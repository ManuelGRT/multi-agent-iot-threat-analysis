# scripts/run_system_audit.py
"""Auditoria E2E de sistema (Fase 5b del plan de cierre).

Ejecuta, sin APIs externas (100% local y determinista):

1. Smoke E2E: N casos por dataset (Edge-IIoTset, TON-IoT host, TON-IoT
   telemetry, IoT-23) recorren el grafo final completo -> % de CaseResult
   validos + auditoria por caso (CaseAuditor, Fase 5a).
2. Metricas batch de los MODELOS DESPLEGADOS (.joblib) reproduciendo el split
   de test exacto de sus scripts de entrenamiento (seed 42) y puntuando con
   ``mcp-evaluation.score_run``: no-regresion frente a las metricas congeladas
   de los artefactos de entrenamiento.
3. Certificacion de los baselines congelados (Fase 0): Edge canonico
   multiclase 0.9363 >= 0.93 y > 0.7479 (Jorge LLM FT), binario >= 0.99,
   artefactos de respaldo presentes. Regla dura del proyecto: los resultados
   congelados NO se re-experimentan; se verifica su integridad y umbrales.
   Este chequeo conserva reproducibilidad historica: NO certifica un split
   deduplicado ni sustituye las fases C-F del plan de validacion posterior.
4. Cobertura del mapeo CAPEC del TFM de Jorge por el catalogo threat intel.
5. Chequeo de target leakage sobre una muestra + caso trampa (self-test del
   detector de leakage).
6. Informe con semaforos: artifacts/informe_auditoria_sistema_<fecha>.md
   (+ .json). Codigo de salida 0 solo si TODOS los semaforos estan en verde.

Uso:
    .venv\\Scripts\\python.exe scripts\\run_system_audit.py
    .venv\\Scripts\\python.exe scripts\\run_system_audit.py --smoke-per-dataset 3 --skip-batch
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.final.auditor import (
    CaseAuditor,
    canonical_leakage_issues,
    feature_leakage_issues,
)
from src.mcp.client import MCPToolClient
from src.mcp.common import artifacts_dir, resolve_confined_path, resolve_path
from src.orchestration.mcp_graph import default_final_agents, run_case

GREEN, AMBER, RED = "VERDE", "AMBAR", "ROJO"

EDGE_DATASET_PATH = (
    "artifacts/datasets/edgeiiot_mistral_standardized_tfm_less_both_partial_cache_recalc_20260621.jsonl"
)
BALANCED450_REPORT = "artifacts/informe_edgeiiot_estandarizado_ml_vs_tfm_jorge_balanced450_20260622.md"
EDGE_CANONICAL_MODEL = "artifacts/models/edgeiiot_canonical_adapter_tfm_less_current.joblib"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auditoria E2E del sistema multiagente MCP.")
    parser.add_argument("--smoke-per-dataset", type=int, default=5)
    parser.add_argument("--leakage-sample", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="Margen de no-regresion frente a las metricas congeladas de entrenamiento.")
    parser.add_argument("--skip-batch", action="store_true",
                        help="Salta las metricas batch (para smoke rapido).")
    parser.add_argument("--out-prefix", default=None,
                        help="Prefijo del informe (por defecto informe_auditoria_sistema_<fecha>).")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# 1. Smoke E2E por dataset
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def sample_rows(path: Path, selector, n: int) -> list[dict[str, Any]]:
    """Primeras n filas que cumplen selector, mitad ataque / mitad benigno."""
    attacks: list[dict[str, Any]] = []
    benigns: list[dict[str, Any]] = []
    want_attack = (n + 1) // 2
    want_benign = n - want_attack
    for row in iter_jsonl(path):
        if not row.get("ok") or not row.get("canonical_event"):
            continue
        if not selector(row):
            continue
        target = row.get("target") or {}
        if target.get("is_attack") and len(attacks) < want_attack:
            attacks.append(row)
        elif not target.get("is_attack") and len(benigns) < want_benign:
            benigns.append(row)
        if len(attacks) >= want_attack and len(benigns) >= want_benign:
            break
    return attacks + benigns


def smoke_datasets(per_dataset: int) -> dict[str, list[dict[str, Any]]]:
    corpus = resolve_path("standardized_dataset")
    edge = PROJECT_ROOT / EDGE_DATASET_PATH

    def top_dir(row: dict[str, Any]) -> str:
        source = str(row.get("source_file") or "").replace("\\", "/")
        parts = source.split("/")
        return parts[1] if len(parts) > 1 else source

    def profile(row: dict[str, Any]) -> str:
        return str((row.get("canonical_event") or {}).get("schema_profile") or "")

    return {
        "edge_iiotset": sample_rows(edge, lambda row: True, per_dataset),
        "ton_iot_host": sample_rows(
            corpus, lambda row: top_dir(row) == "TON_IOT" and profile(row) == "host_metrics", per_dataset
        ),
        "ton_iot_telemetry": sample_rows(
            corpus, lambda row: top_dir(row) == "TON_IOT" and profile(row) == "iot_telemetry", per_dataset
        ),
        "iot23": sample_rows(corpus, lambda row: top_dir(row) == "IOT23", per_dataset),
    }


def run_smoke(per_dataset: int) -> dict[str, Any]:
    agents = default_final_agents()  # cliente in-process compartido: modelos se cargan una vez
    auditor = CaseAuditor()
    per_dataset_results: dict[str, Any] = {}
    all_cases = []
    total = 0
    valid = 0
    for name, rows in smoke_datasets(per_dataset).items():
        cases = []
        for row in rows:
            case = run_case(
                {"dataset": name, "canonical_event": row["canonical_event"]},
                agents=agents,
            )
            cases.append(case)
        audit = auditor.audit_batch(cases)
        dataset_valid = sum(1 for case in cases if case.status in {"completed", "needs_human_review"})
        total += len(cases)
        valid += dataset_valid
        all_cases.extend(cases)
        per_dataset_results[name] = {
            "casos": len(cases),
            "validos": dataset_valid,
            "needs_human_review": sum(1 for c in cases if c.status == "needs_human_review"),
            "auditoria": {
                "by_verdict": audit["by_verdict"],
                "rejected": audit["rejected"],
            },
        }
    rejects = sum(r["auditoria"]["by_verdict"]["reject"] for r in per_dataset_results.values())
    empty_datasets = [name for name, r in per_dataset_results.items() if r["casos"] == 0]
    smoke_green = bool(total) and valid == total and not empty_datasets
    smoke_amber = bool(total) and valid / total >= 0.9 and not empty_datasets
    return {
        "total": total,
        "validos": valid,
        "pct_validos": round(100.0 * valid / total, 1) if total else 0.0,
        "rechazados_por_auditor": rejects,
        "datasets_sin_muestras": empty_datasets,
        "por_dataset": per_dataset_results,
        "casos": all_cases,
        "semaforo_smoke": GREEN if smoke_green else (AMBER if smoke_amber else RED),
        "semaforo_auditoria": GREEN if total and rejects == 0 and not empty_datasets else RED,
    }


# ---------------------------------------------------------------------------
# 2. Metricas batch de los modelos desplegados (split de test reproducido)
# ---------------------------------------------------------------------------

def latest_training_artifact(prefix: str) -> dict[str, Any] | None:
    candidates = sorted((PROJECT_ROOT / "artifacts").glob(f"{prefix}_*.json"))
    if not candidates:
        return None
    with candidates[-1].open("r", encoding="utf-8") as fh:
        return json.load(fh)


def batch_detection(client: MCPToolClient, tolerance: float) -> dict[str, Any]:
    from scripts.train_xgboost_detection_standardized_datasets import (
        load_edge_binary,
        load_prebalanced_binary,
        split_records,
    )
    from src.mcp import model_registry
    from src.mcp.features import event_features

    args = SimpleNamespace(random_state=42, test_size=0.15, val_size=0.15)
    rng = random.Random(args.random_state)
    records = load_prebalanced_binary(resolve_path("standardized_dataset"))
    records += load_edge_binary(PROJECT_ROOT / EDGE_DATASET_PATH, 100, rng)
    rng.shuffle(records)
    _train, _val, test = split_records(records, args)

    model = model_registry.load_model("detection_model")
    features = [event_features(record["event"]) for record in test]
    proba = [float(p[1]) for p in model.predict_proba(features)]
    y_pred = [int(p >= 0.5) for p in proba]
    y_true = [record["label"] for record in test]
    score = client.call("evaluation", "score_run", y_true=y_true, y_pred=y_pred, task="binary", y_score=proba)

    frozen = latest_training_artifact("xgboost_detection_standardized_with_edge")
    frozen_test = (frozen or {}).get("metrics", {}).get("test", {})
    frozen_ok = "f1" in frozen_test and "n" in frozen_test
    frozen_f1 = float(frozen_test.get("f1", 0.0))
    split_ok = frozen_ok and frozen_test.get("n") == len(test)
    live_f1 = float(score.get("f1", 0.0))
    return {
        "n_test": len(test),
        "artefacto_congelado_encontrado": frozen_ok,
        "split_reproducido": split_ok,
        "f1_live": round(live_f1, 4),
        "f1_congelado_entrenamiento": frozen_f1,
        "delta": round(live_f1 - frozen_f1, 4),
        "roc_auc_live": round(float(score.get("roc_auc", 0.0)), 4),
        "modelo": "xgboost_detection_standardized_with_edge_20260705",
        "semaforo": GREEN if frozen_ok and split_ok and live_f1 >= frozen_f1 - tolerance else RED,
    }


def batch_family(client: MCPToolClient, tolerance: float) -> dict[str, Any]:
    from scripts.evaluate_xgboost_attack_family_classification_group_lodo import (
        dataset_group,
        load_edge_attack_families,
        load_prebalanced_attack_families,
    )
    from scripts.train_xgboost_attack_family_balanced_standardized import (
        balance_by_group_and_family,
        split_records,
    )
    from src.mcp import model_registry
    from src.mcp.features import event_features

    args = SimpleNamespace(random_state=42, test_size=0.15, val_size=0.15)
    rng = random.Random(args.random_state)
    pool = load_prebalanced_attack_families(resolve_path("standardized_dataset"))
    pool.extend(load_edge_attack_families(PROJECT_ROOT / EDGE_DATASET_PATH, 5000, rng))
    for record in pool:
        record["dataset_group"] = dataset_group(record["dataset"])
    rng.shuffle(pool)
    records = balance_by_group_and_family(pool, 100, rng)
    _train, _val, test = split_records(records, args)

    model = model_registry.load_model("family_model")
    features = [event_features(record["event"]) for record in test]
    y_pred = model.predict(features)
    y_true = [record["label"] for record in test]
    score = client.call("evaluation", "score_run", y_true=y_true, y_pred=list(y_pred), task="multiclass")

    frozen = latest_training_artifact("xgboost_attack_family_balanced_group")
    frozen_test = (frozen or {}).get("metrics", {}).get("test", {})
    frozen_ok = "weighted_f1" in frozen_test and "n" in frozen_test
    frozen_f1 = float(frozen_test.get("weighted_f1", 0.0))
    # el split se considera reproducido si coincide el tamano Y la
    # distribucion de clases del test congelado
    from collections import Counter

    live_counts = dict(sorted(Counter(y_true).items()))
    frozen_counts = frozen_test.get("class_counts")
    split_ok = (
        frozen_ok
        and frozen_test.get("n") == len(test)
        and (frozen_counts is None or frozen_counts == live_counts)
    )
    live_f1 = float(score.get("f1", 0.0))

    # Slice Edge del test: el generalista frente a la referencia de Jorge.
    edge_pairs = [
        (record["label"], str(pred))
        for record, pred in zip(test, y_pred)
        if record["dataset"] == "EDGE_IIOTSET"
    ]
    edge_slice: dict[str, Any] = {"n": len(edge_pairs)}
    if edge_pairs:
        edge_score = client.call(
            "evaluation",
            "score_run",
            y_true=[p[0] for p in edge_pairs],
            y_pred=[p[1] for p in edge_pairs],
            task="multiclass",
        )
        edge_slice["f1_live"] = round(float(edge_score.get("f1", 0.0)), 4)
        frozen_edge = (
            (frozen or {}).get("metrics", {}).get("test_by_dataset_family", {}).get("EDGE_IIOTSET", {})
        )
        edge_slice["f1_congelado"] = frozen_edge.get("weighted_f1")
        edge_slice["supera_a_jorge_0.7479"] = edge_slice["f1_live"] > 0.7479

    return {
        "n_test": len(test),
        "artefacto_congelado_encontrado": frozen_ok,
        "split_reproducido": split_ok,
        "f1_live_weighted": round(live_f1, 4),
        "f1_congelado_entrenamiento": frozen_f1,
        "delta": round(live_f1 - frozen_f1, 4),
        "slice_edge": edge_slice,
        "modelo": "xgboost_attack_family_balanced_group_20260705",
        "nota": (
            "Metrica del modelo GENERALISTA multi-dataset desplegado; la comparativa con Jorge "
            "(0.9363 vs 0.7479) es sobre Edge canonico y esta congelada en la seccion de baselines."
        ),
        "semaforo": GREEN if frozen_ok and split_ok and live_f1 >= frozen_f1 - tolerance else RED,
    }


# ---------------------------------------------------------------------------
# 3. Baselines congelados + 4. cobertura CAPEC de Jorge
# ---------------------------------------------------------------------------

def check_frozen_baselines(client: MCPToolClient) -> dict[str, Any]:
    baselines = client.call("evaluation", "get_baselines")["baselines"]
    thresholds = baselines["audit_thresholds"]
    edge_multi = baselines["task_multiclass"]["current_system_canonical_edge"]["f1"]
    edge_binary = baselines["task_binary"]["current_system_canonical_edge"]["f1"]
    jorge = baselines["task_multiclass"]["jorge_deepseek_finetuned_less"]["f1"]

    comparison = client.call("evaluation", "compare_with_baseline", f1=edge_multi, task="multiclass")
    checks = {
        "multiclase_edge_canonico_>=_umbral": edge_multi >= thresholds["multiclass_f1_min_canonical_edge"],
        "multiclase_edge_canonico_supera_jorge": comparison["beats_jorge"],
        "binario_edge_canonico_>=_umbral": edge_binary >= thresholds["binary_f1_min"],
        "modelo_edge_canonico_presente": (PROJECT_ROOT / EDGE_CANONICAL_MODEL).exists(),
        "informe_balanced450_presente": (PROJECT_ROOT / BALANCED450_REPORT).exists(),
        "cache_mistral_presente": resolve_path("mistral_cache").exists(),
        "corpus_estandarizado_presente": resolve_path("standardized_dataset").exists(),
    }
    return {
        "f1_multiclase_edge_canonico": edge_multi,
        "f1_binario_edge_canonico": edge_binary,
        "f1_jorge_llm_ft": jorge,
        "delta_vs_jorge": comparison["delta_vs_jorge"],
        "checks": checks,
        "narrativa": baselines.get("narrative"),
        "semaforo": GREEN if all(checks.values()) else RED,
    }


def check_jorge_capec_coverage(client: MCPToolClient) -> dict[str, Any]:
    coverage = client.call("threat_intel", "get_jorge_capec_coverage")
    ok = coverage.get("ok") and coverage.get("covered") == coverage.get("total") == 14
    return {
        "total": coverage.get("total"),
        "cubiertos": coverage.get("covered"),
        "fuente": coverage.get("source"),
        "no_cubiertos": [m for m in coverage.get("mappings", []) if not m.get("covered")],
        "semaforo": GREEN if ok else RED,
    }


# ---------------------------------------------------------------------------
# 5. Target leakage (muestra + trampa)
# ---------------------------------------------------------------------------

def leakage_trap_event() -> dict[str, Any]:
    """Evento trampa CON leakage deliberado: el detector debe cazarlo."""
    return {
        "event_id": "evt-trampa-leakage",
        "modality": "network_flow",
        "schema_profile": "network_flow",
        "label_raw": "DDoS_UDP",
        "attack_family": "ddos",
        "telemetry": {"attack_type": "ddos", "packets": 100},
        "semantic_text": "udp flow label=DDoS_UDP high rate",
        "provenance": {"dataset": "trap"},
        "mapping_confidence": 0.9,
    }


def stride_sample(path: Path, count: int) -> list[dict[str, Any]]:
    """Muestra determinista repartida por TODO el fichero (no solo el inicio).

    El corpus esta ordenado por dataset (BOT_IOT, IOT23, TON_IOT...): tomar
    las primeras filas sesgaria la muestra a un unico origen.
    """
    total_valid = sum(
        1 for row in iter_jsonl(path) if row.get("ok") and row.get("canonical_event")
    )
    if not total_valid:
        return []
    stride = max(1, total_valid // count)
    sampled: list[dict[str, Any]] = []
    index = 0
    for row in iter_jsonl(path):
        if not row.get("ok") or not row.get("canonical_event"):
            continue
        if index % stride == 0 and len(sampled) < count:
            sampled.append(row)
        index += 1
    return sampled


def run_leakage_check(sample_size: int, smoke_cases: list[Any]) -> dict[str, Any]:
    half = max(1, sample_size // 2)
    violations: list[str] = []
    sampled_sources: dict[str, int] = {}
    checked = 0

    # (a) nivel feature: ni siquiera los eventos CRUDOS del corpus (que traen
    # label en el registro) deben producir features con claves target.
    # Muestreo con stride para cubrir todos los datasets del fichero.
    for path in (resolve_path("standardized_dataset"), PROJECT_ROOT / EDGE_DATASET_PATH):
        for position, row in enumerate(stride_sample(path, half)):
            issues = feature_leakage_issues(row["canonical_event"])
            checked += 1
            source = str(row.get("source_file") or "").replace("\\", "/")
            top = source.split("/")[1] if "/" in source else (source or path.name)
            sampled_sources[top] = sampled_sources.get(top, 0) + 1
            if issues:
                violations.append(f"{path.name}#{position}: {issues[:3]}")

    # (b) nivel caso: los eventos canonicos que SALEN del grafo final deben
    # estar sanitizados (sin campos target, sin patrones en semantic_text).
    case_violations: list[str] = []
    for case in smoke_cases:
        issues = canonical_leakage_issues(case.canonical_event or {})
        if issues:
            case_violations.append(f"{case.case_id}: {issues[:3]}")

    # (c) trampa: el detector de leakage DEBE cazar las TRES clases de
    # problema del evento inyectado (campo target, clave anidada y patron en
    # semantic_text) — un conteo simple dejaria pasar una clase rota.
    trap_issues = canonical_leakage_issues(leakage_trap_event())
    trap_kinds = {issue.split(":")[0] for issue in trap_issues}
    trap_detected = {
        "campo_target_con_valor",
        "clave_target_anidada",
        "semantic_text_contiene_patron_target",
    } <= trap_kinds

    ok = not violations and not case_violations and trap_detected
    return {
        "eventos_muestreados_features": checked,
        "muestra_por_origen": sampled_sources,
        "violaciones_features": violations[:10],
        "casos_smoke_revisados": len(smoke_cases),
        "violaciones_casos": case_violations[:10],
        "trampa_detectada": trap_detected,
        "clases_trampa_detectadas": sorted(trap_kinds),
        "problemas_trampa": trap_issues,
        "semaforo": GREEN if ok else RED,
    }


# ---------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------

def semaforo_table(semaforos: dict[str, str]) -> str:
    icon = {GREEN: "🟢", AMBER: "🟡", RED: "🔴"}
    lines = ["| Criterio | Semaforo |", "| --- | --- |"]
    for name, value in semaforos.items():
        lines.append(f"| {name} | {icon.get(value, '')} {value} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = args.out_prefix or f"informe_auditoria_sistema_{stamp}"
    try:
        json_path = resolve_confined_path(
            f"{prefix}.json", artifacts_dir(), allow_absolute=False
        )
    except ValueError as exc:
        print(f"ERROR: prefijo de informe no permitido: {exc}")
        return 2
    client = MCPToolClient(mode="inprocess")

    print("[1/5] Smoke E2E por dataset...")
    smoke = run_smoke(args.smoke_per_dataset)
    smoke_cases = smoke.pop("casos")

    if args.skip_batch:
        detection = {"omitido": True, "semaforo": AMBER}
        family = {"omitido": True, "semaforo": AMBER}
    else:
        print("[2/5] Metricas batch del detector desplegado (split test reproducido)...")
        detection = batch_detection(client, args.tolerance)
        print("[3/5] Metricas batch del clasificador de familia desplegado...")
        family = batch_family(client, args.tolerance)

    print("[4/5] Baselines congelados + cobertura CAPEC de Jorge...")
    baselines = check_frozen_baselines(client)
    jorge_coverage = check_jorge_capec_coverage(client)

    print("[5/5] Chequeo de target leakage (muestra + trampa)...")
    leakage = run_leakage_check(args.leakage_sample, smoke_cases)

    semaforos = {
        "smoke_e2e": smoke["semaforo_smoke"],
        "auditoria_por_caso": smoke["semaforo_auditoria"],
        "batch_deteccion_desplegado": detection["semaforo"],
        "batch_familia_desplegado": family["semaforo"],
        "baselines_congelados_vs_jorge": baselines["semaforo"],
        "cobertura_capec_jorge": jorge_coverage["semaforo"],
        "target_leakage": leakage["semaforo"],
    }
    all_green = all(value == GREEN for value in semaforos.values())

    review_cases = [
        {"case_id": case.case_id, "status": case.status, "issues": case.judge.issues}
        for case in smoke_cases
        if case.status == "needs_human_review"
    ]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "args": {k: v for k, v in vars(args).items()},
        "semaforos": semaforos,
        "todo_verde": all_green,
        "smoke_e2e": {k: v for k, v in smoke.items() if not k.startswith("semaforo")},
        "batch_deteccion": detection,
        "batch_familia": family,
        "baselines_congelados": baselines,
        "cobertura_capec_jorge": jorge_coverage,
        "target_leakage": leakage,
        "casos_revision_humana": review_cases,
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    sections = [
        {"heading": "Semaforos", "body": semaforo_table(semaforos)},
        {
            "heading": "Resumen ejecutivo",
            "body": (
                f"Resultado global: {'TODO VERDE' if all_green else 'HAY CRITERIOS EN ROJO/AMBAR'}. "
                f"Smoke E2E: {smoke['validos']}/{smoke['total']} casos validos "
                f"({smoke['rechazados_por_auditor']} rechazados por el auditor). "
                f"Narrativa: {baselines.get('narrativa', '')}"
            ),
        },
        {"heading": "1. Smoke E2E por dataset", "body": payload["smoke_e2e"]},
        {"heading": "2. Batch deteccion (modelo desplegado)", "body": detection},
        {"heading": "3. Batch familia (modelo desplegado)", "body": family},
        {"heading": "4. Baselines congelados (Edge canonico vs Jorge)", "body": baselines},
        {"heading": "5. Cobertura CAPEC del TFM de Jorge", "body": jorge_coverage},
        {"heading": "6. Target leakage", "body": leakage},
        {"heading": "Casos derivados a revision humana", "body": review_cases or "Ninguno."},
    ]
    report = client.call(
        "evaluation",
        "export_report",
        title="Informe de auditoria E2E del sistema multiagente MCP",
        sections=sections,
        filename=f"{prefix}.md",
    )
    if report.get("ok") is not True or not report.get("path"):
        print(f"ERROR: no se pudo generar el informe Markdown: {report.get('error', report)}")
        return 2

    print(json.dumps({"semaforos": semaforos, "todo_verde": all_green,
                      "informe_md": report.get("path"), "informe_json": str(json_path)},
                     indent=2, ensure_ascii=False))
    return 0 if all_green else 1


if __name__ == "__main__":
    raise SystemExit(main())
