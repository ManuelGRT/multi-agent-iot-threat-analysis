# scripts/run_system_audit.py
"""Auditoria E2E de sistema (Fase 5b del plan de cierre).

Ejecuta, sin APIs externas (100% local y determinista):

1. Smoke E2E: N casos por dataset (Edge-IIoTset, TON-IoT host, TON-IoT
   telemetry, IoT-23) recorren el grafo final completo -> % de CaseResult
   validos + auditoria por caso (CaseAuditor, Fase 5a).
2. Metricas batch de los MODELOS DESPLEGADOS (.joblib). Para el clasificador
   se reconstruye el test exacto de 16 tipos mediante su seleccion congelada;
   si los inputs originales no estan disponibles, solo se informa la metrica
   congelada tras verificar la seleccion, el sidecar y el hash del modelo.
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
from collections import Counter
import hashlib
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
from src.eval.reporting import compare_with_baseline, export_report, load_baselines, score_run
from src.mcp.client import MCPToolClient
from src.mcp.common import artifacts_dir, resolve_confined_path, resolve_path
from src.orchestration.mcp_graph import default_final_agents, run_case

GREEN, AMBER, RED = "VERDE", "AMBAR", "ROJO"

EDGE_DATASET_PATH = (
    "artifacts/datasets/edgeiiot_mistral_standardized_tfm_less_both_partial_cache_recalc_20260621.jsonl"
)
BALANCED450_REPORT = "artifacts/informe_edgeiiot_estandarizado_ml_vs_tfm_jorge_balanced450_20260622.md"
EDGE_CANONICAL_MODEL = "artifacts/models/edgeiiot_canonical_adapter_tfm_less_current.joblib"
ATTACK_TYPE_AUDIT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "validation_2026"
    / "classifier_multidataset16_balanced500_threshold065_20260907"
)
ATTACK_TYPE_REPORT = ATTACK_TYPE_AUDIT_DIR / "classifier_multidataset16_report.json"
ATTACK_TYPE_SELECTION = ATTACK_TYPE_AUDIT_DIR / "global_selection.jsonl"
ATTACK_TYPE_SIDECAR = ATTACK_TYPE_AUDIT_DIR / "taxonomy_mapping_sidecar.jsonl"
ATTACK_TYPE_REPRODUCTION_EPSILON = 1e-9


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


def run_smoke(
    per_dataset: int, client: MCPToolClient | None = None
) -> dict[str, Any]:
    # Auditoria offline rapida: comparte modelos y no pretende medir transporte.
    client = client or MCPToolClient(mode="inprocess")
    agents = default_final_agents(client=client)
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


def batch_detection(tolerance: float) -> dict[str, Any]:
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
    score = score_run(y_true, y_pred, task="binary", y_score=proba)

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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_sha256(rows: list[dict[str, Any]]) -> str:
    """Recalcula la huella con el mismo contrato que genero el corpus."""

    payload = [
        {
            "manifest_id": row["manifest_id"],
            "split": row["classifier_split"],
            "label": row["attack_type"],
            "dataset_origin": row["dataset_origin"],
            "detailed_origin": row["detailed_origin"],
            "mapping_status": row["mapping_status"],
            "mapping_reason": row["mapping_reason"],
        }
        for row in rows
    ]
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_campaign_inputs(
    report: dict[str, Any],
) -> dict[str, Any]:
    """Localiza y verifica los inputs exactos inventariados por la campana.

    Las rutas absolutas originales pueden no existir en otro clon. En ese caso
    se intenta el layout relativo habitual. Una ausencia habilita el modo de
    integridad congelada; un fichero presente con hash distinto es corrupcion.
    """

    resolved: dict[str, list[Path]] = {"manifests": [], "standardized_results": []}
    missing: list[str] = []
    mismatches: list[str] = []
    relative_dirs = {
        "manifests": PROJECT_ROOT / "artifacts" / "validation_2026" / "manifests",
        "standardized_results": (
            PROJECT_ROOT / "artifacts" / "validation_2026" / "standardized"
        ),
    }
    for category in resolved:
        for item in report.get("inputs", {}).get(category, []):
            recorded = Path(str(item.get("path", "")))
            candidates = (recorded, relative_dirs[category] / recorded.name)
            path = next((candidate for candidate in candidates if candidate.is_file()), None)
            if path is None:
                missing.append(recorded.name or f"{category}:ruta_vacia")
                continue
            expected_size = int(item.get("bytes", -1))
            expected_hash = str(item.get("sha256", "")).casefold()
            if path.stat().st_size != expected_size or _sha256_file(path) != expected_hash:
                mismatches.append(str(path))
                continue
            resolved[category].append(path)
    expected_counts = {
        category: len(report.get("inputs", {}).get(category, []))
        for category in resolved
    }
    available = (
        not missing
        and not mismatches
        and all(len(resolved[key]) == expected_counts[key] for key in resolved)
    )
    return {
        "available": available,
        "integrity_ok": not mismatches,
        "paths": resolved,
        "missing": missing,
        "mismatches": mismatches,
    }


def _verify_attack_type_sidecar(
    selection: list[dict[str, Any]], sidecar_path: Path
) -> bool:
    expected = {str(row["manifest_id"]): row for row in selection}
    if len(expected) != len(selection):
        return False
    observed: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(sidecar_path):
        manifest_id = str(row.get("manifest_id", ""))
        if manifest_id not in expected:
            continue
        if manifest_id in observed:
            return False
        observed[manifest_id] = row
    if set(observed) != set(expected):
        return False
    return all(
        observed[manifest_id].get("attack_type") == selected.get("attack_type")
        and observed[manifest_id].get("dataset_origin") == selected.get("dataset_origin")
        and observed[manifest_id].get("detailed_origin") == selected.get("detailed_origin")
        and observed[manifest_id].get("status") == selected.get("mapping_status")
        and observed[manifest_id].get("reason") == selected.get("mapping_reason")
        and observed[manifest_id].get("deduplication_status") == "kept"
        for manifest_id, selected in expected.items()
    )


def _frozen_attack_type_metrics(test_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "accuracy": float(test_report["accuracy"]),
        "precision_macro": float(test_report["macro"]["precision"]),
        "recall_macro": float(test_report["macro"]["recall"]),
        "f1_macro": float(test_report["macro"]["f1"]),
        "precision_weighted": float(test_report["weighted"]["precision"]),
        "recall_weighted": float(test_report["weighted"]["recall"]),
        "f1_weighted": float(test_report["weighted"]["f1"]),
        "top3_accuracy": float(test_report["top3_accuracy"]),
        "selective": dict(test_report["selective"]),
    }


def _recompute_attack_type_test(
    *,
    model: Any,
    test_rows: list[dict[str, Any]],
    campaign_inputs: dict[str, list[Path]],
    threshold: float,
) -> dict[str, Any]:
    from src.eval.classifier_balancing import MappedClassifierRecord
    from src.eval.classifier_evaluation import evaluate_classifier_view
    from src.eval.validation_campaign import PreflightRequirements, load_validation_campaign

    campaign = load_validation_campaign(
        campaign_inputs["manifests"],
        campaign_inputs["standardized_results"],
        PreflightRequirements(
            require_complete=True,
            require_llm=True,
            provider="mistral",
            model="mistral-small-2603",
            dedup_scope="global",
        ),
    )
    prepared = {record.manifest_id: record for record in campaign.joined_records}
    missing = [row["manifest_id"] for row in test_rows if row["manifest_id"] not in prepared]
    if missing:
        raise ValueError(f"Faltan {len(missing)} filas del test exacto en la campana")

    records = []
    for row in test_rows:
        record = prepared[row["manifest_id"]]
        if record.feature_fingerprint != row["feature_fingerprint"]:
            raise ValueError(
                "La huella de features no coincide para " + str(row["manifest_id"])
            )
        records.append(
            MappedClassifierRecord(
                record=record,
                label=str(row["attack_type"]),
                dataset_origin=str(row["dataset_origin"]),
                detailed_origin=str(row["detailed_origin"]),
                mapping_status=str(row["mapping_status"]),
                mapping_reason=str(row["mapping_reason"]),
            )
        )
    return evaluate_classifier_view(
        model,
        records,
        confidence_threshold=threshold,
        complete_label_space=True,
    )


def batch_attack_type(tolerance: float) -> dict[str, Any]:
    """Audita el clasificador desplegado sobre el test exacto de 16 tipos."""

    from src.contracts.attack_taxonomy import (
        MULTIDATASET_ATTACK_CLASSES,
        MULTIDATASET_TAXONOMY_VERSION,
    )
    from src.mcp import model_registry

    base = {
        "artefacto_evaluacion_encontrado": ATTACK_TYPE_REPORT.is_file(),
        "seleccion_encontrada": ATTACK_TYPE_SELECTION.is_file(),
        "sidecar_encontrado": ATTACK_TYPE_SIDECAR.is_file(),
    }
    if not all(base.values()):
        return {
            **base,
            "modo_evaluacion": "no_disponible",
            "metricas_recalculadas": False,
            "motivo": "Faltan artefactos de la evaluacion exacta de 16 tipos",
            "semaforo": RED,
        }

    report = json.loads(ATTACK_TYPE_REPORT.read_text(encoding="utf-8"))
    selection = list(iter_jsonl(ATTACK_TYPE_SELECTION))
    balance = report.get("global_protocol", {}).get("balance", {})
    test_report = report.get("global_protocol", {}).get("evaluation", {}).get("test", {})
    expected_classes = tuple(MULTIDATASET_ATTACK_CLASSES)
    report_classes = tuple(report.get("taxonomy", {}).get("attack_classes", ()))
    test_rows = [row for row in selection if row.get("classifier_split") == "test"]
    support = dict(sorted(Counter(str(row.get("attack_type")) for row in test_rows).items()))
    frozen_support = dict(sorted(test_report.get("class_support", {}).items()))
    selection_ok = (
        len(selection) == int(balance.get("selected_rows", -1))
        and _selection_sha256(selection) == balance.get("selection_sha256")
        and len(test_rows) == int(test_report.get("rows", -1))
        and support == frozen_support
        and set(support) == set(expected_classes)
    )
    sidecar_ok = _verify_attack_type_sidecar(selection, ATTACK_TYPE_SIDECAR)

    model_path = resolve_path("attack_type_model")
    expected_model_hash = str(report.get("candidate_artifact", {}).get("sha256", ""))
    model_hash = _sha256_file(model_path) if model_path.is_file() else None
    hash_ok = bool(model_hash and model_hash.casefold() == expected_model_hash.casefold())
    model = model_registry.load_model("attack_type_model") if hash_ok else None
    model_contract_ok = bool(
        model is not None
        and getattr(model, "task", None) == "attack_subtype"
        and set(map(str, model.classes)) == set(expected_classes)
        and len(model.classes) == len(expected_classes)
        and report.get("taxonomy", {}).get("version") == MULTIDATASET_TAXONOMY_VERSION
        and report_classes == expected_classes
    )

    frozen = _frozen_attack_type_metrics(test_report)
    inputs = _resolve_campaign_inputs(report)
    common = {
        **base,
        "n_test": len(test_rows),
        "numero_tipos": len(support),
        "soporte_por_tipo": support,
        "seleccion_verificada": selection_ok,
        "sidecar_verificado": sidecar_ok,
        "sha256_modelo_desplegado": model_hash,
        "sha256_modelo_documentado": expected_model_hash or None,
        "hash_modelo_coincide": hash_ok,
        "contrato_modelo_valido": model_contract_ok,
        "inputs_campaign_disponibles": inputs["available"],
        "inputs_campaign_integridad": inputs["integrity_ok"],
        "inputs_campaign_ausentes": inputs["missing"],
        "inputs_campaign_alterados": inputs["mismatches"],
        "modelo": getattr(model, "model_name", None) if model is not None else None,
        "metricas_congeladas": frozen,
    }
    integrity_ok = selection_ok and sidecar_ok and hash_ok and model_contract_ok
    if not inputs["integrity_ok"]:
        return {
            **common,
            "modo_evaluacion": "no_disponible",
            "metricas_recalculadas": False,
            "motivo": "Hay inputs inventariados cuyo tamano o SHA-256 no coincide",
            "semaforo": RED,
        }
    if not integrity_ok:
        return {
            **common,
            "modo_evaluacion": "integridad_no_verificada",
            "metricas_recalculadas": False,
            "motivo": (
                "No coinciden el modelo desplegado, su contrato o los artefactos "
                "que fijan la membresia exacta del test"
            ),
            "semaforo": RED,
        }
    if not inputs["available"]:
        return {
            **common,
            "modo_evaluacion": "metricas_congeladas_hash_verificado",
            "metricas_recalculadas": False,
            "motivo": (
                "No estan todos los inputs originales; se informa la metrica congelada "
                "solo tras verificar seleccion, sidecar y hash del modelo desplegado"
            ),
            "semaforo": GREEN if integrity_ok else RED,
        }

    try:
        live_report = _recompute_attack_type_test(
            model=model,
            test_rows=test_rows,
            campaign_inputs=inputs["paths"],
            threshold=float(frozen["selective"]["threshold"]),
        )
    except (OSError, TypeError, ValueError) as exc:
        return {
            **common,
            "modo_evaluacion": "reproduccion_fallida",
            "metricas_recalculadas": False,
            "motivo": f"{type(exc).__name__}: {exc}",
            "semaforo": RED,
        }

    live = _frozen_attack_type_metrics(live_report)
    metric_pairs = {
        "accuracy": (live["accuracy"], frozen["accuracy"]),
        "precision_macro": (live["precision_macro"], frozen["precision_macro"]),
        "recall_macro": (live["recall_macro"], frozen["recall_macro"]),
        "f1_macro": (live["f1_macro"], frozen["f1_macro"]),
        "precision_weighted": (
            live["precision_weighted"], frozen["precision_weighted"]
        ),
        "recall_weighted": (live["recall_weighted"], frozen["recall_weighted"]),
        "f1_weighted": (live["f1_weighted"], frozen["f1_weighted"]),
        "top3_accuracy": (live["top3_accuracy"], frozen["top3_accuracy"]),
        "selective_coverage": (
            live["selective"]["coverage"], frozen["selective"]["coverage"]
        ),
        "selective_accuracy": (
            live["selective"]["accuracy"], frozen["selective"]["accuracy"]
        ),
        "selective_f1_macro": (
            live["selective"]["macro_f1"], frozen["selective"]["macro_f1"]
        ),
    }
    deltas = {key: current - expected for key, (current, expected) in metric_pairs.items()}
    exact_metrics = all(
        abs(delta) <= ATTACK_TYPE_REPRODUCTION_EPSILON for delta in deltas.values()
    )
    no_regression = live["f1_weighted"] >= frozen["f1_weighted"] - tolerance
    return {
        **common,
        "modo_evaluacion": "test_exacto_recalculado",
        "metricas_recalculadas": True,
        "metricas_recalculadas_test": live,
        "deltas_vs_congeladas": deltas,
        "reproduccion_exacta": exact_metrics,
        "sin_regresion_segun_tolerancia": no_regression,
        "tolerancia_no_regresion": tolerance,
        "semaforo": GREEN if integrity_ok and exact_metrics and no_regression else RED,
    }


# ---------------------------------------------------------------------------
# 3. Baselines congelados + 4. cobertura CAPEC de Jorge
# ---------------------------------------------------------------------------

def check_frozen_baselines() -> dict[str, Any]:
    baselines = load_baselines()
    thresholds = baselines["audit_thresholds"]
    edge_multi = baselines["task_multiclass"]["current_system_canonical_edge"]["f1"]
    edge_binary = baselines["task_binary"]["current_system_canonical_edge"]["f1"]
    jorge = baselines["task_multiclass"]["jorge_deepseek_finetuned_less"]["f1"]

    comparison = compare_with_baseline(edge_multi, task="multiclass")
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
    smoke = run_smoke(args.smoke_per_dataset, client=client)
    smoke_cases = smoke.pop("casos")

    if args.skip_batch:
        detection = {"omitido": True, "semaforo": AMBER}
        attack_type = {"omitido": True, "semaforo": AMBER}
    else:
        print("[2/5] Metricas batch del detector desplegado (split test reproducido)...")
        detection = batch_detection(args.tolerance)
        print("[3/5] Auditoria batch del clasificador desplegado de 16 tipos...")
        attack_type = batch_attack_type(args.tolerance)

    print("[4/5] Baselines congelados + cobertura CAPEC de Jorge...")
    baselines = check_frozen_baselines()
    jorge_coverage = check_jorge_capec_coverage(client)

    print("[5/5] Chequeo de target leakage (muestra + trampa)...")
    leakage = run_leakage_check(args.leakage_sample, smoke_cases)

    semaforos = {
        "smoke_e2e": smoke["semaforo_smoke"],
        "auditoria_por_caso": smoke["semaforo_auditoria"],
        "batch_deteccion_desplegado": detection["semaforo"],
        "batch_tipo_ataque_desplegado": attack_type["semaforo"],
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
        "batch_tipo_ataque": attack_type,
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
        {
            "heading": "3. Batch clasificador de 16 tipos (modelo desplegado)",
            "body": attack_type,
        },
        {"heading": "4. Baselines congelados (Edge canonico vs Jorge)", "body": baselines},
        {"heading": "5. Cobertura CAPEC del TFM de Jorge", "body": jorge_coverage},
        {"heading": "6. Target leakage", "body": leakage},
        {"heading": "Casos derivados a revision humana", "body": review_cases or "Ninguno."},
    ]
    report_path = export_report(
        title="Informe de auditoria E2E del sistema multiagente MCP",
        sections=sections,
        filename=f"{prefix}.md",
    )

    print(json.dumps({"semaforos": semaforos, "todo_verde": all_green,
                      "informe_md": str(report_path), "informe_json": str(json_path)},
                     indent=2, ensure_ascii=False))
    return 0 if all_green else 1


if __name__ == "__main__":
    raise SystemExit(main())
