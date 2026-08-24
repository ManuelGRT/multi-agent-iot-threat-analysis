# scripts/demo_mcp_multiagent_case.py
"""Demo reproducible de extremo a extremo (Fase 6 del plan de cierre).

Una sola orden ejecuta casos completos por el grafo final multiagente sobre
la capa MCP, sin APIs externas:

    .venv\\Scripts\\python.exe scripts\\demo_mcp_multiagent_case.py --offline

Casos (seleccion determinista):
1. edge_iiotset      — ataque del dataset Edge-IIoTset estandarizado por
                       Mistral (artefacto congelado 20260621): comparacion
                       directa con el TFM de Jorge (F1 0.9363 vs 0.7479).
                       NOTA de honestidad: el cache Mistral del corpus NO
                       contiene Edge, asi que este evento entra ya canonico
                       (passthrough sanitizado por el final_standardizer);
                       la estandarizacion via tool MCP desde cache la
                       ejercitan los otros tres casos.
2. ton_iot_host      — TON-IoT host metrics DESDE EL CACHE Mistral via
                       ``standardize_event`` (from_cache=True, cero APIs).
3. ton_iot_telemetry — TON-IoT telemetria IoT desde el cache Mistral.
4. iot23             — flujo de red IoT-23 desde el cache Mistral.

Cada caso recorre standardize -> detect -> classify -> explain/mitigate ->
judge, se audita en vivo con el CaseAuditor (Fase 5) y se guarda en
``artifacts/demo/demo_case_<dataset>_<fecha>.json`` con un digest SHA256
estable (sin timestamps ni ids volatiles). La referencia de digests vive en
``artifacts/demo/demo_digests.json`` (sin fecha): repetir la orden — hoy o
cualquier otro dia — debe producir digests IDENTICOS; el script lo comprueba
y falla (exit 1) si algun digest cambia. En la primera ejecucion solo
registra la referencia y lo dice explicitamente.

Modos:
- ``--offline`` (recomendado para la defensa): 100% determinista, LLM del
  mitigador desactivado (modo catalogo), cliente MCP in-process.
- ``--mcp-mode stdio``: cada tool viaja por el protocolo MCP real (un
  subproceso por servidor via SDK oficial). Mas lento; demuestra la capa
  MCP autentica.
- ``--use-llm-mitigator``: activa la contextualizacion LLM anclada al
  catalogo (Fase 4); si el LLM no esta disponible, fallback total.

Codigo de salida: 0 si todos los casos son validos y auditados sin reject.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.final.auditor import CaseAuditor
from src.contracts.case import CaseResult
from src.mcp.client import MCPToolClient
from src.mcp.common import artifacts_dir, iter_jsonl, resolve_path
from src.orchestration.mcp_graph import default_final_agents, run_case

EDGE_DATASET_PATH = (
    "artifacts/datasets/edgeiiot_mistral_standardized_tfm_less_both_partial_cache_recalc_20260621.jsonl"
)

# Prefijos del cache Mistral por caso (heterogeneidad de fuentes/modalidades).
CACHE_PREFIXES = {
    "ton_iot_host": "TON_IOT_linux_",
    "ton_iot_telemetry": "TON_IOT_telemetry_",
    "iot23": "IOT23::",
}

# La demo exige estos 4 casos; si alguno no puede seleccionarse, falla.
EXPECTED_CASES = ["edge_iiotset", *CACHE_PREFIXES.keys()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo E2E del sistema multiagente MCP.")
    parser.add_argument("--offline", action="store_true",
                        help="Modo defensa: determinista, sin LLM, sin APIs (recomendado).")
    parser.add_argument("--mcp-mode", choices=["inprocess", "stdio"], default="inprocess",
                        help="stdio = protocolo MCP real por subprocesos (MUY lento: "
                             "cada tool arranca un servidor que recarga modelos).")
    parser.add_argument("--stdio-smoke", action="store_true",
                        help="Demuestra el protocolo MCP real con un handshake stdio "
                             "al servidor threat_intel (ligero) antes de los casos.")
    parser.add_argument("--use-llm-mitigator", action="store_true",
                        help="Contextualizacion LLM anclada al catalogo (ignorado con --offline).")
    parser.add_argument("--no-persist", action="store_true",
                        help="No registrar los casos en la memoria de casos (SQLite).")
    parser.add_argument("--scan-cap", type=int, default=300,
                        help="Tope de filas a escanear al seleccionar cada caso.")
    parser.add_argument("--out-dir", default=None,
                        help="Directorio de salida (por defecto artifacts/demo).")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Seleccion determinista de casos (el escaneo es utillaje local; el caso en si
# corre integro por el grafo final y las tools MCP)
# ---------------------------------------------------------------------------

def pick_edge_attack_row(scan_cap: int) -> dict[str, Any] | None:
    """Primer ataque Edge con mapping>=0.5, deteccion clara y familia firme."""
    from src.mcp import model_registry

    path = PROJECT_ROOT / EDGE_DATASET_PATH
    for row in iter_jsonl(path, limit=scan_cap):
        if not row.get("ok") or not (row.get("target") or {}).get("is_attack"):
            continue
        event = row.get("canonical_event") or {}
        if float(event.get("mapping_confidence", 0.0) or 0.0) < 0.5:
            continue
        try:
            detection = model_registry.detect(event)
            classification = model_registry.classify(event)
        except Exception:
            return None  # sin modelos no hay seleccion informada
        if detection["probability"] > 0.6 and classification["confidence"] >= 0.65:
            return row
    return None


def pick_cache_entry(prefix: str, scan_cap: int) -> dict[str, Any] | None:
    """Primera entrada ok del cache Mistral con ese prefijo y deteccion clara.

    Fallback: la primera entrada ok del prefijo (aunque sea benigna o de baja
    confianza) — la demo nunca se queda sin caso.
    """
    from src.mcp import model_registry

    path = resolve_path("mistral_cache")
    fallback: dict[str, Any] | None = None
    scanned = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = str(record.get("cache_key") or "")
            if not key.startswith(prefix):
                continue
            if not record.get("ok") or not record.get("event"):
                continue
            if fallback is None:
                fallback = record
            scanned += 1
            try:
                detection = model_registry.detect(record["event"])
                if detection["probability"] > 0.6:
                    return record
            except Exception:
                return fallback
            if scanned >= scan_cap:
                break
    return fallback


def build_case_specs(scan_cap: int) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    edge_row = pick_edge_attack_row(scan_cap)
    if edge_row is not None:
        specs.append(
            {
                "name": "edge_iiotset",
                "titulo": (
                    "Edge-IIoTset: evento canonico estandarizado por Mistral "
                    "(artefacto congelado; passthrough sanitizado) - comparativa con Jorge"
                ),
                "raw_input": {
                    "dataset": "edge_iiotset",
                    "canonical_event": edge_row["canonical_event"],
                },
                "ground_truth": edge_row.get("target"),
            }
        )

    for name, prefix in CACHE_PREFIXES.items():
        record = pick_cache_entry(prefix, scan_cap)
        if record is None:
            continue
        specs.append(
            {
                "name": name,
                "titulo": f"{name}: estandarizacion desde el cache Mistral (from_cache)",
                "raw_input": {"dataset": name, "cache_key": record["cache_key"]},
                "ground_truth": None,
            }
        )
    return specs


# ---------------------------------------------------------------------------
# Digest estable: mismo resultado => mismo hash entre ejecuciones
# ---------------------------------------------------------------------------

def stable_view(case: CaseResult) -> dict[str, Any]:
    """Proyeccion del caso SIN campos volatiles (ids, timestamps, latencias)."""
    return {
        "status": case.status,
        "event_id": case.canonical_event.get("event_id"),
        "standardization": {
            "model": case.standardization.model,
            "from_cache": case.standardization.from_cache,
            "mapping_confidence": round(case.standardization.mapping_confidence, 6),
            "schema_profile": case.standardization.schema_profile,
        },
        "detection": {
            "is_malicious": case.detection.is_malicious,
            "probability": round(case.detection.probability, 6),
            "abstain": case.detection.abstain,
            "model_name": case.detection.model_name,
        },
        "classification": {
            "attack_family": case.classification.attack_family,
            "confidence": round(case.classification.confidence, 6),
            "top_scores": {k: round(v, 6) for k, v in sorted(case.classification.top_scores.items())},
        },
        "explanation": {
            "mitigations": case.explanation.mitigations,
            "references": sorted(
                ref.attack_id or ref.capec_id or "" for ref in case.explanation.references
            ),
            "source": case.explanation.source,
        },
        "judge": {
            "action": case.judge.action,
            "final_label": case.judge.final_label,
            "final_confidence": round(case.judge.final_confidence, 6),
            "issues": case.judge.issues,
        },
        "agentes": [entry.agent for entry in case.trace],
    }


def stable_digest(case: CaseResult) -> str:
    payload = json.dumps(stable_view(case), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Presentacion en consola (ASCII-safe para consolas Windows)
# ---------------------------------------------------------------------------

def print_case(spec: dict[str, Any], case: CaseResult, audit_verdict: str, digest: str) -> None:
    print()
    print("=" * 76)
    print(f"CASO {spec['name']}  [{case.case_id}]")
    print(f"  {spec['titulo']}")
    if spec.get("ground_truth"):
        print(f"  ground truth: {spec['ground_truth']}")
    print("-" * 76)
    std = case.standardization
    print(f"  standardize : modelo={std.model} cache={std.from_cache} "
          f"mapping={std.mapping_confidence:.2f} perfil={std.schema_profile}")
    det = case.detection
    print(f"  detect      : malicioso={det.is_malicious} p={det.probability:.4f} "
          f"abstencion={det.abstain}")
    cls = case.classification
    if cls.attack_family:
        top = ", ".join(f"{k}={v:.3f}" for k, v in list(cls.top_scores.items())[:3])
        print(f"  classify    : familia={cls.attack_family} confianza={cls.confidence:.4f} [{top}]")
    exp = case.explanation
    if exp.mitigations:
        refs = ", ".join((r.attack_id or r.capec_id or "") for r in exp.references[:5])
        print(f"  mitigate    : fuente={exp.source} refs=[{refs}]")
        for mitigation in exp.mitigations[:3]:
            print(f"                - {mitigation}")
        if len(exp.mitigations) > 3:
            print(f"                ... y {len(exp.mitigations) - 3} mas")
    print(f"  judge       : accion={case.judge.action} etiqueta={case.judge.final_label} "
          f"issues={case.judge.issues or 'ninguno'}")
    print(f"  traza       : {' -> '.join(entry.agent for entry in case.trace)}")
    print(f"  estado={case.status} | auditoria={audit_verdict} | digest={digest}")


def print_comparison(client: MCPToolClient) -> dict[str, Any]:
    baselines = client.call("evaluation", "get_baselines")["baselines"]
    edge_f1 = baselines["task_multiclass"]["current_system_canonical_edge"]["f1"]
    comparison = client.call("evaluation", "compare_with_baseline", f1=edge_f1, task="multiclass")
    coverage = client.call("threat_intel", "get_jorge_capec_coverage")

    print()
    print("=" * 76)
    print("COMPARATIVA HISTORICA CON JORGE - pendiente de validacion limpia")
    print("-" * 76)
    print(f"  {'Sistema':<52}{'F1 multiclase':>14}")
    rows = [
        ("Jorge XGBoost (baseline ML)", baselines["task_multiclass"]["jorge_xgboost_less_samples"]["f1"]),
        ("Jorge DeepSeek fine-tuned (mejor LLM previo)", baselines["task_multiclass"]["jorge_deepseek_finetuned_less"]["f1"]),
        ("Baseline historico Edge (no resultado final)", edge_f1),
    ]
    for label, value in rows:
        print(f"  {label:<52}{value:>14.4f}")
    print(f"  Delta vs mejor referencia de Jorge: +{comparison['delta_vs_jorge']:.4f} "
          f"(supera={comparison['beats_jorge']})")
    print(f"  Mitigacion: catalogo cubre {coverage['covered']}/{coverage['total']} mapeos CAPEC "
          f"de Jorge y anade tecnicas ATT&CK + mitigaciones M-* auditables")
    print(f"  Narrativa: {baselines.get('narrative', '')}")
    print("  AVISO: 0.9363 procede del split historico; fases C-F deben reemplazarlo.")
    return {
        "f1_edge_canonico": edge_f1,
        "f1_jorge_llm_ft": baselines["task_multiclass"]["jorge_deepseek_finetuned_less"]["f1"],
        "delta_vs_jorge": comparison["delta_vs_jorge"],
        "beats_jorge": comparison["beats_jorge"],
        "cobertura_capec_jorge": f"{coverage['covered']}/{coverage['total']}",
        "validation_status": "historical_pending_clean_revalidation",
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_dir = Path(args.out_dir) if args.out_dir else artifacts_dir() / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    use_llm = False if args.offline else (args.use_llm_mitigator or None)
    client = MCPToolClient(mode=args.mcp_mode)
    agents = default_final_agents(client=client, use_llm_mitigator=use_llm)
    auditor = CaseAuditor()

    print(f"Demo multiagente MCP | modo={'OFFLINE' if args.offline else 'normal'} "
          f"| mcp={args.mcp_mode} | llm_mitigador={'on' if agents.mitigator.llm else 'off (catalogo)'}")

    if args.stdio_smoke:
        print("Handshake MCP real (stdio) contra el servidor threat_intel...")
        try:
            import asyncio

            from src.mcp.client import list_tools_stdio

            tools = asyncio.run(list_tools_stdio("threat_intel"))
            if "suggest_mitigations" not in tools:
                raise RuntimeError(
                    "handshake incompleto: falta la tool suggest_mitigations"
                )
            print(f"  MCP stdio OK: {len(tools)} tools via protocolo oficial "
                  f"({', '.join(tools[:3])}, ...)")
        except Exception as exc:
            print(f"ERROR: MCP stdio no disponible: {type(exc).__name__}: {exc}")
            return 3

    specs = build_case_specs(args.scan_cap)
    if not specs:
        print("ERROR: no hay artefactos disponibles para seleccionar casos.")
        return 2
    missing = [name for name in EXPECTED_CASES if name not in {s["name"] for s in specs}]
    if missing:
        print(f"ERROR: casos obligatorios sin seleccion posible: {missing} "
              "(artefactos/modelos ausentes o umbrales no alcanzados en scan-cap)")

    # Referencia de digests SIN fecha: la reproducibilidad se comprueba entre
    # ejecuciones de cualquier dia, no solo dentro del mismo dia UTC.
    digest_ref_path = out_dir / "demo_digests.json"
    previous_digests: dict[str, str] = {}
    if digest_ref_path.exists():
        try:
            previous_digests = json.loads(digest_ref_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_digests = {}

    all_ok = not missing
    reproducible: bool | None = True if previous_digests else None
    new_digests: dict[str, str] = {}
    summary_cases = []
    for spec in specs:
        case = run_case(spec["raw_input"], agents=agents, persist=not args.no_persist)
        report = auditor.audit(case)
        digest = stable_digest(case)
        print_case(spec, case, report.verdict, digest)

        case_path = out_dir / f"demo_case_{spec['name']}_{stamp}.json"
        new_digests[spec["name"]] = digest
        previous_digest = previous_digests.get(spec["name"])
        payload = {
            "digest": digest,
            "titulo": spec["titulo"],
            "raw_input_selector": {
                k: v for k, v in spec["raw_input"].items() if k != "canonical_event"
            } | ({"event_id": spec["raw_input"]["canonical_event"].get("event_id")}
                 if "canonical_event" in spec["raw_input"] else {}),
            "ground_truth": spec.get("ground_truth"),
            "auditoria": {"verdict": report.verdict, "issues": report.issues},
            "resultado_estable": stable_view(case),
            "case": case.model_dump(mode="json"),
        }
        case_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )

        if previous_digest is not None:
            same = previous_digest == digest
            reproducible = bool(reproducible) and same
            print(f"  reproducibilidad: digest {'IDENTICO' if same else 'DISTINTO'} "
                  f"al de la ejecucion anterior ({previous_digest})")

        valid = case.status in {"completed", "needs_human_review"} and report.verdict != "reject"
        all_ok = all_ok and valid
        summary_cases.append(
            {
                "caso": spec["name"],
                "case_id": case.case_id,
                "status": case.status,
                "auditoria": report.verdict,
                "digest": digest,
                "fichero": str(case_path),
            }
        )

    comparison = print_comparison(client)

    # actualizar la referencia de digests (solo con la seleccion completa)
    if not missing:
        digest_ref_path.write_text(
            json.dumps(new_digests, indent=2, sort_keys=True), encoding="utf-8"
        )

    summary_path = out_dir / f"demo_summary_{stamp}.json"
    summary_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "modo": {"offline": args.offline, "mcp": args.mcp_mode,
                         "llm_mitigador": bool(agents.mitigator.llm)},
                "casos": summary_cases,
                "casos_faltantes": missing,
                "comparativa_jorge": comparison,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print()
    print(f"Salidas: {len(summary_cases)} casos en {out_dir} | resumen: {summary_path.name}")
    if reproducible is None:
        print("Reproducibilidad: primera ejecucion - digests registrados como referencia "
              f"en {digest_ref_path.name} (sin comparacion previa).")
    elif not reproducible:
        print("AVISO: algun digest cambio respecto a la ejecucion anterior.")
    ok = all_ok and reproducible is not False
    print(f"RESULTADO: {'OK - demo valida y auditada' if ok else 'REVISAR'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
