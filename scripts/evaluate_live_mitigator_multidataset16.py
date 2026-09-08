"""Campaña Mistral en vivo para el mitigador de 16 tipos de ataque.

La campaña reutiliza eventos Mistral ya estandarizados del test congelado para
aislar el tramo que se quiere evaluar. Cada caso recorre los componentes reales
del runtime: passthrough de estandarización, detector, clasificador de 16 tipos,
catálogo MCP, contextualización Mistral, juez, memoria de casos y auditor.

La credencial se lee exclusivamente de ``MISTRAL_API_KEY`` y nunca se escribe
en los artefactos. Las respuestas JSON anteriores al anclaje sí se conservan
para que la evaluación sea reproducible y no dependa solo del CaseResult final.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Keep the production model name as the CLI default.  A campaign that uses a
# different dated model must opt in with ``--model``; the manifest will always
# record the effective identifier instead of relabelling the response.
DEFAULT_MODEL = "mistral-small-2603"
OFFICIAL_MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
DEFAULT_SEED = 42
DEFAULT_MIN_CLASSIFIER_CONFIDENCE = 0.80
DEFAULT_MIN_DETECTION_PROBABILITY = 0.80
DEFAULT_MIN_MAPPING_CONFIDENCE = 0.90
DEFAULT_MCP_CLIENT_MODE = "stdio"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--standardized-dir", type=Path, required=True)
    parser.add_argument("--attack-type-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--min-classifier-confidence",
        type=float,
        default=DEFAULT_MIN_CLASSIFIER_CONFIDENCE,
    )
    parser.add_argument(
        "--min-detection-probability",
        type=float,
        default=DEFAULT_MIN_DETECTION_PROBABILITY,
    )
    parser.add_argument(
        "--min-mapping-confidence",
        type=float,
        default=DEFAULT_MIN_MAPPING_CONFIDENCE,
    )
    parser.add_argument(
        "--mcp-client-mode",
        choices=("stdio", "inprocess"),
        default=DEFAULT_MCP_CLIENT_MODE,
        help=(
            "Transporte MCP empleado durante el replay. Por defecto se usa "
            "stdio, igual que en produccion; inprocess queda disponible para "
            "pruebas aisladas."
        ),
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} no contiene un objeto JSON")
            yield value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(stream: Any, value: Any) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    stream.flush()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def git_metadata(repo: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        # Preserve the two leading porcelain status columns.  Using ``strip``
        # here removes the first blank status column of the first tracked file
        # and corrupts its path (for example, ``src`` becomes ``rc``).
        return completed.stdout.rstrip("\r\n")

    try:
        status = run("status", "--porcelain")
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(status),
            "changed_paths": [line[3:] for line in status.splitlines()],
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def configure_http_log(path: Path) -> logging.Handler:
    logger = logging.getLogger("httpx")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return handler


def validate_raw_payload(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - la campaña instala evaluación
        raise RuntimeError(
            "jsonschema es necesario para certificar las respuestas crudas"
        ) from exc
    jsonschema.validate(instance=payload, schema=schema)


class RecordingLLM:
    """Envoltorio transparente que conserva la respuesta anterior al anclaje."""

    def __init__(self, inner: Any, schema: dict[str, Any]) -> None:
        self.inner = inner
        self.schema = schema
        self.model_name = inner.model_name
        self.records: list[dict[str, Any]] = []

    def contextualize(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        canonical = kwargs.get("canonical_event") or {}
        classification = kwargs.get("classification") or {}
        base_items = kwargs.get("base_items") or []
        started_at = utc_now()
        started = time.perf_counter()
        record: dict[str, Any] = {
            "started_at": started_at,
            "event_id": canonical.get("event_id"),
            "attack_type": classification.get("attack_type"),
            "base_count": len(base_items),
            "base_ids": [item.get("id") for item in base_items],
            "status": "error",
        }
        try:
            payload = self.inner.contextualize(*args, **kwargs)
            if not isinstance(payload, dict):
                raise TypeError("Mistral no devolvió un objeto JSON")
            schema_valid = True
            schema_error = None
            try:
                validate_raw_payload(payload, self.schema)
            except Exception as exc:  # el runtime productivo ancla defensivamente
                schema_valid = False
                schema_error = f"{type(exc).__name__}: {exc}"
            record.update(
                {
                    "status": "ok",
                    "schema_valid": schema_valid,
                    "schema_error": schema_error,
                    "payload": payload,
                }
            )
            return payload
        except Exception as exc:
            response = getattr(exc, "response", None)
            record.update(
                {
                    "schema_valid": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "http_status": getattr(response, "status_code", None),
                }
            )
            raise
        finally:
            record["finished_at"] = utc_now()
            record["latency_s"] = round(time.perf_counter() - started, 6)
            self.records.append(record)


def load_standardized(directory: Path) -> tuple[dict[str, dict[str, Any]], list[Path]]:
    files = sorted(directory.glob("*_standardized.jsonl"))
    if not files:
        raise FileNotFoundError(f"No hay JSONL estandarizados en {directory}")
    by_manifest: dict[str, dict[str, Any]] = {}
    for path in files:
        for row in iter_jsonl(path):
            manifest_id = str(row.get("manifest_id") or "")
            if (
                manifest_id
                and row.get("ok") is True
                and row.get("parsed_by_llm") is True
                and row.get("provider") == "mistral"
                and isinstance(row.get("canonical_event"), dict)
            ):
                # Una campaña reanudada puede contener varios intentos. El último
                # éxito materializado es la salida consolidada de la campaña.
                by_manifest[manifest_id] = row
    return by_manifest, files


def prepare_selection(
    *,
    selection_path: Path,
    standardized_dir: Path,
    per_class: int,
    seed: int,
    min_classifier_confidence: float,
    min_detection_probability: float,
    min_mapping_confidence: float,
) -> tuple[list[dict[str, Any]], dict[str, int], list[Path]]:
    from src.contracts.attack_taxonomy import MULTIDATASET_ATTACK_CLASSES
    from src.eval.data_sanitization import sanitize_canonical_event
    from src.mcp import model_registry
    from src.mcp.standardization_contract import validate_target_free_canonical

    standardized, standardized_files = load_standardized(standardized_dir)
    eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    joined = 0

    for selected in iter_jsonl(selection_path):
        split = selected.get("classifier_split") or selected.get("split")
        if split != "test":
            continue
        attack_type = str(selected.get("attack_type") or "")
        if attack_type not in MULTIDATASET_ATTACK_CLASSES:
            continue
        materialized = standardized.get(str(selected.get("manifest_id") or ""))
        if materialized is None:
            continue
        joined += 1
        canonical = sanitize_canonical_event(materialized["canonical_event"])
        validate_target_free_canonical(canonical)
        mapping_confidence = float(canonical.get("mapping_confidence") or 0.0)
        detection = model_registry.detect(canonical)
        classification = model_registry.classify(canonical, top_k=3)
        if selected.get("mapping_status") != "exact":
            continue
        if mapping_confidence < min_mapping_confidence:
            continue
        if float(detection["probability"]) < min_detection_probability:
            continue
        if classification.get("attack_type") != attack_type:
            continue
        if float(classification["confidence"]) < min_classifier_confidence:
            continue
        eligible[attack_type].append(
            {
                "manifest_id": selected["manifest_id"],
                "attack_type": attack_type,
                "dataset_origin": selected.get("dataset_origin"),
                "detailed_origin": selected.get("detailed_origin"),
                "source_campaign_split": selected.get("source_campaign_split"),
                "classifier_split": split,
                "mapping_status": selected.get("mapping_status"),
                "mapping_reason": selected.get("mapping_reason"),
                "schema_profile": canonical.get("schema_profile"),
                "modality": canonical.get("modality"),
                "mapping_confidence": mapping_confidence,
                "detection_probability": float(detection["probability"]),
                "classifier_confidence": float(classification["confidence"]),
                "standardization_provider": materialized.get("provider"),
                "standardization_model": materialized.get("model"),
                "standardization_latency_s": materialized.get("latency_s"),
                "canonical_event": canonical,
            }
        )

    if joined == 0:
        raise RuntimeError("No se pudo unir ninguna fila de test con la estandarización")

    counts = {attack_type: len(eligible[attack_type]) for attack_type in MULTIDATASET_ATTACK_CLASSES}
    missing = [attack_type for attack_type, count in counts.items() if count < per_class]
    if missing:
        raise RuntimeError(
            "No hay suficientes candidatos para: " + ", ".join(missing)
        )

    # El ranking se define y congela antes de llamar al LLM. Se priorizan casos
    # que llegan con margen al mitigador; el SHA con semilla desempata de forma
    # reproducible sin depender del orden de los archivos.
    def rank(item: dict[str, Any]) -> tuple[float, float, str]:
        tie = sha256_bytes(f"{seed}:{item['manifest_id']}".encode("utf-8"))
        return (
            -float(item["classifier_confidence"]),
            -float(item["detection_probability"]),
            tie,
        )

    chosen: list[dict[str, Any]] = []
    for attack_type in MULTIDATASET_ATTACK_CLASSES:
        chosen.extend(sorted(eligible[attack_type], key=rank)[:per_class])
    return chosen, counts, standardized_files


def trace_entry(case: Any, *, agent: str, tool: str | None = None) -> Any:
    matches = [
        entry
        for entry in case.trace
        if entry.agent == agent and (tool is None or entry.tool == tool)
    ]
    return matches[-1] if matches else None


def case_metrics(
    selected: dict[str, Any],
    case: Any,
    audit: Any,
    raw_record: dict[str, Any] | None,
) -> dict[str, Any]:
    items = case.explanation.mitigation_items
    sources = Counter(item.source for item in items)
    suggested_references = sum(
        reference.source == "llm_suggested" for reference in case.explanation.references
    )
    summary_discarded = any(
        str(evidence).startswith("risk_summary_llm_descartado:")
        for evidence in case.explanation.evidence
    )
    llm_trace = trace_entry(case, agent="final_mitigator", tool="llm_contextualize")
    base_count = int((raw_record or {}).get("base_count") or 0)
    payload = (raw_record or {}).get("payload") or {}
    return {
        "case_id": case.case_id,
        "manifest_id": selected["manifest_id"],
        "expected_attack_type": selected["attack_type"],
        "dataset_origin": selected.get("dataset_origin"),
        "detailed_origin": selected.get("detailed_origin"),
        "schema_profile": selected.get("schema_profile"),
        "standardization_source": case.standardization.source,
        "detection_probability": case.detection.probability,
        "predicted_attack_type": case.classification.attack_type,
        "classifier_confidence": case.classification.confidence,
        "llm_status": getattr(llm_trace, "status", None),
        "llm_trace_error": getattr(llm_trace, "error", None),
        "llm_model": case.explanation.model_name,
        "llm_latency_s": (raw_record or {}).get("latency_s"),
        "raw_schema_valid": bool((raw_record or {}).get("schema_valid", False)),
        "raw_mitigation_count": len(payload.get("mitigations") or []),
        "raw_requires_human_review": bool(
            payload.get("requires_human_review", False)
        ),
        "base_count": base_count,
        "anchored_context_count": sources["llm"],
        "catalog_literal_count": sources["catalog"],
        "llm_suggested_count": sources["llm_suggested"],
        "first_five_catalog_anchored": (
            case.explanation.first_five_catalog_anchored
        ),
        "suggested_reference_count": suggested_references,
        "summary_discarded_for_unknown_reference": summary_discarded,
        "llm_context_summary_present": bool(case.explanation.llm_context_summary),
        "review_reasons": list(case.explanation.review_reasons),
        "judge_action": case.judge.action,
        "judge_requires_human_review": case.judge.requires_human_review,
        "judge_issues": list(case.judge.issues),
        "case_status": case.status,
        "audit_verdict": audit.verdict,
        "audit_passed": audit.passed,
        "audit_hard_failures": len(audit.hard_failures),
        "remote_success": bool(
            raw_record
            and raw_record.get("status") == "ok"
            and getattr(llm_trace, "status", None) == "ok"
            and case.explanation.source == "hybrid"
            and case.explanation.llm_context_summary
            and sources["llm"] > 0
        ),
    }


def aggregate(rows: list[dict[str, Any]], eligible_counts: dict[str, int]) -> dict[str, Any]:
    successful = [row for row in rows if row["remote_success"]]
    latencies = [
        float(row["llm_latency_s"])
        for row in rows
        if isinstance(row.get("llm_latency_s"), (int, float))
    ]
    anchored = sum(int(row["anchored_context_count"]) for row in successful)
    bases = sum(int(row["base_count"]) for row in successful)
    suggestions = sum(int(row["llm_suggested_count"]) for row in successful)
    return {
        "cases": len(rows),
        "eligible_candidates_by_type": eligible_counts,
        "remote_successes": len(successful),
        "fallbacks": sum(row["llm_status"] == "fallback" for row in rows),
        "raw_schema_valid": sum(row["raw_schema_valid"] for row in rows),
        "first_five_catalog_anchored": sum(
            row["first_five_catalog_anchored"] for row in successful
        ),
        "anchored_contexts": anchored,
        "bases_presented": bases,
        "base_coverage_micro": anchored / bases if bases else None,
        "full_base_coverage_cases": sum(
            row["anchored_context_count"] == row["base_count"]
            for row in successful
        ),
        "llm_suggested_items": suggestions,
        "cases_with_llm_suggested": sum(
            row["llm_suggested_count"] > 0 for row in successful
        ),
        "llm_suggested_item_rate": (
            suggestions / (anchored + suggestions)
            if anchored + suggestions
            else None
        ),
        "cases_with_suggested_references": sum(
            row["suggested_reference_count"] > 0 for row in successful
        ),
        "cases_with_discarded_summary": sum(
            row["summary_discarded_for_unknown_reference"] for row in successful
        ),
        "llm_context_summaries_preserved": sum(
            row["llm_context_summary_present"] for row in successful
        ),
        "raw_human_review_requests": sum(
            row["raw_requires_human_review"] for row in successful
        ),
        "raw_review_requests_neutralized": sum(
            row["raw_requires_human_review"]
            and row["first_five_catalog_anchored"]
            and row["judge_action"] == "approve"
            for row in successful
        ),
        "judge_actions": dict(sorted(Counter(row["judge_action"] for row in rows).items())),
        "case_statuses": dict(sorted(Counter(row["case_status"] for row in rows).items())),
        "audit_verdicts": dict(sorted(Counter(row["audit_verdict"] for row in rows).items())),
        "audit_hard_failures": sum(row["audit_hard_failures"] for row in rows),
        "llm_latency_s": {
            "minimum": min(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "p95": percentile(latencies, 0.95),
            "maximum": max(latencies) if latencies else None,
        },
    }


def markdown_summary(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Campaña Mistral en vivo del mitigador multidataset16",
        "",
        f"- Casos: {summary['cases']}",
        f"- Respuestas remotas procesables: {summary['remote_successes']}",
        f"- Fallbacks: {summary['fallbacks']}",
        (
            "- Primeras cinco recomendaciones ancladas: "
            f"{summary['first_five_catalog_anchored']}"
        ),
        (
            "- Cobertura de bases: "
            f"{summary['anchored_contexts']}/{summary['bases_presented']}"
        ),
        f"- Sugerencias LLM no catalogadas: {summary['llm_suggested_items']}",
        f"- Decisiones del juez: {summary['judge_actions']}",
        f"- Veredictos del auditor: {summary['audit_verdicts']}",
        "",
        "| Tipo | Origen/perfil | LLM | Contextualizadas | Primeras 5 | Sugerencias | Referencias ajenas | Juez | Auditor |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        origin = f"{row['detailed_origin'] or row['dataset_origin']}/{row['schema_profile']}"
        lines.append(
            "| {expected_attack_type} | {origin} | {llm_status} | "
            "{anchored_context_count}/{base_count} | {first_five} | "
            "{llm_suggested_count} | {refs} | {judge_action} | {audit_verdict} |".format(
                **row,
                origin=origin,
                first_five="sí" if row["first_five_catalog_anchored"] else "no",
                refs=(
                    row["suggested_reference_count"]
                    + int(row["summary_discarded_for_unknown_reference"])
                ),
            )
        )
    return "\n".join(lines) + "\n"


def write_checksums(output_dir: Path) -> None:
    checksum_path = output_dir / "SHA256SUMS"
    lines = []
    for path in sorted(output_dir.rglob("*"), key=lambda item: str(item)):
        if path.is_file() and path.name != checksum_path.name:
            relative = path.relative_to(output_dir).as_posix()
            lines.append(f"{sha256_file(path)}  {relative}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.per_class < 1:
        raise SystemExit("--per-class debe ser al menos 1")
    if not os.getenv("MISTRAL_API_KEY"):
        raise SystemExit("Falta MISTRAL_API_KEY en el entorno")

    repo = Path(__file__).resolve().parents[1]
    selection_path = args.selection.expanduser().resolve()
    standardized_dir = args.standardized_dir.expanduser().resolve()
    attack_type_model = args.attack_type_model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (selection_path, attack_type_model):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not standardized_dir.is_dir():
        raise NotADirectoryError(standardized_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"El directorio de salida no está vacío; no se sobreescribe: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dir = output_dir / "runtime_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TFM_REPO_ROOT"] = str(repo)
    os.environ["TFM_ATTACK_TYPE_MODEL"] = str(attack_type_model)
    os.environ["TFM_STATE_DIR"] = str(state_dir)
    os.environ["MISTRAL_BASE_URL"] = OFFICIAL_MISTRAL_BASE_URL
    os.environ["MITIGATOR_LLM_MODEL"] = args.model
    os.environ.setdefault("MITIGATOR_LLM_TIMEOUT_SECONDS", "120")
    os.environ.setdefault("MISTRAL_CONNECTION_RETRIES", "3")
    os.environ.setdefault("MISTRAL_RATE_LIMIT_RETRIES", "3")
    os.environ.setdefault("MISTRAL_RETRY_STATUS_CODES", "429,500,502,503,504")
    os.environ.setdefault("MISTRAL_MAX_TOKENS", "2200")

    from src.agents.final.auditor import CaseAuditor
    from src.agents.final.llm_mitigator import (
        LLMMitigationAgent,
        MITIGATION_SCHEMA,
        MITIGATOR_SYSTEM,
    )
    from src.contracts.attack_taxonomy import (
        MULTIDATASET_ATTACK_CLASSES,
        MULTIDATASET_TAXONOMY_VERSION,
    )
    from src.contracts.case import CaseResult
    from src.mcp import model_registry
    from src.mcp.client import MCPToolClient
    from src.mcp.common import package_data_dir, resolve_path
    from src.orchestration.mcp_graph import default_final_agents, run_case

    model_registry.clear_cache()
    selected, eligible_counts, standardized_files = prepare_selection(
        selection_path=selection_path,
        standardized_dir=standardized_dir,
        per_class=args.per_class,
        seed=args.seed,
        min_classifier_confidence=args.min_classifier_confidence,
        min_detection_probability=args.min_detection_probability,
        min_mapping_confidence=args.min_mapping_confidence,
    )
    if len(selected) != len(MULTIDATASET_ATTACK_CLASSES) * args.per_class:
        raise RuntimeError("La selección final no está balanceada por tipo")

    selected_public = [
        {key: value for key, value in row.items() if key != "canonical_event"}
        for row in selected
    ]
    selected_path = output_dir / "selected_cases.jsonl"
    selected_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in selected_public
        ),
        encoding="utf-8",
    )

    http_log_path = output_dir / "httpx_sanitized.log"
    http_handler = configure_http_log(http_log_path)

    inner_llm = LLMMitigationAgent(model=args.model, provider="mistral")
    if inner_llm.agent.base_url != OFFICIAL_MISTRAL_BASE_URL:
        raise RuntimeError("La URL efectiva no es la API oficial de Mistral")
    recording_llm = RecordingLLM(inner_llm, MITIGATION_SCHEMA)
    client = MCPToolClient(mode=args.mcp_client_mode)
    agents = default_final_agents(
        client=client,
        use_llm_mitigator=True,
        mitigator_llm=recording_llm,
    )
    auditor = CaseAuditor()

    detector_model = resolve_path("detection_model")
    catalog_path = package_data_dir() / "threat_intel_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    campaign_manifest = {
        "campaign": "mitigator_mistral_live_multidataset16",
        "started_at": utc_now(),
        "git": git_metadata(repo),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "mcp_client_mode": args.mcp_client_mode,
        },
        "selection": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
            "selected_cases_sha256": sha256_file(selected_path),
            "seed": args.seed,
            "per_class": args.per_class,
            "classes": list(MULTIDATASET_ATTACK_CLASSES),
            "eligible_candidates_by_type": eligible_counts,
            "criteria": {
                "classifier_correct": True,
                "mapping_status": "exact",
                "min_classifier_confidence": args.min_classifier_confidence,
                "min_detection_probability": args.min_detection_probability,
                "min_mapping_confidence": args.min_mapping_confidence,
                "ranking": "classifier_confidence_desc,detection_probability_desc,seeded_sha256",
            },
        },
        "standardized_inputs": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in standardized_files
        ],
        "models": {
            "mistral": {
                "provider": "mistral",
                "model": args.model,
                "base_url": inner_llm.agent.base_url,
                "temperature": 0,
                "max_tokens": inner_llm.agent.max_tokens,
                "timeout_seconds": inner_llm.agent.timeout_seconds,
                "connection_retries": inner_llm.agent.connection_retries,
                "rate_limit_retries": inner_llm.agent.rate_limit_retries,
                "retry_status_codes": sorted(inner_llm.agent.retry_status_codes),
                "credential_present": True,
            },
            "classifier": {
                "path": str(attack_type_model),
                "sha256": sha256_file(attack_type_model),
            },
            "detector": {
                "path": str(detector_model),
                "sha256": sha256_file(detector_model),
            },
        },
        "contracts": {
            "taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
            "catalog_path": str(catalog_path),
            "catalog_sha256": sha256_file(catalog_path),
            "mitigator_system_prompt_sha256": sha256_bytes(
                MITIGATOR_SYSTEM.encode("utf-8")
            ),
            "mitigation_schema_sha256": sha256_json(MITIGATION_SCHEMA),
        },
        "method": {
            "input": "frozen Mistral-standardized test event, sanitized before replay",
            "flow": [
                "final_standardizer_prestandardized_passthrough",
                "final_detector",
                "final_classifier_attack_type_16",
                "final_mitigator_catalog",
                "final_mitigator_mistral_live",
                "final_judge",
                "case_memory_persistence",
                "final_auditor",
            ],
            "failures_retained": True,
            "manual_reruns": False,
        },
    }
    write_json(output_dir / "campaign_manifest.json", campaign_manifest)

    raw_path = output_dir / "raw_mistral_responses.jsonl"
    cases_path = output_dir / "cases.jsonl"
    audits_path = output_dir / "audits.jsonl"
    rows_path = output_dir / "results_by_case.jsonl"
    results: list[dict[str, Any]] = []

    with (
        raw_path.open("w", encoding="utf-8") as raw_stream,
        cases_path.open("w", encoding="utf-8") as cases_stream,
        audits_path.open("w", encoding="utf-8") as audits_stream,
        rows_path.open("w", encoding="utf-8") as rows_stream,
    ):
        for index, selected_case in enumerate(selected, start=1):
            attack_type = selected_case["attack_type"]
            case_id = f"case-mitlive-{index:02d}-{sha256_bytes(selected_case['manifest_id'].encode())[:8]}"
            print(
                f"[{index}/{len(selected)}] {attack_type} "
                f"({selected_case['schema_profile']})",
                flush=True,
            )
            raw_before = len(recording_llm.records)
            raw_input = {
                "dataset": str(selected_case.get("dataset_origin") or "validation_2026"),
                "canonical_event": selected_case["canonical_event"],
                "source_file": "validation_2026/frozen_standardized_replay",
                "row_id": selected_case["manifest_id"],
                "split": "test",
            }
            try:
                case = run_case(
                    raw_input,
                    case_id=case_id,
                    agents=agents,
                    use_llm_mitigator=True,
                    persist=True,
                )
                stored = client.call(
                    "case_memory",
                    "get_case",
                    case_id=case_id,
                    include_trace=False,
                )
                if stored.get("ok") is not True:
                    raise RuntimeError(f"No se recuperó el caso persistido: {stored}")
                persisted = CaseResult.model_validate(stored["case"])
                audit = auditor.audit(persisted)
            except Exception as exc:
                # Un fallo de persistencia/contrato también permanece en el
                # denominador. Se materializa un registro mínimo auditable.
                failure = {
                    "case_id": case_id,
                    "manifest_id": selected_case["manifest_id"],
                    "expected_attack_type": attack_type,
                    "fatal_error_type": type(exc).__name__,
                    "fatal_error": str(exc),
                    "remote_success": False,
                    "llm_status": "fatal_error",
                    "raw_schema_valid": False,
                    "first_five_catalog_anchored": False,
                    "anchored_context_count": 0,
                    "base_count": 0,
                    "llm_suggested_count": 0,
                    "suggested_reference_count": 0,
                    "summary_discarded_for_unknown_reference": False,
                    "llm_context_summary_present": False,
                    "raw_requires_human_review": False,
                    "judge_action": "not_reached",
                    "case_status": "error",
                    "audit_verdict": "not_reached",
                    "audit_hard_failures": 1,
                    "llm_latency_s": None,
                }
                if len(recording_llm.records) > raw_before:
                    raw_record = recording_llm.records[-1]
                    append_jsonl(raw_stream, {"case_id": case_id, **raw_record})
                    failure["llm_latency_s"] = raw_record.get("latency_s")
                append_jsonl(rows_stream, failure)
                results.append(failure)
                continue

            raw_record = (
                recording_llm.records[-1]
                if len(recording_llm.records) > raw_before
                else None
            )
            if raw_record is not None:
                append_jsonl(raw_stream, {"case_id": case_id, **raw_record})
            append_jsonl(cases_stream, persisted.model_dump(mode="json"))
            append_jsonl(audits_stream, audit.model_dump(mode="json"))
            row = case_metrics(selected_case, persisted, audit, raw_record)
            append_jsonl(rows_stream, row)
            results.append(row)

    logging.getLogger("httpx").removeHandler(http_handler)
    http_handler.close()

    summary = aggregate(results, eligible_counts)
    summary["completed_at"] = utc_now()
    summary["model"] = args.model
    summary["taxonomy_version"] = MULTIDATASET_TAXONOMY_VERSION
    summary["catalog_version"] = str(catalog["version"])
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(
        markdown_summary(summary, results), encoding="utf-8"
    )
    write_checksums(output_dir)

    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if summary["remote_successes"] == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
