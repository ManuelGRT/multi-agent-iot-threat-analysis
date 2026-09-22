"""Construye y evalua referencias independientes para contextos del mitigador.

El modo ``export`` extrae tareas ciegas: conserva el evento y la medida base,
pero excluye expresamente el texto generado por Mistral. El modo ``evaluate``
exige una referencia por cada contexto anclado y calcula ROUGE-L y coseno
TF-IDF tanto frente al catalogo como frente a la referencia contextualizada.

Las referencias asistidas por IA permanecen con revision manual pendiente
hasta que el autor marque cada elemento de forma individual. El script nunca
convierte automaticamente un borrador en una referencia revisada.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_live_mitigator_context_similarity import norm_tokens, rouge_l


SCHEMA_VERSION = "1.0"
REFERENCE_AUTHORING_SYSTEM = "OpenAI Codex"
REVIEW_PENDING = "pending_author_review"
REVIEW_APPROVED = "approved_by_author"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="Exporta tareas sin texto Mistral")
    export.add_argument("--campaign-dir", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)

    combine = subparsers.add_parser(
        "combine", help="Congela varios lotes ciegos en una referencia unica"
    )
    combine.add_argument("--input", type=Path, action="append", required=True)
    combine.add_argument("--output", type=Path, required=True)
    combine.add_argument(
        "--review-output",
        type=Path,
        help="Lista Markdown para la revision manual individual del autor",
    )
    combine.add_argument(
        "--approve-all",
        action="store_true",
        help="Marca todas las referencias como revisadas tras confirmacion del autor",
    )
    combine.add_argument(
        "--reviewer",
        help="Identidad que quedara registrada al utilizar --approve-all",
    )
    combine.add_argument(
        "--reviewed-at",
        help="Fecha ISO de revision; por defecto se utiliza la hora UTC actual",
    )

    evaluate = subparsers.add_parser("evaluate", help="Calcula las similitudes")
    evaluate.add_argument("--campaign-dir", type=Path, required=True)
    evaluate.add_argument(
        "--references",
        type=Path,
        action="append",
        required=True,
        help="JSON de referencias; puede repetirse para combinar lotes",
    )
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument(
        "--summary-output",
        type=Path,
        help="JSON resumido sin los textos candidatos ni las referencias",
    )
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def portable_path(path: Path) -> str:
    """Evita incrustar rutas locales cuando el archivo pertenece al repositorio."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _drop_empty(value: Any) -> Any:
    if isinstance(value, dict):
        output = {
            str(key): _drop_empty(item)
            for key, item in value.items()
            if item not in (None, "", [], {})
        }
        return {
            key: item
            for key, item in output.items()
            if item not in (None, "", [], {})
        }
    if isinstance(value, list):
        return [_drop_empty(item) for item in value if item not in (None, "", [], {})]
    return value


def event_evidence_snapshot(event: dict[str, Any]) -> dict[str, Any]:
    """Reduce el evento a evidencia util sin targets ni texto candidato."""
    scalar_fields = (
        "event_id",
        "modality",
        "schema_profile",
        "semantic_text",
        "anomaly_summary",
        "src_ip",
        "src_port",
        "dst_ip",
        "dst_port",
        "transport_proto",
        "app_proto",
        "packet_count",
        "byte_count",
        "duration_ms",
        "traffic_direction",
    )
    nested_fields = (
        "asset_context",
        "host",
        "host_context",
        "telemetry_context",
        "service_context",
        "behavior_tags",
        "attack_indicators",
        "missing_fields",
    )
    snapshot = {field: event.get(field) for field in scalar_fields}
    snapshot.update({field: event.get(field) for field in nested_fields})
    return _drop_empty(snapshot)


def candidate_rows(cases: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        attack_type = str(case["classification"]["attack_type"])
        for item_index, item in enumerate(case["explanation"]["mitigation_items"]):
            if item.get("source") != "llm":
                continue
            base = str(item.get("base") or "").strip()
            context = str(item.get("context") or "").strip()
            if not base or not context:
                continue
            rows.append(
                {
                    "reference_id": f"{case['case_id']}::mitigation::{item_index:02d}",
                    "case_id": str(case["case_id"]),
                    "attack_type": attack_type,
                    "item_index": item_index,
                    "phase": str(item.get("phase") or "unknown"),
                    "base": base,
                    "candidate_context": context,
                }
            )
    return rows


def export_template(cases: Iterable[dict[str, Any]]) -> dict[str, Any]:
    output_cases: list[dict[str, Any]] = []
    for case in cases:
        references = []
        for item_index, item in enumerate(case["explanation"]["mitigation_items"]):
            if item.get("source") != "llm":
                continue
            base = str(item.get("base") or "").strip()
            context = str(item.get("context") or "").strip()
            if not base or not context:
                continue
            references.append(
                {
                    "reference_id": f"{case['case_id']}::mitigation::{item_index:02d}",
                    "item_index": item_index,
                    "phase": str(item.get("phase") or "unknown"),
                    "base": base,
                    "reference_context": "",
                    "manual_review": {
                        "status": REVIEW_PENDING,
                        "reviewer": None,
                        "reviewed_at": None,
                        "notes": None,
                    },
                }
            )
        if not references:
            continue
        event = dict(case["canonical_event"])
        output_cases.append(
            {
                "case_id": str(case["case_id"]),
                "attack_type": str(case["classification"]["attack_type"]),
                "dataset_origin": str(event.get("origin", {}).get("source_name") or ""),
                "evidence": event_evidence_snapshot(event),
                "references": references,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "authoring_method": "ai_assisted_reference_drafting",
            "authoring_system": REFERENCE_AUTHORING_SYSTEM,
            "candidate_context_hidden_during_drafting": True,
            "source_material": "canonical event evidence and immutable catalog base only",
            "manual_review_policy": "each reference requires explicit author approval",
            "manual_review_status": REVIEW_PENDING,
        },
        "cases": output_cases,
    }


def load_references(
    paths: Sequence[Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    provenances: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for path in paths:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Version de referencia incompatible en {path}")
        provenances.append(dict(payload.get("provenance") or {}))
        sources.append(
            {
                "path": portable_path(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        for case in payload.get("cases") or []:
            for reference in case.get("references") or []:
                reference_id = str(reference.get("reference_id") or "")
                if not reference_id:
                    raise ValueError(f"Referencia sin identificador en {path}")
                if reference_id in references:
                    raise ValueError(f"Referencia duplicada: {reference_id}")
                row = dict(reference)
                row["case_id"] = str(case.get("case_id") or "")
                row["attack_type"] = str(case.get("attack_type") or "")
                references[reference_id] = row
    return references, {"reference_sources": sources, "reference_provenance": provenances}


def combine_reference_batches(paths: Sequence[Path]) -> dict[str, Any]:
    """Combina lotes disjuntos y conserva su procedencia verificable."""
    cases: list[dict[str, Any]] = []
    seen_cases: set[str] = set()
    seen_references: set[str] = set()
    sources: list[dict[str, str]] = []
    for path in paths:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Version de referencia incompatible en {path}")
        provenance = dict(payload.get("provenance") or {})
        if provenance.get("candidate_context_hidden_during_drafting") is not True:
            raise ValueError(f"El lote no acredita cegamiento en {path}")
        sources.append(
            {"path": portable_path(path), "sha256": hashlib.sha256(raw).hexdigest()}
        )
        for case in payload.get("cases") or []:
            case_id = str(case.get("case_id") or "")
            if not case_id or case_id in seen_cases:
                raise ValueError(f"case_id vacio o duplicado: {case_id!r}")
            seen_cases.add(case_id)
            for reference in case.get("references") or []:
                reference_id = str(reference.get("reference_id") or "")
                if not reference_id or reference_id in seen_references:
                    raise ValueError(
                        f"reference_id vacio o duplicado: {reference_id!r}"
                    )
                seen_references.add(reference_id)
                if not str(reference.get("reference_context") or "").strip():
                    raise ValueError(f"Referencia vacia: {reference_id}")
                review_status = (reference.get("manual_review") or {}).get("status")
                if review_status not in {REVIEW_PENDING, REVIEW_APPROVED}:
                    raise ValueError(
                        f"Estado de revision no reconocido en {reference_id}: "
                        f"{review_status!r}"
                    )
            cases.append(case)

    cases.sort(key=lambda item: (str(item.get("attack_type")), str(item.get("case_id"))))
    approved = sum(
        (reference.get("manual_review") or {}).get("status") == REVIEW_APPROVED
        for case in cases
        for reference in case.get("references") or []
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "authoring_method": "ai_assisted_blind_draft",
            "authoring_system": REFERENCE_AUTHORING_SYSTEM,
            "candidate_context_hidden_during_drafting": True,
            "source_material": "canonical event evidence and immutable catalog base only",
            "manual_review_policy": "each reference requires explicit author approval",
            "manual_review_status": (
                REVIEW_APPROVED if approved == len(seen_references) else REVIEW_PENDING
            ),
            "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "component_sources": sources,
        },
        "summary": {
            "cases": len(cases),
            "attack_types": len({str(case.get("attack_type")) for case in cases}),
            "references": len(seen_references),
            "manually_approved": approved,
        },
        "cases": cases,
    }


def manual_review_report(payload: dict[str, Any]) -> str:
    """Presenta cada referencia sin marcarla automaticamente como revisada."""
    summary = payload["summary"]
    lines = [
        "# Revisión manual de referencias contextualizadas del mitigador",
        "",
        (
            "Estas referencias se redactaron con apoyo de OpenAI Codex usando solo "
            "el evento canónico y la medida catalogada. El texto candidato de Mistral "
            "permaneció oculto durante la redacción."
        ),
        "",
        (
            f"Cobertura: {summary['references']} referencias de "
            f"{summary['attack_types']} tipos de ataque."
        ),
        "",
        (
            "Una casilla vacía significa que la revisión del autor sigue pendiente. "
            "Marcar la casilla en este documento no modifica por sí solo el estado "
            "auditable del JSON."
        ),
        "",
    ]
    for case in payload["cases"]:
        lines.extend(
            [
                f"## {case['attack_type']}",
                "",
                f"Caso: `{case['case_id']}` · Origen: `{case.get('dataset_origin', '')}`",
                "",
            ]
        )
        for reference in case.get("references") or []:
            review = reference.get("manual_review") or {}
            checked = "x" if review.get("status") == REVIEW_APPROVED else " "
            lines.extend(
                [
                    f"- [{checked}] `{reference['reference_id']}`",
                    "",
                    f"  - Medida catalogada: {reference['base']}",
                    f"  - Referencia contextualizada: {reference['reference_context']}",
                    f"  - Estado auditable: `{review.get('status', REVIEW_PENDING)}`",
                    "",
                ]
            )
    return "\n".join(lines)


def approve_all_references(
    payload: dict[str, Any], *, reviewer: str, reviewed_at: str
) -> dict[str, Any]:
    """Registra una confirmacion explicita; nunca se invoca automaticamente."""
    reviewer = reviewer.strip()
    reviewed_at = reviewed_at.strip()
    if not reviewer:
        raise ValueError("--reviewer es obligatorio con --approve-all")
    if not reviewed_at:
        raise ValueError("--reviewed-at no puede estar vacio")

    approved = 0
    for case in payload["cases"]:
        for reference in case.get("references") or []:
            review = dict(reference.get("manual_review") or {})
            review.update(
                {
                    "status": REVIEW_APPROVED,
                    "reviewer": reviewer,
                    "reviewed_at": reviewed_at,
                }
            )
            reference["manual_review"] = review
            approved += 1
    payload["summary"]["manually_approved"] = approved
    payload["provenance"]["manual_review_status"] = REVIEW_APPROVED
    payload["provenance"]["manual_review_confirmation"] = {
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "scope": "all references individually reviewed",
    }
    return payload


def summary_report(report: dict[str, Any]) -> dict[str, Any]:
    """Elimina los textos por par y conserva resultados y procedencia."""
    return {key: value for key, value in report.items() if key != "pairs"}


def paired_cosine(candidates: Sequence[str], references: Sequence[str]) -> list[float]:
    if len(candidates) != len(references):
        raise ValueError("Candidatos y referencias deben tener la misma longitud")
    documents = [" ".join(norm_tokens(text)) for text in candidates]
    documents.extend(" ".join(norm_tokens(text)) for text in references)
    matrix = TfidfVectorizer().fit_transform(documents)
    count = len(candidates)
    similarities = cosine_similarity(matrix[:count], matrix[count:])
    return [float(similarities[index, index]) for index in range(count)]


def evaluate(
    candidates: Sequence[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    *,
    reference_metadata: dict[str, Any],
) -> dict[str, Any]:
    expected_ids = {row["reference_id"] for row in candidates}
    received_ids = set(references)
    if expected_ids != received_ids:
        raise ValueError(
            "Cobertura de referencias incorrecta: "
            f"faltan={sorted(expected_ids - received_ids)} "
            f"sobran={sorted(received_ids - expected_ids)}"
        )

    pairs: list[dict[str, Any]] = []
    for candidate in candidates:
        reference = references[candidate["reference_id"]]
        if reference["case_id"] != candidate["case_id"]:
            raise ValueError(f"case_id incoherente en {candidate['reference_id']}")
        if reference["attack_type"] != candidate["attack_type"]:
            raise ValueError(f"attack_type incoherente en {candidate['reference_id']}")
        if int(reference.get("item_index", -1)) != candidate["item_index"]:
            raise ValueError(f"item_index incoherente en {candidate['reference_id']}")
        if str(reference.get("base") or "").strip() != candidate["base"]:
            raise ValueError(f"base alterada en {candidate['reference_id']}")
        reference_context = str(reference.get("reference_context") or "").strip()
        if not reference_context:
            raise ValueError(f"Referencia vacia: {candidate['reference_id']}")
        review = dict(reference.get("manual_review") or {})
        pairs.append(
            {
                **candidate,
                "reference_context": reference_context,
                "manual_review": review,
                "rouge_l_vs_catalog": rouge_l(
                    candidate["candidate_context"], candidate["base"]
                ),
                "rouge_l_vs_reference": rouge_l(
                    candidate["candidate_context"], reference_context
                ),
            }
        )

    candidate_texts = [pair["candidate_context"] for pair in pairs]
    catalog_texts = [pair["base"] for pair in pairs]
    reference_texts = [pair["reference_context"] for pair in pairs]
    for pair, value in zip(pairs, paired_cosine(candidate_texts, catalog_texts)):
        pair["tfidf_cosine_vs_catalog"] = value
    for pair, value in zip(pairs, paired_cosine(candidate_texts, reference_texts)):
        pair["tfidf_cosine_vs_reference"] = value

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_type[pair["attack_type"]].append(pair)

    metric_fields = (
        "rouge_l_vs_catalog",
        "tfidf_cosine_vs_catalog",
        "rouge_l_vs_reference",
        "tfidf_cosine_vs_reference",
    )

    def aggregate(values: Sequence[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {"contexts": len(values)}
        for field in metric_fields:
            result[field] = statistics.fmean(float(value[field]) for value in values)
        result["manually_approved"] = sum(
            (value.get("manual_review") or {}).get("status") == REVIEW_APPROVED
            for value in values
        )
        return result

    rows = [
        {"attack_type": attack_type, **aggregate(values)}
        for attack_type, values in sorted(by_type.items())
    ]
    global_metrics = aggregate(pairs)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": {
            "candidate": "Mistral mitigation_items[].context",
            "catalog_reference": "associated immutable mitigation base",
            "independent_reference": (
                "AI-assisted contextualization drafted without the Mistral candidate"
            ),
            "rouge_l": "token-normalized LCS F1",
            "cosine": (
                "separate paired TF-IDF spaces for catalog and independent reference"
            ),
            "interpretation": "auxiliary lexical similarity; not semantic correctness",
            **reference_metadata,
        },
        "global": global_metrics,
        "by_attack_type": rows,
        "pairs": pairs,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Similitud de contextos Mistral frente a catálogo y referencia IA",
        "",
        (
            "| Tipo | Contextos | ROUGE-L catálogo | Coseno catálogo | "
            "ROUGE-L referencia | Coseno referencia | Revisadas |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["by_attack_type"]:
        lines.append(
            f"| {row['attack_type']} | {row['contexts']} | "
            f"{row['rouge_l_vs_catalog']:.4f} | "
            f"{row['tfidf_cosine_vs_catalog']:.4f} | "
            f"{row['rouge_l_vs_reference']:.4f} | "
            f"{row['tfidf_cosine_vs_reference']:.4f} | "
            f"{row['manually_approved']}/{row['contexts']} |"
        )
    row = report["global"]
    lines.append(
        f"| **Media ponderada** | **{row['contexts']}** | "
        f"**{row['rouge_l_vs_catalog']:.4f}** | "
        f"**{row['tfidf_cosine_vs_catalog']:.4f}** | "
        f"**{row['rouge_l_vs_reference']:.4f}** | "
        f"**{row['tfidf_cosine_vs_reference']:.4f}** | "
        f"**{row['manually_approved']}/{row['contexts']}** |"
    )
    lines.extend(
        [
            "",
            "La revisión manual solo se considera completada cuando cada referencia "
            "tiene `manual_review.status=approved_by_author`.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "export":
        campaign_dir = args.campaign_dir.expanduser().resolve()
        cases = read_jsonl(campaign_dir / "cases.jsonl")
        write_json(args.output.expanduser().resolve(), export_template(cases))
        return 0

    if args.command == "combine":
        payload = combine_reference_batches(
            [path.expanduser().resolve() for path in args.input]
        )
        if args.approve_all:
            reviewed_at = args.reviewed_at or datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )
            payload = approve_all_references(
                payload,
                reviewer=args.reviewer or "",
                reviewed_at=reviewed_at,
            )
        elif args.reviewer or args.reviewed_at:
            raise ValueError("--reviewer y --reviewed-at requieren --approve-all")
        write_json(args.output.expanduser().resolve(), payload)
        if args.review_output:
            review_output = args.review_output.expanduser().resolve()
            review_output.parent.mkdir(parents=True, exist_ok=True)
            review_output.write_text(manual_review_report(payload), encoding="utf-8")
        print(json.dumps(payload["summary"], ensure_ascii=False, sort_keys=True))
        return 0

    campaign_dir = args.campaign_dir.expanduser().resolve()
    cases = read_jsonl(campaign_dir / "cases.jsonl")
    references, metadata = load_references(
        [path.expanduser().resolve() for path in args.references]
    )
    report = evaluate(
        candidate_rows(cases), references, reference_metadata=metadata
    )
    output_dir = args.output_dir.expanduser().resolve()
    write_json(output_dir / "reference_similarity.json", report)
    (output_dir / "reference_similarity.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    if args.summary_output:
        write_json(args.summary_output.expanduser().resolve(), summary_report(report))
    print(json.dumps(report["global"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
