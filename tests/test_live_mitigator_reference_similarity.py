from __future__ import annotations

import json

import pytest

from scripts.analyze_live_mitigator_reference_similarity import (
    REVIEW_APPROVED,
    REVIEW_PENDING,
    approve_all_references,
    candidate_rows,
    combine_reference_batches,
    evaluate,
    export_template,
    manual_review_report,
    summary_report,
)


def _case() -> dict:
    return {
        "case_id": "case-1",
        "canonical_event": {
            "event_id": "event-1",
            "modality": "network_flow",
            "schema_profile": "network_flow",
            "src_ip": "192.0.2.10",
            "dst_ip": "198.51.100.20",
            "dst_port": 443,
            "transport_proto": "TCP",
            "semantic_text": "Flujo TCP observado hacia el puerto 443.",
            "label_raw": None,
            "attack_family": None,
            "attack_subtype": None,
            "origin": {"source_name": "fixture"},
        },
        "classification": {"attack_type": "DDoS_TCP"},
        "explanation": {
            "mitigation_items": [
                {
                    "source": "llm",
                    "phase": "containment",
                    "base": "limitar el trafico hacia el servicio afectado",
                    "context": "TEXTO MISTRAL QUE DEBE PERMANECER OCULTO",
                },
                {
                    "source": "catalog",
                    "phase": "prevention",
                    "text": "elemento no contextualizado",
                },
            ]
        },
    }


def test_export_template_is_blind_and_marks_manual_review_pending():
    template = export_template([_case()])
    serialized = json.dumps(template, ensure_ascii=False)

    assert "TEXTO MISTRAL" not in serialized
    assert template["provenance"]["candidate_context_hidden_during_drafting"] is True
    reference = template["cases"][0]["references"][0]
    assert reference["reference_context"] == ""
    assert reference["manual_review"]["status"] == REVIEW_PENDING


def test_evaluate_requires_exact_coverage_and_reports_pending_review():
    cases = [_case()]
    template = export_template(cases)
    reference = template["cases"][0]["references"][0]
    reference["reference_context"] = (
        "Limitar el flujo TCP observado desde 192.0.2.10 hacia "
        "198.51.100.20:443 mientras se valida su legitimidad."
    )
    references = {
        reference["reference_id"]: {
            **reference,
            "case_id": "case-1",
            "attack_type": "DDoS_TCP",
        }
    }

    report = evaluate(
        candidate_rows(cases),
        references,
        reference_metadata={"sources": ["fixture"]},
    )

    assert report["global"]["contexts"] == 1
    assert report["global"]["manually_approved"] == 0
    assert 0.0 <= report["global"]["rouge_l_vs_reference"] <= 1.0
    assert 0.0 <= report["global"]["tfidf_cosine_vs_reference"] <= 1.0

    with pytest.raises(ValueError, match="Cobertura de referencias incorrecta"):
        evaluate(
            candidate_rows(cases),
            {},
            reference_metadata={"sources": []},
        )


def test_combine_preserves_pending_review_and_creates_checklist(tmp_path):
    template = export_template([_case()])
    template["cases"][0]["references"][0]["reference_context"] = (
        "Limitar cautelarmente el flujo observado mientras se valida su legitimidad."
    )
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps(template, ensure_ascii=False), encoding="utf-8")

    combined = combine_reference_batches([batch])
    checklist = manual_review_report(combined)

    assert combined["summary"] == {
        "cases": 1,
        "attack_types": 1,
        "references": 1,
        "manually_approved": 0,
    }
    assert combined["provenance"]["manual_review_status"] == REVIEW_PENDING
    assert combined["provenance"]["component_sources"][0]["path"] == str(batch)
    assert "- [ ] `case-1::mitigation::00`" in checklist
    assert "apoyo de OpenAI Codex" in checklist


def test_summary_report_drops_pair_texts():
    assert summary_report({"global": {"contexts": 1}, "pairs": [{"secret": "x"}]}) == {
        "global": {"contexts": 1}
    }


def test_approval_requires_explicit_reviewer_and_marks_every_reference(tmp_path):
    template = export_template([_case()])
    template["cases"][0]["references"][0]["reference_context"] = "Referencia."
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps(template, ensure_ascii=False), encoding="utf-8")
    combined = combine_reference_batches([batch])

    approved = approve_all_references(
        combined,
        reviewer="author",
        reviewed_at="2026-09-07T12:00:00+00:00",
    )

    review = approved["cases"][0]["references"][0]["manual_review"]
    assert review == {
        "status": REVIEW_APPROVED,
        "reviewer": "author",
        "reviewed_at": "2026-09-07T12:00:00+00:00",
        "notes": None,
    }
    assert approved["summary"]["manually_approved"] == 1
    assert approved["provenance"]["manual_review_status"] == REVIEW_APPROVED

    with pytest.raises(ValueError, match="reviewer"):
        approve_all_references(
            combined,
            reviewer="",
            reviewed_at="2026-09-07T12:00:00+00:00",
        )
