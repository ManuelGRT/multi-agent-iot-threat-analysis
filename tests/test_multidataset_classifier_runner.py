from __future__ import annotations

from types import SimpleNamespace

from scripts import train_balanced_multidataset16_classifier as runner
from src.contracts.attack_taxonomy import (
    MULTIDATASET_ATTACK_CLASSES,
    MULTIDATASET_TAXONOMY_VERSION,
)
from src.eval.classifier_balancing import (
    MappedClassifierRecord,
    RejectedClassifierRecord,
)
from src.eval.classifier_targets import MappingContext


def test_training_taxonomy_report_declares_exact_multidataset_contract():
    report = runner._training_taxonomy_report()

    assert report["version"] == MULTIDATASET_TAXONOMY_VERSION
    assert report["training_classes"] == list(MULTIDATASET_ATTACK_CLASSES)
    assert report["mapping_statuses_used_for_training"] == ["exact", "forced"]
    assert len(report["sha256"]) == 64



def test_mapping_sidecar_preserves_forced_quality_and_raw_context():
    record = SimpleNamespace(
        manifest_id="ton-ddos-1",
        dataset="ton_iot",
        source_file="data/ton.csv",
        row_id=7,
        split="train",
        target_class="ddos",
    )
    mapped = MappedClassifierRecord(
        record=record,
        label="DDoS_TCP",
        dataset_origin="ton_iot",
        detailed_origin="ton_iot_network",
        mapping_reason="ton_ddos_raw_protocol_tcp",
        mapping_status="forced",
    )

    rows = runner._mapping_rows(
        [mapped],
        [],
        native_subclasses={},
        mapping_contexts={
            record.manifest_id: MappingContext(
                transport_proto="tcp", app_proto="http"
            )
        },
        deduplication={"removed_groups": []},
    )

    assert rows[0]["status"] == "forced"
    assert rows[0]["attack_type"] == "DDoS_TCP"
    assert rows[0]["raw_transport_proto"] == "tcp"
    assert rows[0]["mapping_context_source"] == "raw_manifest_row"
    assert rows[0]["mapping_context_used_as_feature"] is False


def test_selection_sidecar_distinguishes_source_and_classifier_splits():
    record = SimpleNamespace(
        manifest_id="row-1",
        split="test",
        feature_fingerprint="fp-1",
    )
    selected = MappedClassifierRecord(
        record=record,
        label="DDoS_TCP",
        dataset_origin="iot23",
        detailed_origin="iot23",
        mapping_reason="iot23_ddos_raw_protocol_tcp",
        mapping_status="exact",
    )

    rows = runner._selection_rows(
        [selected], source_splits={"row-1": "train"}
    )

    assert rows[0]["source_campaign_split"] == "train"
    assert rows[0]["classifier_split"] == "test"
    assert rows[0]["split"] == "test"


def test_deployment_assessment_never_auto_activates_candidate():
    result = SimpleNamespace(
        model=SimpleNamespace(
            task="attack_subtype",
            family_mapping_version=MULTIDATASET_TAXONOMY_VERSION,
            classes=list(MULTIDATASET_ATTACK_CLASSES),
        )
    )
    evaluations = {
        "test": {
            "macro": {"f1": 0.90},
            "per_class": [
                {"recall": 0.80} for _label in MULTIDATASET_ATTACK_CLASSES
            ],
            "selective": {"coverage": 0.80, "risk": 0.03},
        }
    }

    assessment = runner._deployment_assessment(
        result, evaluations, {"accepted_rate": 0.20}
    )

    assert assessment["metrics_eligible_for_manual_activation"] is True
    assert assessment["automatically_activated"] is False


def test_excluded_support_keeps_dataset_status_and_reason_separate():
    first = SimpleNamespace(dataset="ton_iot", target_class="ddos")
    second = SimpleNamespace(dataset="iot23", target_class="FileDownload")
    rows = runner._support_by_native_label(
        [
            RejectedClassifierRecord(
                record=first,
                status="ambiguous",
                reason="missing_protocol",
            ),
            RejectedClassifierRecord(
                record=second,
                status="out_of_taxonomy",
                reason="insufficient_support",
            ),
        ]
    )

    assert rows == [
        {
            "dataset": "iot23",
            "native_class": "FileDownload",
            "status": "out_of_taxonomy",
            "reason": "insufficient_support",
            "rows": 1,
        },
        {
            "dataset": "ton_iot",
            "native_class": "ddos",
            "status": "ambiguous",
            "reason": "missing_protocol",
            "rows": 1,
        },
    ]
