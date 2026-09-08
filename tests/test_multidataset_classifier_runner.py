from __future__ import annotations

import json
from pathlib import Path
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


def test_runner_uses_operational_confidence_threshold_by_default(tmp_path):
    args = runner.parse_args(
        [
            "--manifest-dir",
            str(tmp_path / "manifests"),
            "--results-dir",
            str(tmp_path / "results"),
            "--data-repository-root",
            str(tmp_path / "data"),
            "--out-dir",
            str(tmp_path / "output"),
        ]
    )

    assert args.confidence_threshold == 0.65
    assert args.portable_reference_out is None


def test_portable_reference_builder_strips_local_paths_and_keeps_ton_views():
    metric = {
        "rows": 1,
        "accuracy": 1.0,
        "macro": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
        "weighted": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
        "top3_accuracy": 1.0,
        "selective": {
            "threshold": 0.65,
            "decided_rows": 1,
            "abstained_rows": 0,
            "coverage": 1.0,
            "accuracy": 1.0,
            "macro_f1": 1.0,
            "risk": 0.0,
        },
        "per_class": [
            {
                "label": "Backdoor",
                "support": 1,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
            }
        ],
    }
    origin_group = {
        "minimum_test_rows_per_class": 10,
        "origins": {"ton_iot_windows": {"status": "not_evaluable"}},
    }
    report = {
        "schema_version": "full-v2",
        "generated_at": "2026-09-07T00:00:00+00:00",
        "network_calls": 0,
        "llm_policy": "materialized",
        "inputs": {
            "manifests": [
                {
                    "path": r"C:\private\ton_manifest.jsonl",
                    "bytes": 10,
                    "sha256": "a" * 64,
                }
            ]
        },
        "taxonomy": {
            "version": "taxonomy-v1",
            "sha256": "b" * 64,
            "reference_taxonomy_version": "reference-v1",
            "attack_classes": ["Backdoor"],
            "normal_policy": "binary_only",
            "mapping_statuses_used_for_training": ["exact"],
            "coarsening_policy": {},
            "forced_mapping_policy": {},
            "excluded_labels": {},
        },
        "mapping": {
            "input_rows": 1,
            "mapped_rows": 1,
            "rejected_attack_rows": 0,
            "status_counts": {"exact": 1},
            "mapped_by_dataset": {"ton_iot": 1},
            "mapped_by_status": {"exact": 1},
            "reason_counts": {"exact": 1},
        },
        "excluded_attack_targets": [],
        "candidate_artifact": {
            "path": r"C:\private\candidate.joblib",
            "bytes": 20,
            "sha256": "c" * 64,
            "activated": False,
        },
        "global_protocol": {
            "balance": {
                "policy": "balanced",
                "seed": 42,
                "sampling_with_replacement": False,
                "minimum_global_class_support": 1,
                "limiting_classes": ["Backdoor"],
                "per_class_total": 1,
                "selected_rows": 1,
                "available": {
                    "rows": 1,
                    "class_counts": {"Backdoor": 1},
                },
                "selected": {
                    "rows": 1,
                    "class_counts": {"Backdoor": 1},
                    "mapping_status_counts": {"exact": 1},
                    "splits": {
                        "train": {
                            "rows": 1,
                            "class_counts": {"Backdoor": 1},
                            "by_dataset": {"ton_iot": {"rows": 1}},
                            "by_origin": {"ton_iot_windows": {"rows": 1}},
                        },
                        "val": {
                            "rows": 0,
                            "class_counts": {"Backdoor": 0},
                            "by_dataset": {"ton_iot": {"rows": 0}},
                            "by_origin": {"ton_iot_windows": {"rows": 0}},
                        },
                        "test": {
                            "rows": 0,
                            "class_counts": {"Backdoor": 0},
                            "by_dataset": {"ton_iot": {"rows": 0}},
                            "by_origin": {"ton_iot_windows": {"rows": 0}},
                        },
                    },
                },
                "ratio_units": {"train": 70, "val": 15, "test": 15},
                "quotas_per_class": {"train": 1, "val": 0, "test": 0},
                "split_assignment": {
                    "method": "stratified",
                    "class_totals_exact": True,
                    "origin_counts_use_floor_or_ceil_of_ideal": True,
                    "tie_breaker": "sha256",
                },
                "source_split_used_for_assignment": False,
                "source_split_used_for_corpus_selection": False,
                "final_cross_split_overlap_count": 0,
                "selection_sha256": "d" * 64,
            },
            "training": {
                "model": "xgboost",
                "classes": ["Backdoor"],
                "features": {"count": 3, "names_sha256": "e" * 64},
                "parameters": {"n_estimators": 1},
                "operational_contract": {
                    "task": "attack_subtype",
                    "taxonomy_version": "taxonomy-v1",
                    "confidence_threshold": 0.65,
                    "model_name": "candidate",
                },
                "supports": {"train": {"rows": 1}},
                "data_sha256": {
                    "eligible": "f" * 64,
                    "train": "1" * 64,
                    "val": "2" * 64,
                    "test": "3" * 64,
                },
            },
            "evaluation": {"val": metric, "test": metric},
            "evaluation_by_origin": {
                "policy": "balanced per origin",
                "development_overlap_rows": 0,
                "global_classifier_test": {
                    "rows": 1,
                    "robust_balanced": {
                        "by_dataset": origin_group,
                        "by_detailed_origin": origin_group,
                    },
                },
                "extended_unused_holdout": {
                    "rows": 1,
                    "robust_balanced": {
                        "by_dataset": origin_group,
                        "by_detailed_origin": origin_group,
                    },
                },
            },
            "deployment_assessment": {"automatically_activated": False},
        },
    }

    portable = runner.build_portable_classifier_reference(
        report,
        artifact_filename="deployed.joblib",
        deployed=True,
    )

    assert portable["artifact"]["filename"] == "deployed.joblib"
    assert portable["artifact"]["deployed"] is True
    assert portable["taxonomy"]["training_taxonomy_sha256"] == "b" * 64
    assert (
        portable["taxonomy"]["runtime_taxonomy_sha256"]
        == runner.multidataset_taxonomy_report()["sha256"]
    )
    assert portable["hashes"]["training_taxonomy"] == "b" * 64
    assert (
        portable["hashes"]["runtime_taxonomy"]
        == runner.multidataset_taxonomy_report()["sha256"]
    )
    assert portable["model"]["operational_contract"]["public_task"] == "attack_type"
    assert portable["inputs"]["manifests"][0]["filename"] == "ton_manifest.jsonl"
    assert "C:\\private" not in json.dumps(portable)
    detailed = portable["metrics"]["per_origin"]["global_classifier_test"]
    assert detailed["by_detailed_origin"]["origins"]["ton_iot_windows"] == {
        "status": "not_evaluable"
    }


def test_versioned_portable_reference_matches_deployed_classifier_contract():
    path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "evaluation_references"
        / "xgboost_attack_type_multidataset16_balanced500_20260907.json"
    )
    reference = json.loads(path.read_text(encoding="utf-8"))

    assert reference["schema_version"] == "classifier-evaluation-reference-v1"
    assert reference["artifact"] == {
        "bytes": 2_018_126,
        "deployed": True,
        "filename": "xgboost_attack_subtype_multidataset16_balanced500_20260906.joblib",
        "sha256": "9175adb6f78a69962676c9aba0e9b58a59ef94e1e500c9023f4d9b184fa531b5",
        "source_candidate_activated": False,
    }
    assert reference["taxonomy"]["class_count"] == 16
    assert len(reference["taxonomy"]["attack_classes"]) == 16
    assert reference["taxonomy"]["training_taxonomy_sha256"] == (
        "d016e777a545c78361adb1c05352a70f36e0022c5f85e5aaf4f8e379655bd6ed"
    )
    assert reference["taxonomy"]["runtime_taxonomy_sha256"] == (
        "060a45a18e9a99e302467a12a358fe1f346091707a75375e2082a98e3506fae2"
    )
    assert reference["balancing"]["selected_rows"] == 8_000
    assert reference["balancing"]["per_class_total"] == 500
    assert reference["splits"]["quotas_per_class"] == {
        "train": 350,
        "val": 75,
        "test": 75,
    }
    assert reference["splits"]["supports"]["test"]["rows"] == 1_200
    assert reference["model"]["operational_contract"]["confidence_threshold"] == 0.65
    assert reference["model"]["operational_contract"]["public_task"] == "attack_type"
    assert reference["metrics"]["test"]["accuracy"] == 0.8908333333333334
    assert len(reference["metrics"]["per_class"]) == 16
    per_origin = reference["metrics"]["per_origin"]
    for view_name in ("global_classifier_test", "extended_unused_holdout"):
        ton = per_origin[view_name]["by_detailed_origin"]["origins"]
        assert {
            "ton_iot_linux",
            "ton_iot_network",
            "ton_iot_telemetry",
            "ton_iot_windows",
        }.issubset(ton)
    assert (
        per_origin["extended_unused_holdout"]["by_detailed_origin"]["origins"]
        ["ton_iot_linux"]["rows"]
        == 192
    )


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
