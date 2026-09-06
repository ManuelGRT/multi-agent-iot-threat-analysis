from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.classifier_targets import (
    ClassifierTargetRecoveryError,
    MappingContext,
    bot_iot_manifest_content_hash,
    recover_bot_iot_target_subclasses,
    recover_classifier_mapping_context,
)
from src.eval.validation_campaign import PreparedRecord


RAW_ROWS = [
    {"signal": "1", "attack": "1", "category": "DoS", "subcategory": "HTTP"},
    {"signal": "2", "attack": "1", "category": "DDoS", "subcategory": "TCP"},
    {
        "signal": "3",
        "attack": "1",
        "category": "Reconnaissance",
        "subcategory": "OS_Fingerprint",
    },
]


def _record(manifest_id: str, row_id: int, target_class: str = "DDoS") -> PreparedRecord:
    return PreparedRecord(
        manifest_id=manifest_id,
        dataset="bot_iot",
        source_file="data/BOT_IOT/sample.csv",
        row_id=row_id,
        content_hash=bot_iot_manifest_content_hash(RAW_ROWS[row_id]),
        split="train",
        tasks=("binary", "multiclass"),
        target_class=target_class,
        target_is_attack=True,
        event=None,
        features={"safe": 1.0},
        feature_fingerprint=f"fp-{manifest_id}",
        provider="mistral",
        model="mistral-small-2603",
        mapping_confidence=0.95,
        latency_s=1.0,
        result_source_path=Path("results.jsonl"),
        result_line_number=1,
    )


def _write_csv(root: Path) -> None:
    path = root / "data" / "BOT_IOT" / "sample.csv"
    path.parent.mkdir(parents=True)
    path.write_text(
        "signal,attack,category,subcategory\n"
        "1,1,DoS,HTTP\n"
        "2,1,DDoS,TCP\n"
        "3,1,Reconnaissance,OS_Fingerprint\n",
        encoding="utf-8",
    )


def test_recovers_targets_by_zero_based_row_without_touching_features(tmp_path):
    _write_csv(tmp_path)
    first = _record("ddos", 1)
    second = _record("fingerprint", 2, "Reconnaissance")

    recovered, report = recover_bot_iot_target_subclasses(
        [first, second], repository_root=tmp_path
    )

    assert recovered == {"ddos": "TCP", "fingerprint": "OS_Fingerprint"}
    assert first.features == {"safe": 1.0}
    assert report["feature_use"] is False
    assert report["recovered_rows"] == 2
    assert len(report["mapping_sha256"]) == 64


def test_rejects_category_mismatch(tmp_path):
    _write_csv(tmp_path)
    with pytest.raises(ClassifierTargetRecoveryError, match="no coincide"):
        recover_bot_iot_target_subclasses(
            [_record("wrong", 1, "Reconnaissance")], repository_root=tmp_path
        )


def test_rejects_raw_row_content_hash_mismatch(tmp_path):
    _write_csv(tmp_path)
    record = _record("changed", 1)
    object.__setattr__(record, "content_hash", "0" * 32)
    with pytest.raises(ClassifierTargetRecoveryError, match="content_hash"):
        recover_bot_iot_target_subclasses([record], repository_root=tmp_path)


def test_rejects_source_path_outside_repository(tmp_path):
    record = _record("escape", 1)
    object.__setattr__(record, "source_file", "../outside.csv")
    with pytest.raises(ClassifierTargetRecoveryError, match="sale del repositorio"):
        recover_bot_iot_target_subclasses([record], repository_root=tmp_path)


def test_recovers_forced_mapping_context_from_raw_manifest_not_event(tmp_path):
    record = _record("ton-ddos", 0)
    object.__setattr__(record, "dataset", "ton_iot")
    object.__setattr__(record, "target_class", "ddos")
    object.__setattr__(record, "content_hash", "abc123")
    manifest = tmp_path / "ton_iot_manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "manifest_id": "ton-ddos",
                "dataset": "ton_iot",
                "content_hash": "abc123",
                "row": {"proto": "tcp", "service": "http"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    recovered, report = recover_classifier_mapping_context(
        [record], manifest_paths=[manifest]
    )

    assert recovered == {
        "ton-ddos": MappingContext(transport_proto="tcp", app_proto="http")
    }
    assert report["feature_use"] is False
    assert report["policy"] == "raw_manifest_context_only_not_llm_output"
    assert report["protocol_counts"] == {"tcp": 1}


def test_mapping_context_rejects_manifest_hash_mismatch(tmp_path):
    record = _record("ton-ddos", 0)
    object.__setattr__(record, "dataset", "ton_iot")
    object.__setattr__(record, "target_class", "ddos")
    manifest = tmp_path / "ton_iot_manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "manifest_id": "ton-ddos",
                "dataset": "ton_iot",
                "content_hash": "different",
                "row": {"proto": "udp"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ClassifierTargetRecoveryError, match="content_hash"):
        recover_classifier_mapping_context([record], manifest_paths=[manifest])
