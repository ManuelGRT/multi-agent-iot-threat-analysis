from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.binary_balancing import (
    BinaryBalanceError,
    balance_binary_records,
    logical_binary_origin,
)
from src.eval.validation_campaign import PreparedRecord


TON_SOURCES = {
    "ton_iot_network": "data/TON_IOT/Train_Test_datasets/Train_Test_Network_dataset/network.csv",
    "ton_iot_telemetry": "data/TON_IOT/Train_Test_datasets/Train_Test_IoT_dataset/thermostat.csv",
    "ton_iot_linux": "data/TON_IOT/Train_Test_datasets/Train_Test_Linux_dataset/process.csv",
    "ton_iot_windows": "data/TON_IOT/Train_Test_datasets/Train_Test_Windows_dataset/windows.csv",
}


def _record(
    manifest_id: str,
    *,
    split: str,
    is_attack: bool,
    dataset: str = "ton_iot",
    source_file: str = TON_SOURCES["ton_iot_network"],
) -> PreparedRecord:
    return PreparedRecord(
        manifest_id=manifest_id,
        dataset=dataset,
        source_file=source_file,
        row_id=manifest_id,
        content_hash=f"content-{manifest_id}",
        split=split,
        tasks=("binary",),
        target_class="attack" if is_attack else "normal",
        target_is_attack=is_attack,
        event=None,
        features={"signal": float(is_attack)},
        feature_fingerprint=f"fp-{manifest_id}",
        provider="mistral",
        model="mistral-small-2603",
        mapping_confidence=0.95,
        latency_s=1.0,
        result_source_path=Path("results.jsonl"),
        result_line_number=1,
    )


def test_balances_every_split_and_all_four_ton_origins_deterministically():
    records: list[PreparedRecord] = []
    for split in ("train", "val", "test"):
        for origin, source in TON_SOURCES.items():
            for label, repetitions in ((False, 2), (True, 3)):
                for repetition in range(repetitions):
                    records.append(
                        _record(
                            f"{split}-{origin}-{label}-{repetition}",
                            split=split,
                            is_attack=label,
                            source_file=source,
                        )
                    )

    selected, report = balance_binary_records(records, seed=42)
    reversed_selected, reversed_report = balance_binary_records(
        list(reversed(records)), seed=42
    )

    assert [row.manifest_id for row in selected] == [
        row.manifest_id for row in reversed_selected
    ]
    assert report["selected_manifest_ids_sha256"] == reversed_report[
        "selected_manifest_ids_sha256"
    ]
    assert report["origins"] == sorted(TON_SOURCES)
    assert report["input_rows"] == 60
    assert report["output_rows"] == 48
    assert report["after"]["global"]["class_counts"] == {
        "normal": 24,
        "attack": 24,
    }
    for origin in TON_SOURCES:
        support = report["after"]["by_origin"][origin]
        assert support["class_counts"] == {"normal": 6, "attack": 6}
        assert all(
            values["class_counts"] == {"normal": 2, "attack": 2}
            for values in support["by_split"].values()
        )


@pytest.mark.parametrize(("origin", "source"), TON_SOURCES.items())
def test_logical_origin_expands_ton_modalities(origin: str, source: str):
    record = _record("sample", split="train", is_attack=False, source_file=source)
    assert logical_binary_origin(record) == origin


def test_unknown_ton_source_is_rejected_instead_of_silently_aggregated():
    record = _record(
        "unknown",
        split="train",
        is_attack=False,
        source_file="data/TON_IOT/unclassified.csv",
    )
    with pytest.raises(BinaryBalanceError, match="modalidad TON-IoT"):
        logical_binary_origin(record)


def test_missing_label_in_one_origin_split_is_rejected():
    records = [
        _record(f"{split}-normal", split=split, is_attack=False)
        for split in ("train", "val", "test")
    ]
    with pytest.raises(BinaryBalanceError, match="faltan clases"):
        balance_binary_records(records)
