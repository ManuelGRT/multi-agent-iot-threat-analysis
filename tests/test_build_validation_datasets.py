from __future__ import annotations

import pytest

from scripts import build_validation_datasets as builder


def _row(row_id: int, content_hash: str, split: str) -> dict:
    return {
        "source_file": (
            "data\\TON_IOT\\Train_Test_datasets\\Train_Test_IoT_dataset\\"
            "Train_Test_IoT_Fridge.csv"
        ),
        "row_id": row_id,
        "content_hash": content_hash,
        "class": "normal",
        "is_attack": False,
        "split": split,
    }


def test_curated_rows_are_verified_and_removed_after_split_assignment():
    first = _row(24_117, "d06787265d2f0b7bc938f3e956e4f2ee", "test")
    second = _row(26_045, "64280cb3f7d5cc77408266421c12a833", "train")
    retained_row = _row(99, "other", "val")

    retained, excluded = builder.exclude_curated_rows(
        "ton_iot",
        [first, retained_row, second],
    )

    assert retained == [retained_row]
    assert excluded == [first, second]
    assert [item["split"] for item in excluded] == ["test", "train"]


def test_curated_row_identity_includes_content_target_and_split():
    first = _row(24_117, "changed-content", "test")
    second = _row(26_045, "64280cb3f7d5cc77408266421c12a833", "train")

    with pytest.raises(RuntimeError, match="no coincide con su contrato"):
        builder.exclude_curated_rows("ton_iot", [first, second])


def test_official_seed_requires_both_curated_rows():
    first = _row(24_117, "d06787265d2f0b7bc938f3e956e4f2ee", "test")

    with pytest.raises(RuntimeError, match="distintas de las esperadas"):
        builder.exclude_curated_rows("ton_iot", [first])
