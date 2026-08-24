from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import eval_validacion_por_dataset as evaluation
from src.eval.validation_campaign import PreparedRecord


def _prepared(
    manifest_id: str,
    *,
    dataset: str = "sample",
    split: str = "train",
    task: str = "binary",
    target=True,
    fingerprint: str | None = None,
) -> PreparedRecord:
    return PreparedRecord(
        manifest_id=manifest_id,
        dataset=dataset,
        source_file="source.csv",
        row_id=manifest_id,
        content_hash=f"content-{manifest_id}",
        split=split,
        tasks=(task,),
        target_class=str(target) if task == "multiclass" else None,
        target_is_attack=bool(target) if task == "binary" else None,
        event=None,  # The evaluator consumes only the already-extracted features.
        features={"manifest_feature": manifest_id, "source_split": split},
        feature_fingerprint=fingerprint or f"fp-{manifest_id}",
        provider="mistral",
        model="mistral-small-2603",
        mapping_confidence=0.9,
        latency_s=1.0,
        result_source_path=Path("result.jsonl"),
        result_line_number=1,
    )


def _balanced_records(dataset: str, task: str) -> list[PreparedRecord]:
    records = []
    labels = (False, True) if task == "binary" else ("normal", "scan")
    for split in evaluation.SPLITS:
        for label_index, label in enumerate(labels):
            for repetition in range(2):
                manifest_id = f"{split}-{label_index}-{repetition}"
                records.append(
                    _prepared(
                        manifest_id,
                        dataset=dataset,
                        split=split,
                        task=task,
                        target=label,
                    )
                )
    return records


def _metric(f1: float) -> dict:
    return {
        "accuracy": f1,
        "precision_weighted": f1,
        "recall_weighted": f1,
        "f1_weighted": f1,
    }


def test_task_specific_dedup_removes_unsafe_groups_and_keeps_smallest_id():
    records = [
        _prepared("z-safe", fingerprint="same-safe"),
        _prepared("a-safe", fingerprint="same-safe"),
        _prepared("cross-train", split="train", fingerprint="cross"),
        _prepared("cross-test", split="test", fingerprint="cross"),
        _prepared("label-a", target=True, fingerprint="conflict"),
        _prepared("label-b", target=False, fingerprint="conflict"),
        _prepared("unique", split="val", fingerprint="unique"),
    ]

    kept, report = evaluation.deduplicate_task_records(records, task="binary")

    assert [record.manifest_id for record in kept] == ["a-safe", "unique"]
    assert report["same_split_same_label_groups"] == 1
    assert report["same_split_rows_removed"] == 1
    assert report["cross_split_groups"] == 1
    assert report["conflicting_label_groups"] == 1
    assert report["unsafe_rows_removed"] == 4
    assert report["final_overlap_count"] == 0
    groups = {item["feature_fingerprint"]: item for item in report["removed_groups"]}
    assert groups["cross"]["reasons"] == ["cross_split"]
    assert groups["conflict"]["reasons"] == ["conflicting_labels"]
    assert groups["same-safe"]["kept_manifest_id"] == "a-safe"


def test_evaluate_task_uses_frozen_splits_and_withholds_non_winner_test():
    records = _balanced_records("sample", "binary")
    captured = {}
    estimators = {"random_forest": object(), "xgboost": object()}
    injected_factory = object()

    def fake_runner(
        train_features,
        y_train,
        validation_features,
        y_validation,
        test_features,
        y_test,
        **kwargs,
    ):
        captured.update(
            train_features=train_features,
            y_train=y_train,
            validation_features=validation_features,
            y_validation=y_validation,
            test_features=test_features,
            y_test=y_test,
            kwargs=kwargs,
        )
        return SimpleNamespace(
            report={
                "selection": {"model": "random_forest"},
                "models": {
                    "random_forest": {
                        "status": "ok",
                        "validation": _metric(0.8),
                        "test": _metric(0.7),
                    },
                    "xgboost": {
                        "status": "ok",
                        "validation": _metric(0.6),
                        "test": _metric(0.99),
                    },
                },
            },
            estimators=estimators,
        )

    result = evaluation.evaluate_task(
        records,
        dataset="sample",
        task="binary",
        models=("random_forest", "xgboost"),
        benchmark_runner=fake_runner,
        mlp_backend_factory=injected_factory,
    )

    assert {row["source_split"] for row in captured["train_features"]} == {"train"}
    assert {row["source_split"] for row in captured["validation_features"]} == {"val"}
    assert {row["source_split"] for row in captured["test_features"]} == {"test"}
    assert captured["y_train"] == [False, False, True, True]
    assert captured["kwargs"]["positive_label"] is True
    assert captured["kwargs"]["mlp_backend_factory"] is injected_factory
    assert result["status"] == "ok"
    assert result["benchmark"]["models"]["random_forest"]["test"]["f1_weighted"] == 0.7
    assert "test" not in result["benchmark"]["models"]["xgboost"]
    assert result["benchmark"]["models"]["xgboost"]["test_withheld"] is True
    assert estimators == {}


def test_edge_multiclass_compares_only_preselected_winner_with_all_baselines():
    records = _balanced_records("edge_iiotset", "multiclass")

    def fake_runner(*_args, **_kwargs):
        return SimpleNamespace(
            report={
                "selection": {"model": "xgboost"},
                "models": {
                    "random_forest": {
                        "status": "ok",
                        "validation": _metric(0.7),
                        "test": _metric(0.2),
                    },
                    "xgboost": {
                        "status": "ok",
                        "validation": _metric(0.8),
                        "test": _metric(0.8),
                    },
                },
            },
            estimators={},
        )

    report = evaluation.evaluate_task(
        records,
        dataset="edge_iiotset",
        task="multiclass",
        models=("random_forest", "xgboost"),
        benchmark_runner=fake_runner,
    )

    assert "test" not in report["benchmark"]["models"]["random_forest"]
    comparison = report["jorge_comparison"]
    assert comparison["candidate"] == 0.8
    assert comparison["verdict"] == "exito_pleno"
    assert len(comparison["baselines"]) == 6
    by_name = {item["baseline"]: item for item in comparison["baselines"]}
    assert by_name["Jorge XGBoost"]["f1_weighted"] == 0.4987
    assert by_name["Replica estricta XGBoost"]["f1_weighted"] == 0.4993
    assert by_name["DeepSeek fine-tuned"]["f1_weighted"] == 0.7479


def test_rare_support_is_reported_as_degraded_but_not_always_blocking():
    records = _balanced_records("sample", "binary")
    records = [
        record
        for record in records
        if not (record.split == "test" and record.target_is_attack is True and record.manifest_id.endswith("-1"))
    ]
    supports = evaluation.split_class_supports(records, task="binary")
    assessment = evaluation.assess_task_support(supports)

    assert assessment["trainable"] is True
    assert assessment["degraded"] is True
    assert assessment["rare_classes"]["attack"]["supports"]["test"] == 1
    assert assessment["blocking_reasons"] == []


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _manifest(manifest_id: str, split: str = "train") -> dict:
    return {
        "manifest_id": manifest_id,
        "dataset": "urban_iot",
        "source_file": "urban.csv",
        "row_id": manifest_id,
        "content_hash": f"hash-{manifest_id}",
        "split": split,
        "tasks": ["standardization"],
        "target": {"class": None, "is_attack": None},
        "row": {"packets": manifest_id},
    }


def _result(manifest_id: str, packet_count: int, *, parsed_by_llm: bool = True) -> dict:
    return {
        "manifest_id": manifest_id,
        "ok": True,
        "parsed_by_llm": parsed_by_llm,
        "provider": "mistral",
        "model": "mistral-small-2603",
        "latency_s": 1.0,
        "mapping_confidence": 0.9,
        "canonical_event": {
            "event_id": f"event-{manifest_id}",
            "modality": "network_flow",
            "packet_count": packet_count,
            "semantic_text": "urban flow",
            "provenance": {
                "dataset": "non-authoritative",
                "source_file": "other.csv",
                "row_id": "other",
                "split": "stream",
            },
            "mapping_confidence": 0.9,
        },
    }


def test_validate_only_cli_writes_self_contained_global_and_urban_artifacts(tmp_path):
    manifests = tmp_path / "manifests"
    results = tmp_path / "results"
    out = tmp_path / "out"
    rows = [_manifest("train", "train"), _manifest("val", "val"), _manifest("test", "test")]
    _write_jsonl(manifests / "urban_iot_manifest.jsonl", rows)
    _write_jsonl(
        results / "urban_iot_standardized.jsonl",
        [_result("train", 1), _result("val", 2), _result("test", 3)],
    )

    def forbidden_training(*_args, **_kwargs):
        raise AssertionError("--validate-only must not train")

    exit_code = evaluation.main(
        [
            "--manifest-dir",
            str(manifests),
            "--results-dir",
            str(results),
            "--out-dir",
            str(out),
            "--validate-only",
            "--dataset",
            "urban_iot",
        ],
        benchmark_runner=forbidden_training,
    )

    assert exit_code == 0
    global_report = json.loads((out / "evaluation_global.json").read_text(encoding="utf-8"))
    urban_report = json.loads((out / "urban_iot_evaluation.json").read_text(encoding="utf-8"))
    assert (out / "evaluation_global.md").is_file()
    assert (out / "urban_iot_evaluation.md").is_file()
    assert global_report["status"] == "validated_only"
    assert global_report["comparable"] is True
    assert len(global_report["input_snapshot"]["manifest_files"]) == 1
    assert len(global_report["input_snapshot"]["manifest_files"][0]["sha256"]) == 64
    assert "scikit-learn" in global_report["environment"]["packages"]
    assert urban_report["dataset"]["status"] == "standardization_only"
    assert urban_report["dataset"]["tasks"] == {}


def test_strict_cli_rejects_fallback_and_opt_in_marks_report_non_comparable(tmp_path, capsys):
    manifests = tmp_path / "manifests"
    results = tmp_path / "results"
    strict_out = tmp_path / "strict"
    permissive_out = tmp_path / "permissive"
    _write_jsonl(manifests / "urban_iot_manifest.jsonl", [_manifest("one")])
    _write_jsonl(
        results / "urban_iot_standardized.jsonl",
        [_result("one", 1, parsed_by_llm=False)],
    )
    common = ["--manifest-dir", str(manifests), "--results-dir", str(results), "--validate-only"]

    assert evaluation.main([*common, "--out-dir", str(strict_out)]) == 1
    assert "incompleta" in capsys.readouterr().err
    assert not (strict_out / "evaluation_global.json").exists()

    assert (
        evaluation.main(
            [
                *common,
                "--out-dir",
                str(permissive_out),
                "--allow-incomplete",
                "--include-fallbacks",
            ]
        )
        == 0
    )
    report = json.loads(
        (permissive_out / "evaluation_global.json").read_text(encoding="utf-8")
    )
    assert report["comparable"] is False
    assert report["non_comparable_reasons"] == ["allow_incomplete", "include_fallbacks"]


def test_cli_contract_supports_repeated_filters_csv_models_and_injected_backend():
    args = evaluation.build_parser().parse_args(
        [
            "--dataset",
            "edge_iiotset",
            "--dataset",
            "bot_iot",
            "--task",
            "binary",
            "--task",
            "multiclass",
            "--models",
            "random_forest,xgboost",
            "--seed",
            "42",
            "--mlp-backend",
            "injected",
        ]
    )

    assert args.dataset == ["edge_iiotset", "bot_iot"]
    assert args.task == ["binary", "multiclass"]
    assert args.models == ("random_forest", "xgboost")
    assert args.seed == 42
    assert args.mlp_backend == "injected"


def test_snapshot_detects_content_changes_even_when_file_name_is_stable(tmp_path):
    manifest = _write_jsonl(tmp_path / "sample_manifest.jsonl", [{"id": 1}])
    before = evaluation.snapshot_inputs([manifest], [])
    _write_jsonl(manifest, [{"id": 2}])
    after = evaluation.snapshot_inputs([manifest], [])

    with pytest.raises(evaluation.EvaluationError, match="cambiaron"):
        evaluation.assert_inputs_unchanged(before, after)
