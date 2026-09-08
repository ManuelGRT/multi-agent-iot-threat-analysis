from __future__ import annotations

from pathlib import Path

import pytest

from src.contracts.attack_taxonomy import MULTIDATASET_ATTACK_CLASSES
from src.eval.classifier_balancing import (
    MappedClassifierRecord,
    RejectedClassifierRecord,
    balanced_origin_test_views,
    build_balanced_stratified_classifier_corpus,
    classifier_origins,
    deduplicate_classifier_universe,
    map_multidataset_classifier_records,
)
from src.eval.classifier_targets import MappingContext
from src.eval.validation_campaign import PreparedRecord


def _prepared(
    manifest_id: str,
    *,
    dataset: str,
    split: str,
    target_class: str,
    fingerprint: str | None = None,
    source_file: str | None = None,
    is_attack: bool | None = True,
) -> PreparedRecord:
    return PreparedRecord(
        manifest_id=manifest_id,
        dataset=dataset,
        source_file=source_file or f"data/{dataset}.csv",
        row_id=manifest_id,
        content_hash=f"content-{manifest_id}",
        split=split,  # type: ignore[arg-type]
        tasks=("multiclass",),
        target_class=target_class,
        target_is_attack=is_attack,
        event=None,  # type: ignore[arg-type]
        features={"signal": float(len(manifest_id))},
        feature_fingerprint=fingerprint or f"fp-{manifest_id}",
        provider="mistral",
        model="mistral-small-2603",
        mapping_confidence=0.95,
        latency_s=1.0,
        result_source_path=Path("result.jsonl"),
        result_line_number=1,
    )


def _mapped_corpus() -> list[MappedClassifierRecord]:
    rows: list[MappedClassifierRecord] = []
    needs = {"train": 15, "val": 4, "test": 4}
    for split, count in needs.items():
        for label in MULTIDATASET_ATTACK_CLASSES:
            for index in range(count):
                dataset = "ton_iot" if index % 2 else "edge_iiotset"
                source = (
                    "data/TON_IOT/Train_Test_Windows_dataset/windows.csv"
                    if dataset == "ton_iot"
                    else "data/EDGE_IIOTSET/edge.csv"
                )
                prepared = _prepared(
                    f"{split}-{label}-{index}",
                    dataset=dataset,
                    split=split,
                    target_class=label,
                    source_file=source,
                )
                dataset_origin, detailed_origin = classifier_origins(dataset, source)
                rows.append(
                    MappedClassifierRecord(
                        record=prepared,
                        label=label,
                        dataset_origin=dataset_origin,
                        detailed_origin=detailed_origin,
                        mapping_reason="test",
                    )
                )
    return rows


def test_classifier_origin_splits_all_four_ton_modalities():
    assert classifier_origins("ton_iot", "x/Train_Test_Network_dataset/a.csv")[1] == "ton_iot_network"
    assert classifier_origins("ton_iot", "x/Train_Test_Linux_dataset/a.csv")[1] == "ton_iot_linux"
    assert classifier_origins("ton_iot", "x/Train_Test_Windows_dataset/a.csv")[1] == "ton_iot_windows"
    assert classifier_origins("ton_iot", "x/Train_Test_IoT_dataset/a.csv")[1] == "ton_iot_telemetry"
    with pytest.raises(ValueError, match="modalidad desconocida"):
        classifier_origins("ton_iot", "x/unclassified/a.csv")


def test_deduplication_happens_globally_and_removes_unsafe_groups():
    corpus = _mapped_corpus()[:2]
    base = corpus[0]
    duplicate = MappedClassifierRecord(
        record=_prepared(
            "duplicate-test",
            dataset="edge_iiotset",
            split="test",
            target_class=base.label,
            fingerprint=base.record.feature_fingerprint,
        ),
        label=base.label,
        dataset_origin="edge_iiotset",
        detailed_origin="edge_iiotset",
        mapping_reason="test",
    )
    kept, rejected, report = deduplicate_classifier_universe(
        [*corpus, duplicate], ()
    )
    assert rejected == ()
    assert report["cross_split_groups"] == 1
    assert report["unsafe_rows_removed"] == 2
    assert len(kept) == 1


def test_origin_views_balance_only_classes_present_in_each_origin():
    corpus = _mapped_corpus()
    views = balanced_origin_test_views(corpus, detailed=True, seed=42)
    assert set(views) == {"edge_iiotset", "ton_iot_windows"}
    for rows in views.values():
        counts = {}
        for item in rows:
            counts[item.label] = counts.get(item.label, 0) + 1
        assert len(counts) == len(MULTIDATASET_ATTACK_CLASSES)
        assert len(set(counts.values())) == 1


def test_origin_views_can_exclude_classes_below_minimum_support():
    corpus = _mapped_corpus()
    for label in MULTIDATASET_ATTACK_CLASSES:
        record = _prepared(
            f"test-extra-{label}",
            dataset="edge_iiotset",
            split="test",
            target_class=label,
        )
        corpus.append(
            MappedClassifierRecord(
                record=record,
                label=label,
                dataset_origin="edge_iiotset",
                detailed_origin="edge_iiotset",
                mapping_reason="test",
            )
        )
    views = balanced_origin_test_views(
        corpus,
        detailed=True,
        seed=42,
        minimum_class_support=3,
    )
    assert set(views) == {"edge_iiotset"}
    assert len(views["edge_iiotset"]) == len(MULTIDATASET_ATTACK_CLASSES) * 3
    with pytest.raises(ValueError, match="entero positivo"):
        balanced_origin_test_views(corpus, minimum_class_support=0)


def test_deduplication_covers_exact_and_rejected_ood_across_splits():
    exact = _mapped_corpus()[0]
    rejected = RejectedClassifierRecord(
        record=_prepared(
            "ood-test",
            dataset="iot23",
            split="test",
            target_class="C&C",
            fingerprint=exact.record.feature_fingerprint,
        ),
        status="out_of_taxonomy",
        reason="test_out_of_taxonomy",
    )
    mapped, ood, report = deduplicate_classifier_universe([exact], [rejected])
    assert mapped == ()
    assert ood == ()
    assert report["cross_split_groups"] == 1
    assert report["unsafe_rows_removed"] == 2


def test_mapping_fails_closed_on_contradictory_or_missing_supervision():
    benign_attack = _prepared(
        "normal-as-attack",
        dataset="edge_iiotset",
        split="train",
        target_class="Normal",
        is_attack=True,
    )
    missing = _prepared(
        "missing-target",
        dataset="edge_iiotset",
        split="train",
        target_class="DDoS_TCP",
        is_attack=None,
    )
    with pytest.raises(ValueError, match="clase benigna"):
        map_multidataset_classifier_records([benign_attack])
    with pytest.raises(ValueError, match="target_is_attack invalido"):
        map_multidataset_classifier_records([missing])


def test_multidataset_mapper_keeps_forced_status_and_rejects_missing_ddos_context():
    records = [
        _prepared(
            "ton-ddos-tcp",
            dataset="ton_iot",
            split="train",
            target_class="ddos",
            source_file="x/Train_Test_Network_dataset/a.csv",
        ),
        _prepared(
            "ton-ddos-missing",
            dataset="ton_iot",
            split="train",
            target_class="ddos",
            source_file="x/Train_Test_IoT_dataset/a.csv",
        ),
        _prepared(
            "ton-injection",
            dataset="ton_iot",
            split="train",
            target_class="injection",
            source_file="x/Train_Test_Network_dataset/a.csv",
        ),
        _prepared(
            "bot-service",
            dataset="bot_iot",
            split="train",
            target_class="Reconnaissance",
        ),
        _prepared(
            "iot-c2",
            dataset="iot23",
            split="train",
            target_class="C&C-Torii",
        ),
    ]
    mapped, rejected, report = map_multidataset_classifier_records(
        records,
        native_subclasses={"bot-service": "Service_Scan"},
        mapping_contexts={
            "ton-ddos-tcp": MappingContext("tcp", None),
            "ton-ddos-missing": MappingContext(None, None),
        },
    )

    by_id = {item.record.manifest_id: item for item in mapped}
    assert by_id["ton-ddos-tcp"].label == "DDoS_TCP"
    assert by_id["ton-ddos-tcp"].mapping_status == "forced"
    assert by_id["ton-injection"].label == "SQL_injection"
    assert by_id["bot-service"].label == "Port_Scanning"
    assert by_id["iot-c2"].label == "Command_and_Control"
    assert len(rejected) == 1
    assert rejected[0].reason == "ton_ddos_missing_raw_protocol"
    assert report["status_counts"] == {"ambiguous": 1, "exact": 1, "forced": 3}


def _resplit_rows(
    label: str,
    count: int,
    *,
    dataset: str,
    source_file: str | None = None,
    source_split: str = "train",
) -> list[MappedClassifierRecord]:
    rows: list[MappedClassifierRecord] = []
    source = source_file or f"data/{dataset}.csv"
    dataset_origin, detailed_origin = classifier_origins(dataset, source)
    for index in range(count):
        prepared = _prepared(
            f"resplit-{label}-{dataset}-{Path(source).parent.name}-{index}",
            dataset=dataset,
            split=source_split,
            target_class=label,
            source_file=source,
        )
        rows.append(
            MappedClassifierRecord(
                record=prepared,
                label=label,
                dataset_origin=dataset_origin,
                detailed_origin=detailed_origin,
                mapping_reason="test",
            )
        )
    return rows


def test_current_protocol_defaults_to_the_16_type_taxonomy():
    corpus = [
        item
        for label in MULTIDATASET_ATTACK_CLASSES
        for item in _resplit_rows(label, 20, dataset="edge_iiotset")
    ]

    selected, report = build_balanced_stratified_classifier_corpus(
        corpus,
        per_class_total=20,
        seed=42,
    )

    assert report["classes"] == list(MULTIDATASET_ATTACK_CLASSES)
    assert report["quotas_per_class"] == {"train": 14, "val": 3, "test": 3}
    assert len(selected) == len(MULTIDATASET_ATTACK_CLASSES) * 20


def test_current_protocol_rejects_invalid_exact_split_quota():
    corpus = [
        item
        for label in MULTIDATASET_ATTACK_CLASSES
        for item in _resplit_rows(label, 20, dataset="edge_iiotset")
    ]

    with pytest.raises(ValueError, match="multiplo de 20"):
        build_balanced_stratified_classifier_corpus(corpus, per_class_total=19)
    with pytest.raises(ValueError, match="entero positivo"):
        build_balanced_stratified_classifier_corpus(corpus, per_class_total=20.0)
    with pytest.raises(ValueError, match="entero positivo"):
        build_balanced_stratified_classifier_corpus(corpus, per_class_total=True)


def test_new_protocol_uses_global_minimum_then_splits_exactly_by_class():
    corpus = [
        *_resplit_rows("A", 100, dataset="edge_iiotset"),
        *_resplit_rows("A", 20, dataset="iot23"),
        *_resplit_rows("B", 100, dataset="bot_iot"),
    ]

    selected, report = build_balanced_stratified_classifier_corpus(
        corpus,
        classes=("A", "B"),
        seed=42,
    )

    assert report["minimum_global_class_support"] == 100
    assert report["limiting_classes"] == ["B"]
    assert report["per_class_total"] == 100
    assert report["quotas_per_class"] == {"train": 70, "val": 15, "test": 15}
    assert len(selected) == 200
    assert report["split_membership_preserved"] is False
    assert report["source_split_used_for_corpus_selection"] is False
    assert report["source_split_used_for_assignment"] is False
    assert report["source_to_classifier_split"] == {
        "train": {"test": 30, "train": 140, "val": 30}
    }
    class_a_selection = report["corpus_selection"]["classes"]["A"]
    assert class_a_selection["edge_iiotset"]["selected"] == 83
    assert class_a_selection["iot23"]["selected"] == 17
    for split, quota in report["quotas_per_class"].items():
        assert report["selected"]["splits"][split]["class_counts"] == {
            "A": quota,
            "B": quota,
        }


def test_new_protocol_is_deterministic_and_ignores_source_split_distribution():
    corpus = [
        *_resplit_rows("A", 100, dataset="edge_iiotset", source_split="train"),
        *_resplit_rows("B", 100, dataset="bot_iot", source_split="test"),
    ]
    first, report = build_balanced_stratified_classifier_corpus(
        corpus, classes=("A", "B"), seed=91
    )
    second, second_report = build_balanced_stratified_classifier_corpus(
        tuple(reversed(corpus)), classes=("A", "B"), seed=91
    )

    assert [
        (item.record.manifest_id, item.record.split) for item in first
    ] == [
        (item.record.manifest_id, item.record.split) for item in second
    ]
    assert report["selection_sha256"] == second_report["selection_sha256"]
    assert report["final_cross_split_overlap_count"] == 0


def test_new_protocol_origin_cell_of_100_is_exactly_70_15_15():
    corpus = _resplit_rows("A", 100, dataset="edge_iiotset")
    _selected, report = build_balanced_stratified_classifier_corpus(
        corpus, classes=("A",), seed=42
    )

    origin = report["split_assignment"]["classes"]["A"]["dataset_origins"][
        "edge_iiotset"
    ]
    assert origin["rows"] == 100
    assert origin["split_counts"] == {"train": 70, "val": 15, "test": 15}


def test_new_protocol_stratifies_ton_detailed_origins_and_rejects_oversize():
    corpus = [
        *_resplit_rows(
            "A",
            50,
            dataset="ton_iot",
            source_file="x/Train_Test_Network_dataset/a.csv",
        ),
        *_resplit_rows(
            "A",
            50,
            dataset="ton_iot",
            source_file="x/Train_Test_Windows_dataset/a.csv",
        ),
    ]
    _selected, report = build_balanced_stratified_classifier_corpus(
        corpus, classes=("A",), seed=42
    )
    details = report["split_assignment"]["classes"]["A"]["dataset_origins"][
        "ton_iot"
    ]["detailed_origins"]

    assert set(details) == {"ton_iot_network", "ton_iot_windows"}
    assert sum(value["split_counts"]["train"] for value in details.values()) == 70
    assert sum(value["split_counts"]["val"] for value in details.values()) == 15
    assert sum(value["split_counts"]["test"] for value in details.values()) == 15
    for detail in details.values():
        assert detail["rows"] == 50
        assert set(detail["split_counts"].values()) <= {7, 8, 35}

    with pytest.raises(ValueError, match="superior al minimo global"):
        build_balanced_stratified_classifier_corpus(
            corpus, classes=("A",), per_class_total=120
        )
