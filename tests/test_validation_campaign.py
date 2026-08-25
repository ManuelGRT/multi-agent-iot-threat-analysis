from __future__ import annotations

import json
import math

import pytest

from src.eval import validation_campaign as campaign


def _event(
    event_id: str,
    *,
    packet_count: int = 1,
    mapping_confidence: float = 0.9,
) -> dict:
    return {
        "event_id": event_id,
        "modality": "network_flow",
        "packet_count": packet_count,
        "semantic_text": "flujo de red",
        "provenance": {
            "dataset": "valor_no_autoritativo",
            "source_file": "otro.csv",
            "row_id": "otro",
            "split": "stream",
        },
        "mapping_confidence": mapping_confidence,
    }


def _manifest(
    manifest_id: str,
    *,
    content_hash: str,
    split: str = "train",
    dataset: str = "sample",
    tasks: list[str] | None = None,
    target_class: str = "scan",
    target_is_attack: bool | None = True,
    row: dict | None = None,
) -> dict:
    return {
        "manifest_id": manifest_id,
        "dataset": dataset,
        "source_file": "source.csv",
        "row_id": manifest_id,
        "content_hash": content_hash,
        "split": split,
        "tasks": tasks or ["binary", "multiclass"],
        "target": {"class": target_class, "is_attack": target_is_attack},
        "row": row or {"packets": "3"},
    }


def _result(
    manifest_id: str,
    *,
    event_id: str | None = None,
    packet_count: int = 1,
    ok: bool = True,
    parsed_by_llm: bool = True,
    provider: str = "mistral",
    model: str = "fixed-model",
) -> dict:
    result = {
        "manifest_id": manifest_id,
        "ok": ok,
        "parsed_by_llm": parsed_by_llm,
        "provider": provider,
        "model": model,
        "latency_s": 2.0,
    }
    if ok:
        result["mapping_confidence"] = 0.9
        result["canonical_event"] = _event(
            event_id or f"event-{manifest_id}",
            packet_count=packet_count,
        )
    else:
        result["error"] = "HTTPStatusError"
    return result


def _write_jsonl(path, records, *, final_newline: bool = True):
    text = "\n".join(json.dumps(record) for record in records)
    if final_newline and records:
        text += "\n"
    path.write_text(text, encoding="utf-8")
    return path


def test_manifests_enforce_global_id_and_content_hash_uniqueness(tmp_path):
    first = _write_jsonl(
        tmp_path / "a.jsonl",
        [_manifest("row-a", content_hash="hash-a")],
    )
    duplicate_id = _write_jsonl(
        tmp_path / "b.jsonl",
        [_manifest("row-a", content_hash="hash-b")],
    )

    with pytest.raises(campaign.ManifestContractError, match="manifest_id duplicado"):
        campaign.load_manifests([first, duplicate_id])

    duplicate_hash = _write_jsonl(
        tmp_path / "c.jsonl",
        [_manifest("row-c", content_hash="hash-a")],
    )
    with pytest.raises(campaign.ManifestContractError, match="content_hash duplicado"):
        campaign.load_manifests([first, duplicate_hash])


def test_manifest_rejects_unknown_tasks_missing_targets_and_row_leakage(tmp_path):
    cases = [
        (_manifest("a", content_hash="a", tasks=["unknown"]), "tarea desconocida"),
        (
            _manifest("b", content_hash="b", tasks=["binary"], target_is_attack=None),
            "binary requiere",
        ),
        (
            _manifest("c", content_hash="c", row={"nested": {"ground_truth": 1}}),
            "target/leak",
        ),
    ]
    for index, (record, message) in enumerate(cases):
        path = _write_jsonl(tmp_path / f"bad-{index}.jsonl", [record])
        with pytest.raises(campaign.ManifestContractError, match=message):
            campaign.load_manifests(path)


def test_results_tolerate_only_a_truncated_unterminated_final_line(tmp_path):
    tolerated = tmp_path / "tolerated.jsonl"
    tolerated.write_text(
        json.dumps(_result("good")) + "\n" + '{"manifest_id":"partial"',
        encoding="utf-8",
    )
    index = campaign.load_results(tolerated)

    assert index.total_attempts == 1
    assert index.truncated_final_lines == 1
    assert set(index.latest_attempt) == {"good"}

    corrupt_middle = tmp_path / "corrupt-middle.jsonl"
    corrupt_middle.write_text(
        json.dumps(_result("one")) + "\n" + "{broken\n" + json.dumps(_result("two")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(campaign.JsonlFormatError, match="corrupt-middle.jsonl:2"):
        campaign.load_results(corrupt_middle)

    corrupt_terminated_tail = tmp_path / "corrupt-tail.jsonl"
    corrupt_terminated_tail.write_text("{broken\n", encoding="utf-8")
    with pytest.raises(campaign.JsonlFormatError):
        campaign.load_results(corrupt_terminated_tail)


def test_latest_fallback_does_not_overwrite_latest_valid_llm(tmp_path):
    results_path = _write_jsonl(
        tmp_path / "results.jsonl",
        [
            _result("row", event_id="llm-event"),
            _result("row", event_id="fallback-event", parsed_by_llm=False),
        ],
    )
    index = campaign.load_results(results_path)

    assert index.latest_attempt["row"].canonical_event.event_id == "fallback-event"
    assert index.latest_valid_llm["row"].canonical_event.event_id == "llm-event"
    assert index.attempts_per_manifest == {"row": 2}

    manifests = campaign.load_manifests(
        _write_jsonl(tmp_path / "manifest.jsonl", [_manifest("row", content_hash="hash")])
    )
    prepared = campaign.prepare_validation_campaign(manifests, index)
    assert prepared.records[0].event.event_id == "llm-event"
    assert prepared.quality.global_metrics.latest_fallback_rows == 1
    assert prepared.quality.global_metrics.valid_llm_rows == 1


def test_requested_backend_keeps_its_latest_llm_despite_later_attempts(tmp_path):
    results_path = _write_jsonl(
        tmp_path / "results-by-backend.jsonl",
        [
            _result("row", event_id="model-a-first", model="model-a"),
            _result(
                "row",
                event_id="model-a-fallback",
                parsed_by_llm=False,
                model="model-a",
            ),
            _result("row", event_id="model-b-later", model="model-b"),
        ],
    )
    results = campaign.load_results(results_path)
    backend_key = ("row", "mistral", "model-a")

    assert results.latest_valid_llm["row"].canonical_event.event_id == "model-b-later"
    assert (
        results.latest_valid_llm_by_backend[backend_key].canonical_event.event_id
        == "model-a-first"
    )

    manifests = campaign.load_manifests(
        _write_jsonl(tmp_path / "manifest.jsonl", [_manifest("row", content_hash="hash")])
    )
    prepared = campaign.prepare_validation_campaign(
        manifests,
        results,
        campaign.PreflightRequirements(provider="MISTRAL", model="model-a"),
    )

    assert prepared.records[0].event.event_id == "model-a-first"
    assert prepared.records[0].model == "model-a"
    assert prepared.quality.global_metrics.provider_mismatch_rows == 0
    assert prepared.quality.global_metrics.model_mismatch_rows == 0


def test_result_requires_a_valid_canonical_event(tmp_path):
    invalid = _result("row")
    invalid["canonical_event"] = {"event_id": "missing-most-required-fields"}
    path = _write_jsonl(tmp_path / "invalid.jsonl", [invalid])

    with pytest.raises(campaign.ResultContractError, match="CanonicalEvent invalido"):
        campaign.load_results(path)


def test_join_uses_manifest_as_authority_and_only_canonical_extractor(
    tmp_path,
    monkeypatch,
):
    manifest_path = _write_jsonl(
        tmp_path / "manifest.jsonl",
        [
            _manifest(
                "row",
                content_hash="hash",
                dataset="authoritative_dataset",
                split="test",
                tasks=["binary"],
                target_class="authoritative_class",
                target_is_attack=False,
            )
        ],
    )
    result_path = _write_jsonl(tmp_path / "results.jsonl", [_result("row")])
    seen = []

    def fake_event_features(event):
        seen.append(event)
        return {"safe_feature": 7}

    monkeypatch.setattr(campaign.canonical_features, "event_features", fake_event_features)
    prepared = campaign.load_validation_campaign(manifest_path, result_path)
    record = prepared.records[0]

    assert len(seen) == 1
    assert record.features == {"safe_feature": 7}
    assert record.dataset == "authoritative_dataset"
    assert record.split == "test"
    assert record.tasks == ("binary",)
    assert record.target_for("binary") is False
    assert record.event.provenance.dataset == "valor_no_autoritativo"
    assert record.event.provenance.split == "stream"


def test_requirements_can_enforce_completion_llm_provider_and_model(tmp_path):
    manifests = campaign.load_manifests(
        _write_jsonl(
            tmp_path / "manifest.jsonl",
            [
                _manifest("one", content_hash="h1"),
                _manifest("two", content_hash="h2"),
            ],
        )
    )
    results = campaign.load_results(
        _write_jsonl(
            tmp_path / "results.jsonl",
            [_result("one", provider="mistral", model="model-a")],
        )
    )
    requirements = campaign.PreflightRequirements(
        provider="mistral",
        model="model-a",
    )
    with pytest.raises(campaign.PreflightError, match="seleccionadas=1/2") as captured:
        campaign.prepare_validation_campaign(manifests, results, requirements)
    assert captured.value.report.global_metrics.missing_rows == 1

    partial = campaign.prepare_validation_campaign(
        manifests,
        results,
        campaign.PreflightRequirements(
            require_complete=False,
            provider="MISTRAL",
            model="model-a",
        ),
    )
    assert len(partial.records) == 1
    assert partial.quality.global_metrics.selected_coverage_rate == 0.5

    wrong_model = campaign.PreflightRequirements(
        require_complete=False,
        provider="mistral",
        model="model-b",
    )
    filtered = campaign.prepare_validation_campaign(manifests, results, wrong_model)
    assert filtered.records == ()
    assert filtered.quality.global_metrics.model_mismatch_rows == 1


def test_require_llm_false_allows_fallback_but_true_does_not(tmp_path):
    manifests = campaign.load_manifests(
        _write_jsonl(tmp_path / "manifest.jsonl", [_manifest("row", content_hash="hash")])
    )
    results = campaign.load_results(
        _write_jsonl(
            tmp_path / "results.jsonl",
            [_result("row", parsed_by_llm=False)],
        )
    )

    with pytest.raises(campaign.PreflightError):
        campaign.prepare_validation_campaign(manifests, results)

    allowed = campaign.prepare_validation_campaign(
        manifests,
        results,
        campaign.PreflightRequirements(require_llm=False),
    )
    assert len(allowed.records) == 1
    assert allowed.records[0].event.event_id == "event-row"


def test_orphan_results_are_configurable(tmp_path):
    manifests = campaign.load_manifests(
        _write_jsonl(tmp_path / "manifest.jsonl", [_manifest("known", content_hash="hash")])
    )
    results = campaign.load_results(
        _write_jsonl(
            tmp_path / "results.jsonl",
            [_result("known"), _result("orphan")],
        )
    )
    with pytest.raises(campaign.PreflightError, match="ajenos"):
        campaign.prepare_validation_campaign(manifests, results)

    allowed = campaign.prepare_validation_campaign(
        manifests,
        results,
        campaign.PreflightRequirements(reject_orphan_results=False),
    )
    assert allowed.quality.orphan_result_rows == 1


def test_feature_contract_rejects_labels_target_keys_and_non_finite_values(tmp_path):
    with pytest.raises(campaign.FeatureSafetyError, match="NaN/Inf"):
        campaign.feature_fingerprint({"safe": math.inf})
    with pytest.raises(campaign.FeatureSafetyError, match="target/leak"):
        campaign.feature_fingerprint({"ground_truth": 1})
    with pytest.raises(campaign.FeatureSafetyError, match="target/leak"):
        campaign.feature_fingerprint({"origin.dataset": "sample"})

    manifests = campaign.load_manifests(
        _write_jsonl(tmp_path / "manifest.jsonl", [_manifest("row", content_hash="hash")])
    )
    leaking = _result("row")
    leaking["canonical_event"]["label_raw"] = "attack"
    results = campaign.load_results(_write_jsonl(tmp_path / "results.jsonl", [leaking]))
    with pytest.raises(campaign.FeatureSafetyError, match="contiene targets"):
        campaign.prepare_validation_campaign(manifests, results)


def test_technical_context_paths_are_not_confused_with_attack_targets(tmp_path):
    manifest_path = _write_jsonl(
        tmp_path / "manifest.jsonl",
        [_manifest("row", content_hash="hash")],
    )
    safe = _result("row")
    safe["canonical_event"].update(
        service_context={
            "protocol_family": "icmp",
            "dns": {
                "query_class": "IN",
                "query": {"type": "AAAA"},
                "qry": {"type": "A"},
                "type": "response",
            },
            "mqtt": {"message": {"type": "PUBLISH"}},
            "icmp": {"type": 8},
            "process": {"type": "system"},
        },
        host_context={
            "interface": {"type": "ethernet"},
            "metric_category": "network",
        },
        telemetry_context={
            "dns": {"query_class": "IN", "query": {"type": "A"}},
            "mqtt_message": {"type": "CONNECT"},
            "iot_sensor": {"type": "temperature"},
            "sensor_data": {"type": "float"},
            "geolocation": {"type": "point"},
        },
        feature_groups={"other": ["execution_class"]},
    )
    result_path = _write_jsonl(tmp_path / "safe-results.jsonl", [safe])

    prepared = campaign.load_validation_campaign(manifest_path, result_path)

    assert prepared.records[0].features["context.service.protocol_family.icmp"] == 1

    for index, (container, field, expected_path) in enumerate(
        (
            ("service_context", "attack_family", "service_context.attack_family"),
            ("telemetry_context", "protocol_family", "telemetry_context.protocol_family"),
            ("host_context", "model_family", "host_context.model_family"),
        )
    ):
        leaking = _result("row")
        leaking["canonical_event"][container] = {field: "ddos"}
        leaking_path = _write_jsonl(tmp_path / f"leaking-results-{index}.jsonl", [leaking])
        with pytest.raises(campaign.FeatureSafetyError, match=expected_path):
            campaign.load_validation_campaign(manifest_path, leaking_path)


def test_feature_fingerprint_is_deterministic_and_numeric_semantic(tmp_path):
    first = campaign.feature_fingerprint({"z": "á", "a": 1})
    reordered = campaign.feature_fingerprint({"a": 1.0, "z": "a\u0301"})
    bool_equivalent = campaign.feature_fingerprint({"z": "á", "a": True})

    assert first == reordered == bool_equivalent
    assert len(first) == 64


def test_explicit_dedup_is_deterministic_and_excludes_full_cross_split_group(tmp_path):
    manifest_rows = [
        _manifest("b-same", content_hash="h1", split="train"),
        _manifest("a-same", content_hash="h2", split="train"),
        _manifest("cross-train", content_hash="h3", split="train"),
        _manifest("cross-test", content_hash="h4", split="test"),
        _manifest("unique", content_hash="h5", split="test"),
    ]
    result_rows = [
        _result("b-same", packet_count=1),
        _result("a-same", packet_count=1),
        _result("cross-train", packet_count=2),
        _result("cross-test", packet_count=2),
        _result("unique", packet_count=3),
    ]
    prepared = campaign.load_validation_campaign(
        _write_jsonl(tmp_path / "manifest.jsonl", manifest_rows),
        _write_jsonl(tmp_path / "results.jsonl", result_rows),
    )

    assert len(prepared.records) == 5
    assert prepared.records == prepared.joined_records
    assert prepared.deduplication.applied is False
    assert prepared.deduplication.output_rows == 5

    deduplicated, report = campaign.deduplicate_records(
        prepared.records,
        task="binary",
    )
    assert [record.manifest_id for record in deduplicated] == ["a-same", "unique"]
    assert report.same_split_duplicate_groups == 1
    assert report.same_split_removed_ids == ("b-same",)
    assert report.cross_split_groups == 1
    assert report.cross_split_removed_ids == (
        "cross-test",
        "cross-train",
    )
    assert prepared.quality.global_metrics.selected_rows == 5
    assert prepared.summary()["ready_rows"] == 5
    assert prepared.summary()["records_are_authoritative_join"] is True


def test_prepare_preserves_same_features_with_distinct_tasks(tmp_path):
    manifest_rows = [
        _manifest(
            "binary-row",
            content_hash="binary-hash",
            tasks=["binary"],
            target_is_attack=False,
        ),
        _manifest(
            "multiclass-row",
            content_hash="multiclass-hash",
            tasks=["multiclass"],
            target_class="scan",
        ),
    ]
    result_rows = [_result("binary-row"), _result("multiclass-row")]
    prepared = campaign.load_validation_campaign(
        _write_jsonl(tmp_path / "manifest-tasks.jsonl", manifest_rows),
        _write_jsonl(tmp_path / "results-tasks.jsonl", result_rows),
    )

    assert {record.manifest_id for record in prepared.records} == {
        "binary-row",
        "multiclass-row",
    }
    assert len({record.feature_fingerprint for record in prepared.records}) == 1
    with pytest.raises(campaign.DeduplicationConflictError, match="supervision incompatible"):
        campaign.deduplicate_records(prepared.records)

    binary_only, _ = campaign.deduplicate_records(prepared.records, task="binary")
    assert [record.manifest_id for record in binary_only] == ["binary-row"]


def test_explicit_dedup_rejects_conflicting_labels(tmp_path):
    prepared = campaign.load_validation_campaign(
        _write_jsonl(
            tmp_path / "manifest-conflict.jsonl",
            [
                _manifest(
                    "benign",
                    content_hash="benign-hash",
                    tasks=["binary"],
                    target_is_attack=False,
                ),
                _manifest(
                    "attack",
                    content_hash="attack-hash",
                    tasks=["binary"],
                    target_is_attack=True,
                ),
            ],
        ),
        _write_jsonl(
            tmp_path / "results-conflict.jsonl",
            [_result("benign"), _result("attack")],
        ),
    )

    assert len(prepared.records) == 2
    with pytest.raises(campaign.DeduplicationConflictError, match="ocultaria.*target"):
        campaign.deduplicate_records(prepared.records, task="binary")


def test_quality_metrics_are_available_for_partial_campaign(tmp_path):
    manifests = campaign.load_manifests(
        _write_jsonl(
            tmp_path / "manifest.jsonl",
            [
                _manifest("ok", content_hash="h1", dataset="d1"),
                _manifest("fallback", content_hash="h2", dataset="d1"),
                _manifest("error", content_hash="h3", dataset="d2"),
                _manifest("missing", content_hash="h4", dataset="d2"),
            ],
        )
    )
    results = campaign.load_results(
        _write_jsonl(
            tmp_path / "results.jsonl",
            [
                _result("ok"),
                _result("fallback", parsed_by_llm=False),
                _result("error", ok=False, parsed_by_llm=False),
            ],
        )
    )
    prepared = campaign.prepare_validation_campaign(
        manifests,
        results,
        campaign.PreflightRequirements(require_complete=False),
    )
    metrics = prepared.quality.global_metrics

    assert metrics.manifest_rows == 4
    assert metrics.attempted_rows == 3
    assert metrics.latest_ok_rows == 2
    assert metrics.latest_error_rows == 1
    assert metrics.latest_fallback_rows == 1
    assert metrics.valid_llm_rows == 1
    assert metrics.selected_rows == 1
    assert metrics.missing_rows == 3
    assert metrics.parse_ok_rate == 0.5
    assert metrics.llm_success_rate == 0.25
    assert metrics.fallback_rate == 0.25
    assert set(prepared.quality.by_dataset) == {"d1", "d2"}
