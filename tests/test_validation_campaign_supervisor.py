from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_validation_campaign as campaign


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _manifest(root: Path, dataset: str, ids: list[str]) -> Path:
    path = root / "manifests" / f"{dataset}_manifest.jsonl"
    _write_jsonl(path, [{"manifest_id": item} for item in ids])
    return path


def _llm(manifest_id: str, **overrides) -> dict:
    record = {
        "manifest_id": manifest_id,
        "ok": True,
        "parsed_by_llm": True,
        "provider": campaign.DEFAULT_PROVIDER,
        "model": campaign.DEFAULT_MODEL,
        "canonical_event": {"origin": "test", "features": {"x": 1}},
    }
    record.update(overrides)
    return record


def _fallback(manifest_id: str, **overrides) -> dict:
    record = {
        "manifest_id": manifest_id,
        "ok": True,
        "parsed_by_llm": False,
        "provider": campaign.DEFAULT_PROVIDER,
        "model": campaign.DEFAULT_MODEL,
    }
    record.update(overrides)
    return record


def _error(manifest_id: str, **overrides) -> dict:
    record = {
        "manifest_id": manifest_id,
        "ok": False,
        "parsed_by_llm": False,
        "provider": campaign.DEFAULT_PROVIDER,
        "model": campaign.DEFAULT_MODEL,
    }
    record.update(overrides)
    return record


def _config(root: Path, **overrides) -> campaign.CampaignConfig:
    values = {
        "repo": root,
        "manifest_dir": root / "manifests",
        "results_dir": root / "standardized",
        "evaluation_dir": root / "evaluation",
        "state_dir": root / "state",
        "python_executable": "python-test",
        "workers": 3,
        "max_rounds": 6,
        "max_consecutive_no_progress_rounds": 3,
        "round_backoff_seconds": 0,
        "max_log_bytes": 1024,
    }
    values.update(overrides)
    return campaign.CampaignConfig(**values)


def test_scan_status_requires_nonempty_canonical_event_and_fixed_model(tmp_path):
    _manifest(tmp_path, "sample", ["a", "b", "c", "d", "e", "f"])
    result_path = tmp_path / "standardized" / "sample_standardized.jsonl"
    _write_jsonl(
        result_path,
        [
            _fallback("a"),
            _llm("a"),
            _error("a"),  # un exito LLM valido es terminal para el ID
            _error("b"),
            _fallback("b"),
            _error("c"),
            _llm("d", canonical_event={}),
            _llm("e", provider="otro"),
            _llm("f", model="latest"),
        ],
    )

    inventory = campaign.load_manifest_inventory(tmp_path / "manifests")
    status = campaign.scan_campaign_status(inventory, tmp_path / "standardized")

    assert status.total == 6
    assert status.parsed_by_llm == 1
    assert status.fallback == 1
    assert status.error == 1
    assert status.invalid == 3
    assert status.invalid_result_records == 3
    assert status.missing == 0
    assert status.complete is False


@pytest.mark.parametrize(
    "corruption",
    ["malformed", "orphan", "invalid"],
)
def test_corruption_prevents_complete_even_with_full_llm_coverage(tmp_path, corruption):
    _manifest(tmp_path, "sample", ["a"])
    result_path = tmp_path / "standardized" / "sample_standardized.jsonl"
    initial = [_llm("a", canonical_event={})] if corruption == "invalid" else []
    _write_jsonl(result_path, [*initial, _llm("a")])
    with result_path.open("a", encoding="utf-8") as handle:
        if corruption == "malformed":
            handle.write("{linea-incompleta\n")
        elif corruption == "orphan":
            handle.write(json.dumps(_llm("foreign")) + "\n")

    inventory = campaign.load_manifest_inventory(tmp_path / "manifests")
    status = campaign.scan_campaign_status(inventory, tmp_path / "standardized")

    assert status.parsed_by_llm == status.total == 1
    assert status.complete is False
    assert (
        status.malformed_result_lines
        + status.orphan_result_records
        + status.invalid_result_records
        == 1
    )


def test_manifest_inventory_rejects_duplicate_ids_across_datasets(tmp_path):
    _manifest(tmp_path, "first", ["same"])
    _manifest(tmp_path, "second", ["same"])

    with pytest.raises(campaign.CampaignError, match="duplicado"):
        campaign.load_manifest_inventory(tmp_path / "manifests")


def test_commands_and_phase_environments_are_explicit_and_secret_safe(tmp_path):
    secret = "super-secret-key"
    config = _config(tmp_path, workers=7, provider="mistral", model="fixed-model")

    standardize = campaign.standardization_command(config)
    evaluate = campaign.evaluation_command(config)
    phase_c_env = campaign.standardization_environment(
        config,
        {"PATH": "safe", "MISTRAL_API_KEY": secret, "OTHER_TOKEN": "remove-later"},
    )
    evaluation_env = campaign.evaluation_environment(phase_c_env)

    assert standardize[-3:] == ("--workers", "7", "--retry-failures")
    assert "--manifest-dir" in evaluate
    assert str(config.manifest_dir) in evaluate
    assert str(config.results_dir) in evaluate
    assert str(config.evaluation_dir) in evaluate
    assert evaluate[evaluate.index("--provider") + 1] == "mistral"
    assert evaluate[evaluate.index("--model") + 1] == "fixed-model"
    assert secret not in " ".join(standardize + evaluate)
    assert phase_c_env["MISTRAL_API_KEY"] == secret
    assert "OTHER_TOKEN" not in phase_c_env
    assert phase_c_env["LLM_PROVIDER"] == "mistral"
    assert phase_c_env["INGEST_LLM_MODEL"] == "fixed-model"
    assert "MISTRAL_API_KEY" not in evaluation_env
    assert "OTHER_TOKEN" not in evaluation_env
    assert evaluation_env["PATH"] == "safe"


def test_logged_process_receives_explicit_environment_and_compacts_log(tmp_path):
    observed = {}

    def fake_subprocess_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        kwargs["stdout"].write(b"old line\n" + b"x" * 300 + b"\nlast line\n")
        return SimpleNamespace(returncode=7)

    log_path = tmp_path / "round.log"
    return_code = campaign.run_logged_process(
        ["python", "-u", "safe.py"],
        tmp_path,
        log_path,
        96,
        {"PATH": "safe", "MISTRAL_API_KEY": "in-memory-only"},
        subprocess_run=fake_subprocess_run,
    )

    assert return_code == 7
    assert observed["kwargs"]["env"]["MISTRAL_API_KEY"] == "in-memory-only"
    assert "in-memory-only" not in " ".join(observed["command"])
    assert log_path.stat().st_size <= 96
    assert b"inicio del log truncado" in log_path.read_bytes()


def test_wait_and_backoff_are_split_into_at_most_sixty_seconds():
    sleeps = []
    campaign.sleep_in_chunks(130, sleep=sleeps.append)
    assert sleeps == [60.0, 60.0, 10.0]

    alive = iter([True, False])
    poll_sleeps = []
    campaign.wait_for_pid(
        123,
        poll_seconds=125,
        timeout_seconds=None,
        exists=lambda _pid: next(alive),
        sleep=poll_sleeps.append,
    )
    assert poll_sleeps == [60.0, 60.0, 5.0]


def test_complete_campaign_needs_no_key_and_evaluator_gets_sanitized_env(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setenv("ANOTHER_SECRET", "must-not-leak")
    _manifest(tmp_path, "sample", ["a", "b"])
    _write_jsonl(
        tmp_path / "standardized" / "sample_standardized.jsonl",
        [_llm("a"), _llm("b")],
    )
    calls = []

    def fake_runner(command, cwd, log_path, max_log_bytes, environment):
        calls.append((tuple(command), cwd, log_path, max_log_bytes, environment))
        return 0

    config = _config(tmp_path)
    assert campaign.run_campaign(config, process_runner=fake_runner) == 0

    assert len(calls) == 1
    assert calls[0][0][2] == "scripts/eval_validacion_por_dataset.py"
    assert "MISTRAL_API_KEY" not in calls[0][4]
    assert "ANOTHER_SECRET" not in calls[0][4]
    checkpoint = json.loads(config.checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["phase"] == "complete"
    assert checkpoint["provider"] == "mistral"
    assert checkpoint["model"] == "mistral-small-2603"
    assert checkpoint["status"]["complete"] is True
    assert checkpoint["counters"]["rounds_started"] == 0


def test_incomplete_campaign_requires_key_only_before_standardization(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    _manifest(tmp_path, "sample", ["a"])
    calls = []

    with pytest.raises(campaign.MissingCredentialError, match="incompleta"):
        campaign.run_campaign(
            _config(tmp_path),
            process_runner=lambda *args: calls.append(args) or 0,
        )

    assert calls == []


def test_campaign_retries_until_complete_and_scrubs_key_from_evaluation(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("MISTRAL_API_KEY", "only-phase-c")
    _manifest(tmp_path, "sample", ["a", "b"])
    result_path = tmp_path / "standardized" / "sample_standardized.jsonl"
    calls = []

    def fake_runner(command, _cwd, _log_path, _max_log_bytes, environment):
        calls.append((tuple(command), dict(environment)))
        if command[2] == "scripts/run_live_standardization.py":
            completed = sum(
                call[0][2] == "scripts/run_live_standardization.py" for call in calls
            )
            result_path.parent.mkdir(parents=True, exist_ok=True)
            with result_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_llm("a" if completed == 1 else "b")) + "\n")
        return 0

    assert campaign.run_campaign(_config(tmp_path), process_runner=fake_runner) == 0

    assert len(calls) == 3
    assert calls[0][1]["MISTRAL_API_KEY"] == "only-phase-c"
    assert calls[1][1]["MISTRAL_API_KEY"] == "only-phase-c"
    assert "MISTRAL_API_KEY" not in calls[2][1]


def test_no_progress_requires_three_consecutive_rounds_and_checkpoints_codes(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    _manifest(tmp_path, "sample", ["a"])
    calls = []

    def failed_runner(*_args):
        calls.append(7)
        return 7

    config = _config(tmp_path)
    with pytest.raises(campaign.NoProgressError, match="3 rondas"):
        campaign.run_campaign(config, process_runner=failed_runner)

    assert calls == [7, 7, 7]
    checkpoint = json.loads(config.checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["phase"] == "failed_no_progress"
    assert checkpoint["return_code"] == 7
    assert checkpoint["counters"] == {
        "rounds_started": 3,
        "rounds_finished": 3,
        "nonzero_return_codes": 3,
        "consecutive_no_progress_rounds": 3,
        "total_new_llm_successes": 0,
    }


def test_progress_resets_no_progress_counter_and_max_rounds_is_distinct(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    _manifest(tmp_path, "sample", ["a", "b"])
    result_path = tmp_path / "standardized" / "sample_standardized.jsonl"
    round_number = 0

    def sometimes_progresses(*_args):
        nonlocal round_number
        round_number += 1
        if round_number == 2:
            _write_jsonl(result_path, [_llm("a")])
        return 0

    config = _config(tmp_path, max_rounds=3)
    with pytest.raises(campaign.MaxRoundsExceeded, match="Cobertura incompleta"):
        campaign.run_campaign(config, process_runner=sometimes_progresses)

    checkpoint = json.loads(config.checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["phase"] == "failed_max_rounds"
    assert checkpoint["counters"]["consecutive_no_progress_rounds"] == 1
    assert checkpoint["counters"]["total_new_llm_successes"] == 1


def test_existing_pid_is_waited_before_complete_evaluation_without_key(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    _manifest(tmp_path, "sample", ["a"])
    result_path = tmp_path / "standardized" / "sample_standardized.jsonl"
    alive = iter([True, False])
    sleeps = []
    calls = []

    def process_exists(_pid):
        is_alive = next(alive)
        if not is_alive:
            _write_jsonl(result_path, [_llm("a")])
        return is_alive

    assert campaign.run_campaign(
        _config(tmp_path, wait_pid=987, wait_poll_seconds=0.25),
        process_runner=lambda *args: calls.append(args) or 0,
        process_exists=process_exists,
        sleep=sleeps.append,
    ) == 0

    assert sleeps == [0.25]
    assert len(calls) == 1
    assert calls[0][0][2] == "scripts/eval_validacion_por_dataset.py"


def test_real_runner_rejects_custom_incomplete_directories_before_launch(tmp_path):
    _manifest(tmp_path, "sample", ["a"])

    with pytest.raises(campaign.CampaignError, match="rutas personalizadas"):
        campaign.run_campaign(_config(tmp_path))


def test_supervisor_releases_windows_awake_state_after_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    _manifest(tmp_path, "sample", ["a"])
    flags = []

    with pytest.raises(campaign.NoProgressError):
        campaign.supervise_campaign(
            _config(tmp_path, max_consecutive_no_progress_rounds=1),
            process_runner=lambda *_args: 1,
            platform_name="nt",
            execution_state_setter=lambda value: flags.append(value) or 1,
        )

    assert flags == [0x80000001, 0x80000000]


def test_parser_defaults_allow_long_campaign_and_fixed_model():
    args = campaign.build_parser().parse_args([])

    assert args.max_rounds >= 20
    assert args.max_no_progress_rounds == 3
    assert args.provider == "mistral"
    assert args.model == "mistral-small-2603"
