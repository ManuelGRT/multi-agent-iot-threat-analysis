from __future__ import annotations

import json

import pytest

from scripts import run_live_standardization as live


def test_load_project_env_reads_key_without_echoing_or_overriding(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# credenciales locales\n"
        "MISTRAL_API_KEY='dummy-local-key'\n"
        "export INGEST_LLM_MODEL=mistral-small-test\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setenv("INGEST_LLM_MODEL", "explicit-model")

    assert live.load_project_env(env_file) is True
    assert live.os.environ["MISTRAL_API_KEY"] == "dummy-local-key"
    assert live.os.environ["INGEST_LLM_MODEL"] == "explicit-model"


def test_load_project_env_missing_file_is_safe(tmp_path):
    assert live.load_project_env(tmp_path / "missing.env") is False


def test_campaign_defaults_enable_transient_retries(monkeypatch):
    names = (
        "LLM_PROVIDER",
        "INGEST_LLM_MODEL",
        "MISTRAL_RATE_LIMIT_RETRIES",
        "MISTRAL_RETRY_STATUS_CODES",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)

    try:
        live.configure_campaign_env()

        assert live.os.environ["LLM_PROVIDER"] == "mistral"
        assert live.os.environ["INGEST_LLM_MODEL"] == "mistral-small-2603"
        assert live.os.environ["MISTRAL_RATE_LIMIT_RETRIES"] == "8"
        assert live.os.environ["MISTRAL_RETRY_STATUS_CODES"] == "429,500,502,503,504"
    finally:
        for name in names:
            live.os.environ.pop(name, None)


def test_retry_fallbacks_keeps_only_llm_success_as_done(tmp_path):
    out = tmp_path / "results.jsonl"
    records = [
        {"manifest_id": "llm", "ok": True, "parsed_by_llm": True},
        {"manifest_id": "fallback", "ok": True, "parsed_by_llm": False},
        {"manifest_id": "error", "ok": False, "parsed_by_llm": False},
    ]
    out.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")

    assert live.load_done(out, retry_fallbacks=True) == {"llm"}


def test_windows_awake_request_and_release_use_continuous_state():
    calls = []

    def setter(flags):
        calls.append(flags)
        return 1

    assert live.request_system_awake(platform_name="nt", setter=setter) is True
    assert live.release_system_awake(platform_name="nt", setter=setter) is True
    assert calls == [0x80000001, 0x80000000]


def test_dataset_circuit_breaker_stops_repeated_fallbacks(tmp_path, monkeypatch):
    manifest_dir = tmp_path / "manifests"
    output_dir = tmp_path / "standardized"
    manifest_dir.mkdir()
    rows = [
        {"manifest_id": f"row-{index}", "dataset": "sample", "row": {"value": index}}
        for index in range(5)
    ]
    (manifest_dir / "sample_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(live, "MANIFEST_DIR", manifest_dir)
    monkeypatch.setattr(live, "OUT_DIR", output_dir)
    monkeypatch.setattr(
        live,
        "standardize_one",
        lambda entry: {
            "manifest_id": entry["manifest_id"],
            "ok": True,
            "parsed_by_llm": False,
        },
    )

    with pytest.raises(RuntimeError, match="Cortacircuitos"):
        live.run_dataset(
            "sample",
            workers=1,
            limit=None,
            retry_fallbacks=False,
            max_consecutive_failures=2,
        )

    written = (output_dir / "sample_standardized.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(written) == 2
