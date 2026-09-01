from __future__ import annotations

import json

import pytest

from src.eval import reporting


def test_score_run_multiclass():
    result = reporting.score_run(
        ["benign", "ddos", "ddos"],
        ["benign", "ddos", "benign"],
        task="multiclass",
    )

    assert result["n_samples"] == 3
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert 0.0 <= result["f1"] <= 1.0


def test_export_report_confines_output(tmp_path):
    path = reporting.export_report(
        "Informe",
        [{"heading": "Resultado", "body": {"ok": True}}],
        filename="reports/result.md",
        output_root=tmp_path,
    )

    assert path == (tmp_path / "reports" / "result.md").resolve()
    assert '"ok": true' in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("filename", ["../outside.md", "folder/../../outside.md"])
def test_export_report_rejects_traversal(tmp_path, filename):
    with pytest.raises(ValueError):
        reporting.export_report("Informe", [], filename=filename, output_root=tmp_path)


def test_compare_with_baseline_uses_frozen_values(tmp_path, monkeypatch):
    baseline = {
        "audit_thresholds": {
            "multiclass_f1_min_canonical_edge": 0.8,
            "multiclass_f1_must_beat": 0.7,
            "binary_f1_min": 0.75,
        },
        "task_multiclass": {
            "jorge_deepseek_finetuned_less": {"f1": 0.7},
            "current_system_canonical_edge": {"f1": 0.9},
        },
        "task_binary": {
            "jorge": {"f1": 0.7},
            "current_system_canonical_edge": {"f1": 0.9},
        },
    }
    path = tmp_path / "baselines.json"
    path.write_text(json.dumps(baseline), encoding="utf-8")
    monkeypatch.setenv("TFM_BASELINES", str(path))

    result = reporting.compare_with_baseline(0.85, task="multiclass")

    assert result["beats_jorge"] is True
    assert result["meets_audit_threshold"] is True
    assert result["delta_vs_jorge"] == pytest.approx(0.15)
