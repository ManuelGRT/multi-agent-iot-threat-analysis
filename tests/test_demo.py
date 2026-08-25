# tests/test_demo.py
"""Tests de la demo reproducible (Fase 6)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import scripts.demo_mcp_multiagent_case as demo
from src.contracts.case import (
    CaseResult,
    ClassificationInfo,
    DetectionInfo,
    JudgeInfo,
    StandardizationInfo,
    TraceEntry,
)
from src.mcp.common import resolve_path
from tests.test_final_agents import CANONICAL_EVENT

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _artifacts_available() -> bool:
    return (
        resolve_path("standardized_dataset").exists()
        and resolve_path("mistral_cache").exists()
        and (PROJECT_ROOT / demo.EDGE_DATASET_PATH).exists()
    )


def _models_loadable() -> bool:
    try:
        import xgboost  # noqa: F401
    except ImportError:
        return False
    return resolve_path("detection_model").exists() and resolve_path("family_model").exists()


# ---------------------------------------------------------------------------
# digest estable
# ---------------------------------------------------------------------------

def build_case(case_id: str, created_shift_seconds: int = 0) -> CaseResult:
    start = datetime(2026, 8, 2, 12, 0, 0, tzinfo=timezone.utc) + timedelta(
        seconds=created_shift_seconds
    )
    entry = TraceEntry(
        agent="final_detector",
        status="ok",
        started_at=start,
        finished_at=start + timedelta(milliseconds=100 + created_shift_seconds),
    )
    return CaseResult(
        case_id=case_id,
        canonical_event=dict(CANONICAL_EVENT),
        standardization=StandardizationInfo(mapping_confidence=0.9, from_cache=True),
        detection=DetectionInfo(is_malicious=True, probability=0.97),
        classification=ClassificationInfo(attack_family="ddos", confidence=0.95),
        judge=JudgeInfo(action="approve", approved=True, final_label="ddos"),
        trace=[entry],
    ).close()


def test_stable_digest_ignores_volatile_fields():
    first = demo.stable_digest(build_case("case-aaa111aaa111"))
    second = demo.stable_digest(build_case("case-bbb222bbb222", created_shift_seconds=45))
    assert first == second  # ids y timestamps distintos, mismo resultado


def test_stable_digest_changes_with_substantive_result():
    base = build_case("case-aaa111aaa111")
    changed = build_case("case-aaa111aaa111")
    changed.classification.attack_family = "scanning"
    assert demo.stable_digest(base) != demo.stable_digest(changed)


def test_stable_view_has_no_volatile_keys():
    view = demo.stable_view(build_case("case-aaa111aaa111"))
    dumped = json.dumps(view)
    assert "case-aaa111aaa111" not in dumped
    assert "started_at" not in dumped and "latency" not in dumped


# ---------------------------------------------------------------------------
# seleccion de casos
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (_artifacts_available() and _models_loadable()),
    reason="artefactos o modelos no disponibles",
)
def test_build_case_specs_covers_four_datasets():
    specs = demo.build_case_specs(scan_cap=300)
    names = [spec["name"] for spec in specs]
    assert names == ["edge_iiotset", "ton_iot_host", "ton_iot_telemetry", "iot23"]
    edge = specs[0]
    assert edge["ground_truth"]["is_attack"] is True
    assert "canonical_event" in edge["raw_input"]
    for spec in specs[1:]:
        assert spec["raw_input"]["cache_key"].split("::")[0].startswith(
            ("TON_IOT", "IOT23")
        )


# ---------------------------------------------------------------------------
# criterio de aceptacion F6: una orden -> JSONs; repetir -> mismo resultado
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (_artifacts_available() and _models_loadable()),
    reason="artefactos o modelos no disponibles",
)
def test_demo_offline_is_reproducible_across_processes(tmp_path):
    """Criterio F6 literal: repetir la ORDEN (procesos separados, estado frio)
    produce el mismo resultado. La persistencia va a una DB temporal."""
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env["TFM_CASE_MEMORY_DB"] = str(tmp_path / "cases.db")
    command = [
        sys.executable,
        "scripts/demo_mcp_multiagent_case.py",
        "--offline",
        "--out-dir", str(tmp_path),
    ]

    first = subprocess.run(
        command, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=600
    )
    assert first.returncode == 0, first.stdout[-2000:] + first.stderr[-2000:]
    assert "primera ejecucion" in first.stdout
    case_files = sorted(tmp_path.glob("demo_case_*.json"))
    assert len(case_files) == 4
    first_digests = {
        path.name: json.loads(path.read_text(encoding="utf-8"))["digest"]
        for path in case_files
    }
    reference = json.loads((tmp_path / "demo_digests.json").read_text(encoding="utf-8"))
    assert len(reference) == 4

    second = subprocess.run(
        command, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=600
    )
    assert second.returncode == 0, second.stdout[-2000:] + second.stderr[-2000:]
    assert second.stdout.count("digest IDENTICO") == 4
    assert "DISTINTO" not in second.stdout
    second_digests = {
        path.name: json.loads(path.read_text(encoding="utf-8"))["digest"]
        for path in sorted(tmp_path.glob("demo_case_*.json"))
    }
    assert first_digests == second_digests

    summaries = list(tmp_path.glob("demo_summary_*.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["casos_faltantes"] == []
    assert summary["comparativa_jorge"]["beats_jorge"] is True
    assert summary["comparativa_jorge"]["cobertura_capec_jorge"] == "14/14"
    assert all(c["auditoria"] in {"approve", "review"} for c in summary["casos"])
    # la persistencia real quedo en la DB temporal (8 casos: 4 por ejecucion)
    assert (tmp_path / "cases.db").exists()


def test_missing_expected_case_fails_run(tmp_path, monkeypatch):
    """Si un caso obligatorio no puede seleccionarse, la demo NO devuelve 0."""
    monkeypatch.setattr(demo, "build_case_specs", lambda cap: [])
    assert demo.main(["--offline", "--no-persist", "--out-dir", str(tmp_path)]) == 2

    monkeypatch.setattr(
        demo,
        "build_case_specs",
        lambda cap: [
            {
                "name": "iot23",
                "titulo": "solo un caso",
                "raw_input": {"dataset": "iot23", "row": {"proto": "tcp"}},
                "ground_truth": None,
            }
        ],
    )
    if not (_artifacts_available() and _models_loadable()):
        pytest.skip("artefactos o modelos no disponibles para el caso restante")
    rc = demo.main(["--offline", "--no-persist", "--out-dir", str(tmp_path)])
    assert rc == 1  # faltan edge/ton_*: exit distinto de 0 aunque iot23 corra


def test_requested_stdio_smoke_failure_returns_nonzero(tmp_path, monkeypatch, capsys):
    async def failing_stdio_smoke(*args, **kwargs):
        raise TimeoutError("servidor MCP bloqueado")

    monkeypatch.setattr(
        "src.mcp.client.list_tools_stdio", failing_stdio_smoke
    )
    monkeypatch.setattr(
        demo,
        "build_case_specs",
        lambda cap: pytest.fail("la demo debe cortar antes de seleccionar casos"),
    )

    rc = demo.main(
        ["--offline", "--stdio-smoke", "--no-persist", "--out-dir", str(tmp_path)]
    )

    assert rc == 3
    assert "ERROR: MCP stdio no disponible" in capsys.readouterr().out
