"""Regresiones de la frontera entre preparacion offline y runtime final."""
from __future__ import annotations

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
FINAL_RUNTIME_MODULES = (
    "src/mcp/inference_server.py",
    "src/mcp/standardization_contract.py",
    "src/mcp/features.py",
    "src/agents/llm_ingest_parser.py",
    "src/agents/supervised_general.py",
    "src/agents/final/final_standardizer.py",
    "src/agents/final/final_detector.py",
    "src/agents/final/final_classifier.py",
    "src/agents/final/final_mitigator.py",
    "src/agents/final/final_judge.py",
    "src/agents/final/auditor.py",
    "src/orchestration/mcp_graph.py",
    "src/api/routers.py",
)
OFFLINE_SANITIZATION_MODULES = frozenset(
    {
        "src.eval.data_sanitization",
        "src.eval.predictive_sanitization",
    }
)
LEGACY_RUNTIME_MODULES = frozenset(
    {
        "src.agents.ingest_parser",
        "src.orchestration.graph",
    }
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_final_runtime_does_not_import_offline_sanitization():
    offenders: dict[str, list[str]] = {}
    for relative in FINAL_RUNTIME_MODULES:
        imported = _imports(REPO / relative)
        forbidden = sorted(imported.intersection(OFFLINE_SANITIZATION_MODULES))
        if forbidden:
            offenders[relative] = forbidden

    assert offenders == {}


def test_final_runtime_does_not_import_legacy_graph_or_adapter_router():
    offenders: dict[str, list[str]] = {}
    for relative in FINAL_RUNTIME_MODULES:
        imported = _imports(REPO / relative)
        forbidden = sorted(imported.intersection(LEGACY_RUNTIME_MODULES))
        if forbidden:
            offenders[relative] = forbidden

    assert offenders == {}


def test_sanitization_lives_under_evaluation_namespace_only():
    assert (REPO / "src/eval/data_sanitization.py").is_file()
    assert (REPO / "src/eval/predictive_sanitization.py").is_file()
    assert not (REPO / "src/mcp/standardization_guard.py").exists()
    assert not (REPO / "src/agents/predictive_sanitization.py").exists()
