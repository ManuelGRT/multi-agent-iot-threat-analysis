"""Regresiones del paquete desplegable y de su frontera de escritura."""
from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

from src.mcp.common import package_data_dir, resolve_path


REPO = Path(__file__).resolve().parents[1]
DETECTION_MODEL = "xgboost_detection_validation_2026_20260822.joblib"
CLASSIFIER_MODEL = "xgboost_attack_subtype_multidataset16_balanced500_20260906.joblib"
CLASSIFIER_MODEL_SHA256 = (
    "f49b2a50920b3fa260a999f34696e86a92068786f386bae866b02a5faaa7b185"
)


def test_read_only_runtime_resources_live_inside_package():
    data = package_data_dir()

    assert resolve_path("detection_model") == data / "models" / DETECTION_MODEL
    assert resolve_path("family_model") == data / "models" / CLASSIFIER_MODEL
    assert resolve_path("detection_model").is_file()
    assert resolve_path("family_model").is_file()
    assert (data / "threat_intel_catalog.json").is_file()


def test_packaged_classifier_is_the_reviewed_balanced_16_type_artifact():
    model_path = resolve_path("family_model")

    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == (
        CLASSIFIER_MODEL_SHA256
    )


def test_operational_state_uses_configurable_writable_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("TFM_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("TFM_STANDARDIZATION_CACHE_DB", raising=False)
    monkeypatch.delenv("TFM_CASE_MEMORY_DB", raising=False)

    assert resolve_path("standardization_cache_db") == (
        tmp_path / "cache" / "mistral_standardization_v2.sqlite3"
    )
    assert resolve_path("case_memory_db") == tmp_path / "case_memory.db"


def test_wheel_declares_only_online_packages_and_resources():
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    setuptools = project["tool"]["setuptools"]
    package_data = setuptools["package-data"]

    assert setuptools["include-package-data"] is False
    assert "src.eval*" in setuptools["packages"]["find"]["exclude"]
    assert package_data["src.api"] == ["static/index.html"]
    assert set(package_data["src.mcp"]) == {
        "data/threat_intel_catalog.json",
        f"data/models/{DETECTION_MODEL}",
        f"data/models/{CLASSIFIER_MODEL}",
    }


def test_test_runner_is_not_a_runtime_dependency():
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert not any(requirement.startswith("pytest") for requirement in project["dependencies"])
    assert any(requirement.startswith("pytest") for requirement in project["optional-dependencies"]["test"])
