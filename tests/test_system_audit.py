"""Auditoria portable de los artefactos desplegados actuales."""
from __future__ import annotations

import json
from pathlib import Path

import scripts.run_system_audit as audit_script


def _copy_json(source: Path, target: Path) -> dict:
    value = json.loads(source.read_text(encoding="utf-8"))
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return value


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_current_release_audit_is_fully_green():
    payload = audit_script.run_audit()

    assert payload["todo_verde"] is True
    assert payload["semaforos"] == {
        "detector": audit_script.GREEN,
        "classifier": audit_script.GREEN,
        "catalog": audit_script.GREEN,
        "auditor": audit_script.GREEN,
    }
    assert payload["results"]["detector"]["splits"] == {
        "train": 16496,
        "validation": 3536,
        "test": 3572,
    }
    assert payload["results"]["classifier"]["splits"] == {
        "train": 5600,
        "val": 1200,
        "test": 1200,
    }
    assert payload["results"]["classifier"]["training_taxonomy_sha256"] == (
        "d016e777a545c78361adb1c05352a70f36e0022c5f85e5aaf4f8e379655bd6ed"
    )
    assert payload["results"]["classifier"]["runtime_taxonomy_sha256"] == (
        "060a45a18e9a99e302467a12a358fe1f346091707a75375e2082a98e3506fae2"
    )
    assert payload["results"]["catalog"]["version"] == "3.0"
    assert payload["results"]["auditor"]["detected"] == 16
    assert payload["results"]["auditor"]["false_rejects"] == 0


def test_detector_audit_rejects_a_reference_for_another_model(
    tmp_path, monkeypatch
):
    path = tmp_path / "detector.json"
    reference = _copy_json(audit_script.DETECTOR_REFERENCE, path)
    reference["artifact"]["sha256"] = "0" * 64
    _write_json(path, reference)
    monkeypatch.setattr(audit_script, "DETECTOR_REFERENCE", path)

    result = audit_script.audit_detector()

    assert result["semaforo"] == audit_script.RED
    assert "artifact_sha256" in result["issues"]


def test_detector_audit_rejects_another_balanced_selection(tmp_path, monkeypatch):
    path = tmp_path / "detector.json"
    reference = _copy_json(audit_script.DETECTOR_REFERENCE, path)
    reference["balancing"]["selected_manifest_ids_sha256"] = "0" * 64
    _write_json(path, reference)
    monkeypatch.setattr(audit_script, "DETECTOR_REFERENCE", path)

    result = audit_script.audit_detector()

    assert result["semaforo"] == audit_script.RED
    assert "selection_release_sha256" in result["issues"]


def test_classifier_audit_rejects_a_non_operational_threshold(
    tmp_path, monkeypatch
):
    path = tmp_path / "classifier.json"
    reference = _copy_json(audit_script.CLASSIFIER_REFERENCE, path)
    reference["model"]["operational_contract"]["confidence_threshold"] = 0.8
    _write_json(path, reference)
    monkeypatch.setattr(audit_script, "CLASSIFIER_REFERENCE", path)

    result = audit_script.audit_classifier()

    assert result["semaforo"] == audit_script.RED
    assert "threshold_reference" in result["issues"]


def test_classifier_audit_rejects_a_stale_runtime_taxonomy(
    tmp_path, monkeypatch
):
    path = tmp_path / "classifier.json"
    reference = _copy_json(audit_script.CLASSIFIER_REFERENCE, path)
    reference["taxonomy"]["runtime_taxonomy_sha256"] = "0" * 64
    reference["hashes"]["runtime_taxonomy"] = "0" * 64
    _write_json(path, reference)
    monkeypatch.setattr(audit_script, "CLASSIFIER_REFERENCE", path)

    result = audit_script.audit_classifier()

    assert result["semaforo"] == audit_script.RED
    assert "runtime_taxonomy_sha256" in result["issues"]


def test_classifier_audit_rejects_another_test_split(tmp_path, monkeypatch):
    path = tmp_path / "classifier.json"
    reference = _copy_json(audit_script.CLASSIFIER_REFERENCE, path)
    reference["hashes"]["test"] = "0" * 64
    _write_json(path, reference)
    monkeypatch.setattr(audit_script, "CLASSIFIER_REFERENCE", path)

    result = audit_script.audit_classifier()

    assert result["semaforo"] == audit_script.RED
    assert "release_hash:test" in result["issues"]


def test_classifier_reference_fixes_balanced_16_type_split():
    result = audit_script.audit_classifier()

    assert result["semaforo"] == audit_script.GREEN
    assert result["corpus_rows"] == 8000
    assert result["per_class"] == 500
    assert result["splits"] == {"train": 5600, "val": 1200, "test": 1200}
    assert result["test_top3"] == 0.98
    assert result["selective"]["threshold"] == 0.65


def test_catalog_audit_rejects_an_unpinned_catalog(tmp_path, monkeypatch):
    path = tmp_path / "catalog-reference.json"
    reference = _copy_json(audit_script.CATALOG_REFERENCE, path)
    reference["artifact"]["sha256"] = "f" * 64
    _write_json(path, reference)
    monkeypatch.setattr(audit_script, "CATALOG_REFERENCE", path)

    result = audit_script.audit_catalog()

    assert result["semaforo"] == audit_script.RED
    assert "artifact_sha256" in result["issues"]


def test_catalog_audit_rejects_incorrect_reference_registry_counts(
    tmp_path, monkeypatch
):
    path = tmp_path / "catalog-reference.json"
    reference = _copy_json(audit_script.CATALOG_REFERENCE, path)
    reference["reference_catalog_counts"]["capec_patterns"] = 15
    _write_json(path, reference)
    monkeypatch.setattr(audit_script, "CATALOG_REFERENCE", path)

    result = audit_script.audit_catalog()

    assert result["semaforo"] == audit_script.RED
    assert "reference_catalog_counts" in result["issues"]


def test_auditor_mutations_detect_the_expected_reason():
    result = audit_script.audit_auditor()

    assert result["semaforo"] == audit_script.GREEN
    assert result["mutations"] == result["detected"] == 16
    assert result["false_rejects"] == 0
    assert all(row["detected"] for row in result["results"])
    assert {row["verdict"] for row in result["results"]} <= {"review", "reject"}


def test_auditor_evaluation_rejects_a_wrong_expected_check(
    tmp_path, monkeypatch
):
    path = tmp_path / "auditor-reference.json"
    reference = _copy_json(audit_script.AUDITOR_REFERENCE, path)
    reference["mutations"][0]["expected_check"] = "chequeo_que_no_existe"
    _write_json(path, reference)
    monkeypatch.setattr(audit_script, "AUDITOR_REFERENCE", path)

    result = audit_script.audit_auditor()

    assert result["semaforo"] == audit_script.RED
    assert any(
        issue.startswith("mutation_not_detected:") for issue in result["issues"]
    )


def test_auditor_evaluation_rejects_duplicate_mutation_ids(
    tmp_path, monkeypatch
):
    path = tmp_path / "auditor-reference.json"
    reference = _copy_json(audit_script.AUDITOR_REFERENCE, path)
    reference["mutations"][1] = dict(reference["mutations"][0])
    _write_json(path, reference)
    monkeypatch.setattr(audit_script, "AUDITOR_REFERENCE", path)

    result = audit_script.audit_auditor()

    assert result["semaforo"] == audit_script.RED
    assert "mutation_contract" in result["issues"]


def test_main_no_write_returns_success(capsys):
    assert audit_script.main(["--no-write"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["todo_verde"] is True
    assert output["outputs"] == {}


def test_main_writes_only_inside_artifacts(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(audit_script, "artifacts_dir", lambda: tmp_path)

    assert audit_script.main(["--out-prefix", "current-release"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert Path(output["outputs"]["json"]) == tmp_path / "current-release.json"
    assert Path(output["outputs"]["markdown"]) == tmp_path / "current-release.md"
    assert (tmp_path / "current-release.json").is_file()
    assert (tmp_path / "current-release.md").is_file()


def test_main_rejects_an_escaping_report_prefix(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(audit_script, "artifacts_dir", lambda: tmp_path)

    assert audit_script.main(["--out-prefix", "../escape"]) == 2
    assert "prefijo de informe no permitido" in capsys.readouterr().out


def test_portable_audit_has_no_historical_dataset_or_baseline_dependency():
    source = Path(audit_script.__file__).read_text(encoding="utf-8")

    assert "eval_validacion_por_dataset" not in source
    assert "jorge_and_current_baselines" not in source
    assert "standardized_dataset" not in source
    assert "artifacts/validation_2026" not in source
