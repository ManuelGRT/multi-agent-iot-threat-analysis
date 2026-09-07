# tests/test_system_audit.py
"""Tests del script de auditoria de sistema (Fase 5b)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.run_system_audit as audit_script
from src.mcp.common import resolve_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _corpus_available() -> bool:
    return resolve_path("standardized_dataset").exists()


def _edge_available() -> bool:
    return (PROJECT_ROOT / audit_script.EDGE_DATASET_PATH).exists()


def _models_loadable() -> bool:
    try:
        import xgboost  # noqa: F401
    except ImportError:
        return False
    return resolve_path("detection_model").exists() and resolve_path("attack_type_model").exists()


def _attack_type_audit_artifacts_available() -> bool:
    return all(
        path.exists()
        for path in (
            audit_script.ATTACK_TYPE_REPORT,
            audit_script.ATTACK_TYPE_SELECTION,
            audit_script.ATTACK_TYPE_SIDECAR,
            resolve_path("attack_type_model"),
        )
    )


# ---------------------------------------------------------------------------
# helpers puros
# ---------------------------------------------------------------------------

def test_semaforo_table_renders_all_criteria():
    table = audit_script.semaforo_table({"a": audit_script.GREEN, "b": audit_script.RED})
    assert "VERDE" in table and "ROJO" in table
    assert table.count("|") >= 8


def test_leakage_trap_event_is_actually_leaky():
    from src.agents.final.auditor import canonical_leakage_issues

    issues = canonical_leakage_issues(audit_script.leakage_trap_event())
    kinds = {issue.split(":")[0] for issue in issues}
    assert {
        "campo_target_con_valor",
        "clave_target_anidada",
        "semantic_text_contiene_patron_target",
    } <= kinds


def test_smoke_flags_empty_datasets(monkeypatch):
    monkeypatch.setattr(
        audit_script, "smoke_datasets", lambda per: {"edge_iiotset": [], "iot23": []}
    )
    result = audit_script.run_smoke(2)
    assert result["semaforo_smoke"] == audit_script.RED
    assert result["semaforo_auditoria"] == audit_script.RED
    assert set(result["datasets_sin_muestras"]) == {"edge_iiotset", "iot23"}


def test_trap_check_fails_if_semantic_detector_breaks(monkeypatch):
    """Si la deteccion de patrones en semantic_text regresionara, la trampa
    debe ponerse en ROJO aunque las otras clases de leakage sigan saltando."""
    import src.agents.final.auditor as auditor_mod

    monkeypatch.setattr(auditor_mod, "contains_predictive_target_text", lambda text: False)
    trap_issues = auditor_mod.canonical_leakage_issues(audit_script.leakage_trap_event())
    kinds = {issue.split(":")[0] for issue in trap_issues}
    assert "semantic_text_contiene_patron_target" not in kinds
    # el criterio del script exige las TRES clases: con una rota, no hay verde
    required = {
        "campo_target_con_valor",
        "clave_target_anidada",
        "semantic_text_contiene_patron_target",
    }
    assert not required <= kinds


# ---------------------------------------------------------------------------
# muestreo
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _corpus_available(), reason="corpus estandarizado no disponible")
def test_stride_sample_covers_multiple_sources():
    rows = audit_script.stride_sample(resolve_path("standardized_dataset"), 100)
    assert len(rows) == 100
    tops = set()
    for row in rows:
        source = str(row.get("source_file") or "").replace("\\", "/")
        if "/" in source:
            tops.add(source.split("/")[1])
    # el corpus esta ordenado por dataset: el stride debe cruzar varios
    assert len(tops) >= 2, tops


@pytest.mark.skipif(not _corpus_available(), reason="corpus estandarizado no disponible")
def test_sample_rows_mixes_attack_and_benign():
    rows = audit_script.sample_rows(
        resolve_path("standardized_dataset"), lambda row: True, 4
    )
    labels = {bool((row.get("target") or {}).get("is_attack")) for row in rows}
    assert labels == {True, False}


# ---------------------------------------------------------------------------
# semaforo y contrato del detector desplegado
# ---------------------------------------------------------------------------

def test_batch_detection_red_without_frozen_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        audit_script,
        "DETECTION_EVALUATION",
        tmp_path / "missing-detector-evaluation.json",
    )
    result = audit_script.batch_detection(tolerance=0.02)
    assert result["artefacto_congelado_encontrado"] is False
    assert result["semaforo"] == audit_script.RED


@pytest.mark.skipif(not _models_loadable(), reason="modelos no disponibles")
def test_batch_detection_verifies_the_deployed_balanced_artifact():
    result = audit_script.batch_detection(tolerance=0.02)

    assert result["modo_evaluacion"] == "metricas_congeladas_hash_verificado"
    assert result["metricas_recalculadas"] is False
    assert result["ficha_valida"] is True
    assert result["hash_modelo_coincide"] is True
    assert result["contrato_modelo_valido"] is True
    assert result["corpus_balanceado"] == 23604
    assert result["n_train"] == 16496
    assert result["n_validation"] == 3536
    assert result["n_test"] == 3572
    assert result["f1_ataque_congelado"] == pytest.approx(0.9600223651104277)
    assert result["semaforo"] == audit_script.GREEN


@pytest.mark.skipif(not _models_loadable(), reason="modelos no disponibles")
def test_batch_detection_rejects_internally_inconsistent_metadata(
    tmp_path, monkeypatch
):
    payload = json.loads(audit_script.DETECTION_EVALUATION.read_text(encoding="utf-8"))
    payload["balancing"]["rows"] = 23602
    corrupted = tmp_path / "corrupted-detector-evaluation.json"
    corrupted.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(audit_script, "DETECTION_EVALUATION", corrupted)

    result = audit_script.batch_detection(tolerance=0.02)

    assert result["ficha_valida"] is False
    assert "balanced_rows" in result["incidencias_ficha"]
    assert result["semaforo"] == audit_script.RED


@pytest.mark.skipif(
    not (_models_loadable() and _attack_type_audit_artifacts_available()),
    reason="modelo o artefactos del clasificador de 16 tipos no disponibles",
)
def test_batch_attack_type_reports_frozen_metrics_if_campaign_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        audit_script,
        "_resolve_campaign_inputs",
        lambda report: {
            "available": False,
            "integrity_ok": True,
            "paths": {"manifests": [], "standardized_results": []},
            "missing": ["input_ausente.jsonl"],
            "mismatches": [],
        },
    )

    result = audit_script.batch_attack_type(tolerance=0.02)

    assert result["modo_evaluacion"] == "metricas_congeladas_hash_verificado"
    assert result["metricas_recalculadas"] is False
    assert result["hash_modelo_coincide"] is True
    assert result["seleccion_verificada"] is True
    assert result["sidecar_verificado"] is True
    assert result["n_test"] == 1200
    assert result["numero_tipos"] == 16
    assert set(result["soporte_por_tipo"].values()) == {75}
    assert result["metricas_congeladas"]["f1_weighted"] == pytest.approx(
        0.8916753705196386
    )
    assert result["semaforo"] == audit_script.GREEN


@pytest.mark.skipif(
    not (_models_loadable() and _attack_type_audit_artifacts_available()),
    reason="modelo o artefactos del clasificador de 16 tipos no disponibles",
)
def test_batch_attack_type_reproduces_exact_frozen_test_when_inputs_exist():
    inventory = audit_script._resolve_campaign_inputs(
        json.loads(audit_script.ATTACK_TYPE_REPORT.read_text(encoding="utf-8"))
    )
    if not inventory["available"]:
        pytest.skip("inputs originales de la campana no disponibles")

    result = audit_script.batch_attack_type(tolerance=0.02)

    assert result["modo_evaluacion"] == "test_exacto_recalculado"
    assert result["metricas_recalculadas"] is True
    assert result["reproduccion_exacta"] is True
    assert result["n_test"] == 1200
    assert result["numero_tipos"] == 16
    assert result["metricas_recalculadas_test"]["accuracy"] == pytest.approx(
        0.8908333333333334
    )
    assert result["metricas_recalculadas_test"]["top3_accuracy"] == pytest.approx(0.98)
    assert result["metricas_recalculadas_test"]["selective"]["threshold"] == 0.65
    assert result["semaforo"] == audit_script.GREEN


# ---------------------------------------------------------------------------
# E2E del script (rapido: sin batch)
# ---------------------------------------------------------------------------

def test_main_rejects_absolute_report_prefix_before_running_checks(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        audit_script,
        "run_smoke",
        lambda *_args, **_kwargs: pytest.fail("no debe ejecutar la auditoria"),
    )

    rc = audit_script.main(["--out-prefix", str(tmp_path / "fuera")])

    assert rc == 2
    assert "prefijo de informe no permitido" in capsys.readouterr().out

@pytest.mark.skipif(
    not (_models_loadable() and _corpus_available() and _edge_available()),
    reason="modelos o datasets no disponibles",
)
def test_main_skip_batch_produces_report_with_green_core():
    prefix = "informe_auditoria_pytest_tmp"
    md_path = PROJECT_ROOT / "artifacts" / f"{prefix}.md"
    json_path = PROJECT_ROOT / "artifacts" / f"{prefix}.json"
    try:
        rc = audit_script.main(
            [
                "--smoke-per-dataset", "1",
                "--leakage-sample", "4",
                "--skip-batch",
                "--out-prefix", prefix,
            ]
        )
        # con --skip-batch los semaforos batch quedan AMBAR -> exit 1
        assert rc == 1
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        semaforos = payload["semaforos"]
        assert semaforos["smoke_e2e"] == audit_script.GREEN
        assert semaforos["auditoria_por_caso"] == audit_script.GREEN
        assert semaforos["baselines_congelados_vs_jorge"] == audit_script.GREEN
        assert semaforos["cobertura_capec_jorge"] == audit_script.GREEN
        assert semaforos["target_leakage"] == audit_script.GREEN
        assert semaforos["batch_deteccion_desplegado"] == audit_script.AMBER
        assert semaforos["batch_tipo_ataque_desplegado"] == audit_script.AMBER
        assert "batch_tipo_ataque" in payload
        assert payload["todo_verde"] is False
        assert md_path.exists()
        assert "Semaforos" in md_path.read_text(encoding="utf-8")
    finally:
        md_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
