"""Audita los cuatro artefactos vigentes del sistema en un clon limpio.

La comprobacion no reentrena modelos ni necesita los datasets originales. Usa
fichas portables versionadas para verificar el detector, el clasificador de 16
tipos y sus splits; valida el catalogo operativo v4; y ejecuta la bateria de
mutaciones del auditor sobre casos sinteticos con el contrato actual.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.final.auditor import CaseAuditor
from src.agents.final.final_mitigator import catalog_references
from src.agents.final.llm_mitigator import build_base_items
from src.contracts.attack_taxonomy import (
    MULTIDATASET_ATTACK_CLASSES,
    MULTIDATASET_TAXONOMY_VERSION,
    multidataset_taxonomy_report,
)
from src.contracts.case import (
    CaseResult,
    ClassificationInfo,
    DetectionInfo,
    ExplanationInfo,
    JudgeInfo,
    MitigationItem,
    StandardizationInfo,
    ThreatReference,
    TraceEntry,
)
from src.mcp.common import artifacts_dir, resolve_confined_path, resolve_path


GREEN, RED = "VERDE", "ROJO"
REFERENCE_ROOT = PROJECT_ROOT / "docs" / "evaluation_references"
DETECTOR_REFERENCE = (
    REFERENCE_ROOT / "xgboost_detection_balanced_by_origin_20260905.json"
)
CLASSIFIER_REFERENCE = (
    REFERENCE_ROOT
    / "xgboost_attack_type_multidataset16_balanced500_20260907.json"
)
CATALOG_REFERENCE = REFERENCE_ROOT / "threat_intel_catalog_v4_20260908.json"
AUDITOR_REFERENCE = REFERENCE_ROOT / "auditor_mutation_report_v4_20260908.json"

DETECTOR_RELEASE_HASHES = {
    "artifact": "f2d7d3dbe9c90fbd7f3d1f134d91e77f855dde134119eced812646e285524393",
    "input_snapshot": "e2117f938b9f8551f567b6f5ca1dcb9710913c760c597f70101ce15f46823666",
    "selection": "b97f53b2cea2efbf7174e7a3b49295f130d2f812f9b9f81fcbc99fe31e1246b9",
    "feature_names": "de41afdd00f73945f34c071b99752d75339b88b7af1cb813439b383cca7c9c1b",
    "booster": "26fa71ff717bc36a9305556095a989a660a5d2f596097b1abe424e385d886463",
}
DETECTOR_RELEASE_METRICS = {
    "test_attack_f1": 0.9600223651104277,
    "selective_coverage": 0.9697648376259799,
}
CLASSIFIER_RELEASE_HASHES = {
    "artifact": "9175adb6f78a69962676c9aba0e9b58a59ef94e1e500c9023f4d9b184fa531b5",
    "selection": "d39cf28b3ec96e1558fbc77327d5674a43644a5f880233bd59640e8370168f60",
    "eligible_records": "f4b6aa576b7c7b48b34023232e0abce9368c50f52fde9f3e40dff6a19c38e344",
    "train": "c53a855cbb787b8f34499fd3e61b6dbae3ccf4b2cf0c0c0a30dbc76b9dbe69fc",
    "validation": "bd62dd524b7804fe24d89b22ce57ee5d69b89717f319d48cb8161cef7826fefa",
    "test": "644ce38284d1853b38691eb12aba42f58270ec4217ed074357fec14a2268320c",
    "feature_names": "7562752da27242801d14455990f38e7b1e32771a35f96196fb499b52c0a289cd",
    "training_taxonomy": "d016e777a545c78361adb1c05352a70f36e0022c5f85e5aaf4f8e379655bd6ed",
    "runtime_taxonomy": "060a45a18e9a99e302467a12a358fe1f346091707a75375e2082a98e3506fae2",
}
CLASSIFIER_RELEASE_METRICS = {
    "test_accuracy": 0.8908333333333334,
    "test_macro_f1": 0.8916753705196387,
    "test_top3": 0.98,
    "selective_coverage": 0.9066666666666666,
    "selective_macro_f1": 0.9236604757855131,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-prefix", help="Prefijo dentro de artifacts/.")
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Muestra el resultado sin crear informes JSON/Markdown.",
    )
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _matches_float(value: Any, expected: float) -> bool:
    try:
        return abs(float(value) - expected) <= 1e-12
    except (TypeError, ValueError):
        return False


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} no contiene un objeto JSON")
    return value


def _outcome(component: str, issues: list[str], **details: Any) -> dict[str, Any]:
    return {
        "component": component,
        "semaforo": GREEN if not issues else RED,
        "issues": issues,
        **details,
    }


def _feature_names(model: Any) -> list[str]:
    method = getattr(model.vectorizer, "get_feature_names_out", None)
    return [str(value) for value in method()] if callable(method) else []


def audit_detector() -> dict[str, Any]:
    """Verifica joblib, balance 1:1, splits y metricas del detector actual."""

    issues: list[str] = []
    try:
        reference = _load_json(DETECTOR_REFERENCE)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return _outcome("detector", [f"reference_unavailable:{type(exc).__name__}"])

    artifact = reference.get("artifact") or {}
    balancing = reference.get("balancing") or {}
    splits = balancing.get("splits") or {}
    metrics = reference.get("metrics") or {}
    metadata = reference.get("model") or {}
    inputs = reference.get("inputs") or {}
    model_path = resolve_path("detection_model")

    def require(condition: bool, issue: str) -> None:
        if not condition:
            issues.append(issue)

    require(
        reference.get("schema_version") == "deployed-detector-evaluation-v1",
        "schema",
    )
    require(model_path.is_file(), "model_missing")
    if model_path.is_file():
        require(artifact.get("filename") == model_path.name, "artifact_filename")
        require(
            int(artifact.get("bytes", -1)) == model_path.stat().st_size,
            "artifact_size",
        )
        require(
            artifact.get("sha256") == _sha256_file(model_path),
            "artifact_sha256",
        )
        require(
            artifact.get("sha256") == DETECTOR_RELEASE_HASHES["artifact"],
            "artifact_release_sha256",
        )
    require(artifact.get("task") == "binary_detection", "task_reference")
    require(artifact.get("classes") == [False, True], "classes_reference")
    require(
        inputs.get("snapshot_sha256") == DETECTOR_RELEASE_HASHES["input_snapshot"],
        "input_snapshot_release_sha256",
    )

    model = None
    if model_path.is_file():
        try:
            from src.mcp import model_registry

            model_registry.clear_cache()
            model = model_registry.load_model("detection_model")
        except Exception as exc:  # noqa: BLE001 - se informa como incidencia
            issues.append(f"model_load:{type(exc).__name__}")
    if model is not None:
        names = _feature_names(model)
        require(getattr(model, "task", None) == "binary_detection", "task_model")
        require(
            tuple(getattr(model, "classes", ())) == (False, True),
            "classes_model",
        )
        require(
            len(names) == int(metadata.get("feature_count", -1)),
            "feature_count",
        )
        require(
            getattr(model, "feature_schema_sha256", None)
            == metadata.get("feature_schema_sha256")
            == _sha256_json(names),
            "feature_schema_sha256",
        )
        require(
            metadata.get("feature_schema_sha256")
            == DETECTOR_RELEASE_HASHES["feature_names"],
            "feature_release_sha256",
        )
        try:
            booster_hash = hashlib.sha256(
                model.estimator.get_booster().save_raw(raw_format="ubj")
            ).hexdigest()
        except Exception as exc:  # noqa: BLE001
            issues.append(f"booster_hash:{type(exc).__name__}")
        else:
            require(
                booster_hash
                == metadata.get("booster_raw_sha256")
                == DETECTOR_RELEASE_HASHES["booster"],
                "booster_release_sha256",
            )

    rows = int(balancing.get("rows", -1))
    counts = balancing.get("class_counts") or {}
    require(
        rows > 0 and counts == {"benign": rows // 2, "attack": rows // 2},
        "global_balance",
    )
    require(set(splits) == {"train", "validation", "test"}, "split_names")
    split_rows = 0
    for split, values in splits.items():
        current_rows = int(values.get("rows", -1))
        per_class = int(values.get("per_class", -1))
        require(
            current_rows == 2 * per_class and per_class > 0,
            f"split_balance:{split}",
        )
        split_rows += current_rows
    require(split_rows == rows, "split_total")
    require(balancing.get("seed") == 42, "balance_seed")
    require(
        balancing.get("selected_manifest_ids_sha256")
        == DETECTOR_RELEASE_HASHES["selection"],
        "selection_release_sha256",
    )
    require(
        (reference.get("deduplication") or {}).get(
            "final_cross_split_overlap_count"
        )
        == 0,
        "cross_split_overlap",
    )

    test = metrics.get("test") or {}
    operational = metrics.get("operational_test") or {}
    test_rows = int((splits.get("test") or {}).get("rows", -1))
    require(int(test.get("rows", -2)) == test_rows, "test_rows")
    matrix = test.get("confusion_matrix") or []
    require(
        sum(sum(int(value) for value in row) for row in matrix) == test_rows,
        "test_confusion_total",
    )
    decided = int(operational.get("decided_rows", -1))
    abstained = int(operational.get("abstained_rows", -1))
    require(decided + abstained == test_rows, "selective_total")
    if test_rows > 0:
        require(
            abs(float(operational.get("coverage", -1)) - decided / test_rows)
            <= 1e-12,
            "selective_coverage",
        )
    require(
        _matches_float(
            test.get("attack_f1"), DETECTOR_RELEASE_METRICS["test_attack_f1"]
        ),
        "test_metric_release",
    )
    require(
        _matches_float(
            operational.get("coverage"),
            DETECTOR_RELEASE_METRICS["selective_coverage"],
        ),
        "selective_metric_release",
    )

    return _outcome(
        "detector",
        issues,
        artifact=model_path.name,
        sha256=_sha256_file(model_path) if model_path.is_file() else None,
        corpus_rows=rows,
        splits={key: value.get("rows") for key, value in splits.items()},
        test_attack_f1=test.get("attack_f1"),
        selective_coverage=operational.get("coverage"),
    )


def audit_classifier() -> dict[str, Any]:
    """Verifica joblib, taxonomia, balance y split 70/15/15 del modelo de 16 tipos."""

    issues: list[str] = []
    try:
        reference = _load_json(CLASSIFIER_REFERENCE)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return _outcome("classifier", [f"reference_unavailable:{type(exc).__name__}"])

    artifact = reference.get("artifact") or {}
    taxonomy = reference.get("taxonomy") or {}
    balancing = reference.get("balancing") or {}
    splits = reference.get("splits") or {}
    supports = splits.get("supports") or {}
    hashes = reference.get("hashes") or {}
    model_metadata = reference.get("model") or {}
    contract = model_metadata.get("operational_contract") or {}
    metrics = reference.get("metrics") or {}
    test = metrics.get("test") or {}
    model_path = resolve_path("attack_type_model")
    expected_classes = tuple(MULTIDATASET_ATTACK_CLASSES)
    runtime_taxonomy = multidataset_taxonomy_report()

    def require(condition: bool, issue: str) -> None:
        if not condition:
            issues.append(issue)

    require(
        reference.get("schema_version") == "classifier-evaluation-reference-v1",
        "schema",
    )
    require(model_path.is_file(), "model_missing")
    if model_path.is_file():
        actual_hash = _sha256_file(model_path)
        require(artifact.get("filename") == model_path.name, "artifact_filename")
        require(
            int(artifact.get("bytes", -1)) == model_path.stat().st_size,
            "artifact_size",
        )
        require(
            artifact.get("sha256") == actual_hash == hashes.get("artifact"),
            "artifact_sha256",
        )
        require(
            actual_hash == CLASSIFIER_RELEASE_HASHES["artifact"],
            "artifact_release_sha256",
        )
    require(artifact.get("deployed") is True, "artifact_not_deployed")
    require(
        taxonomy.get("version") == MULTIDATASET_TAXONOMY_VERSION,
        "taxonomy_version",
    )
    require(
        tuple(taxonomy.get("attack_classes") or ()) == expected_classes,
        "taxonomy_classes",
    )
    require(taxonomy.get("class_count") == 16, "taxonomy_class_count")
    training_taxonomy_sha256 = taxonomy.get("training_taxonomy_sha256")
    runtime_taxonomy_sha256 = taxonomy.get("runtime_taxonomy_sha256")
    require(
        training_taxonomy_sha256
        == hashes.get("training_taxonomy")
        == CLASSIFIER_RELEASE_HASHES["training_taxonomy"],
        "training_taxonomy_sha256",
    )
    require(
        runtime_taxonomy_sha256
        == hashes.get("runtime_taxonomy")
        == runtime_taxonomy.get("sha256"),
        "runtime_taxonomy_sha256",
    )
    require(
        runtime_taxonomy_sha256
        == CLASSIFIER_RELEASE_HASHES["runtime_taxonomy"],
        "runtime_taxonomy_release_sha256",
    )

    model = None
    if model_path.is_file():
        try:
            from src.mcp import model_registry

            model_registry.clear_cache()
            model = model_registry.load_model("attack_type_model")
        except Exception as exc:  # noqa: BLE001
            issues.append(f"model_load:{type(exc).__name__}")
    if model is not None:
        names = _feature_names(model)
        require(getattr(model, "task", None) == "attack_subtype", "task_model")
        require(
            tuple(map(str, getattr(model, "classes", ())))
            == tuple(model_metadata.get("classes") or ()),
            "classes_model",
        )
        require(
            set(map(str, getattr(model, "classes", ()))) == set(expected_classes),
            "classes_operational",
        )
        require(
            getattr(model, "family_mapping_version", None)
            == MULTIDATASET_TAXONOMY_VERSION,
            "taxonomy_model",
        )
        require(
            float(getattr(model, "confidence_threshold", -1)) == 0.65,
            "threshold_model",
        )
        require(
            getattr(model, "model_name", None) == contract.get("model_name"),
            "model_name",
        )
        require(
            len(names) == int(model_metadata.get("feature_count", -1)),
            "feature_count",
        )
        require(
            getattr(model, "feature_schema_sha256", None)
            == hashes.get("feature_names")
            == _sha256_json(names),
            "feature_schema_sha256",
        )
        require(
            hashes.get("feature_names")
            == CLASSIFIER_RELEASE_HASHES["feature_names"],
            "feature_release_sha256",
        )
    require(contract.get("task") == "attack_subtype", "task_reference")
    require(contract.get("public_task") == "attack_type", "public_task")
    require(
        contract.get("taxonomy_version") == MULTIDATASET_TAXONOMY_VERSION,
        "contract_taxonomy",
    )
    require(
        float(contract.get("confidence_threshold", -1)) == 0.65,
        "threshold_reference",
    )

    per_class = int(balancing.get("per_class_total", -1))
    selected_rows = int(balancing.get("selected_rows", -1))
    require(
        per_class == 500 and selected_rows == 16 * per_class,
        "global_balance",
    )
    require(balancing.get("seed") == 42, "balance_seed")
    require(
        balancing.get("sampling_with_replacement") is False,
        "sampling_replacement",
    )
    require(
        splits.get("ratio_units") == {"train": 14, "val": 3, "test": 3},
        "split_ratio",
    )
    quotas = splits.get("quotas_per_class") or {}
    require(
        quotas == {"train": 350, "val": 75, "test": 75},
        "split_quotas",
    )
    require(
        splits.get("final_cross_split_overlap_count") == 0,
        "cross_split_overlap",
    )
    split_rows: dict[str, int] = {}
    for split, quota in quotas.items():
        values = supports.get(split) or {}
        class_counts = values.get("class_counts") or {}
        rows = int(values.get("rows", -1))
        split_rows[split] = rows
        require(rows == 16 * quota, f"split_rows:{split}")
        require(
            set(class_counts) == set(expected_classes), f"split_classes:{split}"
        )
        require(
            all(int(value) == quota for value in class_counts.values()),
            f"split_balance:{split}",
        )
    require(sum(split_rows.values()) == selected_rows, "split_total")
    for name, expected_hash in CLASSIFIER_RELEASE_HASHES.items():
        require(hashes.get(name) == expected_hash, f"release_hash:{name}")

    test_rows = int(test.get("rows", -1))
    require(test_rows == split_rows.get("test"), "test_rows")
    require(
        test.get("class_support") == supports.get("test", {}).get("class_counts"),
        "test_support",
    )
    selective = (metrics.get("selective") or {}).get("test") or {}
    require(selective.get("threshold") == 0.65, "selective_threshold")
    require(
        int(selective.get("decided_rows", -1))
        + int(selective.get("abstained_rows", -1))
        == test_rows,
        "selective_total",
    )
    for field, expected in (
        ("accuracy", CLASSIFIER_RELEASE_METRICS["test_accuracy"]),
        ("top3_accuracy", CLASSIFIER_RELEASE_METRICS["test_top3"]),
    ):
        require(
            _matches_float(test.get(field), expected),
            f"test_metric_release:{field}",
        )
    require(
        _matches_float(
            (test.get("macro") or {}).get("f1"),
            CLASSIFIER_RELEASE_METRICS["test_macro_f1"],
        ),
        "test_metric_release:macro_f1",
    )
    require(
        _matches_float(
            selective.get("coverage"),
            CLASSIFIER_RELEASE_METRICS["selective_coverage"],
        ),
        "selective_metric_release:coverage",
    )
    require(
        _matches_float(
            selective.get("macro_f1"),
            CLASSIFIER_RELEASE_METRICS["selective_macro_f1"],
        ),
        "selective_metric_release:macro_f1",
    )
    require(
        (metrics.get("per_origin") or {}).get("development_overlap_rows") == 0,
        "origin_holdout_overlap",
    )
    detailed = (
        (metrics.get("per_origin") or {})
        .get("extended_unused_holdout", {})
        .get("by_detailed_origin", {})
        .get("origins", {})
    )
    for origin in (
        "ton_iot_linux",
        "ton_iot_network",
        "ton_iot_telemetry",
        "ton_iot_windows",
    ):
        require(origin in detailed, f"origin_missing:{origin}")

    return _outcome(
        "classifier",
        issues,
        artifact=model_path.name,
        sha256=_sha256_file(model_path) if model_path.is_file() else None,
        corpus_rows=selected_rows,
        per_class=per_class,
        splits=split_rows,
        test_accuracy=test.get("accuracy"),
        test_macro_f1=(test.get("macro") or {}).get("f1"),
        test_top3=test.get("top3_accuracy"),
        selective=selective,
        training_taxonomy_sha256=training_taxonomy_sha256,
        runtime_taxonomy_sha256=runtime_taxonomy_sha256,
    )


def audit_catalog() -> dict[str, Any]:
    """Valida el fichero v4 y su cobertura exacta de los 16 tipos."""

    issues: list[str] = []
    try:
        reference = _load_json(CATALOG_REFERENCE)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return _outcome("catalog", [f"reference_unavailable:{type(exc).__name__}"])

    try:
        from src.mcp.threat_catalog import (
            CATALOG_PATH,
            THREAT_INTEL_CATALOG_VERSION,
            catalog,
        )
        from src.mcp.threat_intel_server import (
            get_multidataset_attack_type_coverage,
        )

        catalog.cache_clear()
        value = catalog()
        coverage = get_multidataset_attack_type_coverage()
    except Exception as exc:  # noqa: BLE001
        return _outcome("catalog", [f"catalog_invalid:{type(exc).__name__}:{exc}"])

    artifact = reference.get("artifact") or {}
    taxonomy = reference.get("taxonomy") or {}

    def require(condition: bool, issue: str) -> None:
        if not condition:
            issues.append(issue)

    require(
        reference.get("schema_version")
        == "threat-intel-catalog-evaluation-v1",
        "schema",
    )
    require(artifact.get("filename") == CATALOG_PATH.name, "artifact_filename")
    require(
        int(artifact.get("bytes", -1)) == CATALOG_PATH.stat().st_size,
        "artifact_size",
    )
    require(
        artifact.get("sha256") == _sha256_file(CATALOG_PATH), "artifact_sha256"
    )
    require(
        value.get("version")
        == THREAT_INTEL_CATALOG_VERSION
        == artifact.get("version"),
        "catalog_version",
    )
    require(
        taxonomy.get("version") == MULTIDATASET_TAXONOMY_VERSION,
        "taxonomy_version",
    )
    require(
        tuple(taxonomy.get("classes") or ()) == MULTIDATASET_ATTACK_CLASSES,
        "taxonomy_classes",
    )
    require(taxonomy.get("class_count") == 16, "taxonomy_class_count")
    require(coverage.get("ok") is True, "coverage_tool_error")
    require(
        coverage.get("covered") == coverage.get("total") == 16, "coverage"
    )
    reference_counts = reference.get("mitigations_per_attack_type") or {}
    observed_counts = {
        attack_type: sum(len(items) for items in entry["mitigations"].values())
        for attack_type, entry in value["attack_types"].items()
    }
    require(observed_counts == reference_counts, "mitigation_counts")
    require(
        all(count >= 5 for count in observed_counts.values()),
        "minimum_mitigations",
    )
    observed_reference_counts = {
        group: len(entries)
        for group, entries in value["reference_catalog"].items()
    }
    require(
        observed_reference_counts
        == reference.get("reference_catalog_counts"),
        "reference_catalog_counts",
    )
    require(
        reference.get("validation")
        == {
            "coverage": "16/16",
            "minimum_mitigations_per_type": 5,
            "lookup_scope": "attack_type",
            "broad_family_mapping_present": False,
            "historical_jorge_table_present": False,
        },
        "catalog_validation_contract",
    )

    return _outcome(
        "catalog",
        issues,
        artifact=CATALOG_PATH.name,
        sha256=_sha256_file(CATALOG_PATH),
        version=value.get("version"),
        taxonomy_version=taxonomy.get("version"),
        covered_attack_types=coverage.get("covered"),
        mitigations_per_attack_type=observed_counts,
        reference_catalog_counts=observed_reference_counts,
    )


T0 = datetime(2026, 9, 8, 12, 0, 0, tzinfo=timezone.utc)
FULL_PATH = (
    "orchestrator",
    "final_standardizer",
    "final_detector",
    "final_classifier",
    "final_mitigator",
    "final_judge",
)
BENIGN_PATH = (
    "orchestrator",
    "final_standardizer",
    "final_detector",
    "final_judge",
)
AUDITOR_MUTATION_CHECKS = {
    "classification_low_confidence_unreviewed": "umbral_confianza_clasificacion_derivada",
    "classification_threshold_tampered": "consistencia_umbral_clasificacion_operativo",
    "catalog_version_obsolete": "consistencia_mitigacion_catalogo_tipado",
    "catalog_taxonomy_obsolete": "consistencia_mitigacion_catalogo_tipado",
    "catalog_mitigation_invented": "consistencia_contenido_catalogo_tipado",
    "catalog_reference_invented": "consistencia_contenido_catalogo_tipado",
    "benign_with_attack_type": "consistencia_benigno_sin_tipo_ataque",
    "malicious_without_classification": "consistencia_malicioso_con_tipo_ataque",
    "detection_label_probability_mismatch": "consistencia_etiqueta_vs_probabilidad",
    "gray_zone_without_abstention": "umbral_zona_gris_abstiene",
    "trace_empty": "traza_no_vacia",
    "trace_unfinished": "traza_entradas_cerradas",
    "trace_agent_order_invalid": "traza_orden_de_agentes",
    "trace_error_unreported": "traza_errores_derivados",
    "judge_status_incompatible": "consistencia_estado_vs_juez",
    "canonical_target_leakage": "leakage_evento_canonico",
}


def _trace(agents: tuple[str, ...]) -> list[TraceEntry]:
    return [
        TraceEntry(
            agent=agent,
            status="ok",
            started_at=T0 + timedelta(seconds=index),
            finished_at=T0 + timedelta(seconds=index, milliseconds=500),
        )
        for index, agent in enumerate(agents)
    ]


def _canonical_event() -> dict[str, Any]:
    return {
        "event_id": "audit-current-contract",
        "modality": "network_flow",
        "src_ip": "10.0.0.2",
        "dst_ip": "10.0.0.3",
        "src_port": 49152,
        "dst_port": 443,
        "transport_proto": "tcp",
        "packet_count": 120,
        "byte_count": 4096,
        "duration_ms": 1500.0,
        "schema_profile": "network_flow",
        "semantic_text": "tcp flow with high packet rate",
        "provenance": {"dataset": "audit", "row_id": 1},
        "mapping_confidence": 0.95,
    }


def _valid_attack_case() -> CaseResult:
    from src.mcp.threat_intel_server import suggest_mitigations

    catalog_result = suggest_mitigations("DDoS_TCP")
    if catalog_result.get("ok") is not True:
        raise RuntimeError(str(catalog_result.get("error")))
    base_items = build_base_items(catalog_result)
    references = [
        ThreatReference(**value)
        for value in catalog_references(catalog_result.get("references") or {})
    ]
    return CaseResult(
        case_id="case-auditor-valid-attack",
        canonical_event=_canonical_event(),
        standardization=StandardizationInfo(mapping_confidence=0.95),
        detection=DetectionInfo(
            is_malicious=True, probability=0.97, model_name="detector-current"
        ),
        classification=ClassificationInfo(
            attack_type="DDoS_TCP",
            confidence=0.95,
            decision_threshold=0.65,
            model_name="classifier-current",
            taxonomy_version=MULTIDATASET_TAXONOMY_VERSION,
            top_scores={"DDoS_TCP": 0.95, "DDoS_UDP": 0.03, "XSS": 0.02},
        ),
        explanation=ExplanationInfo(
            summary="Respuesta catalogada para DDoS TCP.",
            mitigations=[item["text"] for item in base_items],
            mitigation_items=[
                MitigationItem(
                    text=item["text"], phase=item["phase"], source="catalog"
                )
                for item in base_items
            ],
            references=references,
            confidence=0.95,
            attack_type="DDoS_TCP",
            taxonomy_version=MULTIDATASET_TAXONOMY_VERSION,
            catalog_scope="attack_type",
            catalog_version=str(catalog_result["catalog_version"]),
            catalog_taxonomy_version=str(catalog_result["taxonomy_version"]),
            catalog_compatible_taxonomy_versions=list(
                catalog_result["compatible_taxonomy_versions"]
            ),
            reference_quality=dict(
                catalog_result.get("reference_quality") or {}
            ),
            source="catalog",
        ),
        judge=JudgeInfo(
            action="approve",
            approved=True,
            final_label="DDoS_TCP",
            final_confidence=0.95,
        ),
        trace=_trace(FULL_PATH),
    ).close()


def _valid_benign_case() -> CaseResult:
    return CaseResult(
        case_id="case-auditor-valid-benign",
        canonical_event=_canonical_event(),
        standardization=StandardizationInfo(mapping_confidence=0.95),
        detection=DetectionInfo(
            is_malicious=False,
            probability=0.03,
            model_name="detector-current",
        ),
        judge=JudgeInfo(
            action="approve",
            approved=True,
            final_label="benign",
            final_confidence=0.97,
        ),
        trace=_trace(BENIGN_PATH),
    ).close()


def _valid_review_case() -> CaseResult:
    return CaseResult(
        case_id="case-auditor-valid-review",
        canonical_event=_canonical_event(),
        standardization=StandardizationInfo(mapping_confidence=0.95),
        detection=DetectionInfo(
            is_malicious=None, probability=0.5, abstain=True
        ),
        judge=JudgeInfo(
            action="human_interrupt",
            approved=False,
            requires_human_review=True,
            final_label=None,
            final_confidence=0.0,
            issues=["detector_abstained"],
        ),
        trace=_trace(BENIGN_PATH),
    ).close()


def _mutated_case(mutation_id: str) -> CaseResult:
    case = _valid_attack_case().model_copy(deep=True)
    if mutation_id == "classification_low_confidence_unreviewed":
        case.classification.confidence = 0.60
        case.classification.top_scores = {
            "DDoS_TCP": 0.60,
            "DDoS_UDP": 0.25,
            "XSS": 0.15,
        }
        case.judge.final_confidence = 0.60
    elif mutation_id == "classification_threshold_tampered":
        case.classification.decision_threshold = 0.10
    elif mutation_id == "catalog_version_obsolete":
        case.explanation.catalog_version = "2.0"
    elif mutation_id == "catalog_taxonomy_obsolete":
        case.explanation.catalog_taxonomy_version = (
            "jorge_edge_attack_type_v1_2026-09-06"
        )
    elif mutation_id == "catalog_mitigation_invented":
        case.explanation.mitigations.append("accion no catalogada")
        case.explanation.mitigation_items.append(
            MitigationItem(
                text="accion no catalogada",
                phase="prevention",
                source="catalog",
            )
        )
    elif mutation_id == "catalog_reference_invented":
        case.explanation.references.append(
            ThreatReference(
                attack_id="T9999",
                name="Referencia inexistente",
                url="https://attack.mitre.org/techniques/T9999/",
                source="catalog",
            )
        )
    elif mutation_id == "benign_with_attack_type":
        case = _valid_benign_case().model_copy(deep=True)
        case.classification = ClassificationInfo(
            attack_type="DDoS_TCP",
            confidence=0.95,
            taxonomy_version=MULTIDATASET_TAXONOMY_VERSION,
            top_scores={"DDoS_TCP": 0.95, "DDoS_UDP": 0.03, "XSS": 0.02},
        )
    elif mutation_id == "malicious_without_classification":
        case.classification = ClassificationInfo()
    elif mutation_id == "detection_label_probability_mismatch":
        case.detection.probability = 0.10
    elif mutation_id == "gray_zone_without_abstention":
        case.detection.probability = 0.50
    elif mutation_id == "trace_empty":
        case.trace = []
    elif mutation_id == "trace_unfinished":
        case.trace[-1].finished_at = None
    elif mutation_id == "trace_agent_order_invalid":
        case.trace = _trace(
            (
                "orchestrator",
                "final_standardizer",
                "final_classifier",
                "final_detector",
                "final_mitigator",
                "final_judge",
            )
        )
    elif mutation_id == "trace_error_unreported":
        case.trace[2].status = "error"
    elif mutation_id == "judge_status_incompatible":
        case.judge.approved = False
    elif mutation_id == "canonical_target_leakage":
        case.canonical_event["label_raw"] = "DDoS_TCP"
    else:
        raise KeyError(f"Mutacion desconocida: {mutation_id}")
    return case


def audit_auditor() -> dict[str, Any]:
    """Ejecuta los mutantes declarados y exige el chequeo esperado."""

    issues: list[str] = []
    try:
        reference = _load_json(AUDITOR_REFERENCE)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return _outcome("auditor", [f"reference_unavailable:{type(exc).__name__}"])

    implementation = reference.get("implementation") or {}
    auditor_path = PROJECT_ROOT / "src" / "agents" / "final" / "auditor.py"
    catalog_contract_path = PROJECT_ROOT / "src" / "mcp" / "threat_catalog.py"
    catalog_path = PROJECT_ROOT / "src" / "mcp" / "data" / "threat_intel_catalog.json"
    if reference.get("schema_version") != "auditor-mutation-evaluation-v1":
        issues.append("schema")
    if implementation.get("auditor_sha256") != _sha256_file(auditor_path):
        issues.append("auditor_sha256")
    if implementation.get("catalog_contract_sha256") != _sha256_file(
        catalog_contract_path
    ):
        issues.append("catalog_contract_sha256")
    if implementation.get("catalog_sha256") != _sha256_file(catalog_path):
        issues.append("catalog_sha256")

    expected_protocol = {
        "method": "deterministic_case_mutation_testing",
        "split_policy": "not_applicable_case_contract_evaluation",
        "catalog_version": "4.0",
        "taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
        "classification_threshold": 0.65,
        "valid_controls": 3,
    }
    if reference.get("protocol") != expected_protocol:
        issues.append("mutation_protocol")
    expected_summary = {
        "mutations": len(AUDITOR_MUTATION_CHECKS),
        "detected": len(AUDITOR_MUTATION_CHECKS),
        "false_rejects": 0,
    }
    if reference.get("expected") != expected_summary:
        issues.append("mutation_expected_summary")

    raw_specifications = reference.get("mutations")
    specifications: list[tuple[str, str]] = []
    if not isinstance(raw_specifications, list):
        issues.append("mutation_specifications")
    else:
        for specification in raw_specifications:
            if not isinstance(specification, dict):
                issues.append("mutation_specifications")
                continue
            mutation_id = specification.get("id")
            expected_check = specification.get("expected_check")
            if not isinstance(mutation_id, str) or not isinstance(
                expected_check, str
            ):
                issues.append("mutation_specifications")
                continue
            specifications.append((mutation_id, expected_check))
    declared_mutations = dict(specifications)
    if (
        len(declared_mutations) != len(specifications)
        or declared_mutations != AUDITOR_MUTATION_CHECKS
    ):
        issues.append("mutation_contract")

    auditor = CaseAuditor()
    controls = {
        "valid_attack": (_valid_attack_case(), "approve"),
        "valid_benign": (_valid_benign_case(), "approve"),
        "valid_human_review": (_valid_review_case(), "review"),
    }
    control_results: list[dict[str, Any]] = []
    false_rejects = 0
    for control_id, (case, expected_verdict) in controls.items():
        report = auditor.audit(case)
        accepted = report.verdict == expected_verdict
        false_rejects += int(report.verdict == "reject")
        if not accepted:
            issues.append(f"control:{control_id}:{report.verdict}")
        control_results.append(
            {
                "id": control_id,
                "expected_verdict": expected_verdict,
                "verdict": report.verdict,
                "accepted": accepted,
            }
        )

    mutation_results: list[dict[str, Any]] = []
    for mutation_id, expected_check in specifications:
        try:
            report = auditor.audit(_mutated_case(mutation_id))
        except Exception as exc:  # noqa: BLE001
            issues.append(f"mutation_crash:{mutation_id}:{type(exc).__name__}")
            mutation_results.append(
                {"id": mutation_id, "detected": False, "error": str(exc)}
            )
            continue
        failed_checks = [check.check for check in report.checks if not check.passed]
        detected = report.verdict != "approve" and expected_check in failed_checks
        if not detected:
            issues.append(
                f"mutation_not_detected:{mutation_id}:{expected_check}"
            )
        mutation_results.append(
            {
                "id": mutation_id,
                "expected_check": expected_check,
                "verdict": report.verdict,
                "failed_checks": failed_checks,
                "detected": detected,
            }
        )

    expected = reference.get("expected") or {}
    detected_count = sum(
        int(row.get("detected") is True) for row in mutation_results
    )
    expected_total = int(expected.get("mutations", -1))
    expected_detected = int(expected.get("detected", -1))
    if (
        len(mutation_results) != expected_total
        or detected_count != expected_detected
    ):
        issues.append("mutation_totals")
    if false_rejects != int(expected.get("false_rejects", -1)):
        issues.append("false_rejects")

    return _outcome(
        "auditor",
        issues,
        protocol=reference.get("protocol"),
        mutations=len(mutation_results),
        detected=detected_count,
        false_rejects=false_rejects,
        controls=control_results,
        results=mutation_results,
    )


def run_audit() -> dict[str, Any]:
    results = {
        "detector": audit_detector(),
        "classifier": audit_classifier(),
        "catalog": audit_catalog(),
        "auditor": audit_auditor(),
    }
    semaphores = {name: result["semaforo"] for name, result in results.items()}
    return {
        "schema_version": "current-system-audit-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "semaforos": semaphores,
        "todo_verde": all(value == GREEN for value in semaphores.values()),
        "results": results,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Auditoria de artefactos actuales del sistema multiagente",
        "",
        f"Generado: `{payload['generated_at']}`",
        "",
        "| Componente | Resultado | Incidencias |",
        "|---|---|---|",
    ]
    for name, result in payload["results"].items():
        issues = ", ".join(result["issues"]) if result["issues"] else "ninguna"
        lines.append(f"| {name} | {result['semaforo']} | {issues} |")
    lines.extend(
        [
            "",
            "La auditoria usa fichas portables para los splits y modelos; no "
            "reentrena ni presenta como recalculadas las metricas congeladas.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_audit()
    outputs: dict[str, str] = {}
    if not args.no_write:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        prefix = args.out_prefix or f"informe_auditoria_actual_{stamp}"
        try:
            json_path = resolve_confined_path(
                f"{prefix}.json", artifacts_dir(), allow_absolute=False
            )
            markdown_path = resolve_confined_path(
                f"{prefix}.md", artifacts_dir(), allow_absolute=False
            )
        except ValueError as exc:
            print(f"ERROR: prefijo de informe no permitido: {exc}")
            return 2
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(_markdown(payload), encoding="utf-8")
        outputs = {"json": str(json_path), "markdown": str(markdown_path)}
    print(
        json.dumps(
            {
                "semaforos": payload["semaforos"],
                "todo_verde": payload["todo_verde"],
                "outputs": outputs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if payload["todo_verde"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
