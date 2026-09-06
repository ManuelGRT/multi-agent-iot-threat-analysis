"""Entrena el clasificador ampliado y balanceado de 16 tipos de ataque.

El protocolo conserva las catorce clases de Edge-IIoTset y añade ``DoS`` y
``Command_and_Control``. Las etiquetas genéricas se armonizan con reglas
versionadas. El protocolo crudo empleado para desambiguar DDoS procede del
manifiesto y se usa exclusivamente para construir el target; nunca entra en
las características predictivas. El runner no hace llamadas de red ni activa
automáticamente el candidato producido.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any, Iterable, Sequence


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.contracts.attack_taxonomy import (  # noqa: E402
    MULTIDATASET_ATTACK_CLASSES,
    MULTIDATASET_TAXONOMY_VERSION,
    multidataset_taxonomy_report,
)
from src.eval.classifier_balancing import (  # noqa: E402
    MappedClassifierRecord,
    RejectedClassifierRecord,
    balanced_origin_test_views,
    build_balanced_stratified_classifier_corpus,
    deduplicate_classifier_universe,
    map_multidataset_classifier_records,
)
from src.eval.classifier_evaluation import (  # noqa: E402
    evaluate_classifier_view,
    evaluate_origin_only_baseline,
    evaluate_rejected_attack_confidence,
    select_confidence_threshold,
)
from src.eval.classifier_targets import (  # noqa: E402
    MappingContext,
    recover_bot_iot_target_subclasses,
    recover_classifier_mapping_context,
)
from src.eval.production_models import (  # noqa: E402
    CandidateTrainingResult,
    persist_candidate_model,
    train_labelled_attack_subtype_classifier,
)
from src.eval.validation_campaign import (  # noqa: E402
    PreflightRequirements,
    load_validation_campaign,
)


DEFAULT_PROVIDER = "mistral"
DEFAULT_MODEL = "mistral-small-2603"
MODEL_NAME = "xgboost_attack_subtype_multidataset16_balanced500_20260906"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _print(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def _paths(directory: Path, pattern: str) -> tuple[Path, ...]:
    paths = tuple(sorted(directory.expanduser().resolve().glob(pattern)))
    if not paths:
        raise FileNotFoundError(f"No hay archivos {pattern!r} en {directory}")
    return paths


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _input_inventory(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in paths
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _environment_report() -> dict[str, Any]:
    packages = {}
    for name in ("joblib", "numpy", "scikit-learn", "xgboost"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _training_taxonomy_report() -> dict[str, Any]:
    report = multidataset_taxonomy_report()
    report["training_classes"] = list(MULTIDATASET_ATTACK_CLASSES)
    report["training_protocol"] = (
        "multidataset16_minimum_class_support_then_origin_stratified_split"
    )
    core = {key: value for key, value in report.items() if key != "sha256"}
    report["sha256"] = _json_sha256(core)
    return report


def _train_and_evaluate(
    records: Sequence[MappedClassifierRecord],
    *,
    fixed_threshold: float | None,
    maximum_validation_risk: float,
    minimum_validation_coverage: float,
    seed: int,
) -> tuple[CandidateTrainingResult, dict[str, Any], dict[str, Any]]:
    result = train_labelled_attack_subtype_classifier(
        [(item.record, item.label) for item in records],
        taxonomy_mapping=_training_taxonomy_report(),
        seed=seed,
    )
    by_split = {
        split: [item for item in records if item.record.split == split]
        for split in ("val", "test")
    }
    if fixed_threshold is None:
        threshold_selection = select_confidence_threshold(
            result.model,
            by_split["val"],
            maximum_risk=maximum_validation_risk,
            minimum_coverage=minimum_validation_coverage,
        )
        threshold = float(threshold_selection["selected"]["threshold"])
    else:
        validation = evaluate_classifier_view(
            result.model,
            by_split["val"],
            confidence_threshold=fixed_threshold,
            complete_label_space=True,
        )["selective"]
        threshold = float(fixed_threshold)
        threshold_selection = {
            "selection_split": None,
            "outcome": "fixed_a_priori",
            "selected": validation,
        }
    evaluations = {
        split: evaluate_classifier_view(
            result.model,
            rows,
            confidence_threshold=threshold,
            complete_label_space=True,
        )
        for split, rows in by_split.items()
    }
    return result, evaluations, threshold_selection


def _removed_manifest_ids(deduplication: dict[str, Any]) -> dict[str, list[str]]:
    removed: dict[str, list[str]] = {}
    for group in deduplication["removed_groups"]:
        kept = group.get("kept_manifest_id")
        for manifest_id in group["manifest_ids"]:
            if manifest_id != kept:
                removed[str(manifest_id)] = list(group["reasons"])
    return removed


def _mapping_rows(
    mapped: Sequence[MappedClassifierRecord],
    rejected: Sequence[RejectedClassifierRecord],
    *,
    native_subclasses: dict[str, str],
    mapping_contexts: dict[str, MappingContext],
    deduplication: dict[str, Any],
) -> list[dict[str, Any]]:
    removed = _removed_manifest_ids(deduplication)
    output: list[dict[str, Any]] = []
    for item in mapped:
        record = item.record
        context = mapping_contexts.get(record.manifest_id)
        output.append(
            {
                "manifest_id": record.manifest_id,
                "dataset": record.dataset,
                "source_file": record.source_file,
                "row_id": record.row_id,
                "split": record.split,
                "native_class": record.target_class,
                "native_subclass": native_subclasses.get(record.manifest_id),
                "status": item.mapping_status,
                "reason": item.mapping_reason,
                "attack_type": item.label,
                "dataset_origin": item.dataset_origin,
                "detailed_origin": item.detailed_origin,
                "raw_transport_proto": context.transport_proto if context else None,
                "raw_service": context.app_proto if context else None,
                "mapping_context_source": context.source if context else None,
                "mapping_context_used_as_feature": False,
                "taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
                "deduplication_status": (
                    "removed" if record.manifest_id in removed else "kept"
                ),
                "deduplication_reasons": removed.get(record.manifest_id, []),
            }
        )
    for item in rejected:
        record = item.record
        context = mapping_contexts.get(record.manifest_id)
        output.append(
            {
                "manifest_id": record.manifest_id,
                "dataset": record.dataset,
                "source_file": record.source_file,
                "row_id": record.row_id,
                "split": record.split,
                "native_class": record.target_class,
                "native_subclass": item.native_subclass,
                "status": item.status,
                "reason": item.reason,
                "attack_type": None,
                "raw_transport_proto": context.transport_proto if context else None,
                "raw_service": context.app_proto if context else None,
                "mapping_context_source": context.source if context else None,
                "mapping_context_used_as_feature": False,
                "taxonomy_version": MULTIDATASET_TAXONOMY_VERSION,
                "deduplication_status": (
                    "removed" if record.manifest_id in removed else "kept"
                ),
                "deduplication_reasons": removed.get(record.manifest_id, []),
            }
        )
    return sorted(output, key=lambda row: str(row["manifest_id"]))


def _selection_rows(
    records: Sequence[MappedClassifierRecord],
    *,
    source_splits: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    original = source_splits or {}
    return [
        {
            "manifest_id": item.record.manifest_id,
            "source_campaign_split": original.get(item.record.manifest_id),
            "classifier_split": item.record.split,
            "split": item.record.split,
            "attack_type": item.label,
            "dataset_origin": item.dataset_origin,
            "detailed_origin": item.detailed_origin,
            "mapping_status": item.mapping_status,
            "mapping_reason": item.mapping_reason,
            "feature_fingerprint": item.record.feature_fingerprint,
        }
        for item in records
    ]


def _evaluate_nonempty_view(
    result: CandidateTrainingResult,
    rows: Sequence[MappedClassifierRecord],
    *,
    threshold: float,
) -> dict[str, Any]:
    if not rows:
        return {"rows": 0, "status": "not_evaluable"}
    return evaluate_classifier_view(
        result.model,
        rows,
        confidence_threshold=threshold,
    )


def _mapping_quality_views(
    result: CandidateTrainingResult,
    records: Sequence[MappedClassifierRecord],
    *,
    threshold: float,
) -> dict[str, Any]:
    test = [item for item in records if item.record.split == "test"]
    by_status = {
        status: _evaluate_nonempty_view(
            result,
            [item for item in test if item.mapping_status == status],
            threshold=threshold,
        )
        for status in ("exact", "forced")
    }
    reasons = sorted(
        {item.mapping_reason for item in test if item.mapping_status == "forced"}
    )
    by_forced_rule = {
        reason: _evaluate_nonempty_view(
            result,
            [item for item in test if item.mapping_reason == reason],
            threshold=threshold,
        )
        for reason in reasons
    }
    return {
        "split": "test",
        "by_mapping_status": by_status,
        "by_forced_rule": by_forced_rule,
        "interpretation": (
            "Las filas exactas y forzadas se informan por separado. El rendimiento "
            "sobre targets forzados mide coherencia con la regla de armonizacion, "
            "no valida independientemente que la etiqueta fina sea verdadera."
        ),
    }


def _origin_views(
    result: CandidateTrainingResult,
    selected: Sequence[MappedClassifierRecord],
    unused: Sequence[MappedClassifierRecord],
    *,
    threshold: float,
    seed: int,
) -> dict[str, Any]:
    selected_test = tuple(
        item for item in selected if item.record.split == "test"
    )
    development_ids = {
        item.record.manifest_id
        for item in selected
        if item.record.split in {"train", "val"}
    }
    unused_ids = {item.record.manifest_id for item in unused}
    if development_ids.intersection(unused_ids):
        raise AssertionError("El holdout por origen contiene filas de desarrollo")
    extended_holdout = (
        *selected_test,
        *(
            replace(item, record=replace(item.record, split="test"))
            for item in unused
        ),
    )

    def build(
        records: Sequence[MappedClassifierRecord],
        *,
        detailed: bool,
        minimum_support: int,
    ) -> dict[str, Any]:
        origin_of = (
            (lambda item: item.detailed_origin)
            if detailed
            else (lambda item: item.dataset_origin)
        )
        available: dict[str, Counter[str]] = defaultdict(Counter)
        for item in records:
            if item.record.split != "test":
                raise AssertionError("Una vista por origen contiene filas ajenas a test")
            available[origin_of(item)][item.label] += 1
        views = balanced_origin_test_views(
            records,
            detailed=detailed,
            seed=seed,
            minimum_class_support=minimum_support,
        )
        evaluations: dict[str, Any] = {}
        for origin, counts in sorted(available.items()):
            eligible = sorted(
                label for label, count in counts.items() if count >= minimum_support
            )
            excluded = {
                label: count
                for label, count in sorted(counts.items())
                if count < minimum_support
            }
            rows = views.get(origin, ())
            if len(eligible) < 2 or not rows:
                evaluations[origin] = {
                    "status": "not_evaluable",
                    "available_class_support": dict(sorted(counts.items())),
                    "eligible_classes": eligible,
                    "classes_below_minimum_support": excluded,
                }
                continue
            evaluations[origin] = {
                "status": "evaluated",
                "available_class_support": dict(sorted(counts.items())),
                "eligible_classes": eligible,
                "classes_below_minimum_support": excluded,
                **evaluate_classifier_view(
                    result.model,
                    rows,
                    confidence_threshold=threshold,
                ),
            }
        return {
            "minimum_test_rows_per_class": minimum_support,
            "origins": evaluations,
        }

    def bundle(records: Sequence[MappedClassifierRecord]) -> dict[str, Any]:
        return {
            "rows": len(records),
            "robust_balanced": {
                "by_dataset": build(
                    records, detailed=False, minimum_support=10
                ),
                "by_detailed_origin": build(
                    records, detailed=True, minimum_support=10
                ),
            },
            "all_present_classes_audit": {
                "by_dataset": build(records, detailed=False, minimum_support=1),
                "by_detailed_origin": build(
                    records, detailed=True, minimum_support=1
                ),
            },
        }

    return {
        "global_classifier_test": bundle(selected_test),
        "extended_unused_holdout": bundle(extended_holdout),
        "development_overlap_rows": 0,
        "policy": (
            "La primera vista usa el test del nuevo clasificador. La segunda anade "
            "todas las filas mapeadas que quedaron fuera del corpus global y que "
            "por tanto no participaron en entrenamiento ni validacion. En cada "
            "origen, la vista principal exige al menos 10 filas por clase y "
            "balancea las clases elegibles."
        ),
    }


def _deployment_assessment(
    result: CandidateTrainingResult,
    evaluations: dict[str, Any],
    unresolved: dict[str, Any],
) -> dict[str, Any]:
    test = evaluations["test"]
    recalls = [row["recall"] for row in test["per_class"]]
    checks = {
        "task_is_attack_subtype": result.model.task == "attack_subtype",
        "taxonomy_version_is_multidataset16": (
            result.model.family_mapping_version == MULTIDATASET_TAXONOMY_VERSION
        ),
        "exact_16_classes": set(map(str, result.model.classes))
        == set(MULTIDATASET_ATTACK_CLASSES),
        "test_macro_f1_at_least_0_80": test["macro"]["f1"] >= 0.80,
        "minimum_class_recall_at_least_0_50": min(recalls) >= 0.50,
        "selective_coverage_at_least_0_50": test["selective"]["coverage"] >= 0.50,
        "selective_risk_at_most_0_05": (
            test["selective"]["risk"] is not None
            and test["selective"]["risk"] <= 0.05
        ),
        "unresolved_acceptance_at_most_0_50": (
            unresolved["accepted_rate"] <= 0.50
        ),
    }
    return {
        "metrics_eligible_for_manual_activation": all(checks.values()),
        "checks": checks,
        "automatically_activated": False,
        "activation_policy": (
            "El runner solo persiste un candidato. La activacion requiere una "
            "decision explicita tras revisar metricas y calidad de los targets "
            "forzados."
        ),
    }


def _support_by_native_label(
    rejected: Sequence[RejectedClassifierRecord],
) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str, str]] = Counter(
        (
            str(item.record.dataset),
            str(item.record.target_class),
            item.status,
            item.reason,
        )
        for item in rejected
    )
    return [
        {
            "dataset": dataset,
            "native_class": native_class,
            "status": status,
            "reason": reason,
            "rows": rows,
        }
        for (dataset, native_class, status, reason), rows in sorted(counts.items())
    ]


def _markdown_summary(report: dict[str, Any]) -> str:
    balance = report["global_protocol"]["balance"]
    test = report["global_protocol"]["evaluation"]["test"]
    selective = test["selective"]
    mapping = report["mapping"]
    lines = [
        "# Clasificador multidataset balanceado de 16 tipos de ataque",
        "",
        f"- Taxonomia: `{report['taxonomy']['version']}`.",
        f"- Ataques mapeados antes de deduplicar: {mapping['mapped_rows']}.",
        f"- Mapeos exactos: {mapping['status_counts'].get('exact', 0)}.",
        f"- Mapeos forzados: {mapping['status_counts'].get('forced', 0)}.",
        f"- Ataques ambiguos/excluidos: {mapping['rejected_attack_rows']}.",
        f"- Corpus balanceado: {balance['selected_rows']} filas "
        f"({balance['per_class_total']} por clase).",
        "- Clases limitantes: "
        + ", ".join(f"`{label}`" for label in balance["limiting_classes"])
        + ".",
        "- El corpus se selecciona antes del split; el split fuente no se reutiliza.",
        "- Cuota por clase: "
        + ", ".join(
            f"{split}={quota}"
            for split, quota in balance["quotas_per_class"].items()
        )
        + ".",
        f"- Accuracy test: {test['accuracy']:.6f}.",
        f"- Macro-F1 test: {test['macro']['f1']:.6f}.",
        f"- Top-3 test: {test['top3_accuracy']:.6f}.",
        f"- Cobertura selectiva: {selective['coverage']:.6f}.",
        (
            f"- Riesgo selectivo: {selective['risk']:.6f}."
            if selective["risk"] is not None
            else "- Riesgo selectivo: no calculable."
        ),
        "- Activacion automatica: no.",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_paths = _paths(args.manifest_dir, "*_manifest.jsonl")
    result_paths = _paths(args.results_dir, "*_standardized.jsonl")
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _print("Inventariando entradas y validando la campaña Mistral")
    inputs = {
        "manifests": _input_inventory(manifest_paths),
        "standardized_results": _input_inventory(result_paths),
    }
    campaign = load_validation_campaign(
        manifest_paths,
        result_paths,
        PreflightRequirements(
            require_complete=True,
            require_llm=True,
            provider=args.provider,
            model=args.model,
            dedup_scope="global",
        ),
    )
    _print(f"Campaña validada: {len(campaign.joined_records)} filas")

    _print("Recuperando targets auxiliares y protocolo crudo auditable")
    native_subclasses, subclass_recovery = recover_bot_iot_target_subclasses(
        campaign.joined_records,
        repository_root=args.data_repository_root,
    )
    mapping_contexts, context_recovery = recover_classifier_mapping_context(
        campaign.joined_records,
        manifest_paths=manifest_paths,
    )

    mapped_raw, rejected_raw, mapping_report = map_multidataset_classifier_records(
        campaign.joined_records,
        native_subclasses=native_subclasses,
        mapping_contexts=mapping_contexts,
    )
    mapped, rejected, deduplication = deduplicate_classifier_universe(
        mapped_raw, rejected_raw
    )
    if any("unspecified" in item.label.casefold() for item in mapped):
        raise AssertionError("La taxonomia final no admite clases *_unspecified")
    _print(
        f"Mapeo: {len(mapped)} ataques utilizables y {len(rejected)} "
        "ambiguos/excluidos tras deduplicar"
    )
    _write_jsonl(
        out_dir / "taxonomy_mapping_sidecar.jsonl",
        _mapping_rows(
            mapped_raw,
            rejected_raw,
            native_subclasses=native_subclasses,
            mapping_contexts=mapping_contexts,
            deduplication=deduplication,
        ),
    )

    source_splits = {
        item.record.manifest_id: item.record.split for item in mapped
    }
    selected, balance = build_balanced_stratified_classifier_corpus(
        mapped,
        per_class_total=args.per_class_total,
        classes=MULTIDATASET_ATTACK_CLASSES,
        seed=args.seed,
    )
    _write_jsonl(
        out_dir / "global_selection.jsonl",
        _selection_rows(selected, source_splits=source_splits),
    )
    _print(
        f"Corpus global: {len(selected)} filas; "
        f"{balance['per_class_total']} por clase"
    )

    _print("Entrenando XGBoost de 16 clases")
    result, evaluations, threshold_selection = _train_and_evaluate(
        selected,
        fixed_threshold=args.confidence_threshold,
        maximum_validation_risk=args.maximum_validation_risk,
        minimum_validation_coverage=args.minimum_validation_coverage,
        seed=args.seed,
    )
    if set(map(str, result.model.classes)) != set(MULTIDATASET_ATTACK_CLASSES):
        raise RuntimeError("El candidato no contiene exactamente las 16 clases")
    if result.model.family_mapping_version != MULTIDATASET_TAXONOMY_VERSION:
        raise RuntimeError("El candidato no conserva la version de taxonomia")

    threshold = float(threshold_selection["selected"]["threshold"])
    selected_ids = {item.record.manifest_id for item in selected}
    unused_mapped = [
        item for item in mapped if item.record.manifest_id not in selected_ids
    ]
    unresolved_holdout = list(rejected)
    unresolved_evaluation = evaluate_rejected_attack_confidence(
        result.model,
        unresolved_holdout,
        confidence_threshold=threshold,
    )
    assessment = _deployment_assessment(
        result, evaluations, unresolved_evaluation
    )

    model_path = (
        args.model_out.expanduser().resolve()
        if args.model_out is not None
        else out_dir / "candidate_models" / f"{MODEL_NAME}.joblib"
    )
    persist_candidate_model(
        result,
        model_path,
        confidence_threshold=threshold,
        model_name=MODEL_NAME,
    )
    _print(
        "Test: accuracy={:.4f}, macro-F1={:.4f}, top-3={:.4f}".format(
            evaluations["test"]["accuracy"],
            evaluations["test"]["macro"]["f1"],
            evaluations["test"]["top3_accuracy"],
        )
    )

    global_protocol = {
        "name": (
            "classifier_multidataset16_global_balanced_by_class_then_"
            "origin_stratified_split"
        ),
        "balance": balance,
        "training": result.report,
        "evaluation": evaluations,
        "threshold_selection": threshold_selection,
        "mapping_quality_test_views": _mapping_quality_views(
            result, selected, threshold=threshold
        ),
        "evaluation_by_origin": _origin_views(
            result,
            selected,
            unused_mapped,
            threshold=threshold,
            seed=args.seed,
        ),
        "origin_only_baseline": {
            "dataset": evaluate_origin_only_baseline(selected, detailed=False),
            "detailed": evaluate_origin_only_baseline(selected, detailed=True),
        },
        "unresolved_or_excluded_holdout": unresolved_evaluation,
        "deployment_assessment": assessment,
    }
    report = {
        "schema_version": "classifier_multidataset16_balanced_resplit_evaluation_v2",
        "generated_at": utc_now(),
        "seed": args.seed,
        "network_calls": 0,
        "llm_policy": (
            "Reutilizacion exclusiva de resultados Mistral ya materializados; "
            "el entrenamiento no llama al LLM."
        ),
        "feature_policy": (
            "proto/service crudos se usan solo para construir targets DDoS y "
            "no se inyectan adicionalmente en el evento canonico ni en el vector "
            "predictivo. El evento estandarizado puede conservar por separado "
            "informacion legitima de protocolo; por ello, la evaluacion de los "
            "subtipos DDoS forzados no es una validacion independiente de la regla."
        ),
        "environment": _environment_report(),
        "inputs": inputs,
        "campaign": campaign.summary(),
        "taxonomy": multidataset_taxonomy_report(),
        "target_recovery": {
            "bot_iot_subcategory": subclass_recovery,
            "raw_mapping_context": context_recovery,
        },
        "mapping": mapping_report,
        "deduplication": deduplication,
        "global_protocol": global_protocol,
        "excluded_attack_targets": _support_by_native_label(rejected),
        "candidate_artifact": {
            "path": str(model_path),
            "bytes": model_path.stat().st_size,
            "sha256": _sha256_file(model_path),
            "activated": False,
        },
    }
    _write_json(out_dir / "classifier_multidataset16_report.json", report)
    (out_dir / "classifier_multidataset16_summary.md").write_text(
        _markdown_summary(report), encoding="utf-8", newline="\n"
    )
    _print(f"Informe final: {out_dir / 'classifier_multidataset16_report.json'}")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--data-repository-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--maximum-validation-risk", type=float, default=0.05)
    parser.add_argument("--minimum-validation-coverage", type=float, default=0.50)
    parser.add_argument("--per-class-total", type=int, default=None)
    parser.add_argument("--model-out", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
