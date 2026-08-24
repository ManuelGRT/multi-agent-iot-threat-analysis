"""Fase D: evaluacion reproducible por dataset sobre la campana validada.

El manifiesto congelado es la unica autoridad para splits y targets. Los
resultados de estandarizacion solo aportan el evento canonico y sus features.
Por defecto se exige cobertura completa, parseo LLM, Mistral y un modelo fijo.

La CLI no serializa estimadores: conserva metricas de validacion para todos los
modelos y publica test unicamente para el ganador elegido previamente en val.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.eval.jorge_models import (  # noqa: E402
    BenchmarkRequirementsNotMet,
    SUPPORTED_MODELS,
    MLPBackendFactory,
    run_jorge_benchmarks,
)
from src.eval.external_mlp import (  # noqa: E402
    ExternalTorchMLPFactory,
    detect_torch_interpreter,
)
from src.eval.production_models import train_production_candidates  # noqa: E402
from src.eval.validation_campaign import (  # noqa: E402
    ALLOWED_TASKS,
    ALLOWED_SPLITS,
    PreparedRecord,
    PreflightRequirements,
    ValidationCampaign,
    ValidationCampaignError,
    load_validation_campaign,
)


DEFAULT_MANIFEST_DIR = REPO / "artifacts" / "validation_2026" / "manifests"
DEFAULT_RESULTS_DIR = REPO / "artifacts" / "validation_2026" / "standardized"
DEFAULT_OUT_DIR = REPO / "artifacts" / "validation_2026" / "evaluation"
DEFAULT_PROVIDER = "mistral"
DEFAULT_MODEL = "mistral-small-2603"
TRAINING_TASKS = ("binary", "multiclass")
SPLITS = ("train", "val", "test")
MIN_RECOMMENDED_CLASS_SUPPORT = 2

JORGE_EDGE_BASELINES: Mapping[str, float] = {
    "Jorge RandomForest": 0.4554,
    "Jorge XGBoost": 0.4987,
    "Jorge LightGBM": 0.4942,
    "Jorge MLP": 0.1774,
    "Replica estricta XGBoost": 0.4993,
    "DeepSeek fine-tuned": 0.7479,
}

BenchmarkRunner = Callable[..., Any]


class EvaluationError(RuntimeError):
    """Error controlado de configuracion, reproducibilidad o evaluacion."""


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    manifest_dir: Path = DEFAULT_MANIFEST_DIR
    results_dir: Path = DEFAULT_RESULTS_DIR
    out_dir: Path = DEFAULT_OUT_DIR
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    datasets: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()
    models: tuple[str, ...] = SUPPORTED_MODELS
    seed: int = 42
    validate_only: bool = False
    allow_incomplete: bool = False
    include_fallbacks: bool = False
    mlp_backend: str = "external"
    torch_python: Path | None = None
    train_production_models: bool = True

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.model.strip():
            raise ValueError("provider y model no pueden estar vacios")
        if not isinstance(self.seed, int):
            raise TypeError("seed debe ser entero")
        unknown_tasks = sorted(set(self.tasks).difference(ALLOWED_TASKS))
        if unknown_tasks:
            raise ValueError(f"Tareas desconocidas: {unknown_tasks}")
        unknown_models = sorted(set(self.models).difference(SUPPORTED_MODELS))
        if unknown_models:
            raise ValueError(f"Modelos desconocidos: {unknown_models}")
        if not self.models:
            raise ValueError("Se requiere al menos un modelo")
        for label, values in (
            ("datasets", self.datasets),
            ("tasks", self.tasks),
            ("models", self.models),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} no admite duplicados")
        if self.mlp_backend not in {"external", "local", "injected"}:
            raise ValueError("mlp_backend debe ser 'external', 'local' o 'injected'")

    @property
    def requirements(self) -> PreflightRequirements:
        return PreflightRequirements(
            require_complete=not self.allow_incomplete,
            require_llm=not self.include_fallbacks,
            provider=self.provider,
            model=self.model,
            dedup_scope="dataset",
        )

    @property
    def policy_non_comparable_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.allow_incomplete:
            reasons.append("allow_incomplete")
        if self.include_fallbacks:
            reasons.append("include_fallbacks")
        if set(self.models) != set(SUPPORTED_MODELS):
            reasons.append("modelos_incompletos_respecto_al_protocolo_jorge")
        return tuple(reasons)

    @property
    def policy_comparable(self) -> bool:
        return not self.policy_non_comparable_reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_dir": str(self.manifest_dir.resolve()),
            "results_dir": str(self.results_dir.resolve()),
            "out_dir": str(self.out_dir.resolve()),
            "provider": self.provider,
            "model": self.model,
            "datasets": list(self.datasets),
            "tasks": list(self.tasks),
            "models": list(self.models),
            "seed": self.seed,
            "validate_only": self.validate_only,
            "allow_incomplete": self.allow_incomplete,
            "include_fallbacks": self.include_fallbacks,
            "mlp_backend": self.mlp_backend,
            "torch_python": (
                str(self.torch_python.expanduser().resolve())
                if self.torch_python is not None
                else None
            ),
            "train_production_models": self.train_production_models,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def discover_input_paths(config: EvaluationConfig) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    manifests = tuple(sorted(config.manifest_dir.glob("*_manifest.jsonl")))
    results = tuple(sorted(config.results_dir.glob("*_standardized.jsonl")))
    if not manifests:
        raise EvaluationError(f"No hay manifiestos en {config.manifest_dir}")
    return manifests, results


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_inputs(
    manifest_paths: Sequence[Path],
    result_paths: Sequence[Path],
) -> dict[str, Any]:
    """Genera un inventario de contenido, sin copiar datos ni credenciales."""

    def describe(path: Path) -> dict[str, Any]:
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "name": path.name,
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(path),
        }

    manifests = [describe(path) for path in manifest_paths]
    results = [describe(path) for path in result_paths]
    aggregate_payload = [
        [kind, item["name"], item["bytes"], item["sha256"]]
        for kind, items in (("manifest", manifests), ("result", results))
        for item in items
    ]
    aggregate = hashlib.sha256(
        json.dumps(aggregate_payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "aggregate_sha256": aggregate,
        "manifest_files": manifests,
        "result_files": results,
        "manifest_bytes": sum(item["bytes"] for item in manifests),
        "result_bytes": sum(item["bytes"] for item in results),
    }


def _snapshot_content_signature(snapshot: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        (kind, item["path"], item["bytes"], item["sha256"])
        for kind, key in (("manifest", "manifest_files"), ("result", "result_files"))
        for item in snapshot[key]
    )


def assert_inputs_unchanged(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    if _snapshot_content_signature(before) != _snapshot_content_signature(after):
        raise EvaluationError(
            "Los manifiestos/resultados cambiaron mientras se cargaban; "
            "no se puede congelar un artefacto reproducible"
        )


def _label_identity(label: Any) -> tuple[str, str]:
    return type(label).__name__, json.dumps(label, ensure_ascii=False, sort_keys=True)


def _label_text(label: Any) -> str:
    if isinstance(label, bool):
        return "attack" if label else "normal"
    return str(label)


def deduplicate_task_records(
    records: Iterable[PreparedRecord],
    *,
    task: str,
) -> tuple[tuple[PreparedRecord, ...], dict[str, Any]]:
    """Aplica dedup por dataset+tarea+features usando labels autoritativos.

    Cualquier huella presente en mas de un split o con labels incompatibles se
    elimina por completo. Solo los duplicados del mismo split y label conservan
    una fila, siempre el menor ``manifest_id``.
    """
    if task not in TRAINING_TASKS:
        raise ValueError("La deduplicacion supervisada solo admite binary/multiclass")
    eligible = tuple(record for record in records if task in record.tasks)
    groups: dict[tuple[str, str, str], list[PreparedRecord]] = defaultdict(list)
    for record in eligible:
        groups[(record.dataset, task, record.feature_fingerprint)].append(record)

    kept: list[PreparedRecord] = []
    removed_groups: list[dict[str, Any]] = []
    same_split_groups = 0
    cross_split_groups = 0
    conflicting_label_groups = 0
    rows_removed_same = 0
    rows_removed_unsafe = 0

    for (dataset, _, fingerprint), values in sorted(groups.items()):
        group = sorted(values, key=lambda item: item.manifest_id)
        splits = sorted({item.split for item in group})
        labels = {_label_identity(item.target_for(task)) for item in group}
        reasons: list[str] = []
        if len(splits) > 1:
            reasons.append("cross_split")
            cross_split_groups += 1
        if len(labels) > 1:
            reasons.append("conflicting_labels")
            conflicting_label_groups += 1

        if reasons:
            rows_removed_unsafe += len(group)
            removed_groups.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "feature_fingerprint": fingerprint,
                    "reasons": reasons,
                    "manifest_ids": [item.manifest_id for item in group],
                    "splits": splits,
                    "labels": sorted({_label_text(item.target_for(task)) for item in group}),
                    "kept_manifest_id": None,
                }
            )
            continue

        kept.append(group[0])
        if len(group) > 1:
            same_split_groups += 1
            rows_removed_same += len(group) - 1
            removed_groups.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "feature_fingerprint": fingerprint,
                    "reasons": ["same_split_same_label_duplicate"],
                    "manifest_ids": [item.manifest_id for item in group],
                    "splits": splits,
                    "labels": [_label_text(group[0].target_for(task))],
                    "kept_manifest_id": group[0].manifest_id,
                }
            )

    split_fingerprints: dict[str, set[tuple[str, str]]] = {
        split: {
            (record.dataset, record.feature_fingerprint)
            for record in kept
            if record.split == split
        }
        for split in SPLITS
    }
    overlaps = {
        f"{left}_{right}": sorted(
            fingerprint for _, fingerprint in split_fingerprints[left] & split_fingerprints[right]
        )
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    final_overlap_count = sum(len(values) for values in overlaps.values())
    if final_overlap_count:
        raise EvaluationError(
            f"Invariante rota: quedan {final_overlap_count} huellas entre splits"
        )

    kept.sort(key=lambda item: (item.dataset, SPLITS.index(item.split), item.manifest_id))
    report = {
        "scope": "dataset+task+feature_fingerprint",
        "task": task,
        "input_rows": len(eligible),
        "output_rows": len(kept),
        "removed_rows": len(eligible) - len(kept),
        "same_split_same_label_groups": same_split_groups,
        "same_split_rows_removed": rows_removed_same,
        "cross_split_groups": cross_split_groups,
        "conflicting_label_groups": conflicting_label_groups,
        "unsafe_rows_removed": rows_removed_unsafe,
        "removed_groups": removed_groups,
        "final_split_overlap": overlaps,
        "final_overlap_count": final_overlap_count,
    }
    return tuple(kept), report


def split_class_supports(
    records: Iterable[PreparedRecord],
    *,
    task: str,
) -> dict[str, Any]:
    values = tuple(record for record in records if task in record.tasks)
    split_reports: dict[str, Any] = {}
    total_classes: Counter[str] = Counter()
    for split in SPLITS:
        subset = [record for record in values if record.split == split]
        classes = Counter(_label_text(record.target_for(task)) for record in subset)
        total_classes.update(classes)
        split_reports[split] = {
            "rows": len(subset),
            "classes": dict(sorted(classes.items())),
        }
    return {
        "rows": len(values),
        "classes": dict(sorted(total_classes.items())),
        "splits": split_reports,
    }


def assess_task_support(supports: Mapping[str, Any]) -> dict[str, Any]:
    classes = sorted(supports["classes"])
    rare: dict[str, Any] = {}
    for label in classes:
        counts = {
            split: int(supports["splits"][split]["classes"].get(label, 0))
            for split in SPLITS
        }
        low_splits = [
            split for split, count in counts.items() if count < MIN_RECOMMENDED_CLASS_SUPPORT
        ]
        if low_splits:
            rare[label] = {"supports": counts, "low_support_splits": low_splits}

    blocking: list[str] = []
    for split in SPLITS:
        if not supports["splits"][split]["rows"]:
            blocking.append(f"split_{split}_vacio")
    train_classes = set(supports["splits"]["train"]["classes"])
    if len(train_classes) < 2:
        blocking.append("train_tiene_menos_de_dos_clases")
    for split in ("val", "test"):
        unseen = sorted(set(supports["splits"][split]["classes"]) - train_classes)
        if unseen:
            blocking.append(f"clases_de_{split}_ausentes_en_train:{','.join(unseen)}")

    degraded_reasons = list(blocking)
    if rare:
        degraded_reasons.append("clases_con_soporte_raro")
    return {
        "trainable": not blocking,
        "degraded": bool(degraded_reasons),
        "blocking_reasons": blocking,
        "degraded_reasons": degraded_reasons,
        "rare_class_threshold_per_split": MIN_RECOMMENDED_CLASS_SUPPORT,
        "rare_classes": rare,
    }


def _public_benchmark_report(outcome: Any) -> dict[str, Any]:
    report = copy.deepcopy(outcome.report)
    selection = report.get("selection")
    winner = None if not selection else selection.get("model")
    for model_name, model_report in report.get("models", {}).items():
        if model_name != winner:
            model_report.pop("test", None)
            model_report.pop("test_error", None)
            model_report["test_withheld"] = True
    report["test_reporting_policy"] = (
        "test se publica solo para el ganador preseleccionado con validacion"
    )
    close = getattr(outcome, "close", None)
    if callable(close):
        close()
    else:
        estimators = getattr(outcome, "estimators", None)
        if isinstance(estimators, dict):
            estimators.clear()
    return report


def edge_jorge_comparison(test_f1_weighted: float) -> dict[str, Any]:
    value = float(test_f1_weighted)
    comparisons = [
        {
            "baseline": name,
            "f1_weighted": baseline,
            "delta": value - baseline,
            "surpassed": value > baseline,
        }
        for name, baseline in JORGE_EDGE_BASELINES.items()
    ]
    if value > JORGE_EDGE_BASELINES["DeepSeek fine-tuned"]:
        verdict = "exito_pleno"
    elif value > JORGE_EDGE_BASELINES["Jorge XGBoost"]:
        verdict = "exito_parcial"
    else:
        verdict = "no_supera_jorge_xgboost"
    return {
        "metric": "test.f1_weighted",
        "candidate": value,
        "verdict": verdict,
        "baselines": comparisons,
    }


def _is_edge_dataset(dataset: str) -> bool:
    return re.sub(r"[^a-z0-9]", "", dataset.casefold()).startswith("edgeiiot")


def _split_payload(
    records: Sequence[PreparedRecord], task: str
) -> tuple[list[Mapping[str, Any]], list[Any], list[Mapping[str, Any]], list[Any], list[Mapping[str, Any]], list[Any]]:
    payload: list[Any] = []
    for split in SPLITS:
        subset = [record for record in records if record.split == split]
        payload.extend(
            ([dict(record.features) for record in subset], [record.target_for(task) for record in subset])
        )
    return tuple(payload)  # type: ignore[return-value]


def evaluate_task(
    records: Iterable[PreparedRecord],
    *,
    dataset: str,
    task: str,
    models: Sequence[str] = SUPPORTED_MODELS,
    seed: int = 42,
    validate_only: bool = False,
    policy_comparable: bool = True,
    policy_non_comparable_reasons: Sequence[str] = (),
    benchmark_runner: BenchmarkRunner = run_jorge_benchmarks,
    mlp_backend_factory: MLPBackendFactory | None = None,
) -> dict[str, Any]:
    dataset_records = tuple(record for record in records if record.dataset == dataset)
    deduplicated, dedup_report = deduplicate_task_records(dataset_records, task=task)
    supports = split_class_supports(deduplicated, task=task)
    assessment = assess_task_support(supports)
    reasons = list(policy_non_comparable_reasons)
    reasons.extend(assessment["degraded_reasons"])
    result: dict[str, Any] = {
        "task": task,
        "status": "validated_only" if validate_only else "pending",
        "comparable": policy_comparable and not assessment["degraded"],
        "non_comparable_reasons": list(dict.fromkeys(reasons)),
        "deduplication": dedup_report,
        "supports": supports,
        "support_assessment": assessment,
        "benchmark": None,
    }
    if validate_only:
        return result
    if not assessment["trainable"]:
        result["status"] = "skipped_untrainable"
        result["comparable"] = False
        return result

    try:
        split_payload = _split_payload(deduplicated, task)
        outcome = benchmark_runner(
            *split_payload,
            task=task,
            models=tuple(models),
            seed=seed,
            positive_label=True if task == "binary" else None,
            mlp_backend_factory=mlp_backend_factory,
            require_all_models=True,
        )
        benchmark = _public_benchmark_report(outcome)
    except BenchmarkRequirementsNotMet as exc:
        benchmark = _public_benchmark_report(exc.outcome)
        result.update(
            status="failed",
            comparable=False,
            benchmark=benchmark,
        )
        result["non_comparable_reasons"].append(
            "benchmark_incompleto:" + ";".join(exc.failures)
        )
        return result
    except Exception as exc:
        result.update(
            status="failed",
            comparable=False,
            benchmark={"status": "failed", "message": f"{type(exc).__name__}: {exc}"},
        )
        result["non_comparable_reasons"].append("benchmark_failed")
        return result

    result["benchmark"] = benchmark
    selection = benchmark.get("selection")
    winner = None if not selection else selection.get("model")
    model_reports = benchmark.get("models", {})
    failed_models = [
        name for name in models if model_reports.get(name, {}).get("status") != "ok"
    ]
    winner_test = model_reports.get(winner, {}).get("test") if winner else None
    if winner is None or winner_test is None:
        result["status"] = "failed"
        result["comparable"] = False
        result["non_comparable_reasons"].append("sin_ganador_con_test")
    elif failed_models:
        result["status"] = "degraded"
        result["comparable"] = False
        result["non_comparable_reasons"].append(
            "modelos_no_evaluados:" + ",".join(failed_models)
        )
    elif assessment["degraded"] or not policy_comparable:
        result["status"] = "degraded"
    else:
        result["status"] = "ok"

    result["non_comparable_reasons"] = list(
        dict.fromkeys(result["non_comparable_reasons"])
    )
    if _is_edge_dataset(dataset) and task == "multiclass" and winner_test is not None:
        result["jorge_comparison"] = edge_jorge_comparison(winner_test["f1_weighted"])
    return result


def manifest_supports(campaign: ValidationCampaign, dataset: str) -> dict[str, Any]:
    rows = [record for record in campaign.manifests.records if record.dataset == dataset]
    splits = Counter(record.split for record in rows)
    tasks = Counter(task for record in rows for task in record.tasks)
    classes: dict[str, Any] = {}
    for task in TRAINING_TASKS:
        eligible = [record for record in rows if task in record.tasks]
        if eligible:
            by_split = {
                split: dict(
                    sorted(
                        Counter(
                            _label_text(record.target_for(task))
                            for record in eligible
                            if record.split == split
                        ).items()
                    )
                )
                for split in SPLITS
            }
            classes[task] = by_split
    return {
        "rows": len(rows),
        "splits": {split: splits.get(split, 0) for split in SPLITS},
        "tasks": dict(sorted(tasks.items())),
        "classes": classes,
    }


def environment_report() -> dict[str, Any]:
    packages: dict[str, Any] = {}
    for distribution in (
        "numpy",
        "scikit-learn",
        "joblib",
        "xgboost",
        "lightgbm",
        "torch",
    ):
        try:
            packages[distribution] = {"available": True, "version": metadata.version(distribution)}
        except metadata.PackageNotFoundError:
            packages[distribution] = {"available": False, "version": None}
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": str(Path(sys.executable).resolve()),
        "packages": packages,
    }


def _is_urban_dataset(dataset: str) -> bool:
    return "urban" in re.sub(r"[^a-z0-9]", "", dataset.casefold())


def build_evaluation_report(
    campaign: ValidationCampaign,
    config: EvaluationConfig,
    *,
    input_snapshot: Mapping[str, Any],
    created_at: str | None = None,
    benchmark_runner: BenchmarkRunner = run_jorge_benchmarks,
    mlp_backend_factory: MLPBackendFactory | None = None,
) -> dict[str, Any]:
    available_datasets = sorted({record.dataset for record in campaign.manifests.records})
    unknown = sorted(set(config.datasets).difference(available_datasets))
    if unknown:
        raise EvaluationError(f"Datasets no presentes en manifiestos: {unknown}")
    selected_datasets = list(config.datasets) if config.datasets else available_datasets
    requested_tasks = set(config.tasks) if config.tasks else set(ALLOWED_TASKS)

    dataset_reports: dict[str, Any] = {}
    for dataset in selected_datasets:
        quality = campaign.quality.by_dataset[dataset]
        available_tasks = {
            task
            for record in campaign.manifests.records
            if record.dataset == dataset
            for task in record.tasks
        }
        training_tasks = [
            task for task in TRAINING_TASKS if task in requested_tasks and task in available_tasks
        ]
        forced_standardization = _is_urban_dataset(dataset)
        if forced_standardization:
            training_tasks = []

        task_reports = {
            task: evaluate_task(
                campaign.joined_records,
                dataset=dataset,
                task=task,
                models=config.models,
                seed=config.seed,
                validate_only=config.validate_only,
                policy_comparable=config.policy_comparable,
                policy_non_comparable_reasons=config.policy_non_comparable_reasons,
                benchmark_runner=benchmark_runner,
                mlp_backend_factory=mlp_backend_factory,
            )
            for task in training_tasks
        }
        if not training_tasks:
            status = "standardization_only"
            comparable: bool | None = False if not config.policy_comparable else None
            reasons = (
                list(config.policy_non_comparable_reasons)
                if not config.policy_comparable
                else ["sin_tarea_supervisada_solicitada"]
            )
        else:
            statuses = {report["status"] for report in task_reports.values()}
            status = (
                "validated_only"
                if config.validate_only and statuses == {"validated_only"}
                else "complete"
                if statuses == {"ok"}
                else "degraded"
            )
            comparable = all(report["comparable"] for report in task_reports.values())
            reasons = list(
                dict.fromkeys(
                    reason
                    for report in task_reports.values()
                    for reason in report["non_comparable_reasons"]
                )
            )

        dataset_reports[dataset] = {
            "dataset": dataset,
            "status": status,
            "comparable": comparable,
            "non_comparable_reasons": reasons,
            "available_tasks": sorted(available_tasks),
            "requested_tasks": sorted(requested_tasks),
            "standardization_only": not training_tasks,
            "standardization_quality": asdict(quality),
            "manifest_supports": manifest_supports(campaign, dataset),
            "tasks": task_reports,
        }

    comparable_reports = [
        report["comparable"]
        for report in dataset_reports.values()
        if report["comparable"] is not None
    ]
    global_comparable = config.policy_comparable and all(comparable_reports)
    global_reasons = list(config.policy_non_comparable_reasons)
    global_reasons.extend(
        reason
        for report in dataset_reports.values()
        if report["comparable"] is False
        for reason in report["non_comparable_reasons"]
    )
    statuses = {report["status"] for report in dataset_reports.values()}
    global_status = (
        "validated_only"
        if config.validate_only and statuses.issubset({"validated_only", "standardization_only"})
        else "complete"
        if statuses.issubset({"complete", "standardization_only"})
        else "degraded"
    )
    return {
        "schema_version": "validation-evaluation-v1",
        "created_at": created_at or utc_now(),
        "status": global_status,
        "comparable": global_comparable,
        "non_comparable_reasons": list(dict.fromkeys(global_reasons)),
        "configuration": config.to_dict(),
        "input_snapshot": dict(input_snapshot),
        "environment": environment_report(),
        "campaign": {
            "quality": campaign.quality.to_dict(),
            "preflight_deduplication": campaign.deduplication.to_dict(),
            "joined_rows": len(campaign.joined_records),
            "preflight_ready_rows": len(campaign.records),
            "task_specific_dedup_is_authoritative": True,
        },
        "datasets": dataset_reports,
    }


def run_evaluation(
    config: EvaluationConfig,
    *,
    benchmark_runner: BenchmarkRunner = run_jorge_benchmarks,
    mlp_backend_factory: MLPBackendFactory | None = None,
) -> dict[str, Any]:
    needs_mlp = not config.validate_only and "mlp" in config.models
    if (
        needs_mlp
        and config.mlp_backend in {"external", "injected"}
        and mlp_backend_factory is None
    ):
        raise EvaluationError(
            f"--mlp-backend {config.mlp_backend} requiere una factory disponible"
        )
    if config.mlp_backend == "local":
        mlp_backend_factory = None

    manifest_paths, result_paths = discover_input_paths(config)
    before = snapshot_inputs(manifest_paths, result_paths)
    campaign = load_validation_campaign(
        manifest_paths,
        result_paths,
        config.requirements,
    )
    current_manifest_paths, current_result_paths = discover_input_paths(config)
    after = snapshot_inputs(current_manifest_paths, current_result_paths)
    assert_inputs_unchanged(before, after)
    report = build_evaluation_report(
        campaign,
        config,
        input_snapshot=after,
        benchmark_runner=benchmark_runner,
        mlp_backend_factory=mlp_backend_factory,
    )
    if config.validate_only or not config.train_production_models:
        report["production_models"] = {
            "status": "skipped",
            "reason": "validate_only" if config.validate_only else "disabled_by_cli",
        }
        return report
    if config.datasets or config.tasks:
        report["production_models"] = {
            "status": "skipped",
            "reason": "dataset_or_task_filters_active",
        }
        return report

    model_dir = config.out_dir / "candidate_models"
    detector_path = model_dir / "xgboost_detection_validation_candidate.joblib"
    family_path = model_dir / "xgboost_attack_family_validation_candidate.joblib"
    try:
        candidates = train_production_candidates(
            campaign.joined_records,
            detector_path=detector_path,
            family_path=family_path,
        )
        report["production_models"] = {
            "status": "ok",
            **candidates.report(),
        }
    except Exception as exc:
        report["production_models"] = {
            "status": "failed",
            "message": f"{type(exc).__name__}: {exc}",
        }
        report["status"] = "degraded"
        report["comparable"] = False
        report["non_comparable_reasons"].append("production_model_training_failed")
    return report


def protocol_failures(report: Mapping[str, Any]) -> list[str]:
    """Devuelve fallos que impiden declarar terminada la Fase D."""
    failures: list[str] = []
    for dataset_name, dataset in report.get("datasets", {}).items():
        for task_name, task in dataset.get("tasks", {}).items():
            if task.get("status") in {"failed", "skipped_untrainable"}:
                failures.append(f"{dataset_name}/{task_name}:{task.get('status')}")
    production = report.get("production_models") or {}
    if production.get("status") == "failed":
        failures.append("production_models:failed")
    return failures


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_dataset_markdown(document: Mapping[str, Any]) -> str:
    dataset = document["dataset"]
    lines = [
        f"# Evaluacion: {dataset['dataset']}",
        "",
        f"- Estado: `{dataset['status']}`",
        f"- Comparable: `{dataset['comparable']}`",
        f"- Filas de manifiesto: {dataset['manifest_supports']['rows']}",
        f"- Cobertura seleccionada: {dataset['standardization_quality']['selected_coverage_rate']:.6f}",
        f"- Exitos LLM: {dataset['standardization_quality']['llm_success_rate']:.6f}",
        f"- Fallback: {dataset['standardization_quality']['fallback_rate']:.6f}",
        "",
    ]
    if dataset["standardization_only"]:
        lines.extend(["Este dataset se reporta como `standardization_only`; no se entrena.", ""])
    for task_name, task in dataset["tasks"].items():
        selection = (task.get("benchmark") or {}).get("selection") or {}
        winner = selection.get("model", "-")
        winner_report = (task.get("benchmark") or {}).get("models", {}).get(winner, {})
        test_f1 = (winner_report.get("test") or {}).get("f1_weighted", "-")
        lines.extend(
            [
                f"## {task_name}",
                "",
                f"Estado: `{task['status']}` · comparable: `{task['comparable']}` · "
                f"ganador: `{winner}` · F1 test: `{test_f1}`.",
                "",
                "| Split | Filas | Clases |",
                "|---|---:|---|",
            ]
        )
        for split in SPLITS:
            support = task["supports"]["splits"][split]
            classes = ", ".join(f"{key}: {value}" for key, value in support["classes"].items())
            lines.append(f"| {split} | {support['rows']} | {_md(classes)} |")
        lines.extend(
            [
                "",
                f"Dedup: {task['deduplication']['input_rows']} -> "
                f"{task['deduplication']['output_rows']} filas; overlap final = "
                f"{task['deduplication']['final_overlap_count']}.",
                "",
            ]
        )
        comparison = task.get("jorge_comparison")
        if comparison:
            lines.extend(
                [
                    f"Veredicto Edge: `{comparison['verdict']}`.",
                    "",
                    "| Referencia | F1 | Delta | Superada |",
                    "|---|---:|---:|---|",
                ]
            )
            for baseline in comparison["baselines"]:
                lines.append(
                    f"| {_md(baseline['baseline'])} | {baseline['f1_weighted']:.4f} | "
                    f"{baseline['delta']:+.4f} | {baseline['surpassed']} |"
                )
            lines.append("")
    production = document.get("production_models") or {}
    if production:
        lines.extend(["## Modelos candidatos del sistema", ""])
        for model_name, values in production.items():
            test = values.get("test") or {}
            lines.append(
                f"- `{model_name}`: n={test.get('n', '-')} · "
                f"F1 weighted={test.get('f1_weighted', '-')}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_global_markdown(report: Mapping[str, Any]) -> str:
    quality = report["campaign"]["quality"]["global_metrics"]
    lines = [
        "# Validacion global por dataset",
        "",
        f"- Generado: `{report['created_at']}`",
        f"- Estado: `{report['status']}`",
        f"- Comparable: `{report['comparable']}`",
        f"- Snapshot SHA-256: `{report['input_snapshot']['aggregate_sha256']}`",
        f"- Cobertura seleccionada: {quality['selected_rows']}/{quality['manifest_rows']}",
        "",
        "| Dataset | Estado | Comparable | Filas | Tareas | Ganadores |",
        "|---|---|---|---:|---|---|",
    ]
    for dataset_name, dataset in report["datasets"].items():
        winners = []
        for task_name, task in dataset["tasks"].items():
            selection = (task.get("benchmark") or {}).get("selection") or {}
            winners.append(f"{task_name}: {selection.get('model', '-')}")
        lines.append(
            f"| {_md(dataset_name)} | {dataset['status']} | {dataset['comparable']} | "
            f"{dataset['manifest_supports']['rows']} | "
            f"{_md(', '.join(dataset['available_tasks']))} | {_md(', '.join(winners) or '-')} |"
        )
    production = report.get("production_models") or {}
    lines.extend(["", "## Modelos candidatos del sistema", ""])
    lines.append(f"Estado: `{production.get('status', 'not_run')}`.")
    if production.get("status") == "ok":
        lines.extend(
            [
                "",
                "| Modelo | Artefacto | F1 test global |",
                "|---|---|---:|",
            ]
        )
        for key in ("detector", "family_classifier"):
            candidate = production[key]
            artifact = (candidate.get("artifact") or {}).get("path", "-")
            f1 = candidate["metrics"]["test"]["global"].get("f1_weighted", "-")
            lines.append(f"| {key} | {_md(artifact)} | {f1} |")
    lines.extend(["", "## Entorno", "", "| Paquete | Disponible | Version |", "|---|---|---|"])
    for name, package in report["environment"]["packages"].items():
        lines.append(f"| {name} | {package['available']} | {package['version'] or '-'} |")
    lines.extend(["", "## Entradas congeladas", "", "| Tipo | Fichero | Bytes | SHA-256 |", "|---|---|---:|---|"])
    for kind, key in (("manifest", "manifest_files"), ("result", "result_files")):
        for item in report["input_snapshot"][key]:
            lines.append(f"| {kind} | {_md(item['name'])} | {item['bytes']} | `{item['sha256']}` |")
    return "\n".join(lines).rstrip() + "\n"


def _artifact_slug(dataset: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "_", dataset.casefold()).strip("._")
    if not slug:
        raise EvaluationError(f"Nombre de dataset no apto para artefacto: {dataset!r}")
    return slug


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_artifacts(report: Mapping[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    global_json = out_dir / "evaluation_global.json"
    global_md = out_dir / "evaluation_global.md"
    _write_atomic(global_json, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    _write_atomic(global_md, render_global_markdown(report))
    paths.update(global_json=str(global_json), global_markdown=str(global_md))

    seen_slugs: set[str] = set()
    for dataset_name, dataset in report["datasets"].items():
        slug = _artifact_slug(dataset_name)
        if slug in seen_slugs:
            raise EvaluationError(f"Colision de nombres de artefacto para {dataset_name!r}")
        seen_slugs.add(slug)
        document = {
            "schema_version": report["schema_version"],
            "created_at": report["created_at"],
            "configuration": report["configuration"],
            "input_snapshot": report["input_snapshot"],
            "environment": report["environment"],
            "dataset": dataset,
            "production_models": _production_dataset_slice(report, dataset_name),
        }
        json_path = out_dir / f"{slug}_evaluation.json"
        md_path = out_dir / f"{slug}_evaluation.md"
        _write_atomic(json_path, json.dumps(document, ensure_ascii=False, indent=2) + "\n")
        _write_atomic(md_path, render_dataset_markdown(document))
        paths[f"{dataset_name}.json"] = str(json_path)
        paths[f"{dataset_name}.markdown"] = str(md_path)
    return paths


def _production_dataset_slice(
    report: Mapping[str, Any], dataset: str
) -> dict[str, Any]:
    production = report.get("production_models") or {}
    if production.get("status") != "ok":
        return {}
    sliced: dict[str, Any] = {}
    for key in ("detector", "family_classifier"):
        candidate = production.get(key) or {}
        sliced[key] = {
            "val": (candidate.get("metrics", {}).get("val", {}).get("by_dataset", {}).get(dataset)),
            "test": (candidate.get("metrics", {}).get("test", {}).get("by_dataset", {}).get(dataset)),
        }
    return sliced


def _csv_models(value: str) -> tuple[str, ...]:
    models = tuple(part.strip() for part in value.split(",") if part.strip())
    if not models:
        raise argparse.ArgumentTypeError("la lista CSV de modelos esta vacia")
    unknown = sorted(set(models).difference(SUPPORTED_MODELS))
    if unknown:
        raise argparse.ArgumentTypeError(f"modelos no soportados: {unknown}")
    if len(models) != len(set(models)):
        raise argparse.ArgumentTypeError("la lista de modelos contiene duplicados")
    return models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--task", action="append", choices=sorted(ALLOWED_TASKS), default=[])
    parser.add_argument("--models", type=_csv_models, default=SUPPORTED_MODELS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--include-fallbacks", action="store_true")
    parser.add_argument(
        "--mlp-backend",
        choices=("external", "local", "injected"),
        default="external",
        help="external reutiliza el Python con PyTorch; local lo importa en este entorno",
    )
    parser.add_argument(
        "--torch-python",
        type=Path,
        default=None,
        help="interprete del worker PyTorch; por defecto autodetecta Anaconda",
    )
    parser.add_argument(
        "--skip-production-training",
        action="store_true",
        help="no entrenar los candidatos globales de detector/clasificador",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    benchmark_runner: BenchmarkRunner = run_jorge_benchmarks,
    mlp_backend_factory: MLPBackendFactory | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = EvaluationConfig(
            manifest_dir=args.manifest_dir,
            results_dir=args.results_dir,
            out_dir=args.out_dir,
            provider=args.provider,
            model=args.model,
            datasets=tuple(args.dataset),
            tasks=tuple(args.task),
            models=tuple(args.models),
            seed=args.seed,
            validate_only=args.validate_only,
            allow_incomplete=args.allow_incomplete,
            include_fallbacks=args.include_fallbacks,
            mlp_backend=args.mlp_backend,
            torch_python=args.torch_python,
            train_production_models=not args.skip_production_training,
        )
        owned_factory = None
        if (
            config.mlp_backend == "external"
            and not config.validate_only
            and "mlp" in config.models
            and mlp_backend_factory is None
        ):
            owned_factory = ExternalTorchMLPFactory(
                interpreter=config.torch_python or detect_torch_interpreter()
            )
            mlp_backend_factory = owned_factory
        try:
            report = run_evaluation(
                config,
                benchmark_runner=benchmark_runner,
                mlp_backend_factory=mlp_backend_factory,
            )
        finally:
            if owned_factory is not None:
                owned_factory.close()
        paths = write_artifacts(report, config.out_dir)
    except (EvaluationError, ValidationCampaignError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    failures = protocol_failures(report)
    print(
        f"Evaluacion {report['status']}: {len(report['datasets'])} datasets; "
        f"comparable={report['comparable']}; JSON={paths['global_json']}",
        flush=True,
    )
    if failures:
        print("ERROR: Fase D incompleta: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
