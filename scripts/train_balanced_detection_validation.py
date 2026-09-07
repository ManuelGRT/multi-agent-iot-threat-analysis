"""Retrain only the production detector on the frozen Mistral campaign.

The input manifests and successful Mistral standardisations remain immutable.
After feature-level deduplication, the training function deterministically
balances normal and attack observations inside every split and logical origin,
including the four TON-IoT modalities.  The 16-type attack classifier is never
loaded, trained or overwritten by this command.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.eval.production_models import train_global_detector  # noqa: E402
from src.eval.validation_campaign import (  # noqa: E402
    PreflightRequirements,
    load_validation_campaign,
)


DEFAULT_SOURCE_ROOT = REPO / "artifacts" / "validation_2026"
DEFAULT_OUT_DIR = REPO / "artifacts" / "validation_2026_balanced_detection"
DEPLOYMENT_FILENAME = "xgboost_detection_balanced_by_origin_20260905.joblib"
DEFAULT_MODEL_OUT = DEFAULT_OUT_DIR / DEPLOYMENT_FILENAME
DEFAULT_DEPLOYMENT_METADATA_OUT = DEFAULT_OUT_DIR / (
    "xgboost_detection_balanced_by_origin_20260905.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=DEFAULT_SOURCE_ROOT / "manifests",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_SOURCE_ROOT / "standardized",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    parser.add_argument(
        "--deployment-metadata-out",
        type=Path,
        default=DEFAULT_DEPLOYMENT_METADATA_OUT,
        help="Ficha portable que acompana al candidato para su revision.",
    )
    parser.add_argument("--provider", default="mistral")
    parser.add_argument("--model", default="mistral-small-2603")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def input_snapshot(manifests: Sequence[Path], results: Sequence[Path]) -> dict[str, Any]:
    def describe(path: Path) -> dict[str, Any]:
        return {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    manifest_rows = {path.name: describe(path) for path in manifests}
    result_rows = {path.name: describe(path) for path in results}
    # Las rutas absolutas se conservan como contexto, pero no intervienen en
    # la huella: el mismo corpus copiado a otro equipo debe producir el mismo
    # identificador reproducible.
    core = {"manifests": manifest_rows, "results": result_rows}
    portable_core = {
        group: {
            name: {key: metadata[key] for key in ("bytes", "sha256")}
            for name, metadata in files.items()
        }
        for group, files in core.items()
    }
    aggregate = hashlib.sha256(
        json.dumps(portable_core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**core, "aggregate_sha256": aggregate}


def deployment_metadata(
    report: Mapping[str, Any],
    *,
    artifact_filename: str,
    booster_raw_sha256: str,
) -> dict[str, Any]:
    """Reduce el informe completo a la ficha portable usada por la auditoria."""

    detector = report["detector"]
    inputs = report["inputs"]
    balancing = detector["balancing"]
    supports = detector["supports"]
    metrics = detector["metrics"]
    snapshot = inputs["snapshot"]

    def file_hashes(group: str) -> dict[str, str]:
        return {
            name: str(metadata["sha256"])
            for name, metadata in snapshot[group].items()
        }

    def split_metrics(split: str) -> dict[str, Any]:
        binary = metrics[split]["global"]["binary"]
        return {
            "rows": supports[split]["rows"],
            "accuracy": binary["accuracy"],
            "attack_precision": binary["precision"],
            "attack_recall": binary["recall"],
            "attack_f1": binary["f1"],
            "confusion_matrix": metrics[split]["global"]["confusion_matrix"][
                "matrix"
            ],
        }

    per_origin_per_class: dict[str, Any] = {}
    test_by_origin: dict[str, Any] = {}
    for origin, origin_support in balancing["after"]["by_origin"].items():
        split_support = origin_support["by_split"]
        per_origin_per_class[origin] = {
            "train": split_support["train"]["class_counts"]["normal"],
            "validation": split_support["val"]["class_counts"]["normal"],
            "test": split_support["test"]["class_counts"]["normal"],
            "total": origin_support["class_counts"]["normal"],
        }
        origin_metrics = metrics["test"]["by_origin"][origin]
        operational = metrics["test"]["operational_by_origin"][origin]
        decided = operational["decided_metrics"]
        test_by_origin[origin] = {
            "rows": split_support["test"]["rows"],
            "attack_f1": origin_metrics["binary"]["f1"],
            "coverage": operational["coverage"],
            "attack_f1_decided": (
                decided["binary"]["f1"] if decided is not None else None
            ),
        }

    operational_test = metrics["test"]["operational"]
    decided_test = operational_test["decided_metrics"]
    parameters = detector["parameters"]
    return {
        "schema_version": "deployed-detector-evaluation-v1",
        "created_at": report["created_at"],
        "artifact": {
            "filename": artifact_filename,
            "bytes": detector["artifact"]["bytes"],
            "sha256": detector["artifact"]["sha256"],
            "task": detector["task"],
            "classes": detector["classes"],
        },
        "inputs": {
            "provider": inputs["provider"],
            "model": inputs["model"],
            "successful_standardizations": inputs["llm_success_rows"],
            "binary_eligible_rows": detector["input_rows"],
            "snapshot_sha256": snapshot["aggregate_sha256"],
            "files": {
                "manifests": file_hashes("manifests"),
                "standardized": file_hashes("results"),
            },
        },
        "deduplication": {
            "input_rows": detector["deduplication"]["input_rows"],
            "output_rows": detector["deduplication"]["output_rows"],
            "cross_split_rows_removed": detector["deduplication"][
                "unsafe_rows_removed"
            ],
            "final_cross_split_overlap_count": detector["deduplication"][
                "final_cross_split_overlap_count"
            ],
        },
        "balancing": {
            "policy_version": balancing["policy_version"],
            "seed": balancing["seed"],
            "grouping": balancing["grouping"],
            "selection": balancing["selection"],
            "selected_manifest_ids_sha256": balancing[
                "selected_manifest_ids_sha256"
            ],
            "rows": balancing["output_rows"],
            "removed_rows": balancing["removed_rows"],
            "class_counts": {
                "benign": balancing["after"]["global"]["class_counts"][
                    "normal"
                ],
                "attack": balancing["after"]["global"]["class_counts"][
                    "attack"
                ],
            },
            "splits": {
                "train": {
                    "rows": supports["train"]["rows"],
                    "per_class": balancing["after"]["by_split"]["train"][
                        "class_counts"
                    ]["normal"],
                },
                "validation": {
                    "rows": supports["val"]["rows"],
                    "per_class": balancing["after"]["by_split"]["val"][
                        "class_counts"
                    ]["normal"],
                },
                "test": {
                    "rows": supports["test"]["rows"],
                    "per_class": balancing["after"]["by_split"]["test"][
                        "class_counts"
                    ]["normal"],
                },
            },
            "per_origin_per_class": per_origin_per_class,
        },
        "model": {
            "algorithm": "XGBoost",
            "parameters": {
                key: parameters[key]
                for key in (
                    "n_estimators",
                    "max_depth",
                    "learning_rate",
                    "subsample",
                    "colsample_bytree",
                    "random_state",
                    "objective",
                )
            },
            "feature_count": detector["features"]["count"],
            "feature_schema_sha256": detector["features"]["names_sha256"],
            "booster_raw_sha256": booster_raw_sha256,
        },
        "metrics": {
            "validation": split_metrics("val"),
            "test": split_metrics("test"),
            "operational_test": {
                "gray_zone_inclusive": operational_test["gray_zone_inclusive"],
                "decided_rows": operational_test["decided_rows"],
                "abstained_rows": operational_test["abstained_rows"],
                "coverage": operational_test["coverage"],
                "attack_f1_decided": (
                    decided_test["binary"]["f1"]
                    if decided_test is not None
                    else None
                ),
            },
            "test_by_origin": test_by_origin,
        },
        "scope_notes": [
            "23604 is the complete balanced corpus; fit uses only the 16496 train rows.",
            "Balance is 1:1 inside each split and origin, not equal weight across origins.",
            "Urban-IoT is excluded because it does not declare a binary target.",
            "TON-IoT telemetry is the weakest evaluated origin and Windows has a small test support.",
            "The joblib SHA identifies the deployed wrapper; reproducibility across contract revisions is verified with selection, feature-schema and raw-booster hashes.",
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    detector = report["detector"]
    balancing = detector["balancing"]
    lines = [
        "# Detector XGBoost balanceado por origen",
        "",
        f"- Generado: `{report['created_at']}`",
        f"- Modelo de estandarizacion: `{report['inputs']['provider']}/{report['inputs']['model']}`",
        f"- Politica: `{balancing['policy_version']}`",
        f"- Filas elegibles: {detector['input_rows']}",
        f"- Filas tras deduplicar: {detector['deduplication']['output_rows']}",
        f"- Filas balanceadas: {balancing['output_rows']}",
        f"- Modelo: `{detector['artifact']['path']}`",
        f"- SHA-256 modelo: `{detector['artifact']['sha256']}`",
        "",
        "## Soporte balanceado",
        "",
        "| Origen | Train por clase | Validacion por clase | Test por clase | Total por clase |",
        "|---|---:|---:|---:|---:|",
    ]
    for origin, values in balancing["after"]["by_origin"].items():
        lines.append(
            f"| {origin} | {values['by_split']['train']['class_counts']['normal']} | "
            f"{values['by_split']['val']['class_counts']['normal']} | "
            f"{values['by_split']['test']['class_counts']['normal']} | "
            f"{values['class_counts']['normal']} |"
        )

    lines.extend(
        [
            "",
            "## Metricas globales antes de abstencion",
            "",
        "| Split | n | Accuracy | Precision ponderada | Recall ponderado | F1 ponderada |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split in ("val", "test"):
        metrics = detector["metrics"][split]["global"]
        lines.append(
            f"| {split} | {detector['supports'][split]['rows']} | "
            f"{metrics['accuracy']:.6f} | "
            f"{metrics['precision_weighted']:.6f} | {metrics['recall_weighted']:.6f} | "
            f"{metrics['f1_weighted']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Politica operacional de abstencion",
            "",
            "Zona gris inclusiva: `[0.4, 0.6]`.",
            "",
            "| Split | Cobertura | Abstencion | Filas decididas | F1 ataque decidido |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split in ("val", "test"):
        operational = detector["metrics"][split]["operational"]
        decided = operational["decided_metrics"]
        decided_f1 = "-" if decided is None else f"{decided['binary']['f1']:.6f}"
        lines.append(
            f"| {split} | {operational['coverage']:.6f} | "
            f"{operational['abstention_rate']:.6f} | {operational['decided_rows']} | "
            f"{decided_f1} |"
        )

    lines.extend(
        [
            "",
            "## Metricas de test por origen",
            "",
            "| Origen | n | Accuracy | Precision | Recall | F1 | Cobertura operacional | F1 decidido |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for origin, metrics in detector["metrics"]["test"]["by_origin"].items():
        support = balancing["after"]["by_origin"][origin]["by_split"]["test"]
        operational = detector["metrics"]["test"]["operational_by_origin"][origin]
        decided = operational["decided_metrics"]
        decided_f1 = "-" if decided is None else f"{decided['binary']['f1']:.6f}"
        lines.append(
            f"| {origin} | {support['rows']} | {metrics['accuracy']:.6f} | "
            f"{metrics['precision_weighted']:.6f} | {metrics['recall_weighted']:.6f} | "
            f"{metrics['f1_weighted']:.6f} | {operational['coverage']:.6f} | "
            f"{decided_f1} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_paths = tuple(sorted(args.manifest_dir.resolve().glob("*_manifest.jsonl")))
    result_paths = tuple(sorted(args.results_dir.resolve().glob("*_standardized.jsonl")))
    if not manifest_paths or not result_paths:
        print("ERROR: no se encontraron manifiestos y resultados", file=sys.stderr)
        return 1

    snapshot_before = input_snapshot(manifest_paths, result_paths)
    campaign = load_validation_campaign(
        manifest_paths,
        result_paths,
        PreflightRequirements(
            require_complete=True,
            require_llm=True,
            provider=args.provider,
            model=args.model,
            dedup_scope="dataset",
        ),
    )
    snapshot_after = input_snapshot(manifest_paths, result_paths)
    if snapshot_before != snapshot_after:
        print("ERROR: las entradas cambiaron durante la carga", file=sys.stderr)
        return 1

    result = train_global_detector(
        campaign.joined_records,
        model_path=args.model_out.resolve(),
        balance_seed=args.seed,
    )

    report = {
        "schema_version": "balanced-detection-training-v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "provider": args.provider,
            "model": args.model,
            "snapshot": snapshot_after,
            "joined_rows": len(campaign.joined_records),
            "llm_success_rows": campaign.quality.global_metrics.valid_llm_rows,
        },
        "detector": result.report,
        "classifier_untouched": True,
    }
    booster_raw_sha256 = hashlib.sha256(
        result.model.estimator.get_booster().save_raw(raw_format="ubj")
    ).hexdigest()
    portable_metadata = deployment_metadata(
        report,
        artifact_filename=args.model_out.name,
        booster_raw_sha256=booster_raw_sha256,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "detection_balanced_evaluation.json"
    markdown_path = args.out_dir / "detection_balanced_evaluation.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    metadata_path = args.deployment_metadata_out.resolve()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(portable_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "model": str(result.artifact_path),
                "model_sha256": result.report["artifact"]["sha256"],
                "json": str(json_path.resolve()),
                "markdown": str(markdown_path.resolve()),
                "deployment_metadata": str(metadata_path),
                "balanced_rows": result.report["balancing"]["output_rows"],
                "test": result.report["metrics"]["test"]["global"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
