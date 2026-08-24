from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_xgboost_attack_family_classification_group_lodo import (
    dataset_group,
    load_edge_attack_families,
    load_prebalanced_attack_families,
)
from scripts.train_xgboost_detection_standardized_datasets import (
    DEFAULT_EDGE,
    DEFAULT_STANDARDIZED,
    event_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one global XGBoost attack-family classifier on Mistral-standardized canonical events, "
            "balanced by dataset family and attack family."
        )
    )
    parser.add_argument("--standardized", default=DEFAULT_STANDARDIZED)
    parser.add_argument("--edge-standardized", default=DEFAULT_EDGE)
    parser.add_argument("--per-group-family", type=int, default=100)
    parser.add_argument("--edge-per-family-pool", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--out-prefix", default="artifacts/xgboost_attack_family_balanced_group_20260705")
    parser.add_argument("--model-out", default="artifacts/models/xgboost_attack_family_balanced_group_20260705.joblib")
    return parser.parse_args()


class EncodedClassifier:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.encoder = LabelEncoder()

    def fit(self, x: list[dict[str, Any]], y: list[str]) -> "EncodedClassifier":
        encoded = self.encoder.fit_transform(y)
        self.pipeline.fit(x, encoded)
        return self

    def predict(self, x: list[dict[str, Any]]) -> list[str]:
        encoded = self.pipeline.predict(x)
        return [str(item) for item in self.encoder.inverse_transform(encoded)]

    @property
    def classes(self) -> list[str]:
        return [str(item) for item in self.encoder.classes_]


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    rng = random.Random(args.random_state)

    pool = load_pool(args, rng)
    records = balance_by_group_and_family(pool, args.per_group_family, rng)
    train, val, test = split_records(records, args)

    model = make_model(args)
    train_x, train_y = make_xy(train)
    val_x, val_y = make_xy(val)
    test_x, test_y = make_xy(test)

    fit_started = time.perf_counter()
    model.fit(train_x, train_y)
    fit_seconds = round(time.perf_counter() - fit_started, 3)

    val_pred = model.predict(val_x)
    test_pred = model.predict(test_x)

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "settings": vars(args),
        "target": "attack_family",
        "dataset": {
            "selection_policy": (
                "Use only malicious standardized rows. no_attack is excluded because it belongs to the binary detector. "
                "Rows are capped by broad dataset family and attack_family so no origin/family pair dominates training."
            ),
            "feature_policy": (
                "Features come from canonical_event through the same generalized feature standardizer used in detection. "
                "Dataset origin/provenance, target-like keys, semantic_hint and attack_indicator features are excluded."
            ),
            "pool_rows": len(pool),
            "balanced_rows": len(records),
            "per_group_family_cap": args.per_group_family,
            "class_counts": dict(sorted(Counter(record["label"] for record in records).items())),
            "by_dataset_family": summarize_by(records, "dataset_group"),
            "by_dataset_input": summarize_by(records, "dataset"),
            "splits": {
                "train": split_summary(train),
                "val": split_summary(val),
                "test": split_summary(test),
            },
        },
        "model": {
            "name": "XGBoost",
            "fit_seconds": fit_seconds,
            "classes": model.classes,
            "params": {
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "learning_rate": args.learning_rate,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree,
            },
        },
        "metrics": {
            "validation": multiclass_metrics(val_y, val_pred),
            "test": multiclass_metrics(test_y, test_pred),
            "test_by_dataset_family": metrics_by(test, test_pred, "dataset_group"),
            "test_by_dataset_input": metrics_by(test, test_pred, "dataset"),
        },
    }

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    artifact["model"]["path"] = str(model_path)

    out_base = timestamped_path(Path(args.out_prefix))
    out_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_base.with_suffix(".json")
    md_path = out_base.with_suffix(".md")
    json_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(artifact), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "model": str(model_path),
                "test": artifact["metrics"]["test"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def load_pool(args: argparse.Namespace, rng: random.Random) -> list[dict[str, Any]]:
    records = load_prebalanced_attack_families(Path(args.standardized))
    records.extend(load_edge_attack_families(Path(args.edge_standardized), args.edge_per_family_pool, rng))
    for record in records:
        record["dataset_group"] = dataset_group(record["dataset"])
    rng.shuffle(records)
    return records


def balance_by_group_and_family(
    records: list[dict[str, Any]], per_pair: int, rng: random.Random
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["dataset_group"], record["label"])].append(record)

    selected: list[dict[str, Any]] = []
    for (_group, _family), values in sorted(grouped.items()):
        values = list(values)
        rng.shuffle(values)
        selected.extend(values[: min(per_pair, len(values))])
    rng.shuffle(selected)
    return selected


def split_records(records: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    # Stratify by both target family and broad dataset family where possible. This keeps the split comparable by origin.
    stratify_keys = [f"{record['dataset_group']}::{record['label']}" for record in records]
    if min(Counter(stratify_keys).values()) < 2:
        stratify_keys = [record["label"] for record in records]

    train_val, test = train_test_split(
        records,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify_keys,
    )
    train_val_stratify = [f"{record['dataset_group']}::{record['label']}" for record in train_val]
    if min(Counter(train_val_stratify).values()) < 2:
        train_val_stratify = [record["label"] for record in train_val]
    relative_val_size = args.val_size / (1.0 - args.test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        random_state=args.random_state,
        stratify=train_val_stratify,
    )
    return list(train), list(val), list(test)


def make_model(args: argparse.Namespace) -> EncodedClassifier:
    pipeline = Pipeline(
        [
            ("features", DictVectorizer(sparse=True)),
            (
                "model",
                XGBClassifier(
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth,
                    learning_rate=args.learning_rate,
                    subsample=args.subsample,
                    colsample_bytree=args.colsample_bytree,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=args.random_state,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            ),
        ]
    )
    return EncodedClassifier(pipeline)


def make_xy(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    return [event_features(record["event"]) for record in records], [record["label"] for record in records]


def multiclass_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    labels = sorted(set(y_true) | set(y_pred))
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    return {
        "n": len(y_true),
        "class_counts": dict(sorted(Counter(y_true).items())),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f_macro),
        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "weighted_f1": float(f_weighted),
    }


def metrics_by(records: list[dict[str, Any]], y_pred: list[str], key: str) -> dict[str, Any]:
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for record, pred in zip(records, y_pred, strict=True):
        grouped[str(record[key])].append((record["label"], str(pred)))
    out: dict[str, Any] = {}
    for name, values in sorted(grouped.items()):
        y_true = [item[0] for item in values]
        pred = [item[1] for item in values]
        if len(set(y_true)) < 2:
            out[name] = {
                "n": len(values),
                "class_counts": dict(sorted(Counter(y_true).items())),
                "note": "not enough target classes for multiclass metrics",
            }
            continue
        out[name] = multiclass_metrics(y_true, pred)
    return out


def summarize_by(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record[key])].append(record)
    return {
        name: {
            "rows": len(items),
            "class_counts": dict(sorted(Counter(item["label"] for item in items).items())),
        }
        for name, items in sorted(grouped.items())
    }


def split_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(records),
        "class_counts": dict(sorted(Counter(record["label"] for record in records).items())),
        "by_dataset_family": summarize_by(records, "dataset_group"),
    }


def timestamped_path(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.name}_{stamp}")


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def metric_row(name: str, values: dict[str, Any]) -> str:
    if "accuracy" not in values:
        return f"| {name} | {values.get('n', '-')} | - | - | - | - | {values.get('note', '')} |"
    return (
        f"| {name} | {values['n']} | {fmt(values['accuracy'])} | {fmt(values['balanced_accuracy'])} | "
        f"{fmt(values['macro_f1'])} | {fmt(values['weighted_f1'])} | `{values.get('class_counts', {})}` |"
    )


def render_markdown(artifact: dict[str, Any]) -> str:
    lines = [
        "# XGBoost Clasificacion Global Balanceada De Familia De Ataque",
        "",
        f"Generado: {artifact['generated_at']}",
        "",
        "## Dataset",
        "",
        f"- Pool malicioso disponible: {artifact['dataset']['pool_rows']}",
        f"- Filas usadas tras balanceo: {artifact['dataset']['balanced_rows']}",
        f"- Cap por par `familia_dataset + attack_family`: {artifact['dataset']['per_group_family_cap']}",
        f"- Clases finales: `{artifact['dataset']['class_counts']}`",
        "- Target: `attack_family`.",
        "- `no_attack` se excluye porque la deteccion se evalua aparte.",
        "- Es un unico clasificador global; no se entrena un modelo por dataset.",
        "",
        "## Distribucion Por Familia De Dataset",
        "",
        "| familia dataset | filas | clases |",
        "| --- | ---: | --- |",
    ]
    for group, values in artifact["dataset"]["by_dataset_family"].items():
        lines.append(f"| {group} | {values['rows']} | `{values['class_counts']}` |")

    lines.extend(
        [
            "",
            "## Split Normal: Metricas Globales",
            "",
            "| split | n | accuracy | balanced accuracy | macro-F1 | weighted-F1 | clases |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            metric_row("validation", artifact["metrics"]["validation"]),
            metric_row("test", artifact["metrics"]["test"]),
            "",
            "## Split Normal: Test Por Familia De Dataset",
            "",
            "| familia dataset | n test | accuracy | balanced accuracy | macro-F1 | weighted-F1 | clases en test |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for group, values in artifact["metrics"]["test_by_dataset_family"].items():
        lines.append(metric_row(group, values))
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
