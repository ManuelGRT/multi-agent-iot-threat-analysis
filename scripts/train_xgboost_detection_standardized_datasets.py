from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.agents.predictive_sanitization import is_predictive_target_field
from src.agents.supervised_general import GeneralizedFeatureStandardizer
from src.contracts.canonical import CanonicalEvent
from src.mcp.standardization_guard import sanitize_canonical_event


DEFAULT_STANDARDIZED = "artifacts/datasets/mistral_prebalanced_no_simulated_logs_standardized_20260704_all.jsonl"
DEFAULT_EDGE = "artifacts/datasets/edgeiiot_mistral_standardized_tfm_less_both_partial_cache_recalc_20260621.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost binary attack detector on Mistral-standardized canonical datasets."
    )
    parser.add_argument("--standardized", default=DEFAULT_STANDARDIZED)
    parser.add_argument("--edge-standardized", default=DEFAULT_EDGE)
    parser.add_argument("--edge-per-class", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--out-prefix", default="artifacts/xgboost_detection_standardized_with_edge_20260705")
    parser.add_argument("--model-out", default="artifacts/models/xgboost_detection_standardized_with_edge_20260705.joblib")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    rng = random.Random(args.random_state)

    non_edge = load_prebalanced_binary(Path(args.standardized))
    edge = load_edge_binary(Path(args.edge_standardized), args.edge_per_class, rng)
    records = non_edge + edge
    rng.shuffle(records)

    train, val, test = split_records(records, args)
    train_x, train_y = make_xy(train)
    val_x, val_y = make_xy(val)
    test_x, test_y = make_xy(test)

    model = Pipeline(
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
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=args.random_state,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            ),
        ]
    )
    fit_started = time.perf_counter()
    model.fit(train_x, train_y)
    fit_seconds = round(time.perf_counter() - fit_started, 3)

    val_pred = model.predict(val_x)
    test_pred = model.predict(test_x)
    val_proba = positive_proba(model, val_x)
    test_proba = positive_proba(model, test_x)

    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "settings": vars(args),
        "dataset": {
            "selection_policy": (
                "Use binary_detection rows from the Mistral pre-balanced dataset. "
                "Add Edge-IIoTset with the same per-input binary size: edge_per_class benign and edge_per_class attack. "
                "All rows must have ok=true. Target labels are used only for balancing and scoring."
            ),
            "feature_policy": (
                "Features are generated from canonical_event structured fields with GeneralizedFeatureStandardizer. "
                "Dataset origin/provenance, target-like keys, semantic_hint and attack_indicator features are excluded."
            ),
            "total_rows": len(records),
            "class_counts": dict(sorted(Counter(record["label_name"] for record in records).items())),
            "by_dataset": summarize_by_dataset(records),
            "splits": {
                "train": split_summary(train),
                "val": split_summary(val),
                "test": split_summary(test),
            },
        },
        "feature_stats": {
            "train_feature_dicts": len(train_x),
            "vectorized_features": len(model.named_steps["features"].get_feature_names_out()),
            "top_features_by_gain": top_xgb_features(model, limit=25),
        },
        "model": {
            "name": "XGBoost",
            "fit_seconds": fit_seconds,
            "params": {
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "learning_rate": args.learning_rate,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree,
            },
        },
        "metrics": {
            "validation": metrics(val_y, val_pred, val_proba),
            "test": metrics(test_y, test_pred, test_proba),
            "test_by_dataset": metrics_by_dataset(test, test_pred, test_proba),
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
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "model": str(model_path), "test": artifact["metrics"]["test"]}, indent=2, ensure_ascii=False))


def load_prebalanced_binary(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task") != "binary_detection" or not row.get("ok"):
                continue
            record = record_from_row(row, dataset=str(row.get("input") or "unknown"))
            if record is not None:
                records.append(record)
    return records


def load_edge_binary(path: Path, per_class: int, rng: random.Random) -> list[dict[str, Any]]:
    grouped: dict[bool, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task") != "binary" or not row.get("ok"):
                continue
            record = record_from_row(row, dataset="EDGE_IIOTSET")
            if record is not None:
                grouped[record["is_attack"]].append(record)
    take = min(per_class, len(grouped[False]), len(grouped[True]))
    selected: list[dict[str, Any]] = []
    for label in (False, True):
        values = list(grouped[label])
        rng.shuffle(values)
        selected.extend(values[:take])
    return selected


def record_from_row(row: dict[str, Any], dataset: str) -> dict[str, Any] | None:
    target = row.get("target") or {}
    event = row.get("canonical_event") or {}
    if not event:
        return None
    is_attack = bool(target.get("is_attack"))
    return {
        "sample_id": row.get("sample_id") or f"{dataset}::{row.get('row_index') or row.get('row_id')}",
        "dataset": dataset,
        "source_file": row.get("source_file"),
        "row_id": row.get("row_id", row.get("row_index")),
        "is_attack": is_attack,
        "label": int(is_attack),
        "label_name": "attack" if is_attack else "no_attack",
        "event": event,
    }


def split_records(records: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    labels = [record["label"] for record in records]
    train_val, test = train_test_split(
        records,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels,
    )
    train_val_labels = [record["label"] for record in train_val]
    relative_val_size = args.val_size / (1.0 - args.test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        random_state=args.random_state,
        stratify=train_val_labels,
    )
    return list(train), list(val), list(test)


def make_xy(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[int]]:
    return [event_features(record["event"]) for record in records], [record["label"] for record in records]


def event_features(event_data: dict[str, Any]) -> dict[str, Any]:
    event = CanonicalEvent(**sanitize_canonical_event(event_data))
    features = GeneralizedFeatureStandardizer(include_origin=False, feature_set="full").event_to_features(event)
    clean: dict[str, Any] = {}
    for key, value in features.items():
        key_text = str(key)
        if is_predictive_target_field(key_text):
            continue
        if key_text.startswith(
            (
                "origin.",
                "attack_indicator.",
                "semantic_hint.",
                "behavior.",
                "uncertainty.",
                "asset_context.",
            )
        ):
            continue
        if key_text in {
            "attack_indicator_count",
            "behavior_tag_count",
            "uncertainty_count",
        }:
            continue
        clean[key_text] = value
    return clean


def positive_proba(model: Pipeline, x: list[dict[str, Any]]) -> list[float]:
    proba = model.predict_proba(x)
    return [float(row[1]) for row in proba]


def metrics(y_true: list[int], y_pred: list[int], y_score: list[float]) -> dict[str, Any]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = None
    return {
        "n": len(y_true),
        "class_counts": dict(sorted(Counter("attack" if item else "no_attack" for item in y_true).items())),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc) if auc is not None else None,
        "confusion_matrix_labels": ["no_attack", "attack"],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def metrics_by_dataset(records: list[dict[str, Any]], y_pred: list[int], y_score: list[float]) -> dict[str, Any]:
    grouped: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    for record, pred, score in zip(records, y_pred, y_score, strict=True):
        grouped[record["dataset"]].append((record["label"], int(pred), float(score)))
    result: dict[str, Any] = {}
    for dataset, values in sorted(grouped.items()):
        if len(values) < 2 or len({item[0] for item in values}) < 2:
            result[dataset] = {
                "n": len(values),
                "class_counts": dict(sorted(Counter("attack" if item[0] else "no_attack" for item in values).items())),
                "note": "not enough classes in test split for binary metrics",
            }
            continue
        y_true = [item[0] for item in values]
        pred = [item[1] for item in values]
        score = [item[2] for item in values]
        result[dataset] = metrics(y_true, pred, score)
    return result


def summarize_by_dataset(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["dataset"]].append(record)
    return {
        dataset: {
            "rows": len(items),
            "class_counts": dict(sorted(Counter(item["label_name"] for item in items).items())),
        }
        for dataset, items in sorted(grouped.items())
    }


def split_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(records),
        "class_counts": dict(sorted(Counter(record["label_name"] for record in records).items())),
        "by_dataset": summarize_by_dataset(records),
    }


def top_xgb_features(model: Pipeline, limit: int = 25) -> list[dict[str, Any]]:
    vectorizer: DictVectorizer = model.named_steps["features"]
    estimator: XGBClassifier = model.named_steps["model"]
    names = list(vectorizer.get_feature_names_out())
    booster = estimator.get_booster()
    scores = booster.get_score(importance_type="gain")
    rows = []
    for raw_key, gain in scores.items():
        if not raw_key.startswith("f"):
            continue
        try:
            index = int(raw_key[1:])
        except ValueError:
            continue
        if 0 <= index < len(names):
            rows.append({"feature": names[index], "gain": float(gain)})
    rows.sort(key=lambda item: item["gain"], reverse=True)
    return rows[:limit]


def timestamped_path(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.name}_{stamp}")


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(artifact: dict[str, Any]) -> str:
    test = artifact["metrics"]["test"]
    val = artifact["metrics"]["validation"]
    lines = [
        "# XGBoost Detection Sobre Dataset Estandarizado",
        "",
        f"Generado: {artifact['generated_at']}",
        "",
        "## Dataset",
        "",
        f"- Filas totales: {artifact['dataset']['total_rows']}",
        f"- Balance global: `{artifact['dataset']['class_counts']}`",
        "- Edge-IIoTset se incluye con 100 benignos + 100 ataques para mantener el mismo peso por input que el resto de datasets.",
        "- Las features se extraen de `canonical_event`; se excluyen origen/provenance, targets, `semantic_hint` y `attack_indicator`.",
        "",
        "## Metricas",
        "",
        "| split | n | accuracy | balanced accuracy | precision | recall | f1 | roc_auc | matriz [normal, ataque] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        f"| validacion | {val['n']} | {fmt(val['accuracy'])} | {fmt(val['balanced_accuracy'])} | {fmt(val['precision'])} | {fmt(val['recall'])} | {fmt(val['f1'])} | {fmt(val['roc_auc'])} | `{val['confusion_matrix']}` |",
        f"| test | {test['n']} | {fmt(test['accuracy'])} | {fmt(test['balanced_accuracy'])} | {fmt(test['precision'])} | {fmt(test['recall'])} | {fmt(test['f1'])} | {fmt(test['roc_auc'])} | `{test['confusion_matrix']}` |",
        "",
        "## Por Dataset",
        "",
        "| dataset | n test | accuracy | balanced accuracy | precision | recall | f1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset, values in artifact["metrics"]["test_by_dataset"].items():
        if "accuracy" not in values:
            lines.append(f"| {dataset} | {values['n']} | - | - | - | - | - |")
            continue
        lines.append(
            f"| {dataset} | {values['n']} | {fmt(values['accuracy'])} | {fmt(values['balanced_accuracy'])} | "
            f"{fmt(values['precision'])} | {fmt(values['recall'])} | {fmt(values['f1'])} |"
        )
    lines.extend(
        [
            "",
            "## Top Features XGBoost",
            "",
            "| feature | gain |",
            "| --- | ---: |",
        ]
    )
    for item in artifact["feature_stats"]["top_features_by_gain"]:
        lines.append(f"| `{item['feature']}` | {fmt(item['gain'])} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
