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

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_xgboost_detection_standardized_datasets import DEFAULT_EDGE, DEFAULT_STANDARDIZED, event_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate XGBoost attack-family classification by broad dataset group and LODO."
    )
    parser.add_argument("--standardized", default=DEFAULT_STANDARDIZED)
    parser.add_argument("--edge-standardized", default=DEFAULT_EDGE)
    parser.add_argument("--edge-per-family", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--out-prefix", default="artifacts/xgboost_attack_family_group_lodo_20260705")
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
    records = load_records(args)

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

    lodo = evaluate_lodo(records, args)
    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "settings": vars(args),
        "target": "attack_family",
        "dataset": {
            "total_rows": len(records),
            "class_counts": dict(sorted(Counter(record["label"] for record in records).items())),
            "by_group": summarize_by(records, "dataset_group"),
            "by_input": summarize_by(records, "dataset"),
            "notes": [
                "Only malicious rows are used. no_attack rows are excluded because detection is evaluated separately.",
                "Edge-IIoTset is sampled with edge_per_family rows per attack family to avoid dominating the mixed dataset.",
                "Metrics are aggregate per dataset group; per-attack-family detail is intentionally not reported.",
            ],
        },
        "normal_split": {
            "fit_seconds": fit_seconds,
            "classes": model.classes,
            "splits": {
                "train": split_summary(train),
                "val": split_summary(val),
                "test": split_summary(test),
            },
            "validation": multiclass_metrics(val_y, val_pred),
            "test": multiclass_metrics(test_y, test_pred),
            "test_by_group": metrics_by(test, test_pred, "dataset_group"),
        },
        "lodo": lodo,
    }

    out_base = timestamped_path(Path(args.out_prefix))
    out_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_base.with_suffix(".json")
    md_path = out_base.with_suffix(".md")
    json_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(artifact), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "normal_test": artifact["normal_split"]["test"]}, indent=2, ensure_ascii=False))


def load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = random.Random(args.random_state)
    records = load_prebalanced_attack_families(Path(args.standardized))
    records.extend(load_edge_attack_families(Path(args.edge_standardized), args.edge_per_family, rng))
    for record in records:
        record["dataset_group"] = dataset_group(record["dataset"])
    rng.shuffle(records)
    return records


def load_prebalanced_attack_families(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task") != "multiclass_attack_type" or not row.get("ok"):
                continue
            target = row.get("target") or {}
            if not target.get("is_attack"):
                continue
            record = record_from_row(row, str(row.get("input") or "unknown"))
            if record is not None:
                records.append(record)
    return records


def load_edge_attack_families(path: Path, per_family: int, rng: random.Random) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task") != "multiclass" or not row.get("ok"):
                continue
            target = row.get("target") or {}
            if not target.get("is_attack"):
                continue
            record = record_from_row(row, "EDGE_IIOTSET")
            if record is not None:
                grouped[record["label"]].append(record)
    selected: list[dict[str, Any]] = []
    for family, values in sorted(grouped.items()):
        rng.shuffle(values)
        selected.extend(values[: min(per_family, len(values))])
    return selected


def record_from_row(row: dict[str, Any], dataset: str) -> dict[str, Any] | None:
    target = row.get("target") or {}
    event = row.get("canonical_event") or {}
    family = normalize_family(target.get("attack_family"))
    if not event or not family or family == "benign":
        return None
    return {
        "sample_id": row.get("sample_id") or f"{dataset}::{row.get('row_index') or row.get('row_id')}",
        "dataset": dataset,
        "source_file": row.get("source_file"),
        "row_id": row.get("row_id", row.get("row_index")),
        "label": family,
        "event": event,
    }


def normalize_family(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"", "none", "null", "nan", "normal", "benign", "no_attack"}:
        return ""
    return text


def dataset_group(dataset: str) -> str:
    if dataset.startswith("TON_IOT_linux_"):
        return "TON_IOT_linux"
    if dataset.startswith("TON_IOT_telemetry_"):
        return "TON_IOT_telemetry"
    if dataset.startswith("TON_IOT_windows_"):
        return "TON_IOT_windows"
    return dataset


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


def evaluate_lodo(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for group in sorted({record["dataset_group"] for record in records}):
        test_records = [record for record in records if record["dataset_group"] == group]
        train_records = [record for record in records if record["dataset_group"] != group]
        train_labels = {record["label"] for record in train_records}
        test_labels = {record["label"] for record in test_records}
        unseen = sorted(test_labels - train_labels)
        if len(train_labels) < 2:
            results[group] = {"status": "skipped", "reason": "not enough train classes", "test_rows": len(test_records)}
            continue
        model = make_model(args)
        train_x, train_y = make_xy(train_records)
        test_x, test_y = make_xy(test_records)
        started = time.perf_counter()
        model.fit(train_x, train_y)
        pred = model.predict(test_x)
        result = multiclass_metrics(test_y, pred)
        result.update(
            {
                "status": "ok",
                "train_rows": len(train_records),
                "test_rows": len(test_records),
                "fit_seconds": round(time.perf_counter() - started, 3),
                "unseen_test_classes": unseen,
                "test_by_input": metrics_by(test_records, pred, "dataset"),
            }
        )
        results[group] = result
    return results


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
        if len({item[0] for item in values}) < 2:
            out[name] = {
                "n": len(values),
                "class_counts": dict(sorted(Counter(item[0] for item in values).items())),
                "note": "not enough classes for multiclass metrics",
            }
            continue
        out[name] = multiclass_metrics([item[0] for item in values], [item[1] for item in values])
    return out


def summarize_by(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record[key])].append(record)
    return {name: {"rows": len(items), "class_counts": dict(sorted(Counter(item["label"] for item in items).items()))} for name, items in sorted(grouped.items())}


def split_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(records),
        "class_counts": dict(sorted(Counter(record["label"] for record in records).items())),
        "by_group": summarize_by(records, "dataset_group"),
    }


def timestamped_path(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.name}_{stamp}")


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def metric_row(name: str, values: dict[str, Any]) -> str:
    if values.get("status") == "skipped" or "accuracy" not in values:
        return f"| {name} | {values.get('test_rows', values.get('n', '-'))} | - | - | - | - | {values.get('reason') or values.get('note') or ''} |"
    return (
        f"| {name} | {values['n']} | {fmt(values['accuracy'])} | {fmt(values['balanced_accuracy'])} | "
        f"{fmt(values['macro_f1'])} | {fmt(values['weighted_f1'])} | {', '.join(values.get('unseen_test_classes') or [])} |"
    )


def render_markdown(artifact: dict[str, Any]) -> str:
    lines = [
        "# XGBoost Clasificacion De Familia De Ataque",
        "",
        f"Generado: {artifact['generated_at']}",
        "",
        "## Dataset",
        "",
        f"- Filas maliciosas totales: {artifact['dataset']['total_rows']}",
        f"- Clases: `{artifact['dataset']['class_counts']}`",
        "- Target: `attack_family`.",
        "- `no_attack` se excluye porque la deteccion se evalua aparte.",
        "- Se reportan metricas agregadas por origen; no se desglosa por tipo/familia concreta.",
        "",
        "## Split Normal: Test Por Grupo",
        "",
        "| grupo | n test | accuracy | balanced accuracy | macro-F1 | weighted-F1 | clases no vistas |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for group, values in artifact["normal_split"]["test_by_group"].items():
        lines.append(metric_row(group, values))
    lines.extend(
        [
            "",
            "## LODO: Leave-One-Dataset-Origin-Out",
            "",
            "En cada fila se entrena con todos los grupos salvo el indicado y se testea con ese grupo completo.",
            "",
            "| grupo dejado fuera | n test | accuracy | balanced accuracy | macro-F1 | weighted-F1 | clases no vistas |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for group, values in artifact["lodo"].items():
        lines.append(metric_row(group, values))
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
