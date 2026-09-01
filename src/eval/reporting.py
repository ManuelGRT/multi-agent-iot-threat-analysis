"""Utilidades offline para métricas, baselines e informes de validación.

Estas funciones pertenecen al entorno experimental. No se exponen como
herramientas MCP ni se incluyen en el paquete de ejecución online.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.mcp.common import artifacts_dir, resolve_confined_path, resolve_path


def load_baselines() -> dict[str, Any]:
    path = resolve_path("baselines")
    if not path.exists():
        raise FileNotFoundError(f"Baselines no encontrados en {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def score_run(
    y_true: list[Any],
    y_pred: list[Any],
    *,
    task: str = "multiclass",
    y_score: list[float] | None = None,
) -> dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred deben tener la misma longitud")

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    average = "binary" if task == "binary" else "weighted"
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    result: dict[str, Any] = {
        "task": task,
        "n_samples": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    if task == "binary" and y_score is not None:
        try:
            from sklearn.metrics import roc_auc_score

            result["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            pass
    return result


def compare_with_baseline(f1: float, *, task: str = "multiclass") -> dict[str, Any]:
    baselines = load_baselines()
    thresholds = baselines["audit_thresholds"]
    if task == "multiclass":
        jorge_best = baselines["task_multiclass"]["jorge_deepseek_finetuned_less"]["f1"]
        current_ref = baselines["task_multiclass"]["current_system_canonical_edge"]["f1"]
        min_required = thresholds["multiclass_f1_min_canonical_edge"]
        must_beat = thresholds["multiclass_f1_must_beat"]
    else:
        jorge_best = baselines["task_binary"]["jorge"]["f1"]
        current_ref = baselines["task_binary"]["current_system_canonical_edge"]["f1"]
        min_required = thresholds["binary_f1_min"]
        must_beat = thresholds["binary_f1_min"]
    return {
        "task": task,
        "f1": f1,
        "jorge_best_f1": jorge_best,
        "current_reference_f1": current_ref,
        "beats_jorge": f1 > must_beat,
        "meets_audit_threshold": f1 >= min_required,
        "audit_threshold": min_required,
        "delta_vs_jorge": round(f1 - jorge_best, 4),
        "narrative": baselines.get("narrative"),
    }


def export_report(
    title: str,
    sections: list[dict[str, Any]],
    *,
    filename: str | None = None,
    output_root: str | Path | None = None,
) -> Path:
    root = Path(output_root) if output_root is not None else artifacts_dir()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = filename or f"informe_{stamp}.md"
    path = resolve_confined_path(name, root, allow_absolute=False)
    lines = [f"# {title}", "", f"Generado: {datetime.now(timezone.utc).isoformat()}", ""]
    for section in sections:
        lines.extend((f"## {section.get('heading', 'Sección')}", ""))
        body = section.get("body", "")
        if isinstance(body, (dict, list)):
            lines.extend(("```json", json.dumps(body, indent=2, ensure_ascii=False, default=str), "```"))
        else:
            lines.append(str(body))
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
