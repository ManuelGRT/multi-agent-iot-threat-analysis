# src/mcp/evaluation_server.py
"""Servidor MCP de evaluacion: metricas, comparacion con baselines e informes."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

try:  # SDK MCP oficial; opcional para el modo in-process
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - sin SDK solo se pierde el modo stdio
    FastMCP = None

from src.mcp.common import (
    artifacts_dir,
    register_tools,
    resolve_confined_path,
    resolve_path,
    tool_result,
)

mcp_app = FastMCP("mcp-evaluation") if FastMCP is not None else None


def _load_baselines() -> dict[str, Any]:
    path = resolve_path("baselines")
    if not path.exists():
        raise FileNotFoundError(
            f"Baselines no encontrados en {path}. Ejecuta la Fase 0 (congelacion de resultados)."
        )
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@tool_result
def score_run(
    y_true: list[Any],
    y_pred: list[Any],
    task: str = "multiclass",
    y_score: list[float] | None = None,
) -> dict[str, Any]:
    """Calcula accuracy/precision/recall/F1 (binario o multiclase ponderada)."""
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
        except Exception:
            pass
    return result


@tool_result
def compare_with_baseline(f1: float, task: str = "multiclass") -> dict[str, Any]:
    """Compara un F1 con los baselines congelados de la Fase 0 (incl. Jorge)."""
    baselines = _load_baselines()
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


@tool_result
def get_baselines() -> dict[str, Any]:
    """Devuelve los baselines congelados completos."""
    return {"baselines": _load_baselines()}


@tool_result
def export_report(
    title: str,
    sections: list[dict[str, Any]],
    filename: str | None = None,
) -> dict[str, Any]:
    """Exporta un informe markdown a artifacts/.

    ``filename`` debe ser relativo a artifacts/ y no puede contener ``..``.
    sections: lista de {"heading": str, "body": str | dict | list}.
    Los dict/list se renderizan como bloque JSON.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = filename or f"informe_mcp_{stamp}.md"
    path = resolve_confined_path(name, artifacts_dir(), allow_absolute=False)
    lines = [f"# {title}", "", f"Generado: {datetime.now(timezone.utc).isoformat()}", ""]
    for section in sections:
        lines.append(f"## {section.get('heading', 'Seccion')}")
        lines.append("")
        body = section.get("body", "")
        if isinstance(body, (dict, list)):
            lines.append("```json")
            lines.append(json.dumps(body, indent=2, ensure_ascii=False, default=str))
            lines.append("```")
        else:
            lines.append(str(body))
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return {"path": str(path), "sections": len(sections)}


TOOLS = {
    "score_run": score_run,
    "compare_with_baseline": compare_with_baseline,
    "get_baselines": get_baselines,
    "export_report": export_report,
}
register_tools(mcp_app, TOOLS)


if __name__ == "__main__":
    if mcp_app is None:
        raise SystemExit("SDK MCP no disponible: instala mcp[cli]>=1.2 (extra [mcp]).")
    mcp_app.run()
