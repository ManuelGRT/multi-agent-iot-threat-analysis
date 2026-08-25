# src/mcp/datasets_server.py
"""Servidor MCP de datasets: fuentes crudas y corpus estandarizado."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

try:  # SDK MCP oficial; opcional para el modo in-process
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - sin SDK solo se pierde el modo stdio
    FastMCP = None

from src.mcp.common import (
    data_dir,
    iter_jsonl,
    register_tools,
    resolve_confined_path,
    resolve_path,
    tool_result,
)

mcp_app = FastMCP("mcp-datasets") if FastMCP is not None else None

# Datasets crudos conocidos: subcarpeta de data/ y adapter asociado.
KNOWN_RAW_DATASETS: dict[str, dict[str, str]] = {
    "edge_iiotset": {"folder": "EDGE_IIOTSET", "adapter": "edge_iiotset"},
    "ton_iot": {"folder": "TON_IOT", "adapter": "ton_iot"},
    "bot_iot": {"folder": "BOT_IOT", "adapter": "bot_iot"},
    "iot23": {"folder": "IOT23", "adapter": "iot23"},
    "urban_iot": {"folder": "URBAN_IOT", "adapter": "generic"},
    "simulated": {"folder": "SIMULATED", "adapter": "generic"},
}

STANDARDIZED_KEY = "standardized_corpus"


def _first_csv(folder: Path) -> Path | None:
    folder = folder.expanduser().resolve()
    if not folder.exists():
        return None
    candidates = sorted(folder.rglob("*.csv"))
    return candidates[0] if candidates else None


@tool_result
def list_datasets() -> dict[str, Any]:
    """Lista datasets disponibles (crudos y corpus estandarizado)."""
    items: list[dict[str, Any]] = []
    for name, meta in KNOWN_RAW_DATASETS.items():
        folder = data_dir() / meta["folder"]
        sample = _first_csv(folder)
        items.append(
            {
                "name": name,
                "kind": "raw",
                "adapter": meta["adapter"],
                "available": sample is not None,
                "example_file": str(sample) if sample else None,
            }
        )
    std = resolve_path("standardized_dataset")
    items.append(
        {
            "name": STANDARDIZED_KEY,
            "kind": "standardized_jsonl",
            "adapter": None,
            "available": std.exists(),
            "example_file": str(std) if std.exists() else None,
        }
    )
    return {"datasets": items}


@tool_result
def load_sample(dataset: str, n: int = 1, offset: int = 0, path: str | None = None) -> dict[str, Any]:
    """Carga n registros; cualquier ``path`` debe quedar dentro de TFM_DATA_DIR."""
    return _load(dataset, n=n, offset=offset, path=path)


@tool_result
def load_batch(dataset: str, n: int = 100, offset: int = 0, path: str | None = None) -> dict[str, Any]:
    """Carga un batch; cualquier ``path`` debe quedar dentro de TFM_DATA_DIR."""
    return _load(dataset, n=n, offset=offset, path=path)


def _load(dataset: str, n: int, offset: int, path: str | None) -> dict[str, Any]:
    if dataset == STANDARDIZED_KEY:
        # La ruta configurada del corpus es de confianza. Cualquier override
        # recibido por una tool MCP queda confinado a TFM_DATA_DIR.
        source = (
            resolve_confined_path(path, data_dir())
            if path
            else resolve_path("standardized_dataset")
        )
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"Corpus estandarizado no encontrado: {source}")
        rows = list(iter_jsonl(source, limit=n, offset=offset))
        return {"dataset": dataset, "count": len(rows), "rows": rows, "source_file": str(source)}

    meta = KNOWN_RAW_DATASETS.get(dataset)
    if meta is None:
        raise ValueError(f"Dataset desconocido: {dataset}. Usa list_datasets.")
    candidate = Path(path) if path else _first_csv(data_dir() / meta["folder"])
    source = resolve_confined_path(candidate, data_dir()) if candidate is not None else None
    if source is None or not source.exists() or not source.is_file():
        raise FileNotFoundError(f"No hay CSV disponible para {dataset} en {data_dir() / meta['folder']}")
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for index, row in enumerate(reader):
            if index < offset:
                continue
            if len(rows) >= n:
                break
            rows.append(dict(row))
    return {
        "dataset": dataset,
        "adapter": meta["adapter"],
        "count": len(rows),
        "rows": rows,
        "source_file": str(source),
    }


@tool_result
def get_schema_profile(dataset: str, path: str | None = None) -> dict[str, Any]:
    """Devuelve columnas y un perfil orientativo del dataset."""
    loaded = _load(dataset, n=1, offset=0, path=path)
    rows = loaded.get("rows") or []
    if not rows:
        raise ValueError(f"Dataset vacio: {dataset}")
    first = rows[0]
    if dataset == STANDARDIZED_KEY:
        event = first.get("canonical_event") or {}
        return {
            "dataset": dataset,
            "columns": sorted(event.keys()),
            "schema_profile": (event.get("origin") or {}).get("schema_profile") or event.get("schema_profile"),
            "modality": event.get("modality"),
        }
    columns = sorted(first.keys())
    lowered = {column.lower() for column in columns}
    if {"saddr", "daddr"} & lowered or {"id.orig_h", "src_ip"} & lowered:
        profile = "network_flow"
    elif any("temp" in column or "sensor" in column for column in lowered):
        profile = "iot_telemetry"
    else:
        profile = "unknown"
    return {"dataset": dataset, "columns": columns, "schema_profile": profile, "modality": None}


TOOLS = {
    "list_datasets": list_datasets,
    "load_sample": load_sample,
    "load_batch": load_batch,
    "get_schema_profile": get_schema_profile,
}
register_tools(mcp_app, TOOLS)


if __name__ == "__main__":
    if mcp_app is None:
        raise SystemExit("SDK MCP no disponible: instala mcp[cli]>=1.2 (extra [mcp]).")
    mcp_app.run()
