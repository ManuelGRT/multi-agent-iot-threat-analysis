"""Utilidades offline para exportar informes de validación.

Estas funciones pertenecen al entorno experimental. No se exponen como
herramientas MCP ni se incluyen en el paquete de ejecución online.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.mcp.common import artifacts_dir, resolve_confined_path


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
