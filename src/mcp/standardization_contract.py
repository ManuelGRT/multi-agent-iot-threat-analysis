"""Contrato fail-closed de una estandarizacion aceptada por el flujo final."""
from __future__ import annotations

import math
from typing import Any

from src.contracts.canonical import CanonicalEvent
from src.mcp.standardization_guard import sanitize_canonical_event, sanitize_llm_input


def validate_standardization_success(
    raw_input: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Valida y normaliza un exito MCP o lanza ``ValueError``.

    Se comparte entre el agente final y el runner de campaña para que ninguna
    respuesta de caché, adaptador, otro proveedor o contrato incompleto pueda
    contabilizarse como una estandarización válida.
    """
    if result.get("ok") is not True:
        raise ValueError("la respuesta no declara ok=true")
    if "abstain" in result and result["abstain"] is not False:
        raise ValueError("una respuesta de exito no puede declarar abstencion")
    if (
        "requires_human_review" in result
        and result["requires_human_review"] is not False
    ):
        raise ValueError(
            "una respuesta de exito no puede solicitar revision humana"
        )
    for failure_field in ("failure_code", "failure_reason", "error", "llm_error"):
        if failure_field in result and result[failure_field] not in (None, ""):
            raise ValueError(
                f"una respuesta de exito no puede declarar {failure_field}"
            )
    for fallback_field in ("fallback", "used_fallback"):
        if fallback_field in result and result[fallback_field] is not False:
            raise ValueError(
                f"una respuesta de exito no puede declarar {fallback_field}"
            )
    if result.get("cache_key") not in (None, ""):
        raise ValueError("una respuesta de exito no puede declarar cache_key")
    if result.get("from_cache") is not False:
        raise ValueError("una respuesta de exito debe declarar from_cache=false")

    prestandardized = raw_input.get("canonical_event") is not None
    expected_source = "prestandardized" if prestandardized else "llm"
    if result.get("source") != expected_source:
        raise ValueError(
            "fuente de estandarizacion incompatible: "
            f"esperada={expected_source}, recibida={result.get('source')!r}"
        )

    provider = result.get("provider")
    if prestandardized:
        if provider is not None:
            raise ValueError(
                "un evento preestandarizado no puede declarar proveedor LLM"
            )
    elif provider != "mistral":
        raise ValueError(
            "proveedor de estandarizacion incompatible: "
            f"esperado='mistral', recibido={provider!r}"
        )

    model = result.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model debe ser una cadena no vacia")

    selected_columns = result.get("selected_columns")
    if not isinstance(selected_columns, list) or not all(
        isinstance(column, str) and column.strip()
        for column in selected_columns
    ):
        raise ValueError("selected_columns debe ser una lista de cadenas no vacias")

    row = raw_input.get("row")
    if prestandardized:
        if selected_columns:
            raise ValueError(
                "un evento preestandarizado no puede declarar columnas seleccionadas"
            )
    elif isinstance(row, dict) and row:
        if not selected_columns:
            raise ValueError(
                "una fila tabular requiere seleccion de columnas mediante LLM"
            )
        if len(set(selected_columns)) != len(selected_columns):
            raise ValueError("selected_columns no puede contener duplicados")
        safe_row = sanitize_llm_input({"row": row})["row"]
        unknown = [column for column in selected_columns if column not in safe_row]
        if unknown:
            raise ValueError(
                "selected_columns contiene columnas ausentes o no autorizadas: "
                + ", ".join(unknown)
            )

    notes = result.get("notes")
    if not isinstance(notes, list) or not all(
        isinstance(note, str) for note in notes
    ):
        raise ValueError("notes debe ser una lista de cadenas")
    required_notes = (
        {"security_boundary:sanitized", "source:prestandardized"}
        if prestandardized
        else {"security_boundary:sanitized", "parsed_by_llm"}
    )
    missing_notes = required_notes.difference(notes)
    if missing_notes:
        raise ValueError(
            "faltan marcas de procedencia en notes: "
            + ", ".join(sorted(missing_notes))
        )
    forbidden_note_markers = ("fallback", "adapter", "cache", "llm_failed", "llm_error")
    if any(
        marker in note.strip().casefold()
        for note in notes
        for marker in forbidden_note_markers
    ):
        raise ValueError("notes contiene una marca de fallo, cache o fallback")

    canonical = result.get("canonical_event")
    if not isinstance(canonical, dict):
        raise ValueError("canonical_event debe ser un objeto")
    validated_event = CanonicalEvent(**canonical)
    validated_canonical = validated_event.model_dump(mode="json")
    sanitized_canonical = sanitize_canonical_event(validated_canonical)
    if sanitized_canonical != validated_canonical:
        raise ValueError("canonical_event no respeta la frontera de sanitizacion")

    if "mapping_confidence" not in result:
        raise ValueError("falta mapping_confidence en la respuesta")
    raw_confidence = result["mapping_confidence"]
    if not isinstance(raw_confidence, (int, float)) or isinstance(
        raw_confidence, bool
    ):
        raise ValueError("mapping_confidence debe ser numerico")
    mapping_confidence = float(raw_confidence)
    if not math.isfinite(mapping_confidence):
        raise ValueError("mapping_confidence debe ser finito")
    if abs(mapping_confidence - validated_event.mapping_confidence) > 1e-9:
        raise ValueError("mapping_confidence no coincide con canonical_event")

    return {
        "canonical_event": sanitized_canonical,
        "mapping_confidence": mapping_confidence,
        "selected_columns": list(selected_columns),
        "notes": list(notes),
        "source": expected_source,
        "provider": provider,
        "model": model,
    }
