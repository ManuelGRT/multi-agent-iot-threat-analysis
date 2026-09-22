"""Contrato fail-closed de una estandarizacion aceptada por el flujo final."""
from __future__ import annotations

import math
import re
from typing import Any

from src.contracts.canonical import CanonicalEvent
from src.contracts.leakage import (
    contains_predictive_target_text,
    is_allowed_canonical_target_path,
    is_predictive_target_field,
    is_predictive_target_value,
)
from src.mcp.standardization_cache import compute_content_hash

_TARGET_VALUE_CONTAINERS = frozenset(
    {
        "telemetry",
        "host",
        "service_context",
        "host_context",
        "telemetry_context",
        "behavior_tags",
        "uncertainty",
        "asset_context",
        "traffic_direction",
        "transport_proto",
        "app_proto",
    }
)


def validate_target_free_canonical(value: Any, path: str = "") -> None:
    """Rechaza leakage sin alterar el evento recibido.

    Esto es validacion fail-closed del contrato de salida, no sanitizacion. La
    retirada de targets de los datasets ocurre antes de invocar al sistema.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if (
                item not in (None, "", [], {})
                and is_predictive_target_field(key)
                and not is_allowed_canonical_target_path(child_path)
            ):
                raise ValueError(
                    "canonical_event contiene un campo predictivo no permitido: "
                    + child_path
                )
            validate_target_free_canonical(item, child_path)
    elif isinstance(value, (list, tuple)):
        for item in value:
            if (
                isinstance(item, str)
                and (path == "evidence_fields" or path.startswith("feature_groups."))
                and is_predictive_target_field(item)
            ):
                raise ValueError(
                    f"canonical_event referencia una columna predictiva en {path}: {item}"
                )
            validate_target_free_canonical(item, path)
    elif isinstance(value, str):
        root = path.split(".", 1)[0].split("[", 1)[0]
        if root in _TARGET_VALUE_CONTAINERS and is_predictive_target_value(value):
            raise ValueError(
                f"canonical_event contiene un valor predictivo no permitido: {path}"
            )
        if root in {"semantic_text", "anomaly_summary"} and contains_predictive_target_text(value):
            raise ValueError(
                f"canonical_event contiene un patron predictivo no permitido: {path}"
            )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )


def validate_standardization_success(
    raw_input: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Valida un exito MCP o lanza ``ValueError``.

    La funcion no limpia ni reescribe campos por seguridad. Una entrada cruda
    solo se acepta si procede de Mistral en vivo o de un exito Mistral previo
    recuperado por hash exacto de contenido.
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
        raise ValueError("cache_key no puede proceder del cliente ni de la respuesta")

    from_cache = result.get("from_cache")
    if not isinstance(from_cache, bool):
        raise ValueError("from_cache debe ser booleano")

    prestandardized = raw_input.get("canonical_event") is not None
    expected_source = "prestandardized" if prestandardized else "llm"
    if result.get("source") != expected_source:
        raise ValueError(
            "fuente de estandarizacion incompatible: "
            f"esperada={expected_source}, recibida={result.get('source')!r}"
        )

    provider = result.get("provider")
    if prestandardized:
        if from_cache:
            raise ValueError("un evento preestandarizado no puede proceder de cache")
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
        unknown = [column for column in selected_columns if column not in row]
        if unknown:
            raise ValueError(
                "selected_columns contiene columnas ausentes en la entrada: "
                + ", ".join(unknown)
            )

    notes = result.get("notes")
    if not isinstance(notes, list) or not all(
        isinstance(note, str) for note in notes
    ):
        raise ValueError("notes debe ser una lista de cadenas")
    required_notes = {"source:prestandardized"} if prestandardized else {
        "parsed_by_llm",
        "source:mistral_cache" if from_cache else "source:mistral_live",
    }
    missing_notes = required_notes.difference(notes)
    if missing_notes:
        raise ValueError(
            "faltan marcas de procedencia en notes: "
            + ", ".join(sorted(missing_notes))
        )
    forbidden_note_markers = ("fallback", "adapter", "llm_failed", "llm_error")
    if any(
        marker in note.strip().casefold()
        for note in notes
        for marker in forbidden_note_markers
    ):
        raise ValueError("notes contiene una marca de fallo o fallback")
    if "security_boundary:sanitized" in notes:
        raise ValueError(
            "el runtime no puede declarar una sanitizacion que no realiza"
        )
    if not from_cache and "source:mistral_cache" in notes:
        raise ValueError("una respuesta en vivo no puede declararse como cache")

    cache_content_hash = result.get("cache_content_hash")
    cache_pipeline_hash = result.get("cache_pipeline_hash")
    if from_cache and not _is_sha256(cache_content_hash):
        raise ValueError("un cache hit requiere cache_content_hash SHA-256")
    if from_cache and not _is_sha256(cache_pipeline_hash):
        raise ValueError("un cache hit requiere cache_pipeline_hash SHA-256")
    if cache_content_hash is not None and not _is_sha256(cache_content_hash):
        raise ValueError("cache_content_hash debe ser SHA-256 cuando se declara")
    if cache_pipeline_hash is not None and not _is_sha256(cache_pipeline_hash):
        raise ValueError("cache_pipeline_hash debe ser SHA-256 cuando se declara")
    if not prestandardized and cache_content_hash is not None:
        try:
            expected_content_hash = compute_content_hash(
                row=row if row is not None else None,
                text=raw_input.get("text") if row is None else None,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("no se puede verificar cache_content_hash") from exc
        if cache_content_hash != expected_content_hash:
            raise ValueError("cache_content_hash no coincide con la entrada actual")

    canonical = result.get("canonical_event")
    if not isinstance(canonical, dict):
        raise ValueError("canonical_event debe ser un objeto")
    unknown_canonical_fields = sorted(
        set(canonical).difference(CanonicalEvent.model_fields)
    )
    if unknown_canonical_fields:
        raise ValueError(
            "canonical_event contiene campos fuera del contrato: "
            + ", ".join(unknown_canonical_fields)
        )
    validated_event = CanonicalEvent(**canonical)
    validated_canonical = validated_event.model_dump(mode="json")
    validate_target_free_canonical(validated_canonical)
    if not prestandardized:
        dataset = str(raw_input.get("dataset") or "generic").strip()
        source_file = raw_input.get("source_file", "api")
        source_ref = "inline" if source_file is None else str(source_file)
        row_id = raw_input.get("row_id", 0)
        expected_event_id = f"llm::{dataset}::{source_ref}::{row_id}"
        if validated_event.event_id != expected_event_id:
            raise ValueError(
                "la estandarizacion no reconstruyo event_id para el registro actual"
            )
        origin = validated_event.origin or {}
        if (
            str(origin.get("source_name") or "") != dataset
            or str(origin.get("source_file") or "") != source_ref
            or origin.get("row_id") != row_id
        ):
            raise ValueError(
                "la estandarizacion no reconstruyo origin para el registro actual"
            )
        provenance = validated_event.provenance
        expected_split = raw_input.get("split", "stream")
        if (
            provenance.dataset != dataset
            or provenance.source_file != source_file
            or provenance.row_id != row_id
            or provenance.split != expected_split
        ):
            raise ValueError(
                "la estandarizacion no reconstruyo provenance para el registro actual"
            )

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
        "canonical_event": validated_canonical,
        "mapping_confidence": mapping_confidence,
        "selected_columns": list(selected_columns),
        "notes": list(notes),
        "source": expected_source,
        "provider": provider,
        "model": model,
        "from_cache": from_cache,
        "cache_content_hash": cache_content_hash,
        "cache_pipeline_hash": cache_pipeline_hash,
    }
