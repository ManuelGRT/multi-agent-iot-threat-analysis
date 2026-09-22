"""Preparacion offline sin targets para entrenamiento, validacion y evaluacion.

No se importa desde el grafo multiagente ni desde las tools MCP operacionales.
El sistema final recibe artefactos ya limpios y rechaza contratos invalidos en
lugar de reparar o eliminar campos durante la inferencia.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from src.eval.predictive_sanitization import (
    is_predictive_target_field,
    is_predictive_target_value,
    scrub_predictive_text,
)
from src.contracts.canonical import CanonicalEvent


# ``attack_indicators`` contiene patrones tecnicos derivados, no etiquetas de
# verdad terreno. El auditor lo permite y la featurizacion lo excluye de forma
# explicita; se conserva la clave, aunque se limpian sus textos.
ALLOWED_CANONICAL_TARGET_LIKE_PATHS = frozenset(
    {
        # Conceptos de protocolo/esquema cuyo nombre se parece a un target.
        # Las excepciones son deliberadamente sensibles a la ruta.
        "attack_indicators",
        "service_context.protocol_family",
        "service_context.dns.query_class",
        "telemetry_context.dns.query_class",
        "service_context.dns.query.type",
        "service_context.mqtt.message.type",
        "telemetry_context.mqtt_message.type",
        "service_context.icmp.type",
        "service_context.dns.type",
        "service_context.dns.qry.type",
        "telemetry_context.iot_sensor.type",
        "telemetry_context.dns.query.type",
        "telemetry_context.sensor_data.type",
        "host_context.interface.type",
        "telemetry_context.geolocation.type",
        "service_context.process.type",
        "feature_groups.other.execution_class",
        "host_context.metric_category",
    }
)


def is_allowed_canonical_target_path(path: str) -> bool:
    return path.casefold() in ALLOWED_CANONICAL_TARGET_LIKE_PATHS


_ALLOWED_RAW_TECHNICAL_FIELDS = frozenset(
    {
        "protocol_family",
        "query_class",
        "metric_category",
        "execution_class",
        "dns_query_type",
        "dns_qry_type",
        "mqtt_message_type",
        "icmp_type",
        "process_type",
        "interface_type",
        "iot_sensor_type",
        "sensor_data_type",
        "geolocation_type",
    }
)


def is_allowed_raw_technical_field(field: Any) -> bool:
    normalized = re.sub(
        r"[^a-z0-9]+", "_", str(field or "").strip().casefold()
    ).strip("_")
    return normalized in _ALLOWED_RAW_TECHNICAL_FIELDS


def _is_allowed_raw_target_path(path: str) -> bool:
    field = (
        path.removeprefix("row.")
        if path.startswith("row.")
        else path.rsplit(".", 1)[-1]
    )
    return is_allowed_raw_technical_field(field)


def _sanitize_text(
    text: str,
    path: str,
    *,
    allow_target_path: Callable[[str], bool],
    filter_target_value_path: Callable[[str], bool],
) -> str:
    """Sanea JSON textual con la misma politica sensible a la ruta."""
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            pass
        else:
            if isinstance(parsed, (dict, list)):
                cleaned = _sanitize_value(
                    parsed,
                    path,
                    allow_target_path=allow_target_path,
                    filter_target_value_path=filter_target_value_path,
                )
                if cleaned == parsed:
                    return text
                return json.dumps(
                    cleaned,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
    return scrub_predictive_text(text)


def _sanitize_value(
    value: Any,
    path: str = "",
    *,
    allow_target_path: Callable[[str], bool],
    filter_target_value_path: Callable[[str], bool],
) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            if (
                is_predictive_target_field(key_text)
                and not allow_target_path(child_path)
            ):
                continue
            if filter_target_value_path(child_path) and is_predictive_target_value(
                item
            ):
                continue
            cleaned_item = _sanitize_value(
                item,
                child_path,
                allow_target_path=allow_target_path,
                filter_target_value_path=filter_target_value_path,
            )
            cleaned[key_text] = cleaned_item
        return cleaned
    if isinstance(value, (list, tuple)):
        cleaned_items: list[Any] = []
        for item in value:
            # feature_groups/evidence_fields contienen nombres de columnas como
            # valores, por lo que tambien requieren una comprobacion nominal.
            if isinstance(item, str) and (
                path == "evidence_fields" or path.startswith("feature_groups.")
            ):
                item_path = f"{path}.{item}"
                if (
                    is_predictive_target_field(item)
                    and not allow_target_path(item_path)
                ):
                    continue
            if filter_target_value_path(path) and is_predictive_target_value(item):
                continue
            cleaned_items.append(
                _sanitize_value(
                    item,
                    path,
                    allow_target_path=allow_target_path,
                    filter_target_value_path=filter_target_value_path,
                )
            )
        return cleaned_items
    if isinstance(value, str):
        return _sanitize_text(
            value,
            path,
            allow_target_path=allow_target_path,
            filter_target_value_path=filter_target_value_path,
        )
    return value


def _is_raw_target_value_path(path: str) -> bool:
    root = path.split(".", 1)[0].casefold()
    return root in {"row", "text"}


def _is_canonical_target_value_path(path: str) -> bool:
    root = path.split(".", 1)[0].casefold()
    return root in {
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


def sanitize_llm_input(raw_input: dict[str, Any]) -> dict[str, Any]:
    """Devuelve una copia sin targets apta para enviarse al LLM.

    A diferencia del antiguo saneamiento dentro de ``FinalStandardizer``, esta
    funcion tambien protege entradas textuales antes de la llamada remota.
    """
    cleaned = _sanitize_value(
        raw_input,
        allow_target_path=_is_allowed_raw_target_path,
        filter_target_value_path=_is_raw_target_value_path,
    )
    if not isinstance(cleaned, dict):  # defensivo ante futuros cambios
        raise TypeError("la entrada del LLM debe ser un objeto JSON")
    return cleaned


def sanitize_canonical_event(event_data: dict[str, Any] | CanonicalEvent) -> dict[str, Any]:
    """Valida y limpia un evento canonico sin mutar el objeto de entrada."""
    payload = (
        event_data.model_dump(mode="json")
        if isinstance(event_data, CanonicalEvent)
        else dict(event_data)
    )
    cleaned = _sanitize_value(
        payload,
        allow_target_path=is_allowed_canonical_target_path,
        filter_target_value_path=_is_canonical_target_value_path,
    )

    # CanonicalEvent repone los tres campos legacy como ``None``. Se mantienen
    # por compatibilidad de contrato, nunca con valores utilizables.
    event = CanonicalEvent(**cleaned)
    return event.model_dump(mode="json")
