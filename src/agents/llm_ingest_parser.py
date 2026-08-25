# src/agents/llm_ingest_parser.py
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import os
import re
from typing import Any

from src.agents.base import (
    GoogleAIStudioChatAgent,
    GroqChatAgent,
    MistralChatAgent,
    OllamaChatAgent,
    OpenRouterChatAgent,
    TransformersChatAgent,
)
from src.agents.predictive_sanitization import is_predictive_target_field, scrub_predictive_payload
from src.contracts.canonical import CanonicalEvent, Provenance


CANONICAL_FIELDS = {
    "event_id",
    "modality",
    "ts",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "transport_proto",
    "app_proto",
    "packet_count",
    "byte_count",
    "duration_ms",
    "telemetry",
    "host",
    "severity",
    "schema_profile",
    "origin",
    "feature_groups",
    "evidence_fields",
    "traffic_direction",
    "service_context",
    "host_context",
    "telemetry_context",
    "anomaly_summary",
    "semantic_text",
    "provenance",
    "mapping_confidence",
}


LLM_INGEST_SYSTEM = """
Eres un agente de ingesta para ciberseguridad IoT/IIoT.
Convierte una fila cruda, log o alerta en un evento canonico JSON.
Analiza los nombres de columnas y sus valores. No asumas que el dataset es conocido.
Primero infiere el perfil de esquema y despues mapea los campos relevantes.

Reglas de mapeo generico:
- src_ip: identifica la direccion IP de origen por el significado de la columna, su contexto y el formato del valor.
- dst_ip: identifica la direccion IP de destino por el significado de la columna, su contexto y el formato del valor.
- src_port: identifica el puerto de origen por el significado de la columna, su contexto y el rango numerico del valor.
- dst_port: identifica el puerto de destino por el significado de la columna, su contexto y el rango numerico del valor.
- transport_proto: identifica el protocolo de transporte desde campos que describan protocolo o desde indicadores de trafico asociados.
- modality: network_flow para trafico/red; telemetry solo para sensores/mediciones fisicas; host_log para procesos, memoria, CPU, disco, metricas de sistema operativo o rendimiento de endpoint.
- No uses telemetry para metricas de sistema operativo, CPU, memoria, procesos, disco o interfaces de red del host; en esos casos usa host_log.
- schema_profile debe ser uno de: network_flow, network_packet, host_metrics, iot_telemetry, alert_text, pcap_ref, unknown.
- Usa network_flow para flujos agregados de comunicaciones; network_packet para cabeceras/protocolos de paquetes; host_metrics para senales de endpoint; iot_telemetry para sensores fisicos; alert_text para logs o alertas textuales.
- No generes etiquetas, familias ni subtipos de ataque. Esos son objetivos de otros agentes y no forman parte del evento canonico de entrada.
- origin es metadato de trazabilidad y se completara fuera del razonamiento si no aparece en la entrada. No uses origin como regla de decision.
- feature_groups debe agrupar columnas por funcion tecnica generica: network_endpoint, ports, protocol, network_volume, state_flags, host_metrics, iot_telemetry, textual_alert_context, timing, statistics u other.
- evidence_fields debe listar columnas tecnicas utiles para decidir; excluye etiquetas o anotaciones del dataset.
- Los valores tecnicos seleccionados que no encajen en un campo canonico directo deben preservarse en telemetry o host con su nombre de columna original.
- No dejes telemetry y host vacios si existen senales tecnicas seleccionadas no etiquetadas.
- service_context, host_context y telemetry_context deben resumir senales tecnicas interpretables sin reglas por dataset.
- anomaly_summary debe describir de forma breve que patron tecnico queda representado.

No inventes datos tecnicos: usa null cuando un campo no aparezca.
Incluye SIEMPRE todas las claves del schema aunque su valor sea null, [] o {}.
semantic_text debe resumir la evidencia conservando valores relevantes.
mapping_confidence debe ser calibrada:
- 0.85-1.0 si identificas IPs/puertos/protocolo o senales tecnicas claras del perfil.
- 0.50-0.84 si identificas modalidad, pero faltan campos tecnicos importantes.
- menor de 0.50 si faltan campos esenciales o no entiendes la fila.
missing_fields debe listar los campos canonicos importantes que no pudiste mapear.
Responde exclusivamente con JSON valido.
"""


LLM_COLUMN_SELECTION_SYSTEM = """
Eres un agente de preseleccion de columnas para ingesta de ciberseguridad IoT/IIoT.
Recibiras una fila ancha descrita por nombres de columnas y vistas previas de valores.
Primero infiere el perfil de esquema y despues selecciona solo las columnas utiles para construir un evento canonico posterior.

Prioriza columnas que puedan aportar:
- tiempo del evento.
- origen y destino de comunicacion.
- puertos, protocolo, paquetes, bytes y duracion.
- severidad operacional cuando exista en una alerta.
- senales claras de telemetria fisica o estado de host si no hay trafico de red.
- columnas de CPU, memoria, procesos, disco, sistema operativo o rendimiento de endpoint apuntan a host_metrics/host_log, no a iot_telemetry.

No selecciones anotaciones del dataset: etiquetas, clases, categorias, tipos de ataque, familias, subtipos o labels.
No selecciones columnas por el nombre del origen: usa el perfil de esquema inferido, los nombres de columna y los valores.
No selecciones columnas solo porque tengan valores numericos: deben aportar evidencia semantica.
No inventes nombres de columnas: devuelve solo nombres que existan en la entrada.
Respeta max_selected_columns salvo que sea imprescindible incluir tiempo o campos tecnicos criticos.
Responde exclusivamente con JSON valido.
"""


COLUMN_SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_columns": {"type": "array", "items": {"type": "string"}},
        "modality_guess": {"type": ["string", "null"]},
        "schema_profile_guess": {"type": ["string", "null"]},
        "rationale": {"type": "string"},
    },
    "required": ["selected_columns", "modality_guess", "schema_profile_guess", "rationale"],
}


CANONICAL_EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "event_id": {"type": "string"},
        "modality": {"type": "string", "enum": ["network_flow", "telemetry", "host_log", "alert", "pcap_ref"]},
        "ts": {"type": ["string", "null"]},
        "src_ip": {"type": ["string", "null"]},
        "dst_ip": {"type": ["string", "null"]},
        "src_port": {"type": ["integer", "null"]},
        "dst_port": {"type": ["integer", "null"]},
        "transport_proto": {"type": ["string", "null"]},
        "app_proto": {"type": ["string", "null"]},
        "packet_count": {"type": ["integer", "null"]},
        "byte_count": {"type": ["integer", "null"]},
        "duration_ms": {"type": ["number", "null"]},
        "telemetry": {"type": "object"},
        "host": {"type": "object"},
        "severity": {"type": ["string", "null"], "enum": ["low", "medium", "high", "critical", None]},
        "schema_profile": {
            "type": ["string", "null"],
            "enum": ["network_flow", "network_packet", "host_metrics", "iot_telemetry", "alert_text", "pcap_ref", "unknown", None],
        },
        "origin": {"type": "object"},
        "feature_groups": {
            "type": "object",
            "additionalProperties": {"type": "array", "items": {"type": "string"}},
        },
        "evidence_fields": {"type": "array", "items": {"type": "string"}},
        "traffic_direction": {"type": ["string", "null"]},
        "service_context": {"type": "object"},
        "host_context": {"type": "object"},
        "telemetry_context": {"type": "object"},
        "anomaly_summary": {"type": ["string", "null"]},
        "semantic_text": {"type": "string"},
        "provenance": {
            "type": "object",
            "properties": {
                "dataset": {"type": "string"},
                "source_file": {"type": ["string", "null"]},
                "row_id": {"type": ["string", "integer", "null"]},
                "split": {"type": "string"},
                "parser_version": {"type": "string"},
            },
            "required": ["dataset"],
        },
        "mapping_confidence": {"type": "number"},
        "missing_fields": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "event_id",
        "modality",
        "ts",
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "transport_proto",
        "app_proto",
        "packet_count",
        "byte_count",
        "duration_ms",
        "telemetry",
        "host",
        "severity",
        "schema_profile",
        "origin",
        "feature_groups",
        "evidence_fields",
        "traffic_direction",
        "service_context",
        "host_context",
        "telemetry_context",
        "anomaly_summary",
        "semantic_text",
        "provenance",
        "mapping_confidence",
        "missing_fields",
    ],
}


class LLMIngestParser:
    def __init__(
        self,
        model: str = "gemma3:12b",
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float | None = None,
        enable_column_selection: bool = True,
        column_selection_threshold: int = 80,
        max_selected_columns: int = 40,
        column_selection_timeout_seconds: float | None = None,
        provider: str | None = None,
    ):
        provider_name = (provider or os.getenv("LLM_PROVIDER", "ollama")).strip().lower()
        if provider_name == "openrouter":
            self.agent = OpenRouterChatAgent(
                model=model,
                base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                timeout_seconds=timeout_seconds,
            )
        elif provider_name == "groq":
            self.agent = GroqChatAgent(
                model=model,
                base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
                timeout_seconds=timeout_seconds,
            )
        elif provider_name == "mistral":
            self.agent = MistralChatAgent(
                model=model,
                base_url=os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1"),
                timeout_seconds=timeout_seconds,
            )
        elif provider_name in {"google", "gemini", "google_ai_studio"}:
            self.agent = GoogleAIStudioChatAgent(
                model=model,
                base_url=os.getenv("GOOGLE_AI_STUDIO_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"),
                timeout_seconds=timeout_seconds,
            )
        elif provider_name == "transformers":
            self.agent = TransformersChatAgent(
                model=model,
                adapter_path=os.getenv("INGEST_LORA_ADAPTER") or os.getenv("TRANSFORMERS_ADAPTER_PATH"),
                timeout_seconds=timeout_seconds,
            )
        else:
            self.agent = OllamaChatAgent(model=model, base_url=base_url, timeout_seconds=timeout_seconds)
        self.enable_column_selection = enable_column_selection
        self.column_selection_threshold = column_selection_threshold
        self.max_selected_columns = max_selected_columns
        if column_selection_timeout_seconds is None:
            env_timeout = os.getenv("LLM_COLUMN_SELECTION_TIMEOUT_SECONDS")
            column_selection_timeout_seconds = float(env_timeout) if env_timeout else None
        self.column_selection_timeout_seconds = column_selection_timeout_seconds
        self.last_column_selection: dict[str, Any] | None = None

    async def parse(self, raw_input: dict[str, Any]) -> CanonicalEvent:
        selected_columns = None
        llm_input = raw_input
        selection_metadata = None
        self.last_column_selection = None
        if self._should_select_columns(raw_input):
            selected_columns = await self._select_relevant_columns(raw_input)
            selection_metadata = self._selection_metadata(raw_input, selected_columns)
            self.last_column_selection = selection_metadata
            llm_input = self._filter_raw_input(raw_input, selected_columns)

        payload = await self.agent.invoke_json(
            system_prompt=LLM_INGEST_SYSTEM,
            user_payload=self._compact_input(
                llm_input,
                selection_metadata=selection_metadata,
            ),
            json_schema=CANONICAL_EVENT_SCHEMA,
        )
        payload = self._complete_payload(payload, llm_input)
        return CanonicalEvent(**payload)

    def _should_select_columns(self, raw_input: dict[str, Any]) -> bool:
        if not self.enable_column_selection:
            return False
        row = raw_input.get("row")
        if not isinstance(row, dict):
            return False
        flag = raw_input.get("use_column_selection")
        if flag is True:
            return True
        if flag is False:
            return False
        return len(row) >= self.column_selection_threshold

    async def _select_relevant_columns(self, raw_input: dict[str, Any]) -> list[str]:
        row = raw_input.get("row")
        if not isinstance(row, dict):
            return []

        try:
            selection_call = self.agent.invoke_json(
                system_prompt=LLM_COLUMN_SELECTION_SYSTEM,
                user_payload=self._column_selection_input(raw_input),
                json_schema=COLUMN_SELECTION_SCHEMA,
            )
            if self.column_selection_timeout_seconds and self.column_selection_timeout_seconds > 0:
                payload = await asyncio.wait_for(selection_call, timeout=self.column_selection_timeout_seconds)
            else:
                payload = await selection_call
            selected = payload.get("selected_columns", [])
        except Exception:
            selected = []

        columns = self._sanitize_selected_columns(selected, row)
        if not columns:
            columns = self._fallback_relevant_columns(row)
        return self._limit_selected_columns(columns, row)

    def _column_selection_input(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        row = raw_input.get("row") or {}
        columns = [key for key in row.keys() if not self._is_label_like_key(key)] if isinstance(row, dict) else []
        return {
            "total_columns": len(row) if isinstance(row, dict) else len(columns),
            "max_selected_columns": self.max_selected_columns,
            "columns": [str(key) for key in columns],
            "value_previews": {
                str(key): self._value_preview(row.get(key))
                for key in columns
                if not self._is_empty_or_zero(row.get(key))
            },
            "canonical_output_fields": [
                "origin",
                "schema_profile",
                "feature_groups",
                "evidence_fields",
                "traffic_direction",
                "service_context",
                "host_context",
                "telemetry_context",
                "anomaly_summary",
                "ts",
                "src_ip",
                "dst_ip",
                "src_port",
                "dst_port",
                "transport_proto",
                "app_proto",
                "packet_count",
                "byte_count",
                "duration_ms",
                "telemetry",
                "host",
                "severity",
            ],
        }

    def _value_preview(self, value: Any, limit: int = 80) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if len(text) <= limit:
            return text
        return text[:limit] + "..."

    def _sanitize_selected_columns(self, selected: Any, row: dict[str, Any]) -> list[str]:
        if not isinstance(selected, list):
            selected = []
        available = {str(key): key for key in row}
        ordered = []
        for column in selected:
            key = available.get(str(column))
            if key is not None and key not in ordered and not self._is_label_like_key(key):
                ordered.append(key)

        for key in self._always_keep_columns(row):
            if key not in ordered:
                ordered.append(key)
        return [str(key) for key in ordered]

    def _limit_selected_columns(self, columns: list[str], row: dict[str, Any]) -> list[str]:
        if len(columns) <= self.max_selected_columns:
            return columns
        always_keep = [str(key) for key in self._always_keep_columns(row)]
        limited = []
        for column in always_keep + columns:
            if column in columns and column not in limited:
                limited.append(column)
            if len(limited) >= self.max_selected_columns:
                break
        return limited

    def _always_keep_columns(self, row: dict[str, Any]) -> list[str]:
        time_tokens = {
            "ts",
            "date",
            "time",
            "timestamp",
            "datetime",
            "frame_time",
            "created_at",
            "observed_at",
            "stime",
            "ltime",
        }
        keep = []
        for key in row:
            normalized = str(key).lower()
            normalized_token = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
            if normalized_token in time_tokens or normalized_token == "severity":
                keep.append(key)
        return keep

    def _fallback_relevant_columns(self, row: dict[str, Any]) -> list[str]:
        priority_tokens = (
            "src",
            "source",
            "orig",
            "dst",
            "dest",
            "resp",
            "ip",
            "port",
            "proto",
            "packet",
            "pkt",
            "byte",
            "duration",
            "severity",
            "date",
            "time",
            "timestamp",
            "process",
            "memory",
            "cpu",
            "processor",
            "disk",
            "sensor",
            "temp",
            "humidity",
            "pressure",
        )
        scored = []
        for index, (key, value) in enumerate(row.items()):
            normalized = str(key).lower()
            score = sum(1 for token in priority_tokens if token in normalized)
            if not self._is_empty_or_zero(value):
                score += 1
            if score:
                scored.append((score, -index, str(key)))
        scored.sort(reverse=True)
        return [key for _, _, key in scored[: self.max_selected_columns]]

    def _filter_raw_input(self, raw_input: dict[str, Any], selected_columns: list[str]) -> dict[str, Any]:
        row = raw_input.get("row")
        if not isinstance(row, dict) or not selected_columns:
            return raw_input
        selected = set(selected_columns)
        filtered = {
            key: value
            for key, value in row.items()
            if str(key) in selected and not self._is_label_like_key(key)
        }
        copy = dict(raw_input)
        copy["row"] = filtered
        copy["selected_columns"] = list(filtered.keys())
        copy["source_column_count"] = len(row)
        return copy

    def _selection_metadata(self, raw_input: dict[str, Any], selected_columns: list[str] | None) -> dict[str, Any] | None:
        row = raw_input.get("row")
        if not isinstance(row, dict) or selected_columns is None:
            return None
        return {
            "source_column_count": len(row),
            "selected_column_count": len(selected_columns),
            "selected_columns": selected_columns,
        }

    def _compact_input(
        self,
        raw_input: dict[str, Any],
        selection_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = raw_input.get("row")
        if not isinstance(row, dict):
            return raw_input

        meaningful = {
            str(key): value
            for key, value in row.items()
            if not self._is_empty_or_zero(value) and not self._is_label_like_key(key)
        }
        columns = [key for key in row.keys() if not self._is_label_like_key(key)]
        compacted = {
            "columns": [str(key) for key in columns],
            "meaningful_values": scrub_predictive_payload(meaningful),
            "schema_profiles": [
                "network_flow",
                "network_packet",
                "host_metrics",
                "iot_telemetry",
                "alert_text",
                "pcap_ref",
                "unknown",
            ],
            "instruction": "Infer schema_profile and map meaningful_values to the canonical cybersecurity event schema without generating attack labels, families or subtypes.",
        }
        if selection_metadata is not None:
            compacted["column_selection"] = selection_metadata
        return compacted

    def _is_empty_or_zero(self, value: Any) -> bool:
        if value is None:
            return True
        text = str(value).strip()
        if text in {"", "-", "nan", "NaN", "None", "null"}:
            return True
        try:
            return float(text) == 0.0
        except ValueError:
            return False

    def _complete_payload(self, payload: dict[str, Any], raw_input: dict[str, Any]) -> dict[str, Any]:
        payload = self._clean_empty_values(payload)
        for target_field in ("label_raw", "attack_family", "attack_subtype"):
            payload.pop(target_field, None)
        incoming_provenance = payload.get("provenance")
        if not isinstance(incoming_provenance, dict):
            incoming_provenance = {}
        origin = self._origin_from_raw(raw_input, incoming_provenance=incoming_provenance)
        dataset = str(origin.get("source_name") or incoming_provenance.get("dataset") or "generic")
        source_file = raw_input.get("source_file", incoming_provenance.get("source_file", "llm"))
        row_id = raw_input.get("row_id", incoming_provenance.get("row_id", 0))
        event_id = payload.get("event_id")
        if str(event_id or "").strip().lower() in {"", "0", "default_event_id", "event_id"}:
            payload["event_id"] = f"llm::{dataset}::{source_file}::{row_id}"
        else:
            payload["event_id"] = str(event_id)
        payload["modality"] = self._normalize_modality(payload.get("modality"), raw_input)
        if not payload.get("semantic_text"):
            payload["semantic_text"] = str(raw_input)
        if payload.get("mapping_confidence") is None:
            payload["mapping_confidence"] = 0.5
        payload["schema_profile"] = self._normalize_schema_profile(payload.get("schema_profile"), payload, raw_input)
        payload["origin"] = self._normalize_origin(payload.get("origin"), origin, payload["schema_profile"])
        payload["feature_groups"] = self._normalize_feature_groups(payload.get("feature_groups"), raw_input)
        payload["evidence_fields"] = self._normalize_string_list(
            payload.get("evidence_fields"),
            fallback=self._infer_evidence_fields(raw_input),
        )
        payload["traffic_direction"] = self._normalize_optional_text(payload.get("traffic_direction"))
        payload["service_context"] = self._compact_object(payload.get("service_context"))
        payload["host_context"] = self._compact_object(payload.get("host_context"))
        payload["telemetry_context"] = self._compact_object(payload.get("telemetry_context"))
        payload["anomaly_summary"] = self._normalize_optional_text(payload.get("anomaly_summary"))
        payload["mapping_confidence"] = self._calibrate_mapping_confidence(
            payload.get("mapping_confidence"),
            payload,
        )
        if not isinstance(payload.get("missing_fields"), list):
            payload["missing_fields"] = []
        payload.setdefault("telemetry", {})
        payload.setdefault("host", {})
        payload["ts"] = self._normalize_timestamp(payload.get("ts"), raw_input)
        payload["telemetry"] = scrub_predictive_payload(self._compact_object(payload.get("telemetry"), primitive_only=True))
        payload["host"] = scrub_predictive_payload(self._compact_object(payload.get("host")))
        self._preserve_unmapped_technical_values(payload, raw_input)
        payload["semantic_text"] = scrub_predictive_payload(payload.get("semantic_text") or "")
        payload["feature_groups"] = scrub_predictive_payload(payload.get("feature_groups") or {})
        payload["evidence_fields"] = scrub_predictive_payload(payload.get("evidence_fields") or [])
        for key in ("src_port", "dst_port", "packet_count", "byte_count"):
            payload[key] = self._normalize_optional_int(payload.get(key))
        payload["duration_ms"] = self._normalize_optional_float(payload.get("duration_ms"))
        payload["missing_fields"] = [
            field for field in payload.get("missing_fields", [])
            if field in CANONICAL_FIELDS and self._field_is_missing(payload, field)
        ]
        if payload.get("ts") is None and "ts" not in payload["missing_fields"]:
            payload["missing_fields"].append("ts")
        payload["mapping_confidence"] = max(0.0, min(1.0, float(payload["mapping_confidence"])))
        provenance = payload.get("provenance") or {}
        if not isinstance(provenance, dict):
            provenance = {}
        provenance_dataset = str(provenance.get("dataset") or "").strip().lower()
        if provenance_dataset in {"", "generic", "unknown", "none", "null"}:
            provenance["dataset"] = dataset
        provenance.setdefault("source_file", source_file)
        provenance.setdefault("row_id", row_id)
        if provenance.get("split") not in {"train", "val", "test", "stream"}:
            provenance["split"] = "stream"
        if not isinstance(provenance.get("parser_version"), str) or not provenance.get("parser_version"):
            provenance["parser_version"] = "llm-0.1.0"
        payload["provenance"] = Provenance(**provenance).model_dump(mode="json")
        return payload

    def _preserve_unmapped_technical_values(self, payload: dict[str, Any], raw_input: dict[str, Any]) -> None:
        row = raw_input.get("row")
        if not isinstance(row, dict):
            return
        telemetry = payload.get("telemetry")
        host = payload.get("host")
        if not isinstance(telemetry, dict):
            telemetry = {}
        if not isinstance(host, dict):
            host = {}
        mapped_names = {str(key) for key in telemetry} | {str(key) for key in host}
        for key, value in row.items():
            key_text = str(key)
            if key_text in mapped_names or self._is_blank(value) or self._is_label_like_key(key_text):
                continue
            group = self._generic_feature_group(key_text)
            target = host if payload.get("modality") == "host_log" or group == "host_metrics" else telemetry
            target[key_text] = self._primitive_feature_value(value)
            mapped_names.add(key_text)
            if len(telemetry) + len(host) >= 120:
                break
        payload["telemetry"] = scrub_predictive_payload(self._compact_object(telemetry, primitive_only=True))
        payload["host"] = scrub_predictive_payload(self._compact_object(host))

    @staticmethod
    def _primitive_feature_value(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            if isinstance(value, str) and len(value) > 300:
                return value[:300]
            return value
        text = str(value)
        return text[:300] if len(text) > 300 else text

    def _origin_from_raw(
        self,
        raw_input: dict[str, Any],
        incoming_provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        incoming_provenance = incoming_provenance or {}
        raw_origin = raw_input.get("origin")
        origin: dict[str, Any] = dict(raw_origin) if isinstance(raw_origin, dict) else {}
        source_name = origin.get("source_name") or origin.get("dataset")
        if source_name is None and isinstance(raw_origin, str):
            source_name = raw_origin
        source_name = (
            source_name
            or raw_input.get("dataset_family")
            or raw_input.get("source_dataset")
            or raw_input.get("dataset")
            or incoming_provenance.get("dataset")
            or "generic"
        )
        origin["source_name"] = str(source_name)
        origin["source_file"] = str(raw_input.get("source_file", incoming_provenance.get("source_file", "inline")))
        origin["row_id"] = raw_input.get("row_id", incoming_provenance.get("row_id", 0))
        return origin

    def _normalize_origin(self, value: Any, fallback: dict[str, Any], schema_profile: str | None) -> dict[str, Any]:
        origin = dict(value) if isinstance(value, dict) else {}
        for key, item in fallback.items():
            origin.setdefault(key, item)
        if schema_profile:
            origin["schema_profile"] = schema_profile
        return {
            str(key): item
            for key, item in origin.items()
            if item is not None and not self._is_blank(item)
        }

    def _normalize_feature_groups(self, value: Any, raw_input: dict[str, Any]) -> dict[str, list[str]]:
        if not isinstance(value, dict):
            return self._infer_feature_groups(raw_input)
        groups: dict[str, list[str]] = {}
        for key, item in value.items():
            if isinstance(item, list):
                columns = [str(column) for column in item if not self._is_blank(column)]
            elif self._is_blank(item):
                columns = []
            else:
                columns = [str(item)]
            if columns:
                groups[str(key)] = sorted(set(columns))[:40]
        return groups or self._infer_feature_groups(raw_input)

    def _normalize_string_list(self, value: Any, fallback: list[str] | None = None) -> list[str]:
        if not isinstance(value, list):
            return fallback or []
        normalized = [str(item) for item in value if not self._is_blank(item)]
        return sorted(set(normalized))[:80] or (fallback or [])

    def _normalize_optional_text(self, value: Any) -> str | None:
        if self._is_blank(value):
            return None
        return str(value).strip()

    def _normalize_optional_int(self, value: Any) -> int | None:
        if self._is_blank(value):
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value) if value.is_integer() else None
        if isinstance(value, str):
            cleaned = value.strip().replace(",", "")
            if re.fullmatch(r"[-+]?\d+", cleaned):
                return int(cleaned)
            if re.fullmatch(r"[-+]?\d+\.0+", cleaned):
                return int(float(cleaned))
        return None

    def _normalize_optional_float(self, value: Any) -> float | None:
        if self._is_blank(value):
            return None
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip().replace(",", "")
            if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", cleaned):
                return float(cleaned)
        return None

    def _infer_feature_groups(self, raw_input: dict[str, Any]) -> dict[str, list[str]]:
        row = raw_input.get("row")
        if not isinstance(row, dict):
            return {}
        groups: dict[str, list[str]] = {}
        for key, value in row.items():
            if self._is_blank(value) or self._is_label_like_key(key):
                continue
            group = self._generic_feature_group(str(key))
            groups.setdefault(group, []).append(str(key))
        return {
            group: sorted(set(columns))[:40]
            for group, columns in sorted(groups.items())
            if columns
        }

    def _infer_evidence_fields(self, raw_input: dict[str, Any]) -> list[str]:
        groups = self._infer_feature_groups(raw_input)
        fields = []
        for group, columns in groups.items():
            if group != "other":
                fields.extend(columns)
        return sorted(set(fields))[:80]

    def _generic_feature_group(self, feature: str) -> str:
        raw = feature.lower()
        tokens = [token for token in re.split(r"[^a-z0-9]+", raw) if token]
        token_set = set(tokens)
        compact = "".join(tokens)
        name = " ".join(tokens)
        if any(token in name for token in ("temperature", "thermostat", "sensor", "humidity", "pressure", "water", "weather")):
            return "iot_telemetry"
        if any(token in name for token in ("cpu", "processor", "process", "memory", "disk", "thread", "handle", "pid", "cmd")):
            return "host_metrics"
        if any(token in name for token in ("message", "description", "rule", "alert", "raw", "log", "signature")):
            return "textual_alert_context"
        if any(token in name for token in ("byte", "payload", "tcp len", "frame len", "sbytes", "dbytes", "pkt", "packet", "rate")):
            return "network_volume"
        if any(token in name for token in ("proto", "protocol", "service", "http", "dns", "mqtt", "modbus", "ssl", "tls")):
            return "protocol"
        if (
            any(token in compact for token in ("srcport", "sourceport", "dstport", "destport", "destinationport"))
            or ({"src", "port"} <= token_set)
            or ({"dst", "port"} <= token_set)
            or ({"source", "port"} <= token_set)
            or ({"destination", "port"} <= token_set)
        ):
            return "ports"
        if any(token in name for token in ("src", "source", "orig", "saddr", "dst", "dest", "resp", "daddr")):
            return "network_endpoint"
        if any(token in name for token in ("duration", "dur", "time", "timestamp", "date")):
            return "timing"
        if any(token in name for token in ("status", "state", "flag", "flgs", "history", "conn", "ack", "seq")):
            return "state_flags"
        if any(token in name for token in ("mean", "stddev", "sum", "min", "max", "magnitude", "radius", "covariance")):
            return "statistics"
        return "other"

    def _is_label_like_key(self, key: Any) -> bool:
        return is_predictive_target_field(key)

    def _clean_empty_values(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._clean_empty_values(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._clean_empty_values(item) for item in value]
        if self._is_blank(value):
            return None
        if isinstance(value, str):
            return value.strip()
        return value

    def _compact_object(self, value: Any, primitive_only: bool = False) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        compacted = {}
        for key, item in value.items():
            if item is None or self._is_blank(item):
                continue
            if primitive_only and isinstance(item, (dict, list)):
                item = json.dumps(item, ensure_ascii=True, sort_keys=True, default=str)
            compacted[str(key)] = item
        return compacted

    def _normalize_timestamp(self, value: Any, raw_input: dict[str, Any]) -> str | None:
        candidates = [value]
        row = raw_input.get("row")
        if isinstance(row, dict):
            candidates.extend([
                row.get("ts"),
                row.get("timestamp"),
                row.get("datetime"),
                row.get("frame.time"),
            ])
            if row.get("date") or row.get("time"):
                candidates.append(f"{row.get('date', '')} {row.get('time', '')}".strip())

        for candidate in candidates:
            normalized = self._parse_timestamp(candidate)
            if normalized is not None:
                return normalized
        return None

    def _value_for_key(self, row: dict[str, Any], expected_key: str) -> Any:
        for key, value in row.items():
            if str(key).strip().lower() == expected_key:
                return value
        return None

    def _first_label_candidate(
        self,
        row: dict[str, Any],
        include_tokens: tuple[str, ...],
        exclude_tokens: tuple[str, ...],
        reject_binary: bool = False,
    ) -> Any:
        for key, value in row.items():
            normalized = str(key).lower()
            if any(token in normalized for token in include_tokens) and not any(token in normalized for token in exclude_tokens):
                if not self._is_blank(value):
                    if reject_binary and str(value).strip().lower() in {"0", "0.0", "1", "1.0", "true", "false"}:
                        continue
                    return value
        return None

    def _calibrate_mapping_confidence(self, value: Any, payload: dict[str, Any]) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = 0.5
        return confidence

    def _normalize_modality(self, value: Any, raw_input: dict[str, Any]) -> str:
        modality = str(value or "").strip().lower().replace("-", "_").replace(" ", "_") or "network_flow"
        aliases = {
            "iot_telemetry": "telemetry",
            "sensor": "telemetry",
            "sensor_telemetry": "telemetry",
            "host_metrics": "host_log",
            "endpoint": "host_log",
            "endpoint_metrics": "host_log",
            "network_packet": "network_flow",
            "network_traffic": "network_flow",
            "flow": "network_flow",
            "alert_text": "alert",
            "log": "alert",
        }
        modality = aliases.get(modality, modality)
        if modality == "telemetry" and self._row_has_host_indicators(raw_input):
            return "host_log"
        if modality == "network_flow" and self._row_has_iot_telemetry_indicators(raw_input) and not self._row_has_network_indicators(raw_input):
            return "telemetry"
        return modality

    def _normalize_schema_profile(
        self,
        value: Any,
        payload: dict[str, Any],
        raw_input: dict[str, Any],
    ) -> str:
        allowed = {
            "network_flow",
            "network_packet",
            "host_metrics",
            "iot_telemetry",
            "alert_text",
            "pcap_ref",
            "unknown",
        }
        normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "host_log": "host_metrics",
            "endpoint": "host_metrics",
            "endpoint_metrics": "host_metrics",
            "telemetry": "iot_telemetry",
            "sensor": "iot_telemetry",
            "packet": "network_packet",
            "pcap": "pcap_ref",
            "alert": "alert_text",
            "log": "alert_text",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized in allowed:
            return normalized
        return self._infer_schema_profile(payload, raw_input)

    def _infer_schema_profile(self, payload: dict[str, Any], raw_input: dict[str, Any]) -> str:
        modality = str(payload.get("modality") or "").strip()
        if modality == "host_log" or self._row_has_host_indicators(raw_input):
            return "host_metrics"
        if modality == "telemetry" or (
            self._row_has_iot_telemetry_indicators(raw_input)
            and not self._row_has_network_indicators(raw_input)
        ):
            return "iot_telemetry"
        if modality == "alert":
            return "alert_text"
        if modality == "pcap_ref":
            return "pcap_ref"

        row = raw_input.get("row")
        keys = [str(key).lower() for key in row] if isinstance(row, dict) else []
        packet_prefixes = ("arp.", "eth.", "frame.", "http.", "icmp.", "ip.", "mqtt.", "tcp.", "udp.")
        packet_hits = sum(1 for key in keys if key.startswith(packet_prefixes))
        if packet_hits >= 3:
            return "network_packet"
        if self._row_has_alert_text_indicators(raw_input):
            return "alert_text"
        if self._row_has_network_indicators(raw_input) or modality == "network_flow":
            return "network_flow"
        return "unknown"

    def _row_has_iot_telemetry_indicators(self, raw_input: dict[str, Any]) -> bool:
        row = raw_input.get("row")
        if not isinstance(row, dict):
            return False
        telemetry_indicators = (
            "sensor",
            "temperature",
            "thermostat",
            "humidity",
            "pressure",
            "gps",
            "fridge",
            "garage",
            "motion",
            "light",
            "latitude",
            "longitude",
        )
        return any(any(token in str(key).lower() for token in telemetry_indicators) for key in row)

    def _row_has_network_indicators(self, raw_input: dict[str, Any]) -> bool:
        row = raw_input.get("row")
        if not isinstance(row, dict):
            return False
        network_indicators = (
            "src",
            "source",
            "orig",
            "dst",
            "dest",
            "resp",
            "ip",
            "port",
            "proto",
            "packet",
            "byte",
            "tcp",
            "udp",
            "icmp",
            "http",
            "dns",
            "mqtt",
        )
        return any(any(token in str(key).lower() for token in network_indicators) for key in row)

    def _row_has_alert_text_indicators(self, raw_input: dict[str, Any]) -> bool:
        row = raw_input.get("row")
        if not isinstance(row, dict):
            return False
        alert_indicators = (
            "alert",
            "description",
            "event",
            "log",
            "message",
            "raw",
            "rule",
            "signature",
        )
        return any(any(token in str(key).lower() for token in alert_indicators) for key in row)

    def _row_has_host_indicators(self, raw_input: dict[str, Any]) -> bool:
        row = raw_input.get("row")
        if not isinstance(row, dict):
            return False
        host_indicators = (
            "process",
            "processor",
            "cpu",
            "memory",
            "logicaldisk",
            "disk",
            "paged",
            "privileged",
            "interrupt",
            "dpc",
            "thread",
            "handle",
            "windows",
            "linux",
            "vsize",
            "rsize",
            "majflt",
            "minflt",
        )
        sensor_indicators = (
            "sensor",
            "temperature",
            "thermostat",
            "humidity",
            "pressure",
            "gps",
            "fridge",
            "garage",
            "modbus",
        )
        keys = [str(key).lower() for key in row]
        host_hits = sum(1 for key in keys if any(token in key for token in host_indicators))
        sensor_hits = sum(1 for key in keys if any(token in key for token in sensor_indicators))
        return host_hits > 0 and host_hits >= sensor_hits

    def _parse_timestamp(self, value: Any) -> str | None:
        if self._is_blank(value):
            return None
        if isinstance(value, (int, float)) and value > 1_000_000_000:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()

        text = str(value).strip()
        if self._is_blank(text):
            return None
        try:
            numeric = float(text)
        except ValueError:
            numeric = None
        if numeric is not None and numeric > 1_000_000_000:
            return datetime.fromtimestamp(numeric, tz=timezone.utc).isoformat()

        cleaned = re.sub(r"\s+", " ", text).strip()
        iso_candidate = cleaned.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(iso_candidate).isoformat()
        except ValueError:
            pass

        for fmt in (
            "%d-%b-%y %H:%M:%S",
            "%d-%b-%Y %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
            "%y-%m-%dT%H:%M:%SZ",
            "%y-%m-%dT%H:%M:%S%z",
            "%y-%m-%d %H:%M:%S",
        ):
            try:
                return datetime.strptime(cleaned, fmt).isoformat()
            except ValueError:
                continue
        partial_year_time = self._parse_year_time(cleaned)
        if partial_year_time is not None:
            return partial_year_time
        return None

    def _parse_year_time(self, value: str) -> str | None:
        match = re.fullmatch(r"(?P<year>\d{4})[\s-]+(?P<time>\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)", value.strip())
        if match is None:
            return None
        year = int(match.group("year"))
        time = match.group("time")
        if "." in time:
            base, fraction = time.split(".", 1)
            time = f"{base}.{fraction[:6]}"
            fmt = "%Y-%m-%d %H:%M:%S.%f"
        else:
            fmt = "%Y-%m-%d %H:%M:%S"
        try:
            parsed = datetime.strptime(f"{year}-01-01 {time}", fmt)
        except ValueError:
            return None
        return parsed.isoformat()

    def _is_blank(self, value: Any) -> bool:
        if value is None:
            return True
        if not isinstance(value, str):
            return False
        return value.strip() in {"", "-", "nan", "NaN", "None", "none", "null", "NULL"}

    def _field_is_missing(self, payload: dict[str, Any], field: str) -> bool:
        value = payload.get(field)
        if isinstance(value, dict):
            return len(value) == 0
        if isinstance(value, list):
            return len(value) == 0
        return self._is_blank(value)
