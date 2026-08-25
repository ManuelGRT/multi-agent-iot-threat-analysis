# src/agents/ingest_parser.py
from __future__ import annotations

import re
from typing import Any

from src.adapters.bot_iot import BotIoTAdapter
from src.adapters.edge_iiotset import EdgeIIoTSetAdapter
from src.adapters.generic import GenericAdapter
from src.adapters.iot23 import IoT23Adapter
from src.adapters.ton_iot import TONIoTAdapter
from src.agents.llm_ingest_parser import LLMIngestParser
from src.contracts.agents import IngestOutput
from src.contracts.canonical import CanonicalEvent


ADAPTERS = {
    "iot23": IoT23Adapter(),
    "iot-23": IoT23Adapter(),
    "ton_iot": TONIoTAdapter(),
    "ton-iot": TONIoTAdapter(),
    "bot_iot": BotIoTAdapter(),
    "bot-iot": BotIoTAdapter(),
    "edge_iiotset": EdgeIIoTSetAdapter(),
    "edge-iiotset": EdgeIIoTSetAdapter(),
    "generic": GenericAdapter(),
    "unknown": GenericAdapter(),
}


class IngestParserAgent:
    def __init__(
        self,
        adapters: dict[str, Any] | None = None,
        llm_parser: LLMIngestParser | None = None,
    ):
        self.adapters = adapters or ADAPTERS
        self.llm_parser = llm_parser

    def ingest(self, raw_input: dict[str, Any]) -> tuple[CanonicalEvent, IngestOutput]:
        if raw_input.get("use_llm"):
            raise ValueError("LLM ingestion is async. Use ingest_async or the API endpoint with use_llm=true.")
        return self._ingest_with_adapter(raw_input)

    async def ingest_async(self, raw_input: dict[str, Any]) -> tuple[CanonicalEvent, IngestOutput]:
        if self._should_use_llm(raw_input):
            try:
                event = await self.llm_parser.parse(raw_input)  # type: ignore[union-attr]
                return event, self._output_from_event(event, notes=["parsed_by_llm"])
            except Exception as exc:
                return self._ingest_with_adapter(
                    raw_input,
                    notes=[
                        "llm_failed",
                        f"llm_error_type={type(exc).__name__}",
                        "fallback_adapter",
                    ],
                )
        return self._ingest_with_adapter(raw_input)

    def _should_use_llm(self, raw_input: dict[str, Any]) -> bool:
        if self.llm_parser is None:
            return False
        llm_flag = raw_input.get("use_llm")
        if llm_flag is True:
            return True
        if llm_flag is False:
            return False
        dataset = str(raw_input.get("dataset", "generic")).strip().lower()
        return dataset not in self.adapters or dataset in {"generic", "unknown"}

    def _ingest_with_adapter(
        self,
        raw_input: dict[str, Any],
        notes: list[str] | None = None,
    ) -> tuple[CanonicalEvent, IngestOutput]:
        dataset = str(raw_input.get("dataset", "iot23")).strip().lower()
        row = raw_input.get("row") or raw_input.get("event") or self._row_from_generic_input(raw_input)
        row = dict(row)
        origin_dataset = (
            raw_input.get("dataset_family")
            or raw_input.get("origin")
            or raw_input.get("source_dataset")
            or raw_input.get("dataset", dataset)
        )
        row.setdefault("dataset", origin_dataset)
        source_file = str(raw_input.get("source_file", "inline"))
        row_id = raw_input.get("row_id", raw_input.get("event_id", 0))

        adapter = self.adapters.get(dataset, self.adapters["generic"])

        event = adapter.adapt(row=row, source_file=source_file, row_id=row_id)
        return event, self._output_from_event(event, notes=notes)

    def _output_from_event(self, event: CanonicalEvent, notes: list[str] | None = None) -> IngestOutput:
        merged_notes = list(notes or []) + event.missing_fields
        output = IngestOutput(
            event_id=event.event_id,
            normalized=True,
            mapping_confidence=event.mapping_confidence,
            schema_version="0.1.0",
            modality=event.modality,
            notes=merged_notes,
        )
        return output

    def _row_from_generic_input(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        if "text" in raw_input:
            return self._text_to_row(str(raw_input["text"]), raw_input.get("dataset", "generic"))
        if "raw" in raw_input:
            raw = raw_input["raw"]
            if isinstance(raw, dict):
                return raw
            return self._text_to_row(str(raw), raw_input.get("dataset", "generic"))
        return raw_input

    def _text_to_row(self, text: str, dataset: Any) -> dict[str, Any]:
        row: dict[str, Any] = {"raw_text": text, "dataset": dataset}
        for key, value in re.findall(r"([A-Za-z0-9_.-]+)\s*=\s*([^,\s|]+)", text):
            row[key] = value
        return row
