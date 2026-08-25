import pytest

from src.agents.ingest_parser import IngestParserAgent
from src.agents.llm_ingest_parser import LLMIngestParser
from src.contracts.canonical import CanonicalEvent, Provenance


def llm_payload(**overrides):
    payload = {
        "event_id": "llm::sample::1",
        "modality": "network_flow",
        "ts": None,
        "src_ip": "10.0.0.1",
        "dst_ip": "10.0.0.2",
        "src_port": 4444,
        "dst_port": 23,
        "transport_proto": "tcp",
        "app_proto": "",
        "packet_count": None,
        "byte_count": None,
        "duration_ms": None,
        "telemetry": {"empty": "", "temperature": "31.5"},
        "host": {"process": ""},
        "severity": "",
        "semantic_text": "Mirai telnet traffic",
        "provenance": {"dataset": "generic"},
        "mapping_confidence": 0.95,
        "missing_fields": [],
    }
    payload.update(overrides)
    return payload


def test_llm_payload_sanitizes_invalid_timestamp_and_empty_values():
    parser = LLMIngestParser()

    payload = parser._complete_payload(
        llm_payload(ts="19-04-25T08:59:02Z"),
        {"dataset": "generic", "source_file": "api", "row_id": 1},
    )
    event = CanonicalEvent(**payload)

    assert event.ts.isoformat().startswith("2019-04-25T08:59:02")
    assert event.app_proto is None
    assert event.label_raw is None
    assert event.attack_family is None
    assert event.attack_subtype is None
    assert "empty" not in event.telemetry


def test_llm_payload_normalizes_edge_year_time_timestamp():
    parser = LLMIngestParser()

    payload = parser._complete_payload(
        llm_payload(ts="2021-22:14:30.939803000", missing_fields=["ts", "transport_proto"]),
        {"dataset": "generic", "source_file": "api", "row_id": 1},
    )
    event = CanonicalEvent(**payload)

    assert event.ts.isoformat() == "2021-01-01T22:14:30.939803"
    assert "ts" not in event.missing_fields
    assert "transport_proto" not in event.missing_fields


def test_llm_payload_uses_edge_source_frame_time_when_raw_llm_ts_is_bad():
    parser = LLMIngestParser()

    payload = parser._complete_payload(
        llm_payload(ts="not-a-date"),
        {
            "dataset": "generic",
            "source_file": "api",
            "row_id": 1,
            "row": {"frame.time": " 2021 22:14:30.939803000 "},
        },
    )
    event = CanonicalEvent(**payload)

    assert event.ts.isoformat() == "2021-01-01T22:14:30.939803"


def test_llm_payload_coerces_event_id_and_nested_telemetry():
    parser = LLMIngestParser()

    payload = parser._complete_payload(
        llm_payload(
            event_id=123,
            telemetry={"processor": {"pct_user_time": 38.1}},
            provenance=None,
            missing_fields=None,
        ),
        {"dataset": "generic", "source_file": "api", "row_id": 1},
    )
    event = CanonicalEvent(**payload)

    assert event.event_id == "123"
    assert event.telemetry["processor"] == '{"pct_user_time": 38.1}'
    assert event.provenance.dataset == "generic"


def test_llm_payload_normalizes_invalid_provenance_split():
    parser = LLMIngestParser()

    payload = parser._complete_payload(
        llm_payload(provenance={"dataset": "generic", "split": "unknown"}),
        {"dataset": "generic", "source_file": "api", "row_id": 1},
    )
    event = CanonicalEvent(**payload)

    assert event.provenance.split == "stream"


def test_llm_payload_normalizes_missing_provenance_parser_version():
    parser = LLMIngestParser()

    payload = parser._complete_payload(
        llm_payload(provenance={"dataset": "generic", "parser_version": None}),
        {"dataset": "generic", "source_file": "api", "row_id": 1},
    )
    event = CanonicalEvent(**payload)

    assert event.provenance.parser_version == "llm-0.1.0"


def test_llm_payload_normalizes_host_metrics_to_host_log():
    parser = LLMIngestParser()

    payload = parser._complete_payload(
        llm_payload(modality="telemetry"),
        {
            "dataset": "generic",
            "source_file": "windows.csv",
            "row_id": 1,
            "row": {
                "Processor_pct_User_Time": "38.1",
                "Process_Private_Bytes": "1234",
                "LogicalDisk_Free_Megabytes": "512",
                "type": "dos",
            },
        },
    )
    event = CanonicalEvent(**payload)

    assert event.modality == "host_log"
    assert event.schema_profile == "host_metrics"


def test_llm_payload_normalizes_sensor_metrics_to_telemetry_when_no_network_fields():
    parser = LLMIngestParser()

    payload = parser._complete_payload(
        llm_payload(modality="network_flow"),
        {
            "dataset": "generic",
            "source_file": "thermostat.csv",
            "row_id": 1,
            "row": {
                "current_temperature": "29.2",
                "thermostat_status": "1",
                "label": "1",
                "type": "xss",
            },
        },
    )
    event = CanonicalEvent(**payload)

    assert event.modality == "telemetry"
    assert event.schema_profile == "iot_telemetry"


def test_llm_payload_accepts_common_modality_aliases():
    parser = LLMIngestParser()

    telemetry = CanonicalEvent(**parser._complete_payload(
        llm_payload(modality="iot_telemetry"),
        {
            "dataset": "generic",
            "source_file": "thermostat.csv",
            "row_id": 1,
            "row": {"current_temperature": "29.2", "thermostat_status": "1"},
        },
    ))
    packet = CanonicalEvent(**parser._complete_payload(
        llm_payload(modality="network_packet"),
        {
            "dataset": "generic",
            "source_file": "packets.csv",
            "row_id": 1,
            "row": {"ip.src_host": "10.0.0.1", "tcp.srcport": "443"},
        },
    ))

    assert telemetry.modality == "telemetry"
    assert packet.modality == "network_flow"


def test_llm_payload_nulls_column_names_in_numeric_fields():
    parser = LLMIngestParser()

    payload = parser._complete_payload(
        llm_payload(
            src_port="tcp.srcport",
            dst_port="502.0",
            packet_count="frame.len",
            byte_count="1,024",
            duration_ms="duration",
            missing_fields=["src_port", "packet_count", "duration_ms"],
        ),
        {
            "dataset": "generic",
            "source_file": "packets.csv",
            "row_id": 1,
            "row": {"ip.src_host": "10.0.0.1", "tcp.srcport": "443"},
        },
    )
    event = CanonicalEvent(**payload)

    assert event.src_port is None
    assert event.dst_port == 502
    assert event.packet_count is None
    assert event.byte_count == 1024
    assert event.duration_ms is None
    assert "src_port" in event.missing_fields
    assert "packet_count" in event.missing_fields
    assert "duration_ms" in event.missing_fields


def test_llm_payload_fills_origin_from_dataset_family():
    parser = LLMIngestParser()

    payload = parser._complete_payload(
        llm_payload(origin={}, schema_profile=None),
        {
            "dataset": "generic",
            "dataset_family": "source_family_a",
            "source_file": "events.csv",
            "row_id": 7,
            "row": {"src": "10.0.0.1", "dst": "10.0.0.2", "proto": "tcp"},
        },
    )
    event = CanonicalEvent(**payload)

    assert event.provenance.dataset == "source_family_a"
    assert event.origin["source_name"] == "source_family_a"
    assert event.origin["schema_profile"] == event.schema_profile


def test_llm_payload_does_not_fill_normal_label_from_raw_row():
    parser = LLMIngestParser()

    payload = parser._complete_payload(
        llm_payload(mapping_confidence=0.45),
        {
            "dataset": "generic",
            "source_file": "host.csv",
            "row_id": 1,
            "row": {"label": "0", "type": "normal"},
        },
    )
    event = CanonicalEvent(**payload)

    assert event.label_raw is None
    assert event.mapping_confidence == 0.45


def test_llm_payload_drops_attack_type_from_raw_row():
    parser = LLMIngestParser()

    payload = parser._complete_payload(
        llm_payload(label_raw="1", attack_family="injection", attack_subtype="xss", mapping_confidence=0.45),
        {
            "dataset": "generic",
            "source_file": "host.csv",
            "row_id": 1,
            "row": {"label": "1", "type": "xss"},
        },
    )
    event = CanonicalEvent(**payload)

    assert event.label_raw is None
    assert event.attack_family is None
    assert event.attack_subtype is None
    assert event.mapping_confidence == 0.45


def test_llm_payload_drops_category_and_attack_flags_from_raw_row():
    parser = LLMIngestParser()

    payload = parser._complete_payload(
        llm_payload(label_raw="1", mapping_confidence=0.45),
        {
            "dataset": "generic",
            "source_file": "bot.csv",
            "row_id": 1,
            "row": {"attack": "1", "category": "DDoS", "subcategory": "UDP"},
        },
    )
    event = CanonicalEvent(**payload)

    assert event.label_raw is None
    assert "attack" not in event.telemetry
    assert "category" not in event.telemetry
    assert "subcategory" not in event.telemetry


def test_llm_payload_drops_detailed_label_over_malicious_flag():
    parser = LLMIngestParser()

    payload = parser._complete_payload(
        llm_payload(label_raw="Malicious", mapping_confidence=0.45),
        {
            "dataset": "generic",
            "source_file": "conn.log.labeled",
            "row_id": 1,
            "row": {"label": "Malicious", "detailed-label": "DDoS"},
        },
    )
    event = CanonicalEvent(**payload)

    assert event.label_raw is None
    assert event.mapping_confidence == 0.45


class RecordingLLMAgent:
    def __init__(self):
        self.calls = []

    async def invoke_json(self, system_prompt, user_payload, json_schema):
        self.calls.append({
            "system_prompt": system_prompt,
            "user_payload": user_payload,
            "json_schema": json_schema,
        })
        if "preseleccion" in system_prompt:
            return {
                "selected_columns": ["origin_endpoint", "target_endpoint", "target_service_number"],
                "modality_guess": "network_flow",
                "rationale": "network identifiers and service endpoint",
            }
        return llm_payload(
            src_ip="10.10.0.15",
            dst_ip="172.16.1.20",
            src_port=None,
            dst_port=502,
        )


@pytest.mark.asyncio
async def test_wide_rows_use_llm_column_selection_before_extraction():
    parser = LLMIngestParser(column_selection_threshold=5, max_selected_columns=5)
    recorder = RecordingLLMAgent()
    parser.agent = recorder

    event = await parser.parse({
        "dataset": "generic",
        "source_file": "inline.csv",
        "row_id": 1,
        "row": {
            "origin_endpoint": "10.10.0.15",
            "target_endpoint": "172.16.1.20",
            "origin_service_number": "51544",
            "target_service_number": "502",
            "noise_a": "123",
            "noise_b": "456",
            "incident_category": "Modbus Scan",
        },
    })

    assert event.label_raw is None
    assert len(recorder.calls) == 2
    selector_payload = recorder.calls[0]["user_payload"]
    extractor_payload = recorder.calls[1]["user_payload"]
    assert selector_payload["total_columns"] == 7
    assert "origin" not in selector_payload
    assert "source_file" not in selector_payload
    assert extractor_payload["columns"] == [
        "origin_endpoint",
        "target_endpoint",
        "target_service_number",
    ]
    assert "origin" not in extractor_payload
    assert "source_file" not in extractor_payload
    assert "noise_a" not in extractor_payload["columns"]
    assert extractor_payload["column_selection"]["source_column_count"] == 7


def test_llm_ingest_prompts_do_not_name_specific_datasets():
    from src.agents.llm_ingest_parser import LLM_COLUMN_SELECTION_SYSTEM, LLM_INGEST_SYSTEM

    prompt = f"{LLM_INGEST_SYSTEM}\n{LLM_COLUMN_SELECTION_SYSTEM}".lower()

    for forbidden in ("edge", "ton", "iot23", "bot-iot", "windows", "linux"):
        assert forbidden not in prompt


def test_llm_ingest_schema_does_not_expose_prediction_targets():
    from src.agents.llm_ingest_parser import CANONICAL_EVENT_SCHEMA

    for field in ("label_raw", "attack_family", "attack_subtype"):
        assert field not in CANONICAL_EVENT_SCHEMA["properties"]
        assert field not in CANONICAL_EVENT_SCHEMA["required"]


def test_llm_ingest_filters_target_column_variants():
    parser = LLMIngestParser()
    raw_input = {
        "row": {
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.2",
            "type_Attack": "DDoS",
            "Attack_type": "DDoS_UDP",
            "is_attack": "1",
            "model_family": "botnet",
            "dns.qry.type": "1",
        }
    }

    selector_payload = parser._column_selection_input(raw_input)
    compacted = parser._compact_input(raw_input)

    assert "src_ip" in selector_payload["columns"]
    assert "dns.qry.type" in selector_payload["columns"]
    assert "type_Attack" not in selector_payload["columns"]
    assert "Attack_type" not in selector_payload["columns"]
    assert "is_attack" not in selector_payload["columns"]
    assert "model_family" not in selector_payload["columns"]
    assert "type_Attack" not in compacted["columns"]
    assert "Attack_type" not in compacted["meaningful_values"]


def test_llm_ingest_parser_accepts_direct_api_providers():
    assert LLMIngestParser(model="llama-3.3-70b-versatile", provider="groq").agent.__class__.__name__ == "GroqChatAgent"
    assert LLMIngestParser(model="mistral-small-latest", provider="mistral").agent.__class__.__name__ == "MistralChatAgent"
    assert LLMIngestParser(model="gemini-2.5-flash", provider="gemini").agent.__class__.__name__ == "GoogleAIStudioChatAgent"


@pytest.mark.asyncio
async def test_narrow_rows_skip_llm_column_selection():
    parser = LLMIngestParser(column_selection_threshold=99)
    recorder = RecordingLLMAgent()
    parser.agent = recorder

    await parser.parse({
        "dataset": "generic",
        "source_file": "inline.csv",
        "row_id": 1,
        "row": {
            "origin_endpoint": "10.10.0.15",
            "target_endpoint": "172.16.1.20",
            "incident_category": "Modbus Scan",
        },
    })

    assert len(recorder.calls) == 1
    assert "preseleccion" not in recorder.calls[0]["system_prompt"]


class StubParser:
    async def parse(self, raw_input):
        return CanonicalEvent(
            event_id="stub-llm",
            modality="network_flow",
            src_ip="10.0.0.1",
            dst_ip="10.0.0.2",
            transport_proto="tcp",
            label_raw="normal",
            semantic_text=str(raw_input),
            provenance=Provenance(dataset="generic"),
            mapping_confidence=0.9,
        )


class FailingParser:
    async def parse(self, raw_input):
        raise TimeoutError("slow llm")


@pytest.mark.asyncio
async def test_generic_async_ingest_prefers_llm_when_available():
    agent = IngestParserAgent(llm_parser=StubParser())

    event, output = await agent.ingest_async({"dataset": "generic", "text": "some log"})

    assert event.event_id == "stub-llm"
    assert output.notes == ["parsed_by_llm"]


@pytest.mark.asyncio
async def test_generic_async_ingest_falls_back_to_adapter_when_llm_fails():
    agent = IngestParserAgent(llm_parser=FailingParser())

    event, output = await agent.ingest_async(
        {
            "dataset": "generic",
            "text": "src_ip=10.0.0.1 dst_ip=10.0.0.2 proto=tcp label=normal",
            "row_id": 1,
        }
    )

    assert event.src_ip == "10.0.0.1"
    assert event.label_raw == "normal"
    assert "llm_failed" in output.notes
    assert "fallback_adapter" in output.notes


@pytest.mark.asyncio
async def test_edge_async_ingest_fallback_handles_partial_year_time():
    agent = IngestParserAgent(llm_parser=FailingParser())

    event, output = await agent.ingest_async(
        {
            "dataset": "edge_iiotset",
            "row": {
                "frame.time": " 2021 22:25:14.432732000 ",
                "ip.src_host": "10.0.0.1",
                "ip.dst_host": "10.0.0.2",
                "ip.proto": "tcp",
            },
            "source_file": "edge.csv",
            "row_id": 1947,
            "use_llm": True,
        }
    )

    assert event.ts is not None
    assert event.ts.isoformat() == "2021-01-01T22:25:14.432732"
    assert "llm_failed" in output.notes
    assert "fallback_adapter" in output.notes
