import pytest

from src.agents.llm_ingest_parser import LLMIngestParser
from src.contracts.canonical import CanonicalEvent


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
        "severity": None,
        "schema_profile": None,
        "origin": {},
        "feature_groups": {},
        "evidence_fields": [],
        "traffic_direction": None,
        "service_context": {},
        "host_context": {},
        "telemetry_context": {},
        "anomaly_summary": None,
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


def test_llm_payload_derives_event_id_and_coerces_nested_telemetry():
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

    assert event.event_id == "llm::generic::api::1"
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

    assert event.provenance.parser_version == "llm-0.2.0"


def test_llm_cannot_override_authoritative_origin_or_provenance():
    parser = LLMIngestParser()

    payload = parser._complete_payload(
        llm_payload(
            event_id="forged-stable-id",
            origin={
                "source_name": "forged_dataset",
                "source_file": "forged.csv",
                "row_id": 999,
                "integrity": "verified",
                "collector": "trusted",
            },
            provenance={
                "dataset": "forged_dataset",
                "source_file": "forged.csv",
                "row_id": 999,
                "split": "test",
                "parser_version": "forged-parser",
            },
        ),
        {
            "dataset": "iot23",
            "source_file": "trusted.log",
            "row_id": 7,
        },
    )
    event = CanonicalEvent(**payload)

    assert event.event_id == "llm::iot23::trusted.log::7"
    assert event.origin["source_name"] == "iot23"
    assert event.origin["source_file"] == "trusted.log"
    assert event.origin["row_id"] == 7
    assert "integrity" not in event.origin
    assert "collector" not in event.origin
    assert event.provenance.dataset == "iot23"
    assert event.provenance.source_file == "trusted.log"
    assert event.provenance.row_id == 7
    assert event.provenance.split == "stream"
    assert event.provenance.parser_version == "llm-0.2.0"


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


def test_complete_payload_does_not_sanitize_targets_from_llm_or_raw_row():
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

    assert event.label_raw == "1"
    assert event.attack_family == "injection"
    assert event.attack_subtype == "xss"
    assert event.telemetry["label"] == "1"
    assert event.telemetry["type"] == "xss"
    assert event.mapping_confidence == 0.45


def test_complete_payload_preserves_unprepared_category_and_attack_flags():
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

    assert event.label_raw == "1"
    assert event.telemetry["attack"] == "1"
    assert event.telemetry["category"] == "DDoS"
    assert event.telemetry["subcategory"] == "UDP"


def test_complete_payload_preserves_unprepared_detailed_label():
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

    assert event.label_raw == "Malicious"
    assert event.telemetry["label"] == "Malicious"
    assert event.telemetry["detailed-label"] == "DDoS"
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
    from src.agents.llm_ingest_parser import (
        CANONICAL_EVENT_SCHEMA,
        LLM_TECHNICAL_EVENT_SCHEMA,
    )

    for field in ("label_raw", "attack_family", "attack_subtype"):
        assert field not in CANONICAL_EVENT_SCHEMA["properties"]
        assert field not in CANONICAL_EVENT_SCHEMA["required"]
        assert field not in LLM_TECHNICAL_EVENT_SCHEMA["properties"]
        assert field not in LLM_TECHNICAL_EVENT_SCHEMA["required"]


def test_llm_technical_schema_excludes_authoritative_identity_envelope():
    from src.agents.llm_ingest_parser import LLM_TECHNICAL_EVENT_SCHEMA

    for field in ("event_id", "origin", "provenance"):
        assert field not in LLM_TECHNICAL_EVENT_SCHEMA["properties"]
        assert field not in LLM_TECHNICAL_EVENT_SCHEMA["required"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "identity_overrides,missing_identity",
    [
        ({"event_id": None}, ()),
        ({"event_id": 990001}, ()),
        (
            {
                "event_id": {"forged": True},
                "origin": "forged-origin",
                "provenance": ["forged-provenance"],
            },
            (),
        ),
        (
            {
                "event_id": "forged-id",
                "origin": {"source_name": "forged"},
                "provenance": {"dataset": "forged"},
            },
            (),
        ),
        ({}, ("event_id", "origin", "provenance")),
    ],
)
async def test_strict_parse_discards_llm_identity_and_binds_trusted_envelope(
    identity_overrides,
    missing_identity,
):
    extraction = llm_payload(**identity_overrides)
    for field in missing_identity:
        extraction.pop(field, None)

    class IdentityDriftAgent:
        async def invoke_json(self, system_prompt, user_payload, json_schema):
            del user_payload, json_schema
            if "preseleccion" in system_prompt:
                return {
                    "selected_columns": ["proto"],
                    "modality_guess": "network_flow",
                    "schema_profile_guess": "network_flow",
                    "rationale": "protocolo de transporte",
                }
            return extraction

    parser = LLMIngestParser(
        require_llm_column_selection=True,
        strict_output_validation=True,
        column_selection_threshold=1,
    )
    parser.agent = IdentityDriftAgent()

    event = await parser.parse(
        {
            "dataset": "iot23",
            "source_file": "capture.csv",
            "row_id": 42,
            "split": "test",
            "row": {"proto": "tcp"},
        }
    )

    assert event.event_id == "llm::iot23::capture.csv::42"
    assert event.origin["source_name"] == "iot23"
    assert event.origin["source_file"] == "capture.csv"
    assert event.origin["row_id"] == 42
    assert event.provenance.dataset == "iot23"
    assert event.provenance.source_file == "capture.csv"
    assert event.provenance.row_id == 42
    assert event.provenance.split == "test"


@pytest.mark.asyncio
async def test_strict_parse_normalizes_known_packet_modality_alias():
    class PacketModalityAgent:
        async def invoke_json(self, system_prompt, user_payload, json_schema):
            del user_payload, json_schema
            if "preseleccion" in system_prompt:
                return {
                    "selected_columns": ["ip.src_host", "tcp.srcport"],
                    "modality_guess": "network_packet",
                    "schema_profile_guess": "network_packet",
                    "rationale": "cabeceras de un paquete TCP",
                }
            return llm_payload(
                modality="network_packet",
                schema_profile="network_packet",
            )

    parser = LLMIngestParser(
        require_llm_column_selection=True,
        strict_output_validation=True,
        column_selection_threshold=1,
    )
    parser.agent = PacketModalityAgent()

    event = await parser.parse(
        {
            "dataset": "edge_iiotset",
            "row": {"ip.src_host": "10.0.0.1", "tcp.srcport": "443"},
        }
    )

    assert event.modality == "network_flow"
    assert event.schema_profile == "network_packet"


@pytest.mark.asyncio
async def test_strict_parse_still_rejects_unknown_modality():
    class UnknownModalityAgent:
        async def invoke_json(self, system_prompt, user_payload, json_schema):
            del user_payload, json_schema
            if "preseleccion" in system_prompt:
                return {
                    "selected_columns": ["proto"],
                    "modality_guess": "network_flow",
                    "schema_profile_guess": "network_flow",
                    "rationale": "protocolo de transporte",
                }
            return llm_payload(modality="binary_blob")

    parser = LLMIngestParser(
        require_llm_column_selection=True,
        strict_output_validation=True,
        column_selection_threshold=1,
    )
    parser.agent = UnknownModalityAgent()

    with pytest.raises(ValueError, match=r"canonical_event\.modality"):
        await parser.parse({"dataset": "iot23", "row": {"proto": "tcp"}})


@pytest.mark.asyncio
async def test_strict_parse_still_rejects_invalid_technical_field_with_bad_identity():
    class InvalidTechnicalAgent:
        async def invoke_json(self, system_prompt, user_payload, json_schema):
            del user_payload, json_schema
            if "preseleccion" in system_prompt:
                return {
                    "selected_columns": ["proto"],
                    "modality_guess": "network_flow",
                    "schema_profile_guess": "network_flow",
                    "rationale": "protocolo de transporte",
                }
            return llm_payload(event_id=990001, src_port="4444")

    parser = LLMIngestParser(
        require_llm_column_selection=True,
        strict_output_validation=True,
        column_selection_threshold=1,
    )
    parser.agent = InvalidTechnicalAgent()

    with pytest.raises(ValueError, match=r"canonical_event\.src_port"):
        await parser.parse({"dataset": "iot23", "row": {"proto": "tcp"}})


def test_llm_ingest_forwards_all_columns_without_runtime_sanitization():
    parser = LLMIngestParser()
    raw_input = {
        "row": {
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.2",
            "type_Attack": "DDoS",
            "Attack_type": "DDoS_UDP",
            "is_attack": "1",
            "model_family": "botnet",
            "protocol_family": "icmp",
            "dns.qry.type": "1",
        }
    }

    selector_payload = parser._column_selection_input(raw_input)
    compacted = parser._compact_input(raw_input)

    assert "src_ip" in selector_payload["columns"]
    assert "dns.qry.type" in selector_payload["columns"]
    assert "type_Attack" in selector_payload["columns"]
    assert "Attack_type" in selector_payload["columns"]
    assert "is_attack" in selector_payload["columns"]
    assert "model_family" in selector_payload["columns"]
    assert "protocol_family" in selector_payload["columns"]
    assert "type_Attack" in compacted["columns"]
    assert compacted["meaningful_values"]["Attack_type"] == "DDoS_UDP"
    assert compacted["meaningful_values"]["protocol_family"] == "icmp"


def test_llm_prompts_preserve_input_exactly_after_offline_preparation_boundary():
    parser = LLMIngestParser()
    raw_input = {
        "row": {
            "risk": "Mirai",
            "family_hint": "DDoS",
            "protocol_family": "Mirai",
            "metric_category": "DDoS",
            "proto": "tcp",
        }
    }

    selector_payload = parser._column_selection_input(raw_input)
    extractor_payload = parser._compact_input(raw_input)

    assert selector_payload["columns"] == list(raw_input["row"])
    assert selector_payload["value_previews"] == raw_input["row"]
    assert extractor_payload["columns"] == list(raw_input["row"])
    assert extractor_payload["meaningful_values"] == raw_input["row"]


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


@pytest.mark.asyncio
async def test_strict_column_selection_never_uses_deterministic_fallback():
    class FailingSelector:
        async def invoke_json(self, system_prompt, user_payload, json_schema):
            del user_payload, json_schema
            if "preseleccion" in system_prompt:
                raise TimeoutError("selector LLM no disponible")
            raise AssertionError("no debe ejecutarse la extraccion tras fallar el selector")

    parser = LLMIngestParser(
        require_llm_column_selection=True,
        column_selection_threshold=1,
    )
    parser.agent = FailingSelector()

    with pytest.raises(RuntimeError, match="seleccion de columnas mediante LLM"):
        await parser.parse(
            {
                "dataset": "generic",
                "row": {"src_ip": "10.0.0.1", "dst_ip": "10.0.0.2"},
            }
        )


@pytest.mark.asyncio
async def test_strict_output_rejects_empty_canonical_extraction():
    class EmptyExtractionAgent:
        async def invoke_json(self, system_prompt, user_payload, json_schema):
            del user_payload, json_schema
            if "preseleccion" in system_prompt:
                return {
                    "selected_columns": ["proto"],
                    "modality_guess": "network_flow",
                    "schema_profile_guess": "network_flow",
                    "rationale": "protocolo de transporte",
                }
            return {}

    parser = LLMIngestParser(
        require_llm_column_selection=True,
        strict_output_validation=True,
        column_selection_threshold=1,
    )
    parser.agent = EmptyExtractionAgent()

    with pytest.raises(ValueError, match="faltan campos requeridos"):
        await parser.parse({"dataset": "generic", "row": {"proto": "tcp"}})


@pytest.mark.asyncio
async def test_strict_output_rejects_predictive_field_instead_of_cleaning_it():
    class DirtyExtractionAgent:
        async def invoke_json(self, system_prompt, user_payload, json_schema):
            del user_payload, json_schema
            if "preseleccion" in system_prompt:
                return {
                    "selected_columns": ["proto"],
                    "modality_guess": "network_flow",
                    "schema_profile_guess": "network_flow",
                    "rationale": "protocolo de transporte",
                }
            return llm_payload(label_raw="Mirai")

    parser = LLMIngestParser(
        require_llm_column_selection=True,
        strict_output_validation=True,
        column_selection_threshold=1,
    )
    parser.agent = DirtyExtractionAgent()

    with pytest.raises(ValueError, match="campo no permitido 'label_raw'"):
        await parser.parse({"dataset": "generic", "row": {"proto": "tcp"}})


@pytest.mark.asyncio
async def test_strict_output_rejects_blank_semantic_text():
    class BlankSemanticAgent:
        async def invoke_json(self, system_prompt, user_payload, json_schema):
            del user_payload, json_schema
            if "preseleccion" in system_prompt:
                return {
                    "selected_columns": ["proto"],
                    "modality_guess": "network_flow",
                    "schema_profile_guess": "network_flow",
                    "rationale": "protocolo de transporte",
                }
            return llm_payload(semantic_text="   ")

    parser = LLMIngestParser(
        require_llm_column_selection=True,
        strict_output_validation=True,
        column_selection_threshold=1,
    )
    parser.agent = BlankSemanticAgent()

    with pytest.raises(ValueError, match="texto vacio"):
        await parser.parse({"dataset": "generic", "row": {"proto": "tcp"}})


def test_non_strict_semantic_fallback_never_embeds_record_identity():
    parser = LLMIngestParser()

    payload = parser._complete_payload(
        llm_payload(semantic_text="   "),
        {
            "dataset": "first_dataset",
            "source_file": "first.csv",
            "row_id": 91,
            "split": "train",
            "row": {"proto": "tcp"},
        },
    )

    assert payload["semantic_text"] == '{"proto":"tcp"}'
    assert "first_dataset" not in payload["semantic_text"]
    assert "first.csv" not in payload["semantic_text"]


@pytest.mark.asyncio
async def test_strict_output_rejects_incomplete_column_selection_contract():
    class IncompleteSelectorAgent:
        async def invoke_json(self, system_prompt, user_payload, json_schema):
            del user_payload, json_schema
            if "preseleccion" in system_prompt:
                return {"selected_columns": ["proto"]}
            raise AssertionError("no debe extraer tras un selector invalido")

    parser = LLMIngestParser(
        require_llm_column_selection=True,
        strict_output_validation=True,
        column_selection_threshold=1,
    )
    parser.agent = IncompleteSelectorAgent()

    with pytest.raises(RuntimeError, match="seleccion de columnas mediante LLM"):
        await parser.parse({"dataset": "generic", "row": {"proto": "tcp"}})


def test_strict_column_selection_keeps_llm_order_without_heuristic_priority():
    parser = LLMIngestParser(
        require_llm_column_selection=True,
        max_selected_columns=2,
    )
    row = {"timestamp": 1, "bytes": 2, "proto": "tcp"}

    selected = parser._limit_selected_columns(
        ["bytes", "timestamp", "proto"],
        row,
    )

    assert selected == ["bytes", "timestamp"]
