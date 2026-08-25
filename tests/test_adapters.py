from src.adapters.bot_iot import BotIoTAdapter
from src.adapters.edge_iiotset import EdgeIIoTSetAdapter
from src.adapters.iot23 import IoT23Adapter
from src.adapters.ton_iot import TONIoTAdapter


def test_iot23_adapter_normalizes_string_duration_and_ports():
    event = IoT23Adapter().adapt(
        {
            "id.orig_h": "10.0.0.2",
            "id.resp_h": "10.0.0.3",
            "id.orig_p": "4444",
            "id.resp_p": "23",
            "proto": "tcp",
            "duration": "1.5",
            "orig_pkts": "12",
            "orig_ip_bytes": "1024",
            "label": "Mirai",
        },
        source_file="conn.log",
        row_id=7,
    )

    assert event.event_id == "iot23::conn.log::7"
    assert event.duration_ms == 1500.0
    assert event.src_port == 4444
    assert event.mapping_confidence >= 0.9


def test_iot23_adapter_prefers_detailed_malicious_label():
    event = IoT23Adapter().adapt(
        {
            "id.orig_h": "10.0.0.2",
            "id.resp_h": "10.0.0.3",
            "proto": "tcp",
            "label": "Malicious",
            "detailed-label": "C&C-Torii",
        },
        source_file="conn.log.labeled",
        row_id=8,
    )

    assert event.label_raw == "C&C-Torii"


def test_ton_iot_adapter_handles_telemetry_source():
    event = TONIoTAdapter().adapt(
        {"sensor": "temp", "value": 41.2, "label": "normal"},
        source_file="sensor.csv",
        row_id=1,
    )

    assert event.modality == "telemetry"
    assert "value" in event.telemetry
    assert event.provenance.dataset == "TON_IoT"


def test_generic_adapter_prefers_attack_type_over_binary_label():
    from src.adapters.generic import GenericAdapter

    event = GenericAdapter().adapt(
        {"label": "1", "attack_type": "backdoor"},
        source_file="alerts.csv",
        row_id=1,
    )

    assert event.label_raw == "backdoor"


def test_generic_adapter_prefers_iot23_detailed_label_over_malicious_flag():
    from src.adapters.generic import GenericAdapter

    event = GenericAdapter().adapt(
        {"label": "Malicious", "detailed-label": "C&C"},
        source_file="conn.log.labeled",
        row_id=1,
    )

    assert event.label_raw == "C&C"


def test_generic_adapter_combines_bot_iot_category_and_subcategory():
    from src.adapters.generic import GenericAdapter

    event = GenericAdapter().adapt(
        {"attack": "1", "category": "Reconnaissance", "subcategory": "Service_Scan"},
        source_file="bot.csv",
        row_id=1,
    )

    assert event.label_raw == "Reconnaissance Service_Scan"


def test_generic_adapter_detects_iot_telemetry_and_excludes_labels_from_telemetry():
    from src.adapters.generic import GenericAdapter

    event = GenericAdapter().adapt(
        {
            "date": "27-Apr-19",
            "time": "04:58:07",
            "current_temperature": "29.2904525",
            "thermostat_status": "1",
            "label": "1",
            "type": "xss",
        },
        source_file="thermostat.csv",
        row_id=1,
    )

    assert event.modality == "telemetry"
    assert event.schema_profile == "iot_telemetry"
    assert event.label_raw == "xss"
    assert event.mapping_confidence >= 0.5
    assert event.origin["source_name"] == "generic"
    assert "current_temperature" in event.telemetry
    assert "label" not in event.telemetry
    assert "type" not in event.telemetry
    assert "iot_telemetry" in event.feature_groups
    assert "current_temperature" in event.evidence_fields
    assert event.telemetry_context["telemetry_numeric_count"] >= 1
    assert event.anomaly_summary is not None


def test_generic_adapter_excludes_common_target_aliases_from_features():
    from src.adapters.generic import GenericAdapter

    aliases = {
        "ground_truth": 1,
        "target_value": 1,
        "class_id": 7,
        "outcome": "attack",
        "y": 1,
    }
    event = GenericAdapter().adapt(
        {**aliases, "temperature": 31.5},
        source_file="unknown.csv",
        row_id=2,
    )

    assert "temperature" in event.telemetry
    assert aliases.keys().isdisjoint(event.telemetry)
    assert all(f"{key}=" not in (event.semantic_text or "") for key in aliases)


def test_generic_adapter_adds_schema_context_without_dataset_rules():
    from src.adapters.generic import GenericAdapter

    event = GenericAdapter().adapt(
        {
            "src": "10.0.0.5",
            "dst": "8.8.8.8",
            "dport": "53",
            "proto": "udp",
            "bytes": "2048",
        },
        source_file="network.csv",
        row_id=1,
    )

    assert event.traffic_direction == "internal_to_external"
    assert event.service_context["dst_common_service"] == "dns"
    assert "network_endpoint" in event.feature_groups
    assert "bytes" in event.evidence_fields


def test_ingest_preserves_dataset_family_as_origin():
    from src.agents.ingest_parser import IngestParserAgent

    event, _ = IngestParserAgent().ingest(
        {
            "dataset": "generic",
            "dataset_family": "TON_IOT_windows_host",
            "row": {"process": "svchost", "label": "0"},
            "source_file": "windows.csv",
            "row_id": 1,
        }
    )

    assert event.provenance.dataset == "TON_IOT_windows_host"


def test_bot_iot_and_edge_adapters_have_basic_coverage():
    bot = BotIoTAdapter().adapt(
        {"saddr": "1.1.1.1", "daddr": "2.2.2.2", "proto": "udp", "attack": "DDoS"},
        source_file="bot.csv",
        row_id=1,
    )
    edge = EdgeIIoTSetAdapter().adapt(
        {"ip.src_host": "1.1.1.1", "ip.dst_host": "2.2.2.2", "ip.proto": "tcp", "Attack_type": "Backdoor"},
        source_file="edge.csv",
        row_id=2,
    )

    assert bot.label_raw == "DDoS"
    assert edge.label_raw == "Backdoor"


def test_edge_adapter_normalizes_partial_year_time_and_rejects_invalid_timestamp():
    partial = EdgeIIoTSetAdapter().adapt(
        {"frame.time": " 2021 22:25:14.432732000 ", "ip.proto": "tcp"},
        source_file="edge.csv",
        row_id=1947,
    )
    invalid = EdgeIIoTSetAdapter().adapt(
        {"frame.time": "not-a-date", "ip.proto": "tcp"},
        source_file="edge.csv",
        row_id=1948,
    )

    assert partial.ts is not None
    assert partial.ts.isoformat() == "2021-01-01T22:25:14.432732"
    assert "ts" not in partial.missing_fields
    assert invalid.ts is None
    assert "ts" in invalid.missing_fields
