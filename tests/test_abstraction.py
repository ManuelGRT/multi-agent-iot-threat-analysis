from src.agents.abstraction import BehaviorAbstractionAgent
from src.contracts.canonical import CanonicalEvent, Provenance


def test_behavior_abstraction_adds_generic_network_tags():
    event = CanonicalEvent(
        event_id="evt-network",
        modality="network_flow",
        schema_profile="network_flow",
        src_ip="10.0.0.2",
        dst_ip="8.8.8.8",
        dst_port=53,
        transport_proto="udp",
        packet_count=2,
        traffic_direction="internal_to_external",
        service_context={"dst_common_service": "dns"},
        semantic_text="network event",
        provenance=Provenance(dataset="generic"),
        mapping_confidence=0.9,
    )

    enriched = BehaviorAbstractionAgent().abstract(event)

    assert "communication_observed" in enriched.behavior_tags
    assert "service.dns" in enriched.behavior_tags
    assert "service_probe_pattern" in enriched.attack_indicators
    assert enriched.asset_context == "network_service"
    assert "profile=network_flow" in (enriched.anomaly_summary or "")


def test_behavior_abstraction_adds_telemetry_anomaly_tags():
    event = CanonicalEvent(
        event_id="evt-telemetry",
        modality="telemetry",
        schema_profile="iot_telemetry",
        telemetry={"current_temperature": "22.0", "target_temperature": "80.0"},
        telemetry_context={
            "telemetry_field_count": 2,
            "telemetry_numeric_count": 2,
            "telemetry_range": 58.0,
            "has_sensor_state": True,
        },
        semantic_text="sensor telemetry",
        provenance=Provenance(dataset="generic"),
        mapping_confidence=0.9,
    )

    enriched = BehaviorAbstractionAgent().abstract(event)

    assert "sensor_state_observed" in enriched.behavior_tags
    assert "sensor_value_shift" in enriched.behavior_tags
    assert "telemetry_anomaly" in enriched.attack_indicators
    assert enriched.asset_context == "iot_device"
