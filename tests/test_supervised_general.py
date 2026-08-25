from src.agents.supervised_general import GeneralizedFeatureStandardizer
from src.contracts.canonical import CanonicalEvent, Provenance


def test_generalized_standardizer_can_include_origin_features():
    event = CanonicalEvent(
        event_id="generic::edge::1",
        modality="network_flow",
        telemetry={"tcp.dstport": "80", "ip.proto": "6"},
        semantic_text="network sample",
        provenance=Provenance(
            dataset="EDGE_IIOTSET",
            source_file="data/EDGE_IIOTSET/ML-EdgeIIoT-dataset.csv",
        ),
        mapping_confidence=0.9,
    )

    without_origin = GeneralizedFeatureStandardizer(include_origin=False).event_to_features(event)
    with_origin = GeneralizedFeatureStandardizer(include_origin=True).event_to_features(event)

    assert "origin.dataset" not in without_origin
    assert with_origin["origin.dataset"] == "edge_iiotset"
    assert with_origin["origin_schema.network_packet"] == 1


def test_generalized_standardizer_uses_schema_context_features():
    event = CanonicalEvent(
        event_id="generic::telemetry::1",
        modality="telemetry",
        schema_profile="iot_telemetry",
        telemetry={"current_temperature": "42.5", "thermostat_status": "1"},
        feature_groups={"iot_telemetry": ["current_temperature", "thermostat_status"]},
        evidence_fields=["current_temperature", "thermostat_status"],
        telemetry_context={
            "telemetry_numeric_count": 2,
            "telemetry_range": 41.5,
            "has_sensor_state": True,
        },
        behavior_tags=["sensor_value_shift"],
        attack_indicators=["telemetry_anomaly"],
        asset_context="iot_device",
        uncertainty=["missing_time"],
        anomaly_summary="schema_profile=iot_telemetry; sensor_state=active",
        semantic_text="thermostat telemetry sample",
        provenance=Provenance(dataset="generic"),
        mapping_confidence=0.9,
    )

    features = GeneralizedFeatureStandardizer().event_to_features(event)

    assert features["schema_profile.iot_telemetry"] == 1
    assert features["context_group.iot_telemetry"] == 1
    assert features["context.telemetry.telemetry_numeric_count"] == 2
    assert features["evidence_group.iot_telemetry"] == 1
    assert features["behavior.sensor_value_shift"] == 1
    assert features["attack_indicator.telemetry_anomaly"] == 1
    assert features["asset_context.iot_device"] == 1
    assert features["uncertainty.missing_time"] == 1

    abstract_features = GeneralizedFeatureStandardizer(feature_set="abstract").event_to_features(event)
    assert abstract_features["behavior.sensor_value_shift"] == 1
    assert "current_temperature" not in abstract_features
