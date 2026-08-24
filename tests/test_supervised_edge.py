from src.agents.supervised_edge import (
    EdgeSupervisedClassifier,
    EdgeSupervisedDetector,
    EdgeSupervisedModelBundle,
    TechnicalFeatureStandardizer,
)
from src.agents.explainer import RuleBasedExplainer
from src.contracts.agents import ClassificationOutput
from src.contracts.canonical import CanonicalEvent, Provenance


class DummyDetectorPipeline:
    classes_ = [False, True]

    def predict_proba(self, rows):
        return [[0.2, 0.8] for _ in rows]


class DummyClassifierPipeline:
    classes_ = ["Normal", "DDoS_UDP"]

    def predict(self, rows):
        return ["DDoS_UDP" for _ in rows]

    def predict_proba(self, rows):
        return [[0.1, 0.9] for _ in rows]


def test_technical_standardizer_strips_labels_and_handles_large_values():
    features = TechnicalFeatureStandardizer().row_to_features(
        {
            "Attack_label": "1",
            "Attack_type": "DDoS_UDP",
            "tcp.dstport": "80",
            "huge_numeric_identifier": "1e309",
        }
    )

    assert "Attack_label" not in features
    assert "Attack_type" not in features
    assert features["tcp.dstport"] == 80
    assert features["huge_numeric_identifier"] == "1e309"


def test_edge_supervised_agents_emit_pipeline_outputs():
    event = CanonicalEvent(
        event_id="edge::1",
        modality="network_flow",
        telemetry={"tcp.dstport": "80"},
        semantic_text="edge sample",
        provenance=Provenance(dataset="EDGE_IIOTSET"),
        mapping_confidence=0.9,
    )
    bundle = EdgeSupervisedModelBundle(
        detector_pipeline=DummyDetectorPipeline(),
        classifier_pipeline=DummyClassifierPipeline(),
        model_name="dummy",
    )

    detection = EdgeSupervisedDetector(bundle).detect(event)
    classification = EdgeSupervisedClassifier(bundle).classify(event, detection)

    assert detection.is_malicious is True
    assert detection.next_route == "classify"
    assert detection.probability == 0.8
    assert classification.attack_family == "ddos"
    assert classification.attack_subtype == "DDoS_UDP"
    assert classification.confidence == 0.9


def test_edge_supervised_detector_abstains_outside_edge_domain():
    event = CanonicalEvent(
        event_id="ton::1",
        modality="telemetry",
        telemetry={"current_temperature": "29.2", "thermostat_status": "1"},
        semantic_text="thermostat sample",
        provenance=Provenance(dataset="TON_IOT", source_file="thermostat.csv"),
        mapping_confidence=0.9,
    )
    bundle = EdgeSupervisedModelBundle(
        detector_pipeline=DummyDetectorPipeline(),
        classifier_pipeline=DummyClassifierPipeline(),
        model_name="dummy",
    )

    detection = EdgeSupervisedDetector(bundle).detect(event)
    classification = EdgeSupervisedClassifier(bundle).classify(event, detection)

    assert detection.abstain is True
    assert detection.next_route == "judge"
    assert "edge_domain_guard=out_of_domain" in detection.evidence
    assert classification.next_route == "judge"


def test_explainer_has_specific_mitm_mitigations():
    event = CanonicalEvent(
        event_id="edge::mitm",
        modality="network_flow",
        semantic_text="mitm sample",
        provenance=Provenance(dataset="EDGE_IIOTSET"),
        mapping_confidence=0.9,
    )
    classification = ClassificationOutput(
        event_id=event.event_id,
        attack_family="mitm",
        attack_subtype="MITM",
        confidence=0.9,
        next_route="explain",
    )

    explanation = RuleBasedExplainer().explain(event, classification)

    assert "inspect_arp_tables" in explanation.mitigations
