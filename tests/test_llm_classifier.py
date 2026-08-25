import pytest

from src.agents.classifier import LLMClassificationAgent
from src.contracts.agents import DetectionOutput
from src.contracts.canonical import CanonicalEvent, Provenance


def sample_event(label_raw="Mirai", attack_indicators=None):
    return CanonicalEvent(
        event_id="evt-1",
        modality="network_flow",
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        dst_port=23,
        transport_proto="tcp",
        label_raw=label_raw,
        attack_indicators=attack_indicators or [],
        semantic_text="Possible Mirai telnet activity",
        provenance=Provenance(dataset="test"),
        mapping_confidence=0.9,
    )


def malicious_detection():
    return DetectionOutput(
        event_id="evt-1",
        is_malicious=True,
        probability=0.91,
        evidence=["detected as malicious"],
        model_name="test-detector",
        next_route="classify",
    )


class StubClassificationBackend:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = []

    async def invoke_json(self, system_prompt, user_payload, json_schema):
        self.calls.append({
            "system_prompt": system_prompt,
            "user_payload": user_payload,
            "json_schema": json_schema,
        })
        if self.error:
            raise self.error
        return self.payload


@pytest.mark.asyncio
async def test_llm_classification_agent_returns_structured_classification():
    agent = LLMClassificationAgent(model="test-model")
    agent.agent = StubClassificationBackend({
        "event_id": "evt-1",
        "attack_family": "botnet",
        "attack_subtype": "mirai",
        "confidence": 0.87,
        "cross_dataset_neighbors": ["iot23:mirai"],
        "reason": ["mirai label"],
        "next_route": "explain",
    })

    output = await agent.classify_async(sample_event(), malicious_detection())

    assert output.attack_family == "botnet"
    assert output.attack_subtype == "mirai"
    assert output.next_route == "explain"
    assert output.reason[-1].startswith("llm_classifier_model=")
    assert agent.agent.calls[0]["user_payload"]["detection"]["is_malicious"] is True
    assert "label_raw" not in agent.agent.calls[0]["user_payload"]["event"]
    assert "attack_family" not in agent.agent.calls[0]["user_payload"]["event"]
    assert "attack_subtype" not in agent.agent.calls[0]["user_payload"]["event"]
    assert "origin" not in agent.agent.calls[0]["user_payload"]["event"]
    assert "provenance" not in agent.agent.calls[0]["user_payload"]["event"]


@pytest.mark.asyncio
async def test_llm_classification_agent_falls_back_to_rule_based_classifier():
    agent = LLMClassificationAgent(model="test-model")
    agent.agent = StubClassificationBackend(error=TimeoutError("slow classifier"))

    output = await agent.classify_async(sample_event(attack_indicators=["botnet_pattern"]), malicious_detection())

    assert output.attack_family == "botnet"
    assert output.next_route == "explain"
    assert output.reason[0] == "llm_classifier_failed=TimeoutError"
