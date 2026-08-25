import pytest

from src.agents.explainer import LLMExplanationAgent
from src.contracts.agents import ClassificationOutput
from src.contracts.canonical import CanonicalEvent, Provenance


def sample_event():
    return CanonicalEvent(
        event_id="evt-1",
        modality="network_flow",
        schema_profile="network_flow",
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        dst_port=23,
        transport_proto="tcp",
        behavior_tags=["service.telnet"],
        attack_indicators=["remote_access_service_exposed"],
        semantic_text="Telnet traffic with malicious classification",
        provenance=Provenance(dataset="test"),
        mapping_confidence=0.9,
    )


def sample_classification():
    return ClassificationOutput(
        event_id="evt-1",
        attack_family="botnet",
        attack_subtype="mirai",
        confidence=0.86,
        reason=["classified as botnet"],
        next_route="explain",
    )


class StubExplanationBackend:
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
async def test_llm_explanation_agent_returns_structured_explanation():
    agent = LLMExplanationAgent(model="test-model")
    agent.agent = StubExplanationBackend({
        "event_id": "evt-1",
        "risk_summary": "Possible botnet activity over exposed Telnet.",
        "mitigations": ["isolate_device", "rotate_credentials"],
        "confidence": 0.82,
        "requires_human_review": False,
        "next_route": "judge",
        "model_name": "explainer-test",
    })

    output = await agent.explain_async(sample_event(), sample_classification())

    assert output.risk_summary.startswith("Possible botnet")
    assert output.mitigations[:2] == ["isolate_device", "rotate_credentials"]
    assert output.model_name == "llm_explainer::ollama::test-model"
    assert output.next_route == "judge"
    assert agent.agent.calls[0]["user_payload"]["event"]["behavior_tags"] == ["service.telnet"]
    assert "origin" not in agent.agent.calls[0]["user_payload"]["event"]
    assert "provenance" not in agent.agent.calls[0]["user_payload"]["event"]


@pytest.mark.asyncio
async def test_llm_explanation_agent_falls_back_to_rule_based_explainer():
    agent = LLMExplanationAgent(model="test-model")
    agent.agent = StubExplanationBackend(error=TimeoutError("slow explainer"))

    output = await agent.explain_async(sample_event(), sample_classification())

    assert "llm_explainer_failed=TimeoutError" in output.risk_summary
    assert "isolate_device" in output.mitigations
    assert output.model_name.startswith("llm_explainer::")
