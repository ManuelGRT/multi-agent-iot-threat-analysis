import pytest

from src.agents.detector import LLMDetectionAgent
from src.contracts.canonical import CanonicalEvent, Provenance


def sample_event(label_raw="Mirai", mapping_confidence=0.9):
    return CanonicalEvent(
        event_id="evt-1",
        modality="network_flow",
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        dst_port=23,
        transport_proto="tcp",
        label_raw=label_raw,
        semantic_text="Possible Mirai telnet activity",
        provenance=Provenance(dataset="test"),
        mapping_confidence=mapping_confidence,
    )


class StubDetectionBackend:
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
async def test_llm_detection_agent_returns_structured_detection():
    agent = LLMDetectionAgent(model="test-model")
    agent.agent = StubDetectionBackend({
        "event_id": "evt-1",
        "is_malicious": True,
        "probability": 0.88,
        "evidence": ["attack label and telnet port"],
        "model_name": "llm-test",
        "next_route": "classify",
        "abstain": False,
    })

    output = await agent.detect_async(sample_event())

    assert output.is_malicious is True
    assert output.next_route == "classify"
    assert output.model_name == "llm_detector::ollama::test-model"
    assert output.evidence[-1] == "llm_reported_model_name=llm-test"
    assert agent.agent.calls[0]["user_payload"]["network"]["dst_port"] == 23
    assert "labels" not in agent.agent.calls[0]["user_payload"]
    assert "detector_policy" not in agent.agent.calls[0]["user_payload"]
    assert "origin" not in agent.agent.calls[0]["user_payload"]
    assert "provenance" not in agent.agent.calls[0]["user_payload"]


@pytest.mark.asyncio
async def test_llm_detection_agent_scrubs_target_like_text_from_payload():
    agent = LLMDetectionAgent(model="test-model")
    agent.agent = StubDetectionBackend({
        "event_id": "evt-1",
        "is_malicious": False,
        "probability": 0.2,
        "evidence": ["weak technical signal"],
        "model_name": "detector-test",
        "next_route": "end",
        "abstain": False,
    })
    event = sample_event(label_raw="DDoS_UDP").model_copy(update={
        "semantic_text": "src_ip=10.0.0.1 label=DDoS_UDP | Attack_type=DDoS_UDP | proto=tcp",
        "telemetry": {"raw_text": "dst_ip=10.0.0.2 label=normal", "label": "normal"},
    })

    await agent.detect_async(event)

    user_payload = agent.agent.calls[0]["user_payload"]
    assert "DDoS_UDP" not in user_payload["semantic_text"]
    assert "label=" not in user_payload["telemetry"]["raw_text"]
    assert "label" not in user_payload["telemetry"]


@pytest.mark.asyncio
async def test_llm_detection_agent_falls_back_to_rule_based_detector():
    agent = LLMDetectionAgent(model="test-model")
    agent.agent = StubDetectionBackend(error=TimeoutError("slow detector"))

    output = await agent.detect_async(sample_event())

    assert output.is_malicious is True
    assert output.next_route == "judge"
    assert output.model_name.startswith("llm_detector::")
    assert output.evidence[0] == "llm_detector_failed=TimeoutError"


@pytest.mark.asyncio
async def test_llm_detection_agent_does_not_override_with_hidden_label():
    agent = LLMDetectionAgent(model="test-model")
    agent.agent = StubDetectionBackend({
        "event_id": "evt-1",
        "is_malicious": False,
        "probability": 0.0,
        "evidence": ["missing packet counters"],
        "model_name": "detector-test",
        "next_route": "end",
        "abstain": False,
    })

    output = await agent.detect_async(sample_event(label_raw="DDoS_UDP"))

    assert output.is_malicious is False
    assert output.next_route == "end"
    assert output.abstain is False
    assert output.probability == 0.0
    assert not any(item.startswith("label_detection_guard=") for item in output.evidence)
    assert "labels" not in agent.agent.calls[0]["user_payload"]


@pytest.mark.asyncio
async def test_llm_detection_agent_does_not_force_benign_label_to_malicious():
    agent = LLMDetectionAgent(model="test-model")
    agent.agent = StubDetectionBackend({
        "event_id": "evt-1",
        "is_malicious": False,
        "probability": 0.05,
        "evidence": ["benign label"],
        "model_name": "detector-test",
        "next_route": "end",
        "abstain": False,
    })

    output = await agent.detect_async(sample_event(label_raw="normal"))

    assert output.is_malicious is False
    assert output.next_route == "end"
    assert not any(item.startswith("label_detection_guard=") for item in output.evidence)
    assert "labels" not in agent.agent.calls[0]["user_payload"]


@pytest.mark.asyncio
async def test_llm_detection_agent_keeps_llm_probability_without_label_floor():
    agent = LLMDetectionAgent(model="test-model")
    agent.agent = StubDetectionBackend({
        "event_id": "evt-1",
        "is_malicious": True,
        "probability": 0.52,
        "evidence": ["attack label but sparse telemetry"],
        "model_name": "detector-test",
        "next_route": "classify",
        "abstain": False,
    })

    output = await agent.detect_async(sample_event(label_raw="mitm", mapping_confidence=0.62))

    assert output.is_malicious is True
    assert output.next_route == "classify"
    assert output.probability == 0.52
    assert not any(item.startswith("label_detection_probability_floor=") for item in output.evidence)
