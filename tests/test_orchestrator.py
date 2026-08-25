import pytest

from src.agents.classifier import RuleBasedClassifier
from src.agents.detector import RuleBasedDetector
from src.agents.explainer import RuleBasedExplainer
from src.agents.ingest_parser import IngestParserAgent
from src.agents.judge import RuleBasedJudge
from src.contracts.agents import DetectionOutput
from src.orchestration.graph import AgentBundle, AsyncGraphOrchestrator, LocalOrchestrator, build_async_graph, build_graph


def malicious_iot23_input():
    return {
        "dataset": "iot23",
        "source_file": "conn.log",
        "row_id": 1,
        "row": {
            "id.orig_h": "10.0.0.2",
            "id.resp_h": "10.0.0.3",
            "id.orig_p": 4444,
            "id.resp_p": 23,
            "proto": "tcp",
            "duration": "1.0",
            "orig_pkts": 50,
            "orig_ip_bytes": 4096,
            "label": "Mirai",
        },
    }


def test_local_orchestrator_approves_remote_access_signal():
    result = LocalOrchestrator().invoke({"raw_input": malicious_iot23_input()})

    assert result["route"] == "end"
    assert result["detection_output"]["is_malicious"] is True
    assert result["classification_output"]["attack_family"] == "bruteforce"
    assert result["judge_output"]["action"] == "approve"


def test_build_graph_executes_smoke_flow():
    graph = build_graph(checkpointer=None)
    result = graph.invoke({"raw_input": malicious_iot23_input(), "route": "ingest"})

    assert result["route"] == "end"
    assert result["judge_output"]["approved"] is True


def test_low_confidence_event_goes_to_human_interrupt():
    result = LocalOrchestrator().invoke(
        {
            "raw_input": {
                "dataset": "iot23",
                "source_file": "conn.log",
                "row_id": 99,
                "row": {"duration": "1.0"},
            }
        }
    )

    assert result["judge_output"]["action"] == "human_interrupt"
    assert "mapping_confidence_below_review_threshold" in result["judge_output"]["issues"]


@pytest.mark.asyncio
async def test_build_async_graph_executes_smoke_flow():
    graph = build_async_graph(checkpointer=None, use_llm=False)
    result = await graph.ainvoke({"raw_input": malicious_iot23_input(), "route": "ingest"})

    assert result["route"] == "end"
    assert result["judge_output"]["approved"] is True


@pytest.mark.asyncio
async def test_async_graph_orchestrator_executes_smoke_flow():
    result = await AsyncGraphOrchestrator(use_llm=False).ainvoke({"raw_input": malicious_iot23_input()})

    assert result["route"] == "end"
    assert result["classification_output"]["attack_family"] == "bruteforce"


class StubLLMDetector:
    async def detect_async(self, event):
        return DetectionOutput(
            event_id=event.event_id,
            is_malicious=True,
            probability=0.91,
            evidence=["stub_llm_detector"],
            model_name="stub_llm_detector",
            next_route="classify",
            abstain=False,
        )


class StubLLMClassifier:
    async def classify_async(self, event, detection):
        from src.contracts.agents import ClassificationOutput

        return ClassificationOutput(
            event_id=event.event_id,
            attack_family="botnet",
            attack_subtype="mirai",
            confidence=0.88,
            reason=["stub_llm_classifier"],
            next_route="explain",
        )


@pytest.mark.asyncio
async def test_async_pipeline_can_use_llm_detector_agent():
    agents = AgentBundle(
        ingest=IngestParserAgent(),
        detector=StubLLMDetector(),
        classifier=RuleBasedClassifier(),
        explainer=RuleBasedExplainer(),
        judge=RuleBasedJudge(),
    )

    result = await AsyncGraphOrchestrator(agents=agents, use_llm=False).ainvoke({"raw_input": malicious_iot23_input()})

    assert result["route"] == "end"
    assert result["detection_output"]["model_name"] == "stub_llm_detector"
    assert result["classification_output"]["attack_family"] == "bruteforce"


@pytest.mark.asyncio
async def test_async_pipeline_can_use_llm_classifier_agent():
    agents = AgentBundle(
        ingest=IngestParserAgent(),
        detector=RuleBasedDetector(),
        classifier=StubLLMClassifier(),
        explainer=RuleBasedExplainer(),
        judge=RuleBasedJudge(),
    )

    result = await AsyncGraphOrchestrator(agents=agents, use_llm=False).ainvoke({"raw_input": malicious_iot23_input()})

    assert result["route"] == "end"
    assert result["classification_output"]["attack_subtype"] == "mirai"
    assert result["classification_output"]["reason"] == ["stub_llm_classifier"]
