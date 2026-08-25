# src/orchestration/graph.py
from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Protocol

try:
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt
except ImportError:  # pragma: no cover - exercised only when langgraph is absent
    END = "__end__"
    START = "__start__"
    StateGraph = None
    Command = None

    def interrupt(payload: dict[str, Any]) -> dict[str, Any]:
        return {"interrupted": True, "payload": payload}

from src.agents.classifier import LLMClassificationAgent, RuleBasedClassifier
from src.agents.abstraction import BehaviorAbstractionAgent
from src.agents.detector import LLMDetectionAgent, RuleBasedDetector
from src.agents.explainer import LLMExplanationAgent, RuleBasedExplainer
from src.agents.ingest_parser import IngestParserAgent
from src.agents.judge import RuleBasedJudge
from src.agents.llm_ingest_parser import LLMIngestParser
from src.orchestration.state import OrchestratorState


class RunnableGraph(Protocol):
    def invoke(self, state: OrchestratorState, config: dict[str, Any] | None = None) -> OrchestratorState:
        ...


class AsyncRunnableGraph(Protocol):
    async def ainvoke(self, state: OrchestratorState, config: dict[str, Any] | None = None) -> OrchestratorState:
        ...


@dataclass
class AgentBundle:
    ingest: IngestParserAgent
    detector: Any
    classifier: Any
    explainer: Any
    judge: RuleBasedJudge
    abstraction: BehaviorAbstractionAgent = field(default_factory=BehaviorAbstractionAgent)


def default_agents(
    use_llm: bool = False,
    use_llm_detector: bool | None = None,
    use_llm_classifier: bool | None = None,
    use_llm_explainer: bool | None = None,
) -> AgentBundle:
    llm_parser = None
    if use_llm:
        llm_parser = LLMIngestParser(
            model=_llm_model("INGEST_LLM_MODEL"),
            base_url=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        )
    detector: Any = RuleBasedDetector()
    detector_enabled = use_llm_detector if use_llm_detector is not None else _env_flag("LLM_DETECTOR_ENABLED")
    if detector_enabled:
        detector = LLMDetectionAgent(
            model=_llm_model("DETECTOR_LLM_MODEL"),
            base_url=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            timeout_seconds=_optional_float_env("DETECTOR_LLM_TIMEOUT_SECONDS"),
            provider=os.getenv("DETECTOR_LLM_PROVIDER") or os.getenv("LLM_PROVIDER"),
            fallback_detector=RuleBasedDetector(),
        )
    classifier: Any = RuleBasedClassifier()
    classifier_enabled = use_llm_classifier if use_llm_classifier is not None else _env_flag("LLM_CLASSIFIER_ENABLED")
    if classifier_enabled:
        classifier = LLMClassificationAgent(
            model=_llm_model("CLASSIFIER_LLM_MODEL"),
            base_url=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            timeout_seconds=_optional_float_env("CLASSIFIER_LLM_TIMEOUT_SECONDS"),
            provider=os.getenv("CLASSIFIER_LLM_PROVIDER") or os.getenv("LLM_PROVIDER"),
            fallback_classifier=RuleBasedClassifier(),
        )
    explainer: Any = RuleBasedExplainer()
    explainer_enabled = use_llm_explainer if use_llm_explainer is not None else _env_flag("LLM_EXPLAINER_ENABLED")
    if explainer_enabled:
        explainer = LLMExplanationAgent(
            model=_llm_model("EXPLAINER_LLM_MODEL"),
            base_url=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            timeout_seconds=_optional_float_env("EXPLAINER_LLM_TIMEOUT_SECONDS"),
            provider=os.getenv("EXPLAINER_LLM_PROVIDER") or os.getenv("LLM_PROVIDER"),
            fallback_explainer=RuleBasedExplainer(),
        )
    return AgentBundle(
        ingest=IngestParserAgent(llm_parser=llm_parser),
        detector=detector,
        classifier=classifier,
        explainer=explainer,
        judge=RuleBasedJudge(),
        abstraction=BehaviorAbstractionAgent(),
    )


def ingest_node(state: OrchestratorState, agents: AgentBundle | None = None) -> OrchestratorState:
    agents = agents or default_agents()
    event, output = agents.ingest.ingest(state.get("raw_input", {}))
    event = agents.abstraction.abstract(event)
    route = "judge" if output.mapping_confidence < 0.5 else "detect"
    return {
        "event_id": event.event_id,
        "canonical_event": event.model_dump(mode="json"),
        "ingest_output": output.model_dump(mode="json"),
        "route": route,
    }


async def ingest_node_async(state: OrchestratorState, agents: AgentBundle | None = None) -> OrchestratorState:
    raw_input = state.get("raw_input", {})
    agents = agents or default_agents(
        use_llm=raw_input.get("use_llm") is not False,
        use_llm_detector=raw_input.get("use_llm_detector"),
        use_llm_classifier=raw_input.get("use_llm_classifier"),
        use_llm_explainer=raw_input.get("use_llm_explainer"),
    )
    event, output = await agents.ingest.ingest_async(state.get("raw_input", {}))
    event = agents.abstraction.abstract(event)
    route = "judge" if output.mapping_confidence < 0.5 else "detect"
    return {
        "event_id": event.event_id,
        "canonical_event": event.model_dump(mode="json"),
        "ingest_output": output.model_dump(mode="json"),
        "route": route,
    }


def detect_node(state: OrchestratorState, agents: AgentBundle | None = None) -> OrchestratorState:
    agents = agents or default_agents()
    event = _canonical_from_state(state)
    output = agents.detector.detect(event)
    return _detection_update(output)


async def detect_node_async(state: OrchestratorState, agents: AgentBundle | None = None) -> OrchestratorState:
    raw_input = state.get("raw_input", {})
    agents = agents or default_agents(
        use_llm=raw_input.get("use_llm") is not False,
        use_llm_detector=raw_input.get("use_llm_detector"),
        use_llm_classifier=raw_input.get("use_llm_classifier"),
        use_llm_explainer=raw_input.get("use_llm_explainer"),
    )
    event = _canonical_from_state(state)
    if hasattr(agents.detector, "detect_async"):
        output = await agents.detector.detect_async(event)
    else:
        output = agents.detector.detect(event)
    return _detection_update(output)


def _detection_update(output) -> OrchestratorState:
    if output.abstain or output.next_route == "judge":
        route = "judge"
    elif output.is_malicious:
        route = "classify"
    else:
        route = "end"
    return {"detection_output": output.model_dump(mode="json"), "route": route}


def classify_node(state: OrchestratorState, agents: AgentBundle | None = None) -> OrchestratorState:
    agents = agents or default_agents()
    event = _canonical_from_state(state)
    detection = state["detection_output"]
    output = agents.classifier.classify(event, _detection_from_dict(detection))
    return _classification_update(output)


async def classify_node_async(state: OrchestratorState, agents: AgentBundle | None = None) -> OrchestratorState:
    raw_input = state.get("raw_input", {})
    agents = agents or default_agents(
        use_llm=raw_input.get("use_llm") is not False,
        use_llm_detector=raw_input.get("use_llm_detector"),
        use_llm_classifier=raw_input.get("use_llm_classifier"),
        use_llm_explainer=raw_input.get("use_llm_explainer"),
    )
    event = _canonical_from_state(state)
    detection = _detection_from_dict(state["detection_output"])
    if hasattr(agents.classifier, "classify_async"):
        output = await agents.classifier.classify_async(event, detection)
    else:
        output = agents.classifier.classify(event, detection)
    return _classification_update(output)


def _classification_update(output) -> OrchestratorState:
    route = "judge" if output.next_route == "judge" else output.next_route
    return {"classification_output": output.model_dump(mode="json"), "route": route}


def explain_node(state: OrchestratorState, agents: AgentBundle | None = None) -> OrchestratorState:
    agents = agents or default_agents()
    event = _canonical_from_state(state)
    output = agents.explainer.explain(event, _classification_from_dict(state["classification_output"]))
    return {
        "explanation_output": output.model_dump(mode="json"),
        "needs_human_review": output.requires_human_review,
        "route": "judge",
    }


async def explain_node_async(state: OrchestratorState, agents: AgentBundle | None = None) -> OrchestratorState:
    raw_input = state.get("raw_input", {})
    agents = agents or default_agents(
        use_llm=raw_input.get("use_llm") is not False,
        use_llm_detector=raw_input.get("use_llm_detector"),
        use_llm_classifier=raw_input.get("use_llm_classifier"),
        use_llm_explainer=raw_input.get("use_llm_explainer"),
    )
    event = _canonical_from_state(state)
    classification = _classification_from_dict(state["classification_output"])
    if hasattr(agents.explainer, "explain_async"):
        output = await agents.explainer.explain_async(event, classification)
    else:
        output = agents.explainer.explain(event, classification)
    return {
        "explanation_output": output.model_dump(mode="json"),
        "needs_human_review": output.requires_human_review,
        "route": "judge",
    }


def judge_node(state: OrchestratorState, agents: AgentBundle | None = None) -> OrchestratorState:
    agents = agents or default_agents()
    output = agents.judge.judge(state)
    update: OrchestratorState = {"judge_output": output.model_dump(mode="json")}
    if output.action == "human_interrupt":
        update["human_decision"] = _safe_interrupt({
            "event_id": state.get("event_id"),
            "reason": "low_confidence_or_high_impact",
            "judge_output": output.model_dump(mode="json"),
        })
        update["route"] = "end"
    elif output.action in {"approve", "reject"}:
        update["route"] = "end"
    else:
        update["route"] = "detect"
    return update


def router(state: OrchestratorState) -> str:
    return state.get("route", "ingest")


class LocalOrchestrator:
    def __init__(self, agents: AgentBundle | None = None, max_steps: int = 12):
        self.agents = agents or default_agents()
        self.max_steps = max_steps

    def invoke(self, state: OrchestratorState, config: dict[str, Any] | None = None) -> OrchestratorState:
        del config
        current: OrchestratorState = dict(state)
        current.setdefault("route", "ingest")
        for _ in range(self.max_steps):
            route = router(current)
            if route == "end":
                return current
            if route == "ingest":
                update = ingest_node(current, self.agents)
            elif route == "detect":
                update = detect_node(current, self.agents)
            elif route == "classify":
                update = classify_node(current, self.agents)
            elif route == "explain":
                update = explain_node(current, self.agents)
            elif route == "judge":
                update = judge_node(current, self.agents)
            else:
                raise ValueError(f"Unknown route: {route}")
            current.update(update)
        raise RuntimeError("Orchestrator reached max_steps without ending")


class AsyncLocalOrchestrator:
    def __init__(
        self,
        agents: AgentBundle | None = None,
        use_llm: bool = True,
        use_llm_detector: bool | None = None,
        use_llm_classifier: bool | None = None,
        use_llm_explainer: bool | None = None,
        max_steps: int = 12,
    ):
        self.agents = agents or default_agents(
            use_llm=use_llm,
            use_llm_detector=use_llm_detector,
            use_llm_classifier=use_llm_classifier,
            use_llm_explainer=use_llm_explainer,
        )
        self.max_steps = max_steps

    async def ainvoke(self, state: OrchestratorState, config: dict[str, Any] | None = None) -> OrchestratorState:
        del config
        current: OrchestratorState = dict(state)
        current.setdefault("route", "ingest")
        for _ in range(self.max_steps):
            route = router(current)
            if route == "end":
                return current
            if route == "ingest":
                update = await ingest_node_async(current, self.agents)
            elif route == "detect":
                update = await detect_node_async(current, self.agents)
            elif route == "classify":
                update = await classify_node_async(current, self.agents)
            elif route == "explain":
                update = await explain_node_async(current, self.agents)
            elif route == "judge":
                update = judge_node(current, self.agents)
            else:
                raise ValueError(f"Unknown route: {route}")
            current.update(update)
        raise RuntimeError("Orchestrator reached max_steps without ending")


class AsyncGraphOrchestrator:
    def __init__(
        self,
        checkpointer: Any = None,
        agents: AgentBundle | None = None,
        use_llm: bool = True,
        use_llm_detector: bool | None = None,
        use_llm_classifier: bool | None = None,
        use_llm_explainer: bool | None = None,
        max_steps: int = 12,
    ):
        self.graph = build_async_graph(
            checkpointer=checkpointer,
            agents=agents,
            use_llm=use_llm,
            use_llm_detector=use_llm_detector,
            use_llm_classifier=use_llm_classifier,
            use_llm_explainer=use_llm_explainer,
        )
        self.max_steps = max_steps

    async def ainvoke(self, state: OrchestratorState, config: dict[str, Any] | None = None) -> OrchestratorState:
        merged_config = {"recursion_limit": self.max_steps}
        if config:
            merged_config.update(config)
        return await self.graph.ainvoke(state, config=merged_config)


def build_graph(checkpointer: Any = None, agents: AgentBundle | None = None) -> RunnableGraph:
    agents = agents or default_agents()
    if StateGraph is None:
        return LocalOrchestrator(agents)

    graph = StateGraph(OrchestratorState)
    graph.add_node("ingest", lambda state: ingest_node(state, agents))
    graph.add_node("detect", lambda state: detect_node(state, agents))
    graph.add_node("classify", lambda state: classify_node(state, agents))
    graph.add_node("explain", lambda state: explain_node(state, agents))
    graph.add_node("judge", lambda state: judge_node(state, agents))

    graph.add_conditional_edges(START, router, _route_map())
    for node in ["ingest", "detect", "classify", "explain", "judge"]:
        graph.add_conditional_edges(node, router, _route_map())
    return graph.compile(checkpointer=checkpointer)


def build_async_graph(
    checkpointer: Any = None,
    agents: AgentBundle | None = None,
    use_llm: bool = True,
    use_llm_detector: bool | None = None,
    use_llm_classifier: bool | None = None,
    use_llm_explainer: bool | None = None,
) -> AsyncRunnableGraph:
    agents = agents or default_agents(
        use_llm=use_llm,
        use_llm_detector=use_llm_detector,
        use_llm_classifier=use_llm_classifier,
        use_llm_explainer=use_llm_explainer,
    )
    if StateGraph is None:
        return AsyncLocalOrchestrator(agents)

    async def async_ingest(state: OrchestratorState) -> OrchestratorState:
        return await ingest_node_async(state, agents)

    async def async_detect(state: OrchestratorState) -> OrchestratorState:
        return await detect_node_async(state, agents)

    async def async_classify(state: OrchestratorState) -> OrchestratorState:
        return await classify_node_async(state, agents)

    async def async_explain(state: OrchestratorState) -> OrchestratorState:
        return await explain_node_async(state, agents)

    graph = StateGraph(OrchestratorState)
    graph.add_node("ingest", async_ingest)
    graph.add_node("detect", async_detect)
    graph.add_node("classify", async_classify)
    graph.add_node("explain", async_explain)
    graph.add_node("judge", lambda state: judge_node(state, agents))

    graph.add_conditional_edges(START, router, _route_map())
    for node in ["ingest", "detect", "classify", "explain", "judge"]:
        graph.add_conditional_edges(node, router, _route_map())
    return graph.compile(checkpointer=checkpointer)


def _route_map() -> dict[str, str]:
    return {
        "ingest": "ingest",
        "detect": "detect",
        "classify": "classify",
        "explain": "explain",
        "judge": "judge",
        "end": END,
    }


def _env_flag(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in {"1", "true", "yes", "on"}


def _optional_float_env(name: str) -> float | None:
    value = os.getenv(name)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _llm_model(specific_env: str) -> str:
    if os.getenv(specific_env):
        return os.getenv(specific_env, "")
    if os.getenv("OLLAMA_AGENT_MODEL"):
        return os.getenv("OLLAMA_AGENT_MODEL", "")
    if (os.getenv("LLM_PROVIDER") or "").strip().lower() == "transformers":
        return os.getenv("TRANSFORMERS_AGENT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    if (os.getenv("LLM_PROVIDER") or "").strip().lower() == "openrouter":
        return os.getenv("OPENROUTER_AGENT_MODEL", "openai/gpt-oss-120b:free")
    return "gemma3:12b"


def _safe_interrupt(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return interrupt(payload)
    except RuntimeError:
        return {"interrupted": True, "payload": payload}


def _canonical_from_state(state: OrchestratorState):
    from src.contracts.canonical import CanonicalEvent

    return CanonicalEvent(**state["canonical_event"])


def _detection_from_dict(data: dict[str, Any]):
    from src.contracts.agents import DetectionOutput

    return DetectionOutput(**data)


def _classification_from_dict(data: dict[str, Any]):
    from src.contracts.agents import ClassificationOutput

    return ClassificationOutput(**data)
