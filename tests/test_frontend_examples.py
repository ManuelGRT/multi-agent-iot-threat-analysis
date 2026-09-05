"""Fixtures del frontend: cobertura, contrato HTTP y ausencia de targets."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from src.api.app import app
from src.api.routers import CaseAnalyzeRequest
from src.contracts.leakage import (
    is_predictive_target_field,
    is_predictive_target_value,
)


INDEX_HTML = Path(__file__).resolve().parents[1] / "src" / "api" / "static" / "index.html"
AUDITOR_SOURCE = (
    Path(__file__).resolve().parents[1] / "src" / "agents" / "final" / "auditor.py"
)


def test_frontend_uses_requested_centered_title() -> None:
    html = TestClient(app).get("/").text

    assert (
        "<h1>Sistema Multiagente para Amenazas de Ciberseguridad "
        "en Entornos IoT/IIoT</h1>"
    ) in html
    assert (
        "header h1 { margin: 0; font-size: 18px; font-weight: 600; "
        "text-align: center; }"
    ) in html


class _ExamplesScriptParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._inside_examples = False
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "script" and attributes.get("id") == "demo-examples":
            assert attributes.get("type") == "application/json"
            self._inside_examples = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._inside_examples:
            self._inside_examples = False

    def handle_data(self, data: str) -> None:
        if self._inside_examples:
            self._parts.append(data)

    @property
    def payload(self) -> str:
        return "".join(self._parts)


def _load_examples() -> dict[str, dict[str, Any]]:
    parser = _ExamplesScriptParser()
    parser.feed(INDEX_HTML.read_text(encoding="utf-8"))
    assert parser.payload.strip(), "falta el bloque JSON demo-examples"
    examples = json.loads(parser.payload)
    assert isinstance(examples, dict) and examples
    return examples


def _walk_mapping(value: Any, path: str = ""):
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield child_path, key, item
            yield from _walk_mapping(item, child_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_mapping(item, f"{path}[{index}]")


def test_demo_examples_cover_benign_and_attack_per_dataset_without_subtypes():
    examples = _load_examples()
    profiles_by_dataset: dict[str, set[str]] = defaultdict(set)
    outcomes_by_dataset: dict[str, set[str]] = defaultdict(set)
    dataset_labels = {
        "iot23": "IoT-23",
        "bot_iot": "BoT-IoT",
        "edge_iiotset": "Edge-IIoTset",
        "ton_iot": "TON-IoT",
        "urban_iot": "Urban-IoT",
    }
    outcome_labels = {
        "benign": "Benigno",
        "attack": "Ataque",
    }

    for name, spec in examples.items():
        assert set(spec) == {"expected_profile", "expected_outcome", "payload"}, name
        payload = spec["payload"]
        CaseAnalyzeRequest.model_validate(payload)
        assert sum(
            payload.get(key) is not None
            for key in ("row", "text", "canonical_event")
        ) == 1, name

        if payload["dataset"] == "generic":
            assert name == "JSON libre"
            assert spec["expected_outcome"] == "custom"
            continue

        dataset = payload["dataset"]
        outcome = spec["expected_outcome"]
        assert dataset in dataset_labels, name
        assert outcome in outcome_labels, name
        assert name == f"[{dataset_labels[dataset]}] {outcome_labels[outcome]}"
        assert "row" in payload, f"{name} debe contener una entrada cruda"
        assert payload["row"], name
        outcomes_by_dataset[dataset].add(outcome)
        profiles_by_dataset[dataset].add(spec["expected_profile"])

    assert dict(outcomes_by_dataset) == {
        dataset: {"benign", "attack"}
        for dataset in dataset_labels
    }
    assert dict(profiles_by_dataset) == {
        "iot23": {"network_flow"},
        "bot_iot": {"network_flow"},
        "edge_iiotset": {"network_packet"},
        "ton_iot": {"network_flow"},
        "urban_iot": {"iot_telemetry", "network_flow"},
    }

    labels = "\n".join(examples)
    for forbidden in (
        "Benigno real",
        "Ataque real",
        " · ",
        "/red",
        "/host",
        "/telemetría",
        "botnet",
        "reconocimiento",
        "DDoS",
        "escaneo",
        "inyección",
        "ransomware",
        "Abstención",
    ):
        assert forbidden not in labels


def test_demo_payloads_are_target_free_before_entering_the_multiagent_system():
    for name, spec in _load_examples().items():
        payload = spec["payload"]
        assert str(payload.get("source_file", "")).startswith("demo_"), name

        for path, key, value in _walk_mapping(payload):
            assert not is_predictive_target_field(key), (
                f"{name}: campo target en payload: {path}"
            )
            if isinstance(value, str):
                assert not is_predictive_target_value(value), (
                    f"{name}: valor target en payload: {path}={value!r}"
                )


def test_frontend_sends_only_each_example_payload_to_cases_analyze(monkeypatch):
    from src.api import routers

    received: list[tuple[dict[str, Any], bool | None, bool]] = []

    def fake_run(
        raw_input: dict[str, Any],
        *,
        use_llm_mitigator: bool | None = None,
        persist: bool = False,
    ) -> dict[str, Any]:
        received.append((raw_input, use_llm_mitigator, persist))
        return {"ok": True}

    monkeypatch.setattr(routers, "_run_final_case", fake_run)
    client = TestClient(app)

    for name, spec in _load_examples().items():
        request_payload = spec["payload"]
        response = client.post("/cases/analyze", json=request_payload)
        assert response.status_code == 200, name

        validated = CaseAnalyzeRequest.model_validate(request_payload)
        expected_raw = validated.model_dump(mode="json")
        raw_input, use_llm_mitigator, persist = received[-1]
        assert raw_input == expected_raw, name
        assert use_llm_mitigator is True
        assert persist is True

    first_wrapper = next(iter(_load_examples().values()))
    assert client.post("/cases/analyze", json=first_wrapper).status_code == 422


def test_frontend_prioritizes_abstention_over_internal_binary_class():
    html = TestClient(app).get("/").text

    assert "Ejemplo precargado" not in html
    assert '<label for="ejemplo">' not in html
    assert "const verd = det.abstain ?" in html
    assert "SIN VEREDICTO · ABSTENCIÓN" in html
    assert "JSON.stringify(spec.payload" in html
    assert 'type="checkbox"' not in html
    assert "use_llm_mitigator" not in html
    assert 'id="persist"' not in html


def test_frontend_simplifies_standardization_cache_information():
    html = TestClient(app).get("/").text

    assert "Acceso a caché" in html
    assert "Sí (Estandarizado Previamente)" in html
    assert "No (Llamada en Vivo)" in html
    assert "No (Estandarizado Previamente)" not in html
    assert "Sí (Llamada en Vivo)" not in html
    assert "Caché Mistral" not in html
    assert '<span class="k">Fuente</span>' not in html
    assert '<span class="k">Perfil de esquema</span>' not in html
    assert "Hash del contenido" not in html
    assert "cacheHash" not in html


def test_frontend_shows_auditor_as_an_independent_post_case_stage():
    html = TestClient(app).get("/").text

    assert 'id="panel-auditoria"' in html
    assert "Auditoría posterior independiente" in html
    assert "CaseResult persistido" in html
    assert "Agente auditor" in html
    assert "Se ejecuta después del juez, sobre el caso cerrado." in html
    assert "No modifica su decisión ni su traza operacional" not in html
    assert "final_auditor" not in html
    assert 'fetch(`/cases/${encodeURIComponent(caso.case_id)}/audit`)' in html
    assert "await cargarAuditoria(caso);" in html
    assert html.index('id="panel-traza"') < html.index('id="panel-auditoria"')


def test_frontend_prioritizes_mistral_context_and_shows_every_mitigation():
    html = TestClient(app).get("/").text

    assert 'const visibleText = context || String(item.text || "");' in html
    assert (
        'const visibleSummary = llmSummary || String(exp.summary || "");'
        in html
    )
    assert "Base catalogada:" in html
    assert "Recomendación no respaldada por el catálogo" in html
    assert "Mistral + catálogo" not in html  # se compone dinámicamente
    assert 'llm: `${llmName} + catálogo`' in html
    assert ".slice(0, 6)" not in html
    assert "mitigationItems.length" in html
    assert "escapeHtml(visibleText)" in html
    assert "escapeHtml(visibleSummary)" in html


def test_frontend_escapes_case_result_values_before_using_inner_html():
    html = TestClient(app).get("/").text

    escaped_values = (
        'escapeHtml(std.model ?? "—")',
        'escapeHtml(std.provider ?? "—")',
        'escapeHtml(std.modality ?? "—")',
        'escapeHtml(std.failure_code ?? "standardization_failed")',
        'escapeHtml(det.model_name ?? "—")',
        "escapeHtml(f)",
        "escapeHtml(cls.attack_family)",
        'escapeHtml(juez.action ?? "—")',
        'escapeHtml(juez.final_label ?? "—")',
        "juez.issues.map(issue => escapeHtml(issue))",
        "escapeHtml(t.agent)",
        'escapeHtml(t.tool ?? "—")',
        'escapeHtml(t.status ?? "—")',
        'escapeHtml(t.summary ?? "")',
        "escapeHtml(errorText)",
        "escapeHtml(errorText.slice(0, 90))",
        "escapeHtml(txt)",
        "escapeHtml(caso.case_id)",
    )
    for expression in escaped_values:
        assert expression in html, f"valor dinámico sin protección: {expression}"

    unsafe_interpolations = (
        '${std.model ?? "—"}',
        '${std.source ?? "—"}',
        '${std.provider ?? "—"}',
        '${std.schema_profile ?? "—"}',
        '${std.modality ?? "—"}',
        '${det.model_name ?? "—"}',
        "${cls.attack_family}",
        '${juez.final_label ?? "—"}',
        '${juez.issues.join(", ")}',
        "<tr><td>${t.agent}</td>",
        '${t.summary ?? ""}',
        "${caso.case_id}",
        "t.error.replaceAll",
    )
    for expression in unsafe_interpolations:
        assert expression not in html, f"interpolación insegura: {expression}"

    assert "ratioWidth(std.mapping_confidence)" in html
    assert "ratioWidth(det.probability)" in html
    assert "ratioWidth(s)" in html


def test_frontend_explains_every_auditor_check_without_hard_type():
    html = TestClient(app).get("/").text
    auditor_source = AUDITOR_SOURCE.read_text(encoding="utf-8")
    check_ids = set(re.findall(r'add\(\s*"([^"]+)"', auditor_source))

    assert len(check_ids) == 30
    for check_id in check_ids:
        entry = re.compile(
            rf'"{re.escape(check_id)}":\s*\{{\s*'
            rf'label:\s*"[^"]+",\s*description:\s*"[^"]+"',
        )
        assert entry.search(html), f"falta explicación para {check_id}"

    assert "Descripción" in html
    assert "Qué comprueba" not in html
    assert "Detalle técnico" in html
    assert "Fallos duros" not in html
    assert "<th>Tipo</th>" not in html
    assert "check.hard" not in html
