# Arquitectura final: sistema multiagente de ciberseguridad IoT/IIoT sobre MCP

> La arquitectura descrita esta implementada. La validacion cientifica final no
> esta cerrada: 0.9363 es un baseline historico pendiente de sustitucion por el
> protocolo limpio de `plan_validacion_metricas_por_dataset.md` (fases C-F).

**Proyecto:** `tfm_multiagent` · **Fecha:** 2026-08-06 · **Estado:** arquitectura implementada y suite automatizada; Fase C en ejecucion y evaluador de Fase D listo.

Este documento describe la arquitectura tal y como está construida: capas,
servidores MCP y sus tools, agentes finales, contrato de caso, rutas de
abstención, auditoría E2E y cómo ejecutar cada pieza.

---

## 1. Visión general

```
┌────────────────────────────────────────────────────────────────────────┐
│ FUENTES: IoT-23 · TON-IoT (host/telemetry/network/windows) · Bot-IoT · │
│          Edge-IIoTset · logs · alertas                                 │
└──────────────────────────────┬─────────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│ RAW → MISTRAL EN VIVO → FRONTERA MCP → CanonicalEvent                  │
│ (inference_server.py + standardization_guard.py)                       │
└──────────────────────────────┬─────────────────────────────────────────┘
                               ▼
        ┌──────────────── ORQUESTADOR LangGraph ────────────────┐
        │  src/orchestration/mcp_graph.py — build_final_graph() │
        │  OrchestratorState: case_id + trace[] + salidas       │
        │  standardize → detect → classify → mitigate → judge   │
        │  (abstención / baja confianza → revisión humana)      │
        └──────┬────────────────────────────────────────┬───────┘
               │ agentes finales (clientes MCP)         │
               ▼                                        ▼
┌───────────────────────── CAPA MCP (SDK oficial) ───────────────────────┐
│ mcp-datasets      mcp-inference      mcp-case-memory                   │
│ mcp-threat-intel  mcp-evaluation                                       │
└──────┬──────────────────┬──────────────────┬───────────────────────────┘
       ▼                  ▼                  ▼
  modelos .joblib    Mistral API        catálogo ATT&CK/CAPEC ·
  desplegados        en vivo            SQLite de casos · baselines
               │                                        │
               ▼                                        ▼
   AGENTES FINALES (src/agents/final/)      AGENTE DE AUDITORÍA E2E
   Standardizer · Detector · Classifier     CaseAuditor (por caso) +
   Mitigator · Judge                        scripts/run_system_audit.py
```

**Principio rector:** la aportación no es «otro modelo gana», sino que la
arquitectura produce una **representación canónica más rica y desacoplada**
que habilita mejores modelos posteriores (F1 multiclase 0.9363 sobre evento
canónico Edge-IIoTset frente a 0.7479 del mejor LLM fine-tuned del TFM
previo).

---

## 2. Capa MCP: cinco servidores y sus tools

Construida con `FastMCP` del SDK oficial (`mcp[cli]>=1.2,<2`; la v2 del SDK
eliminó `mcp.server.fastmcp`). Cada servidor es arrancable como proceso MCP
real por stdio (`python -m src.mcp.<server>`) y además expone sus tools en
un dict `TOOLS` para el **modo in-process** (mismo contrato, sin
subprocesos), que usan los tests y la demo `--offline`.

| Servidor | Módulo | Tools | Envuelve |
|---|---|---|---|
| `mcp-datasets` | `src/mcp/datasets_server.py` | `list_datasets`, `load_sample`, `load_batch`, `get_schema_profile` | datasets crudos (`data/`) + corpus estandarizado |
| `mcp-inference` | `src/mcp/inference_server.py` | `standardize_event`, `standardize_batch`, `get_mapping_confidence`, `detect_event`, `detect_batch`, `classify_event`, `classify_batch`, `get_family_scores` | Mistral en vivo obligatorio para entradas crudas, frontera técnica de validación/sanitización para `canonical_event` y modelos `.joblib` |
| `mcp-case-memory` | `src/mcp/case_memory_server.py` | `create_case`, `update_case`, `append_trace`, `get_case`, `list_cases`, `retrieve_similar_cases` | SQLite (`artifacts/case_memory.db`) |
| `mcp-threat-intel` | `src/mcp/threat_intel_server.py` | `list_families`, `map_family_to_attack`, `map_family_to_capec`, `suggest_mitigations`, `get_jorge_capec_coverage` | catálogo estático `src/mcp/data/threat_intel_catalog.json` (v1.1) |
| `mcp-evaluation` | `src/mcp/evaluation_server.py` | `score_run`, `compare_with_baseline`, `get_baselines`, `export_report` | `sklearn.metrics` + baselines congelados |

Notas de diseño:

- Toda tool devuelve JSON serializable con `tool_name`, `tool_version` y
  `latency_ms` (decorador `tool_result`), y **nunca lanza**: los errores se
  devuelven como `{"ok": false, "error": ...}` para que el agente los
  registre en la traza.
- Los modelos `.joblib` se cargan una sola vez por proceso
  (`src/mcp/model_registry.py`, con el shim `EncodedClassifier` para
  deserializar el clasificador de familia).
- `standardize_event` no consulta caché ni adaptadores. Para un `row` crudo
  realiza **dos llamadas a Mistral**: selección semántica de columnas y
  extracción del `CanonicalEvent`. Ambas son obligatorias; si cualquiera
  falla o la salida no valida, devuelve una abstención estructurada y el juez
  termina el caso con `human_interrupt`.
- Un `canonical_event` ya estandarizado no repite Mistral: atraviesa únicamente
  `src/mcp/standardization_guard.py`, la frontera técnica que elimina campos
  sensibles y valida el contrato. Esta sanitización no es una decisión ni una
  responsabilidad cognitiva del agente estandarizador.
- La caché Mistral de la campaña es un artefacto histórico congelado, no una
  ruta de inferencia. Se conserva como procedencia y para obtener los eventos
  canónicos de la demo offline; regla dura: no relanzar campañas masivas.
- Paridad de features entrenamiento/inferencia garantizada:
  `src/mcp/features.py` replica `event_features` del script de
  entrenamiento y un test lo vigila.
- Cliente dual en `src/mcp/client.py`: `MCPToolClient(mode="inprocess")`
  (determinista, tests/demo) o `mode="stdio"` (protocolo MCP auténtico;
  lento porque cada llamada arranca un servidor que recarga modelos —
  útil como demostración, no para operación).

## 3. Agentes finales (`src/agents/final/`)

Cada agente registra en la traza del caso una entrada `TraceEntry` con
inicio, fin, confianza y errores; todos operan contra la capa MCP.

| Agente | Tool principal | Salida / decisión |
|---|---|---|
| `FinalStandardizer` | `standardize_event` | Solicita Mistral obligatoriamente para entrada cruda y consume el resultado ya validado por la frontera MCP. Si la tool se abstiene o `mapping_confidence < 0.5`, enruta al juez; un `canonical_event` preestandarizado solo pasa la validación/sanitización MCP |
| `FinalDetector` | `detect_event` (XGBoost estandarizado con Edge) | `is_malicious`, probabilidad; **zona gris 0.4–0.6 → abstención**; benigno también pasa por el juez |
| `FinalClassifier` | `classify_event` (XGBoost familia balanced_group) | familia, confianza, top-k scores |
| `FinalMitigator` | `suggest_mitigations` + LLM opcional | explicación + mitigaciones por fase + referencias ATT&CK/CAPEC; `source: catalog\|hybrid` |
| `FinalJudge` | — (reglas sobre el estado) | approve / human_interrupt; issues auditables |
| `CaseAuditor` | — (post-hoc, Fase 5) | approve / review / **reject** por consistencia, trazabilidad, leakage y umbrales |

### Mitigación anclada al catálogo (Fase 4)

Flujo en dos pasos: (1) base determinista del catálogo threat intel
(mitigaciones por fase contención/erradicación/prevención + referencias
MITRE ATT&CK, CAPEC y mitigaciones M-*); (2) contextualización LLM opcional
(`LLMMitigationAgent`, multi-proveedor: Mistral/Ollama/OpenRouter/Groq/
Google/Transformers). **Regla dura:** el LLM no puede introducir técnicas ni
referencias fuera del catálogo sin la marca `llm_suggested` — el anclaje
(`anchor_llm_payload`) lo garantiza estructuralmente, incluidos IDs MITRE
citados en texto libre. Si el LLM falla: fallback total al catálogo.

Diferencia clave con el TFM previo (Jorge): aquel usaba **solo CAPEC** como
documentación (mapeo manual de 14 clases, Tabla 3.4) y dejaba al LLM generar
mitigaciones libremente, con evaluación circular (ROUGE-L contra salidas del
propio modelo). Este catálogo (v1.1) es **superset de su tabla** (14/14
cubiertos, tool `get_jorge_capec_coverage`) y añade ATT&CK + M-* + fases +
aplicabilidad por `schema_profile`, con procedencia por ítem
(`MitigationItem.source ∈ {catalog, llm, llm_suggested, fallback}`).

## 4. Contrato de caso (`src/contracts/case.py`)

`CaseResult` es la unidad auditable: `case_id`, entrada cruda, evento
canónico sanitizado, salidas de cada etapa (`StandardizationInfo`,
`DetectionInfo`, `ClassificationInfo`, `ExplanationInfo` con
`mitigation_items[]` y `references[]`, `JudgeInfo`), `trace[]` completa y
`status ∈ {completed, needs_human_review, error}`. La API lo devuelve
íntegro y la memoria de casos lo persiste.

Rutas de abstención (todo caso pasa por el juez y queda auditable; el
`CaseAuditor` se ejecuta después de forma independiente):

- fallo, timeout o respuesta inválida de Mistral durante la selección de
  columnas o la extracción → abstención del estandarizador → juez
  `human_interrupt` (sin caché ni adaptador)
- `canonical_event` inválido en la frontera MCP → abstención → juez
  `human_interrupt`
- `mapping_confidence < 0.5` → juez → revisión humana
- probabilidad de detección en zona gris `[0.4, 0.6]` → abstención → juez
- confianza de clasificación `< 0.65` → juez la deriva a revisión
- error de cualquier tool → registrado en traza y errores → juez

## 5. Auditoría E2E (Fase 5)

- **Por caso** (`CaseAuditor.audit`): consistencia (benigno sin familia,
  abstención flaggeada, estado vs juez, errores derivados), trazabilidad
  (agentes en orden, entradas cerradas, sin retrocesos temporales), target
  leakage (claves target en el evento, patrones en `semantic_text`,
  features que verían los modelos) y umbrales. Veredicto
  approve/review/reject.
- **De sistema** (`scripts/run_system_audit.py`): smoke E2E por dataset,
  métricas batch de los modelos desplegados **reproduciendo el split de
  test exacto** de sus entrenamientos (seed 42; delta 0.0 verificado),
  certificación de baselines congelados (0.9363 ≥ 0.93 y > 0.7479; binario
  ≥ 0.99) **sin re-experimentar**, cobertura CAPEC de Jorge 14/14, y
  chequeo de leakage con caso trampa (self-test). Informe md+json con
  semáforos; exit 0 solo si todo verde. Ejecución real: TODO VERDE, ~7 s,
  determinista.

## 6. Demo reproducible (Fase 6)

`scripts/demo_mcp_multiagent_case.py --offline`: 4 casos obligatorios
(Edge-IIoTset ataque + TON-IoT host + TON-IoT telemetry + IoT-23) entregados
como `canonical_event` congelados. Los tres últimos se seleccionan del
artefacto histórico Mistral, pero el grafo no recibe `cache_key` ni consulta
la caché. Todos atraviesan la validación/sanitización MCP, se auditan en vivo,
con digest SHA256 estable por caso y referencia sin fecha
(`artifacts/demo/demo_digests.json`): repetir la orden produce digests
idénticos y el script lo verifica. `--stdio-smoke` añade un handshake por
protocolo MCP real. Salidas en `artifacts/demo/`.

## 7. Cómo ejecutar

```powershell
# instalación (Windows; usar SIEMPRE el venv del proyecto)
.\.venv\Scripts\python.exe -m pip install -e ".[mcp,inference]"

# suite completa
.\.venv\Scripts\python.exe -m pytest -q

# demo reproducible (defensa)
.\.venv\Scripts\python.exe scripts\demo_mcp_multiagent_case.py --offline

# auditoría E2E de sistema (informe con semáforos en artifacts/)
.\.venv\Scripts\python.exe scripts\run_system_audit.py

# servidor MCP real por stdio (smoke manual)
.\.venv\Scripts\python.exe -m src.mcp.threat_intel_server

# API (POST /cases/analyze devuelve el CaseResult con traza)
.\.venv\Scripts\uvicorn.exe src.api.app:app --reload
```

Ejemplo de análisis de caso vía API:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/cases/analyze -Method Post `
  -ContentType "application/json" `
  -Body '{"dataset":"iot23","row":{"id.orig_h":"10.0.0.2","id.resp_h":"10.0.0.3","id.orig_p":4444,"id.resp_p":23,"proto":"tcp","orig_pkts":120,"orig_ip_bytes":4096},"row_id":42,"persist":true}'
```

### Configuración de Mistral para el estandarizador

```powershell
$env:MISTRAL_API_KEY="..."                 # obligatoria para entradas crudas
$env:INGEST_LLM_MODEL="mistral-small-2603" # opcional; modelo por defecto
$env:INGEST_LLM_TIMEOUT_SECONDS="60"       # opcional; limite por llamada
```

El flujo final no permite desactivar el LLM de estandarización: una entrada
`row` usa siempre Mistral en vivo (selección de columnas + extracción). Si no
está disponible, el caso se abstiene y el juez solicita revisión humana. Un
`canonical_event` congelado no requiere la API porque ya fue estandarizado.

### Configuración del LLM del mitigador (opcional)

```powershell
$env:LLM_MITIGATOR_ENABLED="true"          # activa la contextualización
$env:MITIGATOR_LLM_PROVIDER="mistral"      # mistral | ollama | openrouter | groq | google | transformers
$env:MITIGATOR_LLM_MODEL="mistral-small-latest"   # opcional (default por proveedor)
$env:MISTRAL_API_KEY="..."                 # según proveedor
```

Sin la configuración específica del mitigador (o con fallo de ese LLM), el
mitigador funciona en modo catálogo puro. Este fallback pertenece solo al
mitigador; el estandarizador nunca sustituye Mistral por caché o adaptadores.

## 8. Dónde está cada cosa

| Pieza | Ruta |
|---|---|
| Contratos | `src/contracts/{canonical,agents,case,taxonomy}.py` |
| Capa MCP | `src/mcp/` (cliente en `client.py`; catálogo en `data/threat_intel_catalog.json`) |
| Agentes finales + auditor | `src/agents/final/` |
| Grafo final | `src/orchestration/mcp_graph.py` (`graph.py` es el orquestador previo, intacto) |
| API | `src/api/routers.py` |
| Scripts activos | `scripts/` (5); históricos en `scripts/archive/` con README |
| Baselines congelados | `artifacts/baselines/jorge_and_current_baselines.json` |
| Resultados y narrativa para la memoria | `docs/material_memoria_resultados.md` |
| Diagrama v4 (capa MCP explícita) | `docs/comparativa_arquitectura_tfm_v4_mcp.drawio` |
