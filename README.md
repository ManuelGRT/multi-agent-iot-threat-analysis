# TFM Multiagent Architecture

> **Estado de validacion cientifica:** la arquitectura y su suite son
> ejecutables, pero la cifra 0.9363 es un baseline historico obtenido antes del
> protocolo de deduplicacion. No debe presentarse como resultado final limpio.
> La Fase C de `docs/plan_validacion_metricas_por_dataset.md` esta ejecutandose
> con Mistral en vivo. La Fase D ya esta implementada y probada; sus metricas
> finales se generan automaticamente cuando la cobertura LLM llegue al 100 %.

Arquitectura multiagente de ciberseguridad IoT/IIoT: estandarizacion de
fuentes heterogeneas a un evento canonico, deteccion, clasificacion,
mitigacion anclada a threat intel y auditoria E2E, coordinadas por
LangGraph sobre una capa MCP real (SDK oficial).

## Arquitectura final MCP

Documentacion completa en `docs/arquitectura_mcp_multiagente.md`; diagrama
en `docs/comparativa_arquitectura_tfm_v4_mcp.drawio`; resultados y
narrativa para la memoria en `docs/material_memoria_resultados.md`.

- **Capa MCP** (`src/mcp/`): 5 servidores FastMCP — datasets, inference
  (cache Mistral + modelos XGBoost desplegados), case-memory (SQLite),
  threat-intel (catalogo ATT&CK/CAPEC, superset del mapeo del TFM previo)
  y evaluation (metricas + baselines congelados). Cliente dual
  in-process/stdio en `src/mcp/client.py`.
- **Agentes finales** (`src/agents/final/`): standardizer (sanitizacion
  anti-leakage), detector (abstencion en zona gris 0.4-0.6), classifier,
  mitigator (LLM anclado al catalogo con marca `llm_suggested`; fallback
  total) y judge. Auditor E2E (`CaseAuditor`) por caso.
- **Grafo final** (`src/orchestration/mcp_graph.py`): todo caso devuelve
  un `CaseResult` con `case_id` y `trace[]` completa; la API lo expone en
  `POST /cases/analyze`.
- **Comparativa historica (congelada, no validacion final)**: F1 multiclase **0.9363** (evento
  canonico Edge-IIoTset) frente a **0.7479** (mejor LLM fine-tuned del TFM
  previo).

Comandos de referencia:

```powershell
# demo reproducible (defensa): 4 casos deterministas, digests estables
.\.venv\Scripts\python.exe scripts\demo_mcp_multiagent_case.py --offline

# auditoria E2E: informe con semaforos (exit 0 solo si todo verde)
.\.venv\Scripts\python.exe scripts\run_system_audit.py
```

## Instalacion

Usa Python 3.12 o superior. El stack ML esta fijado a las versiones con las que
se validaron los modelos `joblib`; no reutilices un `.venv` copiado de otro equipo:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[mcp,inference]"
```

Nota: el SDK MCP esta fijado a `mcp[cli]>=1.2,<2` (la v2 elimino
`mcp.server.fastmcp`). Para instalar extras opcionales:

```powershell
python -m pip install -e ".[drift-advanced,postgres,training]"
```

## Tests

```powershell
python -m pytest -q
```

Suite actual: 347 tests (los 74 de la linea base se preservan integros).

## Campana de validacion 2026

La ejecucion larga es reanudable y no guarda la credencial en artefactos. El
supervisor espera al runner activo, reintenta fallbacks con backoff y solo
entonces ejecuta los cuatro modelos de Jorge por dataset, el detector global y
el clasificador global de familias:

```powershell
python scripts\run_validation_campaign.py --wait-pid <PID> --workers 2
```

La Fase D estricta tambien puede lanzarse por separado una vez cerrados los
JSONL. Usa splits congelados, deduplicacion por tarea, seleccion exclusivamente
en validacion y un worker PyTorch externo para el MLP exacto:

```powershell
python scripts\eval_validacion_por_dataset.py
```

Los informes y candidatos quedan bajo `artifacts/validation_2026/evaluation/`;
no sustituyen implicitamente los modelos desplegados.

## Analisis de casos (flujo final MCP)

```powershell
Invoke-RestMethod http://127.0.0.1:8000/cases/analyze -Method Post `
  -ContentType "application/json" `
  -Body '{"dataset":"iot23","row":{"id.orig_h":"10.0.0.2","id.resp_h":"10.0.0.3","id.orig_p":4444,"id.resp_p":23,"proto":"tcp","orig_pkts":120,"orig_ip_bytes":4096},"row_id":42,"persist":true}'
```

Devuelve el `CaseResult` completo (deteccion, familia, mitigaciones con
referencias ATT&CK/CAPEC, veredicto del juez y traza por agente). Con
`"use_llm_mitigator": true` activa la contextualizacion LLM anclada al
catalogo (el proveedor/modelo se configura con `MITIGATOR_LLM_PROVIDER`,
`MITIGATOR_LLM_MODEL` y las credenciales del proveedor; sin peticion
explicita manda la variable `LLM_MITIGATOR_ENABLED`). Si el LLM falla hay
fallback total al catalogo.

## API local

```powershell
uvicorn src.api.app:app --reload
```

Healthcheck:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Analisis de evento:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/events/analyze `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"dataset":"iot23","source_file":"sample.log","row_id":1,"row":{"id.orig_h":"10.0.0.2","id.resp_h":"10.0.0.3","id.orig_p":4444,"id.resp_p":23,"proto":"tcp","duration":"1.2","orig_pkts":120,"orig_ip_bytes":4096,"label":"Mirai"}}'
```

Entrada generica sin adapter especifico:

```powershell
$body = '{"dataset":"future_dataset","text":"src_ip=10.0.0.1 dst_ip=10.0.0.2 proto=tcp label=normal"}'
Invoke-RestMethod -Uri http://127.0.0.1:8000/events/analyze -Method Post -ContentType "application/json" -Body $body
```

En la API async, `generic` y datasets desconocidos intentan usar el parser LLM por defecto si Ollama esta disponible. Si quieres forzar el adapter determinista clasico, envia `use_llm:false`.

Entrada generica forzando LLM local de Ollama:

```powershell
$body = '{"dataset":"future_dataset","use_llm":true,"text":"Connection from 10.0.0.1:4444 to 10.0.0.2:23 over tcp, Mirai attack"}'
Invoke-RestMethod -Uri http://127.0.0.1:8000/events/analyze -Method Post -ContentType "application/json" -Body $body
```

Backend LLM:

```powershell
$env:OLLAMA_HOST="http://127.0.0.1:11434"
$env:OLLAMA_AGENT_MODEL="gemma3:12b"
$env:LLM_BACKEND="langchain" # por defecto; usa ChatOllama de LangChain
```

Para depuracion puedes forzar el cliente HTTP directo anterior:

```powershell
$env:LLM_BACKEND="httpx"
```

Orquestacion:

- `build_graph(...)` compila el grafo LangGraph sincrono para smoke tests deterministas.
- `build_async_graph(...)` compila el grafo LangGraph async para ingesta LLM.
- La API usa `AsyncGraphOrchestrator`, que ejecuta el flujo sobre LangGraph y aplica `recursion_limit`.

Analizar un CSV/log real:

```powershell
$env:TFM_DATA_DIR="C:\ruta\a\datasets" # raiz permitida para lecturas
$body = '{"dataset":"iot23","path":"C:\\ruta\\a\\datasets\\conn.log","limit":10}'
Invoke-RestMethod -Uri http://127.0.0.1:8000/datasets/analyze-file -Method Post -ContentType "application/json" -Body $body
```

La ruta debe estar dentro de `TFM_DATA_DIR` (por defecto, `data/` del
proyecto). El analisis de ficheros usa adaptadores deterministas; para activar
explicitamente la ingesta LLM envia `"use_llm": true`.

Datasets/adapters soportados:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/datasets/supported
```

## Servicios externos

El `docker-compose.yml` levanta Ollama y Postgres con pgvector:

```powershell
docker compose up -d
```

Modelos Ollama sugeridos:

```powershell
ollama pull embeddinggemma
ollama pull gemma3:12b
```
