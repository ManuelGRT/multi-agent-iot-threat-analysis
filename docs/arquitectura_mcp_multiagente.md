# Arquitectura final del sistema multiagente IoT/IIoT

Este documento describe el runtime online desplegable. El repositorio separa
la ruta operacional de las utilidades offline de entrenamiento y evaluación:
la API no ejecuta sanitización, adaptadores ni flujos alternativos.

## 1. Flujo operacional

```text
POST /cases/analyze (`row` o `text` limpios)
                │
                ▼
      caché por hash exacto
        │ hit          │ miss
        │              ▼
        │       Mistral: selección + extracción
        │              │
        └──────────────┤
                       ▼
                evento canónico validado
                       │
                       ▼
 Estandarizador → Detector → Clasificador → Mitigador → Juez
                       │                         │
                       │                         └─ catálogo + contexto Mistral
                       └─ abstención/baja confianza → revisión humana
                                                 │
                                                 ▼
                                    CaseResult persistido y auditable
```

El orquestador activo es `src/orchestration/mcp_graph.py`. Todo caso alcanza
el juez, incluidos los benignos y las abstenciones. El auditor actúa después,
sobre el `CaseResult` ya persistido, mediante
`GET /cases/{case_id}/audit`.

## 2. Tres servidores MCP

Los agentes finales consumen solo tres dominios de herramientas. Cada servidor
puede utilizarse en proceso o mediante el protocolo MCP por stdio con el mismo
contrato JSON.

| Servidor | Módulo | Responsabilidad |
|---|---|---|
| `inference` | `src/mcp/inference_server.py` | Estandarización Mistral, validación canónica, detección y clasificación con los modelos empaquetados |
| `case_memory` | `src/mcp/case_memory_server.py` | Persistencia y consulta de casos y trazas en SQLite |
| `threat_intel` | `src/mcp/threat_intel_server.py` | Catálogo versionado ATT&CK/CAPEC y mitigaciones verificables |

Los servidores históricos de datasets y evaluación no forman parte del
runtime. Los datos crudos, métricas y procesos experimentales pertenecen al
entorno offline.

Toda tool devuelve un objeto JSON con `ok`, metadatos de traza y, ante un
fallo, un error estructurado. Los agentes pueden así cerrar el caso de forma
controlada sin convertir una excepción en un resultado benigno.

## 3. Agentes finales

| Agente | Entrada y operación | Salida o decisión |
|---|---|---|
| `FinalStandardizer` | Entrada limpia `row` o `text`; busca un éxito Mistral por hash exacto y, ante un *miss*, llama obligatoriamente a Mistral | Evento canónico, procedencia y confianza; cualquier fallo o confianza menor de 0,5 deriva al juez |
| `FinalDetector` | Modelo XGBoost binario empaquetado | Veredicto y probabilidad; zona gris `[0.4, 0.6]` implica abstención |
| `FinalClassifier` | Modelo XGBoost multiclase balanceado y empaquetado | Uno de 16 tipos, familia agregada, confianza y top-3; confianza menor de 0,65 solicita revisión |
| `FinalMitigator` | Base del catálogo y contextualización Mistral por defecto | Explicación, recomendaciones y referencias con procedencia explícita |
| `FinalJudge` | Reglas sobre todas las salidas y errores | `approve` o `human_interrupt` |
| `CaseAuditor` | Revisión posterior del caso persistido | `approve`, `review` o `reject` por coherencia, trazabilidad, umbrales y fugas |

Cada paso operacional añade un `TraceEntry` con estado, confianza, tiempos y
errores. El auditor no es un nodo del grafo y emite un informe independiente.

## 4. Estandarización y caché

La entrada pública es siempre cruda y debe estar libre de targets. La API
acepta exclusivamente `row` o `text`; no acepta `canonical_event`.

El estandarizador calcula un hash estable del contenido y consulta una caché
SQLite:

1. En un *hit* reutiliza únicamente una estandarización Mistral validada para
   ese contenido y reconstruye la identidad y procedencia del registro actual.
2. En un *cache miss*, una entrada tabular requiere selección semántica de
   columnas y extracción canónica mediante Mistral. No existe selección
   heurística ni ruta alternativa determinista.
3. Si Mistral no responde, agota el tiempo o devuelve una salida inválida, el
   agente se abstiene y el juez solicita revisión humana.

El mecanismo *single-flight* evita llamadas duplicadas para la misma clave
dentro de un proceso. SQLite conserva el primer éxito entre procesos. La caché
solo almacena respuestas correctas; los fallos no se convierten en resultados
reutilizables.

La sanitización anti-leakage pertenece exclusivamente a la preparación offline
de datasets de entrenamiento, validación y evaluación. No se ejecuta dentro de
la API ni del grafo.

## 5. Detección y clasificación

Los dos artefactos activos están incluidos como recursos del paquete y se
deserializan mediante el contrato estable `src/contracts/inference.py`:

- `xgboost_detection_validation_2026_20260822.joblib`;
- `xgboost_attack_subtype_multidataset16_balanced500_20260906.joblib`.

El registro de modelos los carga de forma perezosa una sola vez por proceso.
El contrato de inferencia no importa módulos de evaluación ni agentes de
entrenamiento. `src/mcp/features.py` mantiene la misma proyección de features
que se usó al entrenarlos.

## 6. Mitigación anclada

El mitigador trabaja en dos capas:

1. `threat_intel` obtiene del catálogo versionado mitigaciones por fase y
   referencias ATT&CK/CAPEC verificables.
2. Mistral intenta contextualizar esa base para el evento observado.

El validador conserva la procedencia de cada elemento. Las incorporaciones sin
respaldo se marcan `llm_suggested` y no se presentan como conocimiento
auditado. Si Mistral falla, el caso mantiene la respuesta completa del
catálogo. Este *fallback* existe únicamente en mitigación: nunca sustituye la
llamada obligatoria del estandarizador ante un *cache miss*.

## 7. Contrato, persistencia y API

`CaseResult` reúne el identificador, la entrada, el evento canónico, las salidas
por etapa, la decisión del juez, los errores y la traza. El endpoint de análisis
persiste siempre el caso en la memoria SQLite.

Superficie pública:

| Método y ruta | Función |
|---|---|
| `POST /cases/analyze` | Analiza una entrada limpia `row` o `text` |
| `GET /cases/{case_id}/audit` | Recalcula la auditoría posterior del caso persistido |
| `GET /health` | Comprueba la disponibilidad de la API |

Las rutas antiguas de eventos, datasets y adaptadores no están registradas y
responden `404`. El contrato de `/cases/analyze` rechaza con `422` cualquier
campo `canonical_event`, por lo que no permite saltarse Mistral.

## 8. Estado y recursos empaquetados

El código, el frontend, el catálogo y los modelos son recursos de solo lectura
del paquete. La caché y la memoria de casos se escriben en una raíz de estado
separada:

- `TFM_STATE_DIR`: raíz escribible del runtime;
- `TFM_STANDARDIZATION_CACHE_DB`: override opcional de la caché Mistral;
- `TFM_CASE_MEMORY_DB`: override opcional de la memoria de casos.

Sin configuración adicional, `TFM_STATE_DIR` corresponde a `artifacts/` en la
raíz de ejecución. La caché queda en
`cache/mistral_standardization_v2.sqlite3` y los casos en `case_memory.db`. En
Docker la raíz se fija a `/var/lib/tfm_multiagent`, respaldada por el volumen
`runtime_state`. Ninguna de estas rutas escribe dentro del paquete instalado.

## 9. Configuración y ejecución

Instalación normal en Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .

$env:MISTRAL_API_KEY="..."
$env:TFM_STATE_DIR=".runtime-state" # opcional
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Configuración opcional:

```powershell
$env:INGEST_LLM_MODEL="mistral-small-2603"
$env:INGEST_LLM_TIMEOUT_SECONDS="60"
$env:MITIGATOR_LLM_MODEL="mistral-small-2603"
$env:TFM_STANDARDIZATION_CACHE_DB=".runtime-state/cache/mistral_standardization_v2.sqlite3"
$env:TFM_CASE_MEMORY_DB=".runtime-state/case_memory.db"
```

La clave `MISTRAL_API_KEY` debe proporcionarse solo mediante el entorno o un
`.env` local ignorado por git.

El extra `mcp` solo es necesario para arrancar los servidores mediante el
protocolo stdio (`python -m pip install ".[mcp]"`); la API usa el mismo
contrato en proceso.

Ejecución en contenedor:

```powershell
docker compose up --build
```

La API y el frontend quedan expuestos en `http://localhost:8000`. El volumen de
estado conserva la caché y los casos entre reinicios; los modelos ya están
incluidos en la imagen a través del paquete.

## 10. Separación online/offline

El runtime online contiene únicamente el grafo final con sus cinco agentes
operacionales, el auditor posterior, los tres servidores MCP, la API, el
frontend, los contratos y los recursos de inferencia. No conserva adaptadores,
grafos alternativos ni agentes legacy.

Las tareas de sanitización, construcción de datasets, entrenamiento y
evaluación son procesos offline. Sus resultados científicos pueden conservarse
en documentación o almacenamiento externo, pero no son dependencias del
servicio online.
