# Arquitectura final del sistema multiagente IoT/IIoT

Este documento describe el runtime online desplegable. El repositorio separa
la ruta operacional de las utilidades offline de entrenamiento y evaluación:
la API no ejecuta sanitización, adaptadores ni flujos alternativos.

## 1. Flujo operacional

```text
POST /cases/analyze (`row` o `text` limpios)
                │
                ▼
      caché por hash exacto (`case_memory`)
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

Los agentes finales consumen solo tres dominios de herramientas. En producción
se comunican por defecto mediante el protocolo MCP sobre `stdio`. El modo
`inprocess` ejecuta directamente las mismas funciones y conserva el contrato
JSON, pero se reserva para la demostración rápida y las pruebas que no evalúan
el transporte. En `stdio`, cada caso crea los servidores de forma perezosa,
mantiene una sesión por servidor durante todas sus llamadas y los cierra tras
la persistencia final, incluso si el flujo termina con error.

`run_case` gestiona automáticamente ese ciclo de vida cuando crea el cliente.
Si se inyecta un cliente manualmente, el llamador conserva su propiedad y debe
cerrarlo mediante `with MCPToolClient(...)` o `close()`.

| Servidor | Módulo | Responsabilidad |
|---|---|---|
| `inference` | `src/mcp/inference_server.py` | Estandarización Mistral viva, validación canónica, detección y clasificación con los modelos empaquetados; no abre almacenamiento persistente |
| `case_memory` | `src/mcp/case_memory_server.py` | Lectura y escritura de la caché de estandarización, los casos y sus trazas en SQLite |
| `threat_intel` | `src/mcp/threat_intel_server.py` | Catálogo versionado ATT&CK/CAPEC, mitigaciones verificables y llamada Mistral para contextualizarlas |

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
| `FinalClassifier` | Modelo XGBoost multiclase balanceado y empaquetado | Uno de 16 tipos, confianza y top-3 de tipos; confianza menor de 0,65 solicita revisión |
| `FinalMitigator` | Invoca por MCP las tools `suggest_mitigations` y `contextualize_mitigations` de `threat_intel` | Explicación, recomendaciones y referencias específicas del tipo, con procedencia explícita |
| `FinalJudge` | Reglas sobre todas las salidas y errores; comprueba que clasificador y mitigador conserven el mismo tipo | `approve` o `human_interrupt` |
| `CaseAuditor` | Revisión posterior del caso persistido | `approve`, `review` o `reject` por coherencia del tipo, trazabilidad, umbrales y fugas |

Cada paso operacional añade un `TraceEntry` con estado, confianza, tiempos y
errores. El auditor no es un nodo del grafo y emite un informe independiente.

## 4. Estandarización y caché

La entrada pública es siempre cruda y debe estar libre de targets. La API
acepta exclusivamente `row` o `text`; no acepta `canonical_event`.

El agente estandarizador coordina las herramientas de los dos servidores. En
primer lugar solicita a `case_memory.lookup_standardization_cache` que calcule
el hash estable del contenido y consulte la caché SQLite:

1. En un *hit* reutiliza únicamente una estandarización Mistral validada para
   ese contenido y reconstruye la identidad y procedencia del registro actual.
2. En un *cache miss*, invoca `inference.standardize_event`: una entrada tabular
   requiere selección semántica de columnas y extracción canónica mediante
   Mistral. No existe selección heurística ni ruta alternativa determinista.
   Un éxito validado se persiste mediante
   `case_memory.store_standardization_cache`.
3. Si Mistral no responde, agota el tiempo o devuelve una salida inválida, el
   agente se abstiene y el juez solicita revisión humana.

El mecanismo *single-flight* del agente evita llamadas duplicadas para la misma
clave dentro de un proceso sin abrir SQLite directamente. `case_memory`
recalcula las claves, valida cada escritura y conserva el primer éxito entre
procesos. Un fallo de caché no invalida un resultado Mistral correcto; los
fallos del modelo nunca se almacenan.

La sanitización anti-leakage pertenece exclusivamente a la preparación offline
de datasets de entrenamiento, validación y evaluación. No se ejecuta dentro de
la API ni del grafo.

## 5. Detección y clasificación

Los dos artefactos activos están incluidos como recursos del paquete y se
deserializan mediante el contrato estable `src/contracts/inference.py`:

- `xgboost_detection_balanced_by_origin_20260905.joblib`;
- `xgboost_attack_subtype_multidataset16_balanced500_20260906.joblib`.

La [ficha reproducible del detector](evaluation_references/xgboost_detection_balanced_by_origin_20260905.json)
documenta el corpus balanceado de 23.604 filas, el ajuste sobre 16.496 filas de
entrenamiento, las particiones de validación y test y los hashes del artefacto,
del esquema y del booster.

El registro de modelos los carga de forma perezosa una sola vez por proceso.
El contrato de inferencia no importa módulos de evaluación ni agentes de
entrenamiento. `src/mcp/features.py` mantiene la misma proyección de features
que se usó al entrenarlos.

El clasificador produce directamente una etiqueta entre los 16 tipos de la
taxonomía desplegada, su confianza y las tres alternativas con mayor
probabilidad. No calcula ni persiste una familia amplia.

## 6. Mitigación anclada

El mitigador trabaja en dos capas:

1. El agente invoca `threat_intel.suggest_mitigations` con el `attack_type`
   predicho; el servidor consulta directamente su
   entrada del catálogo versionado para obtener mitigaciones por fase y
   referencias ATT&CK/CAPEC verificables. No existe una consulta intermedia ni
   una ruta de reserva por familia amplia.
2. El agente invoca `threat_intel.contextualize_mitigations`. El servidor
   resuelve otra vez la entrada autoritativa y llama a Mistral dentro de su
   propio proceso; el agente nunca accede directamente al proveedor.

El servidor valida estrictamente el contrato JSON devuelto por Mistral antes
de responder con `ok=true`. Después, el agente contrasta los metadatos de ambas
respuestas, conserva la procedencia de cada elemento y descarta cualquier
mitigación o referencia que no corresponda con una base catalogada. Si Mistral
falla o su salida es inválida, el caso mantiene la respuesta completa del
catálogo. Este *fallback* existe únicamente en mitigación: nunca sustituye la
llamada obligatoria del estandarizador ante un *cache miss*.

## 7. Contrato, persistencia y API

`CaseResult` reúne el identificador, la entrada, el evento canónico, las salidas
por etapa, la decisión del juez, los errores y la traza. El juez y el auditor
comprueban la continuidad del tipo entre la clasificación, la consulta del
catálogo, la mitigación y el veredicto. El endpoint de análisis persiste siempre
el caso en la memoria SQLite e indexa los casos maliciosos por tipo de ataque.

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
del paquete. Todo el estado mutable se escribe en la única raíz indicada por
`TFM_STATE_DIR`.

Sin configuración adicional, `TFM_STATE_DIR` corresponde a `artifacts/` en la
raíz de ejecución. `case_memory` crea allí dos bases independientes:
`mistral_standardization_v2.sqlite3` para la caché y `case_memory.db` para los
casos y trazas. Los ficheros comparten carpeta, no esquema. En
Docker la raíz se fija a `/var/lib/tfm_multiagent`, respaldada por el volumen
`runtime_state`. Ninguna de estas rutas escribe dentro del paquete instalado.
Los overrides históricos `TFM_STANDARDIZATION_CACHE_DB` y
`TFM_CASE_MEMORY_DB` se conservan por compatibilidad, pero el runtime rechaza
que apunten a directorios diferentes.
En el primer acceso con las rutas predeterminadas, una caché existente en el
antiguo subdirectorio `cache/` se copia mediante el mecanismo de respaldo de
SQLite y el original se conserva sin nuevas escrituras.

## 9. Configuración y ejecución

Instalación normal en Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .

$env:MISTRAL_API_KEY="..."
$env:TFM_STATE_DIR=".runtime-state" # opcional
# MCP_CLIENT_MODE=stdio es el valor predeterminado
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Configuración opcional:

```powershell
$env:INGEST_LLM_MODEL="mistral-small-2603"
$env:INGEST_LLM_TIMEOUT_SECONDS="60"
$env:MITIGATOR_LLM_MODEL="mistral-small-2603"
$env:MITIGATOR_LLM_TIMEOUT_SECONDS="60"
$env:MITIGATOR_LLM_TOTAL_TIMEOUT_SECONDS="105"
$env:MCP_CLIENT_MODE="stdio"
$env:MCP_STDIO_TIMEOUT_SECONDS="120"
$env:TFM_ATTACK_TYPE_MODEL="C:/ruta/modelo-16-tipos.joblib"
```

El timeout de cada intento HTTP se fija en 60 segundos. Los reintentos y sus
esperas comparten además un presupuesto total interno de 105 segundos mediante
`MITIGATOR_LLM_TOTAL_TIMEOUT_SECONDS`, inferior a los 120 segundos de
`MCP_STDIO_TIMEOUT_SECONDS`. Los tres reintentos siguen disponibles cuando los
fallos son rápidos; si consumen demasiado tiempo, el presupuesto total los
cancela y `threat_intel` devuelve un error controlado que activa el fallback al
catálogo antes de que venza el transporte.

`TFM_ATTACK_TYPE_MODEL` solo es necesario para sustituir el artefacto incluido
por otro que implemente exactamente la taxonomía desplegada de 16 tipos. La
antigua variable `TFM_FAMILY_MODEL` no forma parte del contrato operativo.

La clave `MISTRAL_API_KEY` debe proporcionarse solo mediante el entorno o un
`.env` local ignorado por git.

El SDK MCP forma parte de la instalación productiva. Para una demostración
local con menor latencia puede seleccionarse explícitamente
`MCP_CLIENT_MODE=inprocess`; para volver a atravesar el protocolo real basta con
eliminar la variable o asignarle `stdio`.

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
