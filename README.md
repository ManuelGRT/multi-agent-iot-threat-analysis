# Sistema Multiagente para el Análisis de Amenazas IoT/IIoT

Trabajo Fin de Máster (Máster en Inteligencia Artificial Aplicada). Este
proyecto convierte un flujo clásico de detección de intrusiones —acoplado a un
único dataset— en un **sistema multiagente**: los eventos de seguridad de
cualquier fuente IoT/IIoT se transforman a un formato común (el *evento
canónico*) y una cadena de agentes especializados los analiza, propone
mitigaciones verificables y deja una traza completa auditable de cada decisión.

## ¿Cómo funciona?

Cada evento recorre este flujo, coordinado por el orquestador final activo
`src/orchestration/mcp_graph.py` (LangGraph) sobre una capa de herramientas MCP
(Model Context Protocol):

```
entrada cruda → Estandarizador → Detector → Clasificador → Mitigador → Juez
                                    │ (zona gris)              │
                                    └──────► revisión humana ◄─┘
```

| Agente | Qué hace | Qué decide |
|---|---|---|
| **Estandarizador** | Recibe una entrada cruda ya limpia (`row` o `text`). Una caché SQLite por hash exacto reutiliza únicamente éxitos previos de Mistral para contenido duplicado y reconstruye la identidad y procedencia del caso actual. En un *cache miss*, Mistral selecciona las columnas y genera el evento canónico | Confianza del mapeo; si no hay *hit* y Mistral falla, la respuesta es inválida o la confianza es baja (< 0,5), se abstiene y el juez deriva el caso a revisión humana |
| **Detector** | Modelo XGBoost binario: ¿es malicioso? | Veredicto y probabilidad; en la zona gris [0,4–0,6] **se abstiene** |
| **Clasificador** | Modelo XGBoost multiclase balanceado: identifica directamente uno de los 16 tipos de ataque | Tipo, confianza y top-3 de tipos; si la confianza es < 0,65, el caso va a revisión |
| **Mitigador** | Consulta por `attack_type` la entrada específica del **catálogo verificable** (ATT&CK + CAPEC) e intenta siempre contextualizarla con Mistral en el endpoint final | Si el LLM falla conserva el catálogo; todo añadido sin respaldo queda marcado `llm_suggested` |
| **Juez** | Aplica reglas deterministas sobre el caso completo y comprueba que clasificación y mitigación conserven el mismo tipo | Aprobar o derivar a revisión humana |
| **Auditor** | Revisa a posteriori el caso cerrado, incluida la coherencia del tipo entre clasificador, catálogo, mitigador, juez y persistencia | Aprobar, revisar o rechazar |

Ideas clave del diseño:

- **La entrada limpia es una precondición.** La sanitización anti-leakage y la
  separación de columnas target se realizan al preparar los datos de
  entrenamiento, validación y evaluación, siempre fuera del grafo. El runtime
  final no limpia ni inspecciona targets dentro del `row`: presupone una entrada
  preparada. La salida canónica generada por Mistral sí se valida antes de
  alcanzar los modelos. La API pública no acepta eventos preestandarizados.
- **Mistral es obligatorio en cada *cache miss*.** La caché SQLite se indexa
  por el hash exacto del contenido limpio y solo reutiliza respuestas Mistral
  exitosas para duplicados. En un *hit* se reconstruyen la identidad y la
  procedencia actuales y se reutiliza la selección que Mistral hizo para ese
  mismo contenido; no aparece una heurística determinista. Si no hay *hit* y
  Mistral falla, el sistema se abstiene y entrega el caso al juez para revisión
  humana. No existe ningún adaptador en el runtime. El mecanismo *single-flight*
  evita llamadas repetidas para una misma clave dentro de cada proceso; entre
  procesos, SQLite aplica la política del primer éxito escrito, aunque dos
  *misses* simultáneos pueden llegar a invocar Mistral.
- **Todo caso pasa por el juez y queda auditable**: también los benignos se
  revisan operacionalmente y todos producen un `CaseResult` con su traza. El
  `CaseAuditor` es una comprobación posterior e independiente.
- **El tipo es la única etiqueta multiclase operacional.** El clasificador no
  deriva familias amplias: devuelve uno de los 16 tipos junto con su confianza
  y el top-3. Ese mismo tipo identifica directamente la entrada del catálogo,
  se conserva en el veredicto del juez y queda indexado en la memoria de casos.
- **Producción usa MCP real por `stdio`.** Los agentes acceden por defecto a
  los tres servidores mediante el SDK oficial y llamadas MCP sobre procesos
  `stdio`. Cada caso reutiliza una sesión por servidor y la cierra después de
  persistir el resultado. El modo `inprocess` conserva el mismo contrato, pero
  queda reservado para la demostración rápida y las pruebas que no evalúan el
  transporte.
- **El LLM del mitigador está anclado**: puede redactar y contextualizar, pero
  no puede presentar referencias inventadas como conocimiento auditado; lo no
  respaldado por el catálogo siempre conserva la marca `llm_suggested`. Para la
  decisión operacional se examinan las cinco primeras recomendaciones del LLM:
  si las cinco están vinculadas a bases distintas del catálogo, una sugerencia
  posterior o una petición genérica del LLM no fuerza por sí sola la revisión.
  Una recomendación no anclada dentro de esas cinco, una referencia adicional
  desconocida o un identificador inventado en el resumen sí la mantienen.

## Resultados principales

Evaluación sobre 35.637 registros únicos de 5 datasets públicos, con
particiones deduplicadas y congeladas **antes** de experimentar (el diagnóstico
de duplicados que motivó este protocolo detectó hasta un 98,5 % de filas
repetidas en alguna clase de IoT-23):

| Qué se mide | Resultado |
|---|---|
| Multiclase Edge-IIoTset (15 clases, protocolo del TFM previo) | **F1 0,9144** frente a 0,7479 (mejor LLM *fine-tuned* del TFM previo) y 0,4987 (su mejor modelo clásico) |
| Réplica del baseline previo (garantía de reproducción) | 0,4993 frente a 0,4987 ✓ |
| Detector global (test congelado, n=5.064) | F1 0,9676; con abstención: **0,977 sobre lo decidido**, derivando el 4,8 % a revisión |
| Clasificador de 16 tipos (test balanceado, n=1.200) | Accuracy 0,8908; macro-F1 0,8917; top-3 0,9800; con umbral 0,65: F1 0,9237 sobre 1.088 decisiones |
| Mitigador (16 llamadas Mistral en vivo) | 111/113 medidas base contextualizadas; cinco primeras ancladas en 16/16 casos; 15 aprobaciones estructurales y 1 revisión |
| Auditor | 16/16 defectos inyectados detectados; 0 falsos rechazos |

**Límite declarado:** las evaluaciones balanceadas por origen emplean registros
de test de fuentes representadas durante el entrenamiento; no son pruebas
*leave-one-dataset-out* ni demuestran transferencia a una fuente no vista. Los
resultados valen para los dominios representados. La mejora frente al TFM
previo procede de la **representación canónica**, no del algoritmo (la réplica
a igualdad de columnas reproduce su baseline).

## Instalación y ejecución online

```powershell
# 1. Instalar normalmente (Python 3.12+)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .

# 2. Opcional para desarrollo: instalar tests y comprobarlos
python -m pip install ".[test]"
$env:MCP_CLIENT_MODE="inprocess"
python -m pytest -q

# 3. Configurar Mistral y arrancar API + frontend en modo productivo
$env:MISTRAL_API_KEY="..."
$env:MCP_CLIENT_MODE="stdio" # valor predeterminado; se explicita tras los tests
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Los modelos activos, el catálogo y el frontend forman parte del paquete. No es
necesario ejecutar el repositorio en modo editable ni descargar artefactos de
entrenamiento. Abre `http://localhost:8000` para enviar uno de los ejemplos
crudos del visor o llamar a la API.

El clasificador de 16 tipos incluido es el valor predeterminado. Solo si se
necesita desplegar otro artefacto compatible debe definirse
`TFM_ATTACK_TYPE_MODEL` con su ruta; la configuración anterior
`TFM_FAMILY_MODEL` ya no se admite.

Con Docker, guarda `MISTRAL_API_KEY` en un `.env` local no versionado y ejecuta:

```powershell
docker compose up --build
```

La API y el frontend quedarán disponibles en `http://localhost:8000`. Docker
fija `MCP_CLIENT_MODE=stdio` por defecto. La clave se inyecta únicamente por
entorno; nunca debe incorporarse a la imagen ni al repositorio.

## Probar el sistema con el visor web

```powershell
$env:MCP_CLIENT_MODE="inprocess"
uvicorn src.api.app:app --port 8000
```

Abre `http://localhost:8000`: un visor con ejemplos precargados muestra el
flujo completo (pipeline por agente, decisiones, mitigaciones con referencias
y la traza del caso). Tras cerrarlo y persistirlo, el visor ejecuta el agente
auditor en un panel independiente: informa `approve`, `review` o `reject` y
desglosa sus comprobaciones sin modificar la decisión del juez ni añadir una
entrada a la traza operacional. En la tarjeta del mitigador se presenta primero
la contextualización de Mistral; si no existe, se muestra el texto del catálogo.
También se conserva debajo la base catalogada y se muestran todas las
recomendaciones, incluidas las `llm_suggested`. La entrada cruda debe llegar
limpia. El estandarizador busca primero un éxito Mistral con el mismo hash
exacto en la caché SQLite; en
un *miss* necesitas `MISTRAL_API_KEY` y se llama a Mistral en vivo. Si el
proveedor no responde o devuelve una salida inválida, el caso termina en el
juez como abstención y revisión humana. La ruta final nunca usa adaptadores.
`INGEST_LLM_TIMEOUT_SECONDS` permite acotar cada llamada (por ejemplo, `60`
en una demo; una fila tabular realiza selección y extracción por separado).
En el endpoint final, todos los casos se persisten en la memoria SQLite y, si
el detector confirma un caso malicioso, el mitigador intenta siempre la
contextualización anclada con Mistral. Si el modelo no está disponible, el
caso conserva las contramedidas y referencias verificables del catálogo. Este
cliente reintenta por defecto tres veces los fallos transitorios de conexión
(incluidos DNS y timeout de conexión), con esperas breves de 1, 2 y 4 segundos.
`MISTRAL_CONNECTION_RETRIES` permite ajustar o desactivar esos reintentos.
Después de agotarlos se aplica la ruta de reserva correspondiente. Este
es el único *fallback* del flujo: la estandarización no dispone de una ruta
offline ni determinista para sustituir a Mistral en un *cache miss*.

El modo `inprocess` del bloque anterior evita crear procesos MCP durante una
demostración local. Para probar el mismo visor atravesando el protocolo real,
elimina esa variable o asígnale `stdio`.

También puedes llamar a la API directamente:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/cases/analyze -Method Post `
  -ContentType "application/json" `
  -Body '{"dataset":"iot23","row":{"id.orig_h":"10.0.0.2","id.resp_h":"10.0.0.3","id.orig_p":4444,"id.resp_p":23,"proto":"tcp","orig_pkts":120,"orig_ip_bytes":4096},"row_id":42}'

# Con el case_id devuelto, recalcula la auditoría posterior sin mutar el caso
Invoke-RestMethod http://127.0.0.1:8000/cases/<case_id>/audit
```

Devuelve el `CaseResult` completo. La persistencia y la contextualización LLM
anclada son políticas obligatorias de `/cases/analyze`, no parámetros del
cliente. El proveedor online está fijado a Mistral y el modelo se puede elegir
mediante `MITIGATOR_LLM_MODEL`; si falla, hay *fallback* total al catálogo.
La superficie pública se limita a `POST /cases/analyze`, que acepta `row` o
`text`, `GET /cases/{case_id}/audit` y `GET /health`. Los endpoints antiguos
responden `404`; un campo `canonical_event` enviado al endpoint vigente se
rechaza por contrato con `422`.

### Estado persistente

El runtime usa por defecto `artifacts/` como raíz escribible para su estado.
Puede fijarse otra con `TFM_STATE_DIR`; la caché de estandarización y la memoria
de casos pueden sobrescribirse de forma independiente mediante
`TFM_STANDARDIZATION_CACHE_DB` y `TFM_CASE_MEMORY_DB`. Los modelos y el catálogo
son recursos de solo lectura empaquetados y no se escriben durante la operación.
Dentro de la raíz, las rutas por defecto son
`cache/mistral_standardization_v2.sqlite3` y `case_memory.db` para mantener la
compatibilidad con el estado operativo existente.

## Auditoría del sistema

```powershell
python scripts\run_system_audit.py
```

Esta utilidad offline requiere los datasets y resultados congelados de la
campaña, que no se distribuyen en el paquete ni en Git. Reproduce la auditoría
de extremo a extremo, contrasta las métricas de los modelos, revisa baselines,
cobertura del catálogo y fugas de etiquetas. No es necesaria para ejecutar el
servicio online.

## Estructura del repositorio

| Carpeta | Contenido |
|---|---|
| `src/contracts/` | Esquemas Pydantic: evento canónico, salidas de agentes, `CaseResult` y traza |
| `src/agents/final/` | Cinco agentes operacionales y el auditor posterior independiente |
| `src/mcp/` | Tres servidores MCP: inferencia, memoria de casos y catálogo de amenazas; incluye modelos y catálogo empaquetados |
| `src/orchestration/` | Grafo LangGraph del flujo final y su estado |
| `src/eval/` | Sanitización, entrenamiento y evaluación exclusivamente offline |
| `src/api/` | API FastAPI + visor web (`static/index.html`) |
| `scripts/` | Utilidades offline de entrenamiento, evaluación y validación |
| `docs/` | Arquitectura, guion de demostración y material para la memoria |
| `tests/` | Suite automatizada completa |

El repositorio de despliegue no contiene adaptadores, agentes alternativos ni
grafos legacy. La sanitización se conserva como herramienta offline para
preparar entrenamiento y evaluación; nunca forma parte de `/cases/analyze`.
En operación, la caché SQLite solo almacena éxitos Mistral y reconstruye la
identidad y procedencia del registro duplicado actual.

## Reglas del proyecto

1. **No relanzar llamadas masivas al proveedor LLM**: los resultados de la
   campaña experimental están congelados en sus artefactos.
2. **Ninguna columna target puede usarse como feature** (hay verificación
   automática y un caso trampa en la auditoría).
3. La suite de tests debe quedar en verde tras cada cambio.
