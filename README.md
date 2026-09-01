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
| **Estandarizador** | Recibe una entrada cruda ya limpia. Una caché SQLite por hash exacto reutiliza únicamente éxitos previos de Mistral para contenido duplicado y reconstruye la identidad y procedencia del caso actual. En un *cache miss*, Mistral selecciona las columnas y genera el evento; un `canonical_event` previo solo se valida | Confianza del mapeo; si no hay *hit* y Mistral falla, la respuesta es inválida o la confianza es baja (< 0,5), se abstiene y el juez deriva el caso a revisión humana. Nunca usa adaptadores |
| **Detector** | Modelo XGBoost binario: ¿es malicioso? | Veredicto y probabilidad; en la zona gris [0,4–0,6] **se abstiene** |
| **Clasificador** | Modelo XGBoost multiclase: ¿qué familia de ataque? | Familia y confianza; si < 0,65, el caso va a revisión |
| **Mitigador** | Propone contramedidas desde un **catálogo verificable** (ATT&CK + CAPEC) e intenta siempre contextualizarlas con Mistral en el endpoint final | Si el LLM falla conserva el catálogo; todo añadido sin respaldo queda marcado `llm_suggested` |
| **Juez** | Reglas deterministas sobre el caso completo | Aprobar o derivar a revisión humana |
| **Auditor** | Revisa a posteriori el caso cerrado (coherencia, traza, umbrales, fugas) | Aprobar, revisar o rechazar |

Ideas clave del diseño:

- **La entrada limpia es una precondición.** La sanitización anti-leakage y la
  separación de columnas target se realizan al preparar los datos de
  entrenamiento, validación y evaluación, siempre fuera del grafo. El runtime
  final no limpia ni inspecciona targets dentro del `row`: presupone una entrada
  preparada y valida su estructura. Sí rechaza contaminación predictiva en el
  `CanonicalEvent`, tanto si llega preestandarizado como si lo produce Mistral.
- **Mistral es obligatorio en cada *cache miss*.** La caché SQLite se indexa
  por el hash exacto del contenido limpio y solo reutiliza respuestas Mistral
  exitosas para duplicados. En un *hit* se reconstruyen la identidad y la
  procedencia actuales y se reutiliza la selección que Mistral hizo para ese
  mismo contenido; no aparece una heurística determinista. Si no hay *hit* y
  Mistral falla, el sistema se abstiene y entrega el caso al juez para revisión
  humana. No existe ruta por adaptador. El mecanismo *single-flight* evita
  llamadas repetidas para una misma clave dentro de cada proceso; entre
  procesos, SQLite aplica la política del primer éxito escrito, aunque dos
  *misses* simultáneos pueden llegar a invocar Mistral.
- **Todo caso pasa por el juez y queda auditable**: también los benignos se
  revisan operacionalmente y todos producen un `CaseResult` con su traza. El
  `CaseAuditor` es una comprobación posterior e independiente.
- **El LLM del mitigador está anclado**: puede redactar y contextualizar, pero no puede
  inventar referencias — lo no respaldado por el catálogo se marca y el caso
  se supervisa.

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
| Clasificador de familia (n=2.088) | F1 0,9107; con umbral de confianza: 0,953 sobre lo decidido |
| Mitigador | 100 % de ítems con procedencia de catálogo; anclaje verificado también con Mistral en vivo |
| Auditor | 16/16 defectos inyectados detectados; 0 falsos rechazos |

**Límite declarado:** el control *leave-one-dataset-out* muestra que los
modelos no generalizan a una fuente no vista en entrenamiento; los resultados
valen para los dominios representados. La mejora frente al TFM previo procede
de la **representación canónica**, no del algoritmo (la réplica a igualdad de
columnas reproduce su baseline).

## Empezar en tres pasos

```powershell
# 1. Instalar (Python 3.12+)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[mcp,inference]"

# 2. Comprobar que todo está en verde
python -m pytest -q

# 3. Ver el sistema funcionando: demo reproducible de 4 casos (sin APIs externas)
python scripts\demo_mcp_multiagent_case.py --offline
```

La demo es determinista porque utiliza cuatro eventos canónicos congelados:
no envía registros crudos al estandarizador ni realiza llamadas externas. Cada
caso sí atraviesa la validación contractual, se audita en vivo y su
resultado se firma con un resumen SHA-256; repetirla otro día debe producir
firmas idénticas.

## Probar el sistema con el visor web

```powershell
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
caso conserva las contramedidas y referencias verificables del catálogo. Para
una ejecución sin APIs externas utiliza la demo `--offline`, que carga eventos
canónicos congelados.

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
cliente. El proveedor y modelo se pueden configurar mediante
`MITIGATOR_LLM_PROVIDER` y `MITIGATOR_LLM_MODEL`; por defecto se usa Mistral y,
si falla, hay *fallback* total al catálogo.
Los endpoints antiguos `/events/analyze`, `/datasets/adapt` y
`/datasets/analyze-file` responden `410 Gone`: se han cerrado para que ninguna
ruta pública pueda reactivar adaptadores ni omitir la revisión del juez.
`/datasets/supported` conserva únicamente metadatos de formatos históricos;
no habilita su adaptación en el flujo final.

## Auditoría del sistema

```powershell
python scripts\run_system_audit.py
```

Ejecuta la auditoría de extremo a extremo: casos de todas las fuentes por el
grafo completo, certificación de los modelos desplegados contra las métricas
congeladas de la campaña (con integridad SHA-256 de la pertenencia del test),
baselines, cobertura del catálogo y chequeo de fuga de etiquetas (incluido un
caso trampa). El código de salida es 0 solo si los siete semáforos están en
verde.

## Estructura del repositorio

| Carpeta | Contenido |
|---|---|
| `src/contracts/` | Esquemas Pydantic: evento canónico, salidas de agentes, `CaseResult` y traza |
| `src/agents/final/` | Los seis agentes del flujo final |
| `src/mcp/` | Capa de herramientas: 5 servidores MCP (datasets, inferencia, memoria de casos, threat intel, evaluación) y cliente dual in-process/stdio |
| `src/orchestration/` | `mcp_graph.py` es el único grafo final activo; `graph.py` conserva el orquestador previo como código legacy inactivo |
| `src/eval/` | Preparación/sanitización anti-leakage y código de evaluación (contratos de datos, modelos candidatos, benchmarks del TFM previo) |
| `src/api/` | API FastAPI + visor web (`static/index.html`) |
| `scripts/` | Los 5 scripts activos (demo, auditoría, entrenamiento, LODO, estandarización en vivo); los históricos están archivados en `scripts/archive/` |
| `artifacts/` | Artefactos congelados: modelos desplegados, baselines, métricas de evaluación y demo |
| `validacion_fase_c/` | Scripts y resultados de las validaciones por agente del capítulo 5 de la memoria |
| `docs/` | Arquitectura, planes de trabajo y material para la memoria |
| `tests/` | Suite automatizada completa |

Los agentes y adaptadores anteriores que no están bajo `src/agents/final/`,
junto con `src/orchestration/graph.py`, se conservan únicamente como código
legacy/inactivo y compatibilidad histórica. Pueden mantener preprocesamiento de
compatibilidad, pero no forman parte de la ruta final ni contradicen su frontera:
`mcp_graph.py` presupone el `row` preparado y no ejecuta adaptadores.

Los datos crudos (`data/`), las cachés de estandarización y los resultados
masivos de evaluación quedan fuera de git por tamaño, pero **no deben borrarse
del disco**: son la procedencia de los resultados congelados. En operación, la
caché SQLite por hash exacto evita repetir Mistral para contenido limpio
duplicado; solo almacena éxitos y reconstruye la identidad/procedencia del
registro actual. La caché JSONL histórica se conserva intacta como evidencia,
pero no se mezcla automáticamente con la caché operativa porque corresponde a
otro modelo/contrato. La demo offline reutiliza eventos canónicos congelados.

## Reglas del proyecto

1. **No relanzar llamadas masivas al proveedor LLM**: los resultados de la
   campaña experimental están congelados en sus artefactos.
2. **No borrar ni regenerar artefactos históricos** de `artifacts/`.
3. **Ninguna columna target puede usarse como feature** (hay verificación
   automática y un caso trampa en la auditoría).
4. La suite de tests debe quedar en verde tras cada cambio.
