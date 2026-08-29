# Sistema Multiagente para el Análisis de Amenazas IoT/IIoT

Trabajo Fin de Máster (Máster en Inteligencia Artificial Aplicada). Este
proyecto convierte un flujo clásico de detección de intrusiones —acoplado a un
único dataset— en un **sistema multiagente**: los eventos de seguridad de
cualquier fuente IoT/IIoT se transforman a un formato común (el *evento
canónico*) y una cadena de agentes especializados los analiza, propone
mitigaciones verificables y deja una traza completa auditable de cada decisión.

## ¿Cómo funciona?

Cada evento recorre este flujo, coordinado por un orquestador (LangGraph)
sobre una capa de herramientas MCP (Model Context Protocol):

```
entrada cruda → Estandarizador → Detector → Clasificador → Mitigador → Juez
                                    │ (zona gris)              │
                                    └──────► revisión humana ◄─┘
```

| Agente | Qué hace | Qué decide |
|---|---|---|
| **Estandarizador** | Convierte el registro crudo (de cualquier dataset) en un evento canónico, con ayuda de un LLM (Mistral) o de un adaptador determinista | Confianza del mapeo; si es baja (< 0,5), el caso va a revisión |
| **Detector** | Modelo XGBoost binario: ¿es malicioso? | Veredicto y probabilidad; en la zona gris [0,4–0,6] **se abstiene** |
| **Clasificador** | Modelo XGBoost multiclase: ¿qué familia de ataque? | Familia y confianza; si < 0,65, el caso va a revisión |
| **Mitigador** | Propone contramedidas desde un **catálogo verificable** (ATT&CK + CAPEC); un LLM opcional las contextualiza al caso concreto | Todo lo que el LLM añada sin respaldo del catálogo queda marcado `llm_suggested` |
| **Juez** | Reglas deterministas sobre el caso completo | Aprobar o derivar a revisión humana |
| **Auditor** | Revisa a posteriori el caso cerrado (coherencia, traza, umbrales, fugas) | Aprobar, revisar o rechazar |

Ideas clave del diseño:

- **El sistema nunca ve las etiquetas.** La verdad terreno se separa de los
  datos antes de entrar al flujo (como ocurriría en producción, donde los
  eventos llegan sin etiquetar) y solo la usa la evaluación, fuera del grafo.
- **Nada se acepta sin auditar**: también los casos benignos pasan por el juez,
  y todo caso produce un `CaseResult` con su traza paso a paso.
- **El LLM está anclado**: puede redactar y contextualizar, pero no puede
  inventar referencias — lo no respaldado por el catálogo se marca y el caso
  se supervisa.

## Resultados principales (campaña `validation_2026`)

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
# 1. Instalar (Python 3.10+)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[mcp,inference]"

# 2. Comprobar que todo está en verde (237 tests)
python -m pytest -q

# 3. Ver el sistema funcionando: demo reproducible de 4 casos (sin APIs externas)
python scripts\demo_mcp_multiagent_case.py --offline
```

La demo es determinista: cada caso se audita en vivo y su resultado se firma
con un resumen SHA-256; repetirla otro día debe producir firmas idénticas.

## Probar el sistema con el visor web

```powershell
uvicorn src.api.app:app --port 8000
```

Abre `http://localhost:8000`: un visor con ejemplos precargados muestra el
flujo completo (pipeline por agente, decisiones, mitigaciones con referencias
y la traza del caso). Sin las casillas de LLM, el flujo es 100 % determinista;
para la contextualización por LLM define `MISTRAL_API_KEY` en el entorno del
servidor.

También puedes llamar a la API directamente:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/cases/analyze -Method Post `
  -ContentType "application/json" `
  -Body '{"dataset":"iot23","row":{"id.orig_h":"10.0.0.2","id.resp_h":"10.0.0.3","id.orig_p":4444,"id.resp_p":23,"proto":"tcp","orig_pkts":120,"orig_ip_bytes":4096},"row_id":42}'
```

Devuelve el `CaseResult` completo. Con `"use_llm_mitigator": true` se activa la
contextualización LLM anclada (proveedor y modelo vía `MITIGATOR_LLM_PROVIDER`
y `MITIGATOR_LLM_MODEL`; si el LLM falla, hay *fallback* total al catálogo).

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
| `src/orchestration/` | Grafos LangGraph (el flujo final está en `mcp_graph.py`) |
| `src/eval/` | Código de la campaña de validación (contratos de datos, modelos candidatos, benchmarks del TFM previo) |
| `src/api/` | API FastAPI + visor web (`static/index.html`) |
| `scripts/` | Los 5 scripts activos (demo, auditoría, entrenamiento, LODO, estandarización en vivo); los históricos están archivados en `scripts/archive/` |
| `artifacts/` | Artefactos congelados: modelos desplegados, baselines, métricas de la campaña (`validation_2026/evaluation/`), demo |
| `validacion_fase_c/` | Scripts y resultados de las validaciones por agente del capítulo 5 de la memoria |
| `docs/` | Arquitectura, planes de trabajo y material para la memoria |
| `tests/` | Suite completa (237 tests) |

Los datos crudos (`data/`), la caché de estandarización y los resultados
masivos de la campaña quedan fuera de git por tamaño, pero **no deben borrarse
del disco**: son la procedencia de los resultados congelados.

## Reglas del proyecto

1. **No relanzar llamadas masivas al proveedor LLM**: la estandarización de la
   campaña está congelada en `artifacts/validation_2026/standardized/`.
2. **No borrar ni regenerar artefactos históricos** de `artifacts/`.
3. **Ninguna columna target puede usarse como feature** (hay verificación
   automática y un caso trampa en la auditoría).
4. La suite de tests debe quedar en verde tras cada cambio.
