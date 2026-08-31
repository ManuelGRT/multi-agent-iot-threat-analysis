# Guión de la demo completa del sistema (en vivo, con LLMs)

> **AVISO PARA LA DEFENSA:** 0.9363 es una comparativa historica, no una metrica
> final con split limpio. No usar el mensaje central de este guion hasta
> completar las fases C-F de `plan_validacion_metricas_por_dataset.md` o
> reformularlo explicitamente como resultado preliminar.

**Proyecto:** `tfm_multiagent` · Sistema multiagente de ciberseguridad IoT/IIoT sobre MCP
**Modo:** en vivo — la entrada llega limpia; un duplicado exacto puede reutilizar un éxito Mistral y todo *cache miss* exige Mistral; la contextualización LLM del mitigador es opcional
**Duración estimada:** 15–20 minutos (los 5 bloques) · **Plan B:** modo `--offline` con `canonical_event` congelados (100 % determinista)

---

## Mensaje central (repetirlo en apertura y cierre)

> La aportación no es «otro clasificador que gana»: la arquitectura produce una
> **representación canónica más rica y desacoplada** que habilita mejores modelos
> posteriores (F1 multiclase **0,9363** frente a **0,7479** del mejor LLM
> fine-tuned del TFM previo), con **trazabilidad completa por caso** y
> **mitigación anclada a conocimiento verificable** (ATT&CK/CAPEC).

---

## 0. Preparación previa (el día antes — no en directo)

### Qué hacer

```powershell
cd <ruta>\Proyecto_Sistema_Multiagente

# 1) Entorno y suite completa (usar SIEMPRE el venv del proyecto)
.\.venv\Scripts\python.exe -m pytest -q

# 2) Configurar los LLM en vivo (misma terminal que usarás en la demo)
$env:MISTRAL_API_KEY = "<clave>"
$env:INGEST_LLM_MODEL = "mistral-small-2603"      # opcional; es el default actual
$env:INGEST_LLM_TIMEOUT_SECONDS = "60"            # recomendado para acotar cada llamada de la demo
$env:MITIGATOR_LLM_PROVIDER = "mistral"          # mitigacion contextualizada
$env:MITIGATOR_LLM_MODEL = "mistral-small-latest"
$env:MISTRAL_RATE_LIMIT_RETRIES = "5"            # opcional: eleva el default de 3 reintentos a 5
$env:MISTRAL_RETRY_WAIT_SECONDS = "2"            # espera base si la respuesta no aporta Retry-After

# 3) Ensayo general completo (una pasada de los bloques 1-4)
# 4) Red de seguridad: comprobar que el modo offline funciona
.\.venv\Scripts\python.exe scripts\demo_mcp_multiagent_case.py --offline
```

### Resultado esperado

- La suite completa termina en verde (el número de pruebas puede crecer).
- El ensayo offline termina con los 4 `canonical_event` congelados y
  **digests idénticos** a la referencia (`exit 0`). No llama al estandarizador
  LLM: solo aplica la validación contractual MCP. Si algo falla el
  día de la demo, este es el plan B.

> **Regla dura del proyecto:** el grafo recibe entradas limpias; la sanitización
> anti-leakage se realiza antes, al preparar entrenamiento, validación y
> evaluación. En operación, la caché SQLite por hash exacto solo reutiliza
> éxitos Mistral para duplicados; un *miss* siempre requiere Mistral.

---

## 1. Apertura — Auditoría E2E del sistema (~2 min)

**Qué contar mientras corre:** «Antes de enseñar nada, el propio sistema se
audita: humo de extremo a extremo en 4 datasets, métricas de los modelos
desplegados reproduciendo su split de test exacto, certificación de los
baselines congelados de la comparativa con el TFM previo, cobertura del mapeo
CAPEC y chequeo de fuga de etiquetas con un caso trampa que debe ser detectado.»

### Qué hacer

```powershell
.\.venv\Scripts\python.exe scripts\run_system_audit.py
```

### Resultado esperado

- **7 semáforos en VERDE**, ejecución determinista en **menos de 10 segundos**, `exit 0`.
- Smoke E2E: **20/20 casos válidos** (Edge-IIoTset, TON-IoT host y telemetry, IoT-23).
- Batch de modelos desplegados: **delta 0,0** frente a las métricas congeladas.
- Baselines certificados: 0,9363 ≥ 0,93 y > 0,7479 (sin re-experimentar).
- Cobertura CAPEC del TFM previo: **14/14**.
- Leakage: **0/200** y el caso trampa **sí detectado** (valida el detector de fugas).
- Se genera `artifacts/informe_auditoria_sistema_<fecha>.md` (mostrarlo un segundo).

---

## 2. Flujo multiagente completo con mitigación LLM (~5 min)

**Qué contar:** «Ahora el flujo completo sobre cuatro eventos canónicos
congelados: validación técnica → detección → clasificación → mitigación →
juez, con auditoría en vivo de cada caso. Al estar ya estandarizados, no se
repite Mistral; el runtime solo valida el contrato y no sanitiza.
La mitigación combina el catálogo threat intel (base auditable) con el LLM,
que solo contextualiza: nada fuera del catálogo puede presentarse como
conocimiento auditado.»

### Qué hacer

```powershell
.\.venv\Scripts\python.exe scripts\demo_mcp_multiagent_case.py --use-llm-mitigator
```

### Resultado esperado (4 casos por consola)

| Caso | Salida esperada |
|---|---|
| **Edge-IIoTset (DDoS_UDP)** | `detect: malicioso=True p=0.986` · `classify: familia=ddos confianza=0.995` · `judge: accion=approve` · estado `completed`, auditoría `approve` |
| **TON-IoT host** | `standardize: modelo=prestandardized_passthrough cache=False`; evento canónico congelado validado por MCP |
| **TON-IoT telemetry** | `cache=False`; si la confianza cae bajo los umbrales → `needs_human_review` (**explicarlo como comportamiento previsto**, no como fallo) |
| **IoT-23 (flujo de red)** | `cache=False`, flujo completo con traza de 6 agentes |

- En cada caso: `mitigate: fuente=hybrid` (catálogo + LLM) y referencias
  `T14xx / CAPEC-xxx / M1037`. Los ítems contextualizados por el LLM van
  marcados con su procedencia (`llm`); cualquier aportación sin respaldo saldría
  como `llm_suggested`.
- Traza impresa: `orchestrator -> final_standardizer -> final_detector ->
  final_classifier -> final_mitigator -> final_judge`.
- **Nota:** en esta orden la estandarización sigue usando `canonical_event`
  congelados; `--use-llm-mitigator` solo activa el LLM del mitigador. La
  verificación de digests idénticos aplica al modo `--offline`; con el
  mitigador LLM en vivo su redacción puede variar entre ejecuciones.

**Qué señalar en pantalla:** la línea de comparativa con el TFM previo
(0,9363 vs 0,7479, delta +0,1884) y la cobertura CAPEC 14/14 que el script
imprime al final vía tools MCP.

---

## 3. Protocolo MCP real (~1 min)

**Qué contar:** «La capa de herramientas no es una abstracción de papel: cada
servidor habla el protocolo MCP real por stdio. Lo demostramos con un handshake
auténtico contra el servidor de threat intel.»

### Qué hacer

```powershell
.\.venv\Scripts\python.exe scripts\demo_mcp_multiagent_case.py --offline --stdio-smoke
```

### Resultado esperado

- Handshake MCP correcto: el servidor `threat_intel` arranca como proceso,
  lista sus tools (`map_family_to_attack`, `suggest_mitigations`,
  `get_jorge_capec_coverage`…) y responde a una llamada real.
- **Si preguntan por qué no toda la demo va por stdio:** funciona (hay tests),
  pero arrancar un servidor por llamada recarga los modelos y es inviable en
  vivo; la operación real usaría servidores persistentes. Honestidad técnica.

---

## 4. Caso nuevo estandarizado EN VIVO por el LLM, en el frontal web (~5 min)

**Qué contar:** «Hasta ahora hemos usado eventos canónicos congelados. Ahora
entra un registro crudo ya limpio. El runtime calcula su hash exacto: si no hay
un éxito Mistral previo, Mistral realiza dos llamadas obligatorias, primero
selecciona las columnas relevantes y después extrae el evento canónico. La
salida se valida —no se limpia— antes de continuar. Nunca hay adaptador.»

### Qué hacer

Arrancar la API (hereda las variables de entorno de esta terminal) y abrir el
frontal:

```powershell
.\.venv\Scripts\uvicorn.exe src.api.app:app --port 8000
```

Abrir **http://localhost:8000** en el navegador (a pantalla completa para la
proyección) y:

1. Seleccionar el ejemplo **«Flujo IoT-23 sospechoso (telnet)»**.
2. Marcar, si se quiere mostrar la contextualización, **«Mitigación
   contextualizada por LLM»** y marcar **«Persistir caso»**. La
   estandarización viva no tiene casilla: para una entrada cruda Mistral es
   siempre obligatorio.
3. Pulsar **Analizar caso** y narrar la animación del pipeline mientras corre.

### Resultado esperado (y qué señalar en pantalla)

- El pipeline se ilumina agente a agente: Orquestador → Estandarizador →
  Detector → Clasificador → Mitigador → Juez.
- Tarjeta del Estandarizador: modelo, fuente `llm`, proveedor Mistral, perfil
  de esquema, modalidad y confianza de mapeo. **Este es el momento clave del
  bloque: han ocurrido dos llamadas en vivo** (selección + extracción); esa
  latencia es el coste declarado de estandarizar un `row` crudo. La selección
  de columnas queda registrada en `CaseResult.standardization`, pero el visor
  actual no la presenta como un campo independiente.
- Tarjeta del Detector: gauge de probabilidad **con la zona gris [0,4–0,6]
  dibujada** — si cae dentro, la abstención y la derivación al juez SE VEN
  (clasificador y mitigador quedan «omitido»): explicarla como comportamiento
  previsto, no como fallo.
- Tarjeta del Clasificador: familia con su top-5 de puntuaciones en barras (no
  prometer una familia concreta en un evento inventado; el valor está en las
  confianzas visibles).
- Tarjeta del Mitigador: fuente `hybrid`, cada mitigación con su badge de
  procedencia (`catalog`/`llm`/`llm_suggested`) y las referencias como chips
  clicables a attack.mitre.org — clicar una en vivo (T1498) para mostrar que
  las referencias son reales.
- Banner final con el estado y el `case_id`; tabla de traza con los 6 agentes
  y, plegado, el `CaseResult` JSON completo.

**Remate del bloque:** explicar la degradación controlada real: si falla la
selección, la extracción o la validación de Mistral después de un *cache miss*,
el estandarizador se abstiene y el juez cierra el caso con `human_interrupt` /
`needs_human_review`. Un *hit* solo puede reutilizar un éxito Mistral para el
mismo contenido limpio y reconstruye identidad/procedencia; nunca se ejecuta
un adaptador. La sanitización no forma parte del runtime ni del grafo.

**Alternativa sin navegador** (por si falla la proyección): la misma llamada
por consola con `Invoke-RestMethod http://127.0.0.1:8000/cases/analyze` y el
JSON del ejemplo (cuerpo con `"use_llm_mitigator": true`).

---

## 5. Cierre — Trazabilidad y evidencia (~3 min)

### Qué hacer

1. Abrir el JSON del caso persistido (o `artifacts/demo/demo_case_edge_iiotset_*.json`)
   y recorrer en 30 segundos: `case_id` → salidas por etapa → `trace[]` →
   referencias con URL de MITRE.
2. Mostrar el informe de auditoría generado en el bloque 1.
3. Cerrar con la tabla de la memoria (capítulo 5): 0,4993 (crudo) → 0,9363
   (canónico) vs 0,7479 (LLM FT previo).

### Qué decir (cierre)

> «Todo lo que han visto es reproducible: una suite automatizada en verde, artefactos congelados,
> digests estables en modo offline y una auditoría que certifica los baselines
> en cada ejecución. La mejora no viene de un clasificador mejor, sino de
> reorganizar el análisis en torno a una representación canónica y una
> arquitectura auditable de extremo a extremo.»

---

## Riesgos y plan B

| Riesgo | Señal | Reacción |
|---|---|---|
| API de Mistral caída o sin red | Abstención/timeout en el bloque 4 | Un duplicado exacto con éxito Mistral previo puede resolverse desde SQLite; ante *cache miss*, el estandarizador se abstiene y el juez responde `human_interrupt`. Nunca usa adaptadores. El mitigador sí conserva su fallback independiente al catálogo |
| Rate limit de Mistral | HTTP 429 | Los reintentos están configurados (`MISTRAL_RATE_LIMIT_RETRIES`); esperar unos segundos y repetir la llamada |
| Sin red total | Nada en vivo funciona | Ejecutar los bloques 1, 2 y 3 con `--offline`, usando los eventos canónicos congelados. El bloque de entrada cruda se sustituye por un `canonical_event` o se omite |
| Respuesta LLM lenta en el bloque 4 | supera el timeout configurado | Si vence `INGEST_LLM_TIMEOUT_SECONDS`, se registra la abstención y el juez solicita revisión humana |
| Pregunta incómoda: «¿el LLM se inventa mitigaciones?» | — | Enseñar `MitigationItem.source`: nada sale sin procedencia; lo no respaldado se marca `llm_suggested` (hay test que lo verifica) |

## Checklist final antes de empezar

- [ ] Terminal en la raíz del proyecto, venv activo (`.venv\Scripts\...`)
- [ ] Variables de entorno de los LLM exportadas (bloque 0)
- [ ] `pytest -q` verde ese mismo día
- [ ] Ensayo offline con digests OK (plan B verificado)
- [ ] `artifacts/` intacto (no borrar informes ni caché — regla dura)
- [ ] API arrancada y frontal abierto en http://localhost:8000 (bloque 4)
- [ ] Política de reintentos revisada: el cliente usa 3 por defecto; para la demo puede elevarse a 5
