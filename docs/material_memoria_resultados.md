# Material para la memoria: resultados, narrativa y limitaciones

> **DOCUMENTO HISTORICO, NO TABLA FINAL.** El diagnostico posterior de
> `plan_validacion_metricas_por_dataset.md` detecto contaminacion por duplicados
> y una inconsistencia en la procedencia de 0.9363. Estas tablas solo conservan
> trazabilidad; deben sustituirse con los resultados limpios de las fases C-F.

**Fecha:** 2026-08-03. Los números proceden de campañas científicas congeladas;
este documento conserva su trazabilidad, pero el paquete online solo incluye
los dos modelos activos y el catálogo necesarios para inferencia. Los informes
experimentales completos no son una dependencia del servicio.

---

## 1. Narrativa obligatoria

> La mejora no debe presentarse como «otro clasificador gana», sino como
> **«la arquitectura genera una representación canónica más rica y
> reutilizable, que permite mejores modelos posteriores»**. Comparativa
> clave: F1 multiclase **0.9363** (sistema actual, evento canónico
> Edge-IIoTset) frente a **0.7479** (mejor referencia LLM fine-tuned del
> TFM previo).

## 2. Tabla final — multiclase Edge-IIoTset (comparativa con el TFM previo)

Procedencia versionada:
`artifacts/baselines/jorge_and_current_baselines.json`. El informe fuente
completo pertenece al archivo científico de la campaña, no al runtime.

| Sistema | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Jorge — XGBoost (baseline ML) | 0.5164 | 0.6325 | 0.5164 | 0.4987 |
| Réplica estricta de Jorge (7 columnas) | 0.5200 | 0.6624 | 0.5200 | 0.4993 |
| Jorge — DeepSeek fine-tuned (mejor LLM previo) | 0.7502 | 0.8009 | 0.7502 | **0.7479** |
| **Sistema actual — evento canónico Edge** | 0.9360 | 0.9386 | 0.9360 | **0.9363** |
| Sistema actual — corpus estandarizado LLM balanceado | 0.8980 | 0.8978 | 0.8980 | 0.8967 |

Delta frente a la mejor referencia de Jorge: **+0.1884**. La réplica
estricta (0.4993 ≈ 0.4987) demuestra que la referencia está bien
reproducida antes de compararse contra ella.

## 3. Tabla final — binario

| Sistema | F1 binario |
|---|---:|
| Jorge | 1.0000 |
| Sistema actual — evento canónico Edge (ExtraTrees) | 0.9994 |
| Sistema actual — corpus estandarizado LLM balanceado | 0.9966–0.9972 |

Lectura honesta: en binario ambos sistemas están saturados; la mejora
diferencial del TFM está en multiclase y en la arquitectura.

## 4. Modelos generalistas desplegados en la capa MCP

Los modelos que sirven la arquitectura final son **generalistas
multi-dataset** (no específicos de Edge). Los dos artefactos activos se
distribuyen dentro del paquete y usan un contrato ligero de inferencia que no
importa módulos de evaluación. Sus métricas proceden de la campaña final
`validation_2026`, con particiones congeladas y deduplicadas.

| Modelo desplegado | Test (multi-dataset) | Métrica |
|---|---:|---|
| Detección binaria (`xgboost_detection_validation_2026_20260822`) | n=5.064 | F1 0.9676; F1 0.977 sobre casos decididos |
| Tipos de ataque (`xgboost_attack_subtype_multidataset16_balanced500_20260906`) | n=1.200 balanceado | macro-F1 0.8917; 0.9237 sobre casos decididos con umbral 0.65 |

Punto para la memoria: ambos modelos operacionales comparten una representación
canónica multi-dataset y aplican abstención o revisión humana en las regiones
de menor confianza; no deben confundirse con los modelos históricos de julio.

## 5. Comparativa de mitigación (documentación y auditabilidad)

Procedencia: lectura de `TFM_Jorge_Juan_Tejero-Threat_IoT.pdf` (Tabla 3.4)
y su versión artículo (arXiv:2507.02390, Tabla 3); catálogo
`src/mcp/data/threat_intel_catalog.json` v2.0; tool MCP
`get_jorge_capec_coverage` (verificado 14/14 por test y por el auditor).

| Dimensión | TFM previo (Jorge) | Este TFM |
|---|---|---|
| Documentación | Solo MITRE CAPEC (mapeo manual de 14 clases) | Catálogo específico para 16 tipos, que conserva el mapeo CAPEC de Jorge (14/14) y añade técnicas y mitigaciones ATT&CK |
| Generación | LLM libre (DeepSeek, temp 0.7) sobre contramedidas genéricas | LLM **anclado al catálogo**: solo contextualiza; todo lo no respaldado queda marcado `llm_suggested` |
| Priorización | No | Fases contención / erradicación / prevención + acciones por `schema_profile` |
| Trazabilidad | No (texto plano) | Procedencia por ítem (`MitigationItem.source`) y referencias con URL |
| Evaluación | Circular: ROUGE-L/coseno contra salidas del propio DeepSeek | Auditoría estructural: cobertura verificable + regla dura testeada (referencias inventadas no escapan sin marca) |
| Robustez | Depende del LLM | Fallback al catálogo solo en mitigación; un fallo de estandarización en *cache miss* produce abstención |

## 6. Limitación honesta: LODO (leave-one-dataset-out)

Procedencia: informes congelados de la campaña LODO de julio de 2026,
conservados en el archivo científico externo al paquete online.

Detección binaria LODO (F1 al excluir el grupo del entrenamiento):

| Grupo excluido | n | F1 LODO |
|---|---:|---:|
| BOT_IOT | 200 | 0.8065 |
| IOT23 | 200 | 0.7948 |
| TON_IOT_telemetry | 1400 | 0.6641 |
| TON_IOT_windows | 400 | 0.6621 |
| TON_IOT_linux | 600 | 0.5979 |
| EDGE_IIOTSET | 200 | 0.5570 |
| TON_IOT_network | 200 | 0.3270 |

Nota: el artefacto incluye además el grupo degenerado `SIMULATED_ALERTS`
(n=2, F1 0.0), omitido de la tabla por no ser estadísticamente
interpretable; se declara aquí para no ocultar la peor fila.

Clasificación de familia LODO (weighted-F1): entre **0.1272 y 0.3216** según
el grupo excluido — la generalización a orígenes completamente no vistos sigue
siendo el reto abierto. Presentarlo como limitación explícita: la
representación canónica reduce el acoplamiento al dataset, pero no elimina
el domain shift entre familias de datasets; con el grupo presente en
entrenamiento (aunque balanceado a 100 muestras por par), el rendimiento se
recupera (§4).

## 7. Otras limitaciones a declarar

- **Dependencia de Mistral para estandarización en vivo**: el sistema exige
  entrada cruda limpia. Una caché SQLite por hash exacto puede reutilizar solo
  éxitos Mistral previos para contenido duplicado, reconstruyendo la identidad
  y procedencia actuales. En un *cache miss* se requiere la API; un fallo
  produce abstención y revisión humana, sin adaptadores. La sanitización
  anti-leakage pertenece exclusivamente a la preparación de entrenamiento,
  validación y evaluación, fuera del grafo. La API pública solo acepta `row` o
  `text`: no ofrece un modo offline ni una entrada `canonical_event` para
  evitar Mistral.
- **Raw vs canónico en Edge**: parte de la mejora frente al baseline crudo
  proviene de usar todas las columnas y de la representación; la réplica
  estricta de Jorge (0.4993) aísla esa comparación honestamente.
- **Juez por reglas**: el juez final es determinista por umbrales (no LLM);
  la revisión humana es la salida prevista para los casos dudosos (así lo
  muestran la demo y la auditoría, no es un fallo).
- **MCP por stdio**: el runtime productivo atraviesa por defecto el protocolo
  MCP real. Para cada caso, el cliente abre de forma perezosa como máximo un
  proceso y una sesión por servidor, los reutiliza en todas las llamadas del
  flujo y los cierra después de persistir el `CaseResult`. La demo rápida fija
  `inprocess` y ejecuta las mismas tools y contratos sin transporte MCP.
- **Evaluación de mitigaciones**: se audita estructura, cobertura y
  anclaje; las referencias asistidas por IA fueron revisadas una a una por el
  autor, pero no existe validación por expertos humanos independientes.

## 8. Evidencia de sistema (para el capítulo de validación)

- **Auditoría E2E** (`scripts/run_system_audit.py`): 7 semáforos en VERDE,
  determinista (menos de 10 s): smoke E2E 20/20 casos válidos en 4 datasets, batch
  con delta 0.0 vs métricas congeladas, baselines certificados sin
  re-experimentar, cobertura CAPEC 14/14, leakage 0/200 con caso trampa
  detectado. Informe: `artifacts/informe_auditoria_sistema_<fecha>.md`.
- **Demo online**: el frontend envía entradas crudas a `/cases/analyze`, muestra
  las decisiones por agente y recupera la auditoría del caso persistido. Un
  duplicado exacto permite comprobar la reutilización de un éxito Mistral desde
  la caché; un *cache miss* necesita acceso al proveedor.
- **Suite**: pruebas automatizadas para MCP, agentes finales, mitigación
  anclada, auditor y demo; el recuento exacto se obtiene con `pytest -q`.
- **Trazabilidad**: todo caso devuelve `case_id` + `trace[]` completa con
  tiempos, confianzas y errores por agente; el endpoint final persiste siempre
  en la memoria de casos SQLite (la ejecución interna/CLI conserva controles
  explícitos para pruebas aisladas).
