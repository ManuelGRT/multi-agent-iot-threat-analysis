# Material para la memoria: resultados, narrativa y limitaciones

> **DOCUMENTO HISTORICO, NO TABLA FINAL.** El diagnostico posterior de
> `plan_validacion_metricas_por_dataset.md` detecto contaminacion por duplicados
> y una inconsistencia en la procedencia de 0.9363. Estas tablas solo conservan
> trazabilidad; deben sustituirse con los resultados limpios de las fases C-F.

**Fecha:** 2026-08-03. Todos los números provienen de artefactos congelados
del repositorio (se indica la procedencia de cada tabla). Regla del
proyecto: los resultados congelados no se re-experimentan; el agente de
auditoría verifica su integridad y los umbrales en cada ejecución.

---

## 1. Narrativa obligatoria

> La mejora no debe presentarse como «otro clasificador gana», sino como
> **«la arquitectura genera una representación canónica más rica y
> reutilizable, que permite mejores modelos posteriores»**. Comparativa
> clave: F1 multiclase **0.9363** (sistema actual, evento canónico
> Edge-IIoTset) frente a **0.7479** (mejor referencia LLM fine-tuned del
> TFM previo).

## 2. Tabla final — multiclase Edge-IIoTset (comparativa con el TFM previo)

Procedencia: `artifacts/baselines/jorge_and_current_baselines.json`
(congelado en Fase 0 desde
`artifacts/informe_edgeiiot_estandarizado_ml_vs_tfm_jorge_balanced450_20260622.md`).

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
multi-dataset** (no específicos de Edge). Procedencia: artefactos de
entrenamiento `artifacts/xgboost_*_20260705_*.json`; el auditor reproduce
el split de test (seed 42) y verifica no-regresión con **delta 0.0**.

| Modelo desplegado | Test (multi-dataset) | Métrica |
|---|---:|---|
| Detección binaria (`xgboost_detection_standardized_with_edge`) | n=481 | F1 0.7564 · ROC-AUC 0.8667 |
| Familia (`xgboost_attack_family_balanced_group`) | n=480 | weighted-F1 0.7438 |
| — slice Edge-IIoTset del mismo test | n=90 | weighted-F1 **0.7593 > 0.7479 (Jorge)** |

Punto para la memoria: incluso el clasificador **generalista** (entrenado
para 7+ orígenes heterogéneos a la vez) supera en el slice Edge al mejor
LLM fine-tuned específico del TFM previo; el especializado canónico llega
a 0.9363.

## 5. Comparativa de mitigación (documentación y auditabilidad)

Procedencia: lectura de `TFM_Jorge_Juan_Tejero-Threat_IoT.pdf` (Tabla 3.4)
y su versión artículo (arXiv:2507.02390, Tabla 3); catálogo
`src/mcp/data/threat_intel_catalog.json` v1.1; tool MCP
`get_jorge_capec_coverage` (verificado 14/14 por test y por el auditor).

| Dimensión | TFM previo (Jorge) | Este TFM |
|---|---|---|
| Documentación | Solo MITRE CAPEC (mapeo manual de 14 clases) | Superset del mapeo CAPEC de Jorge (14/14) + técnicas ATT&CK + mitigaciones ATT&CK M-* |
| Generación | LLM libre (DeepSeek, temp 0.7) sobre contramedidas genéricas | LLM **anclado al catálogo**: solo contextualiza; todo lo no respaldado queda marcado `llm_suggested` |
| Priorización | No | Fases contención / erradicación / prevención + acciones por `schema_profile` |
| Trazabilidad | No (texto plano) | Procedencia por ítem (`MitigationItem.source`) y referencias con URL |
| Evaluación | Circular: ROUGE-L/coseno contra salidas del propio DeepSeek | Auditoría estructural: cobertura verificable + regla dura testeada (referencias inventadas no escapan sin marca) |
| Robustez | Depende del LLM | Fallback total al catálogo: la demo nunca se rompe |

## 6. Limitación honesta: LODO (leave-one-dataset-out)

Procedencia: `artifacts/xgboost_detection_group_lodo_20260705_*.json` y
`artifacts/xgboost_attack_family_group_lodo_20260705_*.json`.

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
  validación y evaluación, fuera del grafo. Para la defensa, el modo
  `--offline` utiliza `canonical_event` congelados y ya estandarizados.
- **Raw vs canónico en Edge**: parte de la mejora frente al baseline crudo
  proviene de usar todas las columnas y de la representación; la réplica
  estricta de Jorge (0.4993) aísla esa comparación honestamente.
- **Juez por reglas**: el juez final es determinista por umbrales (no LLM);
  la revisión humana es la salida prevista para los casos dudosos (así lo
  muestran la demo y la auditoría, no es un fallo).
- **MCP por stdio**: el protocolo real funciona (tests + handshake en la
  demo), pero arrancar un servidor por llamada recarga los modelos y es
  inviable operativamente; el modo in-process comparte contrato de tools.
- **Evaluación de mitigaciones**: se audita estructura, cobertura y
  anclaje; no hay validación con expertos humanos (igual que en el TFM
  previo, que además tenía evaluación circular — declararlo).

## 8. Evidencia de sistema (para el capítulo de validación)

- **Auditoría E2E** (`scripts/run_system_audit.py`): 7 semáforos en VERDE,
  determinista (menos de 10 s): smoke E2E 20/20 casos válidos en 4 datasets, batch
  con delta 0.0 vs métricas congeladas, baselines certificados sin
  re-experimentar, cobertura CAPEC 14/14, leakage 0/200 con caso trampa
  detectado. Informe: `artifacts/informe_auditoria_sistema_<fecha>.md`.
- **Demo reproducible** (`scripts/demo_mcp_multiagent_case.py --offline`):
  4 casos con digest SHA256 estable idéntico entre ejecuciones y entre
  procesos; casos de baja confianza derivados a revisión humana.
- **Suite**: pruebas automatizadas para MCP, agentes finales, mitigación
  anclada, auditor y demo; el recuento exacto se obtiene con `pytest -q`.
- **Trazabilidad**: todo caso devuelve `case_id` + `trace[]` completa con
  tiempos, confianzas y errores por agente; el endpoint final persiste siempre
  en la memoria de casos SQLite (la ejecución interna/CLI conserva controles
  explícitos para pruebas aisladas).
