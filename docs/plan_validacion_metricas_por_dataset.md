# Plan de validación de métricas por dataset (protocolo Jorge, split limpio)

**Fecha:** 2026-08-03 · **Motivación:** las métricas ~0,93 de Edge canónico provienen
de un benchmark con adapter (no LLM) y split sin deduplicación → casi-duplicados de
train en test. Hay que revalidar TODOS los datasets de entrada a través del agente
de estandarización, con protocolo espejo del TFM de Jorge y particiones limpias.

---

## Diagnóstico que motiva el plan

| Cifra congelada | Origen | Problema |
|---|---|---|
| 0,9363 multiclase Edge | `edgeiiot_canonical_adapter_tfm_less_current` (jun) | Feature source = **adapter** (no LLM); split 70/15/15 **sin dedup** sobre un dataset con >63k duplicados; la cifra del artefacto es 0,9307, no 0,9363 (inconsistencia entre artefactos similares) |
| 0,8967 multiclase Edge | `informe_..._balanced450` (jun, estandarizado LLM) | Metodológicamente mejor (label-leak 0), pero split aleatorio sin dedup → riesgo de casi-duplicados |
| 0,9994 binario Edge | mismo benchmark adapter | Mismo riesgo de dedup (aunque binario está saturado también en Jorge) |

**Consecuencia:** el capítulo 5 de la memoria y los baselines congelados deben
regenerarse con el protocolo de este plan.

## Objetivo

1. Validar el **agente de estandarización** sobre TODOS los datasets de `data/`:
   calidad del mapeo y utilidad de la representación canónica por dataset.
2. Obtener métricas de detección/clasificación **limpias** por dataset
   (split sin contaminación), con protocolo comparable al TFM de Jorge.
3. En EDGE_IIOTSET: comparación directa y honesta contra sus referencias
   (ML 0,4987 · DeepSeek FT 0,7479). Éxito = superar 0,7479 con split limpio;
   mínimo aceptable = superar claramente 0,4987.

## Protocolo común (espejo del TFM de Jorge)

- **Multiclase:** 500 registros por clase nativa del dataset (o el máximo
  disponible si hay menos, declarándolo). Split 70/15/15 estratificado.
- **Binario:** 5.900 normales + 5.900 ataques (o equivalente disponible),
  split 70/15/15.
- **Modelos:** los 4 de Jorge (RandomForest, XGBoost, LightGBM, MLP) con
  hiperparámetros alineados — mantiene la comparabilidad metodológica.
- **Métricas:** accuracy, precision, recall, F1 (weighted) por tarea y dataset.
- **Higiene de split (lo nuevo y crítico):**
  1. Deduplicación por representación canónica ANTES del split (dedup exacto
     sobre el vector de features + dedup de casi-duplicados por hash de la
     tupla de campos técnicos).
  2. Sanitización verificada: 0 campos target en features (chequeo del auditor).
  3. Semilla fija y particiones serializadas (reproducible).
  4. Informe del solapamiento residual train/test (debe ser 0 exacto).
- **Métricas del agente de estandarización por dataset** (el valor central):
  tasa parse_ok, mapping_confidence media, schema_profile asignado,
  label-leak = 0, cobertura de campos técnicos y tasa de abstención/reintento.

## Datasets y volúmenes

| Dataset | Clases nativas | Cache LLM disponible | Necesita estandarización nueva |
|---|---|---|---|
| EDGE_IIOTSET | 15 (14 ataques + normal) | 18.150 filas (cache jun, protocolo Jorge) | Depende de opción (ver abajo) |
| TON_IOT (network/linux/windows/telemetry) | ~7-9 por subset | ~11.000 filas (cache jul) | Parcial |
| IOT23 | ~5-10 etiquetas | 598-1.410 filas | Sí para multiclase 500/clase |
| BOT_IOT | 4-5 categorías | 1.078-1.880 filas | Sí para multiclase 500/clase |
| URBAN_IOT | por determinar (sin adapter) | 0 | Sí (100 % en vivo, vía agente genérico+LLM) |
| SIMULATED | descartado (n=2, degenerado) | — | No |

## Fases

### Fase A — Diagnóstico de contaminación (sin coste LLM, ~0,5-1 día)
Reproducir los benchmarks de junio y medir el solapamiento train/test real
(duplicados exactos y casi-duplicados). Recalcular la métrica tras dedup.
**Entregable:** informe con la métrica corregida de los experimentos antiguos —
cuantifica el efecto de la contaminación antes de re-experimentar.

### Fase B — Muestreo estratificado por dataset (sin coste LLM)
Para cada dataset: inventario de clases nativas, muestreo 500/clase + binario
(desde `data/`), dedup previo, particiones 70/15/15 serializadas y congeladas
ANTES de estandarizar. **Entregable:** manifiestos de partición por dataset.

### Fase C — Estandarización de las muestras
**Estado 2026-08-31:** política final estricta con `mistral-small-2603`, salida
incremental reanudable y supervisor preparado para reintentar abstenciones.
**DECISIÓN (2026-08-03, confirmada por el propietario del proyecto): opción C3 —
el 100 % de las muestras de TODOS los datasets (incluido URBAN_IOT) se
estandariza con el LLM EN VIVO.** La implementación final no acepta caché,
selección heurística ni adaptador de respaldo: un fallo queda como abstención
y se reintenta; nunca entra como resultado válido al corpus. Esta campaña
sustituye a la regla «no llamadas masivas» del handoff y se ejecuta con
manifiesto, reanudación y JSONL incremental como checkpoint del experimento.
Estimación: ~50.000-70.000 llamadas (ajustar con el inventario de Fase B);
lotes con reintentos, throttling y checkpoint por dataset.
Verificación anti-leakage sobre todo lo estandarizado.

### Fase D — Entrenamiento y evaluación limpios (~1 día)
**Estado 2026-08-06:** implementada y cubierta por tests; queda a la espera de
que la Fase C alcance cobertura LLM del 100 % para producir cifras finales.
Script nuevo `scripts/eval_validacion_por_dataset.py`: por dataset y tarea,
entrena los 4 modelos sobre train, selecciona en val, reporta test. Un artefacto
JSON+MD congelado por dataset + tabla global. El test set jamás toca el
entrenamiento (verificado por manifiesto).

### Fase E — Comparativa Edge vs Jorge
Con el split limpio de Edge: tabla directa contra 0,4987 (ML), 0,4993 (réplica)
y 0,7479 (DeepSeek FT). Criterios:
- **Éxito pleno:** F1 multiclase limpio > 0,7479.
- **Éxito parcial:** > 0,4987 pero ≤ 0,7479 → narrativa: representación canónica
  comparable al LLM fine-tuned con coste de inferencia muy inferior y
  arquitectura auditable (sigue siendo defendible).
- Binario: reportar sin sobreinterpretare (saturado también en Jorge).

### Fase F — Actualización de memoria y sistema (~1 día)
1. Regenerar `artifacts/baselines/jorge_and_current_baselines.json` con los
   números limpios (v2, conservando el histórico con nota de deprecación).
2. Ajustar umbrales del auditor (`run_system_audit.py`) a los nuevos baselines.
3. Reescribir el capítulo 5 de la memoria: la tabla central pasa a ser la
   **validación por dataset del agente de estandarización** (lo que tiene valor)
   + comparativa Edge limpia; nota metodológica honesta sobre la corrección.
4. Revisar capítulos 1 y 6 y resumen (las menciones a 0,9363/+0,1884).

## Riesgos

| Riesgo | Mitigación |
|---|---|
| El F1 limpio de Edge cae por debajo de 0,7479 | Plan de contingencia narrativo de Fase E (éxito parcial); la validación multi-dataset del agente sigue siendo la aportación central |
| Coste/tiempo de estandarización en vivo | Opciones C1-C3 con estimaciones; C2 concentra el gasto donde importa |
| Cuotas/rate limit de Mistral en lotes | Lotes con reintentos y reanudación por manifiesto/JSONL; las abstenciones no se aceptan como estandarizaciones |
| Los caches antiguos usan claves distintas | Fase B normaliza el manifiesto (dataset::fichero::fila) y mapea las claves de junio |
| Nuevas cifras rompen el auditor actual | Fase F actualiza baselines y umbrales en el mismo commit |
