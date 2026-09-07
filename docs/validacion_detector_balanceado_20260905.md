# Validacion del detector balanceado por origen

Fecha de ejecucion: 2026-09-05. El detector se reentreno exclusivamente sobre
eventos canonicos ya estandarizados correctamente por Mistral
(`mistral-small-2603`). No se utilizaron adaptadores ni se repitieron llamadas
al LLM. El clasificador de tipos de ataque no participo en este entrenamiento.

## Protocolo

1. Se conservaron los `train`/`val`/`test` congelados en los manifiestos.
2. Se eliminaron primero las huellas de features duplicadas o inseguras entre
   particiones.
3. Se agrupo por `split x origen x etiqueta binaria`.
4. En cada grupo se selecciono, sin reemplazo y con semilla 42, el minimo entre
   observaciones benignas y ataques.
5. TON-IoT se trato como cuatro origenes: red, telemetria, Linux y Windows.

`source_file` solo se utilizo para identificar el suborigen TON durante el
muestreo y el informe. No forma parte de las features predictivas.
El 1:1 es interno a cada origen; no se igualo el numero total entre origenes,
porque hacerlo reduciria todo el corpus al soporte de TON-IoT Windows y
descartaria la mayor parte de las observaciones disponibles.

## Soporte efectivo

De 33.637 observaciones binarias elegibles quedaron 33.635 despues de la
deduplicacion global y 23.604 despues del balance: 11.802 benignas y 11.802
ataques.

| Origen | Train por clase | Validacion por clase | Test por clase | Total por clase |
|---|---:|---:|---:|---:|
| Bot-IoT | 333 | 71 | 73 | 477 |
| Edge-IIoTset | 4.116 | 885 | 885 | 5.886 |
| IoT-23 | 825 | 175 | 181 | 1.181 |
| TON-IoT Linux | 539 | 132 | 116 | 787 |
| TON-IoT red | 1.359 | 290 | 305 | 1.954 |
| TON-IoT telemetria | 875 | 175 | 184 | 1.234 |
| TON-IoT Windows | 201 | 40 | 42 | 283 |
| **Total por etiqueta** | **8.248** | **1.768** | **1.786** | **11.802** |

La distribucion final es 69,88 % / 14,98 % / 15,13 %. La pequena diferencia
entre validacion y test procede del redondeo entero del 70/15/15 original por
clase nativa; no se movieron observaciones despues de congelar las particiones.

## Metricas globales

Estas metricas aplican directamente el umbral 0,5, antes de la abstencion del
agente detector.

| Split | n | Accuracy | Precision ataque | Recall ataque | F1 ataque | Matriz `[[TN, FP], [FN, TP]]` |
|---|---:|---:|---:|---:|---:|---|
| Validacion | 3.536 | 0,962952 | 0,961127 | 0,964932 | 0,963026 | `[[1699, 69], [62, 1706]]` |
| Test | 3.572 | 0,959966 | 0,958682 | 0,961366 | 0,960022 | `[[1712, 74], [69, 1717]]` |

## Metricas operacionales con abstencion

El agente se abstiene cuando la probabilidad pertenece a la zona gris
inclusiva `[0,4, 0,6]`.

| Split | Decididos | Abstenciones | Cobertura | Tasa de abstencion | Precision ataque | Recall ataque | F1 ataque decidido | Matriz decidida `[[TN, FP], [FN, TP]]` |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Validacion | 3.434 | 102 | 0,971154 | 0,028846 | 0,972769 | 0,976731 | 0,974746 | `[[1668, 47], [40, 1679]]` |
| Test | 3.464 | 108 | 0,969765 | 0,030235 | 0,973441 | 0,971198 | 0,972318 | `[[1682, 46], [50, 1686]]` |

## Metricas de test por origen

| Origen | n | F1 ataque antes de abstencion | Cobertura | F1 ataque decidido |
|---|---:|---:|---:|---:|
| Bot-IoT | 146 | 1,000000 | 1,000000 | 1,000000 |
| Edge-IIoTset | 1.770 | 1,000000 | 0,999435 | 1,000000 |
| IoT-23 | 362 | 0,983696 | 0,997238 | 0,983696 |
| TON-IoT Linux | 232 | 0,878261 | 0,935345 | 0,900000 |
| TON-IoT red | 610 | 0,988543 | 0,998361 | 0,988506 |
| TON-IoT telemetria | 368 | 0,742382 | 0,774457 | 0,794118 |
| TON-IoT Windows | 84 | 0,901099 | 0,916667 | 0,941176 |

La estimacion de TON-IoT Windows debe interpretarse con cautela por su soporte
de test reducido. El desglose tambien muestra que agregar TON-IoT ocultaba la
dificultad especifica de la telemetria.

## Reproducibilidad

- Politica: `binary_split_origin_1to1_v1_2026-09-05`.
- Snapshot portable de entradas: `e2117f938b9f8551f567b6f5ca1dcb9710913c760c597f70101ce15f46823666`.
- Seleccion de `manifest_id`: `b97f53b2cea2efbf7174e7a3b49295f130d2f812f9b9f81fcbc99fe31e1246b9`.
- Esquema de 5.264 features: `de41afdd00f73945f34c071b99752d75339b88b7af1cb813439b383cca7c9c1b`.
- Booster XGBoost serializado: `26fa71ff717bc36a9305556095a989a660a5d2f596097b1abe424e385d886463`.
- Modelo: `xgboost_detection_balanced_by_origin_20260905.joblib`.
- SHA-256 del modelo: `f2d7d3dbe9c90fbd7f3d1f134d91e77f855dde134119eced812646e285524393`.

Una reproduccion con el contrato actual obtuvo la misma seleccion, el mismo
esquema, el mismo booster y las mismas metricas. El SHA del contenedor
`joblib` puede cambiar si se anaden campos opcionales al envoltorio Python;
por ello se conserva por separado la huella del booster predictivo.
