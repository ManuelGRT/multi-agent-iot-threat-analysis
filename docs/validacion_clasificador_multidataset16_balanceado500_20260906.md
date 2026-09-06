# Clasificador multidataset de 16 tipos de ataque balanceado a 500 casos por clase

## Estado y objetivo

Este documento describe la campaña `classifier_multidataset16_balanced500_20260906`.
Sustituye como resultado principal a la campaña anterior de 480 casos por
clase. El cambio metodológico es importante: primero se construye el corpus
balanceado usando todos los ataques mapeables y, solo después, se crea una
nueva partición 70/15/15. No se condiciona la cuota a la distribución de las
particiones de una campaña anterior.

La taxonomía conserva los catorce ataques de Edge-IIoTset y añade `DoS` y
`Command_and_Control`. La clase normal no pertenece a este clasificador, ya
que sigue siendo responsabilidad del detector binario. Las dieciséis clases
son:

`Backdoor`, `Command_and_Control`, `DDoS_HTTP`, `DDoS_ICMP`, `DDoS_TCP`,
`DDoS_UDP`, `DoS`, `Fingerprinting`, `MITM`, `Password`, `Port_Scanning`,
`Ransomware`, `SQL_injection`, `Uploading`, `Vulnerability_scanner` y `XSS`.

## Cobertura disponible y determinación de la cuota

La fuente de partida contiene 35.637 resultados de estandarización ya
materializados. Se identificaron 13.478 ataques utilizables por la taxonomía:
11.673 con mapeo exacto o agregado desde una etiqueta nativa y 1.805 con un
mapeo forzado documentado. No se eliminó ninguna fila adicional por
deduplicación.

| Clase | Ataques disponibles |
|---|---:|
| `Backdoor` | 1.000 |
| `Command_and_Control` | 537 |
| `DDoS_HTTP` | 517 |
| `DDoS_ICMP` | **500** |
| `DDoS_TCP` | 1.103 |
| `DDoS_UDP` | 805 |
| `DoS` | 1.000 |
| `Fingerprinting` | 620 |
| `MITM` | 900 |
| `Password` | 1.000 |
| `Port_Scanning` | 1.496 |
| `Ransomware` | 1.000 |
| `SQL_injection` | 1.000 |
| `Uploading` | **500** |
| `Vulnerability_scanner` | **500** |
| `XSS` | 1.000 |

El mínimo global es 500. Por tanto, las clases limitantes son
`DDoS_ICMP`, `Uploading` y `Vulnerability_scanner`, no
`Command_and_Control`, para la que existen 537 casos. La cuota se fija en 500
observaciones por clase y el corpus final contiene 8.000 filas. El muestreo se
realiza sin reemplazo.

Permanecen fuera del espacio supervisado 382 ataques: 300 `DDoS` de TON-IoT
sin protocolo o servicio crudo verificable, 79 `Theft` de Bot-IoT y 3
`FileDownload` independientes de IoT-23. Las variantes C&C de IoT-23 sí se
agregan en `Command_and_Control`.

## Selección proporcional del corpus

Para cada clase, los 500 casos se distribuyen entre los datasets en proporción
a la disponibilidad real. Se usa el reparto de Hamilton: se calcula la cuota
ideal de cada origen, se toma su parte entera y los casos restantes se asignan
por los mayores residuos. El mismo criterio se aplica de forma jerárquica a
los suborígenes y estados de mapeo, siempre sin superar la disponibilidad y sin
reemplazo.

La matriz seleccionada de clase por dataset es:

| Clase | Bot-IoT | Edge-IIoTset | IoT-23 | TON-IoT | Total |
|---|---:|---:|---:|---:|---:|
| `Backdoor` | 0 | 250 | 0 | 250 | 500 |
| `Command_and_Control` | 0 | 0 | 500 | 0 | 500 |
| `DDoS_HTTP` | 2 | 484 | 0 | 14 | 500 |
| `DDoS_ICMP` | 0 | 500 | 0 | 0 | 500 |
| `DDoS_TCP` | 112 | 227 | 101 | 60 | 500 |
| `DDoS_UDP` | 156 | 311 | 1 | 32 | 500 |
| `DoS` | 250 | 0 | 0 | 250 | 500 |
| `Fingerprinting` | 97 | 403 | 0 | 0 | 500 |
| `MITM` | 0 | 222 | 0 | 278 | 500 |
| `Password` | 0 | 250 | 0 | 250 | 500 |
| `Port_Scanning` | 127 | 167 | 39 | 167 | 500 |
| `Ransomware` | 0 | 250 | 0 | 250 | 500 |
| `SQL_injection` | 0 | 250 | 0 | 250 | 500 |
| `Uploading` | 0 | 500 | 0 | 0 | 500 |
| `Vulnerability_scanner` | 0 | 500 | 0 | 0 | 500 |
| `XSS` | 0 | 250 | 0 | 250 | 500 |
| **Total** | **744** | **4.564** | **641** | **2.051** | **8.000** |

La selección conserva 7.248 targets exactos y 752 forzados. Como ejemplos del
reparto proporcional, `Backdoor` se divide 250/250 entre Edge-IIoTset y
TON-IoT; los 517 casos disponibles de `DDoS_HTTP` se convierten en 2/484/14
casos de Bot-IoT, Edge-IIoTset y TON-IoT; y los 537 casos de
`Command_and_Control` se reducen a 500, todos procedentes de IoT-23.

## Partición posterior 70/15/15

La partición se realiza **después** de seleccionar las 8.000 filas. No se
reutiliza la pertenencia `train`/`val`/`test` de la campaña fuente. El nuevo
clasificador se entrena desde cero con esta nueva asignación.

La selección y la asignación se calculan únicamente a partir de etiquetas,
procedencia, identificadores y semilla; no consultan las predicciones ni las
métricas. Una vez construida la nueva partición, esta queda congelada antes de
ajustar el modelo y de seleccionar el umbral.

| Partición | Casos por clase | Casos totales |
|---|---:|---:|
| Entrenamiento | 350 | 5.600 |
| Validación | 75 | 1.200 |
| Prueba | 75 | 1.200 |
| **Total** | **500** | **8.000** |

Los totales 350/75/75 son exactos para cada una de las dieciséis clases. Para
preservar también la composición interna, el reparto entero se efectúa con la
siguiente jerarquía:

1. clase de ataque;
2. dataset de origen;
3. origen detallado (`ton_iot_linux`, `ton_iot_network`,
   `ton_iot_telemetry` o `ton_iot_windows` en TON-IoT);
4. estado de mapeo (`exact` o `forced`).

En cada celda de tamaño \(n\), los objetivos ideales son \(0,70n\),
\(0,15n\) y \(0,15n\). Las asignaciones enteras son el suelo o el techo de
esas cantidades, quedan restringidas por los totales exactos del nivel padre
y los empates se resuelven de forma determinista mediante SHA-256 con semilla
42. Así, un estrato de 100 filas se reparte exactamente 70/15/15. En estratos
muy pequeños no siempre es posible representar las tres particiones; por
ejemplo, dos filas no pueden dividirse de forma no vacía entre tres conjuntos.

Por ejemplo, los 500 casos de `Command_and_Control` de IoT-23 se reparten
350/75/75. En `Backdoor`, las 250 filas de Edge-IIoTset se dividen en
175/37/38 y las 250 de TON-IoT en 175/38/37, manteniendo a la vez el total
exacto 350/75/75 de la clase. Dentro de estas últimas, TON network aporta 128
filas (90/19/19), telemetría 113 (79/17/17) y Windows 9 (6/2/1). Este ejemplo
muestra cómo los residuos enteros se compensan entre estratos sin alterar la
cuota global de la clase.

El resultado agregado por dataset es:

| Dataset | Seleccionado | Train | Val | Test |
|---|---:|---:|---:|---:|
| Bot-IoT | 744 | 520 | 112 | 112 |
| Edge-IIoTset | 4.564 | 3.196 | 684 | 684 |
| IoT-23 | 641 | 449 | 96 | 96 |
| TON-IoT | 2.051 | 1.435 | 308 | 308 |
| **Total** | **8.000** | **5.600** | **1.200** | **1.200** |

La preservación de los cuatro suborígenes TON queda auditada así:

| Suborigen TON-IoT | Seleccionado | Train | Val | Test |
|---|---:|---:|---:|---:|
| Linux | 284 | 200 | 42 | 42 |
| Network | 1.294 | 905 | 194 | 195 |
| Telemetría | 404 | 283 | 60 | 61 |
| Windows | 69 | 47 | 12 | 10 |
| **Total** | **2.051** | **1.435** | **308** | **308** |

El mismo control se aplica al estado del target:

| Estado de mapeo | Seleccionado | Train | Val | Test |
|---|---:|---:|---:|---:|
| Exacto | 7.248 | 5.073 | 1.087 | 1.088 |
| Forzado | 752 | 527 | 113 | 112 |
| **Total** | **8.000** | **5.600** | **1.200** | **1.200** |

Las pequeñas diferencias de una fila entre validación y prueba son el efecto
esperado del redondeo entero. No existe solapamiento de huellas entre las
particiones finales.

## Modelo y selección del umbral

Se entrenó un XGBoost multiclase de 300 árboles, profundidad máxima 6, tasa
de aprendizaje 0,1, `subsample=0.8`, `colsample_bytree=0.8` y objetivo
`multi:softprob`. Se utilizó `tree_method=hist`, `eval_metric=mlogloss` y un
único hilo de entrenamiento (`n_jobs=1`). El vectorizador generó 2.742
características. Tanto el muestreo como el entrenamiento usan la semilla 42.

El umbral de confianza se eligió exclusivamente sobre validación. El valor
seleccionado fue 0,80: cobertura 0,8458, riesgo selectivo 0,0493 y 1.015
decisiones sobre 1.200 casos. La accuracy forzada de validación fue 0,8800, la
macro-F1 0,8792 y la top-3 accuracy 0,9750.

## Resultados globales en prueba

| Métrica | Resultado |
|---|---:|
| Casos | 1.200 |
| Accuracy | 0,8908 |
| Precisión macro | 0,8944 |
| Recall macro | 0,8908 |
| Macro-F1 | 0,8917 |
| Precisión ponderada | 0,8944 |
| Recall ponderado | 0,8908 |
| F1 ponderada | 0,8917 |
| Top-3 accuracy | 0,9800 |
| Cobertura con umbral 0,80 | 0,8383 |
| Casos decididos | 1.006 |
| Casos con abstención | 194 |
| Accuracy entre decisiones | 0,9622 |
| Macro-F1 entre decisiones | 0,9507 |
| Riesgo selectivo | 0,0378 |

El desglose por clase se calcula sobre 75 casos de prueba por etiqueta:

| Clase | Soporte | Precisión | Recall | F1 |
|---|---:|---:|---:|---:|
| `Backdoor` | 75 | 0,9306 | 0,8933 | 0,9116 |
| `Command_and_Control` | 75 | 1,0000 | 1,0000 | 1,0000 |
| `DDoS_HTTP` | 75 | 0,8056 | 0,7733 | 0,7891 |
| `DDoS_ICMP` | 75 | 1,0000 | 1,0000 | 1,0000 |
| `DDoS_TCP` | 75 | 0,9868 | 1,0000 | 0,9934 |
| `DDoS_UDP` | 75 | 0,9867 | 0,9867 | 0,9867 |
| `DoS` | 75 | 0,9726 | 0,9467 | 0,9595 |
| `Fingerprinting` | 75 | 0,9867 | 0,9867 | 0,9867 |
| `MITM` | 75 | 0,9583 | 0,9200 | 0,9388 |
| `Password` | 75 | 0,6176 | 0,5600 | 0,5874 |
| `Port_Scanning` | 75 | 0,9865 | 0,9733 | 0,9799 |
| `Ransomware` | 75 | 0,8933 | 0,8933 | 0,8933 |
| `SQL_injection` | 75 | 0,6250 | 0,7333 | 0,6748 |
| `Uploading` | 75 | 0,8806 | 0,7867 | 0,8310 |
| `Vulnerability_scanner` | 75 | 0,9730 | 0,9600 | 0,9664 |
| `XSS` | 75 | 0,7079 | 0,8400 | 0,7683 |

Como comprobación de la calidad del target, los 1.088 casos exactos de test
alcanzan accuracy 0,8915, macro-F1 0,8883 y top-3 accuracy 0,9835. Los 112
casos con target forzado alcanzan accuracy 0,8839, F1 ponderada 0,9280 y top-3
accuracy 0,9464. La macro-F1 de la vista forzada es 0,3744, pero no debe
compararse directamente con la global: esa vista solo contiene parte de las
clases y su espacio métrico incorpora las clases externas que el modelo haya
predicho.

## Evaluación por origen

Estas evaluaciones no son LODO ni miden transferencia a un dataset nunca
observado. El modelo global ha sido entrenado con las filas `train` de todos
los orígenes. Las vistas siguientes solo reorganizan filas que no participaron
en el desarrollo del modelo para medir su rendimiento dentro de cada fuente.

La primera vista parte exclusivamente de las 1.200 filas del nuevo test. Para
cada dataset conserva las clases con al menos diez casos y las balancea a la
clase elegible menos representada:

| Dataset | Pool de test | Clases evaluadas | Cuota por clase | Filas | Accuracy | Macro-F1 | Top-3 | Cobertura | Riesgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Bot-IoT | 112 | 5 | 15 | 75 | 0,9867 | 0,9867 | 0,9867 | 0,9733 | 0,0000 |
| Edge-IIoTset | 684 | 14 | 25 | 350 | 0,9029 | 0,9005 | 0,9971 | 0,8400 | 0,0340 |
| IoT-23 | 96 | 2 | 15 | 30 | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 0,0000 |
| TON-IoT | 308 | 8 | 25 | 200 | 0,8250 | 0,8274 | 0,9350 | 0,7300 | 0,0548 |

La segunda vista responde a la evaluación ampliada solicitada. Para cada
origen se construye:

\[
E_d = T^{\mathrm{test}}_{d,\,\mathrm{seleccionado}}
      \;\dot\cup\;
      U_{d,\,\mathrm{fuera\ del\ corpus}},
\]

donde el primer término contiene el test del corpus de 8.000 filas y el
segundo todos los ataques mapeados de ese origen que no fueron seleccionados
por el balanceo. Se excluyen expresamente las filas de entrenamiento y
validación del nuevo clasificador. El pool ampliado suma 6.678 observaciones:
1.200 de test y 5.478 no seleccionadas. Su solapamiento con desarrollo es
cero. Después se vuelve a exigir un mínimo de diez casos por clase y se
balancea cada dataset de forma independiente.

| Dataset | Pool ampliado | Clases evaluadas | Cuota por clase | Filas | Accuracy | Macro-F1 | Top-3 | Cobertura | Riesgo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Bot-IoT | 868 | 5 | 38 | 190 | 0,9947 | 0,9947 | 0,9947 | 0,9789 | 0,0000 |
| Edge-IIoTset | 3.020 | 14 | 75 | 1.050 | 0,8905 | 0,8865 | 0,9971 | 0,8419 | 0,0362 |
| IoT-23 | 333 | 3 | 83 | 249 | 0,9920 | 0,7455 | 0,9960 | 0,9598 | 0,0000 |
| TON-IoT | 2.457 | 10 | 25 | 250 | 0,8600 | 0,8596 | 0,9440 | 0,7760 | 0,0567 |

En IoT-23 hubo una predicción `DDoS_UDP` fuera de las tres clases presentes.
El cálculo macro incorpora esa etiqueta predicha con soporte real cero; por
eso la macro-F1 (0,7455) es mucho menor que la accuracy (0,9920). El artefacto
conserva además vistas de auditoría con umbral mínimo de una fila, pero no se
usan como comparación principal por su inestabilidad.

## Reproducibilidad y despliegue

- Semilla: 42.
- Hash de taxonomía: `d016e777a545c78361adb1c05352a70f36e0022c5f85e5aaf4f8e379655bd6ed`.
- Hash de selección: `d39cf28b3ec96e1558fbc77327d5674a43644a5f880233bd59640e8370168f60`.
- Hash de las 8.000 filas seleccionadas que entran al entrenador:
  `f4b6aa576b7c7b48b34023232e0abce9368c50f52fde9f3e40dff6a19c38e344`.
- Hash de train: `c53a855cbb787b8f34499fd3e61b6dbae3ccf4b2cf0c0c0a30dbc76b9dbe69fc`.
- Hash de validación: `bd62dd524b7804fe24d89b22ce57ee5d69b89717f319d48cb8161cef7826fefa`.
- Hash de test: `644ce38284d1853b38691eb12aba42f58270ec4217ed074357fec14a2268320c`.
- Hash de los nombres de características: `7562752da27242801d14455990f38e7b1e32771a35f96196fb499b52c0a289cd`.
- Hash del modelo candidato: `f49b2a50920b3fa260a999f34696e86a92068786f386bae866b02a5faaa7b185`.
- Llamadas de red durante la campaña: 0.
- Una segunda ejecución completa con la misma semilla reprodujo exactamente
  los hashes de selección y de modelo, el umbral y todas las métricas de test.

El entrenamiento reutiliza exclusivamente respuestas de Mistral ya
materializadas; no llama al LLM ni reconstruye determinísticamente la
estandarización. El modelo generado ocupa 2.018.126 bytes. Tras revisar sus
métricas, su contrato y su reproducción determinista, el artefacto con hash
`f49b2a50920b3fa260a999f34696e86a92068786f386bae866b02a5faaa7b185`
se promocionó como clasificador predeterminado del paquete. La variable
`TFM_FAMILY_MODEL` conserva la posibilidad de seleccionar explícitamente otro
artefacto compatible.

Los artefactos completos se encuentran bajo
`artifacts/validation_2026/classifier_multidataset16_balanced500_20260906/`:
informe JSON, resumen, selección global, sidecar de targets y modelo candidato.
El runner reproducible es
`scripts/train_balanced_multidataset16_classifier.py`.
