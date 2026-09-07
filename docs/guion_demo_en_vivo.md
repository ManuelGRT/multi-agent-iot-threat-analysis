# Guion de demostración online

**Objetivo:** enseñar el flujo completo con una entrada cruda, Mistral,
modelos empaquetados, mitigación anclada, persistencia y auditoría posterior.

**Duración estimada:** 12–15 minutos.

## 0. Preparación

La entrada de la demo debe estar libre de targets. La sanitización se realiza
al preparar datasets de entrenamiento o evaluación, nunca durante la petición.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install ".[test]"

$env:MISTRAL_API_KEY="..."
$env:INGEST_LLM_TIMEOUT_SECONDS="60"
$env:TFM_STATE_DIR=".runtime-state"
```

No mostrar ni guardar la clave en capturas, terminales compartidos o archivos
versionados. Antes de la defensa, comprobar:

```powershell
python -m pytest -q
```

## 1. Presentar la arquitectura

Explicación breve:

> «La API admite un registro o texto crudo limpio. El estandarizador busca un
> éxito Mistral para ese contenido y, si no existe, Mistral selecciona las
> columnas y construye el evento canónico. Después actúan detector,
> clasificador, mitigador y juez. El caso se persiste y un auditor independiente
> puede comprobarlo posteriormente.»

Señalar que el runtime contiene tres servidores MCP:

- `inference`: estandarización y modelos XGBoost;
- `case_memory`: casos y trazas SQLite;
- `threat_intel`: catálogo ATT&CK/CAPEC y mitigaciones.

No hay adaptadores, agentes alternativos ni un endpoint para introducir
`canonical_event`.

## 2. Arrancar API y frontend

```powershell
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Abrir `http://localhost:8000` a pantalla completa. Como alternativa de
despliegue, puede mostrarse la misma aplicación con:

```powershell
docker compose up --build
```

La imagen ya incluye el frontend, el catálogo y los dos modelos activos. El
volumen de estado conserva caché y casos; `MISTRAL_API_KEY` se inyecta por
entorno o mediante un `.env` local no versionado.

## 3. Analizar un registro crudo

1. Seleccionar un ejemplo del visor, preferiblemente **IoT-23 · botnet C&C**.
2. Mostrar que el JSON contiene `row` y no contiene targets ni
   `canonical_event`.
3. Pulsar **Analizar caso**.
4. Narrar la secuencia:
   Estandarizador → Detector → Clasificador → Mitigador → Juez.

Qué señalar:

- El estandarizador informa fuente, modelo, perfil y confianza. En un primer
  *cache miss*, Mistral realiza la selección de columnas y la extracción. Si el
  contenido ya tuvo un éxito, la caché evita repetir esas llamadas y reconstruye
  la identidad del caso actual.
- El detector muestra probabilidad y la zona gris `[0.4, 0.6]`; dentro de ella
  se abstiene y el juez solicita revisión humana.
- El clasificador muestra el tipo, su familia agregada, la confianza y la
  distribución top-3. Una confianza inferior a `0.65` también deriva el caso.
- El mitigador muestra primero la contextualización de Mistral y mantiene
  visibles las recomendaciones y referencias del catálogo. Los elementos
  `llm_suggested` aparecen marcados como no respaldados.
- El juez aprueba el caso o produce `human_interrupt`; esta última salida es
  una degradación controlada, no una clasificación maliciosa forzada.

## 4. Distinguir Mistral y catálogo en mitigación

Usar una recomendación mostrada en pantalla para explicar:

- **Catálogo:** aporta el identificador ATT&CK/CAPEC, la medida verificable y
  su URL.
- **Mistral:** redacta el contexto específico del evento y puede reformular la
  acción sin convertirse en fuente de autoridad.

Si Mistral falla en esta etapa, se conserva el catálogo y el juez todavía puede
cerrar el caso. Este *fallback* pertenece solo al mitigador. Si falla Mistral
durante un *cache miss* de estandarización, no existe un sustituto: el
estandarizador se abstiene y el juez solicita revisión humana.

## 5. Mostrar persistencia y auditoría

Al final del visor:

- mostrar el `case_id`, el estado y la traza completa;
- desplegar el `CaseResult` si se necesita justificar una decisión;
- abrir **Auditoría posterior independiente**;
- señalar el veredicto `approve`, `review` o `reject` y la descripción breve de
  cada comprobación.

La misma auditoría puede consultarse por API:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/cases/<case_id>/audit
```

## 6. Repetir para demostrar la caché

Enviar exactamente el mismo contenido una segunda vez. La traza del
estandarizador debe indicar un *cache hit*. Aclarar que:

- solo se reutilizan éxitos Mistral validados;
- la identidad y procedencia se reconstruyen para el caso nuevo;
- la caché no selecciona columnas mediante reglas deterministas;
- los fallos nunca quedan cacheados como estandarizaciones válidas.

## 7. Superficie pública

Mostrar, si surge la pregunta, que la API solo publica:

- `POST /cases/analyze` con `row` o `text`;
- `GET /cases/{case_id}/audit`;
- `GET /health`.

Las rutas antiguas de eventos, datasets y adaptadores no están registradas y
responden `404`. El endpoint vigente rechaza `canonical_event` con `422`, de
modo que tampoco se puede omitir Mistral enviando un evento canónico.

## 8. Riesgos durante la demostración

| Riesgo | Comportamiento esperado | Explicación |
|---|---|---|
| Mistral no responde en un *cache miss* | Abstención y `human_interrupt` | El sistema falla de forma cerrada; no usa un adaptador ni inventa un evento |
| El contenido ya está cacheado | El flujo continúa sin nueva llamada | Se reutiliza un éxito Mistral para el mismo hash exacto |
| Mistral falla solo en mitigación | Se muestran medidas del catálogo | Es el único fallback operacional |
| Rate limit | Error controlado o reintento configurado | Esperar y repetir; no cambiar la entrada para ocultarlo |
| Confianza del detector o clasificador baja | Revisión humana | Es una decisión prevista por los umbrales |

No se presenta una supuesta demo offline autocontenida: la funcionalidad
online requiere Mistral en todo *cache miss*. Si no hay red, puede demostrarse
la degradación controlada o utilizar contenido cuyo éxito ya exista en la caché
persistente, explicándolo explícitamente.

## 9. Cierre

> «El valor del sistema no es solo la predicción. Cada entrada queda
> estandarizada bajo un contrato, pasa por agentes especializados, recibe
> mitigaciones con procedencia y termina en una decisión trazable, persistida y
> auditable. Los componentes experimentales permanecen fuera del runtime.»

Checklist final:

- [ ] `MISTRAL_API_KEY` disponible solo en el entorno;
- [ ] suite en verde;
- [ ] API o contenedor arrancado en `http://localhost:8000`;
- [ ] directorio de estado escribible y persistente;
- [ ] ejemplos revisados para confirmar que no contienen targets;
- [ ] al menos un caso cacheado para demostrar la reutilización de duplicados.
