# Instrucciones: campaña de estandarización en vivo (otro ordenador)

**Qué es:** ejecución reproducible de la fase de estandarización en vivo de la
campaña de validación. Estandariza con Mistral EN VIVO las 35.637 filas de los
manifiestos de
`artifacts/validation_2026/manifests/` (5 datasets). Reanudable: se puede cortar
y relanzar sin perder trabajo.

| Dataset | Filas | Nota |
|---|---:|---|
| edge_iiotset | 12.800 | binario 5.900+5.900 exacto; MITM al máximo único real (400) |
| ton_iot | 11.800 | muestra general balanceada (10 clases, ~655/clase) |
| iot23 | 7.081 | ataques = todo el universo único disponible (1.181) |
| bot_iot | 2.956 | Normal solo 477 en origen (declarado) |
| urban_iot | 1.000 | sin etiquetas: solo validación de estandarización |

Cada fila tabular requiere dos llamadas (selección de columnas y extracción).
La duración y el coste dependen de la cuota y tarifa vigentes del proveedor.
Este runner fuerza `cache_mode=bypass`: la caché operativa del sistema no
interviene y cada resultado aceptado corresponde a una llamada en vivo.

## 1. Qué copiar a la máquina

- El repositorio completo (o `git clone` del repo local + copiar lo no versionado):
  - `artifacts/validation_2026/manifests/` (los 5 `*_manifest.jsonl` + `summary.json`)
    — **imprescindible**: no está en git y contiene las filas a estandarizar.
    Alternativa: si la máquina tiene la carpeta `data/` completa, se pueden
    regenerar idénticos con `python scripts/build_validation_manifests.py`
    (semilla 42 determinista).
- NO hace falta la carpeta `data/` (las filas van dentro de los manifiestos).
- NO hace falta la caché Mistral histórica.

## 2. Preparar el entorno

```powershell
cd <repo>
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

Variables de entorno (misma terminal donde se lance):

```powershell
$env:MISTRAL_API_KEY = "<clave>"
$env:LLM_PROVIDER = "mistral"
$env:INGEST_LLM_MODEL = "mistral-small-2603"    # opcional; modelo fijado por defecto
# opcionales de robustez:
$env:MISTRAL_RATE_LIMIT_RETRIES = "5"
$env:MISTRAL_RETRY_WAIT_SECONDS = "3"
```

## 3. Prueba de humo (obligatoria antes de la campaña)

```powershell
.\.venv\Scripts\python.exe scripts\run_live_standardization.py --dataset iot23 --limit 5
```

Esperado: `llm_ok=5`, salvo que alguna fila quede como abstención reintentable,
y el fichero
`artifacts/validation_2026/standardized/iot23_standardized.jsonl` con 5 líneas
donde todo resultado aceptado tenga `"parsed_by_llm": true`. Si aparecen
abstenciones, revisa las variables de entorno y el campo `error`.

## 4. Campaña completa

```powershell
.\.venv\Scripts\python.exe scripts\run_live_standardization.py --workers 8
```

- Progreso cada 50 filas con ETA por dataset.
- **Reanudación:** si se corta (red, cuota, reinicio), relanzar la misma orden;
  las filas ya procesadas se saltan.
- Si Mistral devuelve muchos 429, bajar a `--workers 4`.
- Al final, reintentar las filas que hayan quedado en abstención o error:

```powershell
.\.venv\Scripts\python.exe scripts\run_live_standardization.py --retry-failures --workers 4
```

## 5. Qué devolver

La carpeta `artifacts/validation_2026/standardized/` completa:
- `<dataset>_standardized.jsonl` (5 ficheros)
- `<dataset>_run_summary.json` (5 ficheros)

La campaña de estandarización termina al verificar la cobertura LLM completa.
El entrenamiento y la evaluación de los modelos vigentes se ejecutan después,
en el equipo principal, mediante sus scripts específicos.

## Criterios de calidad de la campaña

- 100 % de los resultados aceptados con `parsed_by_llm: true`; no se admiten
  resultados de adaptador. Las abstenciones se reintentan con `--retry-failures`.
- `avg_mapping_confidence` ≥ 0,8 en los datasets conocidos; urban_iot puede ser
  menor (fuente desconocida — ese dato ES el resultado).
- 0 errores de tipo «target en features» (el runner aborta si detecta etiquetas).
