"""Fase C de la validacion: estandarizacion EN VIVO de los manifiestos (Mistral).

Disenado para ejecutarse en otra maquina: solo necesita el repo y los
manifiestos de artifacts/validation_2026/manifests/ (las filas crudas van
dentro del manifiesto; no hace falta la carpeta data/).

- 100% LLM en vivo (use_llm=True); si una llamada falla tras los reintentos
  del proveedor, el agente cae al adapter y el resultado queda marcado
  (notes: llm_failed) — se contabiliza como metrica de robustez, y puede
  reintentarse despues con --retry-fallbacks.
- Reanudable: los resultados se anexan a un JSONL por dataset y las filas ya
  procesadas se saltan en la siguiente ejecucion.
- Sin columnas target: el manifiesto ya viene sanitizado y este script lo
  verifica de nuevo antes de cada llamada (defensa en profundidad).

Uso tipico (en la maquina de ejecucion):
    # Basta con guardar MISTRAL_API_KEY en .env. Las variables ya presentes
    # en el proceso tienen precedencia y nunca se imprimen.

    # humo: 5 filas de un dataset
    .venv\\Scripts\\python.exe scripts\\run_live_standardization.py --dataset iot23 --limit 5

    # campana completa (reanudable; ~35k filas)
    .venv\\Scripts\\python.exe scripts\\run_live_standardization.py --workers 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO / "artifacts" / "validation_2026" / "manifests"
OUT_DIR = REPO / "artifacts" / "validation_2026" / "standardized"

TARGET_COLS = {
    "attack_label", "attack_type", "label", "type", "detailed-label",
    "detailed_label", "category", "subcategory", "attack",
}

_local = threading.local()
_write_lock = threading.Lock()
_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_project_env(path: Path | None = None) -> bool:
    """Carga un ``.env`` sencillo sin sobrescribir variables del proceso.

    Se mantiene local al runner para no introducir una dependencia adicional en
    una campana de varios dias. Nunca registra nombres con sus valores.
    """
    env_path = path or REPO / ".env"
    if not env_path.is_file():
        return False

    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not _ENV_KEY.fullmatch(key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)
    return True


def configure_campaign_env() -> None:
    """Aplica defaults seguros para una campana Mistral larga y reanudable."""
    os.environ.setdefault("LLM_PROVIDER", "mistral")
    # Un alias ``latest`` podria cambiar a mitad de una ejecucion de varios dias.
    os.environ.setdefault("INGEST_LLM_MODEL", "mistral-small-2603")
    os.environ.setdefault("MISTRAL_RATE_LIMIT_RETRIES", "8")
    os.environ.setdefault("MISTRAL_RETRY_STATUS_CODES", "429,500,502,503,504")


def request_system_awake(platform_name: str | None = None, setter=None) -> bool:
    """Evita la suspension por inactividad mientras vive el proceso en Windows."""
    if (platform_name or os.name) != "nt":
        return False
    if setter is None:
        import ctypes

        setter = ctypes.windll.kernel32.SetThreadExecutionState
    es_continuous = 0x80000000
    es_system_required = 0x00000001
    return bool(setter(es_continuous | es_system_required))


def release_system_awake(platform_name: str | None = None, setter=None) -> bool:
    """Restaura el comportamiento energetico normal al terminar la campana."""
    if (platform_name or os.name) != "nt":
        return False
    if setter is None:
        import ctypes

        setter = ctypes.windll.kernel32.SetThreadExecutionState
    return bool(setter(0x80000000))


def get_agent():
    """Un IngestParserAgent con LLM por hilo (el cliente HTTP no se comparte)."""
    if getattr(_local, "agent", None) is None:
        from src.orchestration.graph import default_agents
        _local.agent = default_agents(use_llm=True).ingest
    return _local.agent


def standardize_one(entry: dict) -> dict:
    row = entry["row"]
    leaked = [k for k in row if k.strip().lower() in TARGET_COLS]
    if leaked:
        raise RuntimeError(f"target en features del manifiesto: {leaked} ({entry['manifest_id']})")

    raw_input = {
        "dataset": entry["dataset"],
        "row": row,
        "source_file": entry["source_file"],
        "row_id": entry["row_id"],
        "use_llm": True,
    }
    t0 = time.time()
    try:
        event, output = asyncio.run(get_agent().ingest_async(raw_input))
        notes = list(output.notes or [])
        return {
            "manifest_id": entry["manifest_id"],
            "ok": True,
            "parsed_by_llm": "parsed_by_llm" in notes,
            "provider": os.getenv("LLM_PROVIDER", "mistral"),
            "model": os.getenv("INGEST_LLM_MODEL", "mistral-small-2603"),
            "notes": notes,
            "mapping_confidence": output.mapping_confidence,
            "latency_s": round(time.time() - t0, 2),
            "canonical_event": event.model_dump(mode="json"),
        }
    except Exception as exc:
        return {
            "manifest_id": entry["manifest_id"],
            "ok": False,
            "parsed_by_llm": False,
            "provider": os.getenv("LLM_PROVIDER", "mistral"),
            "model": os.getenv("INGEST_LLM_MODEL", "mistral-small-2603"),
            "error": f"{type(exc).__name__}: {exc}",
            "latency_s": round(time.time() - t0, 2),
        }


def load_done(out_path: Path, retry_fallbacks: bool) -> set[str]:
    done: set[str] = set()
    if not out_path.exists():
        return done
    with open(out_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if retry_fallbacks and r.get("ok") and not r.get("parsed_by_llm"):
                continue  # se reintenta
            if r.get("ok") or not retry_fallbacks:
                done.add(r["manifest_id"])
    return done


def iter_bounded_results(entries: list[dict], workers: int):
    """Procesa con como maximo ``workers`` llamadas en vuelo.

    No se encolan decenas de miles de llamadas: asi un cortacircuitos o una
    interrupcion detienen el gasto despues de las solicitudes ya activas.
    """
    iterator = iter(entries)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        for _ in range(workers):
            try:
                entry = next(iterator)
            except StopIteration:
                break
            pending[pool.submit(standardize_one, entry)] = entry

        while pending:
            future = next(as_completed(pending))
            pending.pop(future)
            yield future.result()
            try:
                entry = next(iterator)
            except StopIteration:
                continue
            pending[pool.submit(standardize_one, entry)] = entry


def run_dataset(
    name: str,
    workers: int,
    limit: int | None,
    retry_fallbacks: bool,
    max_consecutive_failures: int = 20,
) -> dict:
    manifest_path = MANIFEST_DIR / f"{name}_manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"No existe {manifest_path}; ejecuta antes build_validation_manifests.py")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{name}_standardized.jsonl"
    done = load_done(out_path, retry_fallbacks)

    entries = []
    with open(manifest_path, encoding="utf-8") as fh:
        for line in fh:
            entry = json.loads(line)
            if entry["manifest_id"] not in done:
                entries.append(entry)
    if limit:
        entries = entries[:limit]

    total = len(entries)
    print(f"[{name}] pendientes {total} (ya procesadas {len(done)})")
    if not total:
        return {"dataset": name, "processed": 0}

    stats = Counter()
    confidences: list[float] = []
    consecutive_failures = 0
    t_start = time.time()
    with open(out_path, "a", encoding="utf-8") as out_fh:
        for i, result in enumerate(iter_bounded_results(entries, workers), 1):
            with _write_lock:
                out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_fh.flush()
            if result.get("ok") and result.get("parsed_by_llm"):
                stats["llm_ok"] += 1
                confidences.append(float(result.get("mapping_confidence") or 0.0))
                consecutive_failures = 0
            elif result.get("ok"):
                stats["fallback_adapter"] += 1
                consecutive_failures += 1
            else:
                stats["error"] += 1
                consecutive_failures += 1
            if i % 50 == 0 or i == total:
                rate = i / max(1e-9, time.time() - t_start)
                eta_min = (total - i) / max(rate, 1e-9) / 60
                print(f"  [{name}] {i}/{total} | llm_ok={stats['llm_ok']} "
                      f"fallback={stats['fallback_adapter']} err={stats['error']} "
                      f"| {rate:.1f} filas/s | ETA {eta_min:.0f} min", flush=True)
            if consecutive_failures >= max_consecutive_failures:
                raise RuntimeError(
                    f"Cortacircuitos: {consecutive_failures} filas consecutivas sin salida LLM; "
                    "se conserva el progreso y se detiene la campana."
                )

    summary = {
        "dataset": name,
        "provider": os.getenv("LLM_PROVIDER", "mistral"),
        "model": os.getenv("INGEST_LLM_MODEL", "mistral-small-2603"),
        "processed": total,
        "llm_ok": stats["llm_ok"],
        "fallback_adapter": stats["fallback_adapter"],
        "errors": stats["error"],
        "avg_mapping_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "elapsed_min": round((time.time() - t_start) / 60, 1),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with _write_lock, open(OUT_DIR / f"{name}_run_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append",
                        help="repetible; por defecto todos los manifiestos presentes")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="procesar solo N filas (prueba de humo)")
    parser.add_argument("--retry-fallbacks", action="store_true",
                        help="reintentar filas que quedaron en fallback de adapter")
    parser.add_argument("--max-consecutive-failures", type=int, default=20,
                        help="detener la campana tras N resultados seguidos sin salida LLM")
    args = parser.parse_args()

    load_project_env()
    configure_campaign_env()
    if not os.getenv("MISTRAL_API_KEY"):
        raise SystemExit("Falta MISTRAL_API_KEY en el entorno o en .env")
    if (os.getenv("LLM_PROVIDER") or "").lower() != "mistral":
        raise SystemExit("Define LLM_PROVIDER=mistral")
    if not os.getenv("INGEST_LLM_MODEL"):
        raise SystemExit("Define INGEST_LLM_MODEL (p. ej. mistral-small-2603); "
                         "sin el, el default gemma3:12b provoca 400 en Mistral")

    datasets = args.dataset or sorted(
        p.name.replace("_manifest.jsonl", "")
        for p in MANIFEST_DIR.glob("*_manifest.jsonl")
    )
    awake = request_system_awake()
    if os.name == "nt" and not awake:
        print("AVISO: Windows no acepto la solicitud para impedir la suspension.", flush=True)
    try:
        print(f"Campana de estandarizacion en vivo | datasets={datasets} | workers={args.workers}")
        for name in datasets:
            run_dataset(
                name,
                args.workers,
                args.limit,
                args.retry_fallbacks,
                args.max_consecutive_failures,
            )
    finally:
        if awake:
            release_system_awake()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
