"""Supervisa la estandarizacion LLM de la campana de forma reanudable.

El supervisor usa los manifiestos como universo autoritativo y los JSONL de
estandarizacion para decidir si todas las filas tienen, al menos, un intento
LLM correcto. El entrenamiento y la evaluacion de los modelos actuales se
ejecutan con sus runners especificos y no forman parte de este supervisor.

Flujo:

1. Opcionalmente espera a que termine un PID ya existente.
2. Ejecuta ``run_live_standardization.py`` en rondas reanudables.
3. Tolera un numero acotado de rondas consecutivas sin progreso, con backoff
   interrumpible, y detiene la campana si se agota ese margen o el numero
   maximo de rondas.
4. Con cobertura LLM del 100 % registra el cierre de la estandarizacion.

La credencial Mistral solo se exige si quedan filas por estandarizar y solo se
entrega al subproceso de Fase C. La credencial nunca se pasa como argumento,
se guarda en un checkpoint ni se imprime.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_DIR = REPO / "artifacts" / "validation_2026" / "manifests"
DEFAULT_RESULTS_DIR = REPO / "artifacts" / "validation_2026" / "standardized"
DEFAULT_STATE_DIR = REPO / "artifacts" / "validation_2026" / "campaign_supervisor"
DEFAULT_PROVIDER = "mistral"
DEFAULT_MODEL = "mistral-small-2603"

_MANIFEST_SUFFIX = "_manifest.jsonl"
_RESULT_SUFFIX = "_standardized.jsonl"
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


class CampaignError(RuntimeError):
    """Error controlado de configuracion o ejecucion de la campana."""


class NoProgressError(CampaignError):
    """Una ronda completa no consiguio ningun nuevo resultado LLM."""


class MaxRoundsExceeded(CampaignError):
    """La campana sigue incompleta despues del maximo de rondas."""


class MissingCredentialError(CampaignError):
    """Falta la credencial necesaria para una Fase C todavia incompleta."""


@dataclass(frozen=True)
class ManifestInventory:
    """IDs esperados, agrupados por el nombre estable de cada dataset."""

    dataset_ids: Mapping[str, frozenset[str]]

    @property
    def total(self) -> int:
        return sum(len(ids) for ids in self.dataset_ids.values())


@dataclass(frozen=True)
class DatasetStatus:
    dataset: str
    total: int
    parsed_by_llm: int
    fallback: int
    error: int
    invalid: int
    missing: int

    @property
    def complete(self) -> bool:
        return self.total == self.parsed_by_llm

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "total": self.total,
            "parsed_by_llm": self.parsed_by_llm,
            "fallback": self.fallback,
            "error": self.error,
            "invalid": self.invalid,
            "missing": self.missing,
            "complete": self.complete,
        }


@dataclass(frozen=True)
class CampaignStatus:
    datasets: tuple[DatasetStatus, ...]
    malformed_result_lines: int = 0
    orphan_result_records: int = 0
    invalid_result_records: int = 0

    @property
    def total(self) -> int:
        return sum(item.total for item in self.datasets)

    @property
    def parsed_by_llm(self) -> int:
        return sum(item.parsed_by_llm for item in self.datasets)

    @property
    def fallback(self) -> int:
        return sum(item.fallback for item in self.datasets)

    @property
    def error(self) -> int:
        return sum(item.error for item in self.datasets)

    @property
    def missing(self) -> int:
        return sum(item.missing for item in self.datasets)

    @property
    def invalid(self) -> int:
        return sum(item.invalid for item in self.datasets)

    @property
    def complete(self) -> bool:
        return (
            self.total > 0
            and self.parsed_by_llm == self.total
            and self.invalid == 0
            and self.invalid_result_records == 0
            and self.malformed_result_lines == 0
            and self.orphan_result_records == 0
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "parsed_by_llm": self.parsed_by_llm,
            "fallback": self.fallback,
            "error": self.error,
            "invalid": self.invalid,
            "invalid_result_records": self.invalid_result_records,
            "missing": self.missing,
            "complete": self.complete,
            "malformed_result_lines": self.malformed_result_lines,
            "orphan_result_records": self.orphan_result_records,
            "datasets": [item.as_dict() for item in self.datasets],
        }


@dataclass(frozen=True)
class CampaignConfig:
    repo: Path = REPO
    manifest_dir: Path = DEFAULT_MANIFEST_DIR
    results_dir: Path = DEFAULT_RESULTS_DIR
    state_dir: Path = DEFAULT_STATE_DIR
    python_executable: str = sys.executable
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    workers: int = 2
    max_rounds: int = 30
    max_consecutive_no_progress_rounds: int = 3
    round_backoff_seconds: float = 60.0
    max_log_bytes: int = 64 * 1024
    wait_pid: int | None = None
    wait_poll_seconds: float = 30.0
    wait_timeout_seconds: float | None = None

    @property
    def checkpoint_path(self) -> Path:
        return self.state_dir / "checkpoint.json"


ProcessRunner = Callable[
    [Sequence[str], Path, Path, int, Mapping[str, str]],
    int,
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_object(raw_line: str, path: Path, line_number: int) -> dict:
    try:
        value = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise CampaignError(f"JSON invalido en {path}:{line_number}") from exc
    if not isinstance(value, dict):
        raise CampaignError(f"Se esperaba un objeto JSON en {path}:{line_number}")
    return value


def load_manifest_inventory(manifest_dir: Path) -> ManifestInventory:
    """Carga y valida el universo de IDs sin retener las filas crudas."""
    paths = sorted(manifest_dir.glob(f"*{_MANIFEST_SUFFIX}"))
    if not paths:
        raise CampaignError(f"No hay manifiestos en {manifest_dir}")

    by_dataset: dict[str, frozenset[str]] = {}
    all_ids: set[str] = set()
    for path in paths:
        dataset = path.name.removesuffix(_MANIFEST_SUFFIX)
        ids: set[str] = set()
        with path.open(encoding="utf-8-sig") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                if not raw_line.strip():
                    continue
                record = _json_object(raw_line, path, line_number)
                manifest_id = record.get("manifest_id")
                if not isinstance(manifest_id, str) or not manifest_id:
                    raise CampaignError(
                        f"manifest_id ausente o invalido en {path}:{line_number}"
                    )
                if manifest_id in ids or manifest_id in all_ids:
                    raise CampaignError(f"manifest_id duplicado: {manifest_id}")
                ids.add(manifest_id)
                all_ids.add(manifest_id)
        if not ids:
            raise CampaignError(f"Manifiesto vacio: {path}")
        by_dataset[dataset] = frozenset(ids)

    return ManifestInventory(dataset_ids=by_dataset)


def result_state(
    record: Mapping[str, object],
    *,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
) -> str:
    """Clasifica un intento sin aceptar exitos LLM estructuralmente vacios.

    Todo intento de esta campana debe declarar el proveedor y el modelo
    fijados. Ademas, un supuesto exito LLM solo es util si contiene un evento
    canonico no vacio. Un intento valido posterior puede reparar el estado de
    ese ID, ya que los JSONL son append-only.
    """
    ok = record.get("ok")
    parsed_by_llm = record.get("parsed_by_llm")
    if not isinstance(ok, bool) or not isinstance(parsed_by_llm, bool):
        return "invalid"
    if record.get("provider") != provider or record.get("model") != model:
        return "invalid"
    if parsed_by_llm:
        canonical_event = record.get("canonical_event")
        if ok and isinstance(canonical_event, Mapping) and bool(canonical_event):
            return "llm"
        return "invalid"
    if ok:
        return "fallback"
    return "error"


def merge_result_state(previous: str | None, current: str) -> str:
    """Un exito LLM es terminal; sin el, prevalece el intento mas reciente."""
    if previous == "llm" or current == "llm":
        return "llm"
    return current


def scan_campaign_status(
    inventory: ManifestInventory,
    results_dir: Path,
    *,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
) -> CampaignStatus:
    """Resume JSONL append-only sin depender de ningun artefacto de Fase D."""
    dataset_statuses: list[DatasetStatus] = []
    malformed = 0
    orphan_records = 0
    invalid_records = 0

    for dataset, expected_ids in sorted(inventory.dataset_ids.items()):
        states: dict[str, str] = {}
        result_path = results_dir / f"{dataset}{_RESULT_SUFFIX}"
        if result_path.is_file():
            with result_path.open(encoding="utf-8-sig", errors="replace") as handle:
                for raw_line in handle:
                    if not raw_line.strip():
                        continue
                    try:
                        record = json.loads(raw_line)
                    except json.JSONDecodeError:
                        malformed += 1
                        continue
                    if not isinstance(record, dict):
                        malformed += 1
                        continue
                    manifest_id = record.get("manifest_id")
                    if not isinstance(manifest_id, str) or manifest_id not in expected_ids:
                        orphan_records += 1
                        continue
                    state = result_state(record, provider=provider, model=model)
                    if state == "invalid":
                        invalid_records += 1
                    states[manifest_id] = merge_result_state(states.get(manifest_id), state)

        counts = {"llm": 0, "fallback": 0, "error": 0, "invalid": 0}
        for state in states.values():
            counts[state] += 1
        total = len(expected_ids)
        dataset_statuses.append(
            DatasetStatus(
                dataset=dataset,
                total=total,
                parsed_by_llm=counts["llm"],
                fallback=counts["fallback"],
                error=counts["error"],
                invalid=counts["invalid"],
                missing=total - len(states),
            )
        )

    return CampaignStatus(
        datasets=tuple(dataset_statuses),
        malformed_result_lines=malformed,
        orphan_result_records=orphan_records,
        invalid_result_records=invalid_records,
    )


def standardization_command(config: CampaignConfig) -> tuple[str, ...]:
    return (
        config.python_executable,
        "-u",
        "scripts/run_live_standardization.py",
        "--workers",
        str(config.workers),
        "--retry-failures",
    )


def compact_log(path: Path, max_bytes: int) -> None:
    """Conserva solo la cola de un log finalizado y limita su tamano."""
    if max_bytes <= 0 or not path.is_file() or path.stat().st_size <= max_bytes:
        return
    marker = b"[... inicio del log truncado ...]\n"
    tail_size = max(1, max_bytes - len(marker))
    with path.open("rb") as handle:
        handle.seek(-tail_size, os.SEEK_END)
        tail = handle.read()
    newline = tail.find(b"\n")
    if newline >= 0:
        tail = tail[newline + 1 :]
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        handle.write(marker)
        handle.write(tail)
    tmp_path.replace(path)


_SENSITIVE_ENV_MARKERS = (
    "API_KEY",
    "APIKEY",
    "ACCESS_KEY",
    "AUTHORIZATION",
    "BEARER",
    "CREDENTIAL",
    "PASSWORD",
    "PASSWD",
    "PRIVATE_KEY",
    "SECRET",
    "TOKEN",
)


def _is_sensitive_env_name(name: str) -> bool:
    normalized = name.upper()
    return (
        normalized == "KEY"
        or normalized.endswith("_KEY")
        or normalized == "PAT"
        or normalized.endswith("_PAT")
        or normalized.endswith("_AUTH")
        or normalized.endswith("_DSN")
        or normalized.endswith("_URI")
        or normalized.endswith("_URL")
        or "CONNECTION_STRING" in normalized
        or any(marker in normalized for marker in _SENSITIVE_ENV_MARKERS)
    )


def standardization_environment(
    config: CampaignConfig,
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Construye el entorno de Fase C y exige la clave solo en este punto."""
    source_environment = os.environ if source is None else source
    mistral_key = source_environment.get("MISTRAL_API_KEY")
    if not mistral_key:
        raise MissingCredentialError(
            "MISTRAL_API_KEY es necesaria porque la estandarizacion esta incompleta"
        )
    environment = scrub_sensitive_environment(source_environment)
    environment["MISTRAL_API_KEY"] = mistral_key
    environment["LLM_PROVIDER"] = config.provider
    environment["INGEST_LLM_MODEL"] = config.model
    return environment


def scrub_sensitive_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Hereda solo variables no sensibles al subproceso de estandarizacion."""
    environment = os.environ if source is None else source
    return {
        name: value
        for name, value in environment.items()
        if not _is_sensitive_env_name(name)
    }


def run_logged_process(
    command: Sequence[str],
    cwd: Path,
    log_path: Path,
    max_log_bytes: int,
    environment: Mapping[str, str],
    *,
    subprocess_run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> int:
    """Ejecuta sin shell con un entorno explicito que nunca se registra."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("wb") as log_handle:
            completed = subprocess_run(
                list(command),
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
                env=dict(environment),
            )
    except OSError as exc:
        raise CampaignError(f"No se pudo iniciar {command[2]}: {exc}") from exc
    finally:
        compact_log(log_path, max_log_bytes)
    return int(completed.returncode)


def write_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    """Reemplaza un unico checkpoint pequeno; nunca recibe el entorno."""
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"updated_at": utc_now(), **payload}
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def pid_exists(pid: int, *, platform_name: str | None = None) -> bool:
    """Comprueba un PID sin enviar senales destructivas en Windows."""
    if pid <= 0:
        return False
    platform = platform_name or os.name
    if platform == "nt":
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        open_process.restype = wintypes.HANDLE
        get_exit_code = kernel32.GetExitCodeProcess
        get_exit_code.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        get_exit_code.restype = wintypes.BOOL
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL

        handle = open_process(process_query_limited_information, False, pid)
        if not handle:
            # Acceso denegado tambien demuestra que el proceso existe.
            return ctypes.get_last_error() == 5
        try:
            exit_code = wintypes.DWORD()
            return bool(get_exit_code(handle, ctypes.byref(exit_code))) and (
                exit_code.value == still_active
            )
        finally:
            close_handle(handle)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_pid(
    pid: int,
    *,
    poll_seconds: float,
    timeout_seconds: float | None,
    exists: Callable[[int], bool] = pid_exists,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    if pid <= 0:
        raise CampaignError("El PID debe ser positivo")
    if poll_seconds <= 0:
        raise CampaignError("El intervalo de espera debe ser positivo")
    started = monotonic()
    while exists(pid):
        if timeout_seconds is not None and monotonic() - started >= timeout_seconds:
            raise CampaignError(f"Tiempo de espera agotado para el PID {pid}")
        sleep_in_chunks(poll_seconds, sleep=sleep)


def sleep_in_chunks(
    seconds: float,
    *,
    sleep: Callable[[float], None] = time.sleep,
    max_chunk_seconds: float = 60.0,
) -> None:
    """Espera en tramos cortos para conservar una interrupcion responsiva."""
    if seconds < 0:
        raise CampaignError("El backoff no puede ser negativo")
    if max_chunk_seconds <= 0 or max_chunk_seconds > 60:
        raise CampaignError("Cada tramo de espera debe estar entre 0 y 60 segundos")
    remaining = float(seconds)
    while remaining > 0:
        chunk = min(remaining, max_chunk_seconds)
        sleep(chunk)
        remaining -= chunk


def request_system_awake(platform_name: str | None = None, setter=None) -> bool:
    if (platform_name or os.name) != "nt":
        return False
    if setter is None:
        import ctypes

        setter = ctypes.windll.kernel32.SetThreadExecutionState
    return bool(setter(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED))


def release_system_awake(platform_name: str | None = None, setter=None) -> bool:
    if (platform_name or os.name) != "nt":
        return False
    if setter is None:
        import ctypes

        setter = ctypes.windll.kernel32.SetThreadExecutionState
    return bool(setter(_ES_CONTINUOUS))


def _status_line(status: CampaignStatus) -> str:
    return (
        f"LLM={status.parsed_by_llm}/{status.total} "
        f"fallback={status.fallback} error={status.error} invalid={status.invalid} "
        f"invalid_records={status.invalid_result_records} "
        f"missing={status.missing} malformed={status.malformed_result_lines} "
        f"orphans={status.orphan_result_records}"
    )


def _validate_config(config: CampaignConfig) -> None:
    if config.workers <= 0:
        raise CampaignError("--workers debe ser positivo")
    if config.max_rounds <= 0:
        raise CampaignError("--max-rounds debe ser positivo")
    if config.max_consecutive_no_progress_rounds <= 0:
        raise CampaignError("--max-no-progress-rounds debe ser positivo")
    if config.round_backoff_seconds < 0:
        raise CampaignError("--backoff-seconds no puede ser negativo")
    if config.max_log_bytes <= 0:
        raise CampaignError("El limite de log debe ser positivo")
    if not config.provider.strip() or not config.model.strip():
        raise CampaignError("provider y model no pueden estar vacios")
    if config.wait_pid == os.getpid():
        raise CampaignError("El supervisor no puede esperar a su propio PID")


def _validate_builtin_standardization_layout(config: CampaignConfig) -> None:
    """Evita escanear rutas que el runner actual de Fase C no puede recibir.

    ``run_live_standardization.py`` usa rutas ligadas a este repositorio y no
    expone argumentos de directorio. Los runners inyectados en tests pueden
    implementar otros layouts; el runner real debe usar exactamente el suyo.
    """
    expected = {
        "repo": REPO,
        "manifest_dir": DEFAULT_MANIFEST_DIR,
        "results_dir": DEFAULT_RESULTS_DIR,
    }
    actual = {
        "repo": config.repo,
        "manifest_dir": config.manifest_dir,
        "results_dir": config.results_dir,
    }
    mismatches = [
        name
        for name in expected
        if actual[name].resolve() != expected[name].resolve()
    ]
    if mismatches:
        raise CampaignError(
            "run_live_standardization.py no admite rutas personalizadas; "
            "no coinciden: " + ", ".join(mismatches)
        )


def _checkpoint(
    config: CampaignConfig,
    phase: str,
    status: CampaignStatus,
    **extra: object,
) -> None:
    write_checkpoint(
        config.checkpoint_path,
        {
            "phase": phase,
            "provider": config.provider,
            "model": config.model,
            "status": status.as_dict(),
            **extra,
        },
    )


def _complete_standardization(
    config: CampaignConfig,
    status: CampaignStatus,
    *,
    counters: Mapping[str, int] | None = None,
) -> int:
    checkpoint_extra = {"counters": dict(counters or {})}
    _checkpoint(
        config,
        "complete",
        status,
        completed_scope="llm_standardization",
        **checkpoint_extra,
    )
    print(
        "Cobertura LLM completa. Estandarizacion cerrada; el entrenamiento y "
        f"la evaluacion se ejecutan con sus runners actuales. {_status_line(status)}",
        flush=True,
    )
    return 0


def run_campaign(
    config: CampaignConfig,
    *,
    process_runner: ProcessRunner = run_logged_process,
    process_exists: Callable[[int], bool] = pid_exists,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> int:
    """Ejecuta la maquina de estados; las dependencias lentas son inyectables."""
    _validate_config(config)

    inventory = load_manifest_inventory(config.manifest_dir)
    status = scan_campaign_status(
        inventory,
        config.results_dir,
        provider=config.provider,
        model=config.model,
    )
    counters = {
        "rounds_started": 0,
        "rounds_finished": 0,
        "nonzero_return_codes": 0,
        "consecutive_no_progress_rounds": 0,
        "total_new_llm_successes": 0,
    }

    if config.wait_pid is not None:
        _checkpoint(
            config,
            "waiting_for_pid",
            status,
            wait_pid=config.wait_pid,
            counters=dict(counters),
        )
        print(f"Esperando al PID {config.wait_pid}. {_status_line(status)}", flush=True)
        wait_for_pid(
            config.wait_pid,
            poll_seconds=config.wait_poll_seconds,
            timeout_seconds=config.wait_timeout_seconds,
            exists=process_exists,
            sleep=sleep,
            monotonic=monotonic,
        )
        status = scan_campaign_status(
            inventory,
            config.results_dir,
            provider=config.provider,
            model=config.model,
        )

    print(f"Estado inicial: {_status_line(status)}", flush=True)
    if status.complete:
        return _complete_standardization(
            config,
            status,
            counters=counters,
        )

    if process_runner is run_logged_process:
        _validate_builtin_standardization_layout(config)
    standardization_env = standardization_environment(config)

    previous_llm = status.parsed_by_llm
    last_return_code: int | None = None
    for round_number in range(1, config.max_rounds + 1):
        counters["rounds_started"] += 1
        _checkpoint(
            config,
            "standardizing",
            status,
            round=round_number,
            counters=dict(counters),
        )
        print(f"Ronda {round_number}/{config.max_rounds}: {_status_line(status)}", flush=True)
        try:
            return_code = process_runner(
                standardization_command(config),
                config.repo,
                config.state_dir / f"standardization_round_{round_number:03d}.log",
                config.max_log_bytes,
                standardization_env,
            )
        except (CampaignError, OSError) as exc:
            _checkpoint(
                config,
                "standardization_process_error",
                status,
                round=round_number,
                error_type=type(exc).__name__,
                counters=dict(counters),
            )
            raise CampaignError(
                f"No se pudo ejecutar el proceso de estandarizacion en la ronda {round_number}"
            ) from exc

        last_return_code = int(return_code)
        counters["rounds_finished"] += 1
        if last_return_code != 0:
            counters["nonzero_return_codes"] += 1
        status = scan_campaign_status(
            inventory,
            config.results_dir,
            provider=config.provider,
            model=config.model,
        )
        progress = status.parsed_by_llm - previous_llm
        counters["total_new_llm_successes"] += max(0, progress)
        if progress > 0:
            counters["consecutive_no_progress_rounds"] = 0
        else:
            counters["consecutive_no_progress_rounds"] += 1
        _checkpoint(
            config,
            "round_finished",
            status,
            round=round_number,
            return_code=last_return_code,
            new_llm_successes=progress,
            counters=dict(counters),
        )
        print(
            f"Fin de ronda {round_number}: rc={last_return_code} "
            f"+{progress} LLM; {_status_line(status)}",
            flush=True,
        )

        if status.complete:
            return _complete_standardization(
                config,
                status,
                counters=counters,
            )
        if (
            counters["consecutive_no_progress_rounds"]
            >= config.max_consecutive_no_progress_rounds
        ):
            _checkpoint(
                config,
                "failed_no_progress",
                status,
                round=round_number,
                return_code=last_return_code,
                counters=dict(counters),
            )
            raise NoProgressError(
                f"{counters['consecutive_no_progress_rounds']} rondas consecutivas "
                f"sin nuevos exitos LLM; ultimo codigo={last_return_code}"
            )
        previous_llm = status.parsed_by_llm

        if round_number < config.max_rounds and config.round_backoff_seconds > 0:
            _checkpoint(
                config,
                "backoff",
                status,
                round=round_number,
                backoff_seconds=config.round_backoff_seconds,
                counters=dict(counters),
            )
            sleep_in_chunks(config.round_backoff_seconds, sleep=sleep)

    _checkpoint(
        config,
        "failed_max_rounds",
        status,
        rounds=config.max_rounds,
        last_return_code=last_return_code,
        counters=dict(counters),
    )
    raise MaxRoundsExceeded(
        f"Cobertura incompleta tras {config.max_rounds} rondas: {_status_line(status)}"
    )


def supervise_campaign(
    config: CampaignConfig,
    *,
    process_runner: ProcessRunner = run_logged_process,
    process_exists: Callable[[int], bool] = pid_exists,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    platform_name: str | None = None,
    execution_state_setter=None,
) -> int:
    """Mantiene Windows despierto durante la espera y la estandarizacion."""
    awake = request_system_awake(platform_name, execution_state_setter)
    if (platform_name or os.name) == "nt" and not awake:
        print("AVISO: Windows no acepto la solicitud para impedir la suspension.", flush=True)
    try:
        return run_campaign(
            config,
            process_runner=process_runner,
            process_exists=process_exists,
            sleep=sleep,
            monotonic=monotonic,
        )
    finally:
        if awake:
            release_system_awake(platform_name, execution_state_setter)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("debe ser mayor que cero")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("debe ser mayor que cero")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("debe ser mayor o igual que cero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=_positive_int, default=None)
    parser.add_argument("--workers", type=_positive_int, default=2)
    parser.add_argument("--max-rounds", type=_positive_int, default=30)
    parser.add_argument(
        "--max-no-progress-rounds",
        type=_positive_int,
        default=3,
        help="rondas consecutivas sin nuevos exitos antes de detenerse",
    )
    parser.add_argument(
        "--backoff-seconds",
        type=_nonnegative_float,
        default=60.0,
        help="espera reanudable entre rondas; se divide en tramos de hasta 60 s",
    )
    parser.add_argument("--wait-poll-seconds", type=_positive_float, default=30.0)
    parser.add_argument(
        "--wait-timeout-hours",
        type=_positive_float,
        default=None,
        help="por defecto espera sin limite",
    )
    parser.add_argument("--max-log-kib", type=_positive_int, default=64)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = CampaignConfig(
        repo=REPO,
        manifest_dir=args.manifest_dir,
        results_dir=args.results_dir,
        state_dir=args.state_dir,
        python_executable=args.python_executable,
        provider=args.provider,
        model=args.model,
        workers=args.workers,
        max_rounds=args.max_rounds,
        max_consecutive_no_progress_rounds=args.max_no_progress_rounds,
        round_backoff_seconds=args.backoff_seconds,
        max_log_bytes=args.max_log_kib * 1024,
        wait_pid=args.wait_pid,
        wait_poll_seconds=args.wait_poll_seconds,
        wait_timeout_seconds=(
            args.wait_timeout_hours * 3600 if args.wait_timeout_hours is not None else None
        ),
    )
    try:
        return supervise_campaign(config)
    except MissingCredentialError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except CampaignError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Campana interrumpida; el progreso JSONL queda conservado.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
