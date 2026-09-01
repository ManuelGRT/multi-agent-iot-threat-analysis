"""Contrato de datos y preflight de la campana de validacion 2026.

Este modulo prepara datos estandarizados para un evaluador posterior, pero no
entrena modelos. Sus invariantes principales son:

* el manifiesto es la unica autoridad para dataset, split, tareas y targets;
* un fallback posterior nunca sustituye a un resultado LLM valido anterior;
* el preflight conserva todas las filas; la deduplicacion es tarea-especifica;
* las features proceden exclusivamente de :func:`src.mcp.features.event_features`;
* no se admiten targets, valores no finitos ni solapamiento exacto entre splits.

La API pensada para un CLI es :func:`load_validation_campaign`. Para inspeccionar
una campana incompleta se puede usar ``PreflightRequirements(require_complete=False)``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence
import unicodedata

from src.eval.predictive_sanitization import is_predictive_target_field
from src.contracts.canonical import CanonicalEvent
from src.mcp import features as canonical_features
from src.eval.data_sanitization import (
    ALLOWED_CANONICAL_TARGET_LIKE_PATHS as _ALLOWED_CANONICAL_TARGET_LIKE_PATHS,
)


Task = Literal["binary", "multiclass", "standardization"]
Split = Literal["train", "val", "test"]
DedupScope = Literal["dataset", "global"]

ALLOWED_TASKS: frozenset[str] = frozenset({"binary", "multiclass", "standardization"})
ALLOWED_SPLITS: frozenset[str] = frozenset({"train", "val", "test"})
_TASK_ORDER = {"binary": 0, "multiclass": 1, "standardization": 2}
_SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}
_FORBIDDEN_FEATURE_PREFIXES = (
    "origin.",
    "attack_indicator.",
    "semantic_hint.",
    "behavior.",
    "uncertainty.",
    "asset_context.",
)
_FORBIDDEN_FEATURE_KEYS = frozenset(
    {"attack_indicator_count", "behavior_tag_count", "uncertainty_count"}
)


class ValidationCampaignError(ValueError):
    """Base para errores que invalidan el contrato de evaluacion."""


class JsonlFormatError(ValidationCampaignError):
    """Un JSONL contiene una linea completa corrupta."""


class ManifestContractError(ValidationCampaignError):
    """Un manifiesto no cumple el esquema o sus invariantes globales."""


class ResultContractError(ValidationCampaignError):
    """Un intento de estandarizacion no cumple su contrato."""


class FeatureSafetyError(ValidationCampaignError):
    """Las features contienen leakage o valores no aptos para entrenamiento."""


class DeduplicationConflictError(ValidationCampaignError):
    """Una deduplicacion intentaria ocultar tareas o targets incompatibles."""


class PreflightError(ValidationCampaignError):
    """La campana es estructuralmente valida, pero aun no esta lista."""

    def __init__(self, message: str, report: QualityReport | None = None):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    manifest_id: str
    dataset: str
    source_file: str
    row_id: str | int
    content_hash: str
    split: Split
    tasks: tuple[Task, ...]
    target_class: str | None
    target_is_attack: bool | None
    source_path: Path
    line_number: int

    def target_for(self, task: Task) -> bool | str | None:
        if task == "binary":
            return self.target_is_attack
        if task == "multiclass":
            return self.target_class
        if task == "standardization":
            return None
        raise ValueError(f"Tarea desconocida: {task}")


@dataclass(frozen=True, slots=True)
class ManifestIndex:
    records: tuple[ManifestRecord, ...]
    by_id: Mapping[str, ManifestRecord]
    by_content_hash: Mapping[str, ManifestRecord]

    def __len__(self) -> int:
        return len(self.records)


@dataclass(frozen=True, slots=True)
class ResultAttempt:
    manifest_id: str
    ok: bool
    parsed_by_llm: bool
    provider: str | None
    model: str | None
    canonical_event: CanonicalEvent | None
    mapping_confidence: float | None
    latency_s: float | None
    error: str | None
    source_path: Path
    line_number: int
    sequence: int

    @property
    def is_valid_success(self) -> bool:
        return self.ok and self.canonical_event is not None

    @property
    def is_valid_llm(self) -> bool:
        return self.is_valid_success and self.parsed_by_llm


@dataclass(frozen=True, slots=True)
class ResultIndex:
    latest_attempt: Mapping[str, ResultAttempt]
    latest_valid_llm: Mapping[str, ResultAttempt]
    latest_valid_success: Mapping[str, ResultAttempt]
    latest_valid_llm_by_backend: Mapping[
        tuple[str, str | None, str | None], ResultAttempt
    ]
    latest_valid_success_by_backend: Mapping[
        tuple[str, str | None, str | None], ResultAttempt
    ]
    attempts_per_manifest: Mapping[str, int]
    total_attempts: int
    truncated_final_lines: int

    def preferred(
        self,
        manifest_id: str,
        require_llm: bool,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> ResultAttempt | None:
        """Devuelve el ultimo exito que satisface el backend solicitado.

        El indice global por ``manifest_id`` se conserva para metricas y
        compatibilidad, pero no es suficiente para seleccionar un artefacto
        reproducible: un exito posterior de otro modelo no debe ocultar el
        ultimo exito del modelo requerido.
        """
        llm = self._latest_matching(
            self.latest_valid_llm_by_backend,
            manifest_id,
            provider=provider,
            model=model,
        )
        if require_llm or llm is not None:
            return llm
        return self._latest_matching(
            self.latest_valid_success_by_backend,
            manifest_id,
            provider=provider,
            model=model,
        )

    @staticmethod
    def _latest_matching(
        index: Mapping[tuple[str, str | None, str | None], ResultAttempt],
        manifest_id: str,
        *,
        provider: str | None,
        model: str | None,
    ) -> ResultAttempt | None:
        normalized_provider = _normalize_provider(provider)
        normalized_model = model.strip() if model is not None else None
        candidates = (
            attempt
            for (
                candidate_id,
                candidate_provider,
                candidate_model,
            ), attempt in index.items()
            if candidate_id == manifest_id
            and (normalized_provider is None or candidate_provider == normalized_provider)
            and (normalized_model is None or candidate_model == normalized_model)
        )
        return max(candidates, key=lambda item: item.sequence, default=None)


@dataclass(frozen=True, slots=True)
class PreflightRequirements:
    """Filtros de disponibilidad antes de exponer filas a un evaluador.

    ``provider`` se compara sin distinguir mayusculas; ``model`` se compara de
    forma exacta para que el artefacto sea reproducible.
    """

    require_complete: bool = True
    require_llm: bool = True
    provider: str | None = None
    model: str | None = None
    reject_orphan_results: bool = True
    dedup_scope: DedupScope = "dataset"

    def __post_init__(self) -> None:
        if self.provider is not None and not self.provider.strip():
            raise ValueError("provider no puede estar vacio")
        if self.model is not None and not self.model.strip():
            raise ValueError("model no puede estar vacio")
        if self.dedup_scope not in {"dataset", "global"}:
            raise ValueError("dedup_scope debe ser 'dataset' o 'global'")


@dataclass(frozen=True, slots=True)
class PreparedRecord:
    """Fila lista para un evaluador, con metadatos autoritativos del manifiesto."""

    manifest_id: str
    dataset: str
    source_file: str
    row_id: str | int
    content_hash: str
    split: Split
    tasks: tuple[Task, ...]
    target_class: str | None
    target_is_attack: bool | None
    event: CanonicalEvent
    features: Mapping[str, bool | int | float | str]
    feature_fingerprint: str
    provider: str | None
    model: str | None
    mapping_confidence: float
    latency_s: float | None
    result_source_path: Path
    result_line_number: int

    def target_for(self, task: Task) -> bool | str | None:
        if task not in self.tasks:
            raise ValueError(f"{self.manifest_id} no participa en {task}")
        if task == "binary":
            return self.target_is_attack
        if task == "multiclass":
            return self.target_class
        return None


@dataclass(frozen=True, slots=True)
class DatasetQualityMetrics:
    manifest_rows: int
    attempted_rows: int
    latest_ok_rows: int
    latest_error_rows: int
    latest_fallback_rows: int
    valid_llm_rows: int
    selected_rows: int
    missing_rows: int
    provider_mismatch_rows: int
    model_mismatch_rows: int
    parse_ok_rate: float
    llm_success_rate: float
    fallback_rate: float
    selected_coverage_rate: float
    avg_mapping_confidence: float | None
    avg_latency_s: float | None
    avg_feature_count: float | None
    min_feature_count: int | None
    max_feature_count: int | None
    split_counts: Mapping[str, int]
    task_counts: Mapping[str, int]
    schema_profile_counts: Mapping[str, int]
    modality_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class QualityReport:
    global_metrics: DatasetQualityMetrics
    by_dataset: Mapping[str, DatasetQualityMetrics]
    result_attempts: int
    retried_manifest_rows: int
    orphan_result_rows: int
    truncated_final_lines: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DeduplicationReport:
    scope: DedupScope
    input_rows: int
    output_rows: int
    same_split_duplicate_groups: int
    same_split_rows_removed: int
    cross_split_groups: int
    cross_split_rows_removed: int
    same_split_removed_ids: tuple[str, ...] = field(default_factory=tuple)
    cross_split_removed_ids: tuple[str, ...] = field(default_factory=tuple)
    applied: bool = True
    task: Task | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ValidationCampaign:
    manifests: ManifestIndex
    results: ResultIndex
    requirements: PreflightRequirements
    joined_records: tuple[PreparedRecord, ...]
    records: tuple[PreparedRecord, ...]
    quality: QualityReport
    deduplication: DeduplicationReport

    def records_for(
        self,
        *,
        dataset: str | None = None,
        task: Task | None = None,
        split: Split | None = None,
    ) -> tuple[PreparedRecord, ...]:
        if task is not None and task not in ALLOWED_TASKS:
            raise ValueError(f"Tarea desconocida: {task}")
        if split is not None and split not in ALLOWED_SPLITS:
            raise ValueError(f"Split desconocido: {split}")
        return tuple(
            record
            for record in self.records
            if (dataset is None or record.dataset == dataset)
            and (task is None or task in record.tasks)
            and (split is None or record.split == split)
        )

    def summary(self) -> dict[str, Any]:
        return {
            "quality": self.quality.to_dict(),
            "deduplication": self.deduplication.to_dict(),
            "joined_rows": len(self.joined_records),
            "ready_rows": len(self.records),
            "records_are_authoritative_join": self.records == self.joined_records,
        }


@dataclass(slots=True)
class _JsonlState:
    truncated_final_line: bool = False


def load_manifests(paths: str | Path | Iterable[str | Path]) -> ManifestIndex:
    """Carga uno o varios manifiestos y comprueba unicidad global."""
    manifest_paths = _paths(paths, allow_empty=False)
    records: list[ManifestRecord] = []
    by_id: dict[str, ManifestRecord] = {}
    by_hash: dict[str, ManifestRecord] = {}

    for path in manifest_paths:
        for line_number, payload in _iter_jsonl(path, tolerate_truncated_final=False):
            record = _parse_manifest(payload, path, line_number)
            previous_id = by_id.get(record.manifest_id)
            if previous_id is not None:
                raise ManifestContractError(
                    f"manifest_id duplicado {record.manifest_id!r}: "
                    f"{_where(previous_id.source_path, previous_id.line_number)} y "
                    f"{_where(path, line_number)}"
                )
            previous_hash = by_hash.get(record.content_hash)
            if previous_hash is not None:
                raise ManifestContractError(
                    f"content_hash duplicado {record.content_hash!r}: "
                    f"{previous_hash.manifest_id!r} y {record.manifest_id!r}"
                )
            records.append(record)
            by_id[record.manifest_id] = record
            by_hash[record.content_hash] = record

    if not records:
        raise ManifestContractError("Los manifiestos no contienen filas")
    ordered = tuple(sorted(records, key=lambda item: item.manifest_id))
    return ManifestIndex(records=ordered, by_id=by_id, by_content_hash=by_hash)


def load_results(paths: str | Path | Iterable[str | Path]) -> ResultIndex:
    """Carga intentos JSONL reanudables.

    Solo se tolera una ultima linea sin terminador que sea JSON/UTF-8 truncado,
    como puede observarse mientras el runner esta escribiendo. Cualquier
    corrupcion en una linea intermedia o terminada invalida el fichero.
    """
    result_paths = _paths(paths, allow_empty=True)
    latest_attempt: dict[str, ResultAttempt] = {}
    latest_llm: dict[str, ResultAttempt] = {}
    latest_success: dict[str, ResultAttempt] = {}
    latest_llm_by_backend: dict[
        tuple[str, str | None, str | None], ResultAttempt
    ] = {}
    latest_success_by_backend: dict[
        tuple[str, str | None, str | None], ResultAttempt
    ] = {}
    attempts: Counter[str] = Counter()
    total = 0
    truncated = 0

    for path in result_paths:
        state = _JsonlState()
        for line_number, payload in _iter_jsonl(
            path,
            tolerate_truncated_final=True,
            state=state,
        ):
            total += 1
            attempt = _parse_result(payload, path, line_number, total)
            attempts[attempt.manifest_id] += 1
            latest_attempt[attempt.manifest_id] = attempt
            backend_key = _result_backend_key(attempt)
            if attempt.is_valid_success:
                latest_success[attempt.manifest_id] = attempt
                latest_success_by_backend[backend_key] = attempt
            if attempt.is_valid_llm:
                latest_llm[attempt.manifest_id] = attempt
                latest_llm_by_backend[backend_key] = attempt
        truncated += int(state.truncated_final_line)

    return ResultIndex(
        latest_attempt=latest_attempt,
        latest_valid_llm=latest_llm,
        latest_valid_success=latest_success,
        latest_valid_llm_by_backend=latest_llm_by_backend,
        latest_valid_success_by_backend=latest_success_by_backend,
        attempts_per_manifest=dict(attempts),
        total_attempts=total,
        truncated_final_lines=truncated,
    )


def load_validation_campaign(
    manifest_paths: str | Path | Iterable[str | Path],
    result_paths: str | Path | Iterable[str | Path],
    requirements: PreflightRequirements | None = None,
) -> ValidationCampaign:
    """Carga, une y valida una campana sin entrenar modelos.

    La deduplicacion depende de la tarea y se deja al evaluador. Tanto
    ``joined_records`` como ``records`` conservan todas las filas autoritativas.
    """
    manifests = load_manifests(manifest_paths)
    results = load_results(result_paths)
    return prepare_validation_campaign(manifests, results, requirements)


def prepare_validation_campaign(
    manifests: ManifestIndex,
    results: ResultIndex,
    requirements: PreflightRequirements | None = None,
) -> ValidationCampaign:
    """Une indices ya cargados y aplica el preflight configurable."""
    policy = requirements or PreflightRequirements()
    manifest_ids = set(manifests.by_id)
    orphan_ids = sorted(set(results.latest_attempt) - manifest_ids)

    selected: dict[str, ResultAttempt] = {}
    provider_mismatches: set[str] = set()
    model_mismatches: set[str] = set()
    for manifest in manifests.records:
        attempt = results.preferred(
            manifest.manifest_id,
            policy.require_llm,
            provider=policy.provider,
            model=policy.model,
        )
        if attempt is None:
            unfiltered = results.preferred(manifest.manifest_id, policy.require_llm)
            if unfiltered is None:
                continue
            provider_match = results.preferred(
                manifest.manifest_id,
                policy.require_llm,
                provider=policy.provider,
            )
            if policy.provider is not None and provider_match is None:
                provider_mismatches.add(manifest.manifest_id)
            elif policy.model is not None:
                model_mismatches.add(manifest.manifest_id)
            continue
        selected[manifest.manifest_id] = attempt

    joined = tuple(
        _prepare_record(manifest, selected[manifest.manifest_id])
        for manifest in manifests.records
        if manifest.manifest_id in selected
    )
    report = _quality_report(
        manifests,
        results,
        joined,
        provider_mismatches,
        model_mismatches,
        len(orphan_ids),
    )

    if policy.reject_orphan_results and orphan_ids:
        sample = ", ".join(repr(value) for value in orphan_ids[:3])
        raise PreflightError(
            f"Hay {len(orphan_ids)} manifest_id de resultados ajenos al manifiesto"
            f" (ejemplos: {sample})",
            report,
        )
    if policy.require_complete and len(joined) != len(manifests):
        metrics = report.global_metrics
        raise PreflightError(
            "Campana incompleta para los requisitos solicitados: "
            f"seleccionadas={len(joined)}/{len(manifests)}, "
            f"sin_resultado_valido={metrics.missing_rows}, "
            f"provider_mismatch={metrics.provider_mismatch_rows}, "
            f"model_mismatch={metrics.model_mismatch_rows}",
            report,
        )

    dedup_report = DeduplicationReport(
        scope=policy.dedup_scope,
        input_rows=len(joined),
        output_rows=len(joined),
        same_split_duplicate_groups=0,
        same_split_rows_removed=0,
        cross_split_groups=0,
        cross_split_rows_removed=0,
        applied=False,
    )
    return ValidationCampaign(
        manifests=manifests,
        results=results,
        requirements=policy,
        joined_records=joined,
        records=joined,
        quality=report,
        deduplication=dedup_report,
    )


def feature_fingerprint(features: Mapping[str, Any]) -> str:
    """Hash SHA-256 estable de un vector de features validado.

    Los numeros se normalizan a IEEE-754 hexadecimal: ``1``, ``1.0`` y
    ``True`` representan el mismo valor numerico que vera ``DictVectorizer``.
    """
    normalized: list[list[str]] = []
    for raw_key in sorted(features, key=lambda item: str(item)):
        if not isinstance(raw_key, str) or not raw_key:
            raise FeatureSafetyError(f"Nombre de feature invalido: {raw_key!r}")
        key = unicodedata.normalize("NFC", raw_key)
        if _is_forbidden_feature(key):
            raise FeatureSafetyError(f"Feature target/leak detectada: {key!r}")
        value = features[raw_key]
        if isinstance(value, (bool, int, float)):
            number = float(value)
            if not math.isfinite(number):
                raise FeatureSafetyError(f"Valor NaN/Inf en feature {key!r}")
            normalized.append([key, "number", number.hex()])
        elif isinstance(value, str):
            normalized.append([key, "string", unicodedata.normalize("NFC", value)])
        else:
            raise FeatureSafetyError(
                f"Valor no escalar en feature {key!r}: {type(value).__name__}"
            )
    payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deduplicate_records(
    records: Sequence[PreparedRecord] | Iterable[PreparedRecord],
    *,
    scope: DedupScope = "dataset",
    task: Task | None = None,
) -> tuple[tuple[PreparedRecord, ...], DeduplicationReport]:
    """Deduplica explicitamente y sin silenciar supervision incompatible.

    Para entrenamiento se recomienda indicar ``task``. Sin ella solo se unen
    filas cuya lista completa de tareas y targets autoritativos coincide; una
    colision incompatible genera :class:`DeduplicationConflictError`.
    """
    if scope not in {"dataset", "global"}:
        raise ValueError("scope debe ser 'dataset' o 'global'")
    if task is not None and task not in ALLOWED_TASKS:
        raise ValueError(f"Tarea desconocida: {task}")
    values = tuple(
        record for record in records if task is None or task in record.tasks
    )
    groups: dict[tuple[str, str], list[PreparedRecord]] = defaultdict(list)
    for record in values:
        namespace = record.dataset if scope == "dataset" else "*"
        groups[(namespace, record.feature_fingerprint)].append(record)

    kept: list[PreparedRecord] = []
    same_removed: list[str] = []
    cross_removed: list[str] = []
    same_groups = 0
    cross_groups = 0
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda item: item.manifest_id)
        if task is None:
            supervision = {
                tuple((item, record.target_for(item)) for item in record.tasks)
                for record in group
            }
        else:
            supervision = {(task, record.target_for(task)) for record in group}
        if len(supervision) > 1:
            ids = ", ".join(record.manifest_id for record in group[:5])
            task_name = task or "todas las tareas"
            raise DeduplicationConflictError(
                "Misma huella de features con supervision incompatible para "
                f"{task_name}: {ids}. Deduplicar ocultaria una tarea o target."
            )
        splits = {item.split for item in group}
        if len(splits) > 1:
            cross_groups += 1
            cross_removed.extend(item.manifest_id for item in group)
            continue
        kept.append(group[0])
        if len(group) > 1:
            same_groups += 1
            same_removed.extend(item.manifest_id for item in group[1:])

    kept.sort(key=lambda item: (item.dataset, _SPLIT_ORDER[item.split], item.manifest_id))
    same_removed.sort()
    cross_removed.sort()
    report = DeduplicationReport(
        scope=scope,
        input_rows=len(values),
        output_rows=len(kept),
        same_split_duplicate_groups=same_groups,
        same_split_rows_removed=len(same_removed),
        cross_split_groups=cross_groups,
        cross_split_rows_removed=len(cross_removed),
        same_split_removed_ids=tuple(same_removed),
        cross_split_removed_ids=tuple(cross_removed),
        applied=True,
        task=task,
    )
    return tuple(kept), report


def _parse_manifest(payload: Any, path: Path, line_number: int) -> ManifestRecord:
    location = _where(path, line_number)
    if not isinstance(payload, dict):
        raise ManifestContractError(f"{location}: cada linea debe ser un objeto JSON")
    manifest_id = _required_string(payload, "manifest_id", ManifestContractError, location)
    dataset = _required_string(payload, "dataset", ManifestContractError, location)
    source_file = _required_string(payload, "source_file", ManifestContractError, location)
    content_hash = _required_string(payload, "content_hash", ManifestContractError, location)
    row_id = payload.get("row_id")
    if isinstance(row_id, bool) or not isinstance(row_id, (str, int)):
        raise ManifestContractError(f"{location}: row_id debe ser str o int")
    split = payload.get("split")
    if split not in ALLOWED_SPLITS:
        raise ManifestContractError(f"{location}: split invalido {split!r}")
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ManifestContractError(f"{location}: tasks debe ser una lista no vacia")
    if any(not isinstance(task, str) or task not in ALLOWED_TASKS for task in raw_tasks):
        raise ManifestContractError(f"{location}: tasks contiene una tarea desconocida")
    if len(set(raw_tasks)) != len(raw_tasks):
        raise ManifestContractError(f"{location}: tasks contiene duplicados")
    tasks = tuple(sorted(raw_tasks, key=_TASK_ORDER.__getitem__))

    target = payload.get("target")
    if not isinstance(target, dict):
        raise ManifestContractError(f"{location}: target debe ser un objeto")
    target_class = target.get("class")
    if target_class is not None and not isinstance(target_class, str):
        raise ManifestContractError(f"{location}: target.class debe ser str o null")
    target_is_attack = target.get("is_attack")
    if target_is_attack is not None and not isinstance(target_is_attack, bool):
        raise ManifestContractError(f"{location}: target.is_attack debe ser bool o null")
    if "binary" in tasks and target_is_attack is None:
        raise ManifestContractError(f"{location}: binary requiere target.is_attack")
    if "multiclass" in tasks and not (target_class and target_class.strip()):
        raise ManifestContractError(f"{location}: multiclass requiere target.class")

    row = payload.get("row")
    if not isinstance(row, dict):
        raise ManifestContractError(f"{location}: row debe ser un objeto")
    leaks = _find_target_keys(row)
    if leaks:
        raise ManifestContractError(
            f"{location}: target/leak dentro de row: {', '.join(leaks[:5])}"
        )
    return ManifestRecord(
        manifest_id=manifest_id,
        dataset=dataset,
        source_file=source_file,
        row_id=row_id,
        content_hash=content_hash,
        split=split,
        tasks=tasks,
        target_class=target_class,
        target_is_attack=target_is_attack,
        source_path=path,
        line_number=line_number,
    )


def _parse_result(
    payload: Any,
    path: Path,
    line_number: int,
    sequence: int,
) -> ResultAttempt:
    location = _where(path, line_number)
    if not isinstance(payload, dict):
        raise ResultContractError(f"{location}: cada linea debe ser un objeto JSON")
    manifest_id = _required_string(payload, "manifest_id", ResultContractError, location)
    ok = payload.get("ok")
    if not isinstance(ok, bool):
        raise ResultContractError(f"{location}: ok debe ser bool")
    parsed_by_llm = payload.get("parsed_by_llm", False)
    if not isinstance(parsed_by_llm, bool):
        raise ResultContractError(f"{location}: parsed_by_llm debe ser bool")
    if parsed_by_llm and not ok:
        raise ResultContractError(f"{location}: parsed_by_llm=true requiere ok=true")

    event: CanonicalEvent | None = None
    event_payload = payload.get("canonical_event")
    if event_payload is not None:
        if not isinstance(event_payload, dict):
            raise ResultContractError(f"{location}: canonical_event debe ser un objeto")
        try:
            event = CanonicalEvent.model_validate(event_payload)
        except Exception as exc:
            raise ResultContractError(
                f"{location}: CanonicalEvent invalido: {exc}"
            ) from exc
    if ok and event is None:
        raise ResultContractError(f"{location}: ok=true requiere canonical_event valido")

    confidence = _optional_finite_number(payload.get("mapping_confidence"), "mapping_confidence", location)
    if confidence is None and event is not None:
        confidence = float(event.mapping_confidence)
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        raise ResultContractError(f"{location}: mapping_confidence fuera de [0, 1]")
    latency = _optional_finite_number(payload.get("latency_s"), "latency_s", location)
    if latency is not None and latency < 0:
        raise ResultContractError(f"{location}: latency_s no puede ser negativa")
    return ResultAttempt(
        manifest_id=manifest_id,
        ok=ok,
        parsed_by_llm=parsed_by_llm,
        provider=_optional_string(payload.get("provider"), "provider", location),
        model=_optional_string(payload.get("model"), "model", location),
        canonical_event=event,
        mapping_confidence=confidence,
        latency_s=latency,
        error=_optional_string(payload.get("error"), "error", location),
        source_path=path,
        line_number=line_number,
        sequence=sequence,
    )


def _prepare_record(manifest: ManifestRecord, attempt: ResultAttempt) -> PreparedRecord:
    event = attempt.canonical_event
    if event is None:  # protegido por ResultAttempt.is_valid_success
        raise ResultContractError(f"Resultado sin CanonicalEvent para {manifest.manifest_id}")
    _validate_event_has_no_targets(event, manifest.manifest_id)
    try:
        raw_features = canonical_features.event_features(event)
    except Exception as exc:
        raise FeatureSafetyError(
            f"No se pudieron extraer features de {manifest.manifest_id}: {exc}"
        ) from exc
    if not isinstance(raw_features, dict) or not raw_features:
        raise FeatureSafetyError(f"Vector de features vacio para {manifest.manifest_id}")
    fingerprint = feature_fingerprint(raw_features)
    confidence = attempt.mapping_confidence
    if confidence is None:
        confidence = float(event.mapping_confidence)
    return PreparedRecord(
        manifest_id=manifest.manifest_id,
        dataset=manifest.dataset,
        source_file=manifest.source_file,
        row_id=manifest.row_id,
        content_hash=manifest.content_hash,
        split=manifest.split,
        tasks=manifest.tasks,
        target_class=manifest.target_class,
        target_is_attack=manifest.target_is_attack,
        event=event,
        features=dict(raw_features),
        feature_fingerprint=fingerprint,
        provider=attempt.provider,
        model=attempt.model,
        mapping_confidence=confidence,
        latency_s=attempt.latency_s,
        result_source_path=attempt.source_path,
        result_line_number=attempt.line_number,
    )


def _validate_event_has_no_targets(event: CanonicalEvent, manifest_id: str) -> None:
    annotations = {
        "label_raw": event.label_raw,
        "attack_family": event.attack_family,
        "attack_subtype": event.attack_subtype,
    }
    present = [name for name, value in annotations.items() if value not in {None, ""}]
    if present:
        raise FeatureSafetyError(
            f"CanonicalEvent de {manifest_id} contiene targets: {', '.join(present)}"
        )
    predictive_containers = {
        "telemetry": event.telemetry,
        "host": event.host,
        "service_context": event.service_context,
        "host_context": event.host_context,
        "telemetry_context": event.telemetry_context,
    }
    leaks: list[str] = []
    for name, value in predictive_containers.items():
        for item in _find_target_keys(value):
            path = f"{name}.{item}"
            if path.casefold() not in _ALLOWED_CANONICAL_TARGET_LIKE_PATHS:
                leaks.append(path)
    for group, columns in (event.feature_groups or {}).items():
        group_path = f"feature_groups.{group}"
        if (
            is_predictive_target_field(group)
            and group_path.casefold() not in _ALLOWED_CANONICAL_TARGET_LIKE_PATHS
        ):
            leaks.append(group_path)
        for column in columns or []:
            column_path = f"{group_path}.{column}"
            if (
                is_predictive_target_field(column)
                and column_path.casefold() not in _ALLOWED_CANONICAL_TARGET_LIKE_PATHS
            ):
                leaks.append(column_path)
    for column in event.evidence_fields or []:
        path = f"evidence_fields.{column}"
        if (
            is_predictive_target_field(column)
            and path.casefold() not in _ALLOWED_CANONICAL_TARGET_LIKE_PATHS
        ):
            leaks.append(path)
    if leaks:
        raise FeatureSafetyError(
            f"CanonicalEvent de {manifest_id} contiene target/leak: {', '.join(leaks[:5])}"
        )


def _quality_report(
    manifests: ManifestIndex,
    results: ResultIndex,
    joined: tuple[PreparedRecord, ...],
    provider_mismatches: set[str],
    model_mismatches: set[str],
    orphan_count: int,
) -> QualityReport:
    selected = {record.manifest_id: record for record in joined}
    datasets = sorted({record.dataset for record in manifests.records})

    def calculate(rows: Sequence[ManifestRecord]) -> DatasetQualityMetrics:
        total = len(rows)
        ids = {row.manifest_id for row in rows}
        latest = [results.latest_attempt[item] for item in ids if item in results.latest_attempt]
        chosen = [selected[item] for item in ids if item in selected]
        latest_ok = sum(attempt.is_valid_success for attempt in latest)
        latest_errors = sum(not attempt.ok for attempt in latest)
        latest_fallbacks = sum(attempt.is_valid_success and not attempt.parsed_by_llm for attempt in latest)
        valid_llm = sum(item in results.latest_valid_llm for item in ids)
        confidences = [record.mapping_confidence for record in chosen]
        latencies = [record.latency_s for record in chosen if record.latency_s is not None]
        feature_counts = [len(record.features) for record in chosen]
        return DatasetQualityMetrics(
            manifest_rows=total,
            attempted_rows=len(latest),
            latest_ok_rows=latest_ok,
            latest_error_rows=latest_errors,
            latest_fallback_rows=latest_fallbacks,
            valid_llm_rows=valid_llm,
            selected_rows=len(chosen),
            missing_rows=total - len(chosen) - len(ids & provider_mismatches) - len(ids & model_mismatches),
            provider_mismatch_rows=len(ids & provider_mismatches),
            model_mismatch_rows=len(ids & model_mismatches),
            parse_ok_rate=_ratio(latest_ok, total),
            llm_success_rate=_ratio(valid_llm, total),
            fallback_rate=_ratio(latest_fallbacks, total),
            selected_coverage_rate=_ratio(len(chosen), total),
            avg_mapping_confidence=fmean(confidences) if confidences else None,
            avg_latency_s=fmean(latencies) if latencies else None,
            avg_feature_count=fmean(feature_counts) if feature_counts else None,
            min_feature_count=min(feature_counts) if feature_counts else None,
            max_feature_count=max(feature_counts) if feature_counts else None,
            split_counts=dict(sorted(Counter(row.split for row in rows).items())),
            task_counts=dict(sorted(Counter(task for row in rows for task in row.tasks).items())),
            schema_profile_counts=dict(sorted(Counter(
                str(record.event.schema_profile or "unknown") for record in chosen
            ).items())),
            modality_counts=dict(sorted(Counter(record.event.modality for record in chosen).items())),
        )

    by_dataset = {
        dataset: calculate(tuple(row for row in manifests.records if row.dataset == dataset))
        for dataset in datasets
    }
    retried = sum(count > 1 for count in results.attempts_per_manifest.values())
    return QualityReport(
        global_metrics=calculate(manifests.records),
        by_dataset=by_dataset,
        result_attempts=results.total_attempts,
        retried_manifest_rows=retried,
        orphan_result_rows=orphan_count,
        truncated_final_lines=results.truncated_final_lines,
    )


def _iter_jsonl(
    path: Path,
    *,
    tolerate_truncated_final: bool,
    state: _JsonlState | None = None,
) -> Iterator[tuple[int, Any]]:
    if not path.is_file():
        raise JsonlFormatError(f"No existe el JSONL: {path}")
    read_state = state or _JsonlState()
    with path.open("rb") as handle:
        previous: tuple[int, bytes] | None = None
        for line_number, raw_line in enumerate(handle, 1):
            if previous is not None:
                parsed = _parse_jsonl_line(previous[1], path, previous[0], allow_truncated=False)
                if parsed is not None:
                    yield previous[0], parsed
            previous = (line_number, raw_line)
        if previous is None:
            return
        line_number, raw_line = previous
        allow_truncated = tolerate_truncated_final and not raw_line.endswith((b"\n", b"\r"))
        try:
            parsed = _parse_jsonl_line(raw_line, path, line_number, allow_truncated=False)
        except JsonlFormatError:
            if allow_truncated:
                read_state.truncated_final_line = True
                return
            raise
        if parsed is not None:
            yield line_number, parsed


def _parse_jsonl_line(
    raw_line: bytes,
    path: Path,
    line_number: int,
    *,
    allow_truncated: bool,
) -> Any | None:
    del allow_truncated  # la decision de tolerancia pertenece a _iter_jsonl
    if not raw_line.strip():
        return None
    try:
        encoding = "utf-8-sig" if line_number == 1 else "utf-8"
        text = raw_line.decode(encoding)
        return json.loads(text, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise JsonlFormatError(f"{_where(path, line_number)}: JSON/UTF-8 invalido: {exc}") from exc


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"constante JSON no finita {value}")


def _paths(
    values: str | Path | Iterable[str | Path],
    *,
    allow_empty: bool,
) -> tuple[Path, ...]:
    if isinstance(values, (str, Path)):
        paths = (Path(values),)
    else:
        paths = tuple(Path(value) for value in values)
    if not paths and not allow_empty:
        raise ValidationCampaignError("Se requiere al menos un fichero")
    return tuple(sorted(paths, key=lambda path: str(path.resolve()).casefold()))


def _required_string(
    payload: Mapping[str, Any],
    key: str,
    error_type: type[ValidationCampaignError],
    location: str,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise error_type(f"{location}: {key} debe ser un string no vacio")
    return value.strip()


def _optional_string(value: Any, key: str, location: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ResultContractError(f"{location}: {key} debe ser str o null")
    stripped = value.strip()
    return stripped or None


def _optional_finite_number(value: Any, key: str, location: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultContractError(f"{location}: {key} debe ser numerico o null")
    number = float(value)
    if not math.isfinite(number):
        raise ResultContractError(f"{location}: {key} contiene NaN/Inf")
    return number


def _find_target_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key)
            path = f"{prefix}.{name}" if prefix else name
            if is_predictive_target_field(name):
                found.append(path)
            found.extend(_find_target_keys(item, path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_find_target_keys(item, f"{prefix}[{index}]"))
    return found


def _is_forbidden_feature(key: str) -> bool:
    lowered = key.casefold()
    return (
        is_predictive_target_field(key)
        or lowered in _FORBIDDEN_FEATURE_KEYS
        or lowered.startswith(_FORBIDDEN_FEATURE_PREFIXES)
    )


def _normalize_provider(provider: str | None) -> str | None:
    if provider is None:
        return None
    normalized = provider.strip().casefold()
    return normalized or None


def _result_backend_key(
    attempt: ResultAttempt,
) -> tuple[str, str | None, str | None]:
    return (
        attempt.manifest_id,
        _normalize_provider(attempt.provider),
        attempt.model,
    )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _where(path: Path, line_number: int) -> str:
    return f"{path}:{line_number}"
