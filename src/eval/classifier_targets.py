"""Recuperacion auditable de etiquetas auxiliares excluidas de las features.

Bot-IoT necesita ``subcategory`` para distinguir los subtipos DDoS y la
huella del sistema operativo. La campana conserva correctamente ese target
fuera del evento estandarizado, de modo que se recupera del CSV autoritativo
mediante ``source_file`` y el ``row_id`` original (indice base cero).
"""
from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from src.contracts.attack_taxonomy import normalise_dataset, normalise_taxonomy_token
from src.eval.data_sanitization import sanitize_llm_input
from src.eval.validation_campaign import PreparedRecord


class ClassifierTargetRecoveryError(ValueError):
    """La etiqueta auxiliar no puede recuperarse de forma inequívoca."""


@dataclass(frozen=True, slots=True)
class MappingContext:
    """Protocolo crudo reservado para construir el target, no una feature LLM."""

    transport_proto: str | None
    app_proto: str | None
    source: str = "raw_manifest_row"


_HISTORICAL_ID_TIME_COLUMNS = frozenset(
    {
        "pkseqid",
        "seq",
        "uid",
        "ts",
        "stime",
        "ltime",
        "time",
        "timestamp",
        "date",
        "frame.time",
        "flow_id",
        "id",
        "node",
    }
)


def recover_bot_iot_target_subclasses(
    records: Iterable[PreparedRecord],
    *,
    repository_root: str | Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Recupera ``subcategory`` sin incorporarla al vector predictivo.

    Solo lee las filas Bot-IoT de ataque cuya clase nativa es ``DDoS`` o
    ``Reconnaissance``. El ``row_id`` sigue exactamente la semantica de
    :func:`scripts.build_validation_manifests.iter_csv`: primera fila de datos
    igual a cero.
    """

    root = Path(repository_root).expanduser().resolve()
    requested: dict[Path, dict[int, list[PreparedRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        if normalise_dataset(record.dataset) != "bot_iot":
            continue
        if "multiclass" not in record.tasks or record.target_is_attack is not True:
            continue
        if normalise_taxonomy_token(record.target_class) not in {
            "ddos",
            "reconnaissance",
        }:
            continue
        row_index = _row_index(record)
        source = _safe_source_path(root, record.source_file)
        requested[source][row_index].append(record)

    recovered: dict[str, str] = {}
    source_report: dict[str, Any] = {}
    csv.field_size_limit(10_000_000)
    for source, rows_by_index in sorted(requested.items(), key=lambda item: str(item[0])):
        if not source.is_file():
            raise ClassifierTargetRecoveryError(f"CSV Bot-IoT inexistente: {source}")
        pending = set(rows_by_index)
        maximum = max(pending)
        with source.open(encoding="utf-8-sig", errors="replace", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or ())
            missing_columns = {"category", "subcategory"} - fields
            if missing_columns:
                raise ClassifierTargetRecoveryError(
                    f"{source}: faltan columnas target {sorted(missing_columns)}"
                )
            for row_index, raw in enumerate(reader):
                if row_index > maximum or not pending:
                    break
                if row_index not in pending:
                    continue
                category = str(raw.get("category") or "").strip()
                subclass = str(raw.get("subcategory") or "").strip()
                if not subclass:
                    raise ClassifierTargetRecoveryError(
                        f"{source}:{row_index}: subcategory vacia"
                    )
                for record in rows_by_index[row_index]:
                    observed_hash = bot_iot_manifest_content_hash(raw)
                    if observed_hash != record.content_hash:
                        raise ClassifierTargetRecoveryError(
                            f"{record.manifest_id}: la fila cruda no coincide con "
                            "content_hash del manifiesto"
                        )
                    if normalise_taxonomy_token(category) != normalise_taxonomy_token(
                        record.target_class
                    ):
                        raise ClassifierTargetRecoveryError(
                            f"{record.manifest_id}: category cruda {category!r} no "
                            f"coincide con target_class {record.target_class!r}"
                        )
                    recovered[record.manifest_id] = subclass
                pending.remove(row_index)
        if pending:
            sample = sorted(pending)[:5]
            raise ClassifierTargetRecoveryError(
                f"{source}: no se encontraron row_id {sample}"
            )
        relative = source.relative_to(root).as_posix()
        source_report[relative] = {
            "requested_rows": sum(len(values) for values in rows_by_index.values()),
            "minimum_row_id": min(rows_by_index),
            "maximum_row_id": max(rows_by_index),
            "bytes": source.stat().st_size,
            "sha256": _file_sha256(source),
        }

    payload = [
        {"manifest_id": manifest_id, "native_subclass": subclass}
        for manifest_id, subclass in sorted(recovered.items())
    ]
    return recovered, {
        "policy": "raw_target_only_by_source_file_and_zero_based_row_id",
        "feature_use": False,
        "recovered_rows": len(recovered),
        "sources": source_report,
        "mapping_sha256": _json_sha256(payload),
        "content_hash_verified_rows": len(recovered),
    }


def recover_classifier_mapping_context(
    records: Iterable[PreparedRecord],
    *,
    manifest_paths: Iterable[str | Path],
) -> tuple[dict[str, MappingContext], dict[str, Any]]:
    """Recupera ``proto`` y ``service`` crudos para DDoS genérico.

    Solo se solicitan ataques DDoS de TON-IoT e IoT-23 destinados a la tarea
    multiclase. El identificador y ``content_hash`` se cotejan contra el
    ``PreparedRecord`` ya validado. No se consulta el evento canónico.
    """

    requested: dict[str, PreparedRecord] = {}
    for record in records:
        if "multiclass" not in record.tasks or record.target_is_attack is not True:
            continue
        if normalise_dataset(record.dataset) not in {"ton_iot", "iot23"}:
            continue
        if normalise_taxonomy_token(record.target_class) != "ddos":
            continue
        if record.manifest_id in requested:
            raise ClassifierTargetRecoveryError(
                f"manifest_id duplicado en registros preparados: {record.manifest_id}"
            )
        requested[record.manifest_id] = record

    recovered: dict[str, MappingContext] = {}
    inventories: list[dict[str, Any]] = []
    paths = sorted(
        (Path(value).expanduser().resolve() for value in manifest_paths), key=str
    )
    for path in paths:
        if not path.is_file():
            raise ClassifierTargetRecoveryError(f"Manifiesto inexistente: {path}")
        matched = 0
        with path.open(encoding="utf-8", errors="strict") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ClassifierTargetRecoveryError(
                        f"{path}:{line_number}: JSON inválido"
                    ) from exc
                manifest_id = str(payload.get("manifest_id") or "")
                expected = requested.get(manifest_id)
                if expected is None:
                    continue
                if manifest_id in recovered:
                    raise ClassifierTargetRecoveryError(
                        f"manifest_id repetido entre manifiestos: {manifest_id}"
                    )
                if str(payload.get("content_hash") or "") != expected.content_hash:
                    raise ClassifierTargetRecoveryError(
                        f"{manifest_id}: content_hash del manifiesto no coincide"
                    )
                if normalise_dataset(payload.get("dataset")) != normalise_dataset(
                    expected.dataset
                ):
                    raise ClassifierTargetRecoveryError(
                        f"{manifest_id}: dataset del manifiesto no coincide"
                    )
                row = payload.get("row")
                if not isinstance(row, dict):
                    raise ClassifierTargetRecoveryError(
                        f"{manifest_id}: row ausente o inválida en manifiesto"
                    )
                recovered[manifest_id] = MappingContext(
                    transport_proto=_raw_optional(row, "proto", "protocol"),
                    app_proto=_raw_optional(row, "service", "app_proto"),
                )
                matched += 1
        inventories.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
                "matched_rows": matched,
            }
        )

    missing = sorted(set(requested) - set(recovered))
    if missing:
        raise ClassifierTargetRecoveryError(
            f"No se recuperó contexto para {len(missing)} filas; muestra={missing[:5]}"
        )

    protocol_counts: dict[str, int] = defaultdict(int)
    service_counts: dict[str, int] = defaultdict(int)
    for context in recovered.values():
        protocol_counts[context.transport_proto or "<missing>"] += 1
        service_counts[context.app_proto or "<missing>"] += 1
    return recovered, {
        "policy": "raw_manifest_context_only_not_llm_output",
        "feature_use": False,
        "requested_rows": len(requested),
        "recovered_rows": len(recovered),
        "protocol_counts": dict(sorted(protocol_counts.items())),
        "service_counts": dict(sorted(service_counts.items())),
        "manifests": inventories,
    }


def _raw_optional(row: dict[str, Any], *names: str) -> str | None:
    folded = {str(key).strip().casefold(): value for key, value in row.items()}
    for name in names:
        value = folded.get(name.casefold())
        if value is None:
            continue
        text = str(value).strip()
        if text and text.casefold() not in {"-", "none", "null", "nan", "n/a"}:
            return text
    return None


def bot_iot_manifest_content_hash(raw_row: dict[str, Any]) -> str:
    """Reproduce el hash de 16 bytes del constructor histórico de manifiestos."""

    normalized = {str(key): (value or "") for key, value in raw_row.items() if key}
    prepared = sanitize_llm_input({"row": normalized})["row"]
    payload = "|".join(
        f"{key}={value}"
        for key, value in sorted(prepared.items())
        if key.strip().casefold() not in _HISTORICAL_ID_TIME_COLUMNS
    )
    return hashlib.sha256(payload.encode("utf-8")).digest()[:16].hex()


def _row_index(record: PreparedRecord) -> int:
    value = record.row_id
    if isinstance(value, bool):
        raise ClassifierTargetRecoveryError(
            f"{record.manifest_id}: row_id booleano invalido"
        )
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise ClassifierTargetRecoveryError(
            f"{record.manifest_id}: row_id no numerico {value!r}"
        ) from exc
    if index < 0 or str(index) != str(value).strip():
        raise ClassifierTargetRecoveryError(
            f"{record.manifest_id}: row_id invalido {value!r}"
        )
    return index


def _safe_source_path(root: Path, source_file: str) -> Path:
    raw = Path(str(source_file).replace("\\", "/"))
    if raw.is_absolute():
        raise ClassifierTargetRecoveryError(
            f"source_file debe ser relativo al repositorio: {source_file!r}"
        )
    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ClassifierTargetRecoveryError(
            f"source_file sale del repositorio: {source_file!r}"
        ) from exc
    return candidate


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ClassifierTargetRecoveryError",
    "MappingContext",
    "bot_iot_manifest_content_hash",
    "recover_bot_iot_target_subclasses",
    "recover_classifier_mapping_context",
]
