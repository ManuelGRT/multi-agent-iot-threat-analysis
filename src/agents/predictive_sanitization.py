from __future__ import annotations

import re
from typing import Any


TARGET_FIELD_NAMES = {
    "attack",
    "attack_cat",
    "attack_category",
    "attack_family",
    "attack_label",
    "attack_subtype",
    "attack_type",
    "category",
    "class",
    "detailed-label",
    "detailed_label",
    "det_label",
    "family",
    "ground_truth",
    "groundtruth",
    "is_attack",
    "label",
    "label_id",
    "label_raw",
    "malicious",
    "outcome",
    "subcategory",
    "subtype",
    "target",
    "target_class",
    "target_value",
    "type",
    "type_attack",
    "class_id",
    "y",
    "y_true",
}

_TARGET_KV_RE = re.compile(
    r"(?i)(?:^|[\s|,;])"
    r"(?:attack(?:[-_](?:cat|category|family|label|subtype|type))?|type[-_]attack|"
    r"detailed[-_]label|det_label|label(?:[-_](?:raw|id))?|target(?:[-_](?:class|value))?|"
    r"ground[-_]?truth|category|subcategory|class(?:[-_]id)?|family|subtype|outcome|"
    r"y(?:[-_]true)?|is[-_]attack|malicious|type)"
    r"\s*[:=]\s*[^|,;]+"
)


def scrub_predictive_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: scrub_predictive_payload(item)
            for key, item in value.items()
            if not _is_target_field(key)
        }
    if isinstance(value, list):
        return [scrub_predictive_payload(item) for item in value]
    if isinstance(value, tuple):
        return [scrub_predictive_payload(item) for item in value]
    if isinstance(value, str):
        return scrub_predictive_text(value)
    return value


def scrub_predictive_text(text: str) -> str:
    scrubbed = _TARGET_KV_RE.sub(" ", text)
    scrubbed = re.sub(r"\s+", " ", scrubbed)
    scrubbed = re.sub(r"\s+\|", " |", scrubbed)
    scrubbed = re.sub(r"\|\s+\|", "|", scrubbed)
    return scrubbed.strip(" |")


def is_predictive_target_field(key: Any) -> bool:
    return _is_target_field(key)


def contains_predictive_target_text(text: str | None) -> bool:
    """True si el texto contiene patrones clave=valor de campos target.

    Lo usa el auditor (Fase 5) para detectar leakage en semantic_text sin
    depender del efecto colateral de normalizacion de scrub_predictive_text.
    """
    return bool(_TARGET_KV_RE.search(text or ""))


def _is_target_field(key: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(key or "").strip().lower()).strip("_")
    tokens = {token for token in normalized.split("_") if token}
    return (
        normalized in TARGET_FIELD_NAMES
        or "attack" in tokens
        or normalized.startswith("attack_")
        or normalized.endswith(("_attack", "_label", "_category", "_subcategory", "_class", "_family", "_target"))
    )
