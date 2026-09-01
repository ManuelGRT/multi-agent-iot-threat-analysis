"""Sanitizacion anti-leakage para preparar datos de evaluacion/entrenamiento.

Este modulo no forma parte del grafo multiagente ni de la inferencia. Las
entradas operacionales se consideran ya preparadas; el runtime valida sus
contratos, pero nunca elimina targets silenciosamente.
"""
from __future__ import annotations

import json
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
    "family_hint",
    "ground_truth",
    "groundtruth",
    "is_attack",
    "label",
    "label_id",
    "label_raw",
    "label_hint",
    "malicious",
    "malware",
    "outcome",
    "prediction",
    "predicted_class",
    "classification",
    "subcategory",
    "subtype",
    "target",
    "target_class",
    "target_value",
    "threat",
    "threat_family",
    "threat_label",
    "threat_type",
    "risk_class",
    "risk_family",
    "risk_label",
    "risk_type",
    "type",
    "type_attack",
    "class_id",
    "y",
    "y_true",
}

# Valores completos que identifican de forma inequivoca una clase predictiva.
# No se buscan subcadenas: un payload tecnico que mencione "password" o "xss"
# dentro de un texto mas largo se conserva. La lista se usa solo en contextos
# derivados del evento.
PREDICTIVE_TARGET_VALUES = frozenset(
    {
        "attack",
        "backdoor",
        "benign",
        "botnet",
        "brute_force",
        "bruteforce",
        "command_and_control",
        "c2",
        "c_c",
        "c_c_filedownload",
        "c_c_torii",
        "ddos",
        "ddos_http",
        "ddos_icmp",
        "ddos_tcp",
        "ddos_udp",
        "dos",
        "exfiltration",
        "fingerprinting",
        "filedownload",
        "injection",
        "malicious",
        "malware",
        "man_in_the_middle",
        "mirai",
        "mitm",
        "part_of_a_horizontal_port_scan",
        "partofahorizontalportscan",
        "password",
        "port_scanning",
        "ransomware",
        "reconnaissance",
        "scanning",
        "sql_injection",
        "theft",
        "unknown_attack",
        "uploading",
        "vulnerability_scanner",
        "xss",
    }
)

_TARGET_KEY_PATTERN = (
    r"(?:attack(?:[-_](?:cat|category|family|label|subtype|type))?|type[-_]attack|"
    r"detailed[-_]label|det_label|label(?:[-_](?:raw|id))?|target(?:[-_](?:class|value))?|"
    r"ground[-_]?truth|category|subcategory|class(?:[-_]id)?|family|subtype|outcome|"
    r"(?:class|family|label|category)[-_]hint|classification|prediction|predicted[-_]class|"
    r"risk[-_](?:class|family|label|type)|"
    r"y(?:[-_]true)?|is[-_]attack|malicious|malware|"
    r"threat(?:[-_](?:family|label|type))?|type)"
)
_TARGET_ASSIGNMENT_RE = re.compile(
    rf"(?i)(?<![A-Za-z0-9_.-])(?P<key>[\"']?{_TARGET_KEY_PATTERN}[\"']?)\s*[:=]\s*"
)
_NEXT_ASSIGNMENT_RE = re.compile(
    r"\s+[\"']?[A-Za-z0-9_.-]+[\"']?\s*[:=]"
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
    # Si la cadena contiene JSON, sanear su estructura evita que claves como
    # ``"label"`` eludan el patron textual y conserva el resto del registro.
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            pass
        else:
            if isinstance(parsed, (dict, list)):
                cleaned = scrub_predictive_payload(parsed)
                if cleaned == parsed:
                    return text
                return json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))

    scrubbed = _remove_target_assignments(text)
    scrubbed = re.sub(r"\s+", " ", scrubbed)
    scrubbed = re.sub(r"\s+\|", " |", scrubbed)
    scrubbed = re.sub(r"\|\s+\|", "|", scrubbed)
    scrubbed = re.sub(r"([,{[])\s*[,;]", r"\1", scrubbed)
    scrubbed = re.sub(r"[,;]\s*([}\]])", r"\1", scrubbed)
    scrubbed = re.sub(r"([,;])\s*\1+", r"\1", scrubbed)
    scrubbed = re.sub(r"\?&+", "?", scrubbed)
    scrubbed = re.sub(r"&&+", "&", scrubbed)
    return scrubbed.strip(" |,;&")


def _remove_target_assignments(text: str) -> str:
    pieces: list[str] = []
    cursor = 0
    while True:
        match = _TARGET_ASSIGNMENT_RE.search(text, cursor)
        if match is None:
            pieces.append(text[cursor:])
            break
        pieces.append(text[cursor : match.start()])
        cursor = _assignment_value_end(text, match.end())
    return "".join(pieces)


def _assignment_value_end(text: str, start: int) -> int:
    if start >= len(text):
        return start
    first = text[start]
    if first in {'"', "'"}:
        return _quoted_value_end(text, start, first)
    if first in "[{":
        return _balanced_value_end(text, start)

    index = start
    while index < len(text):
        char = text[index]
        if char in ",;|&)}]>":
            return index
        if char.isspace() and _NEXT_ASSIGNMENT_RE.match(text, index):
            return index
        index += 1
    return len(text)


def _quoted_value_end(text: str, start: int, quote: str) -> int:
    escaped = False
    for index in range(start + 1, len(text)):
        char = text[index]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == quote:
            return index + 1
    return len(text)


def _balanced_value_end(text: str, start: int) -> int:
    pairs = {"[": "]", "{": "}"}
    stack = [pairs[text[start]]]
    quote: str | None = None
    escaped = False
    for index in range(start + 1, len(text)):
        char = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
        elif char in pairs:
            stack.append(pairs[char])
        elif stack and char == stack[-1]:
            stack.pop()
            if not stack:
                return index + 1
    return len(text)


def is_predictive_target_field(key: Any) -> bool:
    return _is_target_field(key)


def is_predictive_target_value(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = re.sub(
        r"[^a-z0-9]+",
        "_",
        value.strip().casefold(),
    ).strip("_")
    return normalized in PREDICTIVE_TARGET_VALUES


def contains_predictive_target_text(text: str | None) -> bool:
    """True si el texto contiene patrones clave=valor de campos target.

    Lo usa el auditor (Fase 5) para detectar leakage en semantic_text sin
    depender del efecto colateral de normalizacion de scrub_predictive_text.
    """
    return bool(_TARGET_ASSIGNMENT_RE.search(text or ""))


def _is_target_field(key: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(key or "").strip().lower()).strip("_")
    tokens = {token for token in normalized.split("_") if token}
    return (
        normalized in TARGET_FIELD_NAMES
        or "attack" in tokens
        or "threat" in tokens
        or "malware" in tokens
        or normalized.startswith("attack_")
        or normalized.endswith(("_attack", "_label", "_category", "_subcategory", "_class", "_family", "_target"))
    )
