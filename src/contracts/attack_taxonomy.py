"""Taxonomias operativas de tipos de ataque para el clasificador.

El detector binario es responsable de ``Normal``. Por ello, el clasificador
operacional utiliza exclusivamente los catorce tipos de ataque de
Edge-IIoTset empleados en el trabajo de referencia. La familia amplia se
deriva despues de la prediccion para conservar la compatibilidad con el
catalogo de mitigacion. La vista multidataset amplia ese contrato de forma
independiente para cubrir ataques significativos de las demas fuentes.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Literal
import unicodedata


JORGE_TAXONOMY_VERSION = "jorge_edge_attack_type_v1_2026-09-06"
MULTIDATASET_TAXONOMY_VERSION = "multidataset_attack_type_v1_2026-09-06"

JORGE_ATTACK_CLASSES: tuple[str, ...] = (
    "Backdoor",
    "DDoS_HTTP",
    "DDoS_ICMP",
    "DDoS_TCP",
    "DDoS_UDP",
    "Fingerprinting",
    "MITM",
    "Password",
    "Port_Scanning",
    "Ransomware",
    "SQL_injection",
    "Uploading",
    "Vulnerability_scanner",
    "XSS",
)

JORGE_ALL_CLASSES: tuple[str, ...] = ("Normal", *JORGE_ATTACK_CLASSES)

MULTIDATASET_ATTACK_CLASSES: tuple[str, ...] = (
    *JORGE_ATTACK_CLASSES,
    "DoS",
    "Command_and_Control",
)
MULTIDATASET_ALL_CLASSES: tuple[str, ...] = (
    "Normal",
    *MULTIDATASET_ATTACK_CLASSES,
)

SUPPORTED_ATTACK_TAXONOMY_VERSIONS: tuple[str, ...] = (
    JORGE_TAXONOMY_VERSION,
    MULTIDATASET_TAXONOMY_VERSION,
)

JORGE_TO_BROAD_FAMILY: dict[str, str] = {
    "Backdoor": "malware",
    "DDoS_HTTP": "ddos",
    "DDoS_ICMP": "ddos",
    "DDoS_TCP": "ddos",
    "DDoS_UDP": "ddos",
    "Fingerprinting": "scanning",
    "MITM": "mitm",
    "Password": "bruteforce",
    "Port_Scanning": "scanning",
    "Ransomware": "malware",
    "SQL_injection": "injection",
    "Uploading": "injection",
    "Vulnerability_scanner": "scanning",
    "XSS": "injection",
}

MULTIDATASET_TO_BROAD_FAMILY: dict[str, str] = {
    **JORGE_TO_BROAD_FAMILY,
    # El catalogo de inteligencia agrupa DoS y DDoS bajo ``ddos``.
    "DoS": "ddos",
    "Command_and_Control": "botnet",
}

ResolutionStatus = Literal[
    "exact",
    "forced",
    "benign",
    "ambiguous",
    "out_of_taxonomy",
]


@dataclass(frozen=True, slots=True)
class TaxonomyResolution:
    status: ResolutionStatus
    attack_type: str | None
    reason: str


def normalise_taxonomy_token(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def normalise_dataset(value: object) -> str:
    key = normalise_taxonomy_token(value)
    return {
        "edgeiiotset": "edge_iiotset",
        "edge_iiotset": "edge_iiotset",
        "botiot": "bot_iot",
        "bot_iot": "bot_iot",
        "iot_23": "iot23",
        "iot23": "iot23",
        "toniot": "ton_iot",
        "ton_iot": "ton_iot",
        "urban_iot": "urban_iot",
        "urbaniot": "urban_iot",
    }.get(key, key)


_EDGE_DIRECT = {
    normalise_taxonomy_token(label): label for label in JORGE_ATTACK_CLASSES
}

_TON_DIRECT = {
    "backdoor": "Backdoor",
    "mitm": "MITM",
    "password": "Password",
    "ransomware": "Ransomware",
    "xss": "XSS",
}

_BOT_DDOS_SUBTYPES = {
    "http": "DDoS_HTTP",
    "tcp": "DDoS_TCP",
    "udp": "DDoS_UDP",
}

_BOT_RECON_SUBTYPES = {"os_fingerprint": "Fingerprinting"}
_BOT_COMPOSITE_DEFAULTS = {
    "ddos": {
        "status": "ambiguous",
        "reason": "bot_ddos_requires_target_subcategory",
    },
    "reconnaissance": {
        "status": "ambiguous",
        "reason": "bot_reconnaissance_requires_target_subcategory",
    },
}
_BOT_RECON_SPECIAL_REJECTIONS = {
    "service_scan": {
        "status": "ambiguous",
        "reason": "service_scan_is_not_strictly_port_scanning",
    }
}
_IOT23_DIRECT = {"partofahorizontalportscan": "Port_Scanning"}


def _resolve_forced_ddos(
    transport_proto: object | None,
    app_proto: object | None,
    *,
    reason_prefix: str,
) -> TaxonomyResolution:
    """Resuelve DDoS generico solo con protocolo crudo verificable.

    ``app_proto`` representa el campo ``service`` de la fila original. No se
    debe pasar aqui el protocolo inferido por el estandarizador LLM.
    """

    service_key = normalise_taxonomy_token(app_proto)
    transport_key = normalise_taxonomy_token(transport_proto)
    if service_key in {"http", "http_alt", "http_proxy"}:
        return TaxonomyResolution(
            "forced", "DDoS_HTTP", f"{reason_prefix}_raw_service_http"
        )
    if transport_key.startswith("icmp"):
        return TaxonomyResolution(
            "forced", "DDoS_ICMP", f"{reason_prefix}_raw_protocol_icmp"
        )
    if transport_key == "udp":
        return TaxonomyResolution(
            "forced", "DDoS_UDP", f"{reason_prefix}_raw_protocol_udp"
        )
    if transport_key == "tcp":
        return TaxonomyResolution(
            "forced", "DDoS_TCP", f"{reason_prefix}_raw_protocol_tcp"
        )
    return TaxonomyResolution(
        "ambiguous", None, f"{reason_prefix}_missing_raw_protocol"
    )


def resolve_jorge_class(
    dataset: object,
    native_class: object,
    native_subclass: object | None = None,
) -> TaxonomyResolution:
    """Resuelve solo equivalencias defendibles sin mirar features predictivas."""

    dataset_key = normalise_dataset(dataset)
    class_key = normalise_taxonomy_token(native_class)
    subclass_key = normalise_taxonomy_token(native_subclass)

    if class_key in {"normal", "benign", "benigno"}:
        return TaxonomyResolution("benign", None, "handled_by_binary_detector")

    if dataset_key == "edge_iiotset":
        label = _EDGE_DIRECT.get(class_key)
        if label is not None:
            return TaxonomyResolution("exact", label, "edge_native_attack_type")
        return TaxonomyResolution("out_of_taxonomy", None, "edge_label_not_in_jorge")

    if dataset_key == "ton_iot":
        label = _TON_DIRECT.get(class_key)
        if label is not None:
            return TaxonomyResolution("exact", label, "ton_native_exact_label")
        if class_key in {"ddos", "injection", "scanning"}:
            return TaxonomyResolution(
                "ambiguous", None, f"ton_{class_key}_does_not_identify_jorge_subtype"
            )
        if class_key == "dos":
            return TaxonomyResolution("out_of_taxonomy", None, "dos_is_not_ddos")
        return TaxonomyResolution("out_of_taxonomy", None, "ton_label_not_in_jorge")

    if dataset_key == "bot_iot":
        if class_key == "ddos":
            label = _BOT_DDOS_SUBTYPES.get(subclass_key)
            if label is not None:
                return TaxonomyResolution("exact", label, "bot_target_subcategory")
            default = _BOT_COMPOSITE_DEFAULTS["ddos"]
            return TaxonomyResolution(
                default["status"], None, default["reason"]
            )
        if class_key == "reconnaissance":
            label = _BOT_RECON_SUBTYPES.get(subclass_key)
            if label is not None:
                return TaxonomyResolution(
                    "exact", label, "bot_target_os_fingerprint"
                )
            special = _BOT_RECON_SPECIAL_REJECTIONS.get(subclass_key)
            if special is not None:
                return TaxonomyResolution(
                    special["status"], None, special["reason"]
                )
            default = _BOT_COMPOSITE_DEFAULTS["reconnaissance"]
            return TaxonomyResolution(
                default["status"], None, default["reason"]
            )
        if class_key == "dos":
            return TaxonomyResolution("out_of_taxonomy", None, "dos_is_not_ddos")
        if class_key == "theft":
            return TaxonomyResolution(
                "out_of_taxonomy", None, "theft_is_not_a_jorge_attack_type"
            )
        return TaxonomyResolution("out_of_taxonomy", None, "bot_label_not_in_jorge")

    if dataset_key == "iot23":
        label = _IOT23_DIRECT.get(class_key)
        if label is not None:
            return TaxonomyResolution(
                "exact", label, "iot23_horizontal_port_scan"
            )
        if class_key == "ddos":
            return TaxonomyResolution(
                "ambiguous", None, "iot23_ddos_does_not_identify_protocol_subtype"
            )
        return TaxonomyResolution("out_of_taxonomy", None, "iot23_label_not_in_jorge")

    return TaxonomyResolution("out_of_taxonomy", None, "dataset_not_supported")


def resolve_multidataset_class(
    dataset: object,
    native_class: object,
    native_subclass: object | None = None,
    *,
    transport_proto: object | None = None,
    app_proto: object | None = None,
) -> TaxonomyResolution:
    """Resuelve la taxonomia ampliada sin crear clases ``*_unspecified``.

    Los estados ``forced`` identifican equivalencias aproximadas o targets
    derivados del protocolo crudo. Se aceptan para entrenar el experimento
    ampliado, pero deben poder evaluarse por separado de los targets nativos.
    """

    dataset_key = normalise_dataset(dataset)
    class_key = normalise_taxonomy_token(native_class)
    subclass_key = normalise_taxonomy_token(native_subclass)

    if class_key in {"normal", "benign", "benigno"}:
        return TaxonomyResolution("benign", None, "handled_by_binary_detector")

    if dataset_key == "edge_iiotset":
        label = _EDGE_DIRECT.get(class_key)
        if label is not None:
            return TaxonomyResolution("exact", label, "edge_native_attack_type")
        return TaxonomyResolution(
            "out_of_taxonomy", None, "edge_label_not_in_multidataset_taxonomy"
        )

    if dataset_key == "bot_iot":
        if class_key == "ddos":
            label = _BOT_DDOS_SUBTYPES.get(subclass_key)
            if label is not None:
                return TaxonomyResolution("exact", label, "bot_target_subcategory")
            return TaxonomyResolution(
                "ambiguous", None, "bot_ddos_requires_target_subcategory"
            )
        if class_key == "dos":
            return TaxonomyResolution("exact", "DoS", "bot_native_dos_coarsened")
        if class_key == "reconnaissance":
            if subclass_key == "os_fingerprint":
                return TaxonomyResolution(
                    "exact", "Fingerprinting", "bot_target_os_fingerprint"
                )
            if subclass_key == "service_scan":
                return TaxonomyResolution(
                    "forced",
                    "Port_Scanning",
                    "bot_service_scan_forced_to_port_scanning",
                )
            return TaxonomyResolution(
                "ambiguous", None, "bot_reconnaissance_requires_target_subcategory"
            )
        if class_key == "theft":
            return TaxonomyResolution(
                "out_of_taxonomy", None, "bot_theft_insufficient_support"
            )
        return TaxonomyResolution(
            "out_of_taxonomy", None, "bot_label_not_in_multidataset_taxonomy"
        )

    if dataset_key == "iot23":
        label = _IOT23_DIRECT.get(class_key)
        if label is not None:
            return TaxonomyResolution("exact", label, "iot23_horizontal_port_scan")
        if class_key in {"c_c", "c_c_filedownload", "c_c_torii"}:
            return TaxonomyResolution(
                "exact",
                "Command_and_Control",
                "iot23_native_command_and_control",
            )
        if class_key == "ddos":
            return _resolve_forced_ddos(
                transport_proto,
                app_proto,
                reason_prefix="iot23_ddos",
            )
        if class_key == "filedownload":
            return TaxonomyResolution(
                "out_of_taxonomy", None, "iot23_filedownload_insufficient_support"
            )
        return TaxonomyResolution(
            "out_of_taxonomy", None, "iot23_label_not_in_multidataset_taxonomy"
        )

    if dataset_key == "ton_iot":
        label = _TON_DIRECT.get(class_key)
        if label is not None:
            return TaxonomyResolution("exact", label, "ton_native_exact_label")
        if class_key == "ddos":
            return _resolve_forced_ddos(
                transport_proto,
                app_proto,
                reason_prefix="ton_ddos",
            )
        if class_key == "dos":
            return TaxonomyResolution("exact", "DoS", "ton_native_dos_coarsened")
        if class_key == "injection":
            return TaxonomyResolution(
                "forced",
                "SQL_injection",
                "ton_injection_forced_to_sql_injection",
            )
        if class_key == "scanning":
            return TaxonomyResolution(
                "forced",
                "Port_Scanning",
                "ton_scanning_forced_to_port_scanning",
            )
        return TaxonomyResolution(
            "out_of_taxonomy", None, "ton_label_not_in_multidataset_taxonomy"
        )

    return TaxonomyResolution("out_of_taxonomy", None, "dataset_not_supported")


def broad_family_for_attack_type(attack_type: str) -> str:
    try:
        return MULTIDATASET_TO_BROAD_FAMILY[str(attack_type)]
    except KeyError as exc:
        raise ValueError(f"Tipo de ataque fuera de la taxonomia: {attack_type!r}") from exc


def attack_classes_for_taxonomy(version: object) -> tuple[str, ...]:
    """Devuelve el espacio cerrado de clases de una version soportada.

    El chequeo deliberadamente no acepta alias ni versiones ausentes: la
    version forma parte del contrato persistido del modelo y debe coincidir de
    manera exacta durante la inferencia y la auditoria.
    """

    if version == JORGE_TAXONOMY_VERSION:
        return JORGE_ATTACK_CLASSES
    if version == MULTIDATASET_TAXONOMY_VERSION:
        return MULTIDATASET_ATTACK_CLASSES
    raise ValueError(f"Version de taxonomia de ataque no soportada: {version!r}")


def taxonomy_report() -> dict[str, object]:
    native_rules = {
        "edge_iiotset": {
            "direct_by_native_class": dict(sorted(_EDGE_DIRECT.items())),
            "otherwise": "out_of_taxonomy",
        },
        "ton_iot": {
            "direct_by_native_class": dict(sorted(_TON_DIRECT.items())),
            "ambiguous_native_classes": ["ddos", "injection", "scanning"],
            "dos_policy": "out_of_taxonomy_not_ddos",
            "otherwise": "out_of_taxonomy",
        },
        "bot_iot": {
            "composite_by_native_class": {
                "ddos": {
                    "selector": "target_subcategory",
                    "exact_by_selector": dict(
                        sorted(_BOT_DDOS_SUBTYPES.items())
                    ),
                    "unmapped_or_missing_selector": dict(
                        _BOT_COMPOSITE_DEFAULTS["ddos"]
                    ),
                },
                "reconnaissance": {
                    "selector": "target_subcategory",
                    "exact_by_selector": dict(
                        sorted(_BOT_RECON_SUBTYPES.items())
                    ),
                    "special_rejections": {
                        name: dict(rule)
                        for name, rule in sorted(
                            _BOT_RECON_SPECIAL_REJECTIONS.items()
                        )
                    },
                    "unmapped_or_missing_selector": dict(
                        _BOT_COMPOSITE_DEFAULTS["reconnaissance"]
                    ),
                },
            },
            "native_class_rejections": {
                "dos": {
                    "status": "out_of_taxonomy",
                    "reason": "dos_is_not_ddos",
                },
                "theft": {
                    "status": "out_of_taxonomy",
                    "reason": "theft_is_not_a_jorge_attack_type",
                },
            },
            "unmapped_or_missing_native_class": {
                "status": "out_of_taxonomy",
                "reason": "bot_label_not_in_jorge",
            },
        },
        "iot23": {
            "direct_by_native_class": dict(sorted(_IOT23_DIRECT.items())),
            "ambiguous_native_classes": ["ddos"],
            "otherwise": "out_of_taxonomy",
        },
    }
    core: dict[str, object] = {
        "version": JORGE_TAXONOMY_VERSION,
        "attack_classes": list(JORGE_ATTACK_CLASSES),
        "all_classes": list(JORGE_ALL_CLASSES),
        "broad_families": dict(sorted(JORGE_TO_BROAD_FAMILY.items())),
        "normal_policy": "binary_detector_only",
        "native_rules": native_rules,
    }
    encoded = json.dumps(
        core, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {**core, "sha256": hashlib.sha256(encoded).hexdigest()}


def multidataset_taxonomy_report() -> dict[str, object]:
    """Contrato versionado de la taxonomia ampliada y sus mapeos forzados."""

    core: dict[str, object] = {
        "version": MULTIDATASET_TAXONOMY_VERSION,
        "reference_taxonomy_version": JORGE_TAXONOMY_VERSION,
        "attack_classes": list(MULTIDATASET_ATTACK_CLASSES),
        "all_classes": list(MULTIDATASET_ALL_CLASSES),
        "broad_families": dict(sorted(MULTIDATASET_TO_BROAD_FAMILY.items())),
        "normal_policy": "binary_detector_only",
        "mapping_statuses_used_for_training": ["exact", "forced"],
        "forced_mapping_policy": {
            "generic_ddos": {
                "selector_source": "raw_manifest_row_only",
                "precedence": [
                    "service_http_to_DDoS_HTTP",
                    "protocol_icmp_to_DDoS_ICMP",
                    "protocol_udp_to_DDoS_UDP",
                    "protocol_tcp_to_DDoS_TCP",
                ],
                "port_inference": False,
                "llm_inference": False,
                "missing_protocol": "ambiguous_excluded_from_subtype_training",
            },
            "ton_injection": {
                "target": "SQL_injection",
                "quality": "coarse_forced",
            },
            "ton_scanning": {
                "target": "Port_Scanning",
                "quality": "scenario_level_forced",
            },
            "bot_service_scan": {
                "target": "Port_Scanning",
                "quality": "scenario_level_forced",
            },
        },
        "coarsening_policy": {
            "bot_dos": "DoS",
            "ton_dos": "DoS",
            "iot23_c_and_c_variants": "Command_and_Control",
        },
        "excluded_labels": {
            "bot_iot_theft": "insufficient_distinct_support",
            "iot23_filedownload": "insufficient_distinct_support",
        },
    }
    encoded = json.dumps(
        core, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {**core, "sha256": hashlib.sha256(encoded).hexdigest()}


__all__ = [
    "JORGE_ALL_CLASSES",
    "JORGE_ATTACK_CLASSES",
    "JORGE_TAXONOMY_VERSION",
    "JORGE_TO_BROAD_FAMILY",
    "MULTIDATASET_ALL_CLASSES",
    "MULTIDATASET_ATTACK_CLASSES",
    "MULTIDATASET_TAXONOMY_VERSION",
    "MULTIDATASET_TO_BROAD_FAMILY",
    "SUPPORTED_ATTACK_TAXONOMY_VERSIONS",
    "TaxonomyResolution",
    "attack_classes_for_taxonomy",
    "broad_family_for_attack_type",
    "multidataset_taxonomy_report",
    "normalise_dataset",
    "normalise_taxonomy_token",
    "resolve_jorge_class",
    "resolve_multidataset_class",
    "taxonomy_report",
]
