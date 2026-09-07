from __future__ import annotations

import pytest

from src.contracts.attack_taxonomy import (
    JORGE_ALL_CLASSES,
    JORGE_ATTACK_CLASSES,
    MULTIDATASET_ATTACK_CLASSES,
    MULTIDATASET_TAXONOMY_VERSION,
    attack_classes_for_taxonomy,
    multidataset_taxonomy_report,
    resolve_jorge_class,
    resolve_multidataset_class,
    taxonomy_report,
)


def test_jorge_taxonomy_has_14_attacks_and_normal_only_in_reference_view():
    assert len(JORGE_ATTACK_CLASSES) == 14
    assert len(set(JORGE_ATTACK_CLASSES)) == 14
    assert "Normal" not in JORGE_ATTACK_CLASSES
    assert JORGE_ALL_CLASSES == ("Normal", *JORGE_ATTACK_CLASSES)


@pytest.mark.parametrize(
    ("dataset", "native_class", "native_subclass", "expected"),
    [
        ("edge_iiotset", "DDoS_HTTP", None, "DDoS_HTTP"),
        ("edge_iiotset", "SQL_injection", None, "SQL_injection"),
        ("bot_iot", "DDoS", "HTTP", "DDoS_HTTP"),
        ("bot_iot", "DDoS", "TCP", "DDoS_TCP"),
        ("bot_iot", "DDoS", "UDP", "DDoS_UDP"),
        ("bot_iot", "Reconnaissance", "OS_Fingerprint", "Fingerprinting"),
        ("iot23", "PartOfAHorizontalPortScan", None, "Port_Scanning"),
        ("ton_iot", "backdoor", None, "Backdoor"),
        ("ton_iot", "mitm", None, "MITM"),
        ("ton_iot", "password", None, "Password"),
        ("ton_iot", "ransomware", None, "Ransomware"),
        ("ton_iot", "xss", None, "XSS"),
    ],
)
def test_only_defensible_cross_dataset_mappings_are_exact(
    dataset: str,
    native_class: str,
    native_subclass: str | None,
    expected: str,
):
    result = resolve_jorge_class(dataset, native_class, native_subclass)
    assert result.status == "exact"
    assert result.attack_type == expected


@pytest.mark.parametrize("attack_type", JORGE_ATTACK_CLASSES)
def test_every_edge_attack_type_maps_directly(attack_type: str):
    result = resolve_jorge_class("edge_iiotset", attack_type)
    assert (result.status, result.attack_type) == ("exact", attack_type)


@pytest.mark.parametrize(
    ("native_class", "native_subclass", "reason"),
    [
        ("DDoS", "unrepresented_protocol", "bot_ddos_requires_target_subcategory"),
        (
            "Reconnaissance",
            "unrepresented_scan",
            "bot_reconnaissance_requires_target_subcategory",
        ),
    ],
)
def test_bot_composite_defaults_are_ambiguities(
    native_class: str, native_subclass: str, reason: str
):
    result = resolve_jorge_class("bot_iot", native_class, native_subclass)
    assert (result.status, result.attack_type, result.reason) == (
        "ambiguous",
        None,
        reason,
    )


@pytest.mark.parametrize(
    ("dataset", "native_class", "native_subclass", "status"),
    [
        ("bot_iot", "DDoS", None, "ambiguous"),
        ("bot_iot", "Reconnaissance", "Service_Scan", "ambiguous"),
        ("bot_iot", "DoS", "TCP", "out_of_taxonomy"),
        ("bot_iot", "Theft", "Data_Exfiltration", "out_of_taxonomy"),
        ("iot23", "DDoS", None, "ambiguous"),
        ("iot23", "C&C", None, "out_of_taxonomy"),
        ("ton_iot", "ddos", None, "ambiguous"),
        ("ton_iot", "injection", None, "ambiguous"),
        ("ton_iot", "scanning", None, "ambiguous"),
        ("ton_iot", "dos", None, "out_of_taxonomy"),
    ],
)
def test_ambiguous_or_external_labels_are_never_forced_into_jorge_class(
    dataset: str,
    native_class: str,
    native_subclass: str | None,
    status: str,
):
    result = resolve_jorge_class(dataset, native_class, native_subclass)
    assert result.status == status
    assert result.attack_type is None


def test_attack_type_taxonomy_report_is_versioned():
    report = taxonomy_report()
    assert report["normal_policy"] == "binary_detector_only"
    bot_rules = report["native_rules"]["bot_iot"]
    assert (
        bot_rules["composite_by_native_class"]["ddos"]["exact_by_selector"][
            "tcp"
        ]
        == "DDoS_TCP"
    )
    assert bot_rules["composite_by_native_class"]["ddos"][
        "unmapped_or_missing_selector"
    ] == {
        "status": "ambiguous",
        "reason": "bot_ddos_requires_target_subcategory",
    }
    assert bot_rules["composite_by_native_class"]["reconnaissance"][
        "unmapped_or_missing_selector"
    ] == {
        "status": "ambiguous",
        "reason": "bot_reconnaissance_requires_target_subcategory",
    }
    assert bot_rules["composite_by_native_class"]["reconnaissance"][
        "special_rejections"
    ]["service_scan"] == {
        "status": "ambiguous",
        "reason": "service_scan_is_not_strictly_port_scanning",
    }
    assert bot_rules["unmapped_or_missing_native_class"] == {
        "status": "out_of_taxonomy",
        "reason": "bot_label_not_in_jorge",
    }
    assert report["native_rules"]["iot23"]["otherwise"] == "out_of_taxonomy"
    assert len(report["sha256"]) == 64


def test_multidataset_taxonomy_preserves_jorge_and_adds_two_classes():
    assert MULTIDATASET_ATTACK_CLASSES[:14] == JORGE_ATTACK_CLASSES
    assert MULTIDATASET_ATTACK_CLASSES[14:] == ("DoS", "Command_and_Control")
    assert len(MULTIDATASET_ATTACK_CLASSES) == 16
    assert attack_classes_for_taxonomy(MULTIDATASET_TAXONOMY_VERSION) == (
        MULTIDATASET_ATTACK_CLASSES
    )
    with pytest.raises(ValueError, match="no soportada"):
        attack_classes_for_taxonomy("taxonomy_unknown")


@pytest.mark.parametrize(
    (
        "dataset",
        "native_class",
        "native_subclass",
        "transport",
        "service",
        "status",
        "expected",
    ),
    [
        ("bot_iot", "DoS", "TCP", None, None, "exact", "DoS"),
        (
            "bot_iot",
            "Reconnaissance",
            "Service_Scan",
            None,
            None,
            "forced",
            "Port_Scanning",
        ),
        ("iot23", "C&C", None, None, None, "exact", "Command_and_Control"),
        (
            "iot23",
            "C&C-FileDownload",
            None,
            None,
            None,
            "exact",
            "Command_and_Control",
        ),
        ("iot23", "DDoS", None, "tcp", "-", "forced", "DDoS_TCP"),
        ("iot23", "DDoS", None, "udp", "-", "forced", "DDoS_UDP"),
        ("ton_iot", "DDoS", None, "tcp", "http", "forced", "DDoS_HTTP"),
        ("ton_iot", "DDoS", None, "icmp6", "-", "forced", "DDoS_ICMP"),
        ("ton_iot", "injection", None, None, None, "forced", "SQL_injection"),
        ("ton_iot", "scanning", None, None, None, "forced", "Port_Scanning"),
        ("ton_iot", "dos", None, None, None, "exact", "DoS"),
    ],
)
def test_multidataset_mapping_rules(
    dataset: str,
    native_class: str,
    native_subclass: str | None,
    transport: str | None,
    service: str | None,
    status: str,
    expected: str,
):
    resolution = resolve_multidataset_class(
        dataset,
        native_class,
        native_subclass,
        transport_proto=transport,
        app_proto=service,
    )
    assert (resolution.status, resolution.attack_type) == (status, expected)


def test_multidataset_ddos_without_raw_protocol_remains_ambiguous():
    resolution = resolve_multidataset_class("ton_iot", "ddos")
    assert resolution.status == "ambiguous"
    assert resolution.attack_type is None
    assert resolution.reason == "ton_ddos_missing_raw_protocol"


def test_multidataset_report_documents_forced_mapping_without_ports_or_llm():
    report = multidataset_taxonomy_report()
    assert report["version"] == MULTIDATASET_TAXONOMY_VERSION
    assert report["attack_classes"] == list(MULTIDATASET_ATTACK_CLASSES)
    policy = report["forced_mapping_policy"]["generic_ddos"]
    assert policy["port_inference"] is False
    assert policy["llm_inference"] is False
    assert len(report["sha256"]) == 64
