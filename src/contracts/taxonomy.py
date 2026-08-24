# src/contracts/taxonomy.py
from __future__ import annotations

from dataclasses import dataclass


BENIGN_LABELS = {"benign", "normal", "0", "false", "none", "-"}

ATTACK_FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "mitm": ("mitm", "man in the middle", "arp spoof", "spoofing"),
    "ddos": ("ddos", "dos", "flood", "syn", "udp flood", "tcp flood"),
    "scanning": ("scan", "recon", "portscan", "service scan"),
    "botnet": ("botnet", "mirai", "gafgyt", "torii", "c2", "c&c"),
    "bruteforce": ("brute", "password", "login", "credential"),
    "injection": ("injection", "sql", "xss", "command injection"),
    "malware": ("malware", "ransom", "backdoor", "trojan", "worm"),
    "exfiltration": ("exfil", "data theft", "leak"),
}


@dataclass(frozen=True)
class LabelInfo:
    family: str | None
    subtype: str | None
    is_malicious: bool | None


def normalize_label(label: str | None) -> str:
    return (label or "").strip().lower().replace("_", " ").replace("-", " ")


def label_is_benign(label: str | None) -> bool:
    return normalize_label(label) in BENIGN_LABELS


def infer_label_info(label: str | None, text: str = "") -> LabelInfo:
    normalized = normalize_label(label)
    normalized_text = normalize_label(text)
    if not normalized and not normalized_text:
        return LabelInfo(family=None, subtype=None, is_malicious=None)
    if label_is_benign(label):
        return LabelInfo(family=None, subtype=None, is_malicious=False)

    if normalized:
        for family, keywords in ATTACK_FAMILY_KEYWORDS.items():
            if any(keyword in normalized for keyword in keywords):
                return LabelInfo(family=family, subtype=normalized, is_malicious=True)

    haystack = f"{normalized} {normalized_text}".strip()
    for family, keywords in ATTACK_FAMILY_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            return LabelInfo(family=family, subtype=normalized or None, is_malicious=True)

    if normalized:
        return LabelInfo(family="unknown_attack", subtype=normalized, is_malicious=True)
    return LabelInfo(family="unknown_attack", subtype=None, is_malicious=True)
