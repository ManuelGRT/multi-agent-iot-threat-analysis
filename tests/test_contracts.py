import pytest
from pydantic import ValidationError

from src.contracts.canonical import CanonicalEvent, Provenance


def test_canonical_event_accepts_valid_network_flow():
    event = CanonicalEvent(
        event_id="iot23::sample::1",
        modality="network_flow",
        src_ip="10.0.0.2",
        dst_ip="10.0.0.3",
        transport_proto="tcp",
        label_raw="Benign",
        semantic_text="dataset=IoT-23 | label_raw=Benign",
        provenance=Provenance(dataset="IoT-23"),
        mapping_confidence=0.9,
    )

    assert event.is_labeled_malicious is False


def test_canonical_event_rejects_invalid_mapping_confidence():
    with pytest.raises(ValidationError):
        CanonicalEvent(
            event_id="bad",
            modality="network_flow",
            semantic_text="bad",
            provenance=Provenance(dataset="IoT-23"),
            mapping_confidence=1.5,
        )
