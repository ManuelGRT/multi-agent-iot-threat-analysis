# src/eval/ablations.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationConfig:
    name: str
    use_latent_neighbors: bool = True
    use_judge: bool = True
    use_explanations: bool = True


DEFAULT_ABLATIONS = [
    AblationConfig(name="full"),
    AblationConfig(name="no_latent_neighbors", use_latent_neighbors=False),
    AblationConfig(name="no_judge", use_judge=False),
    AblationConfig(name="detector_classifier_only", use_judge=False, use_explanations=False),
]
