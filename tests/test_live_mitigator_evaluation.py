"""Regresiones del runner de evaluación en vivo del mitigador."""
from __future__ import annotations

from scripts.evaluate_live_mitigator_multidataset16 import DEFAULT_MODEL


def test_live_mitigator_runner_keeps_the_production_mistral_model_as_default():
    assert DEFAULT_MODEL == "mistral-small-2603"
