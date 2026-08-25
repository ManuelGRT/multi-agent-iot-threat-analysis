import numpy as np

import src.drift.mmd_monitor as mmd_monitor
from src.drift.adwin_monitor import ScalarDriftMonitor
from src.drift.alerts import alert_from_mmd_result


def test_numpy_mmd_detects_shift(monkeypatch):
    monkeypatch.setattr(mmd_monitor, "MMDDrift", None)
    rng = np.random.default_rng(42)
    reference = rng.normal(0, 0.1, size=(80, 4))
    shifted = rng.normal(2.0, 0.1, size=(20, 4))

    monitor = mmd_monitor.EmbeddingMMDMonitor(reference, p_val=0.05)
    result = monitor.predict(shifted)

    assert result["data"]["is_drift"] == 1
    assert alert_from_mmd_result(result) is not None


def test_adwin_monitor_returns_boolean():
    monitor = ScalarDriftMonitor(delta=0.1)

    assert isinstance(monitor.update(0.1), bool)
