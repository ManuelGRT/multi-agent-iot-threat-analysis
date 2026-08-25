# src/drift/adwin_monitor.py
from river import drift

class ScalarDriftMonitor:
    def __init__(self, delta: float = 0.002):
        self.detector = drift.ADWIN(delta=delta)

    def update(self, value: float) -> bool:
        self.detector.update(value)
        return self.detector.drift_detected
