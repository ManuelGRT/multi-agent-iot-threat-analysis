# src/drift/mmd_monitor.py
from __future__ import annotations

import numpy as np

try:
    from alibi_detect.cd import MMDDrift
except ImportError:  # alibi-detect is optional; tests use the numpy fallback.
    MMDDrift = None


class EmbeddingMMDMonitor:
    def __init__(
        self,
        reference_embeddings: np.ndarray,
        p_val: float = 0.05,
        threshold: float | None = None,
        random_seed: int = 0,
    ):
        self.reference_embeddings = _as_2d(reference_embeddings)
        self.p_val = p_val
        self.threshold = threshold
        self.random_seed = random_seed
        self.detector = MMDDrift(self.reference_embeddings, p_val=p_val) if MMDDrift is not None else None

    def predict(self, batch_embeddings: np.ndarray) -> dict:
        batch = _as_2d(batch_embeddings)
        if self.detector is not None:
            return self.detector.predict(batch)

        distance = mmd2_rbf(self.reference_embeddings, batch)
        threshold = self.threshold
        if threshold is None:
            threshold = bootstrap_threshold(
                self.reference_embeddings,
                sample_size=min(len(batch), max(2, len(self.reference_embeddings) // 2)),
                p_val=self.p_val,
                random_seed=self.random_seed,
            )
        is_drift = int(distance > threshold)
        return {
            "meta": {"backend": "numpy_mmd", "p_val": self.p_val},
            "data": {
                "is_drift": is_drift,
                "distance": float(distance),
                "threshold": float(threshold),
                "p_val": None,
            },
        }


def mmd2_rbf(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
    x = _as_2d(x)
    y = _as_2d(y)
    if gamma is None:
        gamma = _median_gamma(np.vstack([x, y]))
    k_xx = _rbf_kernel(x, x, gamma)
    k_yy = _rbf_kernel(y, y, gamma)
    k_xy = _rbf_kernel(x, y, gamma)
    return float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())


def bootstrap_threshold(reference: np.ndarray, sample_size: int, p_val: float, random_seed: int = 0) -> float:
    reference = _as_2d(reference)
    rng = np.random.default_rng(random_seed)
    sample_size = max(2, min(sample_size, len(reference) // 2))
    if len(reference) < sample_size * 2:
        return 1e-6

    scores = []
    for _ in range(100):
        indices = rng.choice(len(reference), size=sample_size * 2, replace=False)
        left = reference[indices[:sample_size]]
        right = reference[indices[sample_size:]]
        scores.append(mmd2_rbf(left, right))
    return float(np.quantile(scores, 1.0 - p_val))


def _as_2d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("Embeddings must be a 2D array")
    if len(array) == 0:
        raise ValueError("At least one embedding is required")
    return array


def _median_gamma(values: np.ndarray) -> float:
    distances = _squared_distances(values, values)
    upper = distances[np.triu_indices_from(distances, k=1)]
    median = float(np.median(upper[upper > 0])) if np.any(upper > 0) else 1.0
    return 1.0 / (2.0 * median)


def _rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-gamma * _squared_distances(x, y))


def _squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_norm = np.sum(x * x, axis=1)[:, None]
    y_norm = np.sum(y * y, axis=1)[None, :]
    return np.maximum(x_norm + y_norm - 2.0 * x @ y.T, 0.0)
