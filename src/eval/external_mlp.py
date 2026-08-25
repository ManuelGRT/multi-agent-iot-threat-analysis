"""Sklearn-like bridge to Jorge's PyTorch MLP in another interpreter.

The evaluator's process owns feature scaling and the sparse matrices.  PyTorch
stays isolated in :mod:`scripts.run_torch_mlp_worker`: ``fit`` invokes its
``train`` mode exactly once, while every ``predict`` invocation reuses the same
persisted checkpoint.  Cross-interpreter artifacts are deliberately limited to
CSR ``.npz``, integer ``.npy``, JSON and Torch's state-dict ``.pt`` format.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import weakref
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler


JORGE_HIDDEN_SIZES = (256, 128, 64)
DEFAULT_DROPOUT = 0.3
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 100
_SENSITIVE_ENV_MARKERS = ("API_KEY", "PASSWORD", "SECRET", "TOKEN")


class ExternalMLPError(RuntimeError):
    """A safe, user-facing failure from the external MLP bridge."""


def detect_torch_interpreter() -> Path:
    """Prefer the user's conventional Anaconda Python, then this interpreter."""

    user_profile = os.environ.get("USERPROFILE")
    if user_profile:
        candidate = Path(user_profile).expanduser() / "anaconda3" / "python.exe"
        if candidate.is_file():
            return candidate.resolve()
    return Path(sys.executable).resolve()


def default_worker_path() -> Path:
    """Return the repository worker path without relying on the current cwd."""

    return (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_torch_mlp_worker.py"
    )


class ExternalTorchMLPFactory:
    """Factory satisfying :class:`src.eval.jorge_models.MLPBackendFactory`.

    Parameters are supplied later by ``run_jorge_benchmarks``.  A factory may
    be used directly or as a context manager; closing its context closes all
    still-live estimators it created.  Without a context, each estimator owns
    and cleans its own exact temporary directory when closed or collected.
    """

    def __init__(
        self,
        *,
        interpreter: str | os.PathLike[str] | None = None,
        worker: str | os.PathLike[str] | None = None,
        temp_root: str | os.PathLike[str] | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        selected_interpreter = (
            detect_torch_interpreter() if interpreter is None else Path(interpreter).expanduser()
        )
        self.interpreter = selected_interpreter
        self.worker = (
            default_worker_path() if worker is None else Path(worker).expanduser().resolve()
        )
        self.temp_root = (
            None if temp_root is None else Path(temp_root).expanduser().resolve()
        )
        if self.temp_root is not None:
            self.temp_root.mkdir(parents=True, exist_ok=True)
            if not self.temp_root.is_dir():
                raise ValueError("temp_root must be a directory")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive or None")
        self.timeout_seconds = timeout_seconds
        self._estimators: weakref.WeakSet[ExternalTorchMLPClassifier] = weakref.WeakSet()
        self._closed = False

    def __call__(
        self,
        *,
        n_classes: int,
        seed: int,
        parameters: Mapping[str, Any],
    ) -> "ExternalTorchMLPClassifier":
        if self._closed:
            raise RuntimeError("ExternalTorchMLPFactory is closed")
        estimator = ExternalTorchMLPClassifier(
            interpreter=self.interpreter,
            worker=self.worker,
            temp_root=self.temp_root,
            timeout_seconds=self.timeout_seconds,
            n_classes=n_classes,
            seed=seed,
            parameters=parameters,
        )
        self._estimators.add(estimator)
        return estimator

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for estimator in list(self._estimators):
            estimator.close()
        self._estimators.clear()

    def __enter__(self) -> "ExternalTorchMLPFactory":
        if self._closed:
            raise RuntimeError("ExternalTorchMLPFactory is closed")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        del exc_type, exc, traceback
        self.close()
        return False


class ExternalTorchMLPClassifier:
    """Sparse, scaled, subprocess-backed classifier with sklearn semantics."""

    def __init__(
        self,
        *,
        interpreter: str | os.PathLike[str],
        worker: str | os.PathLike[str],
        temp_root: Path | None,
        timeout_seconds: float | None,
        n_classes: int,
        seed: int,
        parameters: Mapping[str, Any],
    ) -> None:
        if isinstance(n_classes, bool) or not isinstance(n_classes, (int, np.integer)):
            raise TypeError("n_classes must be an integer")
        if int(n_classes) < 2:
            raise ValueError("n_classes must be at least 2")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer")
        if int(seed) < 0:
            raise ValueError("seed must be non-negative")

        parsed = _validate_parameters(parameters, int(seed))
        self.interpreter = Path(interpreter).expanduser()
        self.worker = Path(worker).expanduser().resolve()
        self.timeout_seconds = timeout_seconds
        self.n_classes = int(n_classes)
        self.seed = int(seed)
        self.hidden_sizes = parsed["hidden_sizes"]
        self.dropout = parsed["dropout"]
        self.learning_rate = parsed["learning_rate"]
        self.batch_size = parsed["batch_size"]
        self.epochs = parsed["epochs"]
        self.n_jobs = 1

        root = None if temp_root is None else str(temp_root)
        self._temporary = tempfile.TemporaryDirectory(prefix="tfm_jorge_mlp_", dir=root)
        self._work_dir = Path(self._temporary.name).resolve()
        if temp_root is not None and self._work_dir.parent != temp_root.resolve():
            self._temporary.cleanup()
            raise RuntimeError("temporary directory escaped temp_root")
        self.checkpoint_path_ = self._work_dir / "model.pt"
        self.scaler_: StandardScaler | None = None
        self.n_features_in_: int | None = None
        self.classes_ = np.arange(self.n_classes, dtype=np.int64)
        self.train_result_: dict[str, Any] | None = None
        self._train_invoked = False
        self._closed = False
        self._prediction_index = 0

    @property
    def work_dir(self) -> Path:
        """The exact owned directory; useful for diagnostics and tests."""

        return self._work_dir

    def fit(self, x: Any, y: Any) -> "ExternalTorchMLPClassifier":
        if self._closed:
            raise RuntimeError("ExternalTorchMLPClassifier is closed")
        if self._train_invoked:
            raise RuntimeError("fit may only be called once for this external estimator")
        matrix = _as_valid_csr(x, "x")
        labels = _as_valid_labels(y, matrix.shape[0], self.n_classes)

        # Fit only on train.  The fitted object remains in this process and is
        # reused unchanged for validation/test transforms.
        self.scaler_ = StandardScaler(with_mean=False)
        scaled = self.scaler_.fit_transform(matrix)
        scaled = sparse.csr_matrix(scaled, dtype=np.float32)
        self.n_features_in_ = int(scaled.shape[1])

        matrix_path = self._work_dir / "train_matrix.npz"
        labels_path = self._work_dir / "train_labels.npy"
        config_path = self._work_dir / "train.json"
        output_path = self._work_dir / "train_result.json"
        sparse.save_npz(matrix_path, scaled)
        np.save(labels_path, labels, allow_pickle=False)
        _write_json(
            config_path,
            {
                "mode": "train",
                "paths": {
                    "train_matrix": str(matrix_path),
                    "train_labels": str(labels_path),
                    "checkpoint": str(self.checkpoint_path_),
                },
                "training": self._training_config(),
                "num_classes": self.n_classes,
            },
        )
        self._train_invoked = True
        try:
            result = self._invoke_worker("train", config_path, output_path)
            if not self.checkpoint_path_.is_file():
                raise ExternalMLPError("The train worker did not create its checkpoint")
            self.train_result_ = dict(result)
            return self
        finally:
            self._discard_artifacts(matrix_path, labels_path, config_path, output_path)

    def predict(self, x: Any) -> np.ndarray:
        if self._closed:
            raise RuntimeError("ExternalTorchMLPClassifier is closed")
        if not self._train_invoked or self.scaler_ is None or self.n_features_in_ is None:
            raise RuntimeError("ExternalTorchMLPClassifier must be fitted before predict")
        if not self.checkpoint_path_.is_file():
            raise RuntimeError("The fitted external MLP checkpoint is missing")
        matrix = _as_valid_csr(x, "x")
        if int(matrix.shape[1]) != self.n_features_in_:
            raise ValueError(
                f"x has {matrix.shape[1]} features; expected {self.n_features_in_}"
            )
        scaled = self.scaler_.transform(matrix)
        scaled = sparse.csr_matrix(scaled, dtype=np.float32)

        self._prediction_index += 1
        suffix = f"{self._prediction_index:04d}"
        matrix_path = self._work_dir / f"predict_matrix_{suffix}.npz"
        config_path = self._work_dir / f"predict_{suffix}.json"
        output_path = self._work_dir / f"predict_result_{suffix}.json"
        sparse.save_npz(matrix_path, scaled)
        _write_json(
            config_path,
            {
                "mode": "predict",
                "paths": {
                    "matrix": str(matrix_path),
                    "checkpoint": str(self.checkpoint_path_),
                },
                "training": {"seed": self.seed},
            },
        )
        try:
            result = self._invoke_worker("predict", config_path, output_path)
            predictions = np.asarray(result.get("predictions"))
            if predictions.ndim != 1 or predictions.shape[0] != matrix.shape[0]:
                raise ExternalMLPError("The predict worker returned an invalid prediction shape")
            if not np.issubdtype(predictions.dtype, np.integer):
                if not np.all(np.equal(predictions, np.floor(predictions))):
                    raise ExternalMLPError("The predict worker returned non-integral predictions")
            predictions = predictions.astype(np.int64, copy=False)
            if np.any(predictions < 0) or np.any(predictions >= self.n_classes):
                raise ExternalMLPError("The predict worker returned an unknown class index")
            return predictions
        finally:
            self._discard_artifacts(matrix_path, config_path, output_path)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        temporary = self._temporary
        self._temporary = None
        if temporary is not None:
            owned = Path(temporary.name).resolve()
            if owned != self._work_dir:
                raise RuntimeError("refusing to clean a directory not owned by this estimator")
            temporary.cleanup()

    def __enter__(self) -> "ExternalTorchMLPClassifier":
        if self._closed:
            raise RuntimeError("ExternalTorchMLPClassifier is closed")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        del exc_type, exc, traceback
        self.close()
        return False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Destructors must stay best-effort during interpreter shutdown.
            pass

    def _training_config(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "hidden_sizes": list(self.hidden_sizes),
            "batch_size": self.batch_size,
            "seed": self.seed,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
        }

    def _invoke_worker(
        self,
        mode: str,
        config_path: Path,
        output_path: Path,
    ) -> Mapping[str, Any]:
        command = [
            str(self.interpreter),
            str(self.worker),
            "--mode",
            mode,
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ]
        environment = {
            name: value
            for name, value in os.environ.items()
            if not any(marker in name.upper() for marker in _SENSITIVE_ENV_MARKERS)
        }
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_seconds,
                cwd=str(self.worker.parent.parent),
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ExternalMLPError(
                f"Could not complete the external MLP {mode} worker"
            ) from exc

        result: Any = None
        try:
            result = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        if not isinstance(result, Mapping):
            raise ExternalMLPError(
                f"External MLP {mode} worker returned no valid JSON (exit {completed.returncode})"
            )
        if completed.returncode != 0 or result.get("status") != "ok":
            error = result.get("error")
            if isinstance(error, Mapping):
                code = str(error.get("code", "worker_error"))
                message = str(error.get("message", "external worker failed"))
                raise ExternalMLPError(f"External MLP {mode} failed ({code}): {message}")
            raise ExternalMLPError(f"External MLP {mode} worker failed")
        return result

    def _discard_artifacts(self, *paths: Path) -> None:
        for path in paths:
            resolved = path.resolve()
            if resolved.parent != self._work_dir:
                raise RuntimeError("refusing to remove an artifact outside the owned directory")
            try:
                resolved.unlink()
            except FileNotFoundError:
                pass


def _validate_parameters(parameters: Mapping[str, Any], seed: int) -> dict[str, Any]:
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping")
    known = {
        "hidden_sizes",
        "dropout",
        "learning_rate",
        "batch_size",
        "epochs",
        "random_state",
        "n_jobs",
    }
    unknown = sorted(set(parameters).difference(known))
    if unknown:
        raise ValueError(f"Unsupported external MLP parameters: {unknown}")
    hidden_sizes = _integer_tuple(
        parameters.get("hidden_sizes", JORGE_HIDDEN_SIZES), "hidden_sizes"
    )
    if hidden_sizes != JORGE_HIDDEN_SIZES:
        raise ValueError("Jorge's MLP hidden_sizes must remain (256, 128, 64)")
    dropout = _finite_float(parameters.get("dropout", DEFAULT_DROPOUT), "dropout")
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError("dropout must be in [0, 1)")
    learning_rate = _finite_float(
        parameters.get("learning_rate", DEFAULT_LEARNING_RATE), "learning_rate"
    )
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    batch_size = _positive_int(parameters.get("batch_size", DEFAULT_BATCH_SIZE), "batch_size")
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2 for BatchNorm")
    epochs = _positive_int(parameters.get("epochs", DEFAULT_EPOCHS), "epochs")
    random_state = parameters.get("random_state", seed)
    if isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer)):
        raise TypeError("random_state must be an integer")
    if int(random_state) != seed:
        raise ValueError("random_state must match the deterministic factory seed")
    n_jobs = parameters.get("n_jobs", 1)
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)) or int(n_jobs) != 1:
        raise ValueError("n_jobs must remain 1")
    return {
        "hidden_sizes": hidden_sizes,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
    }


def _as_valid_csr(value: Any, field: str) -> sparse.csr_matrix:
    try:
        matrix = sparse.csr_matrix(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be convertible to a numeric CSR matrix") from exc
    if matrix.ndim != 2 or matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
        raise ValueError(f"{field} must be a non-empty two-dimensional matrix")
    if matrix.data.size and not np.isfinite(matrix.data).all():
        raise ValueError(f"{field} contains NaN or infinity")
    return matrix


def _as_valid_labels(value: Any, expected: int, n_classes: int) -> np.ndarray:
    labels = np.asarray(value)
    if labels.ndim != 1 or labels.shape[0] != expected:
        raise ValueError(f"y must have shape ({expected},)")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("y must contain encoded integer labels")
    labels = labels.astype(np.int64, copy=False)
    if np.any(labels < 0) or np.any(labels >= n_classes):
        raise ValueError("y contains a class outside [0, n_classes)")
    if np.unique(labels).size < 2:
        raise ValueError("y must contain at least two classes")
    return labels


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field} must be an integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field} must be positive")
    return parsed


def _integer_tuple(value: Any, field: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be a sequence of integers")
    result = tuple(_positive_int(item, field) for item in value)
    if not result:
        raise ValueError(f"{field} must not be empty")
    return result


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{field} must be numeric")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ExternalMLPError",
    "ExternalTorchMLPClassifier",
    "ExternalTorchMLPFactory",
    "default_worker_path",
    "detect_torch_interpreter",
]
