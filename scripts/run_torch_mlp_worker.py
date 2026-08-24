#!/usr/bin/env python
"""Train and run Jorge's PyTorch MLP from pre-vectorized CSR matrices.

This script is intentionally an isolated subprocess worker.  Its import path has
no PyTorch dependency, which lets the main evaluator validate a job before it
selects the Python interpreter that owns PyTorch.

Example::

    python scripts/run_torch_mlp_worker.py --config mlp_job.json --output result.json

The legacy/default ``benchmark`` mode still trains once and returns validation
and test predictions in one invocation.  The persistent modes split that work
across processes::

    python scripts/run_torch_mlp_worker.py --mode train --config train.json --output train-result.json
    python scripts/run_torch_mlp_worker.py --mode predict --config predict.json --output predictions.json

The configuration schema is::

    {
      "paths": {
        "train_matrix": "train.npz",
        "val_matrix": "val.npz",
        "test_matrix": "test.npz",
        "train_labels": "train_labels.npy",
        "val_labels": "val_labels.npy",
        "test_labels": "test_labels.npy"
      },
      "training": {
        "epochs": 100,
        "hidden_sizes": [256, 128, 64],
        "batch_size": 64,
        "seed": 42
      },
      "num_classes": 2
    }

Paths are resolved relative to the configuration file.  ``num_classes`` is
optional and otherwise inferred from the available integer label arrays.  A
``train`` job writes a versioned Torch checkpoint containing only a state dict
and plain metadata; a ``predict`` job reconstructs the architecture from that
checkpoint.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_EPOCHS = 100
DEFAULT_HIDDEN_SIZES = (256, 128, 64)
DEFAULT_BATCH_SIZE = 64
DEFAULT_SEED = 42
LEARNING_RATE = 0.001
DROPOUT = 0.3
CHECKPOINT_FORMAT = "tfm_jorge_mlp_state_dict"
CHECKPOINT_VERSION = 1

MODES = ("benchmark", "train", "predict")
PATH_KEYS_BY_MODE = {
    "benchmark": (
        "train_matrix",
        "val_matrix",
        "test_matrix",
        "train_labels",
        "val_labels",
        "test_labels",
    ),
    "train": ("train_matrix", "train_labels", "checkpoint"),
    "predict": ("matrix", "checkpoint"),
}
# Kept as a public compatibility alias for callers/tests that imported it.
PATH_KEYS = PATH_KEYS_BY_MODE["benchmark"]


class WorkerError(RuntimeError):
    """Expected, user-actionable worker failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class WorkerConfig:
    paths: Mapping[str, Path]
    epochs: int = DEFAULT_EPOCHS
    hidden_sizes: Tuple[int, ...] = DEFAULT_HIDDEN_SIZES
    batch_size: int = DEFAULT_BATCH_SIZE
    seed: int = DEFAULT_SEED
    num_classes: Optional[int] = None
    mode: str = "benchmark"
    learning_rate: float = LEARNING_RATE
    dropout: float = DROPOUT


@dataclass(frozen=True)
class LoadedData:
    x_train: Any
    x_val: Any
    x_test: Any
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    num_classes: int


@dataclass(frozen=True)
class LoadedTrainingData:
    x_train: Any
    y_train: np.ndarray
    num_classes: int


def positive_int(value: Any, field: str) -> int:
    """Return a strict positive integer (booleans are deliberately rejected)."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise WorkerError("invalid_config", "%s debe ser un entero positivo" % field)
    parsed = int(value)
    if parsed <= 0:
        raise WorkerError("invalid_config", "%s debe ser mayor que cero" % field)
    return parsed


def positive_float(value: Any, field: str) -> float:
    """Return a finite, strictly positive floating-point value."""

    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise WorkerError("invalid_config", "%s debe ser un numero positivo" % field)
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise WorkerError("invalid_config", "%s debe ser mayor que cero" % field)
    return parsed


def dropout_float(value: Any, field: str) -> float:
    """Return a finite dropout probability in the half-open interval [0, 1)."""

    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise WorkerError("invalid_config", "%s debe ser un numero" % field)
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0 or parsed >= 1.0:
        raise WorkerError("invalid_config", "%s debe estar en el intervalo [0, 1)" % field)
    return parsed


def parse_hidden_sizes(value: Any) -> Tuple[int, ...]:
    """Parse hidden widths from a JSON list or a comma-separated CLI value."""

    raw: Iterable[Any]
    if isinstance(value, str):
        if not value.strip():
            raise WorkerError("invalid_config", "hidden_sizes no puede estar vacio")
        raw = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple)):
        raw = value
    else:
        raise WorkerError(
            "invalid_config",
            "hidden_sizes debe ser una lista o una cadena separada por comas",
        )

    parsed: List[int] = []
    for index, item in enumerate(raw):
        if isinstance(item, str):
            try:
                item = int(item)
            except ValueError as exc:
                raise WorkerError(
                    "invalid_config",
                    "hidden_sizes[%d] no es un entero" % index,
                ) from exc
        parsed.append(positive_int(item, "hidden_sizes[%d]" % index))
    if not parsed:
        raise WorkerError("invalid_config", "hidden_sizes debe contener al menos una capa")
    return tuple(parsed)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkerError("invalid_config", "%s debe ser un objeto JSON" % field)
    return value


def validate_config(
    raw: Any,
    base_dir: Path,
    mode_override: Optional[str] = None,
) -> WorkerConfig:
    """Validate raw JSON without importing PyTorch or reading the datasets."""

    root = _mapping(raw, "config")
    raw_mode = mode_override if mode_override is not None else root.get("mode", "benchmark")
    if not isinstance(raw_mode, str) or raw_mode not in MODES:
        raise WorkerError(
            "invalid_config",
            "mode debe ser uno de: %s" % ", ".join(MODES),
        )
    mode = raw_mode
    raw_paths = _mapping(root.get("paths"), "paths")
    paths: Dict[str, Path] = {}
    for key in PATH_KEYS_BY_MODE[mode]:
        value = raw_paths.get(key)
        if not isinstance(value, str) or not value.strip():
            raise WorkerError("invalid_config", "Falta paths.%s" % key)
        path = Path(value).expanduser()
        paths[key] = path if path.is_absolute() else (base_dir / path).resolve()

    training_value = root.get("training", {})
    training = _mapping(training_value, "training")
    epochs = positive_int(training.get("epochs", DEFAULT_EPOCHS), "training.epochs")
    batch_size = positive_int(
        training.get("batch_size", DEFAULT_BATCH_SIZE),
        "training.batch_size",
    )
    seed_value = training.get("seed", DEFAULT_SEED)
    if isinstance(seed_value, bool) or not isinstance(seed_value, (int, np.integer)):
        raise WorkerError("invalid_config", "training.seed debe ser un entero no negativo")
    seed = int(seed_value)
    if seed < 0:
        raise WorkerError("invalid_config", "training.seed debe ser un entero no negativo")

    hidden_sizes = parse_hidden_sizes(
        training.get("hidden_sizes", list(DEFAULT_HIDDEN_SIZES))
    )
    learning_rate = positive_float(
        training.get("learning_rate", LEARNING_RATE),
        "training.learning_rate",
    )
    dropout = dropout_float(training.get("dropout", DROPOUT), "training.dropout")
    raw_num_classes = root.get("num_classes")
    num_classes = None
    if raw_num_classes is not None:
        num_classes = positive_int(raw_num_classes, "num_classes")
        if num_classes < 2:
            raise WorkerError("invalid_config", "num_classes debe ser al menos 2")

    return WorkerConfig(
        paths=paths,
        mode=mode,
        epochs=epochs,
        hidden_sizes=hidden_sizes,
        batch_size=batch_size,
        seed=seed,
        num_classes=num_classes,
        learning_rate=learning_rate,
        dropout=dropout,
    )


def load_config(path: Path, mode_override: Optional[str] = None) -> WorkerConfig:
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError as exc:
        raise WorkerError("config_not_found", "No existe el JSON de configuracion") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkerError("invalid_config", "No se pudo leer el JSON de configuracion") from exc
    return validate_config(raw, path.resolve().parent, mode_override=mode_override)


def apply_cli_overrides(
    config: WorkerConfig,
    epochs: Optional[int] = None,
    hidden_sizes: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> WorkerConfig:
    """Apply the deliberately narrow test-time architecture overrides."""

    updates: Dict[str, Any] = {}
    if epochs is not None:
        updates["epochs"] = positive_int(epochs, "--epochs")
    if hidden_sizes is not None:
        updates["hidden_sizes"] = parse_hidden_sizes(hidden_sizes)
    if batch_size is not None:
        updates["batch_size"] = positive_int(batch_size, "--batch-size")
    return replace(config, **updates)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Worker CPU para el MLP PyTorch exacto del TFM de Jorge."
    )
    parser.add_argument("--config", type=Path, required=True, help="Trabajo JSON de entrada")
    parser.add_argument("--output", type=Path, help="JSON de salida; stdout si se omite")
    parser.add_argument(
        "--mode",
        choices=MODES,
        help="Modo de trabajo; si se omite se usa config.mode o benchmark",
    )
    parser.add_argument("--epochs", type=int, help="Override de epocas para pruebas")
    parser.add_argument(
        "--hidden-sizes",
        help="Override de capas ocultas para pruebas, por ejemplo 8,4",
    )
    parser.add_argument("--batch-size", type=int, help="Override de batch para pruebas")
    return parser


def _load_scipy_sparse() -> Any:
    try:
        return importlib.import_module("scipy.sparse")
    except ImportError as exc:
        raise WorkerError(
            "scipy_unavailable",
            "SciPy no esta instalado en el interprete del worker; se necesita para leer CSR .npz",
        ) from exc


def _load_matrix(sparse: Any, path: Path, name: str) -> Any:
    try:
        matrix = sparse.load_npz(str(path))
    except FileNotFoundError as exc:
        raise WorkerError("input_not_found", "No existe la matriz %s" % name) from exc
    except (OSError, ValueError) as exc:
        raise WorkerError("invalid_input", "No se pudo leer la matriz %s" % name) from exc
    if len(matrix.shape) != 2 or matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
        raise WorkerError("invalid_input", "La matriz %s debe ser bidimensional y no vacia" % name)
    matrix = matrix.tocsr().astype(np.float32, copy=False)
    if matrix.data.size and not np.isfinite(matrix.data).all():
        raise WorkerError("invalid_input", "La matriz %s contiene NaN o infinito" % name)
    return matrix


def _load_labels(path: Path, name: str) -> np.ndarray:
    try:
        labels = np.load(str(path), allow_pickle=False)
    except FileNotFoundError as exc:
        raise WorkerError("input_not_found", "No existe el vector %s" % name) from exc
    except (OSError, ValueError) as exc:
        raise WorkerError("invalid_input", "No se pudo leer el vector %s" % name) from exc
    if labels.ndim != 1 or labels.size == 0:
        raise WorkerError("invalid_input", "El vector %s debe ser unidimensional y no vacio" % name)
    if not np.issubdtype(labels.dtype, np.integer):
        raise WorkerError("invalid_input", "El vector %s debe contener etiquetas enteras" % name)
    labels = labels.astype(np.int64, copy=False)
    if int(labels.min()) < 0:
        raise WorkerError("invalid_input", "El vector %s contiene etiquetas negativas" % name)
    return labels


def load_data(config: WorkerConfig) -> LoadedData:
    """Read and validate the six input artifacts without densifying CSR matrices."""

    sparse = _load_scipy_sparse()
    x_train = _load_matrix(sparse, config.paths["train_matrix"], "train")
    x_val = _load_matrix(sparse, config.paths["val_matrix"], "val")
    x_test = _load_matrix(sparse, config.paths["test_matrix"], "test")
    y_train = _load_labels(config.paths["train_labels"], "train_labels")
    y_val = _load_labels(config.paths["val_labels"], "val_labels")
    y_test = _load_labels(config.paths["test_labels"], "test_labels")

    matrices = (("train", x_train, y_train), ("val", x_val, y_val), ("test", x_test, y_test))
    for name, matrix, labels in matrices:
        if matrix.shape[0] != labels.shape[0]:
            raise WorkerError(
                "invalid_input",
                "La matriz y las etiquetas de %s tienen distinto numero de filas" % name,
            )
    feature_counts = {int(matrix.shape[1]) for _, matrix, _ in matrices}
    if len(feature_counts) != 1:
        raise WorkerError("invalid_input", "Las matrices no tienen el mismo numero de columnas")

    observed_max = max(int(labels.max()) for _, _, labels in matrices)
    num_classes = config.num_classes if config.num_classes is not None else observed_max + 1
    if num_classes < 2:
        raise WorkerError("invalid_input", "Se necesitan al menos dos clases")
    if observed_max >= num_classes:
        raise WorkerError(
            "invalid_input",
            "Hay etiquetas fuera del intervalo [0, num_classes)",
        )
    if np.unique(y_train).size < 2:
        raise WorkerError("invalid_input", "El conjunto train debe contener al menos dos clases")

    return LoadedData(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        num_classes=int(num_classes),
    )


def load_training_data(config: WorkerConfig) -> LoadedTrainingData:
    """Load the two artifacts needed by persistent ``train`` mode."""

    sparse = _load_scipy_sparse()
    x_train = _load_matrix(sparse, config.paths["train_matrix"], "train")
    y_train = _load_labels(config.paths["train_labels"], "train_labels")
    if x_train.shape[0] != y_train.shape[0]:
        raise WorkerError(
            "invalid_input",
            "La matriz y las etiquetas de train tienen distinto numero de filas",
        )
    if np.unique(y_train).size < 2:
        raise WorkerError("invalid_input", "El conjunto train debe contener al menos dos clases")
    observed_max = int(y_train.max())
    num_classes = config.num_classes if config.num_classes is not None else observed_max + 1
    if num_classes < 2:
        raise WorkerError("invalid_input", "Se necesitan al menos dos clases")
    if observed_max >= num_classes:
        raise WorkerError(
            "invalid_input",
            "Hay etiquetas fuera del intervalo [0, num_classes)",
        )
    return LoadedTrainingData(
        x_train=x_train,
        y_train=y_train,
        num_classes=int(num_classes),
    )


def make_epoch_batches(
    size: int,
    batch_size: int,
    rng: np.random.RandomState,
) -> List[np.ndarray]:
    """Shuffle indices and avoid a final singleton where a merge is possible."""

    indices = rng.permutation(size)
    batches = [indices[start : start + batch_size] for start in range(0, size, batch_size)]
    if len(batches) > 1 and len(batches[-1]) == 1 and len(batches[-2]) > 1:
        singleton = batches.pop()
        if len(batches[-1]) == 2:
            # A 2 -> 1+2 rebalance would only move the singleton.  A final
            # batch of three is preferable and remains tightly bounded.
            batches[-1] = np.concatenate((batches[-1], singleton))
        else:
            batches.append(np.concatenate((batches[-1][-1:], singleton)))
            batches[-2] = batches[-2][:-1]
    return batches


def _import_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise WorkerError(
            "torch_unavailable",
            "PyTorch no esta instalado en el interprete del worker; ejecute este proceso con un Python que incluya torch",
        ) from exc


def configure_determinism(torch: Any, seed: int) -> None:
    """Pin the worker to deterministic CPU execution with one compute thread."""

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits this setter only before inter-op work starts.
        pass
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def build_model(
    torch: Any,
    input_size: int,
    hidden_sizes: Sequence[int],
    output_size: int,
    dropout: float = DROPOUT,
) -> Any:
    """Create Linear -> BatchNorm -> ReLU -> Dropout hidden blocks."""

    class SingletonSafeBatchNorm1d(torch.nn.BatchNorm1d):
        def forward(self, inputs: Any) -> Any:
            if self.training and inputs.shape[0] == 1:
                return torch.nn.functional.batch_norm(
                    inputs,
                    self.running_mean,
                    self.running_var,
                    self.weight,
                    self.bias,
                    False,
                    self.momentum,
                    self.eps,
                )
            return super().forward(inputs)

    layers: List[Any] = []
    previous = input_size
    for width in hidden_sizes:
        layers.extend(
            (
                torch.nn.Linear(previous, int(width)),
                SingletonSafeBatchNorm1d(int(width)),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=dropout),
            )
        )
        previous = int(width)
    layers.append(torch.nn.Linear(previous, output_size))
    return torch.nn.Sequential(*layers)


def train_model(torch: Any, model: Any, data: Any, config: WorkerConfig) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    rng = np.random.RandomState(config.seed)
    final_loss = 0.0
    model.train()
    for _epoch in range(config.epochs):
        weighted_loss = 0.0
        seen = 0
        for indices in make_epoch_batches(len(data.y_train), config.batch_size, rng):
            # This is the only training densification point: one CSR batch at a time.
            dense = data.x_train[indices].toarray().astype(np.float32, copy=False)
            inputs = torch.from_numpy(dense)
            targets = torch.from_numpy(data.y_train[indices])
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            batch_len = int(len(indices))
            weighted_loss += float(loss.detach().item()) * batch_len
            seen += batch_len
        final_loss = weighted_loss / max(seen, 1)
    return float(final_loss)


def predict(
    torch: Any,
    model: Any,
    matrix: Any,
    batch_size: int,
    binary: bool,
) -> Tuple[List[int], Optional[List[float]]]:
    predictions: List[int] = []
    probabilities: Optional[List[float]] = [] if binary else None
    model.eval()
    with torch.no_grad():
        for start in range(0, matrix.shape[0], batch_size):
            # Evaluation also densifies only the active CSR batch.
            dense = matrix[start : start + batch_size].toarray().astype(np.float32, copy=False)
            logits = model(torch.from_numpy(dense))
            predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
            if probabilities is not None:
                positive = torch.softmax(logits, dim=1)[:, 1]
                probabilities.extend(float(value) for value in positive.cpu().tolist())
    return predictions, probabilities


def architecture_metadata(
    input_size: int,
    output_size: int,
    config: WorkerConfig,
) -> Dict[str, Any]:
    return {
        "name": "jorge_mlp_pytorch",
        "layer_sizes": [input_size] + list(config.hidden_sizes) + [output_size],
        "hidden_block": "Linear -> BatchNorm1d -> ReLU -> Dropout",
        "batch_normalization": True,
        "activation": "ReLU",
        "dropout": config.dropout,
        "optimizer": "Adam",
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "seed": config.seed,
        "device": "cpu",
        "threads": 1,
        "deterministic_algorithms": True,
    }


def _save_checkpoint(
    torch: Any,
    model: Any,
    checkpoint: Path,
    *,
    input_size: int,
    output_size: int,
    config: WorkerConfig,
) -> None:
    """Atomically persist a state dict plus primitive reconstruction metadata."""

    checkpoint = checkpoint.resolve()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "input_size": int(input_size),
        "num_classes": int(output_size),
        "hidden_sizes": list(config.hidden_sizes),
        "dropout": float(config.dropout),
        "learning_rate": float(config.learning_rate),
        "batch_size": int(config.batch_size),
        "epochs": int(config.epochs),
        "seed": int(config.seed),
        "architecture": architecture_metadata(input_size, output_size, config),
    }
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }
    temporary = checkpoint.with_name(checkpoint.name + ".tmp")
    try:
        torch.save(payload, str(temporary))
        os.replace(str(temporary), str(checkpoint))
    except Exception as exc:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise WorkerError(
            "checkpoint_write_failed",
            "No se pudo guardar el checkpoint Torch",
        ) from exc


def _load_checkpoint(torch: Any, checkpoint: Path) -> Tuple[Any, Mapping[str, Any]]:
    """Load and structurally validate a checkpoint created by this worker."""

    try:
        try:
            payload = torch.load(
                str(checkpoint),
                map_location=torch.device("cpu"),
                weights_only=True,
            )
        except TypeError:
            # ``weights_only`` is unavailable on older supported Torch releases.
            payload = torch.load(str(checkpoint), map_location=torch.device("cpu"))
    except FileNotFoundError as exc:
        raise WorkerError("input_not_found", "No existe el checkpoint") from exc
    except Exception as exc:
        raise WorkerError("invalid_checkpoint", "No se pudo leer el checkpoint Torch") from exc

    if not isinstance(payload, Mapping):
        raise WorkerError("invalid_checkpoint", "El checkpoint no contiene un objeto valido")
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("version") != CHECKPOINT_VERSION:
        raise WorkerError("invalid_checkpoint", "Formato o version de checkpoint no soportado")
    metadata = payload.get("metadata")
    state_dict = payload.get("state_dict")
    if not isinstance(metadata, Mapping) or not isinstance(state_dict, Mapping):
        raise WorkerError("invalid_checkpoint", "Faltan state_dict o metadata en el checkpoint")

    try:
        input_size = positive_int(metadata.get("input_size"), "metadata.input_size")
        num_classes = positive_int(metadata.get("num_classes"), "metadata.num_classes")
        if num_classes < 2:
            raise WorkerError("invalid_config", "metadata.num_classes debe ser al menos 2")
        hidden_sizes = parse_hidden_sizes(metadata.get("hidden_sizes"))
        dropout = dropout_float(metadata.get("dropout"), "metadata.dropout")
    except WorkerError as exc:
        raise WorkerError("invalid_checkpoint", str(exc)) from exc

    model = build_model(
        torch,
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=num_classes,
        dropout=dropout,
    )
    model.to(torch.device("cpu"))
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise WorkerError(
            "invalid_checkpoint",
            "El state_dict no coincide con la arquitectura declarada",
        ) from exc
    return model, metadata


def run_train_job(config: WorkerConfig) -> Dict[str, Any]:
    """Train once and persist the model for later independent predictions."""

    started = time.perf_counter()
    load_started = time.perf_counter()
    data = load_training_data(config)
    load_seconds = time.perf_counter() - load_started

    torch = _import_torch()
    configure_determinism(torch, config.seed)
    input_size = int(data.x_train.shape[1])
    model = build_model(
        torch,
        input_size=input_size,
        hidden_sizes=config.hidden_sizes,
        output_size=data.num_classes,
        dropout=config.dropout,
    )
    model.to(torch.device("cpu"))

    train_started = time.perf_counter()
    final_loss = train_model(torch, model, data, config)
    train_seconds = time.perf_counter() - train_started
    save_started = time.perf_counter()
    _save_checkpoint(
        torch,
        model,
        config.paths["checkpoint"],
        input_size=input_size,
        output_size=data.num_classes,
        config=config,
    )
    save_seconds = time.perf_counter() - save_started
    return {
        "status": "ok",
        "mode": "train",
        "checkpoint": {"format": CHECKPOINT_FORMAT, "version": CHECKPOINT_VERSION},
        "timings_seconds": {
            "load": load_seconds,
            "train": train_seconds,
            "save": save_seconds,
            "total": time.perf_counter() - started,
        },
        "final_train_loss": final_loss,
        "architecture": architecture_metadata(input_size, data.num_classes, config),
        "num_classes": data.num_classes,
        "sample_count": int(data.x_train.shape[0]),
        "torch_version": str(torch.__version__),
    }


def run_predict_job(config: WorkerConfig) -> Dict[str, Any]:
    """Reconstruct one persisted model and predict a CSR matrix."""

    started = time.perf_counter()
    sparse = _load_scipy_sparse()
    matrix = _load_matrix(sparse, config.paths["matrix"], "predict")
    torch = _import_torch()
    configure_determinism(torch, config.seed)
    model, metadata = _load_checkpoint(torch, config.paths["checkpoint"])
    input_size = int(metadata["input_size"])
    num_classes = int(metadata["num_classes"])
    if int(matrix.shape[1]) != input_size:
        raise WorkerError(
            "invalid_input",
            "La matriz predict no tiene el numero de columnas del checkpoint",
        )
    raw_batch_size = metadata.get("batch_size", config.batch_size)
    try:
        batch_size = positive_int(raw_batch_size, "metadata.batch_size")
    except WorkerError as exc:
        raise WorkerError("invalid_checkpoint", str(exc)) from exc
    predict_started = time.perf_counter()
    predictions, probabilities = predict(
        torch,
        model,
        matrix,
        batch_size,
        num_classes == 2,
    )
    predict_seconds = time.perf_counter() - predict_started
    architecture = metadata.get("architecture")
    return {
        "status": "ok",
        "mode": "predict",
        "predictions": predictions,
        "binary_probabilities": probabilities,
        "timings_seconds": {
            "predict": predict_seconds,
            "total": time.perf_counter() - started,
        },
        "architecture": dict(architecture) if isinstance(architecture, Mapping) else None,
        "num_classes": num_classes,
        "sample_count": int(matrix.shape[0]),
        "torch_version": str(torch.__version__),
    }


def run_job(config: WorkerConfig) -> Dict[str, Any]:
    started = time.perf_counter()
    load_started = time.perf_counter()
    data = load_data(config)
    load_seconds = time.perf_counter() - load_started

    # Import after input validation so malformed jobs fail without requiring Torch.
    torch = _import_torch()
    configure_determinism(torch, config.seed)
    model = build_model(
        torch,
        input_size=int(data.x_train.shape[1]),
        hidden_sizes=config.hidden_sizes,
        output_size=data.num_classes,
        dropout=config.dropout,
    )
    model.to(torch.device("cpu"))

    train_started = time.perf_counter()
    final_loss = train_model(torch, model, data, config)
    train_seconds = time.perf_counter() - train_started

    val_started = time.perf_counter()
    val_predictions, val_probabilities = predict(
        torch, model, data.x_val, config.batch_size, data.num_classes == 2
    )
    val_seconds = time.perf_counter() - val_started
    test_started = time.perf_counter()
    test_predictions, test_probabilities = predict(
        torch, model, data.x_test, config.batch_size, data.num_classes == 2
    )
    test_seconds = time.perf_counter() - test_started

    return {
        "status": "ok",
        "predictions": {"val": val_predictions, "test": test_predictions},
        "binary_probabilities": (
            {"val": val_probabilities, "test": test_probabilities}
            if data.num_classes == 2
            else None
        ),
        "timings_seconds": {
            "load": load_seconds,
            "train": train_seconds,
            "predict_val": val_seconds,
            "predict_test": test_seconds,
            "total": time.perf_counter() - started,
        },
        "final_train_loss": final_loss,
        "architecture": architecture_metadata(
            int(data.x_train.shape[1]), data.num_classes, config
        ),
        "num_classes": data.num_classes,
        "sample_counts": {
            "train": int(data.x_train.shape[0]),
            "val": int(data.x_val.shape[0]),
            "test": int(data.x_test.shape[0]),
        },
        "torch_version": str(torch.__version__),
    }


def run_configured_job(config: WorkerConfig) -> Dict[str, Any]:
    """Dispatch one validated configuration without importing Torch early."""

    if config.mode == "train":
        return run_train_job(config)
    if config.mode == "predict":
        return run_predict_job(config)
    return run_job(config)


def _error_result(exc: WorkerError) -> Dict[str, Any]:
    return {
        "status": "error",
        "error": {"code": exc.code, "message": str(exc)},
    }


def emit_result(result: Mapping[str, Any], output: Optional[Path]) -> None:
    payload = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    if output is None:
        sys.stdout.write(payload + "\n")
        return
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(payload + "\n", encoding="utf-8")
    os.replace(str(temporary), str(output))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config, mode_override=args.mode)
        config = apply_cli_overrides(
            config,
            epochs=args.epochs,
            hidden_sizes=args.hidden_sizes,
            batch_size=args.batch_size,
        )
        result = run_configured_job(config)
        emit_result(result, args.output)
        return 0
    except WorkerError as exc:
        emit_result(_error_result(exc), args.output)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
