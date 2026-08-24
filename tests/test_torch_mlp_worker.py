from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from scripts import run_torch_mlp_worker as worker


def valid_raw_config() -> dict:
    return {
        "paths": {
            "train_matrix": "train.npz",
            "val_matrix": "val.npz",
            "test_matrix": "test.npz",
            "train_labels": "train_labels.npy",
            "val_labels": "val_labels.npy",
            "test_labels": "test_labels.npy",
        }
    }


def write_inputs(directory: Path) -> None:
    x_train = sparse.csr_matrix(
        np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.9, 1.0, 0.0],
                [0.2, 0.1, 1.0],
            ],
            dtype=np.float32,
        )
    )
    x_val = sparse.csr_matrix(np.asarray([[0.0, 0.1, 1.0], [1.0, 0.9, 0.0]], dtype=np.float32))
    x_test = sparse.csr_matrix(np.asarray([[0.1, 0.2, 1.0], [0.8, 1.0, 0.0]], dtype=np.float32))
    sparse.save_npz(directory / "train.npz", x_train)
    sparse.save_npz(directory / "val.npz", x_val)
    sparse.save_npz(directory / "test.npz", x_test)
    np.save(directory / "train_labels.npy", np.asarray([0, 0, 1, 1, 0], dtype=np.int64))
    np.save(directory / "val_labels.npy", np.asarray([0, 1], dtype=np.int64))
    np.save(directory / "test_labels.npy", np.asarray([0, 1], dtype=np.int64))


def test_parser_accepts_config_output_and_test_overrides(tmp_path):
    args = worker.build_parser().parse_args(
        [
            "--config",
            str(tmp_path / "job.json"),
            "--output",
            str(tmp_path / "result.json"),
            "--mode",
            "train",
            "--epochs",
            "2",
            "--hidden-sizes",
            "8,4",
            "--batch-size",
            "3",
        ]
    )

    assert args.config == tmp_path / "job.json"
    assert args.output == tmp_path / "result.json"
    assert args.mode == "train"
    assert args.epochs == 2
    assert args.hidden_sizes == "8,4"
    assert args.batch_size == 3


def test_validate_config_uses_exact_jorge_defaults_and_relative_paths(tmp_path):
    config = worker.validate_config(valid_raw_config(), tmp_path)

    assert config.epochs == 100
    assert config.hidden_sizes == (256, 128, 64)
    assert config.batch_size == 64
    assert config.seed == 42
    assert config.num_classes is None
    assert config.paths["train_matrix"] == (tmp_path / "train.npz").resolve()


def test_persistent_modes_require_only_their_own_artifacts(tmp_path):
    train = worker.validate_config(
        {
            "mode": "train",
            "paths": {
                "train_matrix": "train.npz",
                "train_labels": "labels.npy",
                "checkpoint": "model.pt",
            },
            "training": {"dropout": 0.2, "learning_rate": 0.002},
            "num_classes": 3,
        },
        tmp_path,
    )
    prediction = worker.validate_config(
        {
            "mode": "predict",
            "paths": {"matrix": "rows.npz", "checkpoint": "model.pt"},
        },
        tmp_path,
    )

    assert train.mode == "train"
    assert set(train.paths) == {"train_matrix", "train_labels", "checkpoint"}
    assert train.dropout == pytest.approx(0.2)
    assert train.learning_rate == pytest.approx(0.002)
    assert prediction.mode == "predict"
    assert set(prediction.paths) == {"matrix", "checkpoint"}


def test_cli_mode_can_select_schema_for_config_without_mode(tmp_path):
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "paths": {
                    "train_matrix": "train.npz",
                    "train_labels": "labels.npy",
                    "checkpoint": "model.pt",
                }
            }
        ),
        encoding="utf-8",
    )

    config = worker.load_config(config_path, mode_override="train")

    assert config.mode == "train"
    assert config.paths["checkpoint"] == (tmp_path / "model.pt").resolve()


def test_config_and_cli_allow_only_intended_test_size_overrides(tmp_path):
    raw = valid_raw_config()
    raw["training"] = {
        "epochs": 7,
        "hidden_sizes": [12, 6],
        "batch_size": 5,
        "seed": 9,
    }
    raw["num_classes"] = 3
    config = worker.validate_config(raw, tmp_path)
    overridden = worker.apply_cli_overrides(
        config,
        epochs=2,
        hidden_sizes="4, 3",
        batch_size=2,
    )

    assert overridden.epochs == 2
    assert overridden.hidden_sizes == (4, 3)
    assert overridden.batch_size == 2
    assert overridden.seed == 9
    assert overridden.num_classes == 3
    assert worker.LEARNING_RATE == 0.001
    assert worker.DROPOUT == 0.3


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw["paths"].pop("test_labels"), "paths.test_labels"),
        (lambda raw: raw.update({"training": {"epochs": 0}}), "training.epochs"),
        (lambda raw: raw.update({"training": {"hidden_sizes": []}}), "hidden_sizes"),
        (lambda raw: raw.update({"num_classes": 1}), "num_classes"),
    ],
)
def test_validate_config_rejects_invalid_jobs(tmp_path, mutation, message):
    raw = valid_raw_config()
    mutation(raw)

    with pytest.raises(worker.WorkerError, match=message):
        worker.validate_config(raw, tmp_path)


def test_load_data_keeps_matrices_sparse_and_infers_binary_classes(tmp_path):
    write_inputs(tmp_path)
    config = worker.validate_config(valid_raw_config(), tmp_path)

    data = worker.load_data(config)

    assert sparse.isspmatrix_csr(data.x_train)
    assert sparse.isspmatrix_csr(data.x_val)
    assert sparse.isspmatrix_csr(data.x_test)
    assert data.x_train.dtype == np.float32
    assert data.y_train.dtype == np.int64
    assert data.num_classes == 2


def test_load_data_rejects_non_integer_labels_before_torch_import(tmp_path):
    write_inputs(tmp_path)
    np.save(tmp_path / "val_labels.npy", np.asarray([0.0, 1.0], dtype=np.float32))
    config = worker.validate_config(valid_raw_config(), tmp_path)

    with pytest.raises(worker.WorkerError, match="etiquetas enteras"):
        worker.load_data(config)


def test_epoch_batching_avoids_a_trailing_singleton():
    batches = worker.make_epoch_batches(5, 2, np.random.RandomState(42))

    assert sorted(np.concatenate(batches).tolist()) == list(range(5))
    assert [len(batch) for batch in batches] == [2, 3]
    assert all(len(batch) > 1 for batch in batches)


def test_missing_torch_has_a_clear_structured_error(monkeypatch):
    real_import = worker.importlib.import_module

    def fake_import(name):
        if name == "torch":
            raise ImportError("not installed")
        return real_import(name)

    monkeypatch.setattr(worker.importlib, "import_module", fake_import)

    with pytest.raises(worker.WorkerError, match="PyTorch no esta instalado") as caught:
        worker._import_torch()
    assert caught.value.code == "torch_unavailable"


def test_anaconda_torch_subprocess_smoke(tmp_path):
    interpreter = Path(r"C:\Users\34649\anaconda3\python.exe")
    if not interpreter.exists():
        pytest.skip("No existe el interprete Anaconda con PyTorch")
    probe = subprocess.run(
        [str(interpreter), "-c", "import scipy, torch; print(torch.__version__)"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if probe.returncode != 0:
        pytest.skip("El interprete Anaconda no puede importar scipy y torch")

    write_inputs(tmp_path)
    config_path = tmp_path / "job.json"
    output_path = tmp_path / "result.json"
    config_path.write_text(json.dumps(valid_raw_config()), encoding="utf-8")
    completed = subprocess.run(
        [
            str(interpreter),
            str(Path(worker.__file__).resolve()),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--epochs",
            "2",
            "--hidden-sizes",
            "8,4",
            "--batch-size",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert completed.stdout == ""
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["status"] == "ok"
    assert len(result["predictions"]["val"]) == 2
    assert len(result["predictions"]["test"]) == 2
    assert len(result["binary_probabilities"]["val"]) == 2
    assert len(result["binary_probabilities"]["test"]) == 2
    assert result["architecture"]["layer_sizes"] == [3, 8, 4, 2]
    assert result["architecture"]["batch_normalization"] is True
    assert result["architecture"]["dropout"] == 0.3
    assert result["architecture"]["learning_rate"] == 0.001
    assert result["architecture"]["device"] == "cpu"
    assert result["architecture"]["threads"] == 1


def test_anaconda_persistent_train_then_predict_smoke(tmp_path):
    interpreter = Path(r"C:\Users\34649\anaconda3\python.exe")
    if not interpreter.exists():
        pytest.skip("No existe el interprete Anaconda con PyTorch")
    probe = subprocess.run(
        [str(interpreter), "-c", "import scipy, torch"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if probe.returncode != 0:
        pytest.skip("El interprete Anaconda no puede importar scipy y torch")

    write_inputs(tmp_path)
    checkpoint_path = tmp_path / "persistent_model.pt"
    train_config = tmp_path / "persistent_train.json"
    train_output = tmp_path / "persistent_train_result.json"
    train_config.write_text(
        json.dumps(
            {
                "mode": "train",
                "paths": {
                    "train_matrix": "train.npz",
                    "train_labels": "train_labels.npy",
                    "checkpoint": checkpoint_path.name,
                },
                "training": {
                    "epochs": 2,
                    "hidden_sizes": [8, 4],
                    "batch_size": 2,
                    "seed": 42,
                },
                "num_classes": 2,
            }
        ),
        encoding="utf-8",
    )
    trained = subprocess.run(
        [
            str(interpreter),
            str(Path(worker.__file__).resolve()),
            "--mode",
            "train",
            "--config",
            str(train_config),
            "--output",
            str(train_output),
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )

    assert trained.returncode == 0, trained.stderr or trained.stdout
    assert checkpoint_path.is_file()
    train_result = json.loads(train_output.read_text(encoding="utf-8"))
    assert train_result["mode"] == "train"
    assert train_result["architecture"]["layer_sizes"] == [3, 8, 4, 2]

    predict_config = tmp_path / "persistent_predict.json"
    predict_output = tmp_path / "persistent_predict_result.json"
    predict_config.write_text(
        json.dumps(
            {
                "mode": "predict",
                "paths": {
                    "matrix": "val.npz",
                    "checkpoint": checkpoint_path.name,
                },
            }
        ),
        encoding="utf-8",
    )
    predicted = subprocess.run(
        [
            str(interpreter),
            str(Path(worker.__file__).resolve()),
            "--mode",
            "predict",
            "--config",
            str(predict_config),
            "--output",
            str(predict_output),
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )

    assert predicted.returncode == 0, predicted.stderr or predicted.stdout
    prediction_result = json.loads(predict_output.read_text(encoding="utf-8"))
    assert prediction_result["mode"] == "predict"
    assert len(prediction_result["predictions"]) == 2
    assert len(prediction_result["binary_probabilities"]) == 2
    assert prediction_result["architecture"]["layer_sizes"] == [3, 8, 4, 2]
