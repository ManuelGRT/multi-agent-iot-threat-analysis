from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

import src.eval.external_mlp as external_mlp
from src.eval.external_mlp import ExternalTorchMLPFactory
from src.eval.jorge_models import run_jorge_benchmarks


def mlp_parameters(**overrides):
    values = {
        "hidden_sizes": (256, 128, 64),
        "dropout": 0.3,
        "learning_rate": 0.001,
        "batch_size": 2,
        "epochs": 1,
        "random_state": 42,
        "n_jobs": 1,
    }
    values.update(overrides)
    return values


class FakeWorker:
    def __init__(self):
        self.calls = []

    def __call__(self, command, **kwargs):
        mode = command[command.index("--mode") + 1]
        config_path = Path(command[command.index("--config") + 1])
        output_path = Path(command[command.index("--output") + 1])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert config["mode"] == mode
        assert not any("MISTRAL_API_KEY" == key for key in kwargs["env"])

        if mode == "train":
            matrix = sparse.load_npz(config["paths"]["train_matrix"])
            labels = np.load(config["paths"]["train_labels"], allow_pickle=False)
            checkpoint = Path(config["paths"]["checkpoint"])
            checkpoint.write_bytes(b"fake-state-dict")
            result = {
                "status": "ok",
                "mode": "train",
                "sample_count": int(matrix.shape[0]),
            }
            snapshot = {
                "mode": mode,
                "checkpoint": checkpoint,
                "matrix": matrix.copy(),
                "labels": labels.copy(),
            }
        else:
            matrix = sparse.load_npz(config["paths"]["matrix"])
            checkpoint = Path(config["paths"]["checkpoint"])
            predictions = (np.arange(matrix.shape[0]) % 2).astype(int).tolist()
            result = {
                "status": "ok",
                "mode": "predict",
                "predictions": predictions,
            }
            snapshot = {
                "mode": mode,
                "checkpoint": checkpoint,
                "matrix": matrix.copy(),
            }
        self.calls.append(snapshot)
        output_path.write_text(json.dumps(result), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


def test_factory_trains_once_scales_on_train_and_reuses_checkpoint(
    tmp_path, monkeypatch
):
    fake = FakeWorker()
    monkeypatch.setattr(external_mlp.subprocess, "run", fake)
    monkeypatch.setenv("MISTRAL_API_KEY", "must-not-reach-worker")
    temp_root = tmp_path / "owned-root"
    temp_root.mkdir()
    sibling = temp_root / "keep-me.txt"
    sibling.write_text("owned by caller", encoding="utf-8")
    x_train = sparse.csr_matrix(
        np.asarray([[1, 0], [3, 2], [5, 4], [7, 6]], dtype=np.float32)
    )
    y_train = np.asarray([0, 0, 1, 1], dtype=np.int64)

    with ExternalTorchMLPFactory(
        interpreter="fake-python",
        worker=tmp_path / "repo" / "scripts" / "worker.py",
        temp_root=temp_root,
    ) as factory:
        estimator = factory(n_classes=2, seed=42, parameters=mlp_parameters())
        owned_directory = estimator.work_dir
        estimator.fit(x_train, y_train)
        first = estimator.predict(sparse.csr_matrix([[2, 1], [6, 5]], dtype=np.float32))
        second = estimator.predict(sparse.csr_matrix([[4, 3]], dtype=np.float32))

        assert first.tolist() == [0, 1]
        assert second.tolist() == [0]
        assert int(estimator.scaler_.n_samples_seen_) == len(y_train)
        assert owned_directory.is_dir()
        assert estimator.checkpoint_path_.is_file()
        with pytest.raises(RuntimeError, match="only be called once"):
            estimator.fit(x_train, y_train)

    assert [call["mode"] for call in fake.calls] == ["train", "predict", "predict"]
    assert fake.calls[0]["labels"].tolist() == y_train.tolist()
    assert all(call["checkpoint"] == fake.calls[0]["checkpoint"] for call in fake.calls)
    assert not owned_directory.exists()
    assert sibling.read_text(encoding="utf-8") == "owned by caller"


def test_factory_is_directly_accepted_by_run_jorge_benchmarks(tmp_path, monkeypatch):
    fake = FakeWorker()
    monkeypatch.setattr(external_mlp.subprocess, "run", fake)
    train = [
        {"value": 0.0, "kind": "a"},
        {"value": 1.0, "kind": "b"},
        {"value": 2.0, "kind": "a"},
        {"value": 3.0, "kind": "b"},
    ]
    validation = [{"value": 4.0, "kind": "a"}, {"value": 5.0, "kind": "b"}]
    test = [{"value": 6.0, "kind": "a"}, {"value": 7.0, "kind": "b"}]

    with ExternalTorchMLPFactory(
        interpreter="fake-python",
        worker=tmp_path / "worker.py",
        temp_root=tmp_path,
    ) as factory:
        outcome = run_jorge_benchmarks(
            train,
            [0, 1, 0, 1],
            validation,
            [0, 1],
            test,
            [0, 1],
            task="detection",
            models=("mlp",),
            parameter_overrides={"mlp": {"epochs": 1, "batch_size": 2}},
            mlp_backend_factory=factory,
        )

        assert outcome.report["models"]["mlp"]["status"] == "ok"
        assert outcome.best_model_name == "mlp"
        assert outcome.report["models"]["mlp"]["test"]["accuracy"] == 1.0

    assert [call["mode"] for call in fake.calls] == ["train", "predict", "predict"]


def test_worker_failure_does_not_include_captured_secret(tmp_path, monkeypatch):
    secret = "do-not-echo-this-value"

    def failing_worker(command, **kwargs):
        assert secret not in kwargs["env"].values()
        return subprocess.CompletedProcess(command, 7, stdout=secret, stderr=secret)

    monkeypatch.setattr(external_mlp.subprocess, "run", failing_worker)
    monkeypatch.setenv("SOME_API_TOKEN", secret)
    factory = ExternalTorchMLPFactory(
        interpreter="fake-python",
        worker=tmp_path / "worker.py",
        temp_root=tmp_path,
    )
    estimator = factory(n_classes=2, seed=42, parameters=mlp_parameters())

    with pytest.raises(external_mlp.ExternalMLPError) as caught:
        estimator.fit(sparse.eye(4, format="csr"), np.asarray([0, 1, 0, 1]))

    assert secret not in str(caught.value)
    estimator.close()
    factory.close()


def test_autodetect_prefers_user_anaconda_when_present(tmp_path, monkeypatch):
    interpreter = tmp_path / "anaconda3" / "python.exe"
    interpreter.parent.mkdir()
    interpreter.write_bytes(b"")
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    assert external_mlp.detect_torch_interpreter() == interpreter.resolve()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"hidden_sizes": (8, 4)}, "hidden_sizes"),
        ({"batch_size": 1}, "batch_size"),
        ({"n_jobs": 2}, "n_jobs"),
        ({"random_state": 7}, "random_state"),
    ],
)
def test_factory_rejects_non_reproducible_architecture_parameters(
    tmp_path, overrides, message
):
    factory = ExternalTorchMLPFactory(
        interpreter="fake-python",
        worker=tmp_path / "worker.py",
        temp_root=tmp_path,
    )

    with pytest.raises((TypeError, ValueError), match=message):
        factory(n_classes=2, seed=42, parameters=mlp_parameters(**overrides))

    factory.close()
