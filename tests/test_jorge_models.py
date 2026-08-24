from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

import src.eval.jorge_models as jorge_models
from src.eval.jorge_models import (
    BenchmarkRequirementsNotMet,
    compute_classification_metrics,
    run_jorge_benchmarks,
)


class _ScriptedEstimator:
    def __init__(self, validation_predictions, test_predictions=None):
        self.validation_predictions = validation_predictions
        self.test_predictions = (
            validation_predictions if test_predictions is None else test_predictions
        )
        self.predict_calls = 0
        self.closed = False

    def fit(self, x, y):
        self.fit_shape = x.shape
        self.fit_targets = list(y)
        return self

    def predict(self, x):
        self.predict_calls += 1
        if self.predict_calls == 1:
            return self.validation_predictions
        return self.test_predictions

    def close(self):
        self.closed = True


def _small_splits():
    train_features = [
        {"train_numeric": 0.0, "train_category": "a"},
        {"train_numeric": 1.0, "train_category": "b"},
        {"train_numeric": 2.0, "train_category": "a"},
        {"train_numeric": 3.0, "train_category": "b"},
    ]
    validation_features = [
        {"validation_only": float(index)} for index in range(4)
    ]
    test_features = [{"test_only": float(index)} for index in range(4)]
    targets = [0, 1, 0, 1]
    return train_features, targets, validation_features, targets, test_features, targets


def test_selection_uses_validation_not_test_and_vectorizer_fits_train_only(monkeypatch):
    scripted = {
        "random_forest": _ScriptedEstimator(
            validation_predictions=[0, 1, 0, 1],
            test_predictions=[1, 0, 1, 0],
        ),
        "xgboost": _ScriptedEstimator(
            validation_predictions=[0, 0, 0, 0],
            test_predictions=[0, 1, 0, 1],
        ),
    }

    def fake_build(model_name, **kwargs):
        return scripted[model_name], {"fake": True}

    monkeypatch.setattr(jorge_models, "_build_estimator", fake_build)
    outcome = run_jorge_benchmarks(
        *_small_splits(),
        task="detection",
        models=("xgboost", "random_forest"),
    )

    assert outcome.best_model_name == "random_forest"
    assert (
        outcome.report["models"]["xgboost"]["test"]["f1_weighted"]
        > outcome.report["models"]["random_forest"]["test"]["f1_weighted"]
    )
    feature_names = set(outcome.vectorizer.get_feature_names_out())
    assert feature_names == {
        "train_category=a",
        "train_category=b",
        "train_numeric",
    }
    assert all("validation_only" not in name for name in feature_names)
    assert all("test_only" not in name for name in feature_names)
    assert scripted["xgboost"].closed is True
    assert scripted["random_forest"].closed is False
    for model_name in scripted:
        timings = outcome.report["models"][model_name]["timings_seconds"]
        assert timings["fit"] >= 0.0
        assert timings["validation_predict"] >= 0.0
        assert timings["test_predict"] >= 0.0

    outcome.close()
    assert scripted["random_forest"].closed is True


def test_selection_ties_are_broken_by_accuracy_then_model_name(monkeypatch):
    estimators = {
        name: _ScriptedEstimator([0, 1, 0, 1])
        for name in ("xgboost", "random_forest")
    }

    def fake_build(model_name, **kwargs):
        return estimators[model_name], {}

    monkeypatch.setattr(jorge_models, "_build_estimator", fake_build)
    outcome = run_jorge_benchmarks(
        *_small_splits(),
        task="detection",
        models=("xgboost", "random_forest"),
    )

    assert outcome.best_model_name == "random_forest"


def test_metrics_include_weighted_macro_per_class_confusion_and_binary_values():
    metrics = compute_classification_metrics(
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        classes=["normal", "attack"],
        positive_index=1,
    )

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision_weighted"] == pytest.approx(5 / 6)
    assert metrics["recall_weighted"] == pytest.approx(0.75)
    assert metrics["f1_weighted"] == pytest.approx(11 / 15)
    assert metrics["macro"]["f1"] == pytest.approx(11 / 15)
    assert metrics["per_class"] == [
        {
            "label": "normal",
            "precision": 1.0,
            "recall": 0.5,
            "f1": pytest.approx(2 / 3),
            "support": 2,
        },
        {
            "label": "attack",
            "precision": pytest.approx(2 / 3),
            "recall": 1.0,
            "f1": 0.8,
            "support": 2,
        },
    ]
    assert metrics["confusion_matrix"] == {
        "labels": ["normal", "attack"],
        "matrix": [[1, 1], [0, 2]],
    }
    assert metrics["binary"] == {
        "positive_label": "attack",
        "negative_label": "normal",
        "tp": 2,
        "tn": 1,
        "fp": 1,
        "fn": 0,
        "accuracy": 0.75,
        "precision": pytest.approx(2 / 3),
        "recall": 1.0,
        "f1": 0.8,
        "specificity": 0.5,
    }


def test_missing_optional_dependencies_are_reported_without_stopping_rf(monkeypatch):
    def missing_import(name):
        raise ModuleNotFoundError(f"No module named {name}")

    monkeypatch.setattr(jorge_models, "import_module", missing_import)
    outcome = run_jorge_benchmarks(
        *_small_splits(),
        task="detection",
        parameter_overrides={"random_forest": {"n_estimators": 1}},
    )

    assert outcome.report["models"]["random_forest"]["status"] == "ok"
    for model_name, dependency in (
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("mlp", "torch"),
    ):
        model = outcome.report["models"][model_name]
        assert model["status"] == "unavailable"
        assert model["dependency"] == dependency
        assert f"Optional dependency '{dependency}' is unavailable" in model["message"]


def test_external_mlp_factory_bypasses_local_torch_and_receives_exact_defaults(monkeypatch):
    captured = {}

    def forbidden_import(name):
        raise AssertionError(f"local import must not be attempted: {name}")

    def factory(*, n_classes, seed, parameters):
        captured.update(n_classes=n_classes, seed=seed, parameters=dict(parameters))
        return _ScriptedEstimator([0, 1, 0, 1])

    monkeypatch.setattr(jorge_models, "import_module", forbidden_import)
    outcome = run_jorge_benchmarks(
        *_small_splits(),
        task="detection",
        models=("mlp",),
        mlp_backend_factory=factory,
    )

    assert outcome.report["models"]["mlp"]["status"] == "ok"
    assert captured == {
        "n_classes": 2,
        "seed": 42,
        "parameters": {
            "hidden_sizes": (256, 128, 64),
            "dropout": 0.3,
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 100,
            "random_state": 42,
            "n_jobs": 1,
        },
    }


def test_xgboost_and_lightgbm_defaults_match_the_tfm(monkeypatch):
    class CapturingClassifier:
        def __init__(self, **parameters):
            self.parameters = parameters

    modules = {
        "xgboost": SimpleNamespace(XGBClassifier=CapturingClassifier),
        "lightgbm": SimpleNamespace(LGBMClassifier=CapturingClassifier),
    }
    monkeypatch.setattr(jorge_models, "import_module", modules.__getitem__)

    xgb, xgb_parameters = jorge_models._build_estimator(
        "xgboost", n_classes=4, seed=42, overrides={}
    )
    lgbm, lgbm_parameters = jorge_models._build_estimator(
        "lightgbm", n_classes=4, seed=42, overrides={}
    )

    assert xgb.parameters == xgb_parameters
    assert xgb_parameters == {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": 1,
        "verbosity": 0,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": 4,
    }
    assert lgbm.parameters == lgbm_parameters
    assert lgbm_parameters == {
        "n_estimators": 300,
        "max_depth": -1,
        "learning_rate": 0.05,
        "num_leaves": 50,
        "min_data_in_leaf": 10,
        "subsample": 0.7,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": 1,
        "verbosity": -1,
    }


def test_test_features_and_targets_are_delayed_until_after_validation_selection(
    monkeypatch,
):
    estimator = _ScriptedEstimator([0, 1, 0, 1])

    def fake_build(model_name, **kwargs):
        del model_name, kwargs
        return estimator, {}

    class DelayedRows:
        def __iter__(self):
            assert estimator.predict_calls == 1
            return iter([{"test_only": float(index)} for index in range(4)])

    class DelayedTargets:
        def __iter__(self):
            assert estimator.predict_calls == 1
            return iter([0, 1, 0, 1])

    splits = _small_splits()
    monkeypatch.setattr(jorge_models, "_build_estimator", fake_build)
    outcome = run_jorge_benchmarks(
        splits[0],
        splits[1],
        splits[2],
        splits[3],
        DelayedRows(),
        DelayedTargets(),
        task="detection",
        models=("random_forest",),
    )

    assert outcome.best_model_name == "random_forest"
    assert outcome.report["split_sizes"]["test"] == 4
    assert outcome.report["test_evaluation"]["status"] == "ok"


def test_class_seen_only_in_test_preserves_selection_and_marks_test_unavailable(
    monkeypatch,
):
    estimator = _ScriptedEstimator([0, 1, 0, 1])

    def fake_build(model_name, **kwargs):
        del model_name, kwargs
        return estimator, {}

    splits = _small_splits()
    monkeypatch.setattr(jorge_models, "_build_estimator", fake_build)
    outcome = run_jorge_benchmarks(
        splits[0],
        splits[1],
        splits[2],
        splits[3],
        splits[4],
        [0, 1, 2, 2],
        task="detection",
        models=("random_forest",),
    )

    assert outcome.best_model_name == "random_forest"
    assert estimator.predict_calls == 1
    unavailable = outcome.report["models"]["random_forest"]["test_unavailable"]
    assert unavailable["reason"] == "target_class_absent_from_train"
    assert "not present in train" in unavailable["message"]
    assert outcome.report["test_evaluation"]["status"] == "unavailable"


def test_require_all_models_raises_with_complete_context_when_all_fail(monkeypatch):
    closed = []

    class FailingEstimator:
        def fit(self, x, y):
            del x, y
            raise RuntimeError("planned fit failure")

        def predict(self, x):  # pragma: no cover - fit always fails
            raise AssertionError(x)

        def close(self):
            closed.append(True)

    def fake_build(model_name, **kwargs):
        del model_name, kwargs
        return FailingEstimator(), {}

    monkeypatch.setattr(jorge_models, "_build_estimator", fake_build)
    with pytest.raises(BenchmarkRequirementsNotMet) as caught:
        run_jorge_benchmarks(
            *_small_splits(),
            task="detection",
            models=("random_forest", "xgboost"),
            require_all_models=True,
        )

    outcome = caught.value.outcome
    assert outcome.report["selection"] is None
    assert outcome.report["test_evaluation"] == {
        "status": "not_run",
        "reason": "no_model_passed_validation",
    }
    assert all(
        outcome.report["models"][name]["status"] == "failed"
        for name in ("random_forest", "xgboost")
    )
    assert len(closed) == 2
    assert "random_forest status" in str(caught.value)
    assert "validation selection is unavailable" in str(caught.value)


def test_local_torch_mlp_scaler_is_fitted_on_train_only_and_reused(monkeypatch):
    captured_batches = []

    class FakeTensor:
        def __init__(self, values):
            self.values = np.asarray(values)

        def tolist(self):
            return self.values.tolist()

        def argmax(self, dim):
            assert dim == 1
            return FakeTensor(np.argmax(self.values, axis=1))

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class FakeModule:
        def parameters(self):
            return []

        def train(self):
            return self

        def eval(self):
            return self

        def __call__(self, tensor):
            rows = len(tensor.values)
            return FakeTensor(np.column_stack([np.ones(rows), np.zeros(rows)]))

    class FakeLoss:
        def backward(self):
            return None

    class FakeOptimiser:
        def zero_grad(self, **kwargs):
            del kwargs

        def step(self):
            return None

    class FakeGenerator:
        def manual_seed(self, seed):
            del seed
            return self

    class NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            del args
            return False

    fake_torch = SimpleNamespace(
        long="long",
        manual_seed=lambda seed: None,
        Generator=lambda device: FakeGenerator(),
        get_num_threads=lambda: 1,
        set_num_threads=lambda value: None,
        randperm=lambda size, generator: FakeTensor(np.arange(size)),
        from_numpy=lambda values: (
            captured_batches.append(np.asarray(values).copy()) or FakeTensor(values)
        ),
        as_tensor=lambda values, dtype: FakeTensor(values),
        optim=SimpleNamespace(Adam=lambda parameters, lr: FakeOptimiser()),
        nn=SimpleNamespace(
            CrossEntropyLoss=lambda: (lambda logits, labels: FakeLoss())
        ),
        no_grad=lambda: NoGrad(),
    )
    monkeypatch.setattr(
        jorge_models,
        "_make_torch_mlp",
        lambda *args, **kwargs: FakeModule(),
    )
    model = jorge_models._TorchMLPClassifier(
        torch=fake_torch,
        n_classes=2,
        hidden_sizes=(256, 128, 64),
        dropout=0.3,
        learning_rate=0.001,
        batch_size=2,
        epochs=1,
        random_state=42,
        n_jobs=1,
    )
    x_train = sparse.csr_matrix([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y_train = np.asarray([0, 1, 0, 1], dtype=np.int64)
    x_test = sparse.csr_matrix([[100.0, 200.0], [300.0, 400.0]])

    model.fit(x_train, y_train)
    scaler_scale = model.scaler_.scale_.copy()
    assert model.scaler_.with_mean is False
    assert int(model.scaler_.n_samples_seen_) == 4
    captured_batches.clear()
    model.predict(x_test)

    expected = model.scaler_.transform(x_test).toarray().astype(np.float32)
    assert np.vstack(captured_batches) == pytest.approx(expected)
    assert model.scaler_.scale_ == pytest.approx(scaler_scale)
    assert int(model.scaler_.n_samples_seen_) == 4
