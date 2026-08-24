"""Reproducible implementations of the four model families used in Jorge's TFM.

The module deliberately owns no dataset or filesystem concerns.  Callers provide
the already-sanitised feature dictionaries and the frozen train/validation/test
targets.  Feature and target encoders are fitted on train only, and model
selection is completed from validation metrics before the test split is scored.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler


SUPPORTED_MODELS = ("random_forest", "xgboost", "lightgbm", "mlp")


class MLPBackendFactory(Protocol):
    """Factory contract for an injected in-process or subprocess MLP backend.

    The returned object must expose sklearn-compatible ``fit(x, y)`` and
    ``predict(x)`` methods.  This keeps the evaluator usable when PyTorch lives
    in a different Python environment.
    """

    def __call__(
        self,
        *,
        n_classes: int,
        seed: int,
        parameters: Mapping[str, Any],
    ) -> Any: ...


class OptionalDependencyUnavailable(RuntimeError):
    """Raised when one optional benchmark backend cannot be imported."""

    def __init__(self, dependency: str, reason: str) -> None:
        self.dependency = dependency
        super().__init__(
            f"Optional dependency '{dependency}' is unavailable; "
            f"install it to run this benchmark ({reason})."
        )


class BenchmarkRequirementsNotMet(RuntimeError):
    """Strict-mode failure retaining the complete diagnostic outcome."""

    def __init__(
        self,
        outcome: "BenchmarkOutcome",
        failures: Sequence[str],
    ) -> None:
        self.outcome = outcome
        self.failures = tuple(failures)
        super().__init__("Benchmark requirements not met: " + "; ".join(self.failures))


@dataclass(slots=True)
class BenchmarkOutcome:
    """In-memory result retaining the selected estimator for later inference."""

    report: dict[str, Any]
    vectorizer: DictVectorizer
    target_encoder: LabelEncoder
    estimators: dict[str, Any]

    @property
    def best_model_name(self) -> str | None:
        selection = self.report.get("selection")
        return None if selection is None else str(selection["model"])

    @property
    def best_estimator(self) -> Any | None:
        name = self.best_model_name
        return None if name is None else self.estimators.get(name)

    def close(self) -> None:
        """Release resources owned by retained estimators, when supported."""

        for estimator in tuple(self.estimators.values()):
            _close_estimator(estimator)
        self.estimators.clear()

    def __enter__(self) -> "BenchmarkOutcome":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        del exc_type, exc, traceback
        self.close()
        return False


def run_jorge_benchmarks(
    train_features: Sequence[Mapping[str, Any]],
    y_train: Sequence[Any],
    validation_features: Sequence[Mapping[str, Any]],
    y_validation: Sequence[Any],
    test_features: Sequence[Mapping[str, Any]],
    y_test: Sequence[Any],
    task: str,
    *,
    models: Sequence[str] = SUPPORTED_MODELS,
    parameter_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    seed: int = 42,
    positive_label: Any | None = None,
    mlp_backend_factory: MLPBackendFactory | None = None,
    require_all_models: bool = False,
) -> BenchmarkOutcome:
    """Fit and evaluate Jorge's benchmark models without touching disk.

    The canonical defaults reproduce the hyperparameters documented in the TFM.
    ``parameter_overrides`` exists for small/fast validation runs (for example,
    fewer trees or MLP epochs in unit tests); every effective parameter is copied
    into the report.  Models run one after another and all supported estimators
    are constrained to one CPU worker.  ``mlp_backend_factory`` can return an
    sklearn-like estimator backed by another interpreter; when omitted, the
    local optional ``torch`` package is used.

    Selection is lexicographically deterministic: highest validation weighted
    F1, then highest validation accuracy, then ascending model name.  No test
    feature/target transformation, prediction or metric is computed until that
    choice has been made.  With ``require_all_models=True``, a diagnostic
    :class:`BenchmarkRequirementsNotMet` retaining the completed outcome is
    raised unless every requested backend passes validation and the selected
    winner also has test metrics.
    """

    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not isinstance(require_all_models, bool):
        raise TypeError("require_all_models must be a boolean")

    selected_models = tuple(models)
    unknown_models = sorted(set(selected_models).difference(SUPPORTED_MODELS))
    if unknown_models:
        raise ValueError(
            f"Unsupported models: {unknown_models}. Supported models: {list(SUPPORTED_MODELS)}"
        )
    if len(set(selected_models)) != len(selected_models):
        raise ValueError("models must not contain duplicates")

    overrides = {name: dict(values) for name, values in (parameter_overrides or {}).items()}
    unknown_override_models = sorted(set(overrides).difference(selected_models))
    if unknown_override_models:
        raise ValueError(
            "parameter_overrides contains models that were not requested: "
            f"{unknown_override_models}"
        )

    train_rows = _materialise_features("train", train_features)
    validation_rows = _materialise_features("validation", validation_features)
    train_targets = _materialise_targets("train", y_train, len(train_rows))
    validation_targets = _materialise_targets(
        "validation", y_validation, len(validation_rows)
    )

    if not train_rows:
        raise ValueError("train split must contain at least one sample")
    if not validation_rows:
        raise ValueError("validation split must contain at least one sample")

    vectorizer = DictVectorizer(sparse=True, sort=True)
    x_train = vectorizer.fit_transform(train_rows)
    x_validation = vectorizer.transform(validation_rows)

    target_encoder = LabelEncoder()
    y_train_encoded = target_encoder.fit_transform(train_targets)
    if len(target_encoder.classes_) < 2:
        raise ValueError("train split must contain at least two target classes")
    y_validation_encoded = _transform_known_targets(
        target_encoder, validation_targets, "validation"
    )

    classes = [_python_scalar(value) for value in target_encoder.classes_]
    positive_index = _resolve_positive_index(
        target_encoder,
        task=task,
        positive_label=positive_label,
    )

    report: dict[str, Any] = {
        "task": task,
        "seed": seed,
        "classes": classes,
        "n_features": int(x_train.shape[1]),
        "split_sizes": {
            "train": len(train_rows),
            "validation": len(validation_rows),
            # Deliberately unknown until validation-based selection is frozen.
            "test": None,
        },
        "selection_rule": (
            "validation f1_weighted desc, validation accuracy desc, model name asc"
        ),
        "models": {},
        "selection": None,
        "test_evaluation": {"status": "pending"},
    }
    estimators: dict[str, Any] = {}

    # First pass: fit and score validation only.  This ordering is intentional;
    # it makes it impossible for the test metrics to influence selection.
    for model_name in selected_models:
        model_report: dict[str, Any] = {
            "status": "pending",
            "timings_seconds": {},
        }
        report["models"][model_name] = model_report
        estimator: Any | None = None
        try:
            estimator, parameters = _build_estimator(
                model_name,
                n_classes=len(classes),
                seed=seed,
                overrides=overrides.get(model_name, {}),
                mlp_backend_factory=mlp_backend_factory,
            )
            model_report["parameters"] = parameters
            fit_started = time.perf_counter()
            try:
                estimator.fit(x_train, y_train_encoded)
            finally:
                model_report["timings_seconds"]["fit"] = (
                    time.perf_counter() - fit_started
                )
            predict_started = time.perf_counter()
            try:
                raw_validation_predictions = estimator.predict(x_validation)
            finally:
                model_report["timings_seconds"]["validation_predict"] = (
                    time.perf_counter() - predict_started
                )
            validation_predictions = _validated_predictions(
                raw_validation_predictions,
                expected_size=len(validation_rows),
                n_classes=len(classes),
                model_name=model_name,
                split="validation",
            )
            model_report["validation"] = compute_classification_metrics(
                y_validation_encoded,
                validation_predictions,
                classes=classes,
                positive_index=positive_index,
            )
            model_report["status"] = "ok"
            estimators[model_name] = estimator
        except OptionalDependencyUnavailable as exc:
            model_report.update(
                status="unavailable",
                dependency=exc.dependency,
                message=str(exc),
            )
        except Exception as exc:  # isolate one backend without hiding its cause
            model_report.update(
                status="failed",
                message=f"{type(exc).__name__}: {exc}",
            )
        finally:
            if model_report["status"] != "ok" and estimator is not None:
                _close_estimator(estimator)

    successful_names = [
        name for name in selected_models if report["models"][name]["status"] == "ok"
    ]
    if successful_names:
        best_name = min(
            successful_names,
            key=lambda name: (
                -report["models"][name]["validation"]["f1_weighted"],
                -report["models"][name]["validation"]["accuracy"],
                name,
            ),
        )
        best_validation = report["models"][best_name]["validation"]
        report["selection"] = {
            "model": best_name,
            "validation_f1_weighted": best_validation["f1_weighted"],
            "validation_accuracy": best_validation["accuracy"],
        }

    # Second pass: test objects are not even materialised/transformed until the
    # immutable validation-based selection above has been recorded.
    _score_test_after_selection(
        report=report,
        successful_names=successful_names,
        estimators=estimators,
        vectorizer=vectorizer,
        target_encoder=target_encoder,
        test_features=test_features,
        y_test=y_test,
        classes=classes,
        positive_index=positive_index,
    )

    best_name = None if report["selection"] is None else report["selection"]["model"]
    for model_name in tuple(estimators):
        if model_name != best_name:
            _close_estimator(estimators.pop(model_name))

    outcome = BenchmarkOutcome(
        report=report,
        vectorizer=vectorizer,
        target_encoder=target_encoder,
        estimators=estimators,
    )
    if require_all_models:
        validate_benchmark_outcome(outcome, requested_models=selected_models)
    return outcome


def validate_benchmark_outcome(
    outcome: BenchmarkOutcome,
    *,
    requested_models: Sequence[str],
) -> None:
    """Raise a contextual strict-mode error unless a benchmark is complete."""

    failures: list[str] = []
    model_reports = outcome.report.get("models", {})
    for model_name in requested_models:
        model_report = model_reports.get(model_name)
        status = None if not isinstance(model_report, Mapping) else model_report.get("status")
        if status != "ok":
            failures.append(f"{model_name} status is {status!r}, expected 'ok'")

    selection = outcome.report.get("selection")
    if not isinstance(selection, Mapping) or not selection.get("model"):
        failures.append("validation selection is unavailable")
    else:
        winner = str(selection["model"])
        winner_report = model_reports.get(winner, {})
        if not isinstance(winner_report, Mapping) or "test" not in winner_report:
            detail = None
            if isinstance(winner_report, Mapping):
                detail = winner_report.get("test_unavailable") or winner_report.get(
                    "test_error"
                )
            failures.append(
                f"selected model {winner!r} has no test metrics"
                + ("" if detail is None else f": {detail}")
            )

    if failures:
        raise BenchmarkRequirementsNotMet(outcome, failures)


def _score_test_after_selection(
    *,
    report: dict[str, Any],
    successful_names: Sequence[str],
    estimators: Mapping[str, Any],
    vectorizer: DictVectorizer,
    target_encoder: LabelEncoder,
    test_features: Sequence[Mapping[str, Any]],
    y_test: Sequence[Any],
    classes: Sequence[Any],
    positive_index: int | None,
) -> None:
    """Prepare and score test only after ``report['selection']`` is frozen."""

    if not successful_names:
        report["test_evaluation"] = {
            "status": "not_run",
            "reason": "no_model_passed_validation",
        }
        return

    try:
        test_rows = _materialise_features("test", test_features)
    except Exception as exc:
        _mark_test_issue(
            report,
            successful_names,
            field="test_error",
            status="error",
            reason="invalid_test_features",
            exc=exc,
        )
        return

    report["split_sizes"]["test"] = len(test_rows)
    if not test_rows:
        _mark_test_issue(
            report,
            successful_names,
            field="test_unavailable",
            status="unavailable",
            reason="empty_test_split",
            message="test split must contain at least one sample",
        )
        return

    try:
        test_targets = _materialise_targets("test", y_test, len(test_rows))
    except Exception as exc:
        _mark_test_issue(
            report,
            successful_names,
            field="test_unavailable",
            status="unavailable",
            reason="invalid_test_targets",
            exc=exc,
        )
        return

    try:
        y_test_encoded = _transform_known_targets(
            target_encoder, test_targets, "test"
        )
    except ValueError as exc:
        _mark_test_issue(
            report,
            successful_names,
            field="test_unavailable",
            status="unavailable",
            reason="target_class_absent_from_train",
            exc=exc,
        )
        return

    try:
        x_test = vectorizer.transform(test_rows)
    except Exception as exc:
        _mark_test_issue(
            report,
            successful_names,
            field="test_error",
            status="error",
            reason="test_feature_transform_failed",
            exc=exc,
        )
        return

    failed_models: list[str] = []
    for model_name in successful_names:
        estimator = estimators[model_name]
        model_report = report["models"][model_name]
        predict_started = time.perf_counter()
        try:
            raw_test_predictions = estimator.predict(x_test)
        except Exception as exc:
            failed_models.append(model_name)
            model_report["test_error"] = _issue_payload(
                "test_prediction_failed", exc=exc
            )
            continue
        finally:
            model_report["timings_seconds"]["test_predict"] = (
                time.perf_counter() - predict_started
            )

        try:
            test_predictions = _validated_predictions(
                raw_test_predictions,
                expected_size=len(test_rows),
                n_classes=len(classes),
                model_name=model_name,
                split="test",
            )
            model_report["test"] = compute_classification_metrics(
                y_test_encoded,
                test_predictions,
                classes=classes,
                positive_index=positive_index,
            )
        except Exception as exc:
            failed_models.append(model_name)
            model_report["test_error"] = _issue_payload(
                "invalid_test_predictions_or_metrics", exc=exc
            )

    report["test_evaluation"] = {
        "status": "ok" if not failed_models else "partial_failure",
        "scored_models": [
            name for name in successful_names if name not in failed_models
        ],
        "failed_models": failed_models,
    }


def _mark_test_issue(
    report: dict[str, Any],
    model_names: Sequence[str],
    *,
    field: str,
    status: str,
    reason: str,
    exc: Exception | None = None,
    message: str | None = None,
) -> None:
    payload = _issue_payload(reason, exc=exc, message=message)
    report["test_evaluation"] = {"status": status, **payload}
    for model_name in model_names:
        report["models"][model_name][field] = dict(payload)


def _issue_payload(
    reason: str,
    *,
    exc: Exception | None = None,
    message: str | None = None,
) -> dict[str, str]:
    if exc is not None:
        message = f"{type(exc).__name__}: {exc}"
    return {"reason": reason, "message": message or reason}


def _close_estimator(estimator: Any) -> None:
    """Best-effort cleanup for subprocess/resource-owning estimators."""

    close = getattr(estimator, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            # Cleanup must not replace the model's diagnostic failure.
            pass


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    classes: Sequence[Any],
    positive_index: int | None = None,
) -> dict[str, Any]:
    """Return JSON-compatible aggregate, per-class and confusion metrics."""

    truth = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(y_pred, dtype=np.int64)
    if truth.ndim != 1 or predicted.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional")
    if len(truth) != len(predicted):
        raise ValueError("y_true and y_pred must have equal lengths")
    if not len(truth):
        raise ValueError("metrics require at least one sample")
    if len(classes) < 2:
        raise ValueError("metrics require at least two classes")

    labels = np.arange(len(classes), dtype=np.int64)
    if np.any(truth < 0) or np.any(truth >= len(classes)):
        raise ValueError("y_true contains an encoded label outside classes")
    if np.any(predicted < 0) or np.any(predicted >= len(classes)):
        raise ValueError("y_pred contains an encoded label outside classes")

    weighted = precision_recall_fscore_support(
        truth,
        predicted,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    macro = precision_recall_fscore_support(
        truth,
        predicted,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    per_class = precision_recall_fscore_support(
        truth,
        predicted,
        labels=labels,
        average=None,
        zero_division=0,
    )
    matrix = confusion_matrix(truth, predicted, labels=labels)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision_weighted": float(weighted[0]),
        "recall_weighted": float(weighted[1]),
        "f1_weighted": float(weighted[2]),
        "macro": {
            "precision": float(macro[0]),
            "recall": float(macro[1]),
            "f1": float(macro[2]),
        },
        "per_class": [
            {
                "label": _python_scalar(classes[index]),
                "precision": float(per_class[0][index]),
                "recall": float(per_class[1][index]),
                "f1": float(per_class[2][index]),
                "support": int(per_class[3][index]),
            }
            for index in range(len(classes))
        ],
        "confusion_matrix": {
            "labels": [_python_scalar(label) for label in classes],
            "matrix": matrix.astype(int).tolist(),
        },
    }

    if len(classes) == 2:
        resolved_positive = 1 if positive_index is None else positive_index
        if resolved_positive not in (0, 1):
            raise ValueError("positive_index must identify one of the two classes")
        negative_index = 1 - resolved_positive
        tp = int(np.sum((truth == resolved_positive) & (predicted == resolved_positive)))
        tn = int(np.sum((truth == negative_index) & (predicted == negative_index)))
        fp = int(np.sum((truth == negative_index) & (predicted == resolved_positive)))
        fn = int(np.sum((truth == resolved_positive) & (predicted == negative_index)))
        binary_precision = _safe_div(tp, tp + fp)
        binary_recall = _safe_div(tp, tp + fn)
        metrics["binary"] = {
            "positive_label": _python_scalar(classes[resolved_positive]),
            "negative_label": _python_scalar(classes[negative_index]),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": _safe_div(tp + tn, len(truth)),
            "precision": binary_precision,
            "recall": binary_recall,
            "f1": _safe_div(2 * binary_precision * binary_recall, binary_precision + binary_recall),
            "specificity": _safe_div(tn, tn + fp),
        }

    return metrics


def _build_estimator(
    model_name: str,
    *,
    n_classes: int,
    seed: int,
    overrides: Mapping[str, Any],
    mlp_backend_factory: MLPBackendFactory | None = None,
) -> tuple[Any, dict[str, Any]]:
    if model_name == "random_forest":
        defaults: dict[str, Any] = {
            "n_estimators": 100,
            "random_state": seed,
            "n_jobs": 1,
        }
        parameters = _effective_parameters(model_name, defaults, overrides)
        return RandomForestClassifier(**parameters), parameters

    if model_name == "xgboost":
        xgboost = _load_optional_dependency("xgboost")
        defaults = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": seed,
            "n_jobs": 1,
            "verbosity": 0,
            "objective": "binary:logistic" if n_classes == 2 else "multi:softprob",
            "eval_metric": "logloss" if n_classes == 2 else "mlogloss",
        }
        if n_classes > 2:
            defaults["num_class"] = n_classes
        parameters = _effective_parameters(model_name, defaults, overrides)
        return xgboost.XGBClassifier(**parameters), parameters

    if model_name == "lightgbm":
        lightgbm = _load_optional_dependency("lightgbm")
        defaults = {
            "n_estimators": 300,
            "max_depth": -1,
            "learning_rate": 0.05,
            "num_leaves": 50,
            "min_data_in_leaf": 10,
            "subsample": 0.7,
            # LightGBM only applies bagging/subsampling when its frequency is
            # positive.  Without this Jorge parameter would be silently inert.
            "subsample_freq": 1,
            "colsample_bytree": 0.7,
            "class_weight": "balanced",
            "random_state": seed,
            "n_jobs": 1,
            "verbosity": -1,
        }
        parameters = _effective_parameters(model_name, defaults, overrides)
        return lightgbm.LGBMClassifier(**parameters), parameters

    if model_name == "mlp":
        defaults = {
            "hidden_sizes": (256, 128, 64),
            "dropout": 0.3,
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 100,
            "random_state": seed,
            "n_jobs": 1,
        }
        parameters = _effective_parameters(model_name, defaults, overrides)
        if mlp_backend_factory is not None:
            estimator = mlp_backend_factory(
                n_classes=n_classes,
                seed=seed,
                parameters=dict(parameters),
            )
            if not callable(getattr(estimator, "fit", None)) or not callable(
                getattr(estimator, "predict", None)
            ):
                raise TypeError(
                    "mlp_backend_factory must return an object with fit and predict methods"
                )
            return estimator, parameters

        torch = _load_optional_dependency("torch")
        estimator = _TorchMLPClassifier(
            torch=torch,
            n_classes=n_classes,
            **parameters,
        )
        return estimator, parameters

    raise ValueError(f"Unsupported model: {model_name}")


def _effective_parameters(
    model_name: str,
    defaults: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    unknown = sorted(set(overrides).difference(defaults))
    if unknown:
        raise ValueError(f"Unsupported {model_name} parameter overrides: {unknown}")
    protected = sorted(set(overrides).intersection({"random_state", "n_jobs"}))
    if protected:
        raise ValueError(
            f"{model_name} overrides cannot change deterministic settings: {protected}"
        )
    result = dict(defaults)
    result.update(overrides)
    return result


class _TorchMLPClassifier:
    """Small sklearn-like CPU wrapper around Jorge's exact PyTorch MLP."""

    def __init__(
        self,
        *,
        torch: Any,
        n_classes: int,
        hidden_sizes: Sequence[int],
        dropout: float,
        learning_rate: float,
        batch_size: int,
        epochs: int,
        random_state: int,
        n_jobs: int,
    ) -> None:
        if tuple(hidden_sizes) != (256, 128, 64):
            raise ValueError("Jorge's MLP hidden_sizes must remain (256, 128, 64)")
        if n_jobs != 1:
            raise ValueError("MLP n_jobs must remain 1")
        if batch_size < 2:
            raise ValueError("MLP batch_size must be at least 2 for BatchNorm")
        if epochs < 1:
            raise ValueError("MLP epochs must be at least 1")
        self._torch = torch
        self.n_classes = n_classes
        self.hidden_sizes = tuple(hidden_sizes)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.random_state = int(random_state)
        self.n_jobs = n_jobs
        self.module_: Any | None = None
        self.scaler_: StandardScaler | None = None
        self.n_features_in_: int | None = None

    def fit(self, x: Any, y: np.ndarray) -> "_TorchMLPClassifier":
        # Match the external bridge exactly: fit scaling statistics on train
        # only, then reuse the frozen scaler for validation and test.
        self.scaler_ = StandardScaler(with_mean=False)
        scaled_x = self.scaler_.fit_transform(x)
        self.n_features_in_ = int(scaled_x.shape[1])
        torch = self._torch
        torch.manual_seed(self.random_state)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.random_state)
        self.module_ = _make_torch_mlp(
            torch,
            input_size=self.n_features_in_,
            n_classes=self.n_classes,
            dropout=self.dropout,
        )
        optimiser = torch.optim.Adam(self.module_.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            self.module_.train()
            for _ in range(self.epochs):
                permutation = torch.randperm(len(y), generator=generator).tolist()
                for indexes in _non_singleton_batches(permutation, self.batch_size):
                    dense = np.asarray(scaled_x[indexes].toarray(), dtype=np.float32)
                    batch_x = torch.from_numpy(dense)
                    batch_y = torch.as_tensor(y[indexes], dtype=torch.long)
                    optimiser.zero_grad(set_to_none=True)
                    loss = criterion(self.module_(batch_x), batch_y)
                    loss.backward()
                    optimiser.step()
        finally:
            torch.set_num_threads(previous_threads)
        return self

    def predict(self, x: Any) -> np.ndarray:
        if self.module_ is None or self.scaler_ is None or self.n_features_in_ is None:
            raise RuntimeError("MLP must be fitted before predict")
        if int(x.shape[1]) != self.n_features_in_:
            raise ValueError(f"x has {x.shape[1]} features; expected {self.n_features_in_}")
        scaled_x = self.scaler_.transform(x)
        torch = self._torch
        predictions: list[np.ndarray] = []
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            self.module_.eval()
            with torch.no_grad():
                for start in range(0, int(scaled_x.shape[0]), self.batch_size):
                    stop = min(start + self.batch_size, int(scaled_x.shape[0]))
                    dense = np.asarray(
                        scaled_x[start:stop].toarray(), dtype=np.float32
                    )
                    logits = self.module_(torch.from_numpy(dense))
                    predictions.append(logits.argmax(dim=1).cpu().numpy())
        finally:
            torch.set_num_threads(previous_threads)
        return np.concatenate(predictions).astype(np.int64, copy=False)


def _make_torch_mlp(torch: Any, *, input_size: int, n_classes: int, dropout: float) -> Any:
    """Build input -> 256 -> 128 -> 64 -> output with BN and dropout 0.3."""

    return torch.nn.Sequential(
        torch.nn.Linear(input_size, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(256, 128),
        torch.nn.BatchNorm1d(128),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(128, 64),
        torch.nn.BatchNorm1d(64),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(64, n_classes),
    )


def _non_singleton_batches(indexes: list[int], batch_size: int) -> list[list[int]]:
    batches = [indexes[start : start + batch_size] for start in range(0, len(indexes), batch_size)]
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2].extend(batches.pop())
    return batches


def _load_optional_dependency(name: str) -> Any:
    try:
        return import_module(name)
    except Exception as exc:
        raise OptionalDependencyUnavailable(name, f"{type(exc).__name__}: {exc}") from exc


def _materialise_features(
    split: str, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    try:
        materialised = list(rows)
    except TypeError as exc:
        raise TypeError(f"{split}_features must be an iterable of mappings") from exc
    result: list[dict[str, Any]] = []
    for index, row in enumerate(materialised):
        if not isinstance(row, Mapping):
            raise TypeError(f"{split}_features[{index}] must be a mapping")
        result.append(dict(row))
    return result


def _materialise_targets(split: str, targets: Sequence[Any], expected: int) -> list[Any]:
    try:
        result = list(targets)
    except TypeError as exc:
        raise TypeError(f"y_{split} must be an iterable") from exc
    if len(result) != expected:
        raise ValueError(
            f"{split} features/targets length mismatch: {expected} != {len(result)}"
        )
    return result


def _transform_known_targets(
    encoder: LabelEncoder, targets: Sequence[Any], split: str
) -> np.ndarray:
    try:
        return encoder.transform(targets)
    except ValueError as exc:
        raise ValueError(
            f"{split} contains a target class not present in train: {exc}"
        ) from exc


def _resolve_positive_index(
    encoder: LabelEncoder,
    *,
    task: str,
    positive_label: Any | None,
) -> int | None:
    if len(encoder.classes_) != 2:
        if positive_label is not None:
            raise ValueError("positive_label is only valid for a binary target")
        return None
    if positive_label is not None:
        try:
            return int(encoder.transform([positive_label])[0])
        except ValueError as exc:
            raise ValueError("positive_label is not present in train targets") from exc

    classes = [_python_scalar(value) for value in encoder.classes_]
    preferred: tuple[Any, ...] = (
        True,
        1,
        "1",
        "true",
        "attack",
        "attacked",
        "malicious",
        "anomaly",
        "positive",
    )
    for candidate in preferred:
        for index, label in enumerate(classes):
            if label == candidate:
                return index
            if isinstance(label, str) and label.strip().lower() == str(candidate).lower():
                return index

    # LabelEncoder's second class is the conventional positive class.  The task
    # name is retained in the report so callers can override this when semantics
    # differ (for example a label ordered before "normal").
    del task
    return 1


def _validated_predictions(
    predictions: Any,
    *,
    expected_size: int,
    n_classes: int,
    model_name: str,
    split: str,
) -> np.ndarray:
    result = np.asarray(predictions)
    if result.ndim != 1 or len(result) != expected_size:
        raise ValueError(
            f"{model_name} returned invalid {split} prediction shape {result.shape}; "
            f"expected ({expected_size},)"
        )
    if not np.issubdtype(result.dtype, np.integer):
        if not np.all(np.equal(result, np.floor(result))):
            raise ValueError(f"{model_name} returned non-integral encoded predictions")
    result = result.astype(np.int64, copy=False)
    if np.any(result < 0) or np.any(result >= n_classes):
        raise ValueError(f"{model_name} returned a prediction outside the trained classes")
    return result


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
