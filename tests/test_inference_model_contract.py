from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.contracts.inference import ProductionXGBoostModel


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "src" / "mcp" / "data" / "models"
ACTIVE_MODELS = {
    "detection": MODEL_DIR
    / "xgboost_detection_balanced_by_origin_20260905.joblib",
    "classifier": MODEL_DIR
    / "xgboost_attack_subtype_multidataset16_balanced500_20260906.joblib",
}
PROBE_ROWS = [
    {},
    {
        "app_proto": "http",
        "src_port": 443,
        "dst_port": 80,
        "packet_count": 120.0,
        "byte_count": 4096.0,
        "duration_ms": 1500.0,
    },
    {
        "app_proto": "mqtt",
        "src_port": 50000,
        "dst_port": 1883,
        "packet_count": 3.0,
        "byte_count": 180.0,
        "duration_ms": 20.0,
    },
]


@pytest.mark.parametrize(
    ("name", "expected_task", "expected_classes", "expected_predictions"),
    [
        (
            "detection",
            "binary_detection",
            [False, True],
            [True, True, True],
        ),
        (
            "classifier",
            "attack_subtype",
            [
                "Backdoor",
                "Command_and_Control",
                "DDoS_HTTP",
                "DDoS_ICMP",
                "DDoS_TCP",
                "DDoS_UDP",
                "DoS",
                "Fingerprinting",
                "MITM",
                "Password",
                "Port_Scanning",
                "Ransomware",
                "SQL_injection",
                "Uploading",
                "Vulnerability_scanner",
                "XSS",
            ],
            ["Password", "Password", "Password"],
        ),
    ],
)
def test_active_model_loads_from_runtime_contract_and_predicts(
    name: str,
    expected_task: str,
    expected_classes: list[bool | str],
    expected_predictions: list[bool | str],
):
    joblib = pytest.importorskip("joblib")
    pytest.importorskip("xgboost")

    model = joblib.load(ACTIVE_MODELS[name])

    assert isinstance(model, ProductionXGBoostModel)
    assert type(model).__module__ == "src.contracts.inference"
    assert model.task == expected_task
    assert model.classes == expected_classes
    if name == "detection":
        assert model.feature_schema_sha256 == (
            "de41afdd00f73945f34c071b99752d75339b88b7af1cb813439b383cca7c9c1b"
        )
        assert len(model.vectorizer.get_feature_names_out()) == 5264
    else:
        assert model.confidence_threshold == pytest.approx(0.65)
    assert model.predict(PROBE_ROWS) == expected_predictions
    probabilities = model.predict_proba(PROBE_ROWS)
    assert probabilities.shape == (len(PROBE_ROWS), len(expected_classes))
    assert np.all(np.isfinite(probabilities))
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_clean_subprocess_loads_active_models_without_training_modules():
    pytest.importorskip("joblib")
    pytest.importorskip("xgboost")
    code = r"""
import json
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[1])
import joblib

loaded = []
for raw_path in sys.argv[2:]:
    model = joblib.load(Path(raw_path))
    loaded.append(
        {
            "module": type(model).__module__,
            "task": model.task,
            "prediction": model.predict([{}])[0],
            "probability_columns": int(model.predict_proba([{}]).shape[1]),
        }
    )

forbidden = sorted(
    name
    for name in sys.modules
    if name == "src.eval"
    or name.startswith("src.eval.")
    or name == "src.agents.supervised_edge"
)
assert not forbidden, forbidden
assert all(item["module"] == "src.contracts.inference" for item in loaded)
print(json.dumps({"loaded": loaded, "forbidden": forbidden}, sort_keys=True))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            code,
            str(REPO_ROOT),
            *(str(path) for path in ACTIVE_MODELS.values()),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["forbidden"] == []
    assert {item["task"] for item in payload["loaded"]} == {
        "binary_detection",
        "attack_subtype",
    }
