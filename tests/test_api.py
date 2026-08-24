from fastapi.testclient import TestClient

from src.api.app import app
from tests.test_orchestrator import malicious_iot23_input


def test_health_endpoint():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_endpoint_runs_pipeline():
    client = TestClient(app)

    response = client.post("/events/analyze", json=malicious_iot23_input())

    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "end"
    assert payload["judge_output"]["action"] == "approve"


def test_analyze_endpoint_accepts_generic_text():
    client = TestClient(app)

    response = client.post(
        "/events/analyze",
        json={
            "dataset": "future_dataset",
            "use_llm": False,
            "text": "src_ip=10.0.0.1 dst_ip=10.0.0.2 proto=tcp label=normal",
            "row_id": 1,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["canonical_event"]["provenance"]["dataset"] == "future_dataset"
    assert payload["route"] == "judge"
    assert payload["detection_output"]["abstain"] is True


def test_analyze_file_endpoint_reads_csv(tmp_path, monkeypatch):
    client = TestClient(app)
    monkeypatch.setenv("TFM_DATA_DIR", str(tmp_path))
    dataset_file = tmp_path / "iot23.csv"
    dataset_file.write_text(
        "id.orig_h,id.resp_h,id.orig_p,id.resp_p,proto,label\n"
        "10.0.0.2,10.0.0.3,4444,23,tcp,Mirai\n",
        encoding="utf-8",
    )

    response = client.post(
        "/datasets/analyze-file",
        json={"path": str(dataset_file), "dataset": "iot23", "limit": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["results"][0]["classification_output"]["attack_family"] == "bruteforce"


def test_analyze_file_rejects_paths_outside_data_root(tmp_path, monkeypatch):
    client = TestClient(app)
    data_root = tmp_path / "data"
    data_root.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("value\nsecret\n", encoding="utf-8")
    monkeypatch.setenv("TFM_DATA_DIR", str(data_root))

    absolute = client.post(
        "/datasets/analyze-file",
        json={"path": str(outside), "dataset": "generic"},
    )
    traversal = client.post(
        "/datasets/analyze-file",
        json={"path": "../outside.csv", "dataset": "generic"},
    )

    assert absolute.status_code == 400
    assert traversal.status_code == 400


def test_analyze_file_llm_is_explicit_opt_in(tmp_path, monkeypatch):
    client = TestClient(app)
    dataset_file = tmp_path / "events.csv"
    dataset_file.write_text("value\n1\n", encoding="utf-8")
    monkeypatch.setenv("TFM_DATA_DIR", str(tmp_path))
    received: list[bool] = []

    class StubOrchestrator:
        def __init__(self, *, use_llm, **_kwargs):
            received.append(use_llm)

        async def ainvoke(self, _state):
            return {"route": "end"}

    monkeypatch.setattr("src.api.routers.AsyncGraphOrchestrator", StubOrchestrator)

    default_response = client.post(
        "/datasets/analyze-file",
        json={"path": str(dataset_file), "limit": 1},
    )
    opt_in_response = client.post(
        "/datasets/analyze-file",
        json={"path": str(dataset_file), "limit": 1, "use_llm": True},
    )

    assert default_response.status_code == 200
    assert opt_in_response.status_code == 200
    assert received == [False, True]
