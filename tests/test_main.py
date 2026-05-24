import json
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import settings


def test_index(test_client: TestClient) -> None:
    response = test_client.get("/")
    assert response.status_code == 200
    assert "edge-global ingest service" in response.text


def test_auth(test_client: TestClient) -> None:
    response = test_client.post(
        "/auth", json={"user": "test", "password": "test"}
    )
    assert response.status_code == 200
    assert response.json() == {"status": 200, "user_id": "master"}


def test_register(test_client: TestClient) -> None:
    response = test_client.post(
        "/register", json={"user": "test", "password": "test"}
    )
    assert response.status_code == 200
    assert response.json() == {"status": 200, "user_id": "master"}


def test_ingest_batch_no_api_key(
    test_client: TestClient, sample_ingest_batch_payload: dict
) -> None:
    payload = sample_ingest_batch_payload.copy()
    response = test_client.post("/api/ingest_batch", json=payload)
    assert response.status_code == 401


def test_ingest_batch_wrong_api_key(
    test_client: TestClient, sample_ingest_batch_payload: dict
) -> None:
    payload = sample_ingest_batch_payload.copy()
    response = test_client.post(
        "/api/ingest_batch",
        headers={"X-Secret-Api-Key": "wrong_key"},
        json=payload,
    )
    assert response.status_code == 401


def test_ingest_batch_success(
    test_client: TestClient,
    sample_ingest_batch_payload: dict,
    override_settings: None,
    temp_dir: Path,
) -> None:
    payload = sample_ingest_batch_payload.copy()
    response = test_client.post(
        "/api/ingest_batch",
        headers={"X-Secret-Api-Key": settings.STATIC_API_KEY},
        json=payload,
    )
    assert response.status_code == 200
    assert response.json()["batch_id"] == "batch_123"

    log_file = Path(settings.GLOBAL_TXT_PATH)
    assert log_file.exists()
    with open(log_file, "r") as f:
        line = f.readline()
        data = json.loads(line)
    assert data["edge_id"] == "test_edge"
    assert data["batch_id"] == "batch_123"
    assert len(data["frames"]) == 2


def test_ingest_batch_sets_received_at(
    test_client: TestClient,
    sample_ingest_batch_payload: dict,
    override_settings: None,
    temp_dir: Path,
) -> None:
    payload = sample_ingest_batch_payload.copy()
    payload["received_at"] = None  # явно передаём None
    response = test_client.post(
        "/api/ingest_batch",
        headers={"X-Secret-Api-Key": settings.STATIC_API_KEY},
        json=payload,
    )
    assert response.status_code == 200
    log_file = Path(settings.GLOBAL_TXT_PATH)
    with open(log_file, "r") as f:
        data = json.loads(f.readline())
    assert data["received_at"] is not None


def test_ingest_batch_preserves_received_at(
    test_client: TestClient,
    sample_ingest_batch_payload: dict,
    override_settings: None,
    temp_dir: Path,
) -> None:
    payload = sample_ingest_batch_payload.copy()
    provided_time = datetime.now().isoformat()
    payload["received_at"] = provided_time
    response = test_client.post(
        "/api/ingest_batch",
        headers={"X-Secret-Api-Key": settings.STATIC_API_KEY},
        json=payload,
    )
    assert response.status_code == 200
    log_file = Path(settings.GLOBAL_TXT_PATH)
    with open(log_file, "r") as f:
        data = json.loads(f.readline())
    assert data["received_at"] == provided_time
