from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_engine() -> MagicMock:
    mock = MagicMock()
    dummy_img = (np.zeros((10, 10, 3))).astype(np.uint8)
    dummy_data = {
        "faces_count": 1,
        "faces": [
            {
                "bbox": [0, 0, 10, 10],
                "emotion": "Happy",
                "identity": "unknown",
                "identity_confidence": 0.0,
            }
        ],
    }
    mock.process_image.return_value = (dummy_img, dummy_data)
    app.state.engine = mock
    return mock


@pytest.fixture(autouse=True)
def mock_db(mocker: Any) -> None:
    mocker.patch("app.main.get_connection")
    mocker.patch("app.main.save_detection_results")
    mocker.patch("app.main.sync_embeddings")


def test_read_index() -> None:
    response = client.get("/")
    assert response.status_code == 200


def test_api_process_endpoint() -> None:
    file_content = b"fake image content"
    files = {"file": ("test.jpg", file_content, "image/jpeg")}

    response = client.post("/api/process", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "faces" in data
    assert data["faces_count"] == 1


def test_api_process_invalid_file_type() -> None:
    files = {"file": ("test.txt", b"just some text", "text/plain")}

    response = client.post("/api/process", files=files)

    assert response.status_code == 400
    assert response.json()["detail"] == "File provided is not an image"


def test_api_process_engine_error(mocker: Any) -> None:
    app.state.engine.process_image.side_effect = ValueError(
        "Could not decode image"
    )

    files = {"file": ("corrupted.jpg", b"bad_data", "image/jpeg")}

    response = client.post("/api/process", files=files)

    assert response.status_code == 400
    assert response.json()["detail"] == "Could not decode image"


def test_api_process_no_file() -> None:
    response = client.post("/api/process")
    assert response.status_code == 422
