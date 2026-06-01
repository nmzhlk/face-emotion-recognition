from pathlib import Path
from typing import Any, Dict, Generator, Tuple
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from app.core.celery_app import celery_app
from app.core.config import Settings, settings
from public.main import global_app


@pytest.fixture
def test_client() -> Generator[TestClient, None, None]:
    with TestClient(global_app) as client:
        yield client


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def override_settings(
    monkeypatch: MonkeyPatch, temp_dir: Path
) -> Generator[Settings, None, None]:
    original_settings = settings
    test_settings = Settings()
    test_settings.GLOBAL_TXT_PATH = str(temp_dir / "test_ingest.jsonl")
    test_settings.STATIC_API_KEY = "test_secret_key"
    test_settings.REDIS_HOST = "localhost"
    test_settings.REDIS_PORT = "6379"
    for key, value in test_settings.model_dump().items():
        monkeypatch.setattr(settings, key, value)
    yield settings
    for key in test_settings.model_dump().keys():
        monkeypatch.setattr(
            original_settings, key, getattr(original_settings, key)
        )


@pytest.fixture
def celery_eager() -> Generator[None, None, None]:
    celery_app.conf.update(
        task_always_eager=True,
        task_eager_propagates=True,
        broker_url="memory://",
        result_backend="cache+memory://",
    )
    yield
    celery_app.conf.update(
        task_always_eager=False,
        task_eager_propagates=False,
    )


@pytest.fixture
def minio_mock() -> Generator[Tuple[MagicMock, MagicMock], None, None]:
    with patch("app.core.minio_client.Minio") as mock_minio_class:
        mock_client = MagicMock()
        mock_minio_class.return_value = mock_client
        with patch("app.core.minio_client.get_minio_client") as mock_get:
            mock_get.return_value = (mock_client, "photos")
            yield mock_client, mock_get


@pytest.fixture
def test_image_bytes() -> bytes:
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.fixture
def mock_load_image() -> Generator[MagicMock, None, None]:
    img = np.ones((300, 300, 3), dtype=np.uint8) * 100
    with patch(
        "app.services.tasks.load_image_from_minio", return_value=img
    ) as mock:
        yield mock


@pytest.fixture
def mock_yolo_model() -> Generator[MagicMock, None, None]:
    mock_model = MagicMock()
    mock_box = MagicMock()
    mock_box.xyxy = [MagicMock()]
    mock_box.xyxy[0].tolist.return_value = [10, 20, 50, 60]
    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_model.return_value = [mock_result]
    with patch("app.services.tasks.YOLO", return_value=mock_model) as mock:
        yield mock


@pytest.fixture
def mock_recognizer() -> Generator[MagicMock, None, None]:
    embedding = np.array([0.1, 0.2, 0.3])
    mock_recognizer_instance = MagicMock()
    mock_recognizer_instance.return_value = ("test_uuid", 0.95, embedding)
    with patch(
        "ml.src.recognizer.FaceRecognizer",
        return_value=mock_recognizer_instance,
    ) as mock:
        yield mock


@pytest.fixture
def mock_emotion_model() -> Generator[Tuple[MagicMock, MagicMock], None, None]:
    mock_model = MagicMock(spec=torch.nn.Module)
    mock_model.forward = MagicMock(
        return_value=torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    )
    mock_model.return_value = mock_model.forward.return_value
    mock_transforms = MagicMock()
    with patch(
        "app.services.tasks.get_resnet_emotion_model",
        return_value=mock_model,
    ):
        with patch(
            "app.services.tasks.t.Compose", return_value=mock_transforms
        ):
            yield mock_model, mock_transforms


@pytest.fixture
def mock_delete_minio() -> Generator[MagicMock, None, None]:
    with patch("app.services.tasks.delete_minio_task_id") as mock:
        yield mock


@pytest.fixture
def sample_ingest_batch_payload() -> Dict[str, Any]:
    return {
        "edge_id": "test_edge",
        "batch_id": "batch_123",
        "camera_ids": ["0", "1"],
        "processed_count": 2,
        "received_at": None,
        "frames": [
            {
                "edge_id": "test_edge",
                "camera_id": "0",
                "frame_id": "frame_1",
                "timestamp": 1234567890,
                "items": [],
            },
            {
                "edge_id": "test_edge",
                "camera_id": "1",
                "frame_id": "frame_2",
                "timestamp": 1234567891,
                "items": [],
            },
        ],
    }


@pytest.fixture
def mock_tasks_models() -> Generator:
    from app.services import tasks

    original_model = tasks.model
    original_transforms = tasks.transforms
    tasks.model = MagicMock()
    tasks.transforms = MagicMock()
    yield
    tasks.model = original_model
    tasks.transforms = original_transforms


@pytest.fixture(autouse=True)
def mock_db_functions() -> Generator[None, Any, None]:
    with (
        patch("public.main.init_db_pool", return_value=None),
        patch("public.main.create_tables", return_value=None),
        patch("public.main.seed_admin_user", return_value=None),
        patch("public.main.save_batch_to_db", return_value=None),
        patch("public.main.save_frames_to_db", return_value=None),
    ):
        yield
