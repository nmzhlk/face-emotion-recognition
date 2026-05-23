from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from app.services import tasks
from app.services.tasks import merge_results


@pytest.fixture(autouse=True)
def reset_global_models() -> Generator:
    tasks.model = None
    tasks.transforms = None
    yield
    tasks.model = None
    tasks.transforms = None


def test_yolo_task(
    mock_load_image: MagicMock, mock_yolo_model: MagicMock, celery_eager: None
) -> None:
    mock_yolo_instance = MagicMock()
    mock_result = MagicMock()
    mock_box = MagicMock()
    mock_box.xyxy = [[10, 20, 50, 60]]
    mock_result.boxes = [mock_box]
    mock_yolo_instance.return_value = [mock_result]  # вместо __call__
    mock_yolo_model.return_value = mock_yolo_instance
    tasks.model = mock_yolo_instance
    result = tasks.yolo("test_path.jpg")
    assert result["path"] == "test_path.jpg"
    assert len(result["faces"]) == 1
    assert result["faces"][0] == [10, 20, 50, 60]
    mock_load_image.assert_called_once_with("test_path.jpg")


def test_recognizer_task(
    mock_load_image: MagicMock, mock_recognizer: MagicMock, celery_eager: None
) -> None:
    mock_recognizer_instance = MagicMock()
    mock_recognizer_instance.extract_embedding = MagicMock(
        return_value=np.array([0.1, 0.2, 0.3])
    )
    mock_recognizer_instance.return_value = (
        "test_uuid",
        0.95,
        np.array([0.1, 0.2, 0.3]),
    )
    mock_recognizer.return_value = mock_recognizer_instance
    tasks.model = mock_recognizer_instance
    input_data = {"path": "test.jpg", "faces": [[10, 20, 50, 60]]}
    result = tasks.recognizer(input_data)
    assert result["type"] == "recognizer"
    data = result["data"]
    assert data["path"] == "test.jpg"
    assert len(data["identities"]) == 1
    assert data["identities"][0]["bbox"] == [10, 20, 50, 60]
    assert data["identities"][0]["identity"] == "test_uuid"
    mock_recognizer_instance.assert_called_once()


def test_emotions_task(
    mock_load_image: MagicMock, mock_emotion_model: tuple, celery_eager: None
) -> None:
    mock_model, mock_transforms = mock_emotion_model
    mock_model_instance = MagicMock(spec=torch.nn.Module)
    mock_model_instance.return_value = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    )
    mock_model.return_value = mock_model_instance
    tasks.model = mock_model_instance
    tasks.transforms = mock_transforms
    input_data = {"path": "test.jpg", "faces": [[10, 20, 50, 60]]}
    result = tasks.emotions(input_data)
    assert result["type"] == "emotions"
    data = result["data"]
    assert len(data["emotions"]) == 1
    assert data["emotions"][0]["bbox"] == [10, 20, 50, 60]
    assert data["emotions"][0]["emotion"] in [
        "Anger",
        "Contempt",
        "Disgust",
        "Fear",
        "Happy",
        "Neutral",
        "Sad",
        "Surprise",
    ]
    mock_model_instance.assert_called_once()


@patch("app.core.minio_client.delete_minio_task_id")
def test_merge_results_success(mock_delete_minio: MagicMock) -> None:
    recognizer_data = {
        "path": "test.jpg",
        "faces": [[10, 20, 50, 60]],
        "identities": [
            {
                "bbox": [10, 20, 50, 60],
                "identity": "user123",
                "identity_confidence": 0.98,
                "embedding": [0.1, 0.2, 0.3],
            }
        ],
    }
    emotions_data = {
        "path": "test.jpg",
        "faces": [[10, 20, 50, 60]],
        "emotions": [
            {"bbox": [10, 20, 50, 60], "emotion": "Happy", "confidence": 0.85}
        ],
    }
    results = [
        {"type": "recognizer", "data": recognizer_data},
        {"type": "emotions", "data": emotions_data},
    ]
    stream_data = {
        "user_id": "edge1",
        "stream_id": "cam0",
        "frame_id": "frame123",
        "timestamp": 123456,
        "store_path": "test.jpg",
    }
    payload = merge_results(results, stream_data)
    assert payload["user_id"] == "edge1"
    assert payload["frame_id"] == "frame123"
    assert len(payload["items"]) == 1
    assert payload["items"][0]["identity"] == "user123"
    assert payload["items"][0]["emotion"] == "Happy"
    mock_delete_minio.assert_called_once()


def test_merge_results_mismatched_bbox() -> None:
    recognizer_data = {
        "path": "test.jpg",
        "faces": [[10, 20, 50, 60]],
        "identities": [
            {
                "bbox": [10, 20, 50, 60],
                "identity": "user123",
                "identity_confidence": 0.98,
                "embedding": None,
            }
        ],
    }
    emotions_data = {
        "path": "test.jpg",
        "faces": [[10, 20, 50, 60]],
        "emotions": [
            {
                "bbox": [100, 200, 150, 160],
                "emotion": "Anger",
                "confidence": 0.9,
            }
        ],
    }
    results = [
        {"type": "recognizer", "data": recognizer_data},
        {"type": "emotions", "data": emotions_data},
    ]
    stream_data = {
        "user_id": "e",
        "stream_id": "c",
        "frame_id": "f",
        "timestamp": 0,
        "store_path": "p",
    }
    with pytest.raises(ValueError, match="Can't find emotion"):
        merge_results(results, stream_data)
