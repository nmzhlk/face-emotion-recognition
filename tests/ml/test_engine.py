from typing import Any, cast
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from ml.src.engine import EmotionEngine


@pytest.fixture
def engine(mocker: Any) -> EmotionEngine:
    mocker.patch("ml.src.engine.YOLO")
    mocker.patch("ml.src.engine.get_resnet_emotion_model")
    mocker.patch("ml.src.engine.FaceRecognizer")
    return EmotionEngine(yolo_path="fake.pt", resnet_path="fake.pth", device="cpu")


def create_image_bytes(width: int, height: int) -> Any:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()


def test_process_no_faces_found(mocker: Any, engine: EmotionEngine) -> None:
    mock_results = mocker.MagicMock()
    cast(MagicMock, mock_results).boxes = []
    cast(MagicMock, engine.detector).return_value = [mock_results]

    image_bytes = create_image_bytes(640, 480)
    img_out, results = engine.process_image(image_bytes)

    assert results["faces_count"] == 0
    assert len(results["faces"]) == 0
    assert img_out.shape == (480, 640, 3)


def test_process_tiny_image(mocker: Any, engine: EmotionEngine) -> None:
    mock_results = mocker.MagicMock()
    cast(MagicMock, mock_results).boxes = []
    cast(MagicMock, engine.detector).return_value = [mock_results]

    tiny_image = create_image_bytes(1, 1)
    img_out, results = engine.process_image(tiny_image)

    assert results["faces_count"] == 0
    assert img_out.shape == (1, 1, 3)


def test_process_corrupted_data(engine: EmotionEngine) -> None:
    invalid_data = b"this is not a jpeg image content"

    with pytest.raises(ValueError, match="Could not decode image"):
        engine.process_image(invalid_data)


def test_process_face_on_edge(mocker: Any, engine: EmotionEngine) -> None:
    mock_box = mocker.MagicMock()
    cast(Any, mock_box).xyxy = [np.array([10, 10, 10, 50])]

    mock_results = mocker.MagicMock()
    cast(MagicMock, mock_results).boxes = [mock_box]
    cast(MagicMock, engine.detector).return_value = [mock_results]

    image_bytes = create_image_bytes(100, 100)
    _, results = engine.process_image(image_bytes)

    assert results["faces_count"] == 0
