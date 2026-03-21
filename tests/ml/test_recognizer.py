from typing import Any

import numpy as np

from ml.src.recognizer import FaceRecognizer


def test_recognize_face_empty_db() -> None:
    recognizer = FaceRecognizer()
    dummy_emb = np.random.rand(512).astype(np.float32)

    label, score = recognizer.recognize_face(dummy_emb)

    assert label == "unknown"
    assert score == 0.0


def test_recognize_face_with_data() -> None:
    recognizer = FaceRecognizer()
    known_uuid = "user-123"
    known_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    recognizer.set_embeddings({known_uuid: known_emb})

    label, score = recognizer.recognize_face(known_emb)

    assert label == known_uuid
    assert score > 0.9


def test_recognize_face_below_threshold() -> None:
    recognizer = FaceRecognizer()
    known_uuid = "user-123"
    known_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    query_emb = np.array([0.1, 0.9, 0.0], dtype=np.float32)

    recognizer.set_embeddings({known_uuid: known_emb})

    label, score = recognizer.recognize_face(query_emb, threshold=0.45)

    assert label == "unknown"
    assert score < 0.45


def test_extract_embedding_error(mocker: Any) -> None:
    recognizer = FaceRecognizer()
    mocker.patch(
        "ml.src.recognizer.DeepFace.represent", side_effect=Exception("DeepFace Error")
    )

    dummy_roi = np.zeros((100, 100, 3), dtype=np.uint8)
    embedding = recognizer.extract_embedding(dummy_roi)

    assert embedding is None


def test_process_face_full_cycle(mocker: Any) -> None:
    recognizer = FaceRecognizer()
    known_uuid = "valid-user"
    known_emb = np.random.rand(512).astype(np.float32)
    recognizer.set_embeddings({known_uuid: known_emb})

    mocker.patch.object(recognizer, "extract_embedding", return_value=known_emb)

    dummy_roi = np.zeros((100, 100, 3), dtype=np.uint8)
    label, score, emb_out = recognizer.process_face(dummy_roi)

    assert label == known_uuid
    assert score > 0.99

    assert emb_out is not None, "Recognizer returned None instead of embeddings"
    assert np.array_equal(emb_out, known_emb)


def test_extract_embedding_no_faces_found(mocker: Any) -> None:
    recognizer = FaceRecognizer()
    mocker.patch("ml.src.recognizer.DeepFace.represent", return_value=[])

    dummy_roi = np.zeros((100, 100, 3), dtype=np.uint8)
    embedding = recognizer.extract_embedding(dummy_roi)

    assert embedding is None
