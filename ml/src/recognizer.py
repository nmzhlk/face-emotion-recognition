from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from deepface import DeepFace
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity


class FaceRecognizer:
    def __init__(
        self, model_name: str = "Facenet", detector_backend: str = "opencv"
    ) -> None:
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.embeddings_db: Dict[str, NDArray[Any]] = {}

    def set_embeddings(self, data: Dict[str, NDArray[Any]]) -> None:
        self.embeddings_db = data

    def extract_embedding(
        self, face_roi: NDArray[Any]
    ) -> Optional[NDArray[Any]]:
        try:
            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

            result = DeepFace.represent(
                rgb_face,
                model_name=self.model_name,
                detector_backend="skip",
                enforce_detection=False,
            )

            if result and len(result) > 0:
                return np.array(result[0]["embedding"], dtype=np.float32)
            return None

        except Exception as e:
            print(f"Error during embedding extraction: {e}")
            return None

    def recognize_face(
        self, embedding: NDArray[Any], threshold: float = 0.45
    ) -> Tuple[str, float]:
        if not self.embeddings_db:
            return "unknown", 0.0

        best_uuid: str = "unknown"
        max_similarity: float = 0.0

        for human_uuid, stored_emb in self.embeddings_db.items():
            similarity = float(
                cosine_similarity([embedding], [stored_emb])[0][0]
            )

            if similarity > max_similarity:
                max_similarity = similarity
                best_uuid = human_uuid

        if max_similarity < threshold:
            return "unknown", max_similarity

        return best_uuid, max_similarity

    def process_face(
        self, face_roi: NDArray[Any]
    ) -> Tuple[str, float, Optional[NDArray[Any]]]:
        embedding = self.extract_embedding(face_roi)
        if embedding is None:
            return "unknown", 0.0, None

        human_uuid, score = self.recognize_face(embedding)
        return human_uuid, score, embedding
