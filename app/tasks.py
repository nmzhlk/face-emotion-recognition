from typing import Any, Callable, Dict, List, Union

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as t
from celery import chain, chord
from celery.signals import worker_process_init

# from numpy.typing import NDArray
from PIL import Image
from ultralytics import YOLO

from app.celery_app import celery_app
from app.config import settings
from ml.src.recognizer import FaceRecognizer
from ml.src.resnet import EMOTION_LABELS, get_resnet_emotion_model

# from ml.src.engine import EmotionEngine


ModelType = Union[
    YOLO, torch.nn.Module, FaceRecognizer, Callable[..., Any], None
]
model: ModelType = None
transforms: Any = None


@worker_process_init.connect
def init_model(sender: Any, **kwargs: Any) -> None:
    global model
    global transforms
    hostname = sender.hostname
    if "yolo" in hostname:
        model = YOLO(settings.YOLO_PATH)

    elif "recognizer" in hostname:
        model = get_resnet_emotion_model(
            output_shape=len(EMOTION_LABELS), load_path=settings.RESNET_PATH
        )
        transforms = t.Compose(
            [
                t.ToImage(),
                t.ToDtype(torch.float32, scale=True),
                t.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    elif "emotions" in hostname:
        model = FaceRecognizer()

    return None


@celery_app.task
def yolo(image_bytes: bytes) -> Dict[str, Any]:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode image")

    if model is None or not isinstance(model, YOLO):
        raise RuntimeError("YOLO model not initialized")

    results = model(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    faces = []

    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            faces.append([x1, y1, x2, y2])

    return {"image": image_bytes, "faces": faces}


@celery_app.task
def recognizer(data: Dict[str, Any]) -> Dict[str, Any]:
    image_bytes = data["image"]
    faces = data["faces"]

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if model is None or not hasattr(model, "process_face"):
        raise RuntimeError("Recognizer model not initialized")

    if img is None:
        return data

    identities = []

    for x1, y1, x2, y2 in faces:
        face_roi = img[y1:y2, x1:x2]
        human_uuid, identity_conf, embedding = model.process_face(face_roi)

        identities.append(
            {
                "bbox": [x1, y1, x2, y2],
                "identity": human_uuid,
                "identity_confidence": identity_conf,
                "embedding": (
                    embedding.tolist() if embedding is not None else None
                ),
            }
        )

    data["identities"] = identities
    return data


@celery_app.task
def emotions(data: Dict[str, Any]) -> Dict[str, Any]:
    image_bytes = data["image"]
    faces = data["faces"]

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if model is None or transforms is None:
        raise RuntimeError("Emotion model/transforms not initialized")

    if img is None:
        return data

    emotions_out = []

    for x1, y1, x2, y2 in faces:
        face_roi = img[y1:y2, x1:x2]

        if face_roi.size == 0:
            continue

        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_gray = cv2.resize(face_gray, (70, 84))

        face_tensor = transforms(Image.fromarray(face_gray)).unsqueeze(0)

        logits = model(face_tensor)

        if not isinstance(logits, torch.Tensor):
            raise ValueError(
                f"Expected torch.Tensor from model, got {type(logits)}"
            )

        probs = torch.softmax(logits, dim=1)
        pred_idx: int = int(torch.argmax(probs, dim=1).item())

        emotions_out.append(
            {
                "bbox": [x1, y1, x2, y2],
                "emotion": EMOTION_LABELS[pred_idx],
                "confidence": float(probs[0][pred_idx]),
            }
        )

    data["emotions"] = emotions_out
    return data


@celery_app.task
def merge_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:

    recognizer_data = results[0]
    emotions_data = results[1]

    emotions_data["identities"] = recognizer_data.get("identities")

    return emotions_data


def get_etl_pipeline(image_bytes: bytes) -> chain:
    return chain(
        yolo.s(image_bytes),
        chord([recognizer.s(), emotions.s()], merge_results.s()),
    )
