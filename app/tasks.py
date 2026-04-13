import logging
import os
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

from app.api.miniO.db import get_minio_client
from app.celery_app import celery_app
from app.config import settings
from app.schemas.frame import ETLReturnResult, MergedItem
from ml.src.recognizer import FaceRecognizer
from ml.src.resnet import EMOTION_LABELS, get_resnet_emotion_model

ModelType = Union[
    YOLO, torch.nn.Module, FaceRecognizer, Callable[..., Any], None
]
model: ModelType = None
transforms: Any = None

logger = logging.getLogger(__name__)


@worker_process_init.connect
def init_model(sender: Any, **kwargs: Any) -> None:
    global model, transforms
    hostname = os.environ.get("WORKER_HOSTNAME", "")
    logger.warning(
        f"[INIT] worker_process_init triggered. Hostname env: '{hostname}'"
    )

    if "yolo" in hostname:
        logger.warning("[INIT] Loading YOLO model...")
        model = YOLO(settings.YOLO_PATH)
        logger.warning("[INIT] YOLO model loaded")
    elif "recognizer" in hostname:
        logger.warning("[INIT] Loading Recognizer model...")
        model = FaceRecognizer()
        logger.warning("[INIT] Recognizer model loaded")
    elif "emotions" in hostname:
        logger.warning("[INIT] Loading Emotion model...")
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
        logger.warning("[INIT] Emotion model loaded")
    else:
        logger.error(
            f"[INIT] Hostname '{hostname}' did not match any worker type!"
        )
    return None


@celery_app.task
def yolo(store_path: str) -> Dict[str, Any]:
    # client = payload.minio_client
    # bucket_name = payload.bucket_name
    # store_path = payload.store_path

    client, bucket_name = get_minio_client()
    image_bytes: bytes = client.get_object(bucket_name, store_path).read()

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

    return {"path": store_path, "faces": faces}


@celery_app.task
def recognizer(data: Dict[str, Any]) -> Dict[str, Any]:
    faces = data["faces"]
    store_path = data["path"]

    # payload = data["payload"]
    # client = payload.minio_client
    # bucket_name = payload.bucket_name
    # store_path = payload.store_path

    client, bucket_name = get_minio_client()

    image_bytes: bytes = client.get_object(bucket_name, store_path).read()

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if not isinstance(model, FaceRecognizer):
        raise RuntimeError("Recognizer model not initialized")

    if img is None:
        return data

    identities = []

    for x1, y1, x2, y2 in faces:
        face_roi = img[y1:y2, x1:x2]
        human_uuid, identity_conf, embedding = model(face_roi)

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
    return {"type": "recognizer", "data": data}


@celery_app.task
def emotions(data: Dict[str, Any]) -> Dict[str, Any]:
    faces = data["faces"]
    store_path = data["path"]

    client, bucket_name = get_minio_client()

    image_bytes: bytes = client.get_object(bucket_name, store_path).read()

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if not isinstance(model, torch.nn.Module) or transforms is None:
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
    return {"type": "emotions", "data": data}


@celery_app.task
def merge_results(
    results: List[Dict[str, Any]], stream_data: Dict[str, Any]
) -> Dict[str, Any]:

    for r in results:
        if r["type"] == "recognizer":
            recognizer_data = r["data"]
        elif r["type"] == "emotions":
            emotions_data = r["data"]
    merged_results = []

    for identity in recognizer_data["identities"]:

        identity_bbox = identity["bbox"]
        found_emotion = None

        for emotion in emotions_data["emotions"]:
            emotion_bbox = emotion["bbox"]
            if identity_bbox == emotion_bbox:
                found_emotion = emotion
                break

        if found_emotion is None:
            raise ValueError(
                f"Can't find emotion for face with bbox {identity_bbox} in {__file__}"
            )

        merged_item = {
            "bbox": identity["bbox"],
            "identity": identity["identity"],
            "identity_confidence": identity["identity_confidence"],
            "embedding": identity.get("embedding"),
            "emotion": found_emotion["emotion"],
            "emotion_confidence": found_emotion["confidence"],
        }

        merged_results.append(merged_item)

    return ETLReturnResult(
        user_id=stream_data["user_id"],
        stream_id=stream_data["stream_id"],
        frame_id=stream_data["frame_id"],
        timestamp=stream_data["timestamp"],
        store_path=stream_data["store_path"],
        items=[MergedItem(**item) for item in merged_results],
    ).model_dump()


def get_etl_pipeline(payload: ETLReturnResult) -> chain:
    return chain(
        yolo.s(payload.store_path),
        chord(
            [recognizer.s(), emotions.s()],
            merge_results.s(payload.model_dump()),
        ),
    )
