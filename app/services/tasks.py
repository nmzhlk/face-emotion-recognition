import logging
import os
from typing import Any, Callable, Dict, List, Union

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as t
from celery import chain, chord
from celery.signals import worker_process_init
from cv2 import Mat
from numpy import ndarray
from PIL import Image
from ultralytics import YOLO

from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.logging_config import setup_logging
from app.core.minio_client import get_minio_client
from app.schemas.frame import ETLReturnResult, MergedItem
from ml.src.recognizer import FaceRecognizer
from ml.src.resnet import EMOTION_LABELS, get_resnet_emotion_model

setup_logging(service_name=os.environ.get("WORKER_HOSTNAME", "celery-worker"))

ModelType = Union[
    YOLO, torch.nn.Module, FaceRecognizer, Callable[..., Any], None
]
model: ModelType = None
transforms: Any = None

logger = logging.getLogger(__name__)


def load_image_from_minio(store_path: str) -> Mat | ndarray | None:
    client, bucket = get_minio_client()
    return cv2.imdecode(
        np.frombuffer(client.get_object(bucket, store_path).read(), np.uint8),
        cv2.IMREAD_COLOR,
    )


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
    img = load_image_from_minio(store_path)

    if img is None:
        raise ValueError(f"Failed to load image from {store_path}")

    if not isinstance(model, YOLO):
        raise RuntimeError("YOLO model not initialized")

    results = model.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    faces = []

    if results:
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for i in range(len(result.boxes)):
                    box = result.boxes[i]
                    coords = box.xyxy[0].cpu().numpy().tolist()
                    x1, y1, x2, y2 = map(int, coords)
                    faces.append([x1, y1, x2, y2])

    return {"path": store_path, "faces": faces}


@celery_app.task
def recognizer(data: Dict[str, Any]) -> Dict[str, Any]:
    faces = data["faces"]
    store_path = data["path"]

    img = load_image_from_minio(store_path)

    if img is None:
        raise ValueError(f"Failed to load image from {store_path}")

    if not isinstance(model, FaceRecognizer):
        raise RuntimeError("Recognizer model not initialized")

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


@torch.inference_mode()
@celery_app.task
def emotions(data: Dict[str, Any]) -> Dict[str, Any]:
    faces = data["faces"]
    store_path = data["path"]

    img = load_image_from_minio(store_path)

    if img is None:
        raise ValueError(f"Failed to load image from {store_path}")

    if not isinstance(model, torch.nn.Module) or transforms is None:
        raise RuntimeError("Emotion model/transforms not initialized")

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
    recognizer_data: Dict[str, Any] = {}
    emotions_data: Dict[str, Any] = {}

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

    for identity, emotion in zip(
        recognizer_data["identities"], emotions_data["emotions"]
    ):
        merged_item = {
            "bbox": identity["bbox"],
            "identity": identity["identity"],
            "identity_confidence": identity["identity_confidence"],
            "embedding": identity.get("embedding"),
            "emotion": emotion["emotion"],
            "emotion_confidence": emotion["confidence"],
        }
        merged_results.append(merged_item)

    payload = ETLReturnResult(
        user_id=stream_data["user_id"],
        stream_id=stream_data.get("stream_id", stream_data.get("camera_id")),
        frame_id=stream_data["frame_id"],
        timestamp=stream_data["timestamp"],
        store_path=stream_data["store_path"],
        items=[MergedItem(**item) for item in merged_results],
    ).model_dump()

    try:
        client, bucket_name = get_minio_client()
        delete_path = payload.get("store_path")
        if delete_path:
            from app.core.minio_client import delete_minio_task_id

            delete_minio_task_id(client, bucket_name, delete_path)
    except Exception as e:
        logger.warning(f"[CLEANUP] Failed to delete raw frame from MinIO: {e}")

    return payload


def get_etl_pipeline(payload: ETLReturnResult) -> chain:
    yolo_task = celery_app.tasks["app.services.tasks.yolo"]
    recognizer_task = celery_app.tasks["app.services.tasks.recognizer"]
    emotions_task = celery_app.tasks["app.services.tasks.emotions"]
    merge_task = celery_app.tasks["app.services.tasks.merge_results"]

    return chain(
        yolo_task.s(payload.store_path),
        chord(
            [recognizer_task.s(), emotions_task.s()],
            merge_task.s(payload.model_dump()),
        ),
    )
