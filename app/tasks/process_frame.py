import base64
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import redis as sync_redis

from app.celery_app import celery_app
from app.config import settings

logger = logging.getLogger(__name__)

# ── Engine singleton per worker process ───────────────────────────────────────
# EmotionEngine loads heavy models (YOLO + ResNet + DeepFace).
# We initialise it once when the first task runs in this worker process.

_engine: Optional[Any] = None


def get_engine() -> Any:
    global _engine
    if _engine is None:
        from ml.src.engine import EmotionEngine  # noqa: PLC0415

        logger.info("[Worker] Loading EmotionEngine...")
        _engine = EmotionEngine(
            yolo_path=settings.YOLO_PATH,
            resnet_path=settings.RESNET_PATH,
        )
        logger.info("[Worker] EmotionEngine ready")
    return _engine


def get_redis() -> sync_redis.Redis:
    return sync_redis.from_url(settings.redis_url, decode_responses=False)


# ── Tasks ─────────────────────────────────────────────────────────────────────


@celery_app.task(name="process_camera_frame", bind=True, max_retries=0)
def process_camera_frame(self, camera_id: int) -> None:
    """
    Called for every camera frame batch.

    1. Reads the *latest* frame from Redis (previous frames are already dropped).
    2. Runs the full ML pipeline: YOLO → FaceID → Emotion.
    3. Publishes the result to a Redis channel so the WS handler
       can forward it to the browser immediately.
    4. Clears the processing lock so the next frame can be dispatched.
    """
    r = get_redis()
    frame_key = f"frame:{camera_id}"
    processing_key = f"processing:{camera_id}"
    result_channel = f"result_channel:{camera_id}"
    result_key = f"result:{camera_id}"

    try:
        frame_data: Optional[bytes] = r.get(frame_key)
        if frame_data is None:
            logger.warning(f"[Task] No frame found for camera {camera_id}")
            return

        engine = get_engine()
        _, result = engine.process_image(frame_data)

        result["timestamp"] = datetime.now(timezone.utc).isoformat()
        result["camera_id"] = camera_id

        result_json = json.dumps(result)

        # Persist last result (for GET /api/cameras/{id}/emotions)
        r.set(result_key, result_json, ex=60)
        # Push to WebSocket subscriber
        r.publish(result_channel, result_json)

    except Exception as exc:
        logger.error(
            f"[Task] Error processing frame for camera {camera_id}: {exc}",
            exc_info=True,
        )
    finally:
        r.delete(processing_key)


@celery_app.task(name="process_single_image", bind=True, max_retries=0)
def process_single_image(self, image_b64: str) -> Dict[str, Any]:
    """
    One-shot image analysis for the /api/process test endpoint.
    Accepts base64-encoded image bytes, returns the full detection dict.
    """
    image_bytes = base64.b64decode(image_b64)
    engine = get_engine()
    _, result = engine.process_image(image_bytes)
    return result
