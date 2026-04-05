import asyncio
import logging

import redis.asyncio as aioredis
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import settings
from app.tasks.process_frame import process_camera_frame

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/{camera_id}")
async def websocket_endpoint(websocket: WebSocket, camera_id: int) -> None:
    """
    One WebSocket connection = one camera.

    Browser → sends raw JPEG frames as binary messages.
    Server  → replies with JSON results from the ML pipeline.

    Frame-drop logic:
      Each new frame overwrites the previous one in Redis.
      A Celery task is dispatched only when no task is currently
      processing for this camera (SET NX). The task always reads
      the *latest* frame, so stale frames are naturally skipped.
    """
    await websocket.accept()
    logger.info(f"[WS] Camera {camera_id} connected")

    redis_client = aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=False,
    )
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(f"result_channel:{camera_id}")

    frame_key = f"frame:{camera_id}"
    processing_key = f"processing:{camera_id}"
    result_channel = f"result_channel:{camera_id}"

    async def receive_frames() -> None:
        """Read frames from the browser and store the latest one in Redis."""
        async for data in websocket.iter_bytes():
            # Overwrite previous frame — only the freshest matters
            await redis_client.set(frame_key, data)

            # Dispatch a Celery task only if none is running (atomic SET NX)
            acquired = await redis_client.set(
                processing_key, "1", ex=10, nx=True
            )
            if acquired:
                process_camera_frame.delay(camera_id)

    async def send_results() -> None:
        """Forward ML results from Redis pub/sub back to the browser."""
        async for message in pubsub.listen():
            if message["type"] == "message":
                payload = message["data"]
                if isinstance(payload, bytes):
                    payload = payload.decode()
                await websocket.send_text(payload)

    recv_task = asyncio.create_task(receive_frames())
    send_task = asyncio.create_task(send_results())

    try:
        done, pending = await asyncio.wait(
            [recv_task, send_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        # Propagate exceptions if any
        for task in done:
            exc = task.exception()
            if exc and not isinstance(exc, WebSocketDisconnect):
                logger.error(f"[WS] Camera {camera_id} task error: {exc}")
    finally:
        logger.info(f"[WS] Camera {camera_id} disconnected")
        await pubsub.unsubscribe(result_channel)
        await redis_client.aclose()
