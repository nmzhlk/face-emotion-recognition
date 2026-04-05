import json
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

router = APIRouter()

CAMERAS_KEY = "cameras"


# ── helpers ──────────────────────────────────────────────────────────────────


async def _get_cameras(redis) -> List[Dict[str, Any]]:
    data = await redis.get(CAMERAS_KEY)
    return json.loads(data) if data else []


async def _save_cameras(redis, cameras: List[Dict[str, Any]]) -> None:
    await redis.set(CAMERAS_KEY, json.dumps(cameras))


async def _find_camera(redis, camera_id: int) -> Dict[str, Any]:
    cameras = await _get_cameras(redis)
    for c in cameras:
        if c["id"] == camera_id:
            return c
    raise HTTPException(status_code=404, detail="Camera not found")


# ── schemas ───────────────────────────────────────────────────────────────────


class CameraCreate(BaseModel):
    name: str
    url: str


# ── routes ────────────────────────────────────────────────────────────────────


@router.get("")
async def list_cameras(request: Request) -> List[Dict[str, Any]]:
    return await _get_cameras(request.app.state.redis)


@router.post("", status_code=201)
async def add_camera(request: Request, body: CameraCreate) -> Dict[str, Any]:
    redis = request.app.state.redis
    cameras = await _get_cameras(redis)
    new_id = max((c["id"] for c in cameras), default=0) + 1
    camera = {
        "id": new_id,
        "name": body.name,
        "url": body.url,
        "status": "pending",
    }
    cameras.append(camera)
    await _save_cameras(redis, cameras)
    return camera


@router.delete("/{camera_id}", status_code=204)
async def delete_camera(request: Request, camera_id: int) -> None:
    redis = request.app.state.redis
    cameras = await _get_cameras(redis)
    filtered = [c for c in cameras if c["id"] != camera_id]
    if len(filtered) == len(cameras):
        raise HTTPException(status_code=404, detail="Camera not found")
    await _save_cameras(redis, filtered)
    # Cleanup Redis keys for this camera
    await redis.delete(
        f"frame:{camera_id}",
        f"result:{camera_id}",
        f"processing:{camera_id}",
        f"camera:{camera_id}:active",
    )


@router.post("/{camera_id}/start")
async def start_camera(request: Request, camera_id: int) -> Dict[str, str]:
    redis = request.app.state.redis
    cameras = await _get_cameras(redis)
    found = False
    for c in cameras:
        if c["id"] == camera_id:
            c["status"] = "online"
            found = True
            break
    if not found:
        raise HTTPException(status_code=404, detail="Camera not found")
    await _save_cameras(redis, cameras)
    await redis.set(f"camera:{camera_id}:active", "1")
    return {"status": "started"}


@router.post("/{camera_id}/stop")
async def stop_camera(request: Request, camera_id: int) -> Dict[str, str]:
    redis = request.app.state.redis
    cameras = await _get_cameras(redis)
    found = False
    for c in cameras:
        if c["id"] == camera_id:
            c["status"] = "offline"
            found = True
            break
    if not found:
        raise HTTPException(status_code=404, detail="Camera not found")
    await _save_cameras(redis, cameras)
    await redis.delete(f"camera:{camera_id}:active")
    return {"status": "stopped"}


@router.get("/{camera_id}/emotions")
async def get_emotions(request: Request, camera_id: int) -> Dict[str, Any]:
    redis = request.app.state.redis
    data = await redis.get(f"result:{camera_id}")
    if not data:
        return {"timestamp": None, "faces": []}
    return json.loads(data)


# ── stream endpoint (/api/stream/{id}) ───────────────────────────────────────
# Mounted separately in main.py via a stream router, but placed here for clarity.
# Returns the latest raw JPEG frame stored in Redis for this camera.

stream_router = APIRouter()


@stream_router.get("/stream/{camera_id}")
async def get_stream_frame(request: Request, camera_id: int) -> Response:
    """Returns the latest JPEG frame from a camera (single snapshot)."""
    data = await request.app.state.redis.get(f"frame:{camera_id}")
    if not data:
        raise HTTPException(status_code=404, detail="No frame available")
    return Response(content=data, media_type="image/jpeg")
