import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from app.core.config import settings
from app.core.database import close_db_pool, init_db_pool
from app.core.db_queries import (
    create_tables,
    delete_edge_from_db,
    delete_frame_from_db,
    delete_stream_from_db,
    save_batch_to_db,
    save_frames_to_db,
    seed_admin_user,
)
from app.core.minio_client import delete_minio_task_id, get_minio_client
from public.schemas.auth import AuthRequest
from public.schemas.ingest import IngestBatchRequest


def _get_secret_api_key() -> str:
    return getattr(settings, "STATIC_API_KEY", "super_secret_api_key")


def _append_to_txt(payload: IngestBatchRequest, txt_path: str) -> None:
    os.makedirs(os.path.dirname(txt_path) or ".", exist_ok=True)
    line = json.dumps(
        {
            "received_at": (
                payload.received_at.isoformat()
                if payload.received_at
                else None
            ),
            "edge_id": payload.edge_id,
            "batch_id": payload.batch_id,
            "camera_ids": payload.camera_ids,
            "processed_count": payload.processed_count,
            "frames": [
                {
                    "camera_id": f.camera_id,
                    "frame_id": f.frame_id,
                    "timestamp": f.timestamp,
                    "items": f.items,
                }
                for f in payload.frames
            ],
        },
        ensure_ascii=False,
    )
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await init_db_pool()
    await create_tables()
    await seed_admin_user()
    yield
    await close_db_pool()


app = FastAPI(lifespan=lifespan)


async def verify_api_key(request: Request) -> None:
    secret_header = request.headers.get("X-Secret-Api-Key")
    expected = _get_secret_api_key()
    if not secret_header or secret_header != expected:
        raise HTTPException(status_code=401, detail="Invalid secret api key")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse("edge-global ingest service")


@app.post("/auth", response_class=JSONResponse)
async def auth(request: Request, data: AuthRequest) -> Dict[str, Any]:
    return {"status": 200, "user_id": "master"}


@app.post("/register", response_class=JSONResponse)
async def register(request: Request, data: AuthRequest) -> Dict[str, Any]:
    return {"status": 200, "user_id": "master"}


@app.post("/api/ingest_batch")
async def ingest_batch(
    request: Request, payload: IngestBatchRequest
) -> Dict[str, Any]:
    secret_header = request.headers.get("X-Secret-Api-Key")
    expected = _get_secret_api_key()
    if not secret_header or secret_header != expected:
        raise HTTPException(status_code=401, detail="Invalid secret api key")

    if payload.received_at is None:
        payload.received_at = datetime.now(timezone.utc)

    for f in payload.frames:
        print(
            f"[GLOBAL] edge={payload.edge_id} camera_id={f.camera_id} frame_id={f.frame_id}"
        )

    try:
        await save_batch_to_db(
            batch_id=payload.batch_id,
            edge_id=payload.edge_id,
            camera_ids=payload.camera_ids,
            processed_count=payload.processed_count,
            received_at=payload.received_at,
        )
        frames_for_db = []
        for f in payload.frames:
            frames_for_db.append(
                {
                    "edge_id": payload.edge_id,
                    "camera_id": f.camera_id,
                    "frame_id": f.frame_id,
                    "timestamp": f.timestamp,
                    "store_path": "",
                    "items": f.items,
                }
            )
        await save_frames_to_db(frames_for_db)
    except Exception as e:
        print(f"Failed to save to PostgreSQL: {e}")
    txt_path = getattr(
        settings, "GLOBAL_TXT_PATH", "global_ingest_batches.jsonl"
    )
    _append_to_txt(payload, txt_path)

    return {
        "status": 200,
        "batch_id": payload.batch_id,
        "processed_count": payload.processed_count,
    }
