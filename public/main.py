from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from public.schemas.auth import AuthRequest
from public.schemas.ingest import IngestBatchRequest


def _get_secret_api_key() -> str:
    from app.core.config import settings

    return getattr(settings, "STATIC_API_KEY", "super_secret_api_key")


def _append_to_txt(payload: IngestBatchRequest, txt_path: str) -> None:
    import json
    import os

    os.makedirs(os.path.dirname(txt_path) or ".", exist_ok=True)

    # JSONL: one line per batch
    line = json.dumps(
        {
            "received_at": payload.received_at.isoformat() if payload.received_at else None,
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
    yield


app = FastAPI(lifespan=lifespan)


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
async def ingest_batch(request: Request, payload: IngestBatchRequest) -> Dict[str, Any]:
    secret_header = request.headers.get("X-Secret-Api-Key")
    expected = _get_secret_api_key()

    if not secret_header or secret_header != expected:
        raise HTTPException(status_code=401, detail="Invalid secret api key")

    # fill received_at server-side
    if payload.received_at is None:
        payload.received_at = datetime.utcnow()

    # print camera_id for each frame
    for f in payload.frames:
        print(f"[GLOBAL] edge={payload.edge_id} camera_id={f.camera_id} frame_id={f.frame_id}")

    from app.core.config import settings

    txt_path = getattr(settings, "GLOBAL_TXT_PATH", "global_ingest_batches.jsonl")
    _append_to_txt(payload, txt_path)

    return {"status": 200, "batch_id": payload.batch_id, "processed_count": payload.processed_count}

