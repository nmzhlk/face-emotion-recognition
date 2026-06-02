import base64
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from app.core.config import settings
from app.core.database import close_db_pool, init_db_pool
from app.core.db_queries import (
    create_tables,
    save_batch_to_db,
    save_frames_to_db,
    seed_admin_user,
)
from app.core.logging_config import setup_logging
from app.core.minio_client import (
    delete_minio_task_id,
    get_minio_client,
    store_data_in_minio,
)
from app.schemas.frame import ETLReturnResult
from app.services.tasks import get_etl_pipeline
from public.schemas.auth import AuthRequest
from public.schemas.ingest import IngestBatchRequest

setup_logging(service_name="global-api")

logger = logging.getLogger(__name__)


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
app.mount("/static", StaticFiles(directory="public/ui/static"), name="static")
templates = Jinja2Templates(directory="public/ui")


def verify_api_key(request: Request) -> None:
    secret_header = request.headers.get("X-Secret-Api-Key")
    expected = _get_secret_api_key()
    if not secret_header or secret_header != expected:
        raise HTTPException(status_code=401, detail="Invalid secret api key")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@app.post("/login", response_class=JSONResponse)
async def auth(request: Request, data: AuthRequest) -> Dict[str, Any]:
    return {"status": 200, "user_id": "master"}


@app.post("/register", response_class=JSONResponse)
# TODO: create registration page
async def register(request: Request, data: AuthRequest) -> Dict[str, Any]:
    return {"status": 200, "user_id": "master"}


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "login.html")


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "register.html")


@app.get("/logout", response_class=HTMLResponse)
async def logout_page(request: Request) -> RedirectResponse:
    return RedirectResponse(url="/", status_code=303)


@app.get("/cameras", response_class=HTMLResponse)
async def cameras_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "cameras-list.html", {"request": request, "user": "Гость"}
    )


@app.get("/camera/{camera_id}", response_class=HTMLResponse)
async def camera_stream(request: Request, camera_id: str) -> RedirectResponse:
    # TODO: create webpages for cameras
    return RedirectResponse(url="/cameras", status_code=303)


@app.post("/api/ingest_batch")
async def ingest_batch(
    request: Request, payload: IngestBatchRequest
) -> Dict[str, Any]:
    secret_header = request.headers.get("X-Secret-Api-Key")
    expected = _get_secret_api_key()

    if not secret_header or secret_header != expected:
        raise HTTPException(status_code=401, detail="Invalid secret api key")

    # fill received_at server-side
    if payload.received_at is None:
        payload.received_at = datetime.now(timezone.utc)

    # log camera_id forth each frame
    for f in payload.frames:
        logger.info(
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


@app.delete(
    "/api/frames/{edge_id}/{camera_id}/{frame_id}",
    dependencies=[Depends(verify_api_key)],
)
async def delete_frame(
    edge_id: str, camera_id: str, frame_id: str
) -> Dict[str, Any]:
    client, bucket = get_minio_client()
    object_path = f"/{edge_id}/{camera_id}/{frame_id}.jpg"
    try:
        client.stat_object(bucket, object_path)
    except Exception:
        raise HTTPException(status_code=404, detail="Frame not found")
    delete_minio_task_id(client, bucket, object_path)
    logger.info(f"Deleted frame: {object_path}")
    return {"status": "deleted", "path": object_path}


@app.delete(
    "/api/streams/{edge_id}/{camera_id}",
    dependencies=[Depends(verify_api_key)],
)
async def delete_stream(edge_id: str, camera_id: str) -> Dict[str, Any]:
    client, bucket = get_minio_client()
    prefix = f"/{edge_id}/{camera_id}/"
    objects = list(client.list_objects(bucket, prefix=prefix, recursive=True))
    if not objects:
        raise HTTPException(
            status_code=404, detail="No frames found for this stream"
        )
    delete_minio_task_id(client, bucket, prefix)
    logger.info(f"Deleted stream: {prefix}")
    return {"status": "deleted", "prefix": prefix, "count": len(objects)}


@app.delete("/api/edges/{edge_id}", dependencies=[Depends(verify_api_key)])
async def delete_edge(edge_id: str) -> Dict[str, Any]:
    client, bucket = get_minio_client()
    prefix = f"/{edge_id}/"
    objects = list(client.list_objects(bucket, prefix=prefix, recursive=True))
    if not objects:
        raise HTTPException(
            status_code=404, detail="No data found for this edge"
        )
    delete_minio_task_id(client, bucket, prefix)
    logger.info(f"Deleted edge data: {prefix}")
    return {"status": "deleted", "prefix": prefix, "count": len(objects)}


@app.get("/api/logs", dependencies=[Depends(verify_api_key)])
async def search_logs(
    edge_id: Optional[str] = Query(None),
    camera_id: Optional[str] = Query(None),
    frame_id: Optional[str] = Query(None),
    batch_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
) -> List[Dict[str, Any]]:
    from app.core.config import settings

    log_file = Path(settings.GLOBAL_TXT_PATH)
    if not log_file.exists():
        return []
    results: list[Dict[str, Any]] = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if len(results) >= limit:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if edge_id and record.get("edge_id") != edge_id:
                continue
            if batch_id and record.get("batch_id") != batch_id:
                continue
            if camera_id:
                found = any(
                    f.get("camera_id") == camera_id
                    for f in record.get("frames", [])
                )
                if not found:
                    continue
            if frame_id:
                found = any(
                    f.get("frame_id") == frame_id
                    for f in record.get("frames", [])
                )
                if not found:
                    continue
            results.append(record)
    return results


@app.post("/web/process", response_class=HTMLResponse)
async def process_image(
    request: Request, file: UploadFile = File(...)
) -> HTMLResponse:
    contents = await file.read()

    edge_id = "web_user"
    camera_id = "upload"
    frame_id = str(uuid.uuid4())
    timestamp = int(time.time() * 1000)
    store_path = f"/{edge_id}/{camera_id}/{frame_id}.jpg"

    client, bucket = get_minio_client()
    store_data_in_minio(client, bucket, store_path, contents)

    payload = ETLReturnResult(
        user_id=edge_id,
        stream_id=camera_id,
        frame_id=frame_id,
        timestamp=timestamp,
        store_path=store_path,
        items=None,
    )

    pipeline = get_etl_pipeline(payload)
    async_result = pipeline.apply_async()
    result = async_result.get(timeout=60)

    faces_data = result.get("items", [])

    buffered = BytesIO()
    img = Image.open(BytesIO(contents))
    img.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode()

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "image_base64": image_base64,
            "faces_data": faces_data,
        },
    )
