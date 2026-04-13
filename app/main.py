from typing import Any, AsyncIterator, Dict, List, Union

# import asyncio
# import base64
# import json
# import uuid
import logging
from contextlib import asynccontextmanager

# import cv2
from fastapi import (  # File,; HTTPException,; UploadFile,
    FastAPI,
    Request,
    HTTPException,
    status,
    UploadFile,
    File,
    Form
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from celery.result import AsyncResult

from app.api.db.db import get_connection
from app.config import settings
from ml.src.engine import EmotionEngine
from app.celery_app import celery_app
from app.schemas.auth import AuthRequest
from app.schemas.frame import (
    StreamPayload,
    ProcessedFrameData,
    StreamResponse,
    ETLReturnResult,
)

from app.tasks import get_etl_pipeline
from app.api.miniO.db import get_minio_client, store_data_in_minio, delete_minio_task_id
# from venv import logger


async def sync_embeddings(engine: EmotionEngine) -> None:
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM DUAL")
        cur.close()
        conn.close()
        print("[DB] Connection successful")
    except Exception as e:
        print(f"[DB] Warning: Could not sync embeddings: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.engine = EmotionEngine(
        yolo_path=settings.YOLO_PATH, resnet_path=settings.RESNET_PATH
    )

    await sync_embeddings(app.state.engine)
    yield


app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")
templates = Jinja2Templates(directory=settings.TEMPLATES_DIR)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")

# TODO Add JWT
@app.post('/auth', response_class=JSONResponse)
async def auth(request: Request, data: AuthRequest) -> Dict[str, Any]:
    return {
        'status': 200,
        'user_id': 'master'
    }


@app.post('/register', response_class=JSONResponse)
async def auth(request: Request, data: AuthRequest) -> Dict[str, Any]:
    return {
        'status': 200,
        'user_id': 'master'
    }


@app.post("/api/stream", response_model=StreamResponse)
async def stream_frame(request: Request,
    user_id: str = Form(...),
    stream_id: str = Form(...),
    frame_id: str = Form(...),
    timestamp: int = Form(...),
    image: UploadFile = File(...),
) -> StreamResponse:
    try:
        if not image:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image data is required"
            )
        
        store_path = f"/{user_id}/{stream_id}/{frame_id}"
        payload = ETLReturnResult(
            user_id = user_id,
            stream_id = stream_id,
            frame_id = frame_id,
            timestamp = timestamp,
            store_path = store_path,
        )
        (client, bucket_name) = get_minio_client()
        image_bytes = await image.read()
        store_data_in_minio(client, bucket_name, store_path, image_bytes)

        # payload = ETLPayload(
        #     minio_client = minio_client,
        #     bucket_name = bucket_name,
        #     store_path = store_path
        # )

        celery_chain = get_etl_pipeline(payload)
        result = celery_chain.apply_async()
        
        return StreamResponse(
            status=200,
            task_id=result.id,
            message="Task submitted successfully"
        )
        
    except Exception as e:
        # logger.error(f"Error in stream_frame: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error - {str(e)}"
        )
    

@app.get("/api/result/{task_id}", response_model=ProcessedFrameData)
async def get_result(task_id: str):

    result = AsyncResult(task_id, app=celery_app)

    if result.failed():
        raise HTTPException(status_code=500, detail=f"Task failed: {result.result}")

    if result.ready():
        return ETLReturnResult(**result.get())
    else:
        raise HTTPException(status_code=202, detail="Task still processing")


# def save_detection_results(api_data: dict, filename: str) -> None:
#     conn = get_connection()
#     cur = conn.cursor()
#     try:
#         admin_uuid = "admin-uuid-001"
#         img_uuid = str(uuid.uuid4())

#         cur.execute(
#             """INSERT INTO UPLOADED_IMAGES (UUID, USER_ID, IMAGE_URL, ORIGINAL_FILENAME)
#                 VALUES (:1, :2, :3, :4)""",
#             (img_uuid, admin_uuid, f"tmp/{filename}", filename),
#         )

#         for face in api_data.get("faces", []):
#             cur.execute(
#                 """INSERT INTO FACE_DETECTIONS (UUID, SOURCE_PHOTO_ID, DETECTED_BBOX, CONFIDENCE, EMOTION_CODE)
#                    VALUES (:1, :2, :3, :4, :5)""",
#                 (
#                     str(uuid.uuid4()),
#                     img_uuid,
#                     json.dumps(face["bbox"]),
#                     face["identity_confidence"],
#                     face["emotion"],
#                 ),
#             )

#         conn.commit()
#     except Exception as e:
#         print(f"Error: {e}")
#         conn.rollback()
#     finally:
#         cur.close()
#         conn.close()

