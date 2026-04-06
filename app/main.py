# import asyncio
# import base64
# import json
# import uuid
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

# import cv2
from fastapi import (  # File,; HTTPException,; UploadFile,
    FastAPI,
    Request,
    WebSocket,
)
from fastapi.responses import HTMLResponse  # , JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.db.db import get_connection
from app.config import settings
from ml.src.engine import EmotionEngine

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


@app.websocket("/api/stream")
async def stream_one_frame(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        pass
    except Exception as e:
        logging.info(e, exc_info=True)


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


# @app.post("/web/process", response_class=HTMLResponse)
# async def web_process(
#     request: Request, file: UploadFile = File(...)
# ) -> HTMLResponse:
#     image_bytes = await file.read()
#     engine: EmotionEngine = request.app.state.engine

#     loop = asyncio.get_event_loop()
#     processed_image, api_data = await loop.run_in_executor(
#         None, engine.process_image, image_bytes
#     )

#     filename = file.filename or "web_upload"
#     save_detection_results(api_data, filename)

#     success, buffer = cv2.imencode(".png", processed_image)
#     if not success:
#         raise ValueError("Could not encode image to PNG")

#     img_base64 = base64.b64encode(buffer).decode()

#     return templates.TemplateResponse(
#         request,
#         "result.html",
#         {
#             "image_base64": img_base64,
#             "faces_data": api_data.get("faces", []),
#         },
#     )
