import base64

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.tasks.process_frame import process_single_image

router = APIRouter()


@router.post("/process")
async def api_process(file: UploadFile = File(...)) -> JSONResponse:
    """
    Test endpoint: upload a single image, run the full ML pipeline,
    return JSON with detected faces, emotions and identity hashes.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    # Celery задача принимает base64 (JSON-сериализуемо)
    image_b64 = base64.b64encode(image_bytes).decode()

    try:
        task = process_single_image.delay(image_b64)
        result = task.get(timeout=60)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
