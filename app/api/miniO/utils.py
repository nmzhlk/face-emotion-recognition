import cv2
import numpy as np
from typing import Tuple

from app.api.minio.db import get_minio_client


def load_image_from_minio(store_path: str) -> np.ndarray:
    client, bucket_name = get_minio_client()

    try:
        image_bytes: bytes = client.get_object(bucket_name, store_path).read()
    except Exception as e:
        raise RuntimeError(f"Failed to read image from MinIO: {e}")

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Could not decode image from path: {store_path}")

    return img


def load_image_with_metadata(
    store_path: str,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    img = load_image_from_minio(store_path)
    return img, (img.shape[0], img.shape[1])
