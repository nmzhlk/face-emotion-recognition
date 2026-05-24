import logging
from io import BytesIO
from typing import Any, Tuple

from minio import Minio
from minio.deleteobjects import DeleteObject

from app.core.config import settings
from app.core.logging_config import setup_logging

setup_logging(service_name="minio-client")
logger = logging.getLogger(__name__)

server_url = settings.MINIO_URL
access_key = settings.MINIO_ACCESS_KEY
secret_key = settings.MINIO_SECRET_KEY
secure = settings.MINIO_SECURE


def get_minio_client() -> Tuple[Any, str]:
    client = Minio(
        server_url, access_key=access_key, secret_key=secret_key, secure=secure
    )
    return (client, "photos")


def store_data_in_minio(
    client: Minio, bucket: str, path: str, data: bytes
) -> None:
    try:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            logger.info(f"Created bucket: {bucket}")

        data_stream = BytesIO(data)

        client.put_object(
            bucket,
            path,
            data_stream,
            length=len(data),
            content_type="image/jpeg",
        )
        logger.debug(f"Stored object in MinIO: {bucket}/{path}")

    except Exception as error:
        logger.error(f"Error occurred when storing object {path}: {error}")


def delete_minio_task_id(client: Minio, bucket: str, path: str = "/") -> None:
    objects_to_delete = client.list_objects(
        bucket, prefix=path, recursive=True
    )
    delete_list = [DeleteObject(obj.object_name) for obj in objects_to_delete]
    if not delete_list:
        logger.info(f"No objects to delete in {bucket}/{path}")
        return

    errors = client.remove_objects(bucket, delete_list)
    count = len(delete_list)
    for error in errors:
        logger.error(f"Error occurred when deleting object: {error}")
    logger.info(f"Deleted {count} objects from {bucket}/{path}")
