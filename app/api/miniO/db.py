from io import BytesIO
from typing import Any, Tuple

from minio import Minio
from minio.deleteobjects import DeleteObject

from app.config import settings

server_url = settings.MINIO_URL
access_key = settings.MINIO_ACCESS_KEY
secret_key = settings.MINIO_SECRET_KEY
secure = settings.MINIO_SECURE


def get_minio_client() -> Tuple[Any, str]:
    """
    returns (client, bucket)
    """

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

        data_stream = BytesIO(data)

        client.put_object(
            bucket,
            path,
            data_stream,
            length=len(data),
            content_type="image/jpeg",
        )

    except Exception as error:
        print(f"Error occurred when storing_data object {error}")


def delete_minio_task_id(client: Minio, bucket: str, path: str = "/") -> None:
    objects_to_delete = client.list_objects(
        bucket, prefix=path, recursive=True
    )
    delete_list = [DeleteObject(obj.object_name) for obj in objects_to_delete]

    errors = client.remove_objects(bucket, delete_list)

    for error in errors:
        print(f"Error occurred when deleting object: {error}")
