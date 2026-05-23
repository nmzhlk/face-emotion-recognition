from io import BytesIO
from unittest.mock import MagicMock

import pytest
from minio import Minio

from app.core.minio_client import (
    delete_minio_task_id,
    get_minio_client,
    store_data_in_minio,
)


def test_get_minio_client() -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MINIO_URL", "test:9000")
        mp.setenv("MINIO_ACCESS_KEY", "testkey")
        mp.setenv("MINIO_SECRET_KEY", "testsecret")
        from app.core.config import settings

        mp.setattr(settings, "MINIO_URL", "test:9000")
        mp.setattr(settings, "MINIO_ACCESS_KEY", "testkey")
        mp.setattr(settings, "MINIO_SECRET_KEY", "testsecret")
        mp.setattr(settings, "MINIO_SECURE", False)
        client, bucket = get_minio_client()
        assert isinstance(client, Minio)
        assert bucket == "photos"


def test_store_data_in_minio_bucket_exists(minio_mock: tuple) -> None:
    mock_client, _ = minio_mock
    mock_client.bucket_exists.return_value = True
    data = b"test image data"
    store_data_in_minio(mock_client, "photos", "path/test.jpg", data)
    mock_client.bucket_exists.assert_called_once_with("photos")
    mock_client.make_bucket.assert_not_called()
    mock_client.put_object.assert_called_once()
    args, kwargs = mock_client.put_object.call_args
    assert args[0] == "photos"
    assert args[1] == "path/test.jpg"
    assert isinstance(args[2], BytesIO)
    assert kwargs["length"] == len(data)


def test_store_data_in_minio_bucket_not_exists(minio_mock: tuple) -> None:
    mock_client, _ = minio_mock
    mock_client.bucket_exists.return_value = False
    data = b"test data"
    store_data_in_minio(mock_client, "photos", "path/test.jpg", data)
    mock_client.make_bucket.assert_called_once_with("photos")
    mock_client.put_object.assert_called_once()


def test_delete_minio_task_id(minio_mock: tuple) -> None:
    mock_client, _ = minio_mock
    mock_obj1 = MagicMock()
    mock_obj1.object_name = "path/frame1.jpg"
    mock_obj2 = MagicMock()
    mock_obj2.object_name = "path/frame2.jpg"
    mock_client.list_objects.return_value = [mock_obj1, mock_obj2]
    delete_minio_task_id(mock_client, "photos", "path/")
    mock_client.list_objects.assert_called_once_with(
        "photos", prefix="path/", recursive=True
    )
    mock_client.remove_objects.assert_called_once()
    delete_list_arg = mock_client.remove_objects.call_args[0][1]
    assert len(list(delete_list_arg)) == 2
