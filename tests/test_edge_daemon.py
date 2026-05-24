import queue
import threading
from unittest.mock import MagicMock, patch

from app.edge_daemon import CameraSlot, EdgeDaemon, parse_camera_sources


def test_parse_camera_sources() -> None:
    assert parse_camera_sources("0,1,2") == ["0", "1", "2"]
    assert parse_camera_sources("rtsp://cam1;rtsp://cam2") == [
        "rtsp://cam1",
        "rtsp://cam2",
    ]
    assert parse_camera_sources("0;1,2") == ["0", "1", "2"]
    assert parse_camera_sources("") == []
    assert parse_camera_sources("   ") == []


def test_camera_slot() -> None:
    slot = CameraSlot(camera_id="test", source="0")
    assert slot.camera_id == "test"
    assert slot.source == "0"
    assert slot.latest_frame_bytes is None
    assert isinstance(slot.latest_lock, type(threading.Lock()))
    assert slot.last_update_ts == 0.0


@patch("app.edge_daemon.get_minio_client")
@patch("app.edge_daemon.store_data_in_minio")
@patch("app.edge_daemon.get_etl_pipeline")
def test_try_submit_for_camera(
    mock_get_pipeline: MagicMock,
    mock_store: MagicMock,
    mock_get_client: MagicMock,
    test_image_bytes: bytes,
) -> None:
    mock_client = MagicMock()
    mock_get_client.return_value = (mock_client, "photos")
    mock_pipeline = MagicMock()
    mock_async_result = MagicMock()
    mock_async_result.id = "task_123"
    mock_pipeline.apply_async.return_value = mock_async_result
    mock_get_pipeline.return_value = mock_pipeline

    daemon = EdgeDaemon()
    daemon._in_flight_sem = threading.Semaphore(2)
    daemon._submitted = queue.Queue()
    cam = CameraSlot(camera_id="0", source="0")
    cam.latest_frame_bytes = test_image_bytes
    cam.last_update_ts = 1000.0

    task_id = daemon._try_submit_for_camera(cam)
    assert task_id == "task_123"
    assert not daemon._submitted.empty()
    submitted_id, meta = daemon._submitted.get()
    assert submitted_id == "task_123"
    assert meta["camera_id"] == "0"
    mock_store.assert_called_once()
    mock_get_pipeline.assert_called_once()


def test_try_submit_for_camera_no_frame() -> None:
    daemon = EdgeDaemon()
    daemon._in_flight_sem = threading.Semaphore(1)
    cam = CameraSlot(camera_id="0", source="0")
    cam.latest_frame_bytes = None
    result = daemon._try_submit_for_camera(cam)
    assert result is None


@patch("app.edge_daemon.requests.Session.post")
def test_post_batch(mock_post: MagicMock) -> None:
    import json

    daemon = EdgeDaemon()
    daemon.secret_api_key = "test_key"
    batch_id = "batch1"
    frames = [
        {
            "edge_id": "e1",
            "camera_id": "0",
            "frame_id": "f1",
            "timestamp": 123,
            "items": [],
        },
        {
            "edge_id": "e1",
            "camera_id": "1",
            "frame_id": "f2",
            "timestamp": 124,
            "items": [],
        },
    ]
    daemon._post_batch(batch_id, frames)
    mock_post.assert_called_once()
    args = mock_post.call_args[0]
    kwargs = mock_post.call_args[1]
    assert args[0] == daemon.global_ingest_url
    assert kwargs["headers"]["X-Secret-Api-Key"] == "test_key"
    # В production передаётся json-строка, а не словарь
    payload_str = kwargs["json"]
    assert isinstance(payload_str, str)
    payload = json.loads(payload_str)
    assert payload["batch_id"] == batch_id
    assert payload["processed_count"] == 2
    assert payload["camera_ids"] == ["0", "1"]
    assert len(payload["frames"]) == 2
