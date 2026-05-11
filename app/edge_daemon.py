from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.minio_client import get_minio_client, store_data_in_minio
from app.schemas.frame import ETLReturnResult
from app.services.tasks import get_etl_pipeline


@dataclass
class CameraSlot:
    camera_id: str
    source: str
    latest_frame_bytes: Optional[bytes] = None
    latest_lock: threading.Lock = threading.Lock()
    last_update_ts: float = 0.0


def parse_camera_sources(value: str) -> List[str]:
    # supports: "0,1" or "rtsp://..;rtsp://.." or mix
    if not value:
        return []
    normalized = value.replace(";", ",")
    return [s.strip() for s in normalized.split(",") if s.strip()]


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


class EdgeDaemon:
    def __init__(self) -> None:
        self.edge_id = os.getenv("EDGE_ID", "edge-1")
        self.batch_size = env_int("BATCH_SIZE", 100)
        self.max_in_flight = env_int("MAX_IN_FLIGHT", 4)
        self.global_ingest_url = os.getenv(
            "GLOBAL_INGEST_URL", "http://global-api:8000/api/ingest_batch"
        )
        self.secret_api_key = os.getenv(
            "SECRET_API_KEY", "super_secret_api_key"
        )
        raw_sources = os.getenv(
            "CAMERA_SOURCES", "rtsp://host.docker.internal:554/live"
        )
        # raw_sources = "rtsp://192.168.31.98:8554/live"
        print(
            f"DEBUG: raw_sources is '{raw_sources}' type: {type(raw_sources)}"
        )  # ---------------------------------- TODO test
        self.camera_sources = parse_camera_sources(raw_sources)
        print(f"DEBUG: parsed sources: {self.camera_sources}")

        if not self.camera_sources:
            raise RuntimeError("CAMERA_SOURCES is empty")

        self.cameras: List[CameraSlot] = []
        for idx, src in enumerate(self.camera_sources):
            cam_id = str(idx)
            self.cameras.append(CameraSlot(camera_id=cam_id, source=src))

        self._in_flight_sem = threading.Semaphore(self.max_in_flight)
        self._submitted: "queue.Queue[Tuple[str, Dict[str, Any]]]" = (
            queue.Queue()
        )
        self._stop = threading.Event()

        # edge-local MinIO (configured via app/core/config)
        self.minio_bucket = "photos"
        self.minio_path_prefix = (
            f"/{self.edge_id}"  # raw frames under this prefix
        )

        self.session = requests.Session()

    def start(self) -> None:
        # Camera capture threads
        for cam in self.cameras:
            t = threading.Thread(
                target=self._camera_capture_loop,
                args=(cam,),
                daemon=True,
            )
            t.start()

        # Scheduler + result collection loop
        self._scheduler_loop()

    def stop(self) -> None:
        self._stop.set()

    def _camera_capture_loop(self, cam: CameraSlot) -> None:
        print("_camera_capture_loop")
        # Lazy import cv2 to keep edge daemon startup lighter
        import cv2

        src = cam.source
        # create capture
        # if it's an integer index string, use int
        try:
            src_eval: Any = int(src)
        except Exception:
            src_eval = src

        cap = cv2.VideoCapture(src_eval, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {src}")

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            # Encode JPEG
            ok2, buf = cv2.imencode(".jpg", frame)
            if not ok2:
                continue
            data = buf.tobytes()

            with cam.latest_lock:
                cam.latest_frame_bytes = data
                cam.last_update_ts = time.time()

            # no sleeping: we always overwrite newest frame

    def _try_submit_for_camera(self, cam: CameraSlot) -> Any:
        # only submit if we can acquire capacity
        acquired = self._in_flight_sem.acquire(blocking=False)
        if not acquired:
            return None

        with cam.latest_lock:
            if cam.latest_frame_bytes is None:
                self._in_flight_sem.release()
                return None
            frame_bytes = cam.latest_frame_bytes

        frame_id = str(uuid.uuid4())
        timestamp = int(time.time() * 1000)

        # store raw frame to edge MinIO
        # NOTE: tasks will delete this path after processing
        store_path = f"{self.minio_path_prefix}/{cam.camera_id}/{frame_id}.jpg"
        client, bucket_name = get_minio_client()
        store_data_in_minio(client, bucket_name, store_path, frame_bytes)

        payload = ETLReturnResult(
            user_id=self.edge_id,
            stream_id=cam.camera_id,
            frame_id=frame_id,
            timestamp=timestamp,
            store_path=store_path,
            items=None,
        )

        chain_result = get_etl_pipeline(payload).apply_async()
        print(
            f"[DEBUG] Submitted task chain ID: {chain_result.id} for frame {frame_id}"
        )
        # payload metadata for batch send
        self._submitted.put(
            (
                chain_result.id,
                {
                    "edge_id": self.edge_id,
                    "camera_id": cam.camera_id,
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                },
            )
        )
        return chain_result.id

    def _post_batch(self, batch_id: str, frames: List[Dict[str, Any]]) -> None:
        print("_post_batch")
        headers = {"X-Secret-Api-Key": self.secret_api_key}
        payload = {
            "edge_id": self.edge_id,
            "batch_id": batch_id,
            "camera_ids": sorted({f["camera_id"] for f in frames}),
            "processed_count": len(frames),
            "frames": [
                {
                    "edge_id": f["edge_id"],
                    "camera_id": f["camera_id"],
                    "frame_id": f["frame_id"],
                    "timestamp": f["timestamp"],
                    "items": f["items"],
                }
                for f in frames
            ],
        }

        resp = self.session.post(
            self.global_ingest_url, headers=headers, json=payload, timeout=30
        )
        resp.raise_for_status()

    def _scheduler_loop(self) -> None:
        print("_scheduler_loop")
        completed_frames: List[Dict[str, Any]] = []

        # poll interval while waiting for task results
        frame_submitted_count = 0
        while not self._stop.is_set():
            # submit as many as possible (newest frame only: overwrite semantics)
            for cam in self.cameras:
                if len(completed_frames) >= self.batch_size:
                    break
                before = (
                    self._in_flight_sem._value
                    if hasattr(self._in_flight_sem, "_value")
                    else None
                )
                task_id = self._try_submit_for_camera(cam)
                if task_id:
                    frame_submitted_count += 1
                    print(
                        f"[EDGE {self.edge_id}] submitted camera_id={cam.camera_id} frame_task_id={task_id}"
                    )
            if (
                frame_submitted_count
                and frame_submitted_count % self.batch_size == 0
            ):
                print(
                    f"[EDGE {self.edge_id}] submitted total tasks: {frame_submitted_count}"
                )

            # collect completed tasks
            # We drain the queue, attempt to resolve results, and requeue unfinished.
            temp: List[Tuple[str, Dict[str, Any]]] = []
            while True:
                try:
                    task_id, meta = self._submitted.get_nowait()
                except queue.Empty:
                    break

                result = celery_app.AsyncResult(task_id)
                if result.ready():
                    if result.failed():
                        # consider skipping but must release semaphore
                        self._in_flight_sem.release()
                        continue

                    data = result.result
                    # data is ETLReturnResult.model_dump()
                    items = data.get("items") or []

                    completed_frames.append(
                        {
                            "edge_id": meta["edge_id"],
                            "camera_id": meta["camera_id"],
                            "frame_id": meta["frame_id"],
                            "timestamp": meta["timestamp"],
                            "items": items,
                        }
                    )
                    print(
                        f"[EDGE {self.edge_id}] completed camera_id={meta['camera_id']} frame_id={meta['frame_id']} items={len(items)}"
                    )
                    # pipeline is done => release semaphore
                    self._in_flight_sem.release()

                else:
                    temp.append((task_id, meta))

            # put unfinished back
            for item in temp:
                self._submitted.put(item)

            if len(completed_frames) >= self.batch_size:
                batch_id = str(uuid.uuid4())
                to_send = completed_frames[: self.batch_size]
                completed_frames = completed_frames[self.batch_size :]

                print(
                    f"[EDGE {self.edge_id}] sending batch_id={batch_id} processed_count={len(to_send)} camera_ids={[f['camera_id'] for f in to_send]}"
                )
                try:
                    self._post_batch(batch_id, to_send)
                    print(
                        f"[EDGE {self.edge_id}] sent batch_id={batch_id} successfully"
                    )
                except Exception as e:
                    print(
                        f"[EDGE {self.edge_id}] failed to send batch_id={batch_id}: {e}"
                    )
                    # If send failed, keep frames for next attempt
                    completed_frames = to_send + completed_frames
                    time.sleep(2)

            time.sleep(0.05)


def main() -> None:
    EdgeDaemon().start()


if __name__ == "__main__":
    main()
