from __future__ import annotations

import logging
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

import requests
from celery import Task, chain, chord
from celery.result import AsyncResult

import app.services.tasks  # noqa: F401
from app.core.celery_app import celery_app
from app.core.logging_config import setup_logging
from app.core.minio_client import get_minio_client, store_data_in_minio
from app.schemas.frame import ETLReturnResult


@dataclass
class CameraSlot:
    camera_id: str
    source: str
    latest_frame_bytes: Optional[bytes] = None
    latest_lock: threading.Lock = threading.Lock()
    last_update_ts: float = 0.0


def parse_camera_sources(value: str) -> List[str]:
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
        setup_logging(service_name="edge-daemon")
        self.logger = logging.getLogger(__name__)

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
        self.logger.info(
            f"raw_sources is '{raw_sources}' type: {type(raw_sources)}"
        )
        self.camera_sources = parse_camera_sources(raw_sources)
        self.logger.info(f"parsed sources: {self.camera_sources}")

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

        self.minio_bucket = "photos"
        self.minio_path_prefix = f"/{self.edge_id}"

        self.session = requests.Session()
        self._task_cache: Dict[str, Any] = {}

    def _get_task(self, name: str) -> Task:
        if name not in self._task_cache:
            try:
                task = celery_app.tasks[name]
                self._task_cache[name] = task
            except KeyError:
                self.logger.error(f"Task {name} not registered")
                raise
        return cast(Task, self._task_cache[name])

    def start(self) -> None:
        for cam in self.cameras:
            t = threading.Thread(
                target=self._camera_capture_loop,
                args=(cam,),
                daemon=True,
            )
            t.start()
        self._scheduler_loop()

    def stop(self) -> None:
        self._stop.set()

    def _camera_capture_loop(self, cam: CameraSlot) -> None:
        self.logger.info("_camera_capture_loop")
        import cv2

        src = cam.source
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

            ok2, buf = cv2.imencode(".jpg", frame)
            if not ok2:
                continue
            data = buf.tobytes()

            with cam.latest_lock:
                cam.latest_frame_bytes = data
                cam.last_update_ts = time.time()

    def _build_and_submit_chain(self, payload: ETLReturnResult) -> AsyncResult:
        yolo_task = self._get_task("app.services.tasks.yolo")
        recognizer_task = self._get_task("app.services.tasks.recognizer")
        emotions_task = self._get_task("app.services.tasks.emotions")
        merge_task = self._get_task("app.services.tasks.merge_results")

        yolo_sig = yolo_task.s(payload.store_path).set(queue="yolo_queue")
        chord_sig = chord(
            [
                recognizer_task.s().set(queue="recognizer_queue"),
                emotions_task.s().set(queue="resnet_queue"),
            ],
            merge_task.s(payload.model_dump()).set(queue="merge_queue"),
        )
        full_chain: chain = chain(yolo_sig, chord_sig)
        return full_chain.apply_async()

    def _try_submit_for_camera(self, cam: CameraSlot) -> Any:
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

        async_result: Any = self._build_and_submit_chain(payload)

        self.logger.info(
            f"Submitted task chain ID: {async_result.id} for frame {frame_id}"
        )
        self._submitted.put(
            (
                async_result.id,
                {
                    "edge_id": self.edge_id,
                    "camera_id": cam.camera_id,
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                },
            )
        )
        return async_result.id

    def _post_batch(self, batch_id: str, frames: List[Dict[str, Any]]) -> None:
        self.logger.info("_post_batch")
        headers: Dict[str, str] = {"X-Secret-Api-Key": self.secret_api_key}
        payload: Dict[str, Any] = {
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

        resp: requests.Response = self.session.post(
            self.global_ingest_url,
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()

    def _scheduler_loop(self) -> None:
        self.logger.info("_scheduler_loop")
        completed_frames: List[Dict[str, Any]] = []
        frame_submitted_count = 0

        while not self._stop.is_set():
            frame_submitted_count = self._submit_new_tasks(
                frame_submitted_count, completed_frames
            )
            self._collect_completed_tasks(completed_frames)
            if len(completed_frames) >= self.batch_size:
                completed_frames = self._process_batch_if_ready(
                    completed_frames
                )
            time.sleep(0.05)

    def _submit_new_tasks(self, count: int, completed: list) -> int:
        for cam in self.cameras:
            if len(completed) >= self.batch_size:
                break
            task_id = self._try_submit_for_camera(cam)
            if task_id:
                count += 1
                self.logger.info(
                    f"[EDGE {self.edge_id}] submitted camera_id={cam.camera_id} frame_task_id={task_id}"
                )

        if count and count % self.batch_size == 0:
            self.logger.info(
                f"[EDGE {self.edge_id}] submitted total tasks: {count}"
            )
        return count

    def _collect_completed_tasks(self, completed_frames: list) -> None:
        temp: List[Tuple[str, Dict[str, Any]]] = []
        while True:
            try:
                task_id, meta = self._submitted.get_nowait()
            except queue.Empty:
                break

            result: AsyncResult = celery_app.AsyncResult(task_id)
            if not result.ready():
                temp.append((task_id, meta))
                continue

            self._handle_completed_result(result, meta, completed_frames)

        for item in temp:
            self._submitted.put(item)

    def _handle_completed_result(
        self,
        result: AsyncResult[Any],
        meta: Dict[str, Any],
        completed_frames: list,
    ) -> None:
        try:
            if not result.failed():
                data = result.result
                if isinstance(data, dict):
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
                    self.logger.info(
                        f"[EDGE {self.edge_id}] completed camera_id={meta['camera_id']} items={len(items)}"
                    )
        finally:
            self._in_flight_sem.release()

    def _process_batch_if_ready(self, frames: list) -> list:
        batch_id = str(uuid.uuid4())
        to_send = frames[: self.batch_size]
        remaining = frames[self.batch_size :]

        self.logger.info(
            f"[EDGE {self.edge_id}] sending batch_id={batch_id} processed_count={len(to_send)}"
        )
        try:
            self._post_batch(batch_id, to_send)
            self.logger.info(
                f"[EDGE {self.edge_id}] sent batch_id={batch_id} successfully"
            )
            return remaining
        except Exception as e:
            self.logger.error(
                f"[EDGE {self.edge_id}] failed to send batch_id={batch_id}: {e}"
            )
            time.sleep(2)
            return to_send + remaining


def main() -> None:
    EdgeDaemon().start()


if __name__ == "__main__":
    main()
