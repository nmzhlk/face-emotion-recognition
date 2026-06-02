# Face Emotion Recognition – Edge/Global redesign

This repo is split conceptually into:
- **Edge machine (app/):** captures camera frames from cable cameras, runs ML locally via Celery workers, stores raw frames in **edge-local MinIO**, then deletes raw frames after processing. It sends **merged results** to Global every **100 processed frames**.
- **Global machine (public/):** provides a simple ingest endpoint that validates a shared secret key and appends received batches to a txt/JSONL file. For now it only logs and stores JSONL lines.

## Run (containers)
At the moment `docker-compose.yml` contains both services for local demo.


## Global ingest endpoint
- `POST /api/ingest_batch`
- Header: `X-Secret-Api-Key: <SECRET_API_KEY>`
- It writes append-only JSONL to `GLOBAL_TXT_PATH` and prints `camera_id` for every frame received.

## Edge daemon
Runs:
- `python -m app.edge_daemon`

Important env vars (from `.env`):
- `EDGE_ID`
- `CAMERA_SOURCES` (e.g. `0;1` or `rtsp://user:pass@ip:554/..;rtsp://...`)
- `BATCH_SIZE` (default 100)
- `MAX_IN_FLIGHT` (bounded semaphore; prevents overflow)
- `SECRET_API_KEY`

Celery concurrency on the edge:
- `CELERY_YOLO_CONCURRENCY`
- `CELERY_RECOGNIZER_CONCURRENCY`
- `CELERY_EMOTIONS_CONCURRENCY`

## Local non-docker start
Global:
- `uvicorn public.main:app --reload --port 8000`


