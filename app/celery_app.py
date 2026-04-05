from celery import Celery
from app.config import settings

celery_app = Celery(
    "face_recognition",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks.process_frame"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    # Важно для ML — один кадр за раз на воркер
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)
