import ssl

from celery import Celery

from app.config import settings


def get_redis_url(db: int = 0) -> str:
    auth = f":{settings.REDIS_PASSWORD}@" if settings.REDIS_PASSWORD else ""
    return f"rediss://{auth}{settings.REDIS_HOST}:{settings.REDIS_PORT}/{db}"


CELERY_BROKER_URL = get_redis_url(settings.CELERY_BROKER_DB)
CELERY_RESULT_BACKEND = get_redis_url(settings.CELERY_RESULT_DB)

celery_app = Celery(
    "celery_worker", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND
)

ssl_options = {"ssl_cert_reqs": ssl.CERT_NONE}

celery_app.conf.update(
    broker_use_ssl=ssl_options,
    redis_backend_use_ssl=ssl_options,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    enable_utc=True,
    timezone="Europe/Moscow",
    broker_connection_retry_on_startup=True,
)

celery_app.conf.task_routes = {
    "tasks.yolo": {"queue": "yolo_queue"},
    "tasks.recognizer": {"queue": "recognizer_queue"},
    "tasks.emotions": {"queue": "resnet_queue"},
}
