import requests
from celery import Celery
from app.config import ssl_options, redis_url, settings
import redis

celery_app = Celery(
    "celery_worker",
    broker=redis_url,
    backend=redis_url
)


celery_app.conf.update(
    broker_use_ssl=ssl_options,
    redis_backend_use_ssl=ssl_options,
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    enable_utc=True,  # Убедитесь, что UTC включен
    timezone='Europe/Moscow',  # Устанавливаем московское время
    broker_connection_retry_on_startup=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)

redis_database = redis.Redis(host='localhost', port=6373, db=1) # Отдельная БД для кадров

@celery_app.task(
    name='analyse'
)