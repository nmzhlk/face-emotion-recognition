import os
import ssl
import logging

from celery import Celery
from pydantic_settings import BaseSettings, SettingsConfigDict


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='app.log',
    filemode='a'
)



class Settings(BaseSettings):
    YOLO_PATH: str
    RESNET_PATH: str
    
    ORA_USER: str
    ORA_PASS: str
    ORA_DSN: str
    STATIC_DIR: str
    TEMPLATES_DIR: str

    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    model_config = SettingsConfigDict(env_file=f"{BASE_DIR}.env.example")


settings = Settings()

redis_url = f"rediss://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}/0"

ssl_options = {"ssl_cert_reqs": ssl.CERT_NONE}

