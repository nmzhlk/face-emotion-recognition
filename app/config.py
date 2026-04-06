import logging
import os

from pydantic_settings import BaseSettings, SettingsConfigDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="app.log",
    filemode="a",
)

BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class Settings(BaseSettings):
    YOLO_PATH: str = "ml/models/yolo_face_detection.pt"
    RESNET_PATH: str = "ml/models/resnet_best_f1.pth"

    STATIC_DIR: str = "app/ui"
    TEMPLATES_DIR: str = "app/ui"

    ORA_USER: str = "admin"
    ORA_PASS: str = "123"
    ORA_DSN: str = "db:1521/xepdb1"

    REDIS_HOST: str = "localhost"
    REDIS_PORT: str = "6379"
    REDIS_PASSWORD: str = "superpassword"

    CELERY_BROKER_DB: int = 0
    CELERY_RESULT_DB: int = 1

    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env"), extra="ignore"
    )


settings = Settings()
