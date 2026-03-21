from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    YOLO_PATH: str = "ml/models/yolo_face_detection.pt"
    RESNET_PATH: str = "ml/models/resnet_best_f1.pth"

    ORA_USER: str = "admin"
    ORA_PASS: str = "123"
    ORA_DSN: str = "oracle-xe:1521/XEPDB1"

    STATIC_DIR: str = "app/ui"
    TEMPLATES_DIR: str = "app/ui"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
