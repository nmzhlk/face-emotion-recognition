from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    YOLO_PATH: str = "ml/models/yolo_face_detection.pt"
    RESNET_PATH: str = "ml/models/resnet_best_f1.pth"

    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""

    model_config = SettingsConfigDict(env_file=".env.example", extra="ignore")

    @property
    def redis_url(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/0"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"


settings = Settings()
