from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "AI OCR Microservice"
    database_url: str = "sqlite:///./ocr.db"
    upload_dir: str = "uploads"
    max_file_size_mb: int = 20

    model_config = {"env_prefix": "OCR_"}


settings = Settings()
