from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
TEXTBOOKS_DIR = DATA_DIR / "textbooks"

load_dotenv(ROOT_DIR / ".env")


class Settings(BaseSettings):
    app_name: str = "Grade 7 AI Study Buddy"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    vision_model: str = "local-deterministic-vision-captioner"
    vector_store_backend: str = "local"

    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()

