from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AIS"
    database_url: str = Field(default="sqlite:///./ais.db", alias="AIS_DATABASE_URL")
    default_model: str = Field(default="mock-ais", alias="AIS_DEFAULT_MODEL")
    allow_provider_store: bool = Field(default=False, alias="AIS_ALLOW_PROVIDER_STORE")
    data_dir: Path = Path(".")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
