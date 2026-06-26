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

    # CIFT activation-probe lab (opt-in `cift` dependency group). These are read
    # by the cift/ package; they have no effect unless that group is installed.
    cift_model_name: str = Field(default="Qwen/Qwen2.5-1.5B-Instruct", alias="AIS_CIFT_MODEL")
    cift_device: str = Field(default="auto", alias="AIS_CIFT_DEVICE")
    cift_ridge: float = Field(default=1e-2, alias="AIS_CIFT_RIDGE")
    cift_seed: int = Field(default=0, alias="AIS_CIFT_SEED")
    cift_gate_enabled: bool = Field(default=False, alias="AIS_CIFT_GATE")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
