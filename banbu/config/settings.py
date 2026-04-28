from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="BANBU_",
        extra="ignore",
    )

    llm_base_url: str = "http://localhost:8889/v1"
    llm_model: str = "local-model"
    llm_api_key: str = "EMPTY"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_timeout_seconds: int = 60
    llm_toolcall_mode: str = "auto"

    iot_base_url: str = "http://192.168.1.78:8000"
    iot_timeout_seconds: int = 10

    host: str = "0.0.0.0"
    port: int = 9000
    webhook_path: str = "/api/v2/events/batch"

    cp_use_gpu: bool = False

    home_id: str = "home_default"
    fallback_poll_seconds: int = 30
    log_level: str = "INFO"
    db_path: Path = Path("./data/banbu.sqlite")
    devices_file: Path = Path("./banbu/config/devices.yaml")
    scenes_dir: Path = Path("./banbu/config/scenes")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
