from pathlib import Path
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

    vision_enabled: bool = False
    vision_rtsp_url: str = ""
    vision_device_id: str = "entry_camera_vision_1"
    vision_post_base_url: str = "http://127.0.0.1:9000"
    vision_vlm_base_url: str = "http://localhost:30000/v1"
    vision_vlm_model: str = "local-vision-model"
    vision_vlm_api_key: str = "EMPTY"
    vision_vlm_timeout_seconds: int = 60
    vision_frame_interval_seconds: float = 0.12
    vision_jpeg_quality: int = 85
    vision_reconnect_seconds: float = 5.0
    vision_motion_sample_size: int = 32
    vision_motion_pixel_diff_threshold: int = 18
    vision_motion_changed_ratio_threshold: float = 0.03

    cp_use_gpu: bool = False

    home_id: str = "home_default"
    registry_strict: bool = False
    fallback_poll_seconds: int = 30
    log_level: str = "INFO"
    db_path: Path = Path("./data/banbu.sqlite")
    devices_file: Path = Path("./banbu/config/devices.yaml")
    scenes_dir: Path = Path("./banbu/config/scenes")
    policy_file: Path = Path("./banbu/config/policy.yaml")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
