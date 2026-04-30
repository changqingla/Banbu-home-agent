from banbu.config.settings import Settings
from banbu.vision.service import VisionService


def test_vision_service_skips_when_enabled_without_loaded_matching_scenes() -> None:
    service = VisionService(
        Settings(vision_enabled=True, vision_device_id="entry_camera_vision_1"),
        scenes=[],
    )

    service.start()

    assert service._task is None
