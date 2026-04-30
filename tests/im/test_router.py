from fastapi import FastAPI
from fastapi.testclient import TestClient

from banbu.audit.log import AuditLog
from banbu.config.settings import Settings
from banbu.control.plane import ControlPlane
from banbu.devices.definition import DeviceSpec, ResolvedDevice
from banbu.devices.resolver import DeviceResolver
from banbu.im.router import make_router
from banbu.policy.access import AccessPolicy, AccessPolicyFile
from banbu.reactive.runner import ReactiveRunner
from banbu.turn.scheduler import TurnScheduler


class FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict]] = []

    async def run(self, local_id: int, payload: dict) -> None:
        self.calls.append((local_id, payload))


def _resolver() -> DeviceResolver:
    return DeviceResolver(
        [
            ResolvedDevice(
                spec=DeviceSpec(
                    friendly_name="switch_entry_light",
                    room="玄关",
                    role="light_switch",
                    aliases=["玄关灯"],
                ),
                local_id=2,
                ieee_address="0x2",
                model="test",
                capabilities={"state"},
            )
        ]
    )


def _policy(*users: str) -> AccessPolicy:
    return AccessPolicy(
        AccessPolicyFile.model_validate(
            {
                "reactive_users": {
                    user: {
                        "home_id": "home_a",
                        "allowed": [
                            {
                                "device": "switch_entry_light",
                                "actions": ["turn_on", "turn_off"],
                            }
                        ],
                    }
                    for user in users
                },
                "safety": {
                    "high_risk_roles": ["siren"],
                    "high_risk_actions": ["alarm_burglar", "alarm_fire"],
                    "proactive_allowed_scenes": [],
                },
            }
        )
    )


def _client(tmp_path, settings: Settings, users: tuple[str, ...]) -> tuple[TestClient, FakeExecutor]:
    resolver = _resolver()
    audit = AuditLog(tmp_path / "audit.sqlite")
    executor = FakeExecutor()
    control = ControlPlane(executor, resolver, audit, _policy(*users))
    runner = ReactiveRunner(resolver=resolver, control=control, audit=audit)
    scheduler = TurnScheduler()
    app = FastAPI()
    app.include_router(make_router(settings=settings, runner=runner, scheduler=scheduler))
    return TestClient(app), executor


def test_weixin_route_runs_reactive_turn(tmp_path) -> None:
    settings = Settings(
        home_id="home_a",
        im_enabled=True,
        im_weixin_enabled=True,
        im_weixin_bridge_token="secret",
    )
    client, executor = _client(tmp_path, settings, ("weixin:user_1",))

    resp = client.post(
        settings.im_weixin_path,
        headers={"x-banbu-im-token": "secret"},
        json={
            "conversation_id": "conv_1",
            "user_id": "user_1",
            "message_id": "msg_1",
            "text": "打开玄关灯",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["platform"] == "weixin"
    assert body["reply"] == "已打开玄关灯。"
    assert executor.calls == [(2, {"state": "ON"})]


def test_weixin_route_rejects_bad_bridge_token(tmp_path) -> None:
    settings = Settings(
        im_enabled=True,
        im_weixin_enabled=True,
        im_weixin_bridge_token="secret",
    )
    client, _executor = _client(tmp_path, settings, ("weixin:user_1",))

    resp = client.post(
        settings.im_weixin_path,
        headers={"x-banbu-im-token": "bad"},
        json={"conversation_id": "conv_1", "user_id": "user_1", "text": "打开玄关灯"},
    )

    assert resp.status_code == 200
    assert resp.json()["ignored"] is True
    assert "token mismatch" in resp.json()["reason"]
