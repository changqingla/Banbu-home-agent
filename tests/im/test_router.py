from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from banbu.audit.log import AuditLog
from banbu.config.settings import Settings
from banbu.control.plane import ControlPlane
from banbu.devices.definition import DeviceSpec, ResolvedDevice
from banbu.devices.resolver import DeviceResolver
from banbu.im.router import make_router
from banbu.policy.access import AccessPolicy, AccessPolicyFile
from banbu.reactive.agent_runner import ReactiveAgentRunner
from banbu.state.snapshot_cache import SnapshotCache
from banbu.turn.scheduler import TurnScheduler


class FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict]] = []

    async def run(self, local_id: int, payload: dict) -> None:
        self.calls.append((local_id, payload))


class FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    async def create(self, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=self.responses.pop(0)),
                )
            ]
        )


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


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


def _client(
    tmp_path,
    settings: Settings,
    users: tuple[str, ...],
    responses: list[str],
) -> tuple[TestClient, FakeExecutor]:
    resolver = _resolver()
    audit = AuditLog(tmp_path / "audit.sqlite")
    executor = FakeExecutor()
    cache = SnapshotCache(resolver)
    control = ControlPlane(executor, resolver, audit, _policy(*users), cache=cache)
    runner = ReactiveAgentRunner(
        settings=settings,
        resolver=resolver,
        control=control,
        audit=audit,
        cache=cache,
        client=FakeClient(responses),
    )
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
    client, executor = _client(
        tmp_path,
        settings,
        ("weixin:user_1",),
        ['{"intent":"control_request","tool_calls":[{"name":"execute_plan","args":{"local_id":2,"action":"turn_on"}}],"final_message":""}'],
    )

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


def test_weixin_route_answers_greeting_without_execution(tmp_path) -> None:
    settings = Settings(
        home_id="home_a",
        im_enabled=True,
        im_weixin_enabled=True,
        im_weixin_bridge_token="secret",
    )
    client, executor = _client(
        tmp_path,
        settings,
        ("weixin:user_1",),
        ['{"intent":"greeting","tool_calls":[],"final_message":"你好，我是 Banbu。"}'],
    )

    resp = client.post(
        settings.im_weixin_path,
        headers={"x-banbu-im-token": "secret"},
        json={
            "conversation_id": "conv_1",
            "user_id": "user_1",
            "message_id": "msg_1",
            "text": "你好",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["result"]["intent"] == "greeting"
    assert "Banbu" in body["reply"]
    assert executor.calls == []


def test_weixin_route_rejects_bad_bridge_token(tmp_path) -> None:
    settings = Settings(
        im_enabled=True,
        im_weixin_enabled=True,
        im_weixin_bridge_token="secret",
    )
    client, _executor = _client(tmp_path, settings, ("weixin:user_1",), [])

    resp = client.post(
        settings.im_weixin_path,
        headers={"x-banbu-im-token": "bad"},
        json={"conversation_id": "conv_1", "user_id": "user_1", "text": "打开玄关灯"},
    )

    assert resp.status_code == 200
    assert resp.json()["ignored"] is True
    assert "token mismatch" in resp.json()["reason"]
