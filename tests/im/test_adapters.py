from fastapi import Request

from banbu.config.settings import Settings
from banbu.im.feishu_adapter import FeishuAdapter, IMAdapterError
from banbu.im.weixin_adapter import WeixinBridgeAdapter


def test_feishu_adapter_parses_text_event() -> None:
    adapter = FeishuAdapter(Settings(im_feishu_verification_token="verify_me"))

    msg = adapter.parse_event(
        {
            "header": {
                "event_id": "evt_1",
                "token": "verify_me",
                "create_time": "1777372205120",
            },
            "event": {
                "sender": {"sender_id": {"user_id": "user_1"}},
                "message": {
                    "message_id": "msg_1",
                    "chat_id": "oc_chat",
                    "message_type": "text",
                    "content": '{"text":"打开玄关灯"}',
                },
            },
        }
    )

    assert msg.platform == "feishu"
    assert msg.message_id == "msg_1"
    assert msg.chat_id == "oc_chat"
    assert msg.user_id == "feishu:user_1"
    assert msg.text == "打开玄关灯"


def test_feishu_adapter_rejects_bad_verification_token() -> None:
    adapter = FeishuAdapter(Settings(im_feishu_verification_token="expected"))

    try:
        adapter.parse_event({"header": {"token": "wrong"}, "event": {}})
    except IMAdapterError as e:
        assert "token mismatch" in str(e)
    else:
        raise AssertionError("expected token mismatch")


def test_feishu_adapter_parses_sdk_event() -> None:
    adapter = FeishuAdapter(Settings(im_feishu_verification_token="verify_me"))

    class Obj:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    event = Obj(
        header=Obj(event_id="evt_1", token="verify_me", create_time="1777372205120"),
        event=Obj(
            sender=Obj(sender_id=Obj(user_id="user_1", open_id=None, union_id=None)),
            message=Obj(
                message_id="msg_1",
                chat_id="oc_chat",
                message_type="text",
                content='{"text":"打开玄关灯"}',
                create_time="1777372205120",
            ),
        ),
    )

    msg = adapter.parse_sdk_message(event)

    assert msg.platform == "feishu"
    assert msg.message_id == "msg_1"
    assert msg.chat_id == "oc_chat"
    assert msg.user_id == "feishu:user_1"
    assert msg.text == "打开玄关灯"


def test_weixin_bridge_adapter_parses_message() -> None:
    adapter = WeixinBridgeAdapter(Settings(home_id="home_a"))

    msg = adapter.parse_message(
        {
            "conversation_id": "conv_1",
            "user_id": "user_1",
            "message_id": "msg_1",
            "text": "打开玄关灯",
            "attachments": [{"kind": "image", "path": "/tmp/frame.jpg"}],
        }
    )

    assert msg.platform == "weixin"
    assert msg.chat_id == "conv_1"
    assert msg.user_id == "weixin:user_1"
    assert msg.home_id == "home_a"
    assert msg.text == "打开玄关灯"
    assert len(msg.attachments) == 1
    assert msg.attachments[0].path == "/tmp/frame.jpg"


def test_weixin_bridge_adapter_checks_token() -> None:
    adapter = WeixinBridgeAdapter(Settings(im_weixin_bridge_token="secret"))

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "headers": [(b"x-banbu-im-token", b"bad")],
    }
    request = Request(scope)

    try:
        adapter.verify_request(request)
    except IMAdapterError as e:
        assert "token mismatch" in str(e)
    else:
        raise AssertionError("expected token mismatch")
