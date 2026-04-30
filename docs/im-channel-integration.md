# IM Channel Integration

Banbu supports IM messages as a passive entry path into the existing reactive
turn pipeline.

```text
Feishu / personal WeChat
  -> banbu.im adapter
  -> IncomingIMMessage
  -> Turn.from_reactive()
  -> ReactiveRunner
  -> ControlPlane
  -> device execution
```

The IM layer is intentionally an adapter boundary. Platform-specific auth,
payload shape, and reply mechanics stay in `banbu/im/*`; device control still
goes through `ControlPlane`.

## Shared Settings

```env
BANBU_IM_ENABLED=true
```

Each channel also has its own enable switch, so the endpoints can be deployed
incrementally.

## Feishu

Feishu receives events at:

```text
POST /api/v1/im/feishu/events
```

Required settings for inbound messages:

```env
BANBU_IM_FEISHU_ENABLED=true
BANBU_IM_FEISHU_VERIFICATION_TOKEN=<event verification token>
```

Optional settings for sending replies back to Feishu:

```env
BANBU_IM_FEISHU_REPLY_ENABLED=true
BANBU_IM_FEISHU_APP_ID=<app id>
BANBU_IM_FEISHU_APP_SECRET=<app secret>
BANBU_IM_FEISHU_API_BASE_URL=https://open.feishu.cn
```

The adapter handles Feishu URL verification by returning the `challenge`.
Text and post messages are converted into Banbu reactive turns. Attachments are
kept on the unified message for future multimodal handling, but v1 only routes
text commands.

## Personal WeChat

Personal WeChat is exposed through a bridge/connector endpoint:

```text
POST /api/v1/im/weixin/messages
```

Required settings:

```env
BANBU_IM_WEIXIN_ENABLED=true
BANBU_IM_WEIXIN_BRIDGE_TOKEN=<shared bridge token>
```

The bridge should send:

```json
{
  "conversation_id": "wx_conversation",
  "user_id": "wx_user",
  "message_id": "msg_1",
  "text": "打开玄关灯",
  "response_url": "https://optional-bridge-reply-endpoint"
}
```

When `BANBU_IM_WEIXIN_BRIDGE_TOKEN` is set, the request must include either:

```text
X-Banbu-IM-Token: <shared bridge token>
```

or:

```text
Authorization: Bearer <shared bridge token>
```

If the inbound payload contains `response_url`, Banbu posts the text reply back
to that URL. Otherwise the API response itself contains the reply text.

This design matches the personal-WeChat reality: QR login, long polling, and
message upload/download belong to the replaceable bridge, while Banbu owns the
home automation turn handling.

## Authorization

IM users are mapped to reactive user ids:

```text
feishu:<sender user id>
weixin:<bridge user id>
```

Those ids must be allowed in `banbu/config/policy.yaml` before they can execute
device actions.
