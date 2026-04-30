from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal


IMPlatform = Literal["feishu", "weixin"]


@dataclass(frozen=True)
class IMAttachment:
    kind: str
    file_id: str | None = None
    file_name: str | None = None
    url: str | None = None
    path: str | None = None


@dataclass(frozen=True)
class IncomingIMMessage:
    platform: IMPlatform
    message_id: str
    chat_id: str
    user_id: str
    text: str
    home_id: str
    user_display_name: str | None = None
    attachments: list[IMAttachment] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return f"im:{self.platform}"


def make_message_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"
