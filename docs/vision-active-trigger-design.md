# Banbu 视觉主动触发链路设计

## 1. 背景

当前 Banbu 的主动触发链路已经跑通了传感器事件：

```text
IoT / camera-like event source
  -> POST /api/v2/events/batch
  -> ingest.normalizer
  -> DeviceEvent
  -> dispatcher
  -> SceneRuntime
  -> ProactiveTrigger
  -> Turn
  -> Context
  -> Agent
  -> ControlPlane
```

但 `agent-architecture-design.md` 里规划的视觉主动触发链路还没有落地。`video_analyser` 项目已经具备一套可复用的视觉检测能力：

- RTSP 常驻读流。
- 低成本运动检测，避免每帧都调用 VLM。
- 单次 VLM 调用完成多场景匹配。
- 置信度阈值、连续命中、冷却期。
- 命中后输出结构化结果。

本设计的目标不是把 `video_analyser` 整体搬进 Banbu，而是把它改造成 Banbu 的一个视觉事件生产者。视觉模块检测到场景变化后，不直接控制设备，也不直接发业务 webhook，而是统一上报到 Banbu 现有的 `/api/v2/events/batch` 入口。

## 2. 核心决策

### 2.1 视觉事件也走统一 batch 入口

传感器变化现在通过以下接口进入 Banbu：

```text
POST http://192.168.0.188:9000/api/v2/events/batch
```

视觉模块也使用同一个接口。这样主动触发入口只有一个，后续的快照更新、反向索引、SceneRuntime、ProactiveTrigger、Agent 主干都保持一致。

这意味着视觉模块命中后不会调用 `dispatcher.on_event()`，也不会绕过 webhook 直接产 trigger。它只负责生产一条 batch event：

```text
vision monitor task
  -> vision detector
  -> POST /api/v2/events/batch
  -> Banbu ingest pipeline
```

### 2.2 摄像头/视觉检测器建模为虚拟设备

为了复用现有 `DeviceEvent`、`SnapshotCache` 和 `ReverseIndex`，视觉模块在 Banbu 中表现为一个虚拟设备。

示例：

```yaml
devices:
  - friendly_name: entry_camera_vision_1
    role: vision_detector
    care_fields:
      - scene_id
      - confidence
      - detected
      - reason
      - frame_id
      - frame_at
```

这个虚拟设备不一定真实存在于 IoT 平台，但它在 Banbu 的事件语义上等价于一个会上报状态的设备：

- `scene_id`：当前帧最匹配的视觉场景，未命中时为 `null`。
- `confidence`：VLM 对该场景的置信度，范围 `0..1`。
- `detected`：是否有有效命中。
- `reason`：视觉模型给出的短原因。
- `frame_id`：视觉模块生成的帧 ID。
- `frame_at`：帧采集时间。

### 2.3 SceneRuntime 仍然负责“是否命中”

视觉 VLM 的输出只是一个设备状态变化事实，不等于 Banbu Scene 已经命中。

视觉模块可以做第一层过滤：

- 运动检测。
- VLM 场景分类。
- 输出 `scene_id/confidence/reason`。

Banbu SceneRuntime 仍然负责最终命中判断：

- 置信度是否达到阈值。
- 是否连续命中 N 次。
- 是否处于 inflight。
- 是否处于 cooldown。
- 是否满足其他设备 preconditions。

这样可以保持架构原则：主动入口只产生结构化事件，业务场景是否成立由 Banbu 的 Scene 层决定。

## 3. 事件协议

视觉模块上报仍使用当前 batch 格式：

```json
{
  "changed_at": "2026-04-28T10:30:05.120Z",
  "reported_at": "2026-04-28T10:30:05.320Z",
  "source": "vision",
  "payload": [
    {
      "device_id": "entry_camera_vision_1",
      "sequence": 1842,
      "values": {
        "scene_id": "hand_on_cheek_color_temp_light_v1",
        "detected": true,
        "confidence": 0.86,
        "reason": "画面中有人用手托腮",
        "frame_id": "frm_20260428_103005_1842",
        "frame_at": 1777372205.12
      },
      "previous_values": {
        "scene_id": null,
        "detected": false,
        "confidence": 0,
        "reason": "无匹配场景",
        "frame_id": "frm_20260428_103004_1841",
        "frame_at": 1777372204.98
      }
    }
  ]
}
```

未命中时也可以上报，但 v1 建议只在以下情况上报，减少事件量：

- 从未命中变为命中。
- 命中场景发生变化。
- 当前命中场景的置信度跨过关键阈值。
- 从命中恢复为未命中，用于重置运行时连续命中状态。

## 4. Scene 定义

视觉场景可以先复用当前 `kind: sequential` 的结构，但长期看更适合引入新的 scene kind：

```yaml
scene_id: hand_on_cheek_color_temp_light_v1
name: 用手托腮时打开色温灯
kind: vision_match

trigger:
  device: entry_camera_vision_1
  field: payload.scene_id
  value: hand_on_cheek_color_temp_light_v1

vision_policy:
  confidence_threshold: 0.7
  consecutive_hits: 1
  reset_on_miss: true

context_devices:
  trigger:
    - entry_camera_vision_1
  context_only:
    - color_temp_light_1

preconditions:
  - device: color_temp_light_1
    field: payload.state
    op: neq
    value: "ON"
    on_missing: skip

intent: 当视野中有人用手托腮时，打开色温灯
actions_hint:
  - tool: execute_plan
    args:
      device: color_temp_light_1
      action: turn_on

policy:
  cooldown_seconds: 30
  inflight_seconds: 20
  priority: 6
```

`vision_policy` 是视觉运行时的场景级参数。它替代 `video_analyser` 里以 `PHONE_PICKUP_*` 命名的全局环境变量，让每个视觉场景独立配置阈值和连续命中次数。

## 5. 运行时设计

新增一个视觉场景运行时：

```text
VisionMatchSceneRuntime
```

它接收的是普通 `DeviceEvent`，但只关心 `role=vision_detector` 的虚拟设备事件。

运行时状态：

```text
positive_streak: int
last_seen_scene_id: str | None
last_detection_at: float | None
inflight_until: float
cooldown_until: float
```

判定流程：

```text
DeviceEvent(values.scene_id/confidence/detected)
  -> scene_id 是否等于当前 scene_id
  -> detected 是否为 true
  -> confidence 是否 >= confidence_threshold
  -> positive_streak += 1
  -> positive_streak >= consecutive_hits
  -> 检查 inflight/cooldown
  -> 评估 preconditions
  -> 产 ProactiveTrigger
```

任意一帧未命中或置信度不足时，根据 `reset_on_miss` 决定是否清零 `positive_streak`。

## 6. 与 main.py 的关系

`main.py` 可以启动一个 `vision monitor task`，但这个 task 的职责只到“生产 batch event”为止。

推荐 v1 结构：

```text
main.py lifespan
  -> start IoT webhook route
  -> start IoT fallback poller
  -> start vision monitor task

vision monitor task
  -> RTSP 取帧
  -> motion gate
  -> VLM detect
  -> BatchEventPublisher.post(settings.webhook_public_url, payload)
```

注意这里的 `vision monitor task` 不直接持有 `Dispatcher`，也不直接调用 `on_hit`。它和外部 IoT 平台一样，只是 `/api/v2/events/batch` 的事件生产者。

这样做有三个好处：

- 主动入口统一，日志和审计路径更清楚。
- 视觉模块将来拆成独立进程时，不需要改 Banbu 主干。
- 传感器事件和视觉事件的测试方式一致，都可以用 curl 或脚本 POST batch payload。

### 6.1 `.env` 配置

视觉模块所有可变参数都走 `.env`，代码里不写死 RTSP 地址、VLM 地址、模型名或 API key：

```dotenv
BANBU_VISION_ENABLED=false
BANBU_VISION_RTSP_URL=
BANBU_VISION_DEVICE_ID=entry_camera_vision_1
BANBU_VISION_POST_BASE_URL=http://127.0.0.1:9000
BANBU_VISION_VLM_BASE_URL=http://localhost:30000/v1
BANBU_VISION_VLM_MODEL=local-vision-model
BANBU_VISION_VLM_API_KEY=EMPTY
BANBU_VISION_VLM_TIMEOUT_SECONDS=60
BANBU_VISION_FRAME_INTERVAL_SECONDS=0.12
BANBU_VISION_JPEG_QUALITY=85
BANBU_VISION_RECONNECT_SECONDS=5.0
BANBU_VISION_MOTION_SAMPLE_SIZE=32
BANBU_VISION_MOTION_PIXEL_DIFF_THRESHOLD=18
BANBU_VISION_MOTION_CHANGED_RATIO_THRESHOLD=0.03
```

其中 `BANBU_VISION_POST_BASE_URL` 指向 Banbu 服务自身对视觉模块可访问的地址，最终上报地址由它和 `BANBU_WEBHOOK_PATH` 拼出，例如：

```text
http://127.0.0.1:9000/api/v2/events/batch
```

## 7. 需要调整的现有模块

### 7.1 devices registry

当前 `devices/registry.py` 会把 `devices.yaml` 中的设备与 IoT 平台真实设备列表对账。视觉虚拟设备不会出现在 IoT 平台，因此需要支持：

```yaml
devices:
  - friendly_name: entry_camera_vision_1
    role: vision_detector
    virtual: true
    local_id: -1001
    ieee_address: virtual:entry_camera_vision_1
    care_fields: [scene_id, detected, confidence, reason, frame_id, frame_at]
```

虚拟设备不调用：

- `/api/v1/devices`
- `/api/v1/exposes`
- `/devices/report-config`

但仍然加入 `DeviceResolver`，并提供 capabilities 给 scene loader 校验。

### 7.2 normalizer

当前 `normalize_batch()` 已经支持 `device_id` 为字符串，并优先尝试：

- numeric local_id
- friendly_name

所以视觉事件只要使用：

```json
{ "device_id": "entry_camera_vision_1", "values": { ... } }
```

即可被解析为普通 `DeviceEvent`。

需要补充的是：

- 保留 batch 顶层 `source=vision`，写入 `DeviceEvent.source`。
- 对虚拟设备允许负数或固定 local_id。
- 明确 `previous_values` 由视觉模块自己维护，用于产生稳定 diff。

### 7.3 reverse index

视觉场景也进入反向索引：

```text
(entry_camera_vision_1, payload.scene_id)
  -> [(hand_on_cheek_color_temp_light_v1, trigger)]
```

这样视觉事件和传感器事件在 dispatcher 层保持一致。

### 7.4 dispatcher

`Dispatcher` 根据 scene.kind 选择 runtime：

```text
kind=sequential    -> SequentialSceneRuntime
kind=vision_match  -> VisionMatchSceneRuntime
```

`on_event()` 不需要知道事件来自传感器还是视觉，只需要查反向索引并把事件交给对应 runtime。

### 7.5 context assembler

视觉触发产出的 `ProactiveTrigger.facts` 应包含视觉事实：

```json
{
  "vision": {
    "camera": "entry_camera_vision_1",
    "scene_id": "hand_on_cheek_color_temp_light_v1",
    "confidence": 0.86,
    "reason": "画面中有人用手托腮",
    "frame_id": "frm_20260428_103005_1842",
    "frame_at": 1777372205.12
  }
}
```

`context/assembler.py` 应把这部分作为独立 context block 交给 Agent。v1 不把图片直接送给主 Agent，只把 VLM 识别结果作为结构化事实传入。

## 8. 从 video_analyser 迁移哪些能力

适合迁入 Banbu：

- `rtsp_monitor.py` 的 RTSP 读流、JPEG 编码、最新帧缓存。
- 32x32 灰度差分运动检测。
- VLM busy 时跳过新帧的背压策略。
- `build_detection_prompt()` 的多场景分类方式。
- `normalize_scene_result()` 的 JSON 容错解析。
- 连续命中、置信度阈值、冷却期的运行时思想。

不迁入或延后：

- 前端聊天页面。
- `/api/v3/bots/chat/completions` 对话接口。
- 用户视觉问答分支。
- 长期视频记忆。
- `PHONE_PICKUP_WEBHOOK_URL` 这类外部 webhook 告警模式。

## 9. 分期

### 阶段 1：统一事件协议

- 支持 `virtual: true` 设备。
- 支持视觉 batch payload 进入 `normalize_batch()`。
- 写一个脚本手动 POST 视觉事件到 `/api/v2/events/batch`。
- 退出标准：日志出现 `EVENT vision/entry_camera_vision_1 ... scene_id`，并能更新 snapshot。

### 阶段 2：vision_match runtime

- 新增 `Scene.kind = vision_match`。
- 新增 `VisionMatchSceneRuntime`。
- 视觉事件通过 reverse index 进入 runtime。
- 退出标准：连续两条高置信度视觉事件产出 `ProactiveTrigger`。

### 阶段 3：接入 video_analyser 的检测能力

- 新增 RTSP monitor task。
- 新增 VLM detector。
- 检测结果通过 `BatchEventPublisher` POST 到 `/api/v2/events/batch`。
- 退出标准：摄像头画面变化后，视觉模块自动上报 batch event。

### 阶段 4：进入 Agent 主干

- `ProactiveTrigger.facts` 带上视觉事实。
- `context/assembler.py` 输出视觉 context block。
- Agent 根据视觉事实和设备快照决定是否执行动作。
- 退出标准：视觉场景触发后，能完整走到 `ControlPlane.execute()`。

## 10. 第一版推荐方案

第一版只做一个视觉场景：

```text
hand_on_cheek_color_temp_light_v1
```

业务语义：

```text
当视野中有人用手托腮时，色温灯亮起
```

判定边界保持简洁：必须明确看到手接触并支撑脸颊、下巴或脸侧，类似休息或思考姿势；手只是靠近脸、挥手、挠脸、喝水或拿手机都不算命中。这个场景一次命中即可触发，仍然只控制一个低风险设备：色温灯。不要第一版就接安全告警、燃气、烟雾等高风险动作。

第一版的关键验收链路：

```text
RTSP motion
  -> VLM returns hand_on_cheek_color_temp_light_v1
  -> POST /api/v2/events/batch
  -> normalize_batch
  -> SnapshotCache update
  -> ReverseIndex lookup
  -> VisionMatchSceneRuntime HIT
  -> ProactiveTrigger
  -> Turn
  -> Context
  -> Agent
  -> ControlPlane
```

这个方案让视觉模块和传感器模块在入口上真正统一，也保留了将来把视觉模块拆成独立服务的空间。
