# Scene YAML schema (v1)

每个 `*.yaml` 文件描述一个 Scene。文件名建议与 `scene_id` 一致。

```yaml
scene_id: entry_auto_light_v1     # 全局唯一 ID
name: 进门自动开灯                 # 人类可读名称
kind: sequential                  # v1 仅支持 sequential

trigger:
  steps:
    - device: door_sensor_1       # 必须出现在 devices.yaml
      field: payload.contact      # 必须在该设备 /exposes 范围内
      transition: "true->false"   # old->new
    - device: entry_pir_1
      field: payload.occupancy
      transition: "false->true"
      within_seconds: 30          # 仅 step >= 2 需要

context_devices:
  trigger: [door_sensor_1, entry_pir_1]
  context_only: [presence_radar_1, switch_entry_light]

# Runtime 在产 trigger 之前直接判断的硬条件。
preconditions:
  - device: presence_radar_1
    field: payload.illuminance
    op: lt                        # eq / neq / lt / lte / gt / gte / in
    value: 30
    on_missing: skip              # skip / pass / fail
  - device: switch_entry_light
    field: payload.state
    op: neq
    value: "ON"

intent: 进门时如果光线不足且灯未开则开灯
actions_hint:
  - tool: execute_plan
    args: { device: switch_entry_light, action: turn_on }

policy:
  cooldown_seconds: 60            # 仅在执行成功后写入
  inflight_seconds: 30            # 命中到执行结束之间的并发短锁
  priority: 5
```

视觉场景使用 `kind: vision_match`，`trigger` 指向视觉虚拟设备，并可用
`vision_criteria` 描述 VLM 判断标准：

```yaml
kind: vision_match
trigger:
  device: entry_camera_vision_1
  field: payload.scene_id
  value: hand_on_cheek_color_temp_light_v1

vision_criteria:
  - Match only when the visual condition is clearly present.
  - Return null if unsure.
```

字段含义见 `docs/implementation-plan.md` §4.2 / §4.2.1。
