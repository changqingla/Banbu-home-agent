# Banbu Agent 实施计划 (v1)

## 0. 阅读前提

本文档是对 `agent-architecture-design.md` 的工程化映射。架构文档回答“为什么这样设计”，本文档回答“怎么落地”。

v1 目标已锁定：**端到端走通一条线**——以“进门自动开灯”为唯一场景，跑通 `设备事件 → 反向索引 → Scene 运行时 → Turn → ContextPilot → Agent → 设备控制 → 状态回流` 整条链路，对接真实设备 API。

非目标：

- 不做 duration_triggered 扩展（edge_triggered / windowed_all 可作为独立小步补齐）
- 不做权限、家庭多租户、用户画像
- 不做长期记忆、语义事实
- 不做 Web/App 入口，被动入口先用 CLI 模拟

---

## 1. 技术栈

| 层 | 选型 | 理由 |
|---|---|---|
| 语言 | Python 3.11+ | LLM 生态、ContextPilot 原生 Python、设备 API 是 REST |
| Web 框架 | FastAPI | 既要做 webhook 接收（emergency/care 推送），也要做被动入口 HTTP，async 原生 |
| LLM 推理 | 本地 llama.cpp (`llama-server`)，由用户自行部署 | OpenAI 兼容 HTTP 接口；模型/端口/API key 全部走 env，框架不锁死 |
| LLM SDK | `openai` Python SDK，`base_url` 指向本地 llama-server | ContextPilot 输出就是 OpenAI 格式 messages，直接喂；无适配层 |
| 上下文优化 | `contextpilot` 库 | 直接 `cp.optimize(docs, query, conversation_id=...)` |
| 设备层 | 现有 IoT 平台 (`http://192.168.1.78:8000`) | 已提供 |
| 状态存储 | SQLite + 内存 | v1 单进程，Scene runtime / 快照缓存放内存，Scene 定义、审计落 SQLite |
| 配置 | YAML + pydantic-settings | Scene 定义用 YAML，运行时参数走 `.env`（pydantic-settings 自动加载） |
| HTTP 客户端 | `httpx`（async） | 调设备 API、调 llama-server |

**为什么不用 Redis / Postgres**：v1 单机单进程能跑通的事，不引入额外基础设施。Scene runtime 的 cooldown 和游标都是毫秒级访问，内存够用；定义和审计写 SQLite 就够。等 v2 多家庭/横向扩展时再换。

### 1.1 `.env` 配置

`.env.example` 提供模板，所有可变参数集中在这里，框架本身不写死任何模型名/端点：

```dotenv
# ─── LLM ─────────────────────────────────────────────
BANBU_LLM_BASE_URL=http://localhost:8889/v1   # llama-server OpenAI 兼容入口
BANBU_LLM_MODEL=local-model                   # llama-server --alias 或 gguf 文件名
BANBU_LLM_API_KEY=EMPTY                       # llama.cpp 不校验，占位即可
BANBU_LLM_TEMPERATURE=0.0                     # v1 决策可重现性优先
BANBU_LLM_MAX_TOKENS=1024
BANBU_LLM_TIMEOUT_SECONDS=60

# ─── 设备平台 ─────────────────────────────────────────
BANBU_IOT_BASE_URL=http://192.168.1.78:8000
BANBU_IOT_TIMEOUT_SECONDS=10

# ─── 自身服务 ─────────────────────────────────────────
BANBU_HOST=0.0.0.0
BANBU_PORT=9000
BANBU_WEBHOOK_PATH=/ingest/device-event       # IoT 平台 emergency 推送目标

# ─── ContextPilot ────────────────────────────────────
BANBU_CP_USE_GPU=false
# 可选：若用 contextpilot-llama-server 启动，需要
# CONTEXTPILOT_INDEX_URL=http://localhost:8765

# ─── 运行时 ──────────────────────────────────────────
BANBU_HOME_ID=home_default
BANBU_FALLBACK_POLL_SECONDS=30                # webhook 兜底轮询周期
BANBU_LOG_LEVEL=INFO
BANBU_DB_PATH=./data/banbu.sqlite
```

`config/settings.py` 用 `pydantic-settings` 加载，类型校验失败启动直接报错。换模型、换 IoT 平台地址、换端口都不用改代码。

---

## 2. 模块拆分

```
banbu/
├── adapters/
│   └── iot_client.py        # 设备 API 封装：list/state/control/history/report-config
├── ingest/
│   ├── webhook.py           # 接收 emergency/care 推送的 FastAPI 路由
│   └── normalizer.py        # IoT payload → 内部 DeviceEvent
├── devices/
│   ├── definition.py        # DeviceSpec 数据模型（pydantic）
│   ├── registry.py          # 加载 config/devices.yaml + 启动时与 IoT /devices 对账
│   └── resolver.py          # friendly_name ↔ local_id ↔ ieee_address 三向解析
├── scenes/
│   ├── definition.py        # Scene 数据模型（pydantic，含 preconditions）
│   ├── loader.py            # 扫描 config/scenes/*.yaml 全量加载
│   ├── reverse_index.py     # (device, field) → [(scene, role)]
│   └── runtime/
│       ├── base.py          # SceneRuntime 抽象
│       ├── sequential.py    # v1 唯一实现：顺序触发状态机
│       └── lifecycle.py     # inflight_until / cooldown_until 管理（见 §4.2.1）
├── dispatcher.py            # 事件入口：查反向索引 → 路由到 SceneRuntime → 产 trigger
├── turn/
│   ├── model.py             # Turn / ProactiveTrigger 数据模型
│   └── builder.py           # 把 trigger 或用户请求装成 Turn
├── context/
│   ├── selector.py          # 按场景收窄信息范围
│   ├── assembler.py         # 组装上下文块（场景定义/触发事实/设备快照/反馈）
│   └── pilot.py             # 调 contextpilot 的薄封装，设置 conversation_id
├── agent/
│   ├── loop.py              # OpenAI 兼容 tool-use 循环（指向本地 llama-server）
│   ├── tools.py             # 工具定义（read_device_state / execute_plan）
│   └── prompts.py           # system prompt 模板
├── control/
│   ├── plane.py             # 硬边界：能力检查、幂等、动作翻译
│   └── executor.py          # 调 iot_client 真正下发
├── state/
│   ├── snapshot_cache.py    # 内存里维护每个设备最新 payload
│   └── feedback.py          # 最近 N 次执行反馈
├── audit/
│   └── log.py               # SQLite 写审计
├── cli/
│   ├── probe.py             # 阶段 1 用：列出设备 + payload
│   └── reactive.py          # 被动入口（CLI 模拟用户请求）
├── config/
│   ├── settings.py          # pydantic-settings 加载 .env
│   ├── devices.yaml         # ★ 设备清单：用户编辑的设备语义元数据
│   └── scenes/              # ★ 自定义场景目录：用户增删 yaml 文件，启动时全量加载
│       ├── _schema.md       # 场景 YAML 字段说明（开发者参考）
│       └── entry_auto_light_v1.yaml
└── main.py                  # FastAPI app + 启动时载入设备清单/场景、注册 webhook
```

两组用户配置文件是整套系统的**业务输入面**：

- `config/devices.yaml` —— 你管的设备清单（角色、房间、别名）
- `config/scenes/*.yaml` —— 你定义的场景（每个文件一个场景）

两者都在 `main.py` 启动阶段读入并校验，校验失败立刻退出，不带病启动。详见 §4.7。

每个模块只做一件事，模块间通过 pydantic 数据对象串联。

---

## 3. 数据流

### 3.1 主动链路（事件 → 执行）

```
IoT 平台 emergency 推送
   │
   ▼
ingest/webhook.py  ──→  ingest/normalizer.py
   │                       │
   │                       ▼
   │                   DeviceEvent { local_id, field, old, new, ts }
   ▼
state/snapshot_cache  ◀──  始终更新最新 payload
   │
   ▼
dispatcher.py
   │  ① 查 reverse_index → [(scene, role)]
   │  ② role=context_only：仅更新快照，结束
   │  ③ role=trigger：送进 SceneRuntime
   ▼
scenes/runtime/sequential.py
   │  ① 检查 cooldown_until 与 inflight_until（任一未过期则丢弃）
   │  ② 推进游标 / 检查窗口
   │  ③ 评估硬条件（illuminance < threshold、light state != ON 等结构化判定）
   │  ④ 命中 → 产 ProactiveTrigger，并设置 inflight_until = now + 短锁（默认 30s）
   │       （此时不写 cooldown_until，留到执行成功后写）
   ▼
turn/builder.py  → Turn { thread_type=proactive, trigger, conversation_id }
   │
   ▼
context/selector.py → context/assembler.py → context/pilot.py
   │                                              │
   │                                              ▼
   │                            blocks 列表 + conversation_id
   │                                              │
   │                                              ▼
   │                                    cp.optimize(blocks, query, conv_id)
   │                                              │
   ▼                                              ▼
agent/loop.py（拿到 OpenAI 格式 messages，直接调本地 llama-server）
   │  循环：tool_call → control/plane → iot_client → tool_result
   ▼
control/executor.py → iot_client.control(local_id, payload)
   │
   ▼
audit/log.py + state/feedback
   │
   ├─ 执行成功：scenes/runtime 写 cooldown_until = now + cooldown_seconds
   │            清除 inflight_until
   │
   └─ 执行失败 / Agent 决定 skip：
              仅清除 inflight_until，不写 cooldown
              （让下一次同等条件的事件还能再次触发）
```

### 3.2 被动链路（用户请求 → 执行）

CLI 输入 utterance → `turn/builder` 生成 `thread_type=reactive` 的 Turn → 后续路径与主动链路在 `context/selector` 之后完全相同。

差别只在 selector：

- 主动：场景已经选好，按 scene_id 取场景定义和绑定设备
- 被动：先做意图识别（v1 用关键词匹配，把 “开灯/关灯/进门” 落到候选 scene），再走相同选择路径

---

## 4. 关键设计决策

### 4.1 事件来源：emergency 推送 + 启动时拉一次全量

设备 API 没有 WebSocket/MQTT，但有 `report-config` 把 emergency 关键词的变化即时 POST 到外部 endpoint。

策略：

1. 启动时调用 `/devices/allinfo` 把所有设备最新状态灌入 `snapshot_cache`
2. 启动时按 v1 场景关心的字段，给相关设备调 `/devices/report-config` 配置 emergency 关键词（contact, occupancy, presence 等）
3. FastAPI 起一个 webhook 端点 `/ingest/device-event` 接收推送
4. 兜底：起一个低频（30s）轮询任务比对 `allinfo`，防止漏推

这样 v1 不依赖轮询响应延迟，又能在设备 API 没推时不丢事件。

### 4.2 Scene 定义放 YAML，不做 DSL

v1 唯一场景示例（`config/scenes/entry_auto_light_v1.yaml`）：

```yaml
scene_id: entry_auto_light_v1
name: 进门自动开灯
kind: sequential
trigger:
  steps:
    - device: door_sensor_1
      field: payload.contact
      transition: "true->false"
    - device: entry_pir_1
      field: payload.occupancy
      transition: "false->true"
      within_seconds: 30
context_devices:
  trigger:
    - door_sensor_1
    - entry_pir_1
  context_only:
    - presence_radar_1
    - switch_entry_light

# 硬条件：runtime 在产 trigger 之前直接判断；任一不满足就不命中、不调 LLM。
# 字段引用 snapshot_cache 里的最新 payload。op 支持 eq/neq/lt/lte/gt/gte/in。
preconditions:
  - device: presence_radar_1
    field: payload.illuminance
    op: lt
    value: 30                    # 环境足够暗
    on_missing: skip             # 字段缺失或快照过老时的处理：skip / pass / fail
  - device: switch_entry_light
    field: payload.state
    op: neq
    value: "ON"                  # 灯当前未打开

intent: 进门时如果光线不足且灯未开则开灯
actions_hint:
  - tool: execute_plan
    args: { device: switch_entry_light, action: turn_on }
policy:
  cooldown_seconds: 60           # 仅在执行成功后生效
  inflight_seconds: 30           # 命中到执行结束之间的并发短锁
  priority: 5
```

字段说明：

- **preconditions** 是 runtime 在 §3.1 步骤 ③ 直接判断的硬条件，不交给 LLM 自由发挥。这样架构文档第 10.1 节要求的 “环境足够暗” “灯当前未打开” 在工程上就有机器可判定的归宿
- **on_missing** 决定快照缺字段或太旧时的兜底——`skip` 视为不满足、`pass` 视为满足、`fail` 抛错。illuminance 这种来自电池设备、可能久未上报的字段建议设 `skip`
- **inflight_seconds 与 cooldown_seconds 是两个独立锁**，原因见下一节
- 这里的硬条件是后端 “收范围” 的一部分；命中后 LLM 仍会拿到 illuminance、state 的快照值，可以做更软的判断（比如毫米波 presence 还在抖动时选择稍等再开）

`loader.py` 用 pydantic 校验。设备引用用 friendly_name，启动时和 IoT 平台对一遍 local_id 映射。

### 4.2.1 触发生命周期：inflight ≠ cooldown

为避免 “Scene HIT 后 LLM 或设备执行失败，结果 cooldown 把后续应该开的灯也屏蔽掉” 这种情况，runtime 维护两个独立时间戳：

| 字段 | 含义 | 写入时机 | 清除时机 |
|---|---|---|---|
| `inflight_until` | 同场景的并发短锁，防止本次还没结束又被同一串事件二次触发 | 命中即写（`now + inflight_seconds`，默认 30s） | 执行成功 / 执行失败 / skip 都清 |
| `cooldown_until` | 业务 cooldown，防止短时间内重复执行已经成功的动作 | **仅当 execute_plan 真正下发成功后写**（`now + cooldown_seconds`） | 自然到期 |

判定顺序（dispatcher → runtime）：

1. `cooldown_until` 未过期 → 丢弃
2. `inflight_until` 未过期 → 丢弃（已经有一次在处理中）
3. 推进游标 / 评估 preconditions
4. 命中 → 写 `inflight_until`，产 trigger

执行收尾分支：

- **execute_plan 成功**：写 `cooldown_until`，清 `inflight_until`，记 feedback=success
- **execute_plan 失败**（设备返回错、超时）：清 `inflight_until`，**不写 cooldown**，记 feedback=fail。下一次相同事件链仍能再次触发，给恢复留出路
- **Agent 决定 skip**（例如灯其实已经开了）：清 `inflight_until`，**不写 cooldown**。skip 不是 “业务上完成了一次开灯”，再次触发不该被静默
- **超时 / 进程崩溃**：`inflight_until` 自然过期（30s），不会永久卡住

幂等去重（同 trigger_id 重复请求）由 `control/plane.py` 用 5 秒短缓存兜底，与 inflight/cooldown 是两层独立机制。

### 4.3 Agent 工具集（v1 只给两把）

```python
tools = [
    {
        "name": "read_device_state",
        "description": "读取某设备当前 payload",
        "input_schema": {"local_id": int, "fields": list[str] | None},
    },
    {
        "name": "execute_plan",
        "description": "下发一个动作，需要先经过控制平面校验",
        "input_schema": {
            "local_id": int,
            "action": str,        # "turn_on" / "turn_off" 等统一语义
            "params": dict | None,
        },
    },
]
```

控制平面把统一语义翻译成 IoT 平台 payload（`turn_on` → `{"state": "ON"}`），并做：

- 设备能力检查（exposes 里有没有这个动作）
- 幂等去重（同 trigger_id + 同 plan 5 秒内忽略重复）
- 审计落库

Agent 看不到 payload 字段名，看到的只是 `action: turn_on`。这是文档 6.5 节的硬边界。

### 4.4 ContextPilot 集成方式

每次 Turn 调用：

```python
blocks = [
    SYSTEM_POLICY,             # 静态，全局共享
    TOOL_SCHEMA_TEXT,          # 静态
    scene_def_text,            # 按 scene 稳定
    trigger_facts_text,        # 本次唯一
    snapshot_text(door),       # 按设备分块
    snapshot_text(pir),
    snapshot_text(radar),
    snapshot_text(light),
    feedback_text,             # 最近 3 次执行反馈
]
query = "请判断本次场景触发是否需要执行，及执行什么。"

messages = cp.optimize(
    blocks,
    query,
    conversation_id=f"{home_id}_{scene_id}",  # 主动线程 ID
)

# messages 已经是 OpenAI 格式，直接喂 llama-server
resp = await openai_client.chat.completions.create(
    model=settings.llm_model,
    messages=messages,
    tools=TOOLS,                               # OpenAI function calling 风格
    tool_choice="auto",
    temperature=settings.llm_temperature,
)
```

`conversation_id` 用 `home_id + scene_id` 保证主动线程跨轮去重，不污染用户会话。被动线程的 `conversation_id` 用 `home_id + user_id`。

**关于 KV 前缀缓存**：

- 默认走原生 `llama-server`，ContextPilot 在文本层做重排和跨轮去重，前缀复用靠 llama.cpp 自带的 `--cache-reuse` 即可
- 如果用户后续改用 `contextpilot-llama-server` 启动（仅启动命令换前缀），ContextPilot 会通过 LD_PRELOAD 拦截 KV 失效事件，命中率更精确——这是优化项，不影响 v1 跑通
- Agent loop 不感知这层差异，永远只用标准 OpenAI SDK

### 4.5 工具调用兼容性的兜底

不同模型对 OpenAI `tools` 字段的支持质量不一。v1 做两层防御：

1. **首选**：用 `tools` + `tool_choice="auto"` 走结构化函数调用
2. **降级**：如果模型连续 N 次返回非法 tool_call JSON，agent loop 自动切到 ReAct 风格的纯文本 prompt（`Action: execute_plan\nArgs: {...}`），用正则解析。降级状态打日志，便于换模型时观察

env 里 `BANBU_LLM_TOOLCALL_MODE` 控制：`auto`（默认，先试结构化，失败降级）、`structured`（强制）、`react`（强制 ReAct）。

### 4.6 用户配置文件：设备清单与场景目录

两组文件都在 `config/` 下，是用户日常编辑的入口。框架代码不写死任何设备 ID 或场景定义。

#### `config/devices.yaml` —— 设备清单

```yaml
# Banbu 管理的设备子集（不在此列的设备框架不会主动配 emergency 推送、不会响应控制）。
# friendly_name 必须与 IoT 平台 /api/v1/devices 返回的 friendly_name 完全一致；
# local_id 启动时会向 /api/v1/devices 反查并校验，不一致则启动失败。

devices:
  - friendly_name: door_sensor_1
    room: 玄关
    role: door_sensor              # 语义角色，影响 prompt 模板与意图匹配
    aliases: [入户门磁, 大门门磁]   # 被动入口用户语句里可能出现的叫法
    care_fields: [contact]         # 启动时给该设备配的 emergency 关键词

  - friendly_name: entry_pir_1
    room: 玄关
    role: motion_sensor
    aliases: [玄关人体, 玄关红外]
    care_fields: [occupancy]

  - friendly_name: presence_radar_1
    room: 玄关
    role: presence_radar
    aliases: [玄关毫米波]
    care_fields: [presence, illuminance]

  - friendly_name: switch_entry_light
    room: 玄关
    role: light_switch
    aliases: [玄关灯, 入户灯]
    care_fields: [state]
    actions:                       # 可选：覆盖 control/plane.py 默认翻译表
      turn_on: { state: "ON" }
      turn_off: { state: "OFF" }
```

字段含义：

| 字段 | 必填 | 用途 |
|---|---|---|
| `friendly_name` | 是 | 主键。Scene YAML 引用、运行时与 IoT 平台对齐都用它 |
| `room` | 是 | 上下文装配时 prompt 里展示，被动入口意图匹配用 |
| `role` | 是 | 语义角色，决定默认动作翻译表（如 `light_switch` → `{state: ON/OFF}`）和上下文模板 |
| `aliases` | 否 | 被动入口的关键词扩展，用户说"开玄关灯"能落到 `switch_entry_light` |
| `care_fields` | 否 | 启动时框架自动调 `/devices/report-config` 配 emergency 关键词；不填则不配 |
| `actions` | 否 | 当 role 默认翻译表不够用时手动覆盖 |

`devices/registry.py` 启动时做的事：

1. 读 `config/devices.yaml`，pydantic 校验
2. 调 `/api/v1/devices` 拉真实清单
3. 对账：每个 friendly_name 找对应 `local_id` 和 `ieee_address`，找不到就报错退出
4. 调 `/api/v1/exposes?local_id=...` 拉每个设备的能力 spec，校验 `care_fields` 和 `actions` 在能力范围内
5. 按 `care_fields` 自动调 `/devices/report-config` 配 emergency 关键词
6. 把解析结果放 `devices/resolver.py` 的内存索引，供后续 friendly_name ↔ local_id ↔ ieee_address 互转

#### `config/scenes/*.yaml` —— 自定义场景目录

每个文件一个场景，文件名建议和 `scene_id` 一致（不强制）。`scenes/loader.py` 启动时 `glob('config/scenes/*.yaml')` 全部加载，pydantic 校验每个，反向索引一并构建。

YAML 完整字段已在 §4.2 给出，关键点：

- 场景里 `device:` 引用的 friendly_name 必须出现在 `devices.yaml` 里（loader 交叉校验，找不到就报错）
- `field:` 必须在该设备的 `/exposes` 能力范围内（防止写错字段名）
- 多个场景可以引用同一个设备；一个设备字段会被多个场景共用，反向索引把它表达成 `(device, field) → [(scene_a, trigger), (scene_b, context_only)]`

新增场景的工作流：

1. 在 `config/scenes/` 加一个 `living_room_no_one_off.yaml`
2. 如果引用了新设备，`devices.yaml` 同步加条目
3. 重启服务，启动日志会打印"载入 N 个场景，反向索引 M 条"
4. 校验失败的场景**整体丢弃**（不会半载入），日志给出确切原因

#### 与 `.env` 的分工

| 内容类型 | 放哪里 | 为什么 |
|---|---|---|
| 模型/端口/密钥/超时 | `.env` | 部署相关，不进 git |
| 设备清单 | `config/devices.yaml` | 业务相关，进 git |
| 场景定义 | `config/scenes/*.yaml` | 业务相关，进 git |
| 代码逻辑 | `banbu/**` | 框架本身 |

换部署环境只动 `.env`；换业务行为只动 `config/`；扩展能力才动代码。

### 4.7 状态存放在哪里

| 数据 | 存放 | 生命周期 |
|---|---|---|
| 设备清单（业务侧） | `config/devices.yaml` → 内存 | 启动时校验对账 |
| 设备能力 spec | `/api/v1/exposes` 启动拉取 → 内存 | 进程内（重启重拉） |
| Scene 定义 | `config/scenes/*.yaml` → 内存 | 启动时全量加载 |
| 反向索引 | 内存（dict） | 启动时构建 |
| Scene runtime（cursor / inflight_until / cooldown_until） | 内存 | 进程内，重启丢失（v1 接受） |
| Snapshot cache | 内存 | 启动 allinfo 灌入 + 持续更新 |
| 执行反馈（最近 N） | 内存 deque | 进程内 |
| 审计日志 | SQLite | 持久 |
| 用户会话历史 | SQLite | 持久（v1 简单的 turn 列表） |

v2 把 runtime 和 snapshot 搬 Redis 即可横向扩展，对上层接口零改动。

---

## 5. 实施分期

总计 5 个阶段，每个阶段都是可运行的可观察产物，避免“先写 1000 行再调”。

### 阶段 1：设备适配层 + 设备清单 + 启动同步（约 1.5 天）

- [ ] `iot_client.py`：封装 list/exposes/info/allinfo/control/history/report-config
- [ ] `devices/registry.py` + `config/devices.yaml`：加载、对账、能力校验、自动配 emergency
- [ ] `devices/resolver.py`：friendly_name ↔ local_id ↔ ieee_address 三向索引
- [ ] `snapshot_cache.py`：内存设备状态缓存，启动从 `/devices/allinfo` 拉取（只缓存清单内的设备）
- [ ] `cli/probe.py`：列出清单内所有设备 + 当前 payload + 与 IoT 实际状态对账结果
- [ ] **退出标准**：写一份 4 个设备的 `devices.yaml`，启动后 probe 能看到对账绿勾、状态实时；故意改错 friendly_name 启动会清晰报错

### 阶段 2：事件入口贯通（约 1 天）

- [ ] `ingest/webhook.py` + `normalizer.py`：FastAPI 收 emergency 推送
- [ ] 启动时给目标设备配 `report-config`（contact/occupancy/presence/state）
- [ ] 兜底 30s 轮询任务
- [ ] 打日志：每条事件打印 `local_id, field, old, new`
- [ ] **退出标准**：手动开关门/触发人体传感器，控制台立即看到事件

### 阶段 3：Scene 引擎 + 反向索引（约 2 天）

- [ ] Scene 数据模型 + `scenes/loader.py`（扫描 `config/scenes/*.yaml`，与 `devices.yaml` 交叉校验）
- [ ] `reverse_index.py`：构建 `(device, field) → [(scene, role)]`
- [ ] `runtime/sequential.py`：顺序触发状态机 + 窗口 + preconditions 评估
- [ ] `runtime/lifecycle.py`：inflight_until / cooldown_until 管理
- [ ] `dispatcher.py`：把事件路由到 runtime
- [ ] 写 1 份 `config/scenes/entry_auto_light_v1.yaml`
- [ ] **退出标准**：手动模拟“先开门、再触发 PIR”，日志输出 `Scene HIT, inflight 已写, ProactiveTrigger 已生成`，但还不调 LLM；故意把 illuminance 设亮，日志显示 precondition 拒绝命中

### 阶段 4：Turn + Context + Agent + 控制平面（约 3 天）

- [ ] `turn/builder.py`、`context/selector.py`、`context/assembler.py`、`context/pilot.py`
- [ ] `agent/loop.py`：OpenAI 兼容 tool use 循环 + ReAct 降级路径
- [ ] `control/plane.py` + `control/executor.py`：动作语义翻译 + 调 `iot_client.control`
- [ ] 审计写 SQLite
- [ ] **退出标准**：阶段 3 的 trigger 真的能让 LLM 输出 `execute_plan(turn_on)`，玄关灯实际亮起，cooldown 写入

### 阶段 5：被动入口 + 联调收尾（约 1 天）

- [ ] `cli/reactive.py`：终端输入“把玄关灯关了”这类自然语言
- [ ] 简单意图匹配 → 走相同主干
- [ ] 跑一遍完整 demo：开门 → 灯亮 → 终端说“关灯” → 灯灭 → 60 秒内再开门不会触发
- [ ] **退出标准**：录一段视频/日志，整链路可复现

合计约 8.5 个工作日。

---

## 6. 验证方式

| 模块 | 验证手段 |
|---|---|
| iot_client | 单测 + 真机 list/control |
| reverse_index | 单测：给定场景配置，断言索引结构 |
| sequential runtime | 单测：注入伪造事件序列，断言命中 / 不命中 / cooldown |
| dispatcher | 集成测：mock iot_client，喂事件，断言 Trigger 产出 |
| context/pilot | 单测：blocks 顺序、conversation_id 透传 |
| agent loop | 录制式：把一条 trigger 跑过 LLM，断言至少调用一次 `execute_plan` |
| 端到端 | 真机：开门触发 → 玄关灯 ON |

LLM 调用的非确定性：v1 默认 `temperature=0`（env 可改），并对 prompt 里 actions_hint 给的足够明确，让行为可重现。模型在用户那边自部署，调试时可以换模型只改 `.env`，不重启代码逻辑。

---

## 7. 与架构文档的对应关系

| 架构文档章节 | 落地模块 |
|---|---|
| 5.1 Scene | `scenes/definition.py` + `config/scenes/*.yaml`（含 `preconditions` 块表达硬条件） |
| 5.2 Trigger | `scenes/runtime/sequential.py` + `scenes/runtime/edge.py` + `scenes/runtime/windowed_all.py` |
| 5.3 Context Devices | YAML 字段 + `context/selector.py` |
| 5.4 Actions Hint | YAML 字段 → 注入 system prompt |
| 5.5 Runtime State | `scenes/runtime/*` 内存对象（`cursor` / `inflight_until` / `cooldown_until` 三段独立） |
| 6.3 后端硬边界 | `control/plane.py` |
| 6.5 统一设备协议 | `control/plane.py` 翻译表（`turn_on` → `{state: ON}`） |
| 7.1 Device Snapshot | `state/snapshot_cache.py`（来源 `/devices/allinfo`） |
| 7.1 Device 清单与语义 | `config/devices.yaml` + `devices/registry.py` + `devices/resolver.py` |
| 7.3 Trigger Event | `turn/model.py: ProactiveTrigger` |
| 7.4 Turn | `turn/model.py: Turn` |
| 8.1 反向索引 | `scenes/reverse_index.py` |
| 8.2 Scene Runtime | `scenes/runtime/sequential.py` |
| 8.3 命中后输出 | dispatcher 产 trigger + 写 `inflight_until`；`cooldown_until` 由执行成功分支回写（见 §4.2.1） |
| 9.1 统一入口 | `main.py` + `cli/reactive.py` + webhook |
| 9.2 会话层 | `conversation_id` 命名规则 |
| 9.3 上下文选择 | `context/selector.py` |
| 9.4 上下文装配与优化 | `context/assembler.py` + `context/pilot.py` |

---

## 8. 已知风险与暂缓项

| 项 | 状态 | 说明 |
|---|---|---|
| 设备 API 没有 push channel | 接受 | 用 emergency report-config 推送 + 兜底轮询 |
| 进程重启 runtime 丢失 | 接受 | v1 单机；v2 上 Redis |
| 多家庭 / 多用户 | 暂缓 | 字段已预留 home_id/user_id，逻辑只跑单家庭 |
| 用户画像 / 长期记忆 | 暂缓 | context/selector 留扩展点，不实现 |
| 复杂触发类型 | 部分支持 | 已支持 sequential / edge_triggered / windowed_all；duration 暂缓 |
| LLM 失控 | 受控 | 只暴露 2 个工具；execute_plan 必经 control plane |
| 本地模型 tool calling 不稳 | 已设防 | `BANBU_LLM_TOOLCALL_MODE=auto` 自动从结构化降级 ReAct |
| 本地模型 / 端点变更 | 零代码改动 | 全走 `.env`，重启即可生效 |

---

## 9. 一句话总结

**先用 8 天把"进门自动开灯"这一条线从真实门磁推到真实玄关灯打通，每个模块只做架构文档里给它的那一件事，其余全部用 v1 验收线砍掉。**
