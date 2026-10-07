# Repository Guidelines

## Project Structure & Module Organization

Source lives in `src/typesafe_mario/`: `state.py` parses emulator telemetry and RAM, `actions.py` defines controller macros, `policy.py` implements Jev and heuristic policies, `runner.py` manages episodes, `dashboard.py` renders the live UI, and `cli.py` exposes commands. Keep model-facing structured JSON separate from debug views; timing calculations belong in code, while Jev chooses actions.

Tests live in `tests/`. CI is defined in `.github/workflows/ci.yml`; packaging and dependencies are in `pyproject.toml`. Generated decision logs go to ignored `artifacts/run-<timestamp>.jsonl`. No game assets or ROMs are tracked.

## Build, Test, and Development Commands

Use Python 3.13 or newer. Create and activate a virtual environment, then run:

- `python -m pip install -e ".[dev]"` — install the package and development tools, matching CI.
- `python -m pip install -e ".[mario,dev]"` — also install emulator and dashboard dependencies for gameplay.
- `typesafe-mario state-demo` — inspect sample model input without launching the emulator or calling the API.
- `typesafe-mario play --env SuperMarioBros-1-1-v0 --frames-per-decision 8` — launch the live dashboard.
- `ruff format src tests` — format Python files.
- `ruff format --check src tests` and `ruff check src tests` — run CI formatting and lint checks.
- `python -m pytest -q` — run the test suite.

Packaging uses Hatchling; there is no separate application build script.

## Coding Style & Naming Conventions

Use four-space indentation, Ruff formatting, and the configured 100-character line length. Follow existing type annotations and modern Python syntax. Use `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants and enum members.

## Testing Guidelines

Existing tests use `unittest.TestCase`, discovered by pytest. Name files `test_*.py` and methods `test_<behavior>`. Add deterministic regression cases for parser, action, or dashboard logic changes, using synthetic telemetry and RAM. Keep unit tests independent of API credentials and gameplay. No numeric coverage threshold is configured.

## Commit & Pull Request Guidelines

The single existing commit uses `feat: add TypeSafe Mario agent demo`; follow that short, imperative, type-prefixed style. In PRs, describe the behavior change, link relevant issues, report validation commands and results, and include screenshots for dashboard changes. Run all CI checks before requesting review.

## Security & Configuration

Provide `TYPESAFE_API_KEY` through the environment. Never commit credentials, `.env`, generated logs, or ROM files. Gameplay requires a lawful local game setup.

## 事实修复与决策策略变更规则

1. **优先修复事实性问题**：核对 RAM、角色定义、坐标、状态、时间单位、日志与实际执行的一致性。每项修复必须有可复现证据。
2. **事实修复不夹带策略调整**：不顺带修改 Jev 提示词、动作描述、威胁判断或预测职责；必要的字段变化也只限于纠正事实。
3. **决策预测策略单独处理**：先明确范围，经确认后完整调整涉及的输入、预测、提示和测试，并做从开局到终局的完整运行验证，不能用固定旧动作前缀的局部测试代替。
4. **分开报告结果**：事实是否正确、执行是否一致、策略是否改善，分别给出证据。

## Jev 请求与响应消息格式

以下格式依据 `src/typesafe_mario/policy.py`、`state.py`、`actions.py` 和本地 `typesafe-sdk 0.7.2` 核对。实际提示词以 `TypeSafePolicy.choose()` 为准，动作说明以 `ACTION_DESCRIPTIONS` 为准；本节只描述消息契约，不新增策略。

### 请求：结构化状态与三个命名问题

调用入口为 `client.system_one(state=snapshot.to_state(), questions=questions)`，SDK 发送 `POST /v1/systemone`。请求体使用 `state`、`model`、`questions`，不是聊天接口的 `messages`/`role` 格式。当前代码未显式指定模型，本地 SDK 默认模型为 `jev-latest`。

下面是结构模板：`<...>` 表示从代码填入的文本，`state` 中的空对象仅省略字段展示，不代表实际发送空对象。完整状态由 `MarioSnapshot.to_state()` 生成，不应手工维护第二套字段。

```json
{
  "model": "jev-latest",
  "state": {
    "objective": "Reach the flag in World 1-1 without dying. Prioritize survival and forward progress, then safely obtain mushroom/fire-flower upgrades, collect visible coins, hit reachable reward blocks from below, and stomp suitable enemies for additional score when the observed geometry supports a low-risk opportunity.",
    "scoring": {},
    "opportunities": {},
    "level": {},
    "player": {},
    "trajectory": {},
    "hazard": {},
    "terrain": {},
    "reaction_timing": {},
    "recent_control": {},
    "committed_control": null,
    "episode": {}
  },
  "questions": {
    "next_action": {
      "type": "choice",
      "instructions": {
        "question": "Which controller macro should Mario commit to next?",
        "goal": "<snapshot.goal，与 state.objective 一致>",
        "reward_blocks": "<使用 head_bump_geometry 判断靠近、对齐及向上顶击，并考虑惯性、承诺控制和 pursuit 约束>",
        "powerups": "<强化砖与已生成道具的顶击、接取、惯性及安全约束>",
        "scoring": "<沿途收益取舍、追分边界及不确定性说明>",
        "timing": "<按 decision_horizon_frames 生成的动作周期说明>",
        "geometry": "<当前地形提示词>",
        "trajectory": "<当前轨迹提示词>",
        "stall": "<当前受阻提示词>",
        "enemy_timing": "<当前敌人时序提示词>",
        "delay": "<当前动作延迟提示词>"
      },
      "criteria": {
        "noop": "<ACTION_DESCRIPTIONS[Action.NOOP]>",
        "right": "<ACTION_DESCRIPTIONS[Action.RIGHT]>",
        "right_jump": "<ACTION_DESCRIPTIONS[Action.RIGHT_JUMP]>",
        "right_run": "<ACTION_DESCRIPTIONS[Action.RIGHT_RUN]>",
        "right_run_jump": "<ACTION_DESCRIPTIONS[Action.RIGHT_RUN_JUMP]>",
        "jump": "<ACTION_DESCRIPTIONS[Action.JUMP]>",
        "left": "<ACTION_DESCRIPTIONS[Action.LEFT]>"
      }
    },
    "jump_needed": {
      "type": "noul",
      "instructions": "<当前是否需要起跳或保持跳跃的提示词>"
    },
    "danger": {
      "type": "score",
      "instructions": "How dangerous is Mario's immediate situation?",
      "criteria": [
        "Safe open movement",
        "Potential obstacle or enemy soon",
        "Immediate collision, fall, or enemy threat"
      ]
    }
  }
}
```

`next_action.criteria` 实际只包含传入 `choose(snapshot, actions)` 的可选动作；上面展示全部七种动作。`jump_needed` 未设置 `criteria`，SDK 序列化时省略该可选字段。

| `state` 字段 | 内容与单位 |
| --- | --- |
| `objective`、`level` | 目标，以及 `world`、`stage`、`area`。 |
| `player` | `x`、`y`（像素），水平/垂直速度（像素/帧），`grounded`、`jump_phase`、`powerup_status`；垂直速度为正表示上升。 |
| `trajectory` | 腾空帧数、起跳后水平位移（像素）、`crossing_known_gap`、起跳时记录的坑宽（瓦片）。 |
| `hazard` | 敌人相对位置、水平接触估计、落地估计及紧迫性标记。距离为像素，时间为模拟器帧；未知估计为 `null`，不是零。恒速水平接触估计不等于实际碰撞预测。 |
| `terrain` | 地形几何、可靠性及 `last_grounded_preview`；距离/宽度字段按名称区分瓦片或像素，`*_world_x` 为世界像素坐标。几何不可用时部分字段会省略，不能把缺失理解为没有障碍或坑。 |
| `reaction_timing` | `action_horizon_frames`、`last_inference_delay_frames`、`total_reaction_horizon_frames`，均为模拟器帧；总窗口为前两项之和，不是 API 墙钟耗时。 |
| `recent_control` | 最近实际输入、观察帧数、前进像素及动作结果摘要。 |
| `committed_control` | 已承诺宏 `action`、下一帧实际输入 `first_frame_action`、所选宏开始前的帧数 `frames_before_selected_action`，以及 `release_jump_while_falling`、`press_jump_on_landing`；未附加时为 `null`。 |
| `episode` | 生命数、游戏剩余时间、当前/最佳进度、受阻帧数、死亡及通关标记。 |
| `scoring` | 实际游戏分数、当前金币计数、上一实际帧分数差、确认收集/踩踏累计及未归因分数；未知为 null。确认计数是下界，金币计数不是累计收集量。 |
| `scoring.pursuit` | `allow_extra_effort`、未刷新最佳横向进度的帧数、限制原因。游戏时间 <=100、时间未知或连续 48 帧未刷新最佳进度时停止额外追分，仍可在前进中顺便得分。 |
| `opportunities` | availability、观测帧、世界横向观测边界 [left,right)、最多 8 个当前前方候选及 truncated。金币砖包含 Mario 当前所在列。相对坐标为像素，Y 正值向下；碰撞框为屏幕坐标；金币瓦片原点不冒充碰撞框。coin_block 区分 question/brick，required_interaction 为 hit_from_below，tile_bounds_screen_xyxy 是半开屏幕瓦片边界，不是碰撞框；另识别 0xc1 的 powerup_block；不包含隐藏砖、空砖或其他特殊奖励砖，不估计剩余金币数或顶砖成功率。 |

每个机会包含 target_id、kind、observed_frame、relative_x_pixels、relative_y_pixels、
world_x、collision_box_screen_xyxy、source、validity；敌人还包含 enemy_kind、
motion_state、stompable 和 relative_velocity_x。无关或未知字段为 null。
stompable 只表示已验证的对象状态允许踩踏，不是踩踏成功预测；未识别敌人继续留在 hazard，
不能因为机会列表为空而认为没有危险。完整当前候选与事件证据保留在 debug_state。
三个问题仍由 Jev 同时回答，jump_needed 现在也考虑符合通关优先要求的取币和踩踏跳跃。

可用 `typesafe-mario state-demo` 查看完整示例状态；该命令不调用 Jev，也不展示完整 HTTP 请求。`debug_state`、`state_text`、RAM 和截图不在当前 Jev 请求中。每次调用发送当前结构化状态及问题，代码不附加聊天历史。

### 响应：按问题名称返回答案

奖励砖的 `head_bump_geometry` 描述当前背景碰撞头部探测点，包含头部世界 X、
屏幕 Y、砖块横向边界相对头部的半开区间、头部到砖底的上升距离（正数表示在下方）
及砖块弹跳是否进行中。小 Mario/蹲下与站立大 Mario 的偏移不同；不支持的游泳
状态为 null。字段不是未来命中预测。`reward_blocks` 提示仅在 pursuit 允许且当前
候选列表有已知头部几何的金币砖或强化砖时发送；Jev 判断对齐、惯性、承诺动作和安全性，
选择现有宏，执行器不覆盖选择。

powerup_block.expected_powerup_if_hit_now 依据当前 RAM 状态给出条件性产物；顶出后以
独立槽位的 powerup.powerup_kind 为准。emergence_phase 区分 emerging/active，早期
未更新接触几何时碰撞框为 null。道具用触碰接取，不作为敌人或踩踏目标；player 新增
collision_box_screen_xyxy 用于比较双方屏幕碰撞框。powerups 提示仅在 pursuit 允许
且当前列表存在强化砖或道具时启用；顶出、消失或分数变化都不单独证明已获得强化。

以下为格式示例，概率、模型名称和 token 数仅为演示值，不是实测结果。响应的 `model` 是服务返回的模型名称，可能不同于请求中的别名。

```json
{
  "model": "jev-latest",
  "answers": {
    "next_action": {
      "type": "choice",
      "choice": "right_jump",
      "confidence": 0.7,
      "probabilities": {
        "noop": 0.02,
        "right": 0.08,
        "right_jump": 0.7,
        "right_run": 0.05,
        "right_run_jump": 0.1,
        "jump": 0.03,
        "left": 0.02
      }
    },
    "jump_needed": {"type": "noul", "noul": 0.85},
    "danger": {
      "type": "score",
      "score": 1.6,
      "confidence": 0.8,
      "legend": {
        "0": "Safe open movement",
        "1": "Potential obstacle or enemy soon",
        "2": "Immediate collision, fall, or enemy threat"
      },
      "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}
    }
  },
  "usage": {"input_tokens": 1000, "output_tokens": 20}
}
```

`choice` 对应请求中的一个动作名；`confidence` 和各项概率取值为 0–1，概率分布之和约为 1。`noul` 是“是”的概率，不是布尔值。`danger.score` 是等级 0、1、2 的概率加权平均，范围为 0–2，可为小数；它不是死亡概率。HTTP JSON 的评分等级键为字符串，SDK 的 `ScoreAnswer.legend`/`probabilities` 使用整数键。

### SDK 读取、本地决策与日志

`TypeSafePolicy._answer()` 优先从 SDK 的分类视图读取，缺少对应条目时回退到 `response.answers[question_id]`；两处均缺失则抛出 `KeyError`。

| SDK 读取路径 | 本地 `Decision` 字段 |
| --- | --- |
| `response.choices["next_action"].choice` | `action`，转换为 `Action` 枚举 |
| `response.choices["next_action"].confidence` | `confidence` |
| `response.choices["next_action"].probabilities` | `probabilities` |
| `response.nouls["jump_needed"].noul` | `jump_needed_probability` |
| `response.scores["danger"].score` | `danger_score` |
| 本地 `system_one()` 调用前后的计时差 | `latency_ms`，不是服务响应字段 |

`choices`、`nouls`、`scores` 是 SDK 从 `answers` 派生的访问视图，不是三个额外的 HTTP 响应顶层字段。当前代码直接采用 `next_action.choice`；跳跃概率和危险评分不作为二次动作覆盖规则。宏随后由执行器逐帧展开，因此所选宏与某帧实际按键可以不同。

`artifacts/run-*.jsonl` 保存状态、解析后的决策、执行信息和奖励等本地记录，不是原始 HTTP 响应归档；当前记录不保留响应 `model`、`usage` 或完整 `danger` 分布。分析时应区分模型输入、模型选择和实际执行。

得分扩展：决策日志的 scoring_result 记录 (start_frame,end_frame] 实际执行区间的收益，
不回填进请求时 state。trace schema_version=2 记录确认的收集、踩踏、受伤与分数变化，
episode_end.summary 包含累计收益和观测覆盖情况。初始 NOOP 收益只进入整局累计。
score_before_clear 为首次 clear 前紧邻观测的分数，score_at_clear 为 clear 帧分数；
两者都不是结束动画结算后的最终分数。缺失分数不按零计算，确认事件数只提供下界。
无法唯一证明来源的分数归入 unattributed_score_gain；不会仅凭同时发生就把分数归因给踩踏或金币。
