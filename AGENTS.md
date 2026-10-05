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
    "objective": "Reach the flag in World 1-1 without dying.",
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
        "goal": "Advance toward the stage flag while avoiding death.",
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

可用 `typesafe-mario state-demo` 查看完整示例状态；该命令不调用 Jev，也不展示完整 HTTP 请求。`debug_state`、`state_text`、RAM 和截图不在当前 Jev 请求中。每次调用发送当前结构化状态及问题，代码不附加聊天历史。

### 响应：按问题名称返回答案

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
