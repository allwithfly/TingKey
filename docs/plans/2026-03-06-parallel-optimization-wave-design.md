# Parallel Optimization Wave Design

**Date:** 2026-03-06

**Goal**

以低冲突、高并行的方式，同时优化 GUI、REST API、CLI/Shell 三条子系统线，优先消除稳定性与可维护性痛点，并在主控集成波中统一验收与回归。

## Why Wave-Based Parallelism

当前仓库的痛点分布较分散：

- `desktop_gui_app.py` 中存在超长 worker 逻辑、职责混杂和静默失败。
- `desktop_service/` 中存在配置恢复、会话状态与 HTTP 错误语义不够清晰的问题。
- `asr_cli.py` 与 `asr_service_shell.py` 中存在重复逻辑和用户可见行为不一致。

这些问题域彼此相对独立，但若直接同时改共享层（例如 `speech_output.py`、`asr_media_utils.py`）会显著增加冲突风险。因此本轮采用“**并行子系统优化 + 主控统一集成**”结构。

## Wave 1: Three Independent Agents

### Agent 1: GUI Stability + Maintainability

**Write scope:**

- `desktop_gui_app.py`
- 可选新增一个仅供 GUI 使用的辅助模块
- 可选新增对应测试文件

**Primary goals:**

- 收窄 `_StreamPartialWorker.run` 的静默失败行为，避免无痕吞错。
- 将 `AsrTaskWorker` 中可抽离的纯逻辑拆出为可测试辅助函数或更小的方法。
- 保持 GUI 行为不变前提下，提升故障可观测性与可维护性。

### Agent 2: REST API Hardening

**Write scope:**

- `desktop_service/api_server.py`
- `desktop_service/session_manager.py`
- `desktop_service/config_store.py`
- 可选新增 API / service 单测文件

**Primary goals:**

- 统一参数无效、会话不存在、状态不合法时的错误语义。
- 避免配置文件损坏时静默覆盖而不留痕。
- 改善队列/会话边界条件处理，减少无提示降级。

### Agent 3: CLI / Shell Consistency

**Write scope:**

- `asr_cli.py`
- `asr_service_shell.py`
- 可选新增仅供这两者共享的辅助模块
- 可选新增对应测试文件

**Primary goals:**

- 统一 CLI / Shell 的参数校验和 FFmpeg 路径解析逻辑。
- 收敛重复 helper，减少后续修改成本。
- 修顺用户可见输出与错误提示的一致性。

## Wave 2: Main Integration

主控不让多个 agent 同时改共享层。Wave 1 完成后，由主控负责：

- 审查三路结果是否冲突
- 统一命名、边界处理和错误语义
- 决定是否需要第 2 轮共享层收口
- 运行集中验证命令

## Data / Control Flow

1. 主控写设计和计划文档。
2. 主控按明确边界生成 3 个并行任务。
3. 各 agent 在各自负责的文件范围内独立修改。
4. 主控读取改动结果，逐项集成。
5. 主控运行统一验证，确认各子系统改动可共存。

## Conflict Prevention

- Wave 1 不允许多个 agent 同时修改 `asr_media_utils.py`、`speech_output.py` 等共享核心文件。
- 每个 agent 只对自己的文件集负责。
- 若某个 agent 认为必须改共享层，只能报告建议，交由主控在集成波处理。

## Error Handling Strategy

- GUI：避免静默吞错，至少记录或可回传失败原因。
- API：将 400 / 404 / 409 / 500 语义尽量区分清楚。
- CLI / Shell：参数错误尽早失败，避免进入半运行状态后才报错。

## Testing Strategy

- 优先补纯 Python `unittest`，避免引入新的测试框架。
- 每个 agent 先写失败用例，再补最小实现。
- 主控最终跑聚合验证：定向单测 + `py_compile`。

## Success Criteria

- 三个子系统各自至少解决一个高价值痛点。
- 改动边界清晰，合并时无文件冲突或仅有可控冲突。
- 新增或调整的行为均有最小验证支撑。
- 主控可以用一轮集中验证确认改动协同工作。
