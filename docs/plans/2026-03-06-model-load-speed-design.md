# Model Load Speed Design

**Date:** 2026-03-06

**Goal**

同时缩短 GUI 启动预加载、CLI/Shell 首次启动、以及模型“已加载但第一次真正转写仍卡顿”的等待时间，并把优化方案统一到一条可复用的加载链路里。

## Current Bottlenecks

- `speech_output.py` 中 `QwenSpeechOutput.__init__` 每次都会走依赖导入、设备与 dtype 决策、`Qwen3ASRModel.from_pretrained(...)`，这是所有入口的共同冷启动成本。
- GUI 和服务端只做了**进程内缓存**，无法帮助一次一进程的 CLI 冷启动。
- 时间戳相关 forced aligner 在构造阶段决定，导致普通转写和时间戳转写无法完全解耦。
- “首次识别前卡顿”本质上是模型虽然已构造，但权重首次真正跑推理时仍有额外 warmup 成本。

## Recommended Strategy

采用双层方案：

1. **进程内加载优化**
   - 为所有入口共享统一加载阶段：依赖导入、模型核心加载、可选 aligner 加载、可选 warmup。
   - 增加共享 runtime registry，避免同进程内重复构建相同模型实例。
   - 将 aligner 延迟到真正请求时间戳时再加载。
   - 增加显式 warmup，提前吃掉首次推理卡顿。

2. **CLI 优先复用本地常驻 runtime**
   - 为 CLI 提供优先连接本地 runtime 的快路径。
   - 如果本地 runtime 可用，则复用已经热启动的模型。
   - 如果不可用，则自动回退到当前本地加载流程，不破坏原有使用方式。

## Why This Is The Best Fit

### Option A: Only In-Process Optimization

优点：改动小、风险低。  
缺点：一次性 CLI 冷启动仍然必须完整加载模型，无法真正覆盖“所有情况”。

### Option B: Dual-Layer Strategy (Recommended)

优点：
- GUI / Shell / Service 获得进程内提速
- CLI 在本地 runtime 存在时显著受益
- 首次识别卡顿可通过 warmup 统一缓解

缺点：
- 增加中等复杂度的 runtime 复用逻辑
- 需要更明确的错误回退和测试设计

### Option C: Aggressive Model Format Changes

优点：潜在提速最大。  
缺点：需要量化 / 导出 / 后端迁移，风险最高，不适合先手。

## Architecture

### 1. Shared Load Pipeline

在 `speech_output.py` 中引入更清晰的 runtime 结构：

- 加载 key：由 model path、device、dtype、backend、attention、batch size 等组成
- runtime registry：同进程内按 key 复用已构造模型
- staged load API：
  - `load_core()`
  - `ensure_aligner_loaded()`
  - `warmup()`

### 2. Lazy Aligner

当前 forced aligner 配置在构造时就参与模型构建。新方案里：

- 普通转写先只加载模型核心
- 只有 `return_time_stamps=True` 时才确保 aligner 已就绪
- 若 aligner 缺失，则保留现有回退行为

### 3. Warmup Handling

增加轻量 warmup 接口：

- GUI：模型核心 ready 后，后台 warmup
- Shell / Service：启动后可选 warmup
- CLI：本地加载模式可选 warmup；runtime 模式通常无需额外 warmup

### 4. CLI Runtime Reuse

为 CLI 增加“优先复用本地 runtime”的路径：

- 尝试连接本地 runtime
- 成功：直接请求热模型推理
- 失败：回退到本地加载

runtime 可以基于现有 `desktop_service/` 扩展为直接转写接口，而不是另起一套完全独立协议。

## Behavior Changes

- GUI 会更早进入“可用”状态，warmup 移到后台进行。
- Shell / Service 首次启动后，后续转写保持热态。
- CLI 在本地 runtime 可用时显著缩短启动等待；不可用时行为与当前兼容。
- 时间戳需求不会再强迫普通转写路径承担 aligner 首次加载成本。

## Error Handling

- runtime 不可用时 CLI 自动回退，不应直接失败。
- aligner 延迟加载失败时，继续沿用当前 coarse segment fallback。
- warmup 失败只记录，不应阻塞主功能。

## Testing Strategy

- 新增纯 Python `unittest`，覆盖：
  - runtime registry key 与复用逻辑
  - lazy aligner 触发条件
  - warmup 调度与失败容错
  - CLI runtime 连接成功 / 回退路径
- 集成验证：
  - GUI preload 时间
  - CLI 首次执行时间
  - 首次真实转写时间

## Success Criteria

- GUI 首次“模型已就绪”时间下降
- CLI 在 runtime 可用时显著缩短启动等待
- 首次识别前卡顿下降
- 时间戳路径与普通路径解耦更清晰
- 原有回退逻辑保持兼容
