<p align="center">
  <h1 align="center">⚡ 听键 TingKey</h1>
  <p align="center">
    免费版 Typeless，开源版 闪电说 · Free & Open-Source Alternative to Typeless
    <br />
    基于 Qwen3-ASR 的本地桌面语音输入工具
    <br />
    <a href="./README.md">English</a> · <a href="https://github.com/allwithfly/TingKey/issues">报告 Bug</a> · <a href="https://github.com/allwithfly/TingKey/issues">功能建议</a>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License" />
  <img src="https://img.shields.io/badge/python-3.10+-brightgreen.svg" alt="Python" />
  <img src="https://img.shields.io/badge/platform-Windows-lightgrey.svg" alt="Platform" />
  <img src="https://img.shields.io/badge/model-Qwen3--ASR-orange.svg" alt="Model" />
</p>

---

**听键（TingKey）** 将 [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) 封装为一套可直接使用的本地语音识别桌面工具。按下快捷键说话，松开即可将识别文字输入到光标位置——完全本地运行，无需网络，隐私无忧。

## ✨ 功能亮点

- 🎙️ **快捷键录音** — 按住说话，松开即写入光标位置
- 🖥️ **桌面 GUI** — 模型预加载，实时进度，可中断任务
- 📁 **命令行批处理** — 支持通配符，批量转写音频/视频
- 🔄 **常驻交互 Shell** — 模型常驻内存，随时转写
- 📊 **基准测试** — 一键对比 0.6B 与 1.7B 模型性能
- 🌐 **REST API 后端** — 可选，方便集成
- 📝 **字幕导出** — 支持 `.srt` 时间戳字幕
- ✏️ **文本增强** — 去除语气词、自动标点、空格规整、用户词典
- 🔒 **100% 本地** — 无需联网，数据不出设备

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/allwithfly/TingKey.git
cd TingKey
```

### 2. 下载模型权重

选择**任一**来源：

<details>
<summary>📦 Hugging Face</summary>

```bash
pip install -U huggingface_hub
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir model
# 可选：更大模型
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir model-1.7b
```

</details>

<details>
<summary>📦 ModelScope 魔搭（国内推荐）</summary>

```bash
pip install -U modelscope
modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir model
# 可选：更大模型
modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir model-1.7b
```

</details>

### 3. 安装依赖

```bash
pip install -r requirements.txt
pip install PySide6  # GUI 需要
```

> **GPU 加速**：请安装与你 CUDA 版本匹配的 PyTorch 以获得最佳性能。

### 4. 启动 GUI

```bash
python desktop_gui_app.py --model-size 0.6b --device cuda:0
```

## 📖 使用指南

### GUI 桌面应用

```bash
python desktop_gui_app.py --model-size 0.6b --device cuda:0
```

| 参数 | 说明 |
|:---|:---|
| `--model-size` | `0.6b` 或 `1.7b` |
| `--model-dir` | 自定义模型目录路径 |
| `--device` | `auto`、`cpu` 或 `cuda:0` |
| `--ffmpeg-bin` | ffmpeg 可执行文件路径 |
| `--ffprobe-bin` | ffprobe 可执行文件路径 |

**GUI 能力：**
- 启动即预加载模型，识别无等待
- 按住快捷键录音，松开后识别并写入当前光标位置
- 支持音频/视频文件转写，支持通配符
- 实时进度与当前识别文本展示
- 可中断长时间任务
- 时间戳与 `.srt` 字幕导出
- 文本增强规则、用户词典、个性化设置

### 命令行 CLI

```bash
# 基础转写
python asr_cli.py --model-dir model --audio ./recording.wav

# 通配符批量转写
python asr_cli.py --model-size 0.6b --audio "*.wav"

# 视频转字幕
python asr_cli.py --model-size 0.6b --audio "*.mp4" --subtitle-dir ./subtitles

# 输出 JSON
python asr_cli.py --model-dir model --audio ./1.wav ./2.wav --output-format json --save ./result.json

# 低延迟模式
python asr_cli.py --model-size 0.6b --audio ./1.wav --device cuda:0 --dtype float16 --fast-mode
```

**注意事项：**
- `model_inference_time` 是模型识别耗时，不包含首次加载时间
- `audio_duration=N/A` 通常是 `ffprobe` 不可用或媒体元数据读取失败
- 请求时间戳但未提供 aligner 时，会自动降级为粗粒度区间

### 常驻服务 Shell

模型常驻内存，交互式转写：

```bash
# 默认启动即加载模型
python asr_service_shell.py --model-size 0.6b --device cuda:0 --dtype float16

# 仅进入 Shell（不自动加载）
python asr_service_shell.py --no-auto-start
```

交互命令：`start`、`stop`、`status`、`help`、`exit`，也可直接输入文件路径。

### 基准测试（0.6B vs 1.7B）

```bash
python benchmark_qwen_asr_models.py --inputs "path/to/recordings/*.wav"

# 预热 + 多次重复，结果更稳定
python benchmark_qwen_asr_models.py --inputs "*.wav" --warmup-runs 2 --repeat-per-file 3

# 附带 CER 准确率评估
python benchmark_qwen_asr_models.py --inputs "*.wav" --reference-file references.json
```

输出：`benchmark_report.json` 和 `benchmark_report.md`

### REST API 后端（可选）

```bash
python -m desktop_service.run_server --host 127.0.0.1 --port 8765
```

| 接口 | 方法 | 说明 |
|:---|:---|:---|
| `/health` | GET | 健康检查 |
| `/v1/config` | GET / PUT | 读取或替换配置 |
| `/v1/config/{section}` | PATCH | 部分配置更新 |
| `/v1/sessions/start` | POST | 创建新的识别会话 |
| `/v1/sessions/{id}/chunk` | POST | 发送音频片段 |
| `/v1/sessions/{id}/partial` | POST | 获取部分识别结果 |
| `/v1/sessions/{id}/stop` | POST | 停止会话并获取结果 |
| `/v1/sessions/{id}/events` | GET | SSE 事件流 |

## 🗂️ 项目结构

```
TingKey/
├── desktop_gui_app.py              # 桌面 GUI 应用（PySide6）
├── asr_cli.py                      # 命令行批处理工具
├── asr_service_shell.py            # 常驻交互式 Shell
├── benchmark_qwen_asr_models.py    # 0.6B vs 1.7B 基准测试
├── speech_output.py                # Qwen3-ASR 模型封装
├── asr_media_utils.py              # 音视频处理工具函数
├── desktop_service/                # 可选 REST API 后端
│   ├── api_server.py               # FastAPI 路由
│   ├── config_models.py            # 配置数据结构
│   ├── config_store.py             # 配置持久化
│   ├── final_asr.py                # ASR 集成
│   ├── models.py                   # 数据模型
│   ├── session_manager.py          # 会话生命周期管理
│   └── run_server.py               # 服务入口
├── requirements.txt                # Python 依赖
├── desktop_service_config.example.json  # 配置文件模板
├── model/                          # 0.6B 模型权重（不包含）
└── model-1.7b/                     # 1.7B 模型权重（不包含）
```

## ⚙️ FFmpeg

处理视频文件和探测媒体时长需要 FFmpeg。程序默认在 `./ffmpeg/bin/ffmpeg.exe` 位置查找。

**下载**：[FFmpeg 官方构建](https://ffmpeg.org/download.html)

也可通过参数显式指定路径：

```bash
python desktop_gui_app.py --ffmpeg-bin /path/to/ffmpeg --ffprobe-bin /path/to/ffprobe
```

## 💡 注意事项

- **Windows 优先**：GUI 使用 Windows 特定 API 实现快捷键监听和文本输入，暂不支持 Linux/macOS。
- **vLLM**：原生 Windows 不支持 vLLM。如需使用请在 WSL2 或 Linux 下运行。Windows 推荐 `transformers` + CUDA + `float16`。
- **配置文件**：首次使用请将 `desktop_service_config.example.json` 复制为 `desktop_service_config.json` 并按需修改。

## 🤝 参与贡献

欢迎贡献！请提交 Pull Request。

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交改动 (`git commit -m 'Add amazing feature'`)
4. 推送到远端 (`git push origin feature/amazing-feature`)
5. 发起 Pull Request

## 📄 许可证

本项目基于 MIT 许可证开源 — 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)（阿里通义千问团队）— 听键的核心语音识别模型
- [qwen-asr](https://pypi.org/project/qwen-asr/) — Qwen3-ASR Python SDK
- [FFmpeg](https://ffmpeg.org/) — 多媒体处理框架
