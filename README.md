<p align="center">
  <h1 align="center">⚡ TingKey</h1>
  <p align="center">
    Free & Open-Source Alternative to <a href="https://www.typeless.ch/">Typeless</a> · 免费版 Typeless，开源版<a href="https://shandiansays.com/">闪电说</a>
    <br />
    Local desktop voice input powered by Qwen3-ASR
    <br />
    <a href="./README_zh.md">中文文档</a> · <a href="https://github.com/allwithfly/TingKey/issues">Report Bug</a> · <a href="https://github.com/allwithfly/TingKey/issues">Request Feature</a>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License" />
  <img src="https://img.shields.io/badge/python-3.10+-brightgreen.svg" alt="Python" />
  <img src="https://img.shields.io/badge/platform-Windows-lightgrey.svg" alt="Platform" />
  <img src="https://img.shields.io/badge/model-Qwen3--ASR-orange.svg" alt="Model" />
</p>

---

**TingKey（听键）** turns [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) into a practical, fully local speech-to-text desktop toolkit. Press a hotkey, speak, and the recognized text is typed directly at your cursor — no cloud, no latency, full privacy.

## ✨ Features

- 🎙️ **Press-and-hold hotkey recording** — speak, release, text appears at your cursor
- 🖥️ **Desktop GUI** with model preloading, live progress, cancel support
- 📁 **Batch CLI** for transcribing audio/video files with wildcard support
- 🔄 **Resident shell** — keep the model loaded, transcribe on demand
- 📊 **Benchmark tool** — compare 0.6B vs 1.7B model performance
- 🌐 **Optional REST API** backend for integrations
- 📝 **Subtitle export** (`.srt`) with timestamp support
- ✏️ **Text enhancement** — filler removal, punctuation, spacing normalization, user dictionary
- 🔒 **100% local** — no network required, your data stays on your device

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/allwithfly/TingKey.git
cd TingKey
```

### 2. Download model weights

> **Note**: Model weights are **not** included in this repository. You must download them separately.

Official model pages:
- Qwen3-ASR-0.6B: [Hugging Face](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) · [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-ASR-0.6B)
- Qwen3-ASR-1.7B: [Hugging Face](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) · [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B)

Choose **one** of the following download methods:

<details>
<summary>📦 Hugging Face (global)</summary>

```bash
pip install -U huggingface_hub

# Download 0.6B model (required)
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir model

# Optional: download 1.7B model for higher accuracy
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir model-1.7b
```

</details>

<details>
<summary>📦 ModelScope (recommended for China / 国内推荐)</summary>

```bash
pip install -U modelscope

# Download 0.6B model (required)
modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir model

# Optional: download 1.7B model for higher accuracy
modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir model-1.7b
```

</details>

After downloading, your project directory should look like:

```
TingKey/
├── model/          ← 0.6B weights (required)
├── model-1.7b/     ← 1.7B weights (optional)
├── desktop_gui_app.py
├── asr_cli.py
└── ...
```

> You can also place models anywhere and use `--model-dir <path>` to specify the location.

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install PySide6  # for the GUI app
```

> **GPU acceleration**: Install the PyTorch build matching your CUDA version for best performance.

### 4. Launch the GUI

```bash
python desktop_gui_app.py --model-size 0.6b --device cuda:0
```

## 📖 Usage

### GUI App

```bash
python desktop_gui_app.py --model-size 0.6b --device cuda:0
```

| Option | Description |
|:---|:---|
| `--model-size` | `0.6b` or `1.7b` |
| `--model-dir` | Custom model directory path |
| `--device` | `auto`, `cpu`, or `cuda:0` |
| `--ffmpeg-bin` | Path to ffmpeg binary |
| `--ffprobe-bin` | Path to ffprobe binary |

**GUI capabilities:**
- Preload model on startup for instant recognition
- Press-and-hold hotkey recording with direct text insertion at caret
- Audio/video file transcription with wildcard support
- Live progress and partial text display
- Cancel running jobs
- Timestamp and `.srt` subtitle export
- Text enhancement rules and user dictionary

### CLI

```bash
# Basic transcription
python asr_cli.py --model-dir model --audio ./recording.wav

# Batch with wildcards
python asr_cli.py --model-size 0.6b --audio "*.wav"

# Video to subtitles
python asr_cli.py --model-size 0.6b --audio "*.mp4" --subtitle-dir ./subtitles

# JSON output
python asr_cli.py --model-dir model --audio ./1.wav ./2.wav --output-format json --save ./result.json

# Low-latency mode
python asr_cli.py --model-size 0.6b --audio ./1.wav --device cuda:0 --dtype float16 --fast-mode
```

### Resident Shell

Keep the model loaded in memory and transcribe interactively:

```bash
# Start with auto-load
python asr_service_shell.py --model-size 0.6b --device cuda:0 --dtype float16

# Start without auto-load
python asr_service_shell.py --no-auto-start
```

Interactive commands: `start`, `stop`, `status`, `help`, `exit`, or directly type file paths.

### Benchmark (0.6B vs 1.7B)

```bash
python benchmark_qwen_asr_models.py --inputs "path/to/recordings/*.wav"

# With warmup and repetitions for stable results
python benchmark_qwen_asr_models.py --inputs "*.wav" --warmup-runs 2 --repeat-per-file 3

# With accuracy evaluation
python benchmark_qwen_asr_models.py --inputs "*.wav" --reference-file references.json
```

Outputs: `benchmark_report.json` and `benchmark_report.md`

### REST API (Optional)

```bash
python -m desktop_service.run_server --host 127.0.0.1 --port 8765
```

| Endpoint | Method | Description |
|:---|:---|:---|
| `/health` | GET | Health check |
| `/v1/config` | GET / PUT | Read or replace config |
| `/v1/config/{section}` | PATCH | Partial config update |
| `/v1/sessions/start` | POST | Start a new ASR session |
| `/v1/sessions/{id}/chunk` | POST | Send audio chunk |
| `/v1/sessions/{id}/partial` | POST | Get partial result |
| `/v1/sessions/{id}/stop` | POST | Stop session and get result |
| `/v1/sessions/{id}/events` | GET | SSE event stream |

## 🗂️ Project Structure

```
TingKey/
├── desktop_gui_app.py              # Desktop GUI application (PySide6)
├── asr_cli.py                      # Batch CLI tool
├── asr_service_shell.py            # Interactive resident shell
├── benchmark_qwen_asr_models.py    # 0.6B vs 1.7B benchmark
├── speech_output.py                # Qwen3-ASR model wrapper
├── asr_media_utils.py              # Audio/video processing utilities
├── desktop_service/                # Optional REST API backend
│   ├── api_server.py               # FastAPI routes
│   ├── config_models.py            # Config schema
│   ├── config_store.py             # Config persistence
│   ├── final_asr.py                # ASR integration
│   ├── models.py                   # Data models
│   ├── session_manager.py          # Session lifecycle
│   └── run_server.py               # Server entry point
├── requirements.txt                # Python dependencies
├── desktop_service_config.example.json  # Config template
├── model/                          # 0.6B model weights (not included)
└── model-1.7b/                     # 1.7B model weights (not included)
```

## ⚙️ FFmpeg (Optional)

FFmpeg is **not** included in this repository. It is optional but recommended.

**Download**: [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) (choose a "full build" or "essentials build")

**Installation**: Extract the download and place it so the binaries are at:

```
TingKey/
└── ffmpeg/
    └── bin/
        ├── ffmpeg.exe
        └── ffprobe.exe
```

The app automatically looks for `./ffmpeg/bin/ffmpeg.exe` and `./ffmpeg/bin/ffprobe.exe`. You can also specify custom paths:

```bash
python desktop_gui_app.py --ffmpeg-bin /path/to/ffmpeg --ffprobe-bin /path/to/ffprobe
```

**What happens without FFmpeg?**

| Feature | Without FFmpeg |
|:---|:---|
| `.wav` audio transcription | ✅ Works normally |
| Video file transcription (`.mp4`, `.mkv`, etc.) | ❌ Not available — audio cannot be extracted |
| Media duration display | ❌ Shows `N/A` — duration cannot be probed |
| `.srt` subtitle export | ⚠️ Partially affected — timestamps may be less accurate |

## 💡 Notes

- **Windows-first**: The GUI uses Windows-specific APIs for hotkey handling and text insertion. Linux/macOS support is not yet available.
- **vLLM**: Not supported on native Windows. Use WSL2 or Linux if you need the vLLM backend. On Windows, use `transformers` + CUDA + `float16`.
- **Config**: Copy `desktop_service_config.example.json` to `desktop_service_config.json` and edit as needed.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) by Alibaba Qwen Team — the speech recognition model powering TingKey
- [qwen-asr](https://pypi.org/project/qwen-asr/) — Python SDK for Qwen3-ASR
- [FFmpeg](https://ffmpeg.org/) — multimedia framework for audio/video processing
