from __future__ import annotations

import argparse
import gc
import shlex
import tempfile
import time
from pathlib import Path

import torch

from asr_frontend_common import (
    build_subtitle_path,
    default_binary_path,
    format_transcription_result_lines,
    validate_runtime_numeric_args,
)
from asr_media_utils import (
    batched_transcribe_rows,
    build_time_segments,
    default_batch_size,
    expand_media_inputs,
    format_hms,
    normalize_cli_path,
    prepare_media_inputs,
    resolve_model_dir,
    save_srt_file,
    transcribe_with_timestamp_fallback,
)
from speech_output import QwenSpeechOutput


def split_user_inputs(line: str) -> list[str]:
    try:
        parts = shlex.split(line, posix=False)
    except ValueError:
        parts = line.split()

    tokens: list[str] = []
    for part in parts:
        for token in part.split(","):
            value = normalize_cli_path(token)
            if value:
                tokens.append(value)
    return tokens


class ASRResidentShell:
    def __init__(
        self,
        model_dir_arg: str | None,
        model_size: str,
        aligner_dir: str | None,
        device: str,
        dtype: str,
        backend: str,
        gpu_memory_utilization: float,
        attn_implementation: str,
        enable_tf32: bool,
        cudnn_benchmark: bool,
        warmup_audio: str | None,
        max_inference_batch_size: int | None,
        max_new_tokens: int,
        language: str | None,
        return_time_stamps: bool,
        subtitle_dir: str | None,
        ffmpeg_bin: str,
        ffprobe_bin: str,
    ) -> None:
        self.default_model_dir_arg = model_dir_arg
        self.default_model_size = model_size
        self.aligner_dir = aligner_dir
        self.device = device
        self.dtype = dtype
        self.backend = backend
        self.gpu_memory_utilization = gpu_memory_utilization
        self.attn_implementation = attn_implementation
        self.enable_tf32 = enable_tf32
        self.cudnn_benchmark = cudnn_benchmark
        self.warmup_audio = warmup_audio
        self.max_inference_batch_size = max_inference_batch_size
        self.max_new_tokens = max_new_tokens
        self.language = language
        self.return_time_stamps = return_time_stamps
        self.subtitle_dir = subtitle_dir
        self.ffmpeg_bin = ffmpeg_bin
        self.ffprobe_bin = ffprobe_bin

        self.engine: QwenSpeechOutput | None = None
        self.running_model_dir: Path | None = None
        self.running_model_size: str | None = None
        self.temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
        self.timestamps_supported: bool | None = None

    def print_help(self) -> None:
        print("可用命令:")
        print("  启动 / start [0.6b|1.7b|模型目录]  -> 启动服务")
        print("  停止 / stop                         -> 停止服务并释放模型")
        print("  状态 / status                       -> 查看当前服务状态")
        print("  帮助 / help                         -> 查看帮助")
        print("  退出 / exit                         -> 退出程序")
        print("  直接输入音频/视频文件或通配符       -> 执行识别")
        print('    示例: 1.wav 2.wav')
        print('    示例: "*.wav"')
        print('    示例: "*.mp4"')

    def status(self) -> None:
        if self.engine is None:
            print("状态: 未启动")
            print(
                f"默认模型: size={self.default_model_size}, "
                f"dir={resolve_model_dir(self.default_model_dir_arg, self.default_model_size)}"
            )
            return
        print("状态: 运行中")
        print(f"模型目录: {self.running_model_dir}")
        print(f"模型规格: {self.running_model_size}")
        print(f"后端: {self.backend}")
        print(f"设备: {self.device}, dtype: {self.dtype}")
        print(f"时间戳: {'on' if self._need_timestamps() else 'off'}")
        print(f"字幕输出目录: {self.subtitle_dir or 'off'}")

    def _need_timestamps(self) -> bool:
        return self.return_time_stamps or bool(self.subtitle_dir)

    def start(self, hint: str | None = None) -> None:
        try:
            validate_runtime_numeric_args(
                max_inference_batch_size=self.max_inference_batch_size,
                max_new_tokens=self.max_new_tokens,
                gpu_memory_utilization=self.gpu_memory_utilization,
            )
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            return

        model_dir_arg = self.default_model_dir_arg
        model_size = self.default_model_size

        if hint:
            hint_clean = normalize_cli_path(hint).lower()
            if hint_clean in {"0.6b", "1.7b"}:
                model_size = hint_clean
                model_dir_arg = None
            else:
                model_dir_arg = hint

        model_dir = resolve_model_dir(model_dir_arg, model_size)
        if not model_dir.exists():
            print(f"[ERROR] 模型目录不存在: {model_dir}")
            return

        if self.engine is not None:
            print("检测到服务已启动，先执行重启...")
            self.stop()

        if self._need_timestamps() and not self.aligner_dir:
            print(
                "[WARN] 未指定 --aligner-dir，时间戳可能退化为整段音频区间。",
            )

        batch_size = (
            self.max_inference_batch_size
            if self.max_inference_batch_size is not None
            else default_batch_size(model_size)
        )

        print(f"正在启动服务，加载模型: {model_dir}")
        t0 = time.time()
        try:
            self.engine = QwenSpeechOutput(
                model_path=model_dir,
                device=self.device,
                dtype=self.dtype,
                backend=self.backend,  # type: ignore[arg-type]
                gpu_memory_utilization=self.gpu_memory_utilization,
                attn_implementation=self.attn_implementation,  # type: ignore[arg-type]
                enable_tf32=self.enable_tf32,
                cudnn_benchmark=self.cudnn_benchmark,
                max_inference_batch_size=batch_size,
                max_new_tokens=self.max_new_tokens,
                forced_aligner_path=self.aligner_dir,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] 服务启动失败: {exc}")
            self.engine = None
            return
        self.temp_dir_obj = tempfile.TemporaryDirectory(prefix="qwen3_asr_media_shell_")
        self.running_model_dir = model_dir
        self.running_model_size = model_size
        self.timestamps_supported = None
        if self.warmup_audio:
            try:
                warmup_paths, _ = expand_media_inputs([self.warmup_audio])
                if warmup_paths:
                    warmup_audio = warmup_paths[0]
                    _ = self.engine.transcribe(
                        audio=warmup_audio,
                        language=self.language,
                        return_time_stamps=False,
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] warmup failed: {exc}")
        print(f"服务启动完成，用时 {time.time() - t0:.2f}s")

    def stop(self) -> None:
        if self.engine is None:
            print("服务未启动")
            return

        self.engine = None
        self.running_model_dir = None
        self.running_model_size = None
        self.timestamps_supported = None

        if self.temp_dir_obj is not None:
            self.temp_dir_obj.cleanup()
            self.temp_dir_obj = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("服务已停止")

    def transcribe(self, raw_inputs: list[str]) -> None:
        if self.engine is None:
            print("Service is not started. Run `start` first.")
            return
        if self.temp_dir_obj is None:
            print("[ERROR] Temporary directory is unavailable. Restart the service.")
            return

        media_paths, missing = expand_media_inputs(raw_inputs)
        if missing:
            print(f"[WARN] Skipped unmatched inputs: {missing}")
        if not media_paths:
            print("[WARN] No valid input files found.")
            return

        need_timestamps = self._need_timestamps()
        print(f"Starting transcription for {len(media_paths)} file(s)...")

        prepared_rows = []
        for media_path in media_paths:
            try:
                prepared_rows.extend(
                    prepare_media_inputs(
                        [media_path],
                        temp_dir=self.temp_dir_obj.name,
                        ffmpeg_bin=self.ffmpeg_bin,
                        ffprobe_bin=self.ffprobe_bin,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] Failed to prepare media: {media_path}")
                print(f"detail: {exc}")

        if not prepared_rows:
            return

        batch_size = (
            self.max_inference_batch_size
            if self.max_inference_batch_size is not None
            else default_batch_size(self.running_model_size or self.default_model_size)
        )

        if not need_timestamps:
            try:
                normalized = batched_transcribe_rows(
                    prepared_rows,
                    batch_size=batch_size,
                    transcribe_batch=lambda batch_audio: self.engine.transcribe(
                        audio=batch_audio,
                        language=self.language,
                        return_time_stamps=False,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                print("[ERROR] Batch transcription failed.")
                print(f"detail: {exc}")
                return

            for index, item in enumerate(normalized, start=1):
                for line in format_transcription_result_lines(
                    item,
                    header_label=f"{index}/{len(normalized)}",
                ):
                    print(line)
            return

        for index, row in enumerate(prepared_rows, start=1):
            try:
                request_timestamps = need_timestamps and self.timestamps_supported is not False
                t0 = time.time()
                result, used_timestamps = transcribe_with_timestamp_fallback(
                    lambda return_ts: self.engine.transcribe(
                        audio=row.prepared_audio,
                        language=self.language,
                        return_time_stamps=return_ts,
                    ),
                    request_timestamps=request_timestamps,
                )
                model_inference_time_s = time.time() - t0
                if request_timestamps and not used_timestamps:
                    if self.timestamps_supported is not False:
                        print(
                            "[WARN] Timestamps disabled because forced_aligner is unavailable. Falling back to coarse segments."
                        )
                    self.timestamps_supported = False
                elif request_timestamps and used_timestamps:
                    self.timestamps_supported = True

                text = result.get("text", "")
                segments = build_time_segments(
                    result.get("time_stamps") if used_timestamps else [],
                    text=text,
                    fallback_duration_s=row.audio_duration_s,
                ) if need_timestamps else []

                subtitle_path: str | None = None
                if self.subtitle_dir:
                    target = build_subtitle_path(self.subtitle_dir, row.source_media)
                    save_srt_file(segments, target)
                    subtitle_path = str(target)

                shell_result = {
                    "audio": row.source_media,
                    "prepared_audio": row.prepared_audio,
                    "source_type": row.source_type,
                    "language": result.get("language"),
                    "audio_duration_s": row.audio_duration_s,
                    "model_inference_time_s": model_inference_time_s,
                    "segments": segments,
                    "subtitle_path": subtitle_path,
                    "text": text,
                }
                for line in format_transcription_result_lines(
                    shell_result,
                    header_label=f"{index}/{len(prepared_rows)}",
                ):
                    print(line)
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] Transcription failed: {row.source_media}")
                print(f"detail: {exc}")

    def run(self) -> None:
        print("Qwen3-ASR 常驻服务（对话模式）")
        print("输入 `帮助` 查看命令。")
        while True:
            try:
                line = input("asr> ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                self.stop()
                print("已退出")
                return

            if not line:
                continue

            tokens = split_user_inputs(line)
            if not tokens:
                continue

            command = tokens[0].lower()
            args = tokens[1:]

            if command in {"帮助", "help", "h", "?"}:
                self.print_help()
                continue
            if command in {"启动", "start"}:
                hint = args[0] if args else None
                self.start(hint=hint)
                continue
            if command in {"停止", "stop"}:
                self.stop()
                continue
            if command in {"状态", "status"}:
                self.status()
                continue
            if command in {"退出", "exit", "quit", "q"}:
                self.stop()
                print("已退出")
                return

            self.transcribe(tokens)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR resident shell with start/stop and wildcard media input",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help='Custom model directory. If omitted, it uses --model-size: "model" or "model-1.7b".',
    )
    parser.add_argument(
        "--model-size",
        default="0.6b",
        choices=["0.6b", "1.7b"],
        help="Model preset when --model-dir is not set",
    )
    parser.add_argument(
        "--aligner-dir",
        default=None,
        help="Optional forced aligner model directory for timestamps",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Runtime device: "auto", "cpu", "cuda:0"...',
    )
    parser.add_argument(
        "--backend",
        default="transformers",
        choices=["transformers", "vllm"],
        help="Inference backend",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.7,
        help="vLLM gpu memory utilization",
    )
    parser.add_argument(
        "--attn-implementation",
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        help="Attention implementation for transformers backend",
    )
    parser.add_argument(
        "--disable-tf32",
        action="store_true",
        help="Disable TF32 matmul/cudnn acceleration on CUDA",
    )
    parser.add_argument(
        "--disable-cudnn-benchmark",
        action="store_true",
        help="Disable cudnn benchmark autotune",
    )
    parser.add_argument(
        "--warmup-audio",
        default=None,
        help="Optional audio path/pattern used to warm up model after startup",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype",
    )
    parser.add_argument(
        "--language",
        default=None,
        help='Optional language hint, e.g. "Chinese" or "English"',
    )
    parser.add_argument(
        "--max-inference-batch-size",
        type=int,
        default=None,
        help="Batch size limit used by qwen-asr (default: 8 for 0.6B, 2 for 1.7B)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max generated tokens per sample",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Low-latency preset for short utterances (may reduce long-audio robustness)",
    )
    parser.add_argument(
        "--return-time-stamps",
        action="store_true",
        help="Return timestamps. Best used with --aligner-dir",
    )
    parser.add_argument(
        "--subtitle-dir",
        default=None,
        help="Optional output directory for generated .srt subtitle files",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default=default_binary_path("ffmpeg.exe", "ffmpeg", base_dir=Path(__file__).resolve().parent),
        help="ffmpeg executable path used for video audio extraction",
    )
    parser.add_argument(
        "--ffprobe-bin",
        default=default_binary_path("ffprobe.exe", "ffprobe", base_dir=Path(__file__).resolve().parent),
        help="ffprobe executable path used for duration probing",
    )
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Do not start the service automatically on launch",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.fast_mode:
        if args.dtype == "auto":
            args.dtype = "float16"
        if args.max_new_tokens == 256:
            args.max_new_tokens = 96
        if args.attn_implementation == "auto":
            args.attn_implementation = "flash_attention_2"

    try:
        validate_runtime_numeric_args(
            max_inference_batch_size=args.max_inference_batch_size,
            max_new_tokens=args.max_new_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    except ValueError as exc:
        parser.error(str(exc))

    shell = ASRResidentShell(
        model_dir_arg=args.model_dir,
        model_size=args.model_size,
        aligner_dir=normalize_cli_path(args.aligner_dir) if args.aligner_dir else None,
        device=args.device,
        dtype=args.dtype,
        backend=args.backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
        attn_implementation=args.attn_implementation,
        enable_tf32=not args.disable_tf32,
        cudnn_benchmark=not args.disable_cudnn_benchmark,
        warmup_audio=normalize_cli_path(args.warmup_audio) if args.warmup_audio else None,
        max_inference_batch_size=args.max_inference_batch_size,
        max_new_tokens=args.max_new_tokens,
        language=args.language,
        return_time_stamps=args.return_time_stamps,
        subtitle_dir=normalize_cli_path(args.subtitle_dir) if args.subtitle_dir else None,
        ffmpeg_bin=normalize_cli_path(args.ffmpeg_bin),
        ffprobe_bin=normalize_cli_path(args.ffprobe_bin),
    )
    if not args.no_auto_start:
        shell.start()
    shell.run()


if __name__ == "__main__":
    main()
