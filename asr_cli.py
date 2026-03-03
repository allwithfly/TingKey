from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

from asr_media_utils import (
    build_time_segments,
    default_batch_size,
    expand_media_inputs,
    format_hms,
    prepare_audio_input,
    resolve_model_dir,
    save_srt_file,
    transcribe_with_timestamp_fallback,
    get_media_duration_seconds,
    normalize_cli_path,
)
from speech_output import QwenSpeechOutput


def default_binary_path(exe_name: str, fallback: str) -> str:
    candidates: list[Path] = []
    env_dir = os.environ.get("QWEN_ASR_FFMPEG_DIR", "").strip()
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.append(Path(__file__).resolve().parent / "ffmpeg" / "bin")
    candidates.append(Path(__file__).resolve().parent)
    for root in candidates:
        target = (root / exe_name).resolve()
        if target.exists():
            return str(target)
    return fallback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR local speech-to-text CLI",
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
        "--audio",
        nargs="+",
        required=True,
        help='One or more audio/video paths or patterns, e.g. 1.wav "*.wav" "*.mp4"',
    )
    parser.add_argument(
        "--language",
        default=None,
        help='Optional language hint, e.g. "Chinese" or "English"',
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
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype",
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
        "--fast-mode",
        action="store_true",
        help="Low-latency preset for short utterances (may reduce long-audio robustness)",
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
        default=default_binary_path("ffmpeg.exe", "ffmpeg"),
        help="ffmpeg executable path used for video audio extraction",
    )
    parser.add_argument(
        "--ffprobe-bin",
        default=default_binary_path("ffprobe.exe", "ffprobe"),
        help="ffprobe executable path used for duration probing",
    )
    parser.add_argument(
        "--output-format",
        default="text",
        choices=["text", "json"],
        help="Output mode",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional output file path",
    )
    return parser


def format_segments_text(segments: list[dict]) -> list[str]:
    lines: list[str] = []
    for index, segment in enumerate(segments):
        lines.append(
            f"segment[{index}] start={format_hms(segment.get('start_time'))} "
            f"end={format_hms(segment.get('end_time'))} text={segment.get('text', '')}"
        )
    return lines


def to_text_lines(results: list[dict]) -> str:
    lines: list[str] = []
    for index, result in enumerate(results):
        lines.append(f"[{index}] audio={result['audio']}")
        if result.get("source_type") == "video":
            lines.append(f"source_type=video extracted_audio={result.get('prepared_audio')}")
        lines.append(f"language={result.get('language')}")
        lines.append(f"audio_duration={format_hms(result.get('audio_duration_s'))}")
        lines.append(
            f"model_inference_time={format_hms(result.get('model_inference_time_s'))}"
        )
        if result.get("segments"):
            lines.extend(format_segments_text(result["segments"]))
        if result.get("subtitle_path"):
            lines.append(f"subtitle={result['subtitle_path']}")
        lines.append(f"text={result.get('text', '')}")
    return "\n".join(lines)


def build_subtitle_path(subtitle_dir: str, source_media: str) -> Path:
    source = Path(source_media)
    return Path(subtitle_dir).expanduser().resolve() / f"{source.stem}.srt"


def main() -> None:
    args = build_parser().parse_args()

    if args.fast_mode:
        if args.dtype == "auto":
            args.dtype = "float16"
        if args.max_new_tokens == 256:
            args.max_new_tokens = 96
        if args.attn_implementation == "auto":
            args.attn_implementation = "flash_attention_2"

    model_dir = resolve_model_dir(args.model_dir, args.model_size)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model path not found: {model_dir}")

    media_paths, missing_media = expand_media_inputs(args.audio)
    if missing_media and not media_paths:
        raise FileNotFoundError(
            f"Media file not found: {missing_media[0]} "
            "(check for hidden characters or wildcard patterns)"
        )
    if missing_media:
        print(
            f"[WARN] Some inputs were not matched and were skipped: {missing_media}",
            file=sys.stderr,
        )

    max_inference_batch_size = args.max_inference_batch_size
    if max_inference_batch_size is None:
        max_inference_batch_size = default_batch_size(args.model_size)

    need_timestamps = args.return_time_stamps or bool(args.subtitle_dir)
    if need_timestamps and not args.aligner_dir:
        print(
            "[WARN] No --aligner-dir provided. Timestamps may fall back to a full-audio segment.",
            file=sys.stderr,
        )
    if args.backend == "vllm" and platform.system().lower().startswith("win"):
        print(
            "[ERROR] vLLM backend is not supported natively on Windows. Use WSL2/Linux.",
            file=sys.stderr,
        )
        return

    engine = QwenSpeechOutput(
        model_path=model_dir,
        device=args.device,
        dtype=args.dtype,
        backend=args.backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
        attn_implementation=args.attn_implementation,
        enable_tf32=not args.disable_tf32,
        cudnn_benchmark=not args.disable_cudnn_benchmark,
        max_inference_batch_size=max_inference_batch_size,
        max_new_tokens=args.max_new_tokens,
        forced_aligner_path=normalize_cli_path(args.aligner_dir) if args.aligner_dir else None,
    )

    normalized: list[dict] = []
    timestamps_supported: bool | None = None
    with tempfile.TemporaryDirectory(prefix="qwen3_asr_media_") as temp_dir:
        for media_path in media_paths:
            prepared_audio, was_extracted = prepare_audio_input(
                media_path,
                temp_dir=temp_dir,
                ffmpeg_bin=args.ffmpeg_bin,
            )
            audio_duration_s = get_media_duration_seconds(
                prepared_audio,
                ffprobe_bin=args.ffprobe_bin,
            )

            request_timestamps = need_timestamps and timestamps_supported is not False
            start_time = time.perf_counter()
            result, used_timestamps = transcribe_with_timestamp_fallback(
                lambda return_ts: engine.transcribe(
                    audio=prepared_audio,
                    language=args.language,
                    return_time_stamps=return_ts,
                ),
                request_timestamps=request_timestamps,
            )
            model_inference_time_s = time.perf_counter() - start_time
            if request_timestamps and not used_timestamps:
                if timestamps_supported is not False:
                    print(
                        "[WARN] Timestamps disabled because forced aligner is not initialized. "
                        "Falling back to coarse full-audio segments.",
                        file=sys.stderr,
                    )
                timestamps_supported = False
            elif request_timestamps and used_timestamps:
                timestamps_supported = True

            text = result.get("text", "")
            segments = build_time_segments(
                result.get("time_stamps") if used_timestamps else [],
                text=text,
                fallback_duration_s=audio_duration_s,
            ) if need_timestamps else []

            subtitle_path: str | None = None
            if args.subtitle_dir:
                subtitle_target = build_subtitle_path(args.subtitle_dir, media_path)
                save_srt_file(segments, subtitle_target)
                subtitle_path = str(subtitle_target)

            normalized.append(
                {
                    "audio": media_path,
                    "prepared_audio": prepared_audio,
                    "source_type": "video" if was_extracted else "audio",
                    "language": result.get("language"),
                    "audio_duration_s": audio_duration_s,
                    "model_inference_time_s": model_inference_time_s,
                    "text": text,
                    "segments": segments,
                    "subtitle_path": subtitle_path,
                }
            )

    if args.output_format == "json":
        output = json.dumps(normalized, ensure_ascii=False, indent=2)
    else:
        output = to_text_lines(normalized)

    print(output)

    if args.save:
        output_path = Path(normalize_cli_path(args.save)).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
