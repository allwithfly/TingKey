from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

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
    prepare_media_inputs,
    resolve_model_dir,
    save_srt_file,
    transcribe_with_timestamp_fallback,
    normalize_cli_path,
)
from speech_output import QwenSpeechOutput


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
        default=default_binary_path("ffmpeg.exe", "ffmpeg", base_dir=Path(__file__).resolve().parent),
        help="ffmpeg executable path used for video audio extraction",
    )
    parser.add_argument(
        "--ffprobe-bin",
        default=default_binary_path("ffprobe.exe", "ffprobe", base_dir=Path(__file__).resolve().parent),
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
        lines.extend(format_transcription_result_lines(result, header_label=str(index)))
    return "\n".join(lines)


def _local_service_url() -> str:
    return os.environ.get("QWEN_ASR_SERVICE_URL", "http://127.0.0.1:8765").rstrip("/")


def _service_runtime_ready(service_url: str, timeout_s: float = 0.15) -> bool:
    try:
        with urllib_request.urlopen(f"{service_url}/health", timeout=timeout_s) as response:
            return int(getattr(response, "status", 200)) == 200
    except (urllib_error.URLError, TimeoutError, OSError, ValueError):
        return False


def _request_service_transcribe(
    *,
    service_url: str,
    audio: list[str],
    language: str | None,
    return_time_stamps: bool,
    model_dir: str | Path,
    device: str,
    backend: str,
    dtype: str,
    gpu_memory_utilization: float,
    attn_implementation: str,
    enable_tf32: bool,
    cudnn_benchmark: bool,
    max_inference_batch_size: int,
    max_new_tokens: int,
    aligner_dir: str | None,
    timeout_s: float = 300.0,
) -> list[dict] | None:
    payload = {
        "audio": audio,
        "language": language,
        "return_time_stamps": return_time_stamps,
        "model_dir": str(Path(model_dir).expanduser().resolve()),
        "device": device,
        "backend": backend,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "attn_implementation": attn_implementation,
        "enable_tf32": enable_tf32,
        "cudnn_benchmark": cudnn_benchmark,
        "max_inference_batch_size": max_inference_batch_size,
        "max_new_tokens": max_new_tokens,
        "aligner_dir": aligner_dir,
    }
    request = urllib_request.Request(
        f"{service_url}/v1/asr/transcribe-files",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
    except (urllib_error.HTTPError, urllib_error.URLError, TimeoutError, OSError, ValueError):
        return None

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


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

    aligner_dir = normalize_cli_path(args.aligner_dir) if args.aligner_dir else None
    service_url = _local_service_url()
    service_runtime_ready = _service_runtime_ready(service_url)
    engine: QwenSpeechOutput | None = None

    def get_engine() -> QwenSpeechOutput:
        nonlocal engine
        if engine is None:
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
                forced_aligner_path=aligner_dir,
            )
        return engine

    def transcribe_batch(batch_audio: list[str]) -> list[dict]:
        if service_runtime_ready:
            rows = _request_service_transcribe(
                service_url=service_url,
                audio=batch_audio,
                language=args.language,
                return_time_stamps=False,
                model_dir=model_dir,
                device=args.device,
                backend=args.backend,
                dtype=args.dtype,
                gpu_memory_utilization=args.gpu_memory_utilization,
                attn_implementation=args.attn_implementation,
                enable_tf32=not args.disable_tf32,
                cudnn_benchmark=not args.disable_cudnn_benchmark,
                max_inference_batch_size=max_inference_batch_size,
                max_new_tokens=args.max_new_tokens,
                aligner_dir=aligner_dir,
            )
            if rows is not None:
                return rows
        return get_engine().transcribe(
            audio=batch_audio,
            language=args.language,
            return_time_stamps=False,
        )

    def transcribe_one(audio_path: str, return_ts: bool) -> list[dict]:
        if service_runtime_ready:
            rows = _request_service_transcribe(
                service_url=service_url,
                audio=[audio_path],
                language=args.language,
                return_time_stamps=return_ts,
                model_dir=model_dir,
                device=args.device,
                backend=args.backend,
                dtype=args.dtype,
                gpu_memory_utilization=args.gpu_memory_utilization,
                attn_implementation=args.attn_implementation,
                enable_tf32=not args.disable_tf32,
                cudnn_benchmark=not args.disable_cudnn_benchmark,
                max_inference_batch_size=max_inference_batch_size,
                max_new_tokens=args.max_new_tokens,
                aligner_dir=aligner_dir,
            )
            if rows is not None:
                return rows
        return get_engine().transcribe(
            audio=audio_path,
            language=args.language,
            return_time_stamps=return_ts,
        )

    normalized: list[dict] = []
    timestamps_supported: bool | None = None
    with tempfile.TemporaryDirectory(prefix="qwen3_asr_media_") as temp_dir:
        prepared_rows = prepare_media_inputs(
            media_paths,
            temp_dir=temp_dir,
            ffmpeg_bin=args.ffmpeg_bin,
            ffprobe_bin=args.ffprobe_bin,
        )
        if not need_timestamps:
            normalized = batched_transcribe_rows(
                prepared_rows,
                batch_size=max_inference_batch_size,
                transcribe_batch=transcribe_batch,
            )
        else:
            for row in prepared_rows:
                request_timestamps = need_timestamps and timestamps_supported is not False
                start_time = time.perf_counter()
                result, used_timestamps = transcribe_with_timestamp_fallback(
                    lambda return_ts: transcribe_one(row.prepared_audio, return_ts),
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
                    fallback_duration_s=row.audio_duration_s,
                )

                subtitle_path: str | None = None
                if args.subtitle_dir:
                    subtitle_target = build_subtitle_path(args.subtitle_dir, row.source_media)
                    save_srt_file(segments, subtitle_target)
                    subtitle_path = str(subtitle_target)

                normalized.append(
                    {
                        "audio": row.source_media,
                        "prepared_audio": row.prepared_audio,
                        "source_type": row.source_type,
                        "language": result.get("language"),
                        "audio_duration_s": row.audio_duration_s,
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
