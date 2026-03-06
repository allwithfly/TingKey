from __future__ import annotations

import os
from pathlib import Path

from asr_media_utils import format_hms


def default_binary_path(
    exe_name: str,
    fallback: str,
    *,
    base_dir: str | Path,
    env_dir: str | None = None,
) -> str:
    candidates: list[Path] = []
    resolved_base_dir = Path(base_dir).expanduser().resolve()
    configured_env_dir = env_dir if env_dir is not None else os.environ.get("QWEN_ASR_FFMPEG_DIR", "").strip()
    if configured_env_dir:
        candidates.append(Path(configured_env_dir).expanduser())
    candidates.append(resolved_base_dir / "ffmpeg" / "bin")
    candidates.append(resolved_base_dir)
    for root in candidates:
        target = (root / exe_name).resolve()
        if target.exists():
            return str(target)
    return fallback


def validate_runtime_numeric_args(
    *,
    max_inference_batch_size: int | None,
    max_new_tokens: int,
    gpu_memory_utilization: float,
) -> None:
    if max_inference_batch_size is not None and max_inference_batch_size < 1:
        raise ValueError("max_inference_batch_size must be >= 1")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if not 0.0 < gpu_memory_utilization <= 1.0:
        raise ValueError("gpu_memory_utilization must be within (0, 1]")


def build_subtitle_path(subtitle_dir: str | Path, source_media: str | Path) -> Path:
    source = Path(source_media)
    return Path(subtitle_dir).expanduser().resolve() / f"{source.stem}.srt"


def format_transcription_result_lines(
    result: dict,
    *,
    header_label: str,
) -> list[str]:
    lines: list[str] = [f"[{header_label}] audio={result['audio']}"]
    if result.get("source_type") == "video":
        lines.append(f"source_type=video extracted_audio={result.get('prepared_audio')}")
    lines.append(f"language={result.get('language')}")
    lines.append(f"audio_duration={format_hms(result.get('audio_duration_s'))}")
    lines.append(f"model_inference_time={format_hms(result.get('model_inference_time_s'))}")
    for index, segment in enumerate(result.get("segments") or []):
        lines.append(
            f"segment[{index}] start={format_hms(segment.get('start_time'))} "
            f"end={format_hms(segment.get('end_time'))} text={segment.get('text', '')}"
        )
    if result.get("subtitle_path"):
        lines.append(f"subtitle={result['subtitle_path']}")
    lines.append(f"text={result.get('text', '')}")
    return lines
