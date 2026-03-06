from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


def should_use_chunked_transcription(
    audio_duration_s: float | None,
    audio_path: str,
    threshold_s: float,
) -> bool:
    if audio_duration_s is None or audio_duration_s < threshold_s:
        return False
    return Path(audio_path).suffix.lower() == ".wav"


def requires_wav_conversion_for_chunking(
    audio_duration_s: float | None,
    audio_path: str,
    threshold_s: float,
) -> bool:
    if audio_duration_s is None or audio_duration_s < threshold_s:
        return False
    return Path(audio_path).suffix.lower() != ".wav"


def extract_partial_text(rows: Sequence[dict[str, Any]] | None) -> str:
    if not rows:
        return ""
    first_row = rows[0]
    text = first_row.get("text", "") if isinstance(first_row, dict) else getattr(first_row, "text", "")
    return str(text or "").strip()


def build_file_start_payload(*, index: int, total: int, source_media: str) -> dict[str, Any]:
    return {
        "event": "file_start",
        "index": index,
        "total": total,
        "source_media": source_media,
        "progress_percent": int(((index - 1) / float(max(total, 1))) * 100),
        "current_text": "",
    }


def build_file_done_payload(
    *,
    index: int,
    total: int,
    source_media: str,
    audio_path: str,
    was_extracted: bool,
    language: str | None,
    text: str,
    audio_duration_s: float | None,
    model_inference_time_s: float,
    segments: list[dict[str, Any]],
    subtitle_path: str | None,
) -> dict[str, Any]:
    return {
        "event": "file_done",
        "index": index,
        "total": total,
        "source_media": source_media,
        "audio_path": audio_path,
        "was_extracted": was_extracted,
        "language": language,
        "text": text,
        "audio_duration_s": audio_duration_s,
        "model_inference_time_s": model_inference_time_s,
        "segments": segments,
        "subtitle_path": subtitle_path,
        "progress_percent": int((index / float(max(total, 1))) * 100),
    }


def build_file_partial_payload(
    *,
    index: int,
    total: int,
    source_media: str,
    file_progress: float,
    global_progress: float,
    current_text: str,
    model_time_s: float,
) -> dict[str, Any]:
    return {
        "event": "file_partial",
        "index": index,
        "total": total,
        "source_media": source_media,
        "file_progress_percent": int(max(0.0, min(1.0, file_progress)) * 100),
        "progress_percent": int(max(0.0, min(1.0, global_progress)) * 100),
        "current_text": current_text,
        "model_inference_time_s": model_time_s,
    }


def build_subtitle_output_path(
    subtitle_dir: str | None,
    source_media: str,
    *,
    has_segments: bool,
) -> Path | None:
    if not subtitle_dir or not has_segments:
        return None
    return Path(subtitle_dir).expanduser().resolve() / f"{Path(source_media).stem}.srt"


def format_worker_failure(exc: Exception, traceback_text: str) -> str:
    return f"{exc}\n{traceback_text}"
