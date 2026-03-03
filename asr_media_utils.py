from __future__ import annotations

import glob
import hashlib
import subprocess
import unicodedata
import wave
from pathlib import Path
from typing import Any, Callable


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".flv",
    ".webm",
    ".m4v",
    ".ts",
    ".wmv",
    ".mpeg",
    ".mpg",
    ".3gp",
}

FORCED_ALIGNER_REQUIRED_FRAGMENT = (
    "return_time_stamps=True requires `forced_aligner`"
)


def normalize_cli_path(raw_path: str) -> str:
    cleaned = "".join(ch for ch in raw_path if unicodedata.category(ch) != "Cf")
    return cleaned.strip().strip('"').strip("'")


def contains_glob(pattern: str) -> bool:
    return any(char in pattern for char in "*?[]")


def expand_media_inputs(raw_inputs: list[str]) -> tuple[list[str], list[str]]:
    expanded: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()

    for raw in raw_inputs:
        token = normalize_cli_path(raw)
        if not token:
            continue

        path_like = Path(token).expanduser()
        pattern = str(path_like)

        if contains_glob(pattern):
            matches = [
                str(Path(p).resolve())
                for p in glob.glob(pattern, recursive=True)
                if Path(p).is_file()
            ]
            if not matches:
                missing.append(pattern)
                continue
            for match in matches:
                if match not in seen:
                    seen.add(match)
                    expanded.append(match)
            continue

        abs_path = str(path_like.resolve())
        if Path(abs_path).is_file():
            if abs_path not in seen:
                seen.add(abs_path)
                expanded.append(abs_path)
        else:
            missing.append(abs_path)

    return expanded, missing


def resolve_model_dir(model_dir_arg: str | None, model_size: str) -> Path:
    if model_dir_arg:
        return Path(normalize_cli_path(model_dir_arg)).expanduser().resolve()
    if model_size == "1.7b":
        return (Path.cwd() / "model-1.7b").resolve()
    return (Path.cwd() / "model").resolve()


def default_batch_size(model_size: str) -> int:
    return 2 if model_size == "1.7b" else 8


def is_video_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def extract_audio_with_ffmpeg(
    input_media: str | Path,
    output_wav: str | Path,
    ffmpeg_bin: str = "ffmpeg",
) -> None:
    candidates: list[str] = []
    for item in [ffmpeg_bin, "ffmpeg"]:
        token = str(item).strip().strip('"').strip("'")
        if token and token not in candidates:
            candidates.append(token)

    errors: list[str] = []
    for binary in candidates:
        cmd = [
            binary,
            "-y",
            "-i",
            str(input_media),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(output_wav),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except OSError as exc:
            errors.append(f"{binary}: not found or not executable ({exc})")
            continue

        if proc.returncode == 0:
            return

        error_text = (proc.stderr or proc.stdout or "").strip()
        if not error_text:
            unsigned = proc.returncode & 0xFFFFFFFF
            if unsigned == 0xC0000135:
                error_text = (
                    f"exit={proc.returncode} (0x{unsigned:08X}), ffmpeg failed to start. "
                    "Windows usually reports this when required DLL/runtime is missing. "
                    "Use a full ffmpeg build, not a single copied exe."
                )
            else:
                error_text = f"exit={proc.returncode} (0x{unsigned:08X})"
        errors.append(f"{binary}: {error_text[:600]}")

    raise RuntimeError(
        f"ffmpeg extract failed for {input_media}. "
        f"Tried {len(candidates)} command(s): " + " | ".join(errors)
    )


def prepare_audio_input(
    source_path: str | Path,
    temp_dir: str | Path,
    ffmpeg_bin: str = "ffmpeg",
) -> tuple[str, bool]:
    source = Path(source_path).resolve()
    if not is_video_file(source):
        return str(source), False

    temp_root = Path(temp_dir).resolve()
    temp_root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5(str(source).encode("utf-8")).hexdigest()[:10]
    out_wav = temp_root / f"{source.stem}_{digest}.wav"
    extract_audio_with_ffmpeg(source, out_wav, ffmpeg_bin=ffmpeg_bin)
    return str(out_wav), True


def get_media_duration_seconds(path: str | Path, ffprobe_bin: str = "ffprobe") -> float | None:
    path_obj = Path(path).expanduser().resolve()
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path_obj),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError:
        proc = None

    if proc is not None and proc.returncode == 0:
        text = (proc.stdout or "").strip()
        if text:
            try:
                return float(text)
            except ValueError:
                pass

    # Fallback for wav when ffprobe is unavailable.
    if path_obj.suffix.lower() == ".wav":
        try:
            with wave.open(str(path_obj), "rb") as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                if sample_rate > 0:
                    return frames / float(sample_rate)
        except wave.Error:
            pass

    # Optional fallback for other formats if torchaudio is available.
    try:
        import torchaudio

        info = torchaudio.info(str(path_obj))
        if info.sample_rate > 0:
            return info.num_frames / float(info.sample_rate)
    except Exception:
        pass

    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _segment_from_item(item: Any) -> dict[str, Any] | None:
    start: float | None = None
    end: float | None = None
    text: str | None = None

    if isinstance(item, dict):
        start = _to_float(
            item.get("start_time", item.get("start", item.get("begin", item.get("from"))))
        )
        end = _to_float(item.get("end_time", item.get("end", item.get("to"))))
        text_value = item.get("text", item.get("token", item.get("word", item.get("char"))))
        text = str(text_value) if text_value is not None else None
    elif isinstance(item, (list, tuple)):
        if len(item) >= 2:
            start = _to_float(item[0])
            end = _to_float(item[1])
        if len(item) >= 3 and item[2] is not None:
            text = str(item[2])
    else:
        start_attr = getattr(item, "start_time", None)
        end_attr = getattr(item, "end_time", None)
        text_attr = getattr(item, "text", None)
        start = _to_float(start_attr)
        end = _to_float(end_attr)
        text = str(text_attr) if text_attr is not None else None

    if start is None or end is None:
        return None
    if end < start:
        start, end = end, start
    return {"start_time": start, "end_time": end, "text": text}


def build_time_segments(
    time_stamps: Any,
    text: str,
    fallback_duration_s: float | None,
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    raw_items: list[Any] = []

    if time_stamps is None:
        raw_items = []
    elif isinstance(time_stamps, list):
        raw_items = time_stamps
    elif isinstance(time_stamps, tuple):
        raw_items = list(time_stamps)
    else:
        raw_items = [time_stamps]

    for item in raw_items:
        segment = _segment_from_item(item)
        if segment is not None:
            segments.append(segment)

    if not segments:
        if fallback_duration_s is not None:
            return [
                {
                    "start_time": 0.0,
                    "end_time": max(0.0, fallback_duration_s),
                    "text": text,
                }
            ]
        return []

    if len(segments) == 1 and not segments[0].get("text"):
        segments[0]["text"] = text
    return segments


def is_forced_aligner_required_error(exc: Exception) -> bool:
    return FORCED_ALIGNER_REQUIRED_FRAGMENT in str(exc)


def transcribe_with_timestamp_fallback(
    transcribe_once: Callable[[bool], list[dict[str, Any]]],
    request_timestamps: bool,
) -> tuple[dict[str, Any], bool]:
    try:
        return transcribe_once(request_timestamps)[0], request_timestamps
    except ValueError as exc:
        if not request_timestamps or not is_forced_aligner_required_error(exc):
            raise
    return transcribe_once(False)[0], False


def format_hms(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    sec = seconds - hours * 3600 - minutes * 60
    return f"{hours:02d}:{minutes:02d}:{sec:06.3f}"


def format_srt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    hours = ms // 3_600_000
    ms %= 3_600_000
    minutes = ms // 60_000
    ms %= 60_000
    secs = ms // 1000
    ms %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def build_srt_content(segments: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        start = format_srt_time(float(segment["start_time"]))
        end = format_srt_time(float(segment["end_time"]))
        text = str(segment.get("text") or "").strip()
        if not text:
            text = "..."
        lines.append(str(index))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def save_srt_file(segments: list[dict[str, Any]], output_path: str | Path) -> Path:
    content = build_srt_content(segments)
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target
