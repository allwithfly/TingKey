from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from queue import Queue
from typing import Any
import wave


class SessionStatus(str, Enum):
    RECORDING = "recording"
    STOPPED = "stopped"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass
class SessionRecord:
    session_id: str
    created_at: str
    updated_at: str
    status: SessionStatus
    wav_path: Path
    sample_rate: int
    channels: int
    sample_width: int
    language: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_count: int = 0
    total_bytes: int = 0
    final_text: str | None = None
    final_model_time_s: float | None = None
    error_message: str | None = None
    wave_writer: wave.Wave_write | None = None
    event_history: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=200))
    subscribers: dict[str, Queue] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status.value,
            "wav_path": str(self.wav_path),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width": self.sample_width,
            "language": self.language,
            "metadata": self.metadata,
            "chunk_count": self.chunk_count,
            "total_bytes": self.total_bytes,
            "final_text": self.final_text,
            "final_model_time_s": self.final_model_time_s,
            "error_message": self.error_message,
        }

