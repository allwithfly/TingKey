from __future__ import annotations

import threading
import uuid
from pathlib import Path
from queue import Full, Queue
from typing import Any
import wave

from desktop_service.models import SessionRecord, SessionStatus, utc_now_iso


class SessionManager:
    def __init__(self, audio_root: str | Path) -> None:
        self.audio_root = Path(audio_root).expanduser().resolve()
        self.audio_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._sessions: dict[str, SessionRecord] = {}

    def list_sessions(self) -> list[SessionRecord]:
        with self._lock:
            return list(self._sessions.values())

    def get_session(self, session_id: str) -> SessionRecord:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"session not found: {session_id}")
            return session

    def start_session(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
        language: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionRecord:
        with self._lock:
            session_id = uuid.uuid4().hex
            wav_path = self.audio_root / f"{session_id}.wav"
            writer = wave.open(str(wav_path), "wb")
            writer.setnchannels(channels)
            writer.setsampwidth(sample_width)
            writer.setframerate(sample_rate)

            now = utc_now_iso()
            session = SessionRecord(
                session_id=session_id,
                created_at=now,
                updated_at=now,
                status=SessionStatus.RECORDING,
                wav_path=wav_path,
                sample_rate=sample_rate,
                channels=channels,
                sample_width=sample_width,
                language=language,
                metadata=metadata or {},
                wave_writer=writer,
            )
            self._sessions[session_id] = session
            self._publish(
                session,
                {
                    "type": "session_started",
                    "session_id": session_id,
                    "wav_path": str(wav_path),
                },
            )
            return session

    def add_chunk(self, session_id: str, data: bytes) -> SessionRecord:
        if not data:
            raise ValueError("empty audio chunk")
        with self._lock:
            session = self.get_session(session_id)
            if session.status != SessionStatus.RECORDING:
                raise ValueError(f"session {session_id} is not recording")
            if session.wave_writer is None:
                raise RuntimeError(f"session {session_id} writer unavailable")

            session.wave_writer.writeframesraw(data)
            session.chunk_count += 1
            session.total_bytes += len(data)
            session.updated_at = utc_now_iso()
            self._publish(
                session,
                {
                    "type": "chunk_received",
                    "session_id": session_id,
                    "chunk_count": session.chunk_count,
                    "total_bytes": session.total_bytes,
                    "received_bytes": len(data),
                },
            )
            return session

    def stop_session(self, session_id: str) -> SessionRecord:
        with self._lock:
            session = self.get_session(session_id)
            if session.status == SessionStatus.RECORDING and session.wave_writer is not None:
                session.wave_writer.close()
                session.wave_writer = None
            session.status = SessionStatus.STOPPED
            session.updated_at = utc_now_iso()
            self._publish(
                session,
                {
                    "type": "session_stopped",
                    "session_id": session_id,
                    "wav_path": str(session.wav_path),
                    "chunk_count": session.chunk_count,
                    "total_bytes": session.total_bytes,
                },
            )
            return session

    def mark_processing(self, session_id: str) -> SessionRecord:
        with self._lock:
            session = self.get_session(session_id)
            session.status = SessionStatus.PROCESSING
            session.updated_at = utc_now_iso()
            self._publish(
                session,
                {"type": "session_processing", "session_id": session_id},
            )
            return session

    def mark_completed(
        self,
        session_id: str,
        final_text: str,
        final_model_time_s: float | None = None,
    ) -> SessionRecord:
        with self._lock:
            session = self.get_session(session_id)
            session.status = SessionStatus.COMPLETED
            session.final_text = final_text
            session.final_model_time_s = final_model_time_s
            session.updated_at = utc_now_iso()
            self._publish(
                session,
                {
                    "type": "final_result",
                    "session_id": session_id,
                    "final_text": final_text,
                    "final_model_time_s": final_model_time_s,
                },
            )
            return session

    def mark_error(self, session_id: str, message: str) -> SessionRecord:
        with self._lock:
            session = self.get_session(session_id)
            session.status = SessionStatus.ERROR
            session.error_message = message
            session.updated_at = utc_now_iso()
            self._publish(
                session,
                {
                    "type": "error",
                    "session_id": session_id,
                    "message": message,
                },
            )
            return session

    def publish_partial(self, session_id: str, text: str) -> SessionRecord:
        with self._lock:
            session = self.get_session(session_id)
            session.updated_at = utc_now_iso()
            self._publish(
                session,
                {
                    "type": "partial_result",
                    "session_id": session_id,
                    "text": text,
                },
            )
            return session

    def subscribe(
        self,
        session_id: str,
        replay_history: bool = True,
        queue_size: int = 256,
    ) -> tuple[str, Queue]:
        with self._lock:
            session = self.get_session(session_id)
            subscriber_id = uuid.uuid4().hex
            q: Queue = Queue(maxsize=queue_size)
            session.subscribers[subscriber_id] = q
            if replay_history:
                for event in session.event_history:
                    self._queue_put(q, event)
            return subscriber_id, q

    def unsubscribe(self, session_id: str, subscriber_id: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session.subscribers.pop(subscriber_id, None)

    def _publish(self, session: SessionRecord, event: dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("ts", utc_now_iso())
        session.event_history.append(payload)
        for q in session.subscribers.values():
            self._queue_put(q, payload)

    @staticmethod
    def _queue_put(q: Queue, payload: dict[str, Any]) -> None:
        try:
            q.put_nowait(payload)
        except Full:
            try:
                q.get_nowait()
            except Exception:
                pass
            try:
                q.put_nowait(payload)
            except Exception:
                pass

