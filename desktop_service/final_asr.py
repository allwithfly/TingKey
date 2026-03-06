from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from speech_output import QwenSpeechOutput, _build_silent_warmup_audio


logger = logging.getLogger(__name__)

_RUNTIME_BUSY = "runtime busy"
_WARMUP_IN_PROGRESS = "warmup already in progress"


@dataclass
class FinalAsrConfig:
    enabled: bool = False
    model_dir: str = "model"
    model_size: str = "0.6b"
    device: str = "auto"
    backend: str = "transformers"
    dtype: str = "auto"
    max_new_tokens: int = 256
    max_inference_batch_size: int = 8
    attn_implementation: str = "auto"
    enable_tf32: bool = True
    cudnn_benchmark: bool = True


def warmup_engine_if_idle(
    engine: QwenSpeechOutput,
    *,
    language: str | None = None,
) -> tuple[bool, str | None]:
    runtime = getattr(engine, "_runtime", None)
    if runtime is None:
        warmed = bool(engine.warmup(language=language))
        return warmed, None if warmed else "warmup failed"

    if getattr(runtime, "warmup_succeeded", False):
        return True, None

    warmup_lock = getattr(runtime, "warmup_lock", None)
    infer_lock = getattr(runtime, "infer_lock", None)
    if warmup_lock is None or infer_lock is None:
        warmed = bool(engine.warmup(language=language))
        return warmed, getattr(runtime, "warmup_error", None)

    if not warmup_lock.acquire(blocking=False):
        return False, _WARMUP_IN_PROGRESS

    try:
        if getattr(runtime, "warmup_succeeded", False):
            return True, None
        if not infer_lock.acquire(blocking=False):
            return False, _RUNTIME_BUSY
        try:
            warmup_audio = _build_silent_warmup_audio()
            if warmup_audio is None:
                runtime.warmup_error = "warmup audio is unavailable"
                return False, runtime.warmup_error
            runtime.model.transcribe(
                audio=warmup_audio,
                language=language,
                return_time_stamps=False,
            )
        except Exception as exc:  # noqa: BLE001
            runtime.warmup_error = str(exc)
            return False, runtime.warmup_error
        finally:
            infer_lock.release()
        runtime.warmup_succeeded = True
        runtime.warmup_error = None
        return True, None
    finally:
        warmup_lock.release()


class FinalAsrRunner:
    def __init__(self, cfg: FinalAsrConfig) -> None:
        self.cfg = cfg
        self._engine: QwenSpeechOutput | None = None
        self._lock = threading.Lock()
        self._warmup_state_lock = threading.Lock()
        self._warmup_thread: threading.Thread | None = None

    def transcribe_file(self, wav_path: str | Path, language: str | None = None) -> dict:
        engine = self._get_engine()
        t0 = time.perf_counter()
        result = engine.transcribe(
            audio=str(Path(wav_path).resolve()),
            language=language,
            return_time_stamps=False,
        )[0]
        elapsed = time.perf_counter() - t0
        return {
            "text": result.get("text", ""),
            "language": result.get("language"),
            "model_inference_time_s": elapsed,
        }

    def schedule_warmup(self) -> bool:
        if not self.cfg.enabled:
            return False

        engine = self._get_engine()
        runtime = getattr(engine, "_runtime", None)
        if runtime is not None and getattr(runtime, "warmup_succeeded", False):
            return False

        with self._warmup_state_lock:
            if self._warmup_thread is not None and self._warmup_thread.is_alive():
                return False
            self._warmup_thread = threading.Thread(
                target=self._run_warmup,
                name="final-asr-warmup",
                daemon=True,
            )
            self._warmup_thread.start()
            return True

    def _run_warmup(self) -> None:
        try:
            engine = self._get_engine()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Final ASR warmup failed for %s: %s", self.cfg.model_dir, exc)
            return

        time.sleep(0.1)
        last_error: str | None = None
        for retry_delay_s in (0.0, 0.25, 0.5):
            if retry_delay_s:
                time.sleep(retry_delay_s)
            warmed, error = warmup_engine_if_idle(engine)
            if warmed:
                return
            last_error = error
            if error not in {_RUNTIME_BUSY, _WARMUP_IN_PROGRESS}:
                break

        if last_error:
            logger.warning("Final ASR warmup failed for %s: %s", self.cfg.model_dir, last_error)

    def _get_engine(self) -> QwenSpeechOutput:
        if self._engine is not None:
            return self._engine

        with self._lock:
            if self._engine is not None:
                return self._engine

            configured = self.cfg.model_dir.strip()
            configured_name = Path(configured).name if configured else ""
            if not configured or (
                self.cfg.model_size == "1.7b" and configured_name.lower() == "model"
            ):
                model_dir = str(
                    (Path.cwd() / ("model-1.7b" if self.cfg.model_size == "1.7b" else "model")).resolve()
                )
            else:
                model_dir = configured
                if not Path(model_dir).exists():
                    model_dir = str(
                        (Path.cwd() / ("model-1.7b" if self.cfg.model_size == "1.7b" else "model")).resolve()
                    )

            self._engine = QwenSpeechOutput(
                model_path=model_dir,
                device=self.cfg.device,
                dtype=self.cfg.dtype,  # type: ignore[arg-type]
                backend=self.cfg.backend,  # type: ignore[arg-type]
                max_inference_batch_size=self.cfg.max_inference_batch_size,
                max_new_tokens=self.cfg.max_new_tokens,
                attn_implementation=self.cfg.attn_implementation,  # type: ignore[arg-type]
                enable_tf32=self.cfg.enable_tf32,
                cudnn_benchmark=self.cfg.cudnn_benchmark,
            )
            return self._engine
