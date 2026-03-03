from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

from speech_output import QwenSpeechOutput


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


class FinalAsrRunner:
    def __init__(self, cfg: FinalAsrConfig) -> None:
        self.cfg = cfg
        self._engine: QwenSpeechOutput | None = None
        self._lock = threading.Lock()

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
