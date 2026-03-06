from __future__ import annotations

import os
import platform
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import torch


RuntimeDType = Literal["auto", "float32", "float16", "bfloat16"]
RuntimeBackend = Literal["transformers", "vllm"]
RuntimeAttention = Literal["auto", "flash_attention_2", "sdpa", "eager"]
RuntimeKey = tuple[Any, ...]


_RUNTIME_REGISTRY: dict[RuntimeKey, "_SharedRuntime"] = {}
_RUNTIME_REGISTRY_LOCK = threading.Lock()


def _load_qwen_asr_model():
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:
        pass

    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: qwen-asr. Install with `pip install -U qwen-asr`."
        ) from exc
    return Qwen3ASRModel


def _load_qwen_forced_aligner():
    try:
        from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: qwen-asr forced aligner. Install with `pip install -U qwen-asr`."
        ) from exc
    return Qwen3ForcedAligner


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def _resolve_dtype(
    dtype: RuntimeDType,
    device: str,
) -> torch.dtype:
    if dtype == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype]


def _resolve_model_path(model_path: str | Path) -> Path:
    return Path(model_path).expanduser().resolve()


def _resolve_aligner_path(forced_aligner_path: str | Path | None) -> Path | None:
    if not forced_aligner_path:
        return None
    return Path(forced_aligner_path).expanduser().resolve()


def _build_runtime_key(
    model_path: str | Path,
    device: str = "auto",
    dtype: RuntimeDType = "auto",
    backend: RuntimeBackend = "transformers",
    gpu_memory_utilization: float = 0.7,
    attn_implementation: RuntimeAttention = "auto",
    enable_tf32: bool = True,
    cudnn_benchmark: bool = True,
    max_inference_batch_size: int = 8,
    max_new_tokens: int = 256,
    forced_aligner_path: str | Path | None = None,
) -> RuntimeKey:
    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype(dtype, resolved_device)
    model_dir = _resolve_model_path(model_path)
    aligner_dir = _resolve_aligner_path(forced_aligner_path)
    return (
        str(model_dir),
        backend,
        resolved_device,
        str(resolved_dtype),
        float(gpu_memory_utilization),
        attn_implementation,
        bool(enable_tf32),
        bool(cudnn_benchmark),
        int(max_inference_batch_size),
        int(max_new_tokens),
        str(aligner_dir) if aligner_dir is not None else None,
    )


def _apply_cuda_runtime_settings(
    device: str,
    *,
    enable_tf32: bool,
    cudnn_benchmark: bool,
) -> None:
    if not device.startswith("cuda"):
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(enable_tf32)
    torch.backends.cudnn.allow_tf32 = bool(enable_tf32)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _build_aligner_kwargs(
    *,
    dtype: torch.dtype,
    device: str,
    attn_implementation: RuntimeAttention,
) -> dict[str, Any]:
    aligner_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": device,
    }
    if attn_implementation != "auto":
        aligner_kwargs["attn_implementation"] = attn_implementation
    return aligner_kwargs


def _build_silent_warmup_audio() -> tuple[Any, int] | None:
    try:
        import numpy as np
    except Exception:
        return None
    return (np.zeros(1600, dtype=np.float32), 16000)


@dataclass
class _SharedRuntime:
    model: Any
    key: RuntimeKey
    aligner_path: str | None
    aligner_kwargs: dict[str, Any]
    infer_lock: threading.RLock = field(default_factory=threading.RLock)
    aligner_lock: threading.Lock = field(default_factory=threading.Lock)
    warmup_lock: threading.Lock = field(default_factory=threading.Lock)
    aligner_loaded: bool = False
    warmup_succeeded: bool = False
    warmup_error: str | None = None

    def ensure_aligner(self) -> bool:
        if not self.aligner_path:
            return False
        if self.aligner_loaded and getattr(self.model, "forced_aligner", None) is not None:
            return True

        with self.aligner_lock:
            if self.aligner_loaded and getattr(self.model, "forced_aligner", None) is not None:
                return True

            if hasattr(self.model, "load_forced_aligner"):
                self.model.load_forced_aligner(self.aligner_path, **self.aligner_kwargs)
            else:
                Qwen3ForcedAligner = _load_qwen_forced_aligner()
                self.model.forced_aligner = Qwen3ForcedAligner.from_pretrained(
                    self.aligner_path,
                    **self.aligner_kwargs,
                )
            self.aligner_loaded = True
            return True

    def warmup(
        self,
        audio: str | tuple[Any, int] | None = None,
        *,
        language: str | None = None,
    ) -> bool:
        warmup_audio = audio if audio is not None else _build_silent_warmup_audio()
        if warmup_audio is None:
            self.warmup_error = "warmup audio is unavailable"
            return False

        with self.warmup_lock:
            if self.warmup_succeeded:
                return True
            try:
                with self.infer_lock:
                    self.model.transcribe(
                        audio=warmup_audio,
                        language=language,
                        return_time_stamps=False,
                    )
            except Exception as exc:
                self.warmup_error = str(exc)
                return False
            self.warmup_succeeded = True
            self.warmup_error = None
            return True


def _create_runtime(
    *,
    key: RuntimeKey,
    model_dir: Path,
    device: str,
    dtype: torch.dtype,
    backend: RuntimeBackend,
    gpu_memory_utilization: float,
    attn_implementation: RuntimeAttention,
    enable_tf32: bool,
    cudnn_benchmark: bool,
    max_inference_batch_size: int,
    max_new_tokens: int,
    aligner_path: Path | None,
) -> _SharedRuntime:
    _apply_cuda_runtime_settings(
        device,
        enable_tf32=enable_tf32,
        cudnn_benchmark=cudnn_benchmark,
    )

    Qwen3ASRModel = _load_qwen_asr_model()
    aligner_kwargs = _build_aligner_kwargs(
        dtype=dtype,
        device=device,
        attn_implementation=attn_implementation,
    )

    if backend == "transformers":
        model_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "device_map": device,
            "max_inference_batch_size": max_inference_batch_size,
            "max_new_tokens": max_new_tokens,
        }
        if attn_implementation != "auto":
            model_kwargs["attn_implementation"] = attn_implementation
        try:
            model = Qwen3ASRModel.from_pretrained(
                str(model_dir),
                **model_kwargs,
            )
        except Exception as exc:
            if attn_implementation == "flash_attention_2":
                warnings.warn(
                    "flash_attention_2 init failed, fallback to default attention.",
                    RuntimeWarning,
                )
                model_kwargs.pop("attn_implementation", None)
                model = Qwen3ASRModel.from_pretrained(
                    str(model_dir),
                    **model_kwargs,
                )
            else:
                raise exc
        return _SharedRuntime(
            model=model,
            key=key,
            aligner_path=str(aligner_path) if aligner_path is not None else None,
            aligner_kwargs=aligner_kwargs,
        )

    if backend == "vllm":
        if platform.system().lower().startswith("win"):
            raise RuntimeError(
                "vLLM backend is not supported natively on Windows. "
                "Please run this in WSL2/Linux."
            )
        model = Qwen3ASRModel.LLM(
            model=str(model_dir),
            gpu_memory_utilization=gpu_memory_utilization,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
        )
        return _SharedRuntime(
            model=model,
            key=key,
            aligner_path=str(aligner_path) if aligner_path is not None else None,
            aligner_kwargs=aligner_kwargs,
        )

    raise ValueError(f"Unsupported backend: {backend}")


def _get_or_create_runtime(
    *,
    model_path: str | Path,
    device: str,
    dtype: RuntimeDType,
    backend: RuntimeBackend,
    gpu_memory_utilization: float,
    attn_implementation: RuntimeAttention,
    enable_tf32: bool,
    cudnn_benchmark: bool,
    max_inference_batch_size: int,
    max_new_tokens: int,
    forced_aligner_path: str | Path | None,
) -> _SharedRuntime:
    model_dir = _resolve_model_path(model_path)
    aligner_dir = _resolve_aligner_path(forced_aligner_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model path not found: {model_dir}")
    if aligner_dir is not None and not aligner_dir.exists():
        raise FileNotFoundError(f"Forced aligner path not found: {aligner_dir}")

    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype(dtype, resolved_device)
    key = _build_runtime_key(
        model_path=model_dir,
        device=resolved_device,
        dtype=dtype,
        backend=backend,
        gpu_memory_utilization=gpu_memory_utilization,
        attn_implementation=attn_implementation,
        enable_tf32=enable_tf32,
        cudnn_benchmark=cudnn_benchmark,
        max_inference_batch_size=max_inference_batch_size,
        max_new_tokens=max_new_tokens,
        forced_aligner_path=aligner_dir,
    )

    with _RUNTIME_REGISTRY_LOCK:
        runtime = _RUNTIME_REGISTRY.get(key)
        if runtime is None:
            runtime = _create_runtime(
                key=key,
                model_dir=model_dir,
                device=resolved_device,
                dtype=resolved_dtype,
                backend=backend,
                gpu_memory_utilization=gpu_memory_utilization,
                attn_implementation=attn_implementation,
                enable_tf32=enable_tf32,
                cudnn_benchmark=cudnn_benchmark,
                max_inference_batch_size=max_inference_batch_size,
                max_new_tokens=max_new_tokens,
                aligner_path=aligner_dir,
            )
            _RUNTIME_REGISTRY[key] = runtime
        return runtime


class QwenSpeechOutput:
    """Local Qwen3-ASR speech-to-text helper."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "auto",
        dtype: RuntimeDType = "auto",
        backend: RuntimeBackend = "transformers",
        gpu_memory_utilization: float = 0.7,
        attn_implementation: RuntimeAttention = "auto",
        enable_tf32: bool = True,
        cudnn_benchmark: bool = True,
        max_inference_batch_size: int = 8,
        max_new_tokens: int = 256,
        forced_aligner_path: str | Path | None = None,
    ) -> None:
        self.device = _resolve_device(device)
        self.dtype = _resolve_dtype(dtype, self.device)
        self.backend = backend
        self._runtime = _get_or_create_runtime(
            model_path=model_path,
            device=device,
            dtype=dtype,
            backend=backend,
            gpu_memory_utilization=gpu_memory_utilization,
            attn_implementation=attn_implementation,
            enable_tf32=enable_tf32,
            cudnn_benchmark=cudnn_benchmark,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
            forced_aligner_path=forced_aligner_path,
        )
        self.model = self._runtime.model

    def warmup(
        self,
        audio: str | tuple[Any, int] | None = None,
        *,
        language: str | None = None,
    ) -> bool:
        return self._runtime.warmup(audio=audio, language=language)

    def transcribe(
        self,
        audio: str | Sequence[str],
        language: str | Sequence[str] | None = None,
        return_time_stamps: bool = False,
    ) -> list[dict[str, Any]]:
        """Transcribe one or many audio files."""

        if isinstance(audio, str):
            audio_input: str | list[str] = audio
        else:
            audio_input = list(audio)

        if return_time_stamps:
            self._runtime.ensure_aligner()

        with self._runtime.infer_lock:
            results = self.model.transcribe(
                audio=audio_input,
                language=language,
                return_time_stamps=return_time_stamps,
            )

        output: list[dict[str, Any]] = []
        for item in results:
            row: dict[str, Any] = {
                "language": getattr(item, "language", None),
                "text": getattr(item, "text", ""),
            }
            if return_time_stamps:
                row["time_stamps"] = getattr(item, "time_stamps", [])
            output.append(row)
        return output


def speech_output_method(
    audio_path: str,
    model_path: str | Path,
    language: str | None = None,
    device: str = "auto",
) -> str:
    """Single-file shortcut that returns plain text."""

    engine = QwenSpeechOutput(
        model_path=model_path,
        device=device,
        dtype="auto",
    )
    return engine.transcribe(audio=audio_path, language=language)[0]["text"]
