from __future__ import annotations

import os
import platform
import warnings
from pathlib import Path
from typing import Any, Literal, Sequence

import torch


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


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def _resolve_dtype(
    dtype: Literal["auto", "float32", "float16", "bfloat16"],
    device: str,
) -> torch.dtype:
    if dtype == "auto":
        # Prefer fp16 on most consumer NVIDIA GPUs for best latency.
        return torch.float16 if device.startswith("cuda") else torch.float32

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype]


class QwenSpeechOutput:
    """Local Qwen3-ASR speech-to-text helper."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "auto",
        dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto",
        backend: Literal["transformers", "vllm"] = "transformers",
        gpu_memory_utilization: float = 0.7,
        attn_implementation: Literal["auto", "flash_attention_2", "sdpa", "eager"] = "auto",
        enable_tf32: bool = True,
        cudnn_benchmark: bool = True,
        max_inference_batch_size: int = 8,
        max_new_tokens: int = 256,
        forced_aligner_path: str | Path | None = None,
    ) -> None:
        model_dir = Path(model_path).expanduser().resolve()
        if not model_dir.exists():
            raise FileNotFoundError(f"Model path not found: {model_dir}")

        self.device = _resolve_device(device)
        self.dtype = _resolve_dtype(dtype, self.device)
        self.backend = backend

        if self.device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = bool(enable_tf32)
            torch.backends.cudnn.allow_tf32 = bool(enable_tf32)
            torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        Qwen3ASRModel = _load_qwen_asr_model()
        aligner_dir: Path | None = None
        if forced_aligner_path:
            aligner_dir = Path(forced_aligner_path).expanduser().resolve()
            if not aligner_dir.exists():
                raise FileNotFoundError(f"Forced aligner path not found: {aligner_dir}")

        if backend == "transformers":
            model_kwargs: dict[str, Any] = {
                "dtype": self.dtype,
                "device_map": self.device,
                "max_inference_batch_size": max_inference_batch_size,
                "max_new_tokens": max_new_tokens,
            }
            if attn_implementation != "auto":
                model_kwargs["attn_implementation"] = attn_implementation
            if aligner_dir:
                model_kwargs["forced_aligner"] = str(aligner_dir)
                model_kwargs["forced_aligner_kwargs"] = {
                    "dtype": self.dtype,
                    "device_map": self.device,
                }
                if attn_implementation != "auto":
                    model_kwargs["forced_aligner_kwargs"]["attn_implementation"] = (
                        attn_implementation
                    )
            try:
                self.model = Qwen3ASRModel.from_pretrained(
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
                    if "forced_aligner_kwargs" in model_kwargs:
                        model_kwargs["forced_aligner_kwargs"].pop("attn_implementation", None)
                    self.model = Qwen3ASRModel.from_pretrained(
                        str(model_dir),
                        **model_kwargs,
                    )
                else:
                    raise exc
            return

        if backend == "vllm":
            if platform.system().lower().startswith("win"):
                raise RuntimeError(
                    "vLLM backend is not supported natively on Windows. "
                    "Please run this in WSL2/Linux."
                )
            model_kwargs = {
                "model": str(model_dir),
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_inference_batch_size": max_inference_batch_size,
                "max_new_tokens": max_new_tokens,
            }
            if aligner_dir:
                model_kwargs["forced_aligner"] = str(aligner_dir)
                model_kwargs["forced_aligner_kwargs"] = {
                    "dtype": self.dtype,
                    "device_map": self.device,
                }
            self.model = Qwen3ASRModel.LLM(**model_kwargs)
            return

        raise ValueError(f"Unsupported backend: {backend}")

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
