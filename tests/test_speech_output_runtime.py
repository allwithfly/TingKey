from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import speech_output


class _FakeLoadedModel:
    def __init__(self, recorder: dict[str, list[dict[str, object]]], load_kwargs: dict[str, object]) -> None:
        self._recorder = recorder
        self._load_kwargs = dict(load_kwargs)
        self.forced_aligner = load_kwargs.get("forced_aligner")

    def load_forced_aligner(self, forced_aligner: str, **kwargs: object) -> None:
        self.forced_aligner = SimpleNamespace(forced_aligner=forced_aligner, kwargs=dict(kwargs))
        self._recorder["aligner_loads"].append(
            {"forced_aligner": forced_aligner, "kwargs": dict(kwargs)}
        )

    def transcribe(
        self,
        audio: str | list[str],
        language: str | list[str] | None = None,
        return_time_stamps: bool = False,
    ) -> list[SimpleNamespace]:
        self._recorder["transcribe_calls"].append(
            {
                "audio": list(audio) if isinstance(audio, list) else audio,
                "language": language,
                "return_time_stamps": return_time_stamps,
                "aligner_loaded": self.forced_aligner is not None,
            }
        )
        audio_items = [audio] if isinstance(audio, str) else list(audio)
        if audio_items == ["warmup-fail.wav"]:
            raise RuntimeError("warmup boom")
        if return_time_stamps and self.forced_aligner is None:
            raise ValueError("return_time_stamps=True requires `forced_aligner`")
        time_stamps = (
            [{"start_time": 0.0, "end_time": 1.0, "text": "hello"}]
            if return_time_stamps
            else []
        )
        return [
            SimpleNamespace(language="en", text=f"text:{item}", time_stamps=time_stamps)
            for item in audio_items
        ]


class _FakeQwenFactory:
    def __init__(self) -> None:
        self.load_calls: list[dict[str, object]] = []
        self.aligner_loads: list[dict[str, object]] = []
        self.transcribe_calls: list[dict[str, object]] = []

    def build_model_class(self):
        factory = self
        recorder = {
            "aligner_loads": self.aligner_loads,
            "transcribe_calls": self.transcribe_calls,
        }

        class FakeQwen3ASRModel:
            @classmethod
            def from_pretrained(cls, model_path: str, **kwargs: object) -> _FakeLoadedModel:
                factory.load_calls.append(
                    {"model_path": model_path, "kwargs": dict(kwargs)}
                )
                return _FakeLoadedModel(recorder, dict(kwargs))

        return FakeQwen3ASRModel


class _FakeAlignerFactory:
    def __init__(self) -> None:
        self.load_calls: list[dict[str, object]] = []

    def build_aligner_class(self):
        factory = self

        class FakeQwen3ForcedAligner:
            @classmethod
            def from_pretrained(cls, aligner_path: str, **kwargs: object) -> SimpleNamespace:
                factory.load_calls.append(
                    {"aligner_path": aligner_path, "kwargs": dict(kwargs)}
                )
                return SimpleNamespace(aligner_path=aligner_path, kwargs=dict(kwargs))

        return FakeQwen3ForcedAligner


class SpeechOutputRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_dir = Path(self.temp_dir.name) / "model"
        self.model_dir.mkdir()
        self.aligner_dir = Path(self.temp_dir.name) / "aligner"
        self.aligner_dir.mkdir()
        registry = getattr(speech_output, "_RUNTIME_REGISTRY", None)
        if registry is not None:
            registry.clear()

    def tearDown(self) -> None:
        registry = getattr(speech_output, "_RUNTIME_REGISTRY", None)
        if registry is not None:
            registry.clear()
        self.temp_dir.cleanup()

    def _base_config(self, **overrides: object) -> dict[str, object]:
        config: dict[str, object] = {
            "model_path": self.model_dir,
            "device": "cpu",
            "dtype": "float32",
            "backend": "transformers",
            "gpu_memory_utilization": 0.7,
            "attn_implementation": "auto",
            "enable_tf32": True,
            "cudnn_benchmark": True,
            "max_inference_batch_size": 8,
            "max_new_tokens": 256,
            "forced_aligner_path": None,
        }
        config.update(overrides)
        return config

    def test_runtime_key_normalizes_equivalent_config(self) -> None:
        with mock.patch.object(speech_output.torch.cuda, "is_available", return_value=False):
            key_one = speech_output._build_runtime_key(
                **self._base_config(model_path=self.model_dir / ".", device="auto", dtype="auto")
            )
            key_two = speech_output._build_runtime_key(
                **self._base_config(model_path=str(self.model_dir.resolve()))
            )

        self.assertEqual(key_one, key_two)

    def test_same_config_reuses_one_runtime_in_process(self) -> None:
        fake_factory = _FakeQwenFactory()
        with mock.patch.object(speech_output, "_load_qwen_asr_model", return_value=fake_factory.build_model_class()):
            first = speech_output.QwenSpeechOutput(**self._base_config())
            second = speech_output.QwenSpeechOutput(**self._base_config())

        self.assertIs(first._runtime, second._runtime)

    def test_different_configs_do_not_share_runtime(self) -> None:
        fake_factory = _FakeQwenFactory()
        with mock.patch.object(speech_output, "_load_qwen_asr_model", return_value=fake_factory.build_model_class()):
            first = speech_output.QwenSpeechOutput(**self._base_config(max_new_tokens=128))
            second = speech_output.QwenSpeechOutput(**self._base_config(max_new_tokens=512))

        self.assertIsNot(first._runtime, second._runtime)

    def test_aligner_is_not_loaded_during_normal_transcription(self) -> None:
        fake_factory = _FakeQwenFactory()
        aligner_factory = _FakeAlignerFactory()
        with mock.patch.object(speech_output, "_load_qwen_asr_model", return_value=fake_factory.build_model_class()), \
            mock.patch.object(speech_output, "_load_qwen_forced_aligner", return_value=aligner_factory.build_aligner_class()):
            engine = speech_output.QwenSpeechOutput(
                **self._base_config(forced_aligner_path=self.aligner_dir)
            )
            result = engine.transcribe("plain.wav", return_time_stamps=False)

        self.assertEqual(result[0]["text"], "text:plain.wav")
        self.assertEqual(aligner_factory.load_calls, [])
        self.assertNotIn("forced_aligner", fake_factory.load_calls[0]["kwargs"])
        self.assertFalse(fake_factory.transcribe_calls[-1]["aligner_loaded"])

    def test_aligner_loads_only_when_timestamps_are_requested(self) -> None:
        fake_factory = _FakeQwenFactory()
        aligner_factory = _FakeAlignerFactory()
        with mock.patch.object(speech_output, "_load_qwen_asr_model", return_value=fake_factory.build_model_class()), \
            mock.patch.object(speech_output, "_load_qwen_forced_aligner", return_value=aligner_factory.build_aligner_class()):
            engine = speech_output.QwenSpeechOutput(
                **self._base_config(forced_aligner_path=self.aligner_dir)
            )
            engine.transcribe("plain.wav", return_time_stamps=False)
            result = engine.transcribe("timed.wav", return_time_stamps=True)

        self.assertEqual(result[0]["time_stamps"], [{"start_time": 0.0, "end_time": 1.0, "text": "hello"}])
        lazy_load_count = len(aligner_factory.load_calls) + len(fake_factory.aligner_loads)
        self.assertEqual(lazy_load_count, 1)
        loaded_path = (
            aligner_factory.load_calls[0]["aligner_path"]
            if aligner_factory.load_calls
            else fake_factory.aligner_loads[0]["forced_aligner"]
        )
        self.assertEqual(loaded_path, str(self.aligner_dir.resolve()))
        self.assertTrue(fake_factory.transcribe_calls[-1]["aligner_loaded"])

    def test_warmup_failure_does_not_break_inference_availability(self) -> None:
        fake_factory = _FakeQwenFactory()
        with mock.patch.object(speech_output, "_load_qwen_asr_model", return_value=fake_factory.build_model_class()):
            engine = speech_output.QwenSpeechOutput(**self._base_config())
            warmed = engine._runtime.warmup("warmup-fail.wav")
            result = engine.transcribe("after-warmup.wav", return_time_stamps=False)

        self.assertFalse(warmed)
        self.assertEqual(result[0]["text"], "text:after-warmup.wav")
        self.assertEqual(fake_factory.transcribe_calls[-1]["audio"], "after-warmup.wav")


if __name__ == "__main__":
    unittest.main()
