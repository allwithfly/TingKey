from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from desktop_service.final_asr import FinalAsrConfig, FinalAsrRunner


class _ImmediateThread:
    def __init__(self, *, target, name: str, daemon: bool) -> None:  # type: ignore[no-untyped-def]
        self._target = target
        self.name = name
        self.daemon = daemon
        self._alive = False

    def start(self) -> None:
        self._alive = True
        try:
            self._target()
        finally:
            self._alive = False

    def is_alive(self) -> bool:
        return self._alive


class _FakeEngine:
    def __init__(self, *, text: str = 'done') -> None:
        self.text = text
        self.transcribe_calls: list[dict[str, object]] = []

    def transcribe(self, audio: str, language: str | None = None, return_time_stamps: bool = False) -> list[dict[str, object]]:
        self.transcribe_calls.append(
            {
                'audio': audio,
                'language': language,
                'return_time_stamps': return_time_stamps,
            }
        )
        return [{'text': self.text, 'language': language}]


class FinalAsrWarmupTests(unittest.TestCase):
    def test_enabled_runner_starts_background_warmup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / 'model'
            model_dir.mkdir()
            engine = _FakeEngine()
            with mock.patch('desktop_service.final_asr.threading.Thread', _ImmediateThread),                 mock.patch('desktop_service.final_asr.time.sleep', return_value=None),                 mock.patch('desktop_service.final_asr.warmup_engine_if_idle', return_value=(True, None)) as warmup_call,                 mock.patch('desktop_service.final_asr.QwenSpeechOutput', return_value=engine):
                runner = FinalAsrRunner(
                    FinalAsrConfig(enabled=True, model_dir=str(model_dir), model_size='0.6b')
                )
                scheduled = runner.schedule_warmup()

        self.assertIsNotNone(runner)
        self.assertTrue(scheduled)
        warmup_call.assert_called_once_with(engine)

    def test_warmup_failure_does_not_block_transcribe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / 'model'
            model_dir.mkdir()
            engine = _FakeEngine(text='after-warmup')
            with mock.patch('desktop_service.final_asr.threading.Thread', _ImmediateThread),                 mock.patch('desktop_service.final_asr.time.sleep', return_value=None),                 mock.patch('desktop_service.final_asr.warmup_engine_if_idle', return_value=(False, 'warmup failed')) as warmup_call,                 mock.patch('desktop_service.final_asr.QwenSpeechOutput', return_value=engine):
                runner = FinalAsrRunner(
                    FinalAsrConfig(enabled=True, model_dir=str(model_dir), model_size='0.6b')
                )
                scheduled = runner.schedule_warmup()
                result = runner.transcribe_file('demo.wav', language='zh')

        self.assertTrue(scheduled)
        warmup_call.assert_called_once_with(engine)
        self.assertEqual(result['text'], 'after-warmup')
        self.assertEqual(result['language'], 'zh')
        self.assertEqual(engine.transcribe_calls[0]['language'], 'zh')
        self.assertFalse(engine.transcribe_calls[0]['return_time_stamps'])


if __name__ == '__main__':
    unittest.main()
