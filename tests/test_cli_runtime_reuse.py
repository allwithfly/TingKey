from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import asr_cli
from asr_media_utils import PreparedMediaInput


class _FakeEngine:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def transcribe(
        self,
        audio: str | list[str],
        language: str | None = None,
        return_time_stamps: bool = False,
    ) -> list[dict[str, object]]:
        self.calls.append(
            {
                'audio': audio,
                'language': language,
                'return_time_stamps': return_time_stamps,
            }
        )
        if isinstance(audio, list):
            return [{'language': 'zh', 'text': f'local:{item}'} for item in audio]
        return [{'language': 'zh', 'text': f'local:{audio}', 'time_stamps': []}]


class CliRuntimeReuseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model_dir_obj = tempfile.TemporaryDirectory()
        self.addCleanup(self.model_dir_obj.cleanup)
        self.model_dir = Path(self.model_dir_obj.name)

    def _prepared_rows(self) -> list[PreparedMediaInput]:
        return [
            PreparedMediaInput(
                source_media='one.wav',
                prepared_audio='one.wav',
                source_type='audio',
                audio_duration_s=1.0,
            ),
            PreparedMediaInput(
                source_media='two.wav',
                prepared_audio='two.wav',
                source_type='audio',
                audio_duration_s=2.0,
            ),
        ]

    def test_cli_prefers_service_runtime_when_available(self) -> None:
        service_rows = [
            {'language': 'zh', 'text': 'svc:one.wav'},
            {'language': 'zh', 'text': 'svc:two.wav'},
        ]
        stdout = io.StringIO()
        with mock.patch.object(asr_cli, 'resolve_model_dir', return_value=self.model_dir),             mock.patch.object(asr_cli, 'expand_media_inputs', return_value=(['one.wav', 'two.wav'], [])),             mock.patch.object(asr_cli, 'prepare_media_inputs', return_value=self._prepared_rows()),             mock.patch.object(asr_cli, '_service_runtime_ready', return_value=True),             mock.patch.object(asr_cli, '_request_service_transcribe', return_value=service_rows) as service_call,             mock.patch.object(asr_cli, 'QwenSpeechOutput', side_effect=AssertionError('local engine should not be created')),             mock.patch('sys.argv', ['asr_cli.py', '--audio', 'one.wav', 'two.wav']),             mock.patch('sys.stdout', stdout):
            asr_cli.main()

        self.assertIn('svc:one.wav', stdout.getvalue())
        self.assertIn('svc:two.wav', stdout.getvalue())
        service_call.assert_called_once()

    def test_cli_falls_back_to_local_engine_when_service_unavailable(self) -> None:
        created_engines: list[_FakeEngine] = []

        def make_engine(*args, **kwargs):  # type: ignore[no-untyped-def]
            engine = _FakeEngine()
            created_engines.append(engine)
            return engine

        stdout = io.StringIO()
        with mock.patch.object(asr_cli, 'resolve_model_dir', return_value=self.model_dir),             mock.patch.object(asr_cli, 'expand_media_inputs', return_value=(['one.wav', 'two.wav'], [])),             mock.patch.object(asr_cli, 'prepare_media_inputs', return_value=self._prepared_rows()),             mock.patch.object(asr_cli, '_service_runtime_ready', return_value=False),             mock.patch.object(asr_cli, 'QwenSpeechOutput', side_effect=make_engine),             mock.patch('sys.argv', ['asr_cli.py', '--audio', 'one.wav', 'two.wav']),             mock.patch('sys.stdout', stdout):
            asr_cli.main()

        self.assertEqual(len(created_engines), 1)
        self.assertEqual(created_engines[0].calls[0]['audio'], ['one.wav', 'two.wav'])
        self.assertIn('local:one.wav', stdout.getvalue())

    def test_cli_falls_back_to_local_engine_when_service_call_fails(self) -> None:
        created_engines: list[_FakeEngine] = []

        def make_engine(*args, **kwargs):  # type: ignore[no-untyped-def]
            engine = _FakeEngine()
            created_engines.append(engine)
            return engine

        stdout = io.StringIO()
        with mock.patch.object(asr_cli, 'resolve_model_dir', return_value=self.model_dir),             mock.patch.object(asr_cli, 'expand_media_inputs', return_value=(['one.wav', 'two.wav'], [])),             mock.patch.object(asr_cli, 'prepare_media_inputs', return_value=self._prepared_rows()),             mock.patch.object(asr_cli, '_service_runtime_ready', return_value=True),             mock.patch.object(asr_cli, '_request_service_transcribe', return_value=None),             mock.patch.object(asr_cli, 'QwenSpeechOutput', side_effect=make_engine),             mock.patch('sys.argv', ['asr_cli.py', '--audio', 'one.wav', 'two.wav']),             mock.patch('sys.stdout', stdout):
            asr_cli.main()

        self.assertEqual(len(created_engines), 1)
        self.assertEqual(created_engines[0].calls[0]['audio'], ['one.wav', 'two.wav'])
        self.assertIn('local:one.wav', stdout.getvalue())


if __name__ == '__main__':
    unittest.main()
