from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import asr_cli
import asr_service_shell
from asr_media_utils import PreparedMediaInput, batched_transcribe_rows, prepare_audio_input


class BatchedTranscribeRowsTests(unittest.TestCase):
    def test_preserves_result_order_across_batches(self) -> None:
        rows = [
            PreparedMediaInput(
                source_media=f"source-{index}.wav",
                prepared_audio=f"prepared-{index}.wav",
                source_type="audio",
                audio_duration_s=1.5 + index,
            )
            for index in range(3)
        ]
        seen_batches: list[list[str]] = []

        def fake_transcribe(batch_audio: list[str]) -> list[dict[str, object]]:
            seen_batches.append(list(batch_audio))
            return [
                {"language": "zh", "text": f"text:{audio_path}"}
                for audio_path in batch_audio
            ]

        output = batched_transcribe_rows(rows, batch_size=2, transcribe_batch=fake_transcribe)

        self.assertEqual(
            seen_batches,
            [["prepared-0.wav", "prepared-1.wav"], ["prepared-2.wav"]],
        )
        self.assertEqual([item["audio"] for item in output], [f"source-{i}.wav" for i in range(3)])
        self.assertEqual(
            [item["text"] for item in output],
            [f"text:prepared-{i}.wav" for i in range(3)],
        )
        self.assertTrue(all(item["model_inference_time_s"] >= 0 for item in output))

    def test_raises_when_batch_output_count_mismatches(self) -> None:
        rows = [
            PreparedMediaInput(
                source_media="one.wav",
                prepared_audio="one.wav",
                source_type="audio",
                audio_duration_s=1.0,
            ),
            PreparedMediaInput(
                source_media="two.wav",
                prepared_audio="two.wav",
                source_type="audio",
                audio_duration_s=2.0,
            ),
        ]

        with self.assertRaisesRegex(ValueError, "returned 1 result"):
            batched_transcribe_rows(
                rows,
                batch_size=2,
                transcribe_batch=lambda _batch: [{"language": "en", "text": "only-one"}],
            )


class PrepareAudioInputCacheTests(unittest.TestCase):
    def test_reuses_cached_extraction_when_output_is_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "clip.mp4"
            source.write_bytes(b"video")

            def fake_extract(input_media: str | Path, output_wav: str | Path, ffmpeg_bin: str = "ffmpeg") -> None:
                Path(output_wav).write_bytes(b"wav")

            with mock.patch("asr_media_utils.extract_audio_with_ffmpeg", side_effect=fake_extract) as patched_extract:
                first_output, first_extracted = prepare_audio_input(source, temp_dir=temp_dir, ffmpeg_bin="ffmpeg")
                second_output, second_extracted = prepare_audio_input(source, temp_dir=temp_dir, ffmpeg_bin="ffmpeg")

            self.assertTrue(first_extracted)
            self.assertTrue(second_extracted)
            self.assertEqual(first_output, second_output)
            self.assertEqual(patched_extract.call_count, 1)

    def test_reextracts_when_source_is_newer_than_cached_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "clip.mp4"
            source.write_bytes(b"video")
            output_path: Path | None = None

            def fake_extract(input_media: str | Path, output_wav: str | Path, ffmpeg_bin: str = "ffmpeg") -> None:
                Path(output_wav).write_bytes(b"wav")

            with mock.patch("asr_media_utils.extract_audio_with_ffmpeg", side_effect=fake_extract):
                first_output, _ = prepare_audio_input(source, temp_dir=temp_dir, ffmpeg_bin="ffmpeg")
                output_path = Path(first_output)

            old_mtime = source.stat().st_mtime - 100
            os.utime(output_path, (old_mtime, old_mtime))
            new_mtime = source.stat().st_mtime + 100
            os.utime(source, (new_mtime, new_mtime))

            with mock.patch("asr_media_utils.extract_audio_with_ffmpeg", side_effect=fake_extract) as patched_extract:
                second_output, _ = prepare_audio_input(source, temp_dir=temp_dir, ffmpeg_bin="ffmpeg")

            self.assertEqual(str(output_path), second_output)
            self.assertEqual(patched_extract.call_count, 1)


class AsrCliBatchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model_dir_obj = tempfile.TemporaryDirectory()
        self.addCleanup(self.model_dir_obj.cleanup)
        self.model_dir = Path(self.model_dir_obj.name)

    def _prepared_rows(self) -> list[PreparedMediaInput]:
        return [
            PreparedMediaInput(
                source_media="one.wav",
                prepared_audio="one.wav",
                source_type="audio",
                audio_duration_s=1.0,
            ),
            PreparedMediaInput(
                source_media="two.wav",
                prepared_audio="two.wav",
                source_type="audio",
                audio_duration_s=2.0,
            ),
        ]

    def test_main_batches_non_timestamp_requests(self) -> None:
        created_engines: list[FakeEngine] = []

        def make_engine(*args, **kwargs):  # type: ignore[no-untyped-def]
            engine = FakeEngine()
            created_engines.append(engine)
            return engine

        stdout = io.StringIO()
        with mock.patch.object(asr_cli, "resolve_model_dir", return_value=self.model_dir), \
            mock.patch.object(asr_cli, "expand_media_inputs", return_value=(["one.wav", "two.wav"], [])), \
            mock.patch.object(asr_cli, "prepare_media_inputs", return_value=self._prepared_rows()), \
            mock.patch.object(asr_cli, "QwenSpeechOutput", side_effect=make_engine), \
            mock.patch("sys.argv", ["asr_cli.py", "--audio", "one.wav", "two.wav"]), \
            mock.patch("sys.stdout", stdout):
            asr_cli.main()

        self.assertEqual(len(created_engines), 1)
        self.assertEqual(len(created_engines[0].calls), 1)
        self.assertEqual(created_engines[0].calls[0]["audio"], ["one.wav", "two.wav"])
        self.assertFalse(created_engines[0].calls[0]["return_time_stamps"])

    def test_main_keeps_per_file_calls_when_timestamps_requested(self) -> None:
        created_engines: list[FakeEngine] = []

        def make_engine(*args, **kwargs):  # type: ignore[no-untyped-def]
            engine = FakeEngine()
            created_engines.append(engine)
            return engine

        stdout = io.StringIO()
        with mock.patch.object(asr_cli, "resolve_model_dir", return_value=self.model_dir), \
            mock.patch.object(asr_cli, "expand_media_inputs", return_value=(["one.wav", "two.wav"], [])), \
            mock.patch.object(asr_cli, "prepare_media_inputs", return_value=self._prepared_rows()), \
            mock.patch.object(asr_cli, "QwenSpeechOutput", side_effect=make_engine), \
            mock.patch("sys.argv", ["asr_cli.py", "--audio", "one.wav", "two.wav", "--return-time-stamps"]), \
            mock.patch("sys.stdout", stdout), \
            mock.patch("sys.stderr", io.StringIO()):
            asr_cli.main()

        self.assertEqual(len(created_engines), 1)
        self.assertEqual(len(created_engines[0].calls), 2)
        self.assertEqual(created_engines[0].calls[0]["audio"], "one.wav")
        self.assertEqual(created_engines[0].calls[1]["audio"], "two.wav")
        self.assertTrue(all(call["return_time_stamps"] for call in created_engines[0].calls))


class AsrResidentShellBatchingTests(unittest.TestCase):
    def _build_shell(self) -> asr_service_shell.ASRResidentShell:
        shell = asr_service_shell.ASRResidentShell(
            model_dir_arg=None,
            model_size="0.6b",
            aligner_dir=None,
            device="auto",
            dtype="auto",
            backend="transformers",
            gpu_memory_utilization=0.7,
            attn_implementation="auto",
            enable_tf32=True,
            cudnn_benchmark=True,
            warmup_audio=None,
            max_inference_batch_size=8,
            max_new_tokens=256,
            language=None,
            return_time_stamps=False,
            subtitle_dir=None,
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
        )
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        shell.temp_dir_obj = temp_dir
        shell.engine = FakeEngine()
        return shell

    def test_shell_batches_non_timestamp_requests(self) -> None:
        shell = self._build_shell()
        stdout = io.StringIO()
        def fake_prepare(media_paths: list[str], temp_dir: str, ffmpeg_bin: str, ffprobe_bin: str) -> list[PreparedMediaInput]:
            return [
                PreparedMediaInput(
                    source_media=media_paths[0],
                    prepared_audio=media_paths[0],
                    source_type="audio",
                    audio_duration_s=1.0,
                )
            ]

        with mock.patch.object(asr_service_shell, "expand_media_inputs", return_value=(["one.wav", "two.wav"], [])), \
            mock.patch.object(asr_service_shell, "prepare_media_inputs", side_effect=fake_prepare), \
            mock.patch("sys.stdout", stdout):
            shell.transcribe(["one.wav", "two.wav"])

        assert shell.engine is not None
        self.assertEqual(len(shell.engine.calls), 1)
        self.assertEqual(shell.engine.calls[0]["audio"], ["one.wav", "two.wav"])
        self.assertFalse(shell.engine.calls[0]["return_time_stamps"])


class FakeEngine:
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
                "audio": audio,
                "language": language,
                "return_time_stamps": return_time_stamps,
            }
        )
        if isinstance(audio, list):
            return [{"language": "zh", "text": f"text:{item}"} for item in audio]
        return [{"language": "zh", "text": f"text:{audio}", "time_stamps": []}]


if __name__ == "__main__":
    unittest.main()
