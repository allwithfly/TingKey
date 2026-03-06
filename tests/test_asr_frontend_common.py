from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import asr_cli
import asr_service_shell


class FrontendBinaryPathTests(unittest.TestCase):
    def test_prefers_env_ffmpeg_directory(self) -> None:
        from asr_frontend_common import default_binary_path

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_dir = root / "env-bin"
            env_dir.mkdir()
            expected = env_dir / "ffmpeg.exe"
            expected.write_text("", encoding="utf-8")

            fallback_root = root / "app"
            (fallback_root / "ffmpeg" / "bin").mkdir(parents=True)
            ((fallback_root / "ffmpeg" / "bin") / "ffmpeg.exe").write_text("", encoding="utf-8")

            resolved = default_binary_path(
                "ffmpeg.exe",
                "ffmpeg",
                base_dir=fallback_root,
                env_dir=str(env_dir),
            )

        self.assertEqual(resolved, str(expected.resolve()))

    def test_returns_fallback_when_binary_is_missing(self) -> None:
        from asr_frontend_common import default_binary_path

        with tempfile.TemporaryDirectory() as temp_dir:
            resolved = default_binary_path(
                "ffprobe.exe",
                "ffprobe",
                base_dir=temp_dir,
                env_dir="",
            )

        self.assertEqual(resolved, "ffprobe")

    def test_build_subtitle_path_uses_source_stem(self) -> None:
        from asr_frontend_common import build_subtitle_path

        with tempfile.TemporaryDirectory() as temp_dir:
            resolved = build_subtitle_path(temp_dir, "C:/tmp/example.clip.mp4")

        self.assertEqual(resolved.name, "example.clip.srt")

    def test_format_transcription_result_lines_renders_shared_output(self) -> None:
        from asr_frontend_common import format_transcription_result_lines

        result = {
            "audio": "demo.wav",
            "prepared_audio": "prepared.wav",
            "source_type": "video",
            "language": "zh",
            "audio_duration_s": 12.5,
            "model_inference_time_s": 1.25,
            "segments": [
                {"start_time": 0.0, "end_time": 1.0, "text": "hello"},
            ],
            "subtitle_path": "subtitles/demo.srt",
            "text": "hello world",
        }

        lines = format_transcription_result_lines(result, header_label="1/2")

        self.assertEqual(lines[0], "[1/2] audio=demo.wav")
        self.assertIn("source_type=video extracted_audio=prepared.wav", lines)
        self.assertIn("language=zh", lines)
        self.assertIn("subtitle=subtitles/demo.srt", lines)
        self.assertIn("text=hello world", lines)
        self.assertTrue(any(line.startswith("segment[0] start=") for line in lines))


class FrontendValidationTests(unittest.TestCase):
    def test_rejects_invalid_numeric_runtime_args(self) -> None:
        from asr_frontend_common import validate_runtime_numeric_args

        with self.assertRaisesRegex(ValueError, "max_inference_batch_size"):
            validate_runtime_numeric_args(
                max_inference_batch_size=0,
                max_new_tokens=256,
                gpu_memory_utilization=0.7,
            )

        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            validate_runtime_numeric_args(
                max_inference_batch_size=1,
                max_new_tokens=0,
                gpu_memory_utilization=0.7,
            )

        with self.assertRaisesRegex(ValueError, "gpu_memory_utilization"):
            validate_runtime_numeric_args(
                max_inference_batch_size=1,
                max_new_tokens=16,
                gpu_memory_utilization=1.5,
            )

    def test_cli_exits_early_for_invalid_batch_size(self) -> None:
        with mock.patch.object(asr_cli, "QwenSpeechOutput") as mocked_engine, \
            mock.patch("sys.argv", ["asr_cli.py", "--audio", "demo.wav", "--max-inference-batch-size", "0"]):
            with self.assertRaises(SystemExit) as exc:
                asr_cli.main()

        self.assertEqual(exc.exception.code, 2)
        mocked_engine.assert_not_called()

    def test_shell_exits_early_for_invalid_gpu_utilization(self) -> None:
        with mock.patch.object(asr_service_shell, "ASRResidentShell") as mocked_shell, \
            mock.patch("sys.argv", ["asr_service_shell.py", "--gpu-memory-utilization", "0"]):
            with self.assertRaises(SystemExit) as exc:
                asr_service_shell.main()

        self.assertEqual(exc.exception.code, 2)
        mocked_shell.assert_not_called()


if __name__ == "__main__":
    unittest.main()
