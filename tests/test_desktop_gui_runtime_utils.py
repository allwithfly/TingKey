from __future__ import annotations

import unittest

from desktop_gui_runtime_utils import (
    build_subtitle_output_path,
    build_file_done_payload,
    build_file_partial_payload,
    build_file_start_payload,
    extract_partial_text,
    format_worker_failure,
    requires_wav_conversion_for_chunking,
    should_use_chunked_transcription,
)


class DesktopGuiRuntimeUtilsTests(unittest.TestCase):
    def test_should_use_chunked_transcription_requires_long_wav(self) -> None:
        self.assertFalse(should_use_chunked_transcription(None, "demo.wav", 70.0))
        self.assertFalse(should_use_chunked_transcription(30.0, "demo.wav", 70.0))
        self.assertFalse(should_use_chunked_transcription(90.0, "demo.mp3", 70.0))
        self.assertTrue(should_use_chunked_transcription(90.0, "demo.wav", 70.0))

    def test_extract_partial_text_returns_trimmed_first_row_text(self) -> None:
        self.assertEqual(extract_partial_text([]), "")
        self.assertEqual(extract_partial_text([{}]), "")
        self.assertEqual(extract_partial_text([{"text": "  hello world  "}]), "hello world")
        self.assertEqual(extract_partial_text([{"text": None}]), "")

    def test_requires_wav_conversion_for_chunking_only_for_long_non_wav(self) -> None:
        self.assertFalse(requires_wav_conversion_for_chunking(None, "demo.mp3", 70.0))
        self.assertFalse(requires_wav_conversion_for_chunking(30.0, "demo.mp3", 70.0))
        self.assertFalse(requires_wav_conversion_for_chunking(90.0, "demo.wav", 70.0))
        self.assertTrue(requires_wav_conversion_for_chunking(90.0, "demo.mp3", 70.0))

    def test_build_progress_payload_helpers(self) -> None:
        start = build_file_start_payload(index=2, total=5, source_media="demo.wav")
        self.assertEqual(start["event"], "file_start")
        self.assertEqual(start["index"], 2)
        self.assertEqual(start["total"], 5)
        self.assertEqual(start["source_media"], "demo.wav")
        self.assertEqual(start["current_text"], "")

        done = build_file_done_payload(
            index=2,
            total=5,
            source_media="demo.wav",
            audio_path="prepared.wav",
            was_extracted=False,
            language="zh",
            text="hello",
            audio_duration_s=12.3,
            model_inference_time_s=1.2,
            segments=[],
            subtitle_path=None,
        )
        self.assertEqual(done["event"], "file_done")
        self.assertEqual(done["audio_path"], "prepared.wav")
        self.assertEqual(done["text"], "hello")
        self.assertEqual(done["progress_percent"], 40)

        partial = build_file_partial_payload(
            index=2,
            total=5,
            source_media="demo.wav",
            file_progress=1.2,
            global_progress=-0.5,
            current_text="hello",
            model_time_s=1.2,
        )
        self.assertEqual(partial["event"], "file_partial")
        self.assertEqual(partial["file_progress_percent"], 100)
        self.assertEqual(partial["progress_percent"], 0)
        self.assertEqual(partial["current_text"], "hello")
        self.assertEqual(partial["model_inference_time_s"], 1.2)

    def test_format_worker_failure_combines_exception_and_traceback(self) -> None:
        detail = format_worker_failure(RuntimeError("boom"), "Traceback line 1\nTraceback line 2")

        self.assertIn("boom", detail)
        self.assertIn("Traceback line 1", detail)
        self.assertIn("Traceback line 2", detail)

    def test_build_subtitle_output_path_requires_dir_and_segments(self) -> None:
        self.assertIsNone(build_subtitle_output_path(None, "demo.wav", has_segments=True))
        self.assertIsNone(build_subtitle_output_path("subtitles", "demo.wav", has_segments=False))

        target = build_subtitle_output_path("subtitles", "folder/demo.wav", has_segments=True)

        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.name, "demo.srt")
        self.assertEqual(target.parent.name, "subtitles")


if __name__ == "__main__":
    unittest.main()
