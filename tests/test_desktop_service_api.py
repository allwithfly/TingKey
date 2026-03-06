from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from desktop_service.api_server import create_app
from desktop_service.config_store import ConfigStore
from desktop_service.session_manager import SessionManager


class ConfigStoreRecoveryTests(unittest.TestCase):
    def test_invalid_config_is_preserved_before_reset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "desktop_service_config.json"
            config_path.write_text("{not-json", encoding="utf-8")

            store = ConfigStore(config_path)

            self.assertIsNotNone(store.get())
            backup_path = config_path.with_suffix(config_path.suffix + ".invalid")
            self.assertTrue(backup_path.exists())
            self.assertEqual(backup_path.read_text(encoding="utf-8"), "{not-json")
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertIsInstance(payload, dict)

    def test_invalid_config_backup_does_not_overwrite_existing_invalid_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "desktop_service_config.json"
            existing_backup = config_path.with_suffix(config_path.suffix + ".invalid")
            existing_backup.write_text("older-invalid", encoding="utf-8")
            config_path.write_text("{new-invalid", encoding="utf-8")

            ConfigStore(config_path)

            self.assertEqual(existing_backup.read_text(encoding="utf-8"), "older-invalid")
            next_backup = config_path.with_suffix(config_path.suffix + ".invalid.1")
            self.assertTrue(next_backup.exists())
            self.assertEqual(next_backup.read_text(encoding="utf-8"), "{new-invalid")


class SessionManagerStateTests(unittest.TestCase):
    def test_stop_session_rejects_already_stopped_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SessionManager(audio_root=temp_dir)
            session = manager.start_session()
            manager.stop_session(session.session_id)

            with self.assertRaisesRegex(ValueError, "already stopped|cannot be stopped from status stopped"):
                manager.stop_session(session.session_id)

    def test_mark_processing_rejects_recording_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SessionManager(audio_root=temp_dir)
            session = manager.start_session()

            with self.assertRaisesRegex(ValueError, "cannot enter processing"):
                manager.mark_processing(session.session_id)
            manager.stop_session(session.session_id)

    def test_mark_completed_rejects_recording_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SessionManager(audio_root=temp_dir)
            session = manager.start_session()

            with self.assertRaisesRegex(ValueError, "cannot complete"):
                manager.mark_completed(session.session_id, final_text="done")
            manager.stop_session(session.session_id)

    def test_mark_error_closes_recording_writer_and_preserves_error_message(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SessionManager(audio_root=temp_dir)
            session = manager.start_session()

            errored = manager.mark_error(session.session_id, "boom")

            self.assertEqual(errored.status.value, "error")
            self.assertEqual(errored.error_message, "boom")
            self.assertIsNone(errored.wave_writer)
            with self.assertRaisesRegex(ValueError, "cannot be stopped from status error"):
                manager.stop_session(session.session_id)


class ApiServerInputTests(unittest.TestCase):
    def test_invalid_base64_chunk_returns_400(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "desktop_service_config.json"
            audio_root = Path(temp_dir) / "recordings"
            app = create_app(config_path=config_path, audio_root=audio_root)
            with TestClient(app) as client:
                started = client.post("/v1/sessions/start", json={})
                self.assertEqual(started.status_code, 200)
                session_id = started.json()["session_id"]

                try:
                    response = client.post(
                        f"/v1/sessions/{session_id}/chunk",
                        json={"audio_base64": "%%%not-base64%%%"},
                    )

                    self.assertEqual(response.status_code, 400)
                    self.assertIn("invalid base64", response.json()["detail"].lower())
                finally:
                    client.post(
                        f"/v1/sessions/{session_id}/stop",
                        json={"run_final_asr": False},
                    )


if __name__ == "__main__":
    unittest.main()
