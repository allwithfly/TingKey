from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from desktop_service.config_models import AppConfig


def _model_dump(model: AppConfig) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # pydantic v2
    return model.dict()  # pydantic v1


class ConfigStore:
    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._config = self._load_or_default()

    def get(self) -> AppConfig:
        with self._lock:
            return self._config

    def get_dict(self) -> dict[str, Any]:
        with self._lock:
            return _model_dump(self._config)

    def set(self, config: AppConfig) -> AppConfig:
        with self._lock:
            self._config = config
            self._save()
            return self._config

    def update_section(self, section: str, partial: dict[str, Any]) -> AppConfig:
        with self._lock:
            if not hasattr(self._config, section):
                raise KeyError(f"unknown config section: {section}")
            section_obj = getattr(self._config, section)
            base = section_obj.model_dump() if hasattr(section_obj, "model_dump") else section_obj.dict()
            merged = {**base, **partial}
            updated_section = section_obj.__class__(**merged)
            updated_data = self.get_dict()
            updated_data[section] = (
                updated_section.model_dump()
                if hasattr(updated_section, "model_dump")
                else updated_section.dict()
            )
            self._config = AppConfig(**updated_data)
            self._save()
            return self._config

    def _load_or_default(self) -> AppConfig:
        if not self.config_path.exists():
            cfg = AppConfig()
            self._save_data(_model_dump(cfg))
            return cfg

        try:
            payload = json.loads(self.config_path.read_text(encoding="utf-8"))
            return AppConfig(**payload)
        except Exception:
            cfg = AppConfig()
            self._save_data(_model_dump(cfg))
            return cfg

    def _save(self) -> None:
        self._save_data(self.get_dict())

    def _save_data(self, payload: dict[str, Any]) -> None:
        self.config_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

