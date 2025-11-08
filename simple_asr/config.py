"""Configuration persistence for the Simple ASR application."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

from .app import AppConfig


DEFAULT_CONFIG_PATH = Path.home() / ".simple_asr" / "config.json"


class ConfigManager:
    """Load and store application configuration on disk."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or DEFAULT_CONFIG_PATH
        self._lock = threading.Lock()

    def load(self) -> AppConfig:
        with self._lock:
            if not self.path.exists():
                return AppConfig()

            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return AppConfig()

        return AppConfig.from_dict(data)

    def save(self, config: AppConfig) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(config.to_dict(), indent=2, sort_keys=True)
            self.path.write_text(payload, encoding="utf-8")

    def update(self, config: AppConfig) -> AppConfig:
        self.save(config)
        return config

