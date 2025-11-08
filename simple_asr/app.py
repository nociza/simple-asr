"""Application wiring for the simple ASR hotkey recorder."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from .audio import AudioRecorder
from .hotkeys import HotkeyTranscriber
from .providers import get_provider


LOGGER = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """Runtime configuration for the ASR application."""

    provider_name: str = "canary"
    hotkey: str = "f8"
    model_id: str | None = None
    sample_rate: int = 16000
    provider_options: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:  # type: ignore[override]
        return {
            "provider_name": self.provider_name,
            "hotkey": self.hotkey,
            "model_id": self.model_id,
            "sample_rate": self.sample_rate,
            "provider_options": dict(self.provider_options),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        return cls(
            provider_name=data.get("provider_name", "canary"),
            hotkey=data.get("hotkey", "f8"),
            model_id=data.get("model_id"),
            sample_rate=int(data.get("sample_rate", 16000)),
            provider_options=dict(data.get("provider_options", {})),
        )

    def copy(self) -> "AppConfig":
        return AppConfig.from_dict(self.to_dict())


class SimpleASRApp:
    """Bootstrap the provider, recorder, and hotkey listener."""

    def __init__(self, config: AppConfig):
        self.config = config
        provider_cls = get_provider(config.provider_name)
        provider_kwargs = {"model_id": config.model_id, **config.provider_options}
        self.provider = provider_cls(**provider_kwargs)
        self.recorder = AudioRecorder(sample_rate=config.sample_rate)
        self.listener: HotkeyTranscriber | None = None
        self._listener_thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def run(self) -> None:
        """Start the ASR application and block until the user exits."""

        LOGGER.info("Loading provider '%s'", self.provider.name)
        print(f"Loading ASR model for provider '{self.provider.name}'. This may take a moment...")

        def report_progress(message: str) -> None:
            print(f"  -> {message}")

        self.provider.load(report_progress=report_progress)
        print("Model loaded. Ready when you are!")

        listener = HotkeyTranscriber(
            provider=self.provider,
            recorder=self.recorder,
            hotkey=self.config.hotkey,
        )
        self.listener = listener

        print(
            "Hold the '{key}' key to record. Release it to transcribe. Press Ctrl+C to exit.".format(
                key=listener.hotkey_label
            )
        )

        try:
            listener.start()
        except KeyboardInterrupt:
            print("Exiting per user request.")
        finally:
            self.recorder.close()

    def apply_settings(self, new_config: AppConfig) -> None:
        with self._lock:
            listener = self.listener

            if new_config.hotkey != self.config.hotkey and listener is not None:
                listener.update_hotkey(new_config.hotkey)

            if new_config.sample_rate != self.config.sample_rate:
                self.recorder.close()
                self.recorder = AudioRecorder(sample_rate=new_config.sample_rate)
                if listener is not None:
                    listener.update_recorder(self.recorder)

            new_provider_name = new_config.provider_name
            if new_provider_name != self.config.provider_name or new_config.model_id != self.config.model_id:
                raise ValueError(
                    "Switching providers or model IDs at runtime is not yet supported. Restart the application."
                )

            new_options = dict(new_config.provider_options)
            new_vocab = list(new_options.get("vocabulary", []))
            old_vocab = list(self.config.provider_options.get("vocabulary", []))
            if new_vocab != old_vocab:
                if hasattr(self.provider, "clear_vocabulary"):
                    self.provider.clear_vocabulary()
                if new_vocab and hasattr(self.provider, "add_vocabulary"):
                    self.provider.add_vocabulary(new_vocab)

            new_max_tokens = new_options.get("max_new_tokens")
            old_max_tokens = self.config.provider_options.get("max_new_tokens")
            if new_max_tokens is not None and new_max_tokens != old_max_tokens:
                if hasattr(self.provider, "max_new_tokens"):
                    self.provider.max_new_tokens = int(new_max_tokens)  # type: ignore[attr-defined]

            self.config = new_config.copy()
            print("Settings updated and persisted.")

