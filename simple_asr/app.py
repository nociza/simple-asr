"""Application wiring for the simple ASR hotkey recorder."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

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


class SimpleASRApp:
    """Bootstrap the provider, recorder, and hotkey listener."""

    def __init__(self, config: AppConfig):
        self.config = config
        provider_cls = get_provider(config.provider_name)
        provider_kwargs = {"model_id": config.model_id, **config.provider_options}
        self.provider = provider_cls(**provider_kwargs)
        self.recorder = AudioRecorder(sample_rate=config.sample_rate)

    def run(self) -> None:
        """Start the ASR application and block until the user exits."""

        LOGGER.info("Loading provider '%s'", self.provider.name)
        print(f"Loading ASR model for provider '{self.provider.name}'. This may take a moment...")
        self.provider.load()
        print("Model loaded. Ready when you are!")

        listener = HotkeyTranscriber(
            provider=self.provider,
            recorder=self.recorder,
            hotkey=self.config.hotkey,
        )

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

