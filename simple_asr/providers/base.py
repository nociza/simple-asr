"""Abstract base class for ASR providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Protocol


class ProgressReporter(Protocol):
    """Callable interface used to surface progress updates to the caller."""

    def __call__(self, message: str) -> None:
        ...


class BaseASRProvider(ABC):
    """Common interface for all ASR providers."""

    name: str = "base"

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id

    @abstractmethod
    def load(self, report_progress: ProgressReporter | None = None) -> None:
        """Ensure any heavy resources are ready for inference."""

    @abstractmethod
    def transcribe(self, audio_path: Path) -> str:
        """Return the transcription for the given audio file."""

    def add_vocabulary(self, phrases: list[str]) -> None:
        """Extend the provider with custom bias phrases.

        Implementations can override or extend this to hook into model-specific
        vocabulary or prompt modifiers.
        """

    def clear_vocabulary(self) -> None:
        """Remove any custom vocabulary previously registered."""

