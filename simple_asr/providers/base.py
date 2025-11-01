"""Abstract base class for ASR providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseASRProvider(ABC):
    """Common interface for all ASR providers."""

    name: str = "base"

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id

    @abstractmethod
    def load(self) -> None:
        """Ensure any heavy resources are ready for inference."""

    @abstractmethod
    def transcribe(self, audio_path: Path) -> str:
        """Return the transcription for the given audio file."""

