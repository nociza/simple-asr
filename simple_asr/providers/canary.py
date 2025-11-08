"""Provider implementation for NVIDIA's Canary SALM ASR model."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import List

import soundfile as sf
from nemo.collections.speechlm2.models import SALM

from .base import BaseASRProvider, ProgressReporter


LOGGER = logging.getLogger(__name__)


class CanaryProvider(BaseASRProvider):
    """ASR provider that wraps the Canary Qwen 2.5B model."""

    name = "canary"
    DEFAULT_MODEL_ID = "nvidia/canary-qwen-2.5b"

    def __init__(
        self,
        model_id: str | None = None,
        max_new_tokens: int = 128,
        vocabulary: list[str] | None = None,
    ) -> None:
        super().__init__(model_id=model_id)
        self.max_new_tokens = max_new_tokens
        self._model: SALM | None = None
        self._vocabulary: list[str] = vocabulary or []

    def load(self, report_progress: ProgressReporter | None = None) -> None:
        if self._model is not None:
            if report_progress is not None:
                report_progress("Canary model already initialized; reusing existing instance.")
            return

        model_id = self.model_id or self.DEFAULT_MODEL_ID

        def emit(message: str) -> None:
            LOGGER.info(message)
            if report_progress is not None:
                report_progress(message)

        emit("Checking local cache for Canary model resources...")
        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.utils import LocalEntryNotFoundError
        except ImportError:
            emit(
                "huggingface_hub not available; skipping explicit cache warm-up (downloads will still occur automatically if needed)."
            )
        else:
            try:
                snapshot_download(repo_id=model_id, local_files_only=True)
                emit("Canary model weights already present in the local Hugging Face cache.")
            except LocalEntryNotFoundError:
                emit(
                    f"Downloading Canary model '{model_id}' from Hugging Face — first run may take several minutes..."
                )
                snapshot_download(repo_id=model_id)
                emit("Download complete. Continuing with model initialization...")
            except Exception as exc:  # pragma: no cover - best effort logging
                LOGGER.warning("Cache warm-up encountered an issue: %s", exc, exc_info=True)
                if report_progress is not None:
                    report_progress("Encountered an issue while checking the cache; continuing with model load.")

        emit("Initializing Canary SALM model (loading weights into memory)...")
        start_time = time.perf_counter()
        self._model = SALM.from_pretrained(model_id)
        duration = time.perf_counter() - start_time
        emit(f"Canary model initialized in {duration:.1f}s.")

    @property
    def model(self) -> SALM:
        if self._model is None:
            self.load()
        return self._model  # type: ignore[return-value]

    def transcribe(self, audio_path: Path) -> str:
        model = self.model

        prompt_payload: List[List[dict]] = [
            [
                {
                    "role": "user",
                    "content": f"Transcribe the following: {model.audio_locator_tag}",
                    "audio": [str(audio_path)],
                }
            ]
        ]

        adaptive_tokens = self._estimate_max_tokens(audio_path)

        if self._vocabulary:
            vocab_prompt = "\n".join(self._vocabulary)
            prompt_payload[0][0]["content"] += (
                f"\nUse the following terminology when appropriate: {vocab_prompt}"
            )

        LOGGER.debug(
            "Invoking model.generate for audio %s (max_new_tokens=%s)",
            audio_path,
            adaptive_tokens,
        )
        answer_ids = model.generate(
            prompts=prompt_payload,
            max_new_tokens=adaptive_tokens,
        )

        text = model.tokenizer.ids_to_text(answer_ids[0].cpu())
        return text.strip()

    def _estimate_max_tokens(self, audio_path: Path) -> int:
        try:
            info = sf.info(str(audio_path))
            duration = max(info.duration, 0.1)
        except RuntimeError:
            LOGGER.debug("Falling back to default max_new_tokens due to unreadable audio metadata.")
            return self.max_new_tokens

        estimated = int(duration * 6)  # Empirical factor for Canary decoding speed.
        bounded = max(24, min(self.max_new_tokens, estimated))
        return bounded

    def add_vocabulary(self, phrases: list[str]) -> None:
        normalized = [phrase.strip() for phrase in phrases if phrase.strip()]
        self._vocabulary.extend(p for p in normalized if p not in self._vocabulary)

    def clear_vocabulary(self) -> None:
        self._vocabulary.clear()

