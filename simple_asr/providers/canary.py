"""Provider implementation for NVIDIA's Canary SALM ASR model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import soundfile as sf
from nemo.collections.speechlm2.models import SALM

from .base import BaseASRProvider


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

    def load(self) -> None:
        if self._model is not None:
            return

        model_id = self.model_id or self.DEFAULT_MODEL_ID
        LOGGER.info("Loading Canary SALM model '%s'", model_id)
        self._model = SALM.from_pretrained(model_id)

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

