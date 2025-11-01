"""Provider implementation for NVIDIA's Canary SALM ASR model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

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
    ) -> None:
        super().__init__(model_id=model_id)
        self.max_new_tokens = max_new_tokens
        self._model: SALM | None = None

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

        LOGGER.debug("Invoking model.generate for audio %s", audio_path)
        answer_ids = model.generate(
            prompts=prompt_payload,
            max_new_tokens=self.max_new_tokens,
        )

        text = model.tokenizer.ids_to_text(answer_ids[0].cpu())
        return text.strip()

