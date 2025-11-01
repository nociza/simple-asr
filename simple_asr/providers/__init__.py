"""Provider registry for the simple ASR application."""

from __future__ import annotations

from typing import Type

from .base import BaseASRProvider
from .canary import CanaryProvider


PROVIDERS: dict[str, Type[BaseASRProvider]] = {
    CanaryProvider.name: CanaryProvider,
}


def get_provider(name: str) -> Type[BaseASRProvider]:
    try:
        return PROVIDERS[name.lower()]
    except KeyError as exc:  # pragma: no cover - defensive programming
        available = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"Unknown provider '{name}'. Available providers: {available}") from exc


__all__ = ["BaseASRProvider", "CanaryProvider", "get_provider"]

