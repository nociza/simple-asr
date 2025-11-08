"""Helpers for managing custom vocabulary stored in a TOML file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

try:  # Python 3.11+
    import tomllib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for older versions
    import tomli as tomllib  # type: ignore


DEFAULT_VOCAB_FILE = Path.cwd() / "vocab.toml"


def load_vocabulary(path: Path | None = None) -> List[str]:
    file_path = path or DEFAULT_VOCAB_FILE
    if not file_path.exists():
        return []

    try:
        data = tomllib.loads(file_path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError):
        return []

    raw_phrases = data.get("phrases") or data.get("vocabulary")
    if not isinstance(raw_phrases, list):
        return []

    phrases: list[str] = []
    for item in raw_phrases:
        if isinstance(item, str):
            phrase = item.strip()
            if phrase and phrase not in phrases:
                phrases.append(phrase)
    return phrases


def save_vocabulary(phrases: Iterable[str], path: Path | None = None) -> None:
    file_path = path or DEFAULT_VOCAB_FILE
    file_path.parent.mkdir(parents=True, exist_ok=True)
    normalized: list[str] = []
    for phrase in phrases:
        phrase = str(phrase).strip()
        if phrase and phrase not in normalized:
            normalized.append(phrase)

    content_lines = ["phrases = ["]
    if normalized:
        formatted = ",\n".join(f"  {json.dumps(item)}" for item in normalized)
        content_lines.append(formatted)
    content_lines.append("]\n")

    file_path.write_text("\n".join(content_lines), encoding="utf-8")


__all__ = ["DEFAULT_VOCAB_FILE", "load_vocabulary", "save_vocabulary"]


