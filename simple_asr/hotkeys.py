"""Keyboard hotkey integration for triggering audio capture and transcription."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, Optional, Union

from pynput import keyboard

LOGGER = logging.getLogger(__name__)


KeyType = Union[keyboard.Key, str]


def _build_special_key_map() -> dict[str, keyboard.Key]:
    mapping: dict[str, keyboard.Key] = {
        "space": keyboard.Key.space,
        "enter": keyboard.Key.enter,
        "return": keyboard.Key.enter,
        "tab": keyboard.Key.tab,
        "esc": keyboard.Key.esc,
        "escape": keyboard.Key.esc,
    }

    for index in range(1, 13):
        mapping[f"f{index}"] = getattr(keyboard.Key, f"f{index}")

    return mapping


SPECIAL_KEYS = _build_special_key_map()


class HotkeyTranscriber:
    """Manage the record/transcribe lifecycle tied to a single hotkey."""

    def __init__(
        self,
        provider,
        recorder,
        hotkey: str = "f8",
        on_transcription: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.provider = provider
        self.recorder = recorder
        self._hotkey, self.hotkey_label = self._normalize_hotkey(hotkey)
        self._on_transcription = on_transcription
        self._recording = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """Block while listening for the configured hotkey."""

        with keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        ) as listener:
            listener.join()

    def _normalize_hotkey(self, hotkey: str) -> tuple[KeyType, str]:
        key = hotkey.strip().lower()
        if not key:
            raise ValueError("Hotkey must not be empty.")

        if key in SPECIAL_KEYS:
            return SPECIAL_KEYS[key], key.upper()

        if len(key) == 1:
            return key, key.upper()

        raise ValueError(f"Unsupported hotkey '{hotkey}'.")

    def _matches_hotkey(self, key: keyboard.Key | keyboard.KeyCode) -> bool:
        target = self._hotkey
        if isinstance(target, keyboard.Key):
            return key == target

        try:
            return key.char is not None and key.char.lower() == target
        except AttributeError:
            return False

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        if not self._matches_hotkey(key):
            return

        with self._lock:
            if self._recording:
                return
            self._recording = True

        try:
            self.recorder.start()
        except Exception as exc:  # pragma: no cover - user environment specific
            LOGGER.exception("Failed to start recording: %s", exc)
            print(f"[Error] Could not start recording: {exc}")
            with self._lock:
                self._recording = False
            return

        print("Recording... release the key to transcribe.")

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        if not self._matches_hotkey(key):
            return

        with self._lock:
            if not self._recording:
                return
            self._recording = False

        audio_path = self.recorder.stop()

        if audio_path is None:
            print("No audio captured. Try again.")
            return

        self._transcribe(audio_path)

    def _transcribe(self, audio_path: Path) -> None:
        print("Transcribing...")
        try:
            text = self.provider.transcribe(audio_path)
        except Exception as exc:  # pragma: no cover - user environment specific
            LOGGER.exception("Transcription failed: %s", exc)
            print(f"[Error] Transcription failed: {exc}")
            text = None
        finally:
            try:
                audio_path.unlink()
            except OSError:
                LOGGER.warning("Failed to delete temporary audio file at %s", audio_path)

        if text is None:
            return

        if self._on_transcription:
            self._on_transcription(text)

        print(f"Transcript: {text}")

