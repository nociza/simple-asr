"""Keyboard hotkey integration for triggering audio capture and transcription."""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Union

import pyperclip
from pynput import keyboard
from pynput.keyboard import Key

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
        self._keyboard_controller = keyboard.Controller()
        self._paste_modifier = Key.cmd if sys.platform == "darwin" else Key.ctrl
        self._stop_event = threading.Event()
        self._listener: keyboard.Listener | None = None
        self._ctrl_active = False
        self._shutdown_requested = False
        self._record_started_at: float | None = None
        self._record_stopped_at: float | None = None
        self._transcribe_started_at: float | None = None

    def start(self) -> None:
        """Block while listening for the configured hotkey."""

        self._stop_event.clear()
        self._shutdown_requested = False

        listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener = listener
        listener.start()

        try:
            self._stop_event.wait()
        except KeyboardInterrupt:
            self._shutdown_requested = True
            raise
        finally:
            listener.stop()
            listener.join()
            self._listener = None

        if self._shutdown_requested:
            raise KeyboardInterrupt

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
        if key in (Key.ctrl, Key.ctrl_l, Key.ctrl_r):
            self._ctrl_active = True

        if self._ctrl_active:
            char = getattr(key, "char", None)
            if char and char.lower() == "c":
                self._request_shutdown()
                return

        if self._shutdown_requested:
            return

        if not self._matches_hotkey(key):
            return

        with self._lock:
            if self._recording:
                return
            self._recording = True
            self._record_started_at = time.perf_counter()
            self._record_stopped_at = None
            self._transcribe_started_at = None

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
        if key in (Key.ctrl, Key.ctrl_l, Key.ctrl_r):
            self._ctrl_active = False

        if self._shutdown_requested:
            return

        if not self._matches_hotkey(key):
            return

        with self._lock:
            if not self._recording:
                return
            self._recording = False
            self._record_stopped_at = time.perf_counter()

        audio_path = self.recorder.stop()

        if audio_path is None:
            print("No audio captured. Try again.")
            return

        self._start_transcription(audio_path)

    def _start_transcription(self, audio_path: Path) -> None:
        threading.Thread(
            target=self._transcribe,
            args=(audio_path,),
            daemon=True,
        ).start()

    def _transcribe(self, audio_path: Path) -> None:
        print("Transcribing...")
        self._transcribe_started_at = time.perf_counter()
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
                LOGGER.warning(
                    "Failed to delete temporary audio file at %s", audio_path
                )

        if text is None:
            return

        if self._on_transcription:
            self._on_transcription(text)

        self._deliver_transcript(text)

    def _deliver_transcript(self, text: str) -> None:
        print(f"Transcript: {text}")

        if not text:
            return

        completed_at = time.perf_counter()
        capture_ms = (
            (self._record_stopped_at - self._record_started_at) * 1000
            if self._record_started_at and self._record_stopped_at
            else None
        )
        decode_ms = (
            (completed_at - self._transcribe_started_at) * 1000
            if self._transcribe_started_at
            else None
        )
        end_to_end_ms = (
            (completed_at - self._record_started_at) * 1000
            if self._record_started_at
            else None
        )

        if (
            capture_ms is not None
            and decode_ms is not None
            and end_to_end_ms is not None
        ):
            LOGGER.info(
                "Session timings (ms): capture=%0.0f decode=%0.0f total=%0.0f",
                capture_ms,
                decode_ms,
                end_to_end_ms,
            )
            print(
                "⏱️  Durations — capture: {cap:.0f} ms | decode: {dec:.0f} ms | total: {tot:.0f} ms".format(
                    cap=capture_ms,
                    dec=decode_ms,
                    tot=end_to_end_ms,
                )
            )

        try:
            pyperclip.copy(text)
        except pyperclip.PyperclipException as exc:
            LOGGER.debug("Clipboard copy failed: %s. Falling back to typing.", exc)
            self._keyboard_controller.type(text)
            return

        # Give the OS a moment to register the clipboard update before pasting.
        time.sleep(0.05)

        try:
            self._keyboard_controller.press(self._paste_modifier)
            self._keyboard_controller.press("v")
            self._keyboard_controller.release("v")
            self._keyboard_controller.release(self._paste_modifier)
        except Exception as exc:  # pragma: no cover - dependent on host OS
            LOGGER.exception("Failed to send paste key sequence: %s", exc)
            self._keyboard_controller.type(text)

    def _request_shutdown(self) -> None:
        if self._shutdown_requested:
            return

        self._shutdown_requested = True

        with self._lock:
            was_recording = self._recording
            self._recording = False

        if was_recording:
            audio_path = self.recorder.stop()
            if audio_path is not None:
                try:
                    audio_path.unlink()
                except OSError:
                    LOGGER.debug(
                        "Failed to remove temporary audio file during shutdown: %s",
                        audio_path,
                    )

        self._recording = False
        self._record_started_at = None
        self._record_stopped_at = None
        self._transcribe_started_at = None

        self._stop_event.set()
        listener = self._listener
        if listener is not None:
            listener.stop()

    def stop(self) -> None:
        self._request_shutdown()

    def update_hotkey(self, hotkey: str) -> None:
        normalized_key, label = self._normalize_hotkey(hotkey)
        with self._lock:
            self._hotkey = normalized_key
            self.hotkey_label = label
        print(f"Hotkey updated to '{self.hotkey_label}'.")

    def update_recorder(self, recorder) -> None:
        with self._lock:
            self.recorder = recorder
        print("Recorder updated to new settings.")

    def update_provider(self, provider) -> None:
        with self._lock:
            self.provider = provider
