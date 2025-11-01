"""Audio recording utilities for the ASR application."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

LOGGER = logging.getLogger(__name__)


class AudioRecorder:
    """Simple microphone recorder driven by explicit start/stop calls."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._stream: Optional[sd.InputStream] = None
        self._frames: list[np.ndarray] = []
        self._frames_lock = threading.Lock()

    def start(self) -> None:
        """Begin capturing audio from the default input device."""

        if self._stream is not None:
            LOGGER.debug("Recorder already running; ignoring start request.")
            return

        self._frames = []

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
            LOGGER.debug("Audio stream started.")
        except Exception:
            self._stream = None
            LOGGER.exception("Failed to start audio input stream.")
            raise

    def stop(self) -> Optional[Path]:
        """Stop the recording session and persist the audio to a temporary file."""

        if self._stream is None:
            LOGGER.debug("Recorder not running; ignoring stop request.")
            return None

        try:
            self._stream.stop()
            self._stream.close()
            LOGGER.debug("Audio stream stopped.")
        finally:
            self._stream = None

        with self._frames_lock:
            if not self._frames:
                LOGGER.info("No audio captured during the session.")
                return None

            audio = np.concatenate(self._frames, axis=0)
            self._frames = []

        audio_path = self._write_temp_wav(audio)
        LOGGER.debug("Wrote temporary audio file to %s", audio_path)
        return audio_path

    def close(self) -> None:
        """Release any resources held by the recorder."""

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                LOGGER.exception("Error while shutting down the audio stream.")
            finally:
                self._stream = None

        with self._frames_lock:
            self._frames = []

    def _callback(self, indata: np.ndarray, frames: int, time, status) -> None:  # type: ignore[override]
        if status:
            LOGGER.warning("Audio input status: %s", status)

        with self._frames_lock:
            self._frames.append(indata.copy())

    def _write_temp_wav(self, audio: np.ndarray) -> Path:
        with NamedTemporaryFile(prefix="simple-asr-", suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, self.sample_rate)
            return Path(tmp.name)

