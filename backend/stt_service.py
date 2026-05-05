from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional


@dataclass
class STTResult:
    text: str
    language: Optional[str] = None
    duration_sec: float = 0.0


class STTService:
    """Local speech-to-text service powered by faster-whisper."""

    def __init__(self, model_size: str = "base") -> None:
        self.model_size = model_size
        self._model = None
        self._lock = Lock()

    def set_model(self, model_size: str) -> None:
        if model_size == self.model_size:
            return
        with self._lock:
            self.model_size = model_size
            self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from faster_whisper import WhisperModel  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "faster-whisper is not installed. Install backend requirements to use voice STT."
                ) from exc

            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            return self._model

    def transcribe(self, audio_file: str | Path, language: Optional[str] = None) -> STTResult:
        model = self._load_model()

        segments, info = model.transcribe(
            str(audio_file),
            beam_size=5,
            vad_filter=True,
            language=language,
            condition_on_previous_text=False,
        )

        parts = []
        duration = 0.0
        for seg in segments:
            text = (seg.text or "").strip()
            if text:
                parts.append(text)
            try:
                duration = max(duration, float(seg.end))
            except Exception:
                pass

        return STTResult(
            text=" ".join(parts).strip(),
            language=getattr(info, "language", None),
            duration_sec=duration,
        )
