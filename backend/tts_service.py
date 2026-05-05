from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from threading import Lock


def strip_markdown(text: str) -> str:
    cleaned = text or ""
    cleaned = re.sub(r"```[\\s\\S]*?```", " ", cleaned)
    cleaned = re.sub(r"`([^`]+)`", r"\\1", cleaned)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\\1", cleaned)
    cleaned = re.sub(r"[_~#>-]", " ", cleaned)
    cleaned = re.sub(r"\[(.*?)\]\((.*?)\)", r"\\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


class TTSService:
    """Local text-to-speech service that supports Piper and Coqui."""

    def __init__(self, engine: str = "piper") -> None:
        self.engine = engine
        self._lock = Lock()
        self._coqui = None

    def set_engine(self, engine: str) -> None:
        if engine == self.engine:
            return
        with self._lock:
            self.engine = engine

    def synthesize(self, text: str) -> str:
        clean_text = strip_markdown(text)
        if not clean_text:
            raise RuntimeError("Cannot synthesize empty text")

        with self._lock:
            if self.engine == "coqui":
                return self._synthesize_coqui(clean_text)
            return self._synthesize_piper(clean_text)

    def _synthesize_piper(self, text: str) -> str:
        piper_bin = os.getenv("PIPER_BIN", "piper")
        piper_model = os.getenv("PIPER_MODEL_PATH")
        piper_config = os.getenv("PIPER_CONFIG_PATH")

        if not piper_model:
            raise RuntimeError("PIPER_MODEL_PATH is not set")

        fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="tts_")
        os.close(fd)

        cmd = [piper_bin, "--model", piper_model, "--output_file", out_path]
        if piper_config:
            cmd.extend(["--config", piper_config])

        try:
            proc = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            _ = proc
        except Exception as exc:
            try:
                Path(out_path).unlink(missing_ok=True)
            except Exception:
                pass
            raise RuntimeError(
                "Failed to synthesize with Piper. Ensure piper binary and model files are configured."
            ) from exc

        return out_path

    def _synthesize_coqui(self, text: str) -> str:
        if self._coqui is None:
            try:
                from TTS.api import TTS as CoquiTTS  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("Coqui TTS is not installed") from exc

            model_name = os.getenv("COQUI_MODEL_NAME", "tts_models/en/ljspeech/tacotron2-DDC")
            self._coqui = CoquiTTS(model_name=model_name, progress_bar=False, gpu=False)

        fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="tts_")
        os.close(fd)
        self._coqui.tts_to_file(text=text, file_path=out_path)
        return out_path
