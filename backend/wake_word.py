from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class WakeWordEvent:
    event_id: int
    ts: float
    phrase: str


class WakeWordService:
    """Background wake-word detection using openWakeWord in a lightweight thread."""

    def __init__(self, wake_word_text: str = "alexa") -> None:
        self.wake_word_text = wake_word_text.lower()
        self.enabled = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._events: deque[WakeWordEvent] = deque(maxlen=128)
        self._next_event_id = 1
        self._lock = threading.Lock()
        self._status_message = "Wake word disabled"
        self._threshold = 0.7

    def configure(self, enabled: bool, wake_word_text: str | None = None) -> dict[str, Any]:
        if wake_word_text:
            self.wake_word_text = wake_word_text.lower()

        if enabled:
            self.start()
        else:
            self.stop()

        return self.status()

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "wake_word_text": self.wake_word_text,
            "status": self._status_message,
        }

    def poll(self, after_id: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "id": evt.event_id,
                    "timestamp": evt.ts,
                    "wake_word": evt.phrase,
                }
                for evt in self._events
                if evt.event_id > after_id
            ]

    def start(self) -> None:
        if self.enabled:
            return

        if self._thread and self._thread.is_alive():
            self.enabled = True
            return

        self.enabled = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="wake-word-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.enabled = False
        self._stop_event.set()

    def _run_loop(self) -> None:
        try:
            import openwakeword
            from openwakeword.model import Model
            import sounddevice as sd
        except Exception:
            self._status_message = "openWakeWord or sounddevice packages not installed"
            self.enabled = False
            return

        # Initialize model
        # Map user-friendly names to openwakeword models if possible
        model_name = self.wake_word_text
        if model_name == "hey smart":
            model_name = "alexa" # Fallback to alexa for now, or user can provide custom path
        
        try:
            # Check if it's a valid pretrained model name
            available_models = [os.path.basename(p).split('_v')[0] for p in openwakeword.get_pretrained_model_paths()]
            if model_name not in available_models:
                # Try to find closest match or fallback
                model_name = "alexa"

            oww_model = Model(wakeword_models=[model_name], inference_framework='tflite')
            self._status_message = f"Listening for '{model_name}'"
            
            # Audio parameters
            RATE = 16000
            CHUNK_SIZE = 1280 # 80ms at 16kHz
            
            def audio_callback(indata, frames, callback_time, status):
                if not self.enabled or self._stop_event.is_set():
                    return
                
                if status:
                    print(f"[wake-word] Status: {status}")
                
                # openwakeword expects int16 numpy array
                audio_data = indata.flatten()
                
                # Predict
                prediction = oww_model.predict(audio_data)
                
                # Check detections
                for phrase, score in prediction.items():
                    if score > self._threshold:
                        with self._lock:
                            evt = WakeWordEvent(
                                event_id=self._next_event_id,
                                ts=time.time(),
                                phrase=phrase,
                            )
                            self._events.append(evt)
                            self._next_event_id += 1
                        print(f"[wake-word] Detected '{phrase}' with score {score:.2f}")

            with sd.InputStream(samplerate=RATE, channels=1, dtype='int16', 
                              blocksize=CHUNK_SIZE, callback=audio_callback):
                while not self._stop_event.is_set():
                    time.sleep(0.5)

        except Exception as exc:
            self._status_message = f"Wake word error: {type(exc).__name__}: {str(exc)}"
        finally:
            if self.enabled:
                self._status_message = "Wake word stopped"
            self.enabled = False
