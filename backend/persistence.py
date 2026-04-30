import json
import threading
from pathlib import Path


class JsonPersistence:
    """Simple thread-safe JSON persistence for small app state."""

    def __init__(self, file_path: str, default_data):
        self.path = Path(file_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.default_data = default_data
        self._lock = threading.Lock()

        if not self.path.exists():
            self.write(self.default_data)

    def read(self):
        with self._lock:
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                return self.default_data

    def write(self, data):
        with self._lock:
            self.path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
