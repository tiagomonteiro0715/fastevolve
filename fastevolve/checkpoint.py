import os
import pickle
import struct
import threading
from pathlib import Path
from queue import Queue

from .telemetry import log

_SENTINEL = object()


class Checkpointer:
    """Append-only background log: O(1) main-thread cost per iteration."""

    def __init__(self, path: str | None):
        self.path = Path(path) if path else None
        self._q: Queue = Queue()
        self._thread: threading.Thread | None = None
        if self.path:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def append(self, record) -> None:
        if self.path:
            self._q.put(record)

    def close(self) -> None:
        if self._thread:
            self._q.put(_SENTINEL)
            self._thread.join(timeout=5)

    def load(self) -> list:
        records: list = []
        if not (self.path and self.path.exists()):
            return records
        try:
            with open(self.path, "rb") as f:
                while True:
                    hdr = f.read(4)
                    if len(hdr) < 4:
                        break
                    (n,) = struct.unpack("!I", hdr)
                    records.append(pickle.loads(f.read(n)))
        except Exception:
            log.exception("checkpoint load failed: %s", self.path)
        return records

    def _worker(self) -> None:
        try:
            with open(self.path, "ab") as f:
                while True:
                    rec = self._q.get()
                    if rec is _SENTINEL:
                        return
                    data = pickle.dumps(rec, protocol=pickle.HIGHEST_PROTOCOL)
                    f.write(struct.pack("!I", len(data)))
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
        except Exception:
            log.exception("checkpoint worker died")
