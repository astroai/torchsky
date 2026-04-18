"""LRU tile cache with byte-level size tracking."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Optional


class LRUTileCache:
    """
    Thread-safe LRU cache for PNG tile bytes.

    Keys are arbitrary hashable objects (typically (survey_id, order, pix) tuples).
    Evicts least-recently-used entries when max_bytes is exceeded.
    """

    def __init__(self, max_mb: int = 512) -> None:
        self._max_bytes = max_mb * 1024 * 1024
        self._cache: OrderedDict[object, bytes] = OrderedDict()
        self._size_bytes: int = 0
        self._lock = threading.Lock()
        self.hits: int = 0
        self.misses: int = 0

    def get(self, key: object) -> Optional[bytes]:
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]

    def put(self, key: object, value: bytes) -> None:
        with self._lock:
            if key in self._cache:
                self._size_bytes -= len(self._cache[key])
                del self._cache[key]
            self._cache[key] = value
            self._size_bytes += len(value)
            self._cache.move_to_end(key)
            # Evict LRU entries until under budget
            while self._size_bytes > self._max_bytes and self._cache:
                _, oldest = self._cache.popitem(last=False)
                self._size_bytes -= len(oldest)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._size_bytes = 0

    @property
    def size_mb(self) -> float:
        return self._size_bytes / (1024 * 1024)

    @property
    def n_entries(self) -> int:
        return len(self._cache)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "size_mb": round(self.size_mb, 2),
            "n_entries": self.n_entries,
        }
