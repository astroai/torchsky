from __future__ import annotations

from torchsky._io_engine import caches


class _FakeHandle:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_clear_python_caches_resets_data_handles_meta_and_stats() -> None:
    handle = _FakeHandle()
    caches.file_cache[("a.fits", 0)] = object()
    caches.file_handle_cache["a.fits"] = handle
    caches.file_handle_sig_cache["a.fits"] = (1, 2, 3)
    caches.image_meta_cache[("a.fits", 0)] = object()
    caches.hdu_type_cache[("a.fits", 0)] = "IMAGE"
    caches.cold_nommap_cache[("a.fits", 0)] = True
    caches.auto_mmap_cache[("a.fits", 0)] = False
    caches.auto_hdu_cache[("a.fits", "payload")] = ((1, 2, 3), 0)
    caches.cache_stats.update(
        {"total_requests": 3, "hits": 1, "misses": 2, "cache_size": 1}
    )

    caches.clear_python_caches()

    assert handle.closed
    assert not caches.file_cache
    assert not caches.file_handle_cache
    assert not caches.file_handle_sig_cache
    assert not caches.image_meta_cache
    assert not caches.hdu_type_cache
    assert not caches.cold_nommap_cache
    assert not caches.auto_mmap_cache
    assert not caches.auto_hdu_cache
    assert caches.get_cache_performance() == {
        "cache_size": 0,
        "hit_rate": 0.0,
        "miss_rate": 0.0,
        "total_requests": 0,
        "hits": 0,
        "misses": 0,
    }
