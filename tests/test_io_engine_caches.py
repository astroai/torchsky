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


def test_named_cache_subsystem_policies_are_explicit_and_fail_closed() -> None:
    assert set(caches.IO_CACHE_SUBSYSTEMS) == {
        "all",
        "fits_header_metadata",
        "fits_image_data",
        "fits_table_data",
        "fits_wcs_extraction",
    }
    image_policy = caches.cache_subsystem_policy("fits_image_data")
    table_policy = caches.cache_subsystem_policy("fits_table_data")
    assert image_policy["data"] is True
    assert image_policy["meta"] is False
    assert table_policy["table_handles"] is True

    try:
        caches.cache_subsystem_policy("unknown")
    except KeyError as exc:
        assert "unknown FITS I/O cache subsystem" in str(exc)
    else:
        raise AssertionError("unknown cache subsystem did not fail closed")


def test_clear_named_cache_subsystem_targets_expected_buckets() -> None:
    handle = _FakeHandle()
    caches.file_cache[("a.fits", 0)] = object()
    caches.file_handle_cache["a.fits"] = handle
    caches.image_meta_cache[("a.fits", 0)] = object()
    caches.hdu_type_cache[("a.fits", 0)] = "IMAGE"

    caches.clear_cache_subsystem("fits_header_metadata")

    assert caches.file_cache
    assert caches.file_handle_cache
    assert not handle.closed
    assert not caches.image_meta_cache
    assert not caches.hdu_type_cache

    caches.clear_cache_subsystem("fits_image_data")

    assert not caches.file_cache
    assert caches.file_handle_cache
    assert not handle.closed

    caches.clear_cache_subsystem("all", cpp_module=object())

    assert not caches.file_handle_cache
    assert handle.closed
