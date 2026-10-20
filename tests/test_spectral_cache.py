import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from torchsky.sphere._spectral_cache import SpectralCacheManager, value_nbytes
import torchsky.sphere.spectral as spectral


def _wait_for_waiters(manager: SpectralCacheManager, count: int) -> None:
    deadline = time.monotonic() + 5.0
    while manager.info().waits < count:
        if time.monotonic() >= deadline:
            raise AssertionError(f"expected {count} cache waiters")
        time.sleep(0.001)


def test_same_key_construction_is_single_flight() -> None:
    manager = SpectralCacheManager(max_bytes=1 << 20, max_entry_bytes=1 << 20)
    started = threading.Event()
    release = threading.Event()
    build_count = 0
    count_lock = threading.Lock()

    def build() -> torch.Tensor:
        nonlocal build_count
        with count_lock:
            build_count += 1
        started.set()
        assert release.wait(timeout=5)
        return torch.arange(8, dtype=torch.float64)

    with ThreadPoolExecutor(max_workers=8) as executor:
        first = executor.submit(manager.get_or_build, "basis", 1, build)
        assert started.wait(timeout=5)
        others = [
            executor.submit(manager.get_or_build, "basis", 1, build)
            for _ in range(7)
        ]
        _wait_for_waiters(manager, 7)
        release.set()
        values = [first.result(timeout=5)] + [future.result(timeout=5) for future in others]

    assert build_count == 1
    assert all(value is values[0] for value in values)
    assert manager.info().waits == 7


def test_different_keys_build_concurrently() -> None:
    manager = SpectralCacheManager(max_bytes=1 << 20, max_entry_bytes=1 << 20)
    barrier = threading.Barrier(2, timeout=5)

    def build(value: int) -> torch.Tensor:
        barrier.wait()
        return torch.tensor([value])

    with ThreadPoolExecutor(max_workers=2) as executor:
        one = executor.submit(manager.get_or_build, "layout", 1, lambda: build(1))
        two = executor.submit(manager.get_or_build, "layout", 2, lambda: build(2))
        assert int(one.result(timeout=5).item()) == 1
        assert int(two.result(timeout=5).item()) == 2


def test_clear_during_build_does_not_publish_stale_generation() -> None:
    manager = SpectralCacheManager(max_bytes=1 << 20, max_entry_bytes=1 << 20)
    started = threading.Event()
    release = threading.Event()

    def build_old() -> torch.Tensor:
        started.set()
        assert release.wait(timeout=5)
        return torch.tensor([1.0])

    with ThreadPoolExecutor(max_workers=1) as executor:
        old_future = executor.submit(manager.get_or_build, "angles", 4, build_old)
        assert started.wait(timeout=5)
        manager.clear()
        current = manager.get_or_build("angles", 4, lambda: torch.tensor([2.0]))
        release.set()
        old = old_future.result(timeout=5)

    cached = manager.get_or_build("angles", 4, lambda: torch.tensor([3.0]))
    torch.testing.assert_close(old, torch.tensor([1.0]))
    torch.testing.assert_close(current, torch.tensor([2.0]))
    assert cached is current
    assert manager.info().entries == 1


def test_byte_budget_and_lru_eviction_are_deterministic() -> None:
    sample = torch.arange(16, dtype=torch.float64)
    entry_bytes = value_nbytes(sample)
    manager = SpectralCacheManager(
        max_bytes=2 * entry_bytes, max_entry_bytes=entry_bytes
    )
    cache = manager.namespace("index", device="cpu", dtype=torch.float64)

    a = cache.get_or_build("a", lambda: sample.clone())
    cache.get_or_build("b", lambda: sample.clone())
    assert cache["a"] is a
    cache.get_or_build("c", lambda: sample.clone())

    assert list(cache) == ["a", "c"]
    assert manager.info().evictions == 1
    assert manager.info().total_bytes <= manager.info().max_bytes


def test_oversized_entry_is_returned_but_not_cached() -> None:
    sample = torch.arange(32, dtype=torch.float64)
    manager = SpectralCacheManager(
        max_bytes=value_nbytes(sample) * 2,
        max_entry_bytes=value_nbytes(sample) - 1,
    )
    builds = 0

    def build() -> torch.Tensor:
        nonlocal builds
        builds += 1
        return sample.clone()

    manager.get_or_build("dense", 1, build)
    manager.get_or_build("dense", 1, build)

    assert builds == 2
    assert manager.info().entries == 0
    assert manager.info().rejected == 2


def test_failed_build_wakes_waiters_and_allows_retry() -> None:
    manager = SpectralCacheManager(max_bytes=1 << 20, max_entry_bytes=1 << 20)
    started = threading.Event()
    release = threading.Event()
    build_count = 0

    def fail() -> torch.Tensor:
        nonlocal build_count
        build_count += 1
        started.set()
        assert release.wait(timeout=5)
        raise RuntimeError("injected failure")

    with ThreadPoolExecutor(max_workers=4) as executor:
        first = executor.submit(manager.get_or_build, "failure", 1, fail)
        assert started.wait(timeout=5)
        others = [
            executor.submit(manager.get_or_build, "failure", 1, fail)
            for _ in range(3)
        ]
        _wait_for_waiters(manager, 3)
        release.set()
        for future in [first, *others]:
            with pytest.raises(RuntimeError, match="injected failure"):
                future.result(timeout=5)

    value = manager.get_or_build("failure", 1, lambda: torch.tensor([4.0]))
    torch.testing.assert_close(value, torch.tensor([4.0]))
    assert build_count == 1
    assert manager.info().failures == 1


def test_device_and_dtype_are_part_of_cache_key() -> None:
    manager = SpectralCacheManager(max_bytes=1 << 20, max_entry_bytes=1 << 20)
    key = (8, 7)
    f32 = manager.get_or_build(
        "phase", key, lambda: torch.ones(2, dtype=torch.float32), dtype=torch.float32
    )
    f64 = manager.get_or_build(
        "phase", key, lambda: torch.ones(2, dtype=torch.float64), dtype=torch.float64
    )
    meta = manager.get_or_build(
        "phase",
        key,
        lambda: torch.ones(2, dtype=torch.float32, device="meta"),
        device="meta",
        dtype=torch.float32,
    )

    assert f32.dtype == torch.float32
    assert f64.dtype == torch.float64
    assert meta.device.type == "meta"
    assert manager.info().entries == 3


def test_clear_preserves_in_use_snapshot_and_resets_accounting() -> None:
    manager = SpectralCacheManager(max_bytes=1 << 20, max_entry_bytes=1 << 20)
    value = manager.get_or_build("basis", 1, lambda: torch.arange(6))
    snapshot = value.clone()

    manager.clear()

    torch.testing.assert_close(value, snapshot)
    assert manager.info().entries == 0
    assert manager.info().total_bytes == 0


def test_spectral_results_are_deterministic_across_cache_clear() -> None:
    nside = 4
    lmax = 3
    generator = torch.Generator().manual_seed(742)
    sky = torch.randn(12 * nside * nside, generator=generator, dtype=torch.float64)

    spectral.clear_spectral_caches()
    alm_first = spectral.map2alm(sky, nside=nside, lmax=lmax, iter=0)
    map_first = spectral.alm2map(alm_first, nside=nside, lmax=lmax)
    spectral.clear_spectral_caches()
    alm_second = spectral.map2alm(sky, nside=nside, lmax=lmax, iter=0)
    map_second = spectral.alm2map(alm_second, nside=nside, lmax=lmax)

    torch.testing.assert_close(alm_first, alm_second, atol=0.0, rtol=0.0)
    torch.testing.assert_close(map_first, map_second, atol=0.0, rtol=0.0)
