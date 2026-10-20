from __future__ import annotations

import sys
import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterator, Mapping
from dataclasses import dataclass
from typing import Generic, NamedTuple, TypeVar, cast

import torch
from torch import Tensor


T = TypeVar("T")
_MISSING = object()


@dataclass(frozen=True)
class SpectralCacheInfo:
    hits: int
    misses: int
    waits: int
    builds: int
    failures: int
    rejected: int
    evictions: int
    entries: int
    total_bytes: int
    max_bytes: int
    max_entry_bytes: int
    generation: int


class _CacheKey(NamedTuple):
    namespace: str
    key: Hashable
    device: str
    dtype: str


@dataclass(frozen=True)
class _CacheEntry(Generic[T]):
    value: T
    size_bytes: int
    generation: int


class _InFlight(Generic[T]):
    def __init__(self, generation: int) -> None:
        self.generation = generation
        self.owner = threading.get_ident()
        self.event = threading.Event()
        self.value: T | object = _MISSING
        self.error: BaseException | None = None


def _device_tag(device: torch.device | str | None) -> str:
    return str(torch.device("cpu" if device is None else device))


def _dtype_tag(dtype: object | None) -> str:
    if dtype is None:
        return "none"
    if isinstance(dtype, tuple):
        return "(" + ",".join(_dtype_tag(item) for item in dtype) + ")"
    return str(dtype)


def value_nbytes(value: object) -> int:
    seen_objects: set[int] = set()
    seen_storages: set[tuple[str, int | None, int, int]] = set()

    def measure(item: object) -> int:
        if isinstance(item, Tensor):
            storage = item.untyped_storage()
            storage_bytes = int(storage.nbytes())
            try:
                storage_address = int(storage.data_ptr())
            except RuntimeError:
                storage_address = id(storage)
            storage_key = (
                item.device.type,
                item.device.index,
                storage_address,
                storage_bytes,
            )
            payload = 0
            if storage_key not in seen_storages:
                seen_storages.add(storage_key)
                payload = storage_bytes
            object_id = id(item)
            if object_id in seen_objects:
                return payload
            seen_objects.add(object_id)
            return int(sys.getsizeof(item)) + payload

        object_id = id(item)
        if object_id in seen_objects:
            return 0
        seen_objects.add(object_id)
        size = int(sys.getsizeof(item))
        if isinstance(item, Mapping):
            return size + sum(measure(key) + measure(val) for key, val in item.items())
        if isinstance(item, (tuple, list)):
            return size + sum(measure(val) for val in item)
        return size

    return measure(value)


class SpectralCacheManager:
    def __init__(self, *, max_bytes: int, max_entry_bytes: int) -> None:
        if max_bytes < 0 or max_entry_bytes < 0:
            raise ValueError("spectral cache byte limits must be non-negative")
        self._max_bytes = int(max_bytes)
        self._max_entry_bytes = min(int(max_entry_bytes), self._max_bytes)
        self._lock = threading.RLock()
        self._entries: OrderedDict[_CacheKey, _CacheEntry[object]] = OrderedDict()
        self._inflight: dict[_CacheKey, _InFlight[object]] = {}
        self._total_bytes = 0
        self._generation = 0
        self._hits = 0
        self._misses = 0
        self._waits = 0
        self._builds = 0
        self._failures = 0
        self._rejected = 0
        self._evictions = 0

    def namespace(
        self,
        name: str,
        *,
        device: torch.device | str | None = None,
        dtype: object | None = None,
        max_entry_bytes: int | None = None,
    ) -> SpectralCacheNamespace[T]:
        return SpectralCacheNamespace[T](
            self,
            name,
            device=device,
            dtype=dtype,
            max_entry_bytes=max_entry_bytes,
        )

    def _full_key(
        self,
        namespace: str,
        key: Hashable,
        device: torch.device | str | None,
        dtype: object | None,
    ) -> _CacheKey:
        return _CacheKey(namespace, key, _device_tag(device), _dtype_tag(dtype))

    def get_or_build(
        self,
        namespace: str,
        key: Hashable,
        builder: Callable[[], T],
        *,
        device: torch.device | str | None = None,
        dtype: object | None = None,
        estimated_bytes: int | None = None,
        max_entry_bytes: int | None = None,
    ) -> T:
        full_key = self._full_key(namespace, key, device, dtype)
        return self._get_or_build(
            full_key,
            builder,
            estimated_bytes=estimated_bytes,
            max_entry_bytes=max_entry_bytes,
        )

    def _get_or_build(
        self,
        full_key: _CacheKey,
        builder: Callable[[], T],
        *,
        estimated_bytes: int | None,
        max_entry_bytes: int | None,
    ) -> T:
        with self._lock:
            entry = self._entries.get(full_key)
            if entry is not None:
                self._entries.move_to_end(full_key)
                self._hits += 1
                return cast(T, entry.value)
            generation = self._generation
            flight = self._inflight.get(full_key)
            if flight is not None and flight.generation == generation:
                if flight.owner == threading.get_ident():
                    raise RuntimeError(f"recursive spectral cache build for {full_key}")
                self._waits += 1
                build = False
            else:
                flight = _InFlight[object](generation)
                self._inflight[full_key] = flight
                self._misses += 1
                build = True

        if not build:
            flight.event.wait()
            if flight.error is not None:
                raise flight.error
            if flight.value is _MISSING:
                raise RuntimeError("spectral cache construction completed without a value")
            return cast(T, flight.value)

        try:
            value = builder()
            size_bytes = value_nbytes(value)
        except BaseException as exc:
            with self._lock:
                self._failures += 1
                flight.error = exc
                if self._inflight.get(full_key) is flight:
                    del self._inflight[full_key]
                flight.event.set()
            raise

        limit = self._max_entry_bytes
        if max_entry_bytes is not None:
            limit = min(limit, max(0, int(max_entry_bytes)))
        estimate_too_large = estimated_bytes is not None and int(estimated_bytes) > limit

        with self._lock:
            self._builds += 1
            flight.value = value
            current_flight = self._inflight.get(full_key)
            can_publish = (
                current_flight is flight
                and self._generation == generation
                and not estimate_too_large
                and size_bytes <= limit
                and size_bytes <= self._max_bytes
            )
            if can_publish:
                while self._entries and self._total_bytes + size_bytes > self._max_bytes:
                    _, evicted = self._entries.popitem(last=False)
                    self._total_bytes -= evicted.size_bytes
                    self._evictions += 1
                self._entries[full_key] = _CacheEntry(value, size_bytes, generation)
                self._total_bytes += size_bytes
            elif self._generation == generation and current_flight is flight:
                self._rejected += 1
            if current_flight is flight:
                del self._inflight[full_key]
            flight.event.set()
        return value

    def get(
        self,
        namespace: str,
        key: Hashable,
        *,
        device: torch.device | str | None = None,
        dtype: object | None = None,
        default: T | None = None,
    ) -> object | T | None:
        full_key = self._full_key(namespace, key, device, dtype)
        return self._get(full_key, default)

    def _get(self, full_key: _CacheKey, default: T | None) -> object | T | None:
        with self._lock:
            entry = self._entries.get(full_key)
            if entry is None:
                return default
            self._entries.move_to_end(full_key)
            self._hits += 1
            return entry.value

    def clear(self) -> None:
        with self._lock:
            self._generation += 1
            self._entries.clear()
            self._inflight.clear()
            self._total_bytes = 0

    def clear_namespace(self, namespace: str) -> None:
        with self._lock:
            self._generation += 1
            retained: OrderedDict[_CacheKey, _CacheEntry[object]] = OrderedDict()
            total_bytes = 0
            for key, entry in self._entries.items():
                if key.namespace != namespace:
                    retained[key] = entry
                    total_bytes += entry.size_bytes
            self._entries = retained
            self._inflight = {
                key: flight
                for key, flight in self._inflight.items()
                if key.namespace != namespace
            }
            self._total_bytes = total_bytes

    def info(self) -> SpectralCacheInfo:
        with self._lock:
            return SpectralCacheInfo(
                hits=self._hits,
                misses=self._misses,
                waits=self._waits,
                builds=self._builds,
                failures=self._failures,
                rejected=self._rejected,
                evictions=self._evictions,
                entries=len(self._entries),
                total_bytes=self._total_bytes,
                max_bytes=self._max_bytes,
                max_entry_bytes=self._max_entry_bytes,
                generation=self._generation,
            )

    def _namespace_items(
        self,
        namespace: str,
        device: torch.device | str | None,
        dtype: object | None,
    ) -> tuple[tuple[Hashable, object], ...]:
        device_tag = _device_tag(device)
        dtype_tag = _dtype_tag(dtype)
        with self._lock:
            return tuple(
                (key.key, entry.value)
                for key, entry in self._entries.items()
                if key.namespace == namespace
                and key.device == device_tag
                and key.dtype == dtype_tag
            )


class SpectralCacheNamespace(Mapping[Hashable, T], Generic[T]):
    def __init__(
        self,
        manager: SpectralCacheManager,
        name: str,
        *,
        device: torch.device | str | None,
        dtype: object | None,
        max_entry_bytes: int | None,
    ) -> None:
        self._manager = manager
        self._name = str(name)
        self._device = device
        self._dtype = dtype
        self._device_tag = _device_tag(device)
        self._dtype_tag = _dtype_tag(dtype)
        self._max_entry_bytes = max_entry_bytes

    def _full_key(self, key: Hashable) -> _CacheKey:
        return _CacheKey(self._name, key, self._device_tag, self._dtype_tag)

    def get_or_build(
        self,
        key: Hashable,
        builder: Callable[[], T],
        *,
        estimated_bytes: int | None = None,
    ) -> T:
        return self._manager._get_or_build(
            self._full_key(key),
            builder,
            estimated_bytes=estimated_bytes,
            max_entry_bytes=self._max_entry_bytes,
        )

    def __getitem__(self, key: Hashable) -> T:
        value = self._manager._get(self._full_key(key), _MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return cast(T, value)

    def __iter__(self) -> Iterator[Hashable]:
        return iter(
            key
            for key, _ in self._manager._namespace_items(
                self._name, self._device, self._dtype
            )
        )

    def __len__(self) -> int:
        return len(
            self._manager._namespace_items(self._name, self._device, self._dtype)
        )

    def clear(self) -> None:
        self._manager.clear_namespace(self._name)
