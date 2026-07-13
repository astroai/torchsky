"""Compatibility facade for cache controls now owned by :mod:`torchfits.cache`."""

from torchfits.cache import (
    CacheConfig,
    CacheManager,
    clear,
    clear_cache,
    configure_cache,
    configure_for_environment,
    get_cache_manager,
    get_cache_stats,
    get_optimal_cache_config,
    optimize_for_dataset,
    stats,
)

__all__ = [
    "CacheConfig",
    "CacheManager",
    "clear",
    "clear_cache",
    "configure_cache",
    "configure_for_environment",
    "get_cache_manager",
    "get_cache_stats",
    "get_optimal_cache_config",
    "optimize_for_dataset",
    "stats",
]
