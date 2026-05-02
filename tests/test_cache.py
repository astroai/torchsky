"""
Test caching functionality.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

import torchsky
from torchsky.cache import CacheConfig


class TestCaching:
    """Test file caching functionality."""

    def create_test_fits(self, shape=(100, 100)):
        """Create a test FITS file."""
        data = np.random.normal(0, 1, shape).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            hdu = fits.PrimaryHDU(data)
            hdu.writeto(f.name, overwrite=True)
            return f.name, data

    def test_cache_performance_tracking(self):
        """Test cache hit/miss tracking."""
        filepath, _ = self.create_test_fits()

        try:
            # Clear cache first
            torchsky.clear_file_cache()

            # First read should be a cache miss
            torchsky.read(filepath)
            stats1 = torchsky.get_cache_performance()

            # Second read should be a cache hit
            torchsky.read(filepath)
            stats2 = torchsky.get_cache_performance()

            # Verify cache behavior
            assert stats2["total_requests"] > stats1["total_requests"]

        finally:
            os.unlink(filepath)
            torchsky.clear_file_cache()

    def test_get_cache_stats(self):
        """Test get_cache_stats returns expected dictionary structure."""
        from torchsky.cache import get_cache_stats, clear_cache

        # Clear cache to start with a known state
        clear_cache()

        stats = get_cache_stats()

        # Verify it's a dictionary
        assert isinstance(stats, dict)

        # Check for expected keys
        expected_keys = {
            "hits",
            "misses",
            "evictions",
            "memory_usage_mb",
            "disk_usage_gb",
            "cpp_cache_size",
            "config",
            "hit_rate",
        }
        assert expected_keys.issubset(stats.keys())

        # Verify types of specific fields
        assert isinstance(stats["hits"], int)
        assert isinstance(stats["misses"], int)
        assert isinstance(stats["hit_rate"], float)
        assert isinstance(stats["config"], dict)

        # Check config keys
        expected_config_keys = {
            "max_files",
            "max_memory_mb",
            "disk_cache_gb",
            "prefetch_enabled",
        }
        assert expected_config_keys.issubset(stats["config"].keys())

        # Basic hit_rate calculation check (should be 0.0 when hits=0, misses=0)
        assert stats["hit_rate"] == 0.0

    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        filepath, _ = self.create_test_fits()

        try:
            # Read file to populate cache
            torchsky.read(filepath)

            # Clear cache
            torchsky.clear_file_cache()

            # Verify cache is cleared
            torchsky.get_cache_performance()
            # After clearing, stats should be reset or show no cached entries

        finally:
            os.unlink(filepath)

    def test_multiple_file_caching(self):
        """Test caching with multiple files."""
        files = []

        try:
            # Create multiple test files
            for i in range(5):
                filepath, _ = self.create_test_fits((50 + i * 10, 50 + i * 10))
                files.append(filepath)

            # Clear cache
            torchsky.clear_file_cache()

            # Read all files
            for filepath in files:
                torchsky.read(filepath)

            # Read them again (should hit cache)
            for filepath in files:
                torchsky.read(filepath)

            # Check cache performance
            stats = torchsky.get_cache_performance()
            assert stats["total_requests"] >= len(files) * 2

        finally:
            for f in files:
                if os.path.exists(f):
                    os.unlink(f)
            torchsky.clear_file_cache()

    def test_cached_numpy_reads_survive_repeated_cache_clears(self):
        """Regression: cached numpy reads should remain stable across cache clears."""
        cpp = pytest.importorskip("torchsky.cpp")
        if not hasattr(cpp, "read_full_numpy_cached"):
            pytest.skip("read_full_numpy_cached unavailable in this build")

        file_a, _ = self.create_test_fits((257,))
        file_b, _ = self.create_test_fits((33, 17))

        try:
            for i in range(300):
                torchsky.clear_file_cache()
                if i % 2 == 0:
                    arr = cpp.read_full_numpy_cached(file_a, 0, True)
                    assert arr.shape == (257,)
                else:
                    arr = cpp.read_full_numpy_cached(file_b, 0, True)
                    assert arr.shape == (33, 17)
        finally:
            for path in (file_a, file_b):
                if os.path.exists(path):
                    os.unlink(path)
            torchsky.clear_file_cache()

    def test_cached_multibyte_read_matches_nocache_reference(self):
        """Regression: cached mmap raw path must match the no-cache path exactly."""
        cpp = pytest.importorskip("torchsky.cpp")
        if not hasattr(cpp, "read_full_cached") or not hasattr(
            cpp, "read_full_nocache"
        ):
            pytest.skip("cached/nocache read methods unavailable in this build")

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            data = np.arange(128 * 64, dtype=np.int32).reshape(128, 64) - 1000
            fits.PrimaryHDU(data).writeto(f.name, overwrite=True)
            path = f.name

        try:
            torchsky.clear_file_cache()
            cached = cpp.read_full_cached(path, 0, True).numpy()
            reference = cpp.read_full_nocache(path, 0, True).numpy()
            np.testing.assert_array_equal(cached, reference)
        finally:
            if os.path.exists(path):
                os.unlink(path)
            torchsky.clear_file_cache()

    def test_cold_nommap_heuristic_with_cache_enabled_int16(self):
        """Large int16 images should prefer non-mmap even when cache is enabled."""
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            data = np.arange(1024 * 1024, dtype=np.int16).reshape(1024, 1024)
            fits.PrimaryHDU(data).writeto(f.name, overwrite=True)
            path = f.name

        try:
            torchsky.clear_file_cache()
            assert torchsky._should_use_cold_nommap(
                path, 0, cache_capacity=10, mmap=True
            )
        finally:
            if os.path.exists(path):
                os.unlink(path)
            torchsky.clear_file_cache()

    def test_cold_nommap_heuristic_float64_and_small_guard(self):
        """64-bit and sub-1MiB payloads should keep mmap enabled by default."""
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f_large:
            from astropy.io import fits

            large = np.random.randn(1024, 1024).astype(np.float64)  # ~8 MiB payload
            fits.PrimaryHDU(large).writeto(f_large.name, overwrite=True)
            large_path = f_large.name
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f_small:
            from astropy.io import fits

            small = np.random.randn(128, 128).astype(np.float64)  # <1 MiB payload
            fits.PrimaryHDU(small).writeto(f_small.name, overwrite=True)
            small_path = f_small.name

        try:
            torchsky.clear_file_cache()
            assert not torchsky._should_use_cold_nommap(
                large_path, 0, cache_capacity=10, mmap=True
            )
            assert not torchsky._should_use_cold_nommap(
                small_path, 0, cache_capacity=10, mmap=True
            )
        finally:
            for path in (large_path, small_path):
                if os.path.exists(path):
                    os.unlink(path)
            torchsky.clear_file_cache()

    def test_read_mmap_auto_defaults_to_false_for_compressed(self, monkeypatch):
        """In smart policy, `mmap='auto'` should disable mmap for compressed HDUs."""
        cpp = pytest.importorskip("torchsky.cpp")
        if not hasattr(cpp, "read_full_cached"):
            pytest.skip("read_full_cached unavailable in this build")

        observed = []

        monkeypatch.setattr(
            torchsky,
            "_get_image_meta",
            lambda path, hdu: (-32, 2, (64, 64), 1.0, 0.0, True),
        )
        monkeypatch.setattr(
            torchsky,
            "_should_use_cold_nommap",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError(
                    "cold nommap heuristic should not run for compressed auto mode"
                )
            ),
        )
        monkeypatch.setattr(
            cpp,
            "read_full_cached",
            lambda path, hdu, use_mmap: (
                observed.append(bool(use_mmap)),
                torch.zeros((8, 8), dtype=torch.float32),
            )[1],
        )
        monkeypatch.setattr(
            cpp,
            "read_full",
            lambda path, hdu, use_mmap: (
                observed.append(bool(use_mmap)),
                torch.zeros((8, 8), dtype=torch.float32),
            )[1],
        )

        out = torchsky.read(
            "synthetic_compressed.fits",
            hdu=1,
            mmap="auto",
            policy="smart",
            cache_capacity=10,
            handle_cache_capacity=16,
        )

        assert isinstance(out, torch.Tensor)
        assert observed == [False]

    def test_read_mmap_true_is_explicit_override(self, monkeypatch):
        """In smart policy, `mmap=True` must stay enabled for compressed metadata."""
        cpp = pytest.importorskip("torchsky.cpp")
        if not hasattr(cpp, "read_full_cached"):
            pytest.skip("read_full_cached unavailable in this build")

        observed = []

        monkeypatch.setattr(
            torchsky,
            "_get_image_meta",
            lambda path, hdu: (-32, 2, (64, 64), 1.0, 0.0, True),
        )
        monkeypatch.setattr(
            torchsky,
            "_should_use_cold_nommap",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError(
                    "cold nommap heuristic should not run for explicit mmap=True"
                )
            ),
        )
        monkeypatch.setattr(
            cpp,
            "read_full_cached",
            lambda path, hdu, use_mmap: (
                observed.append(bool(use_mmap)),
                torch.zeros((8, 8), dtype=torch.float32),
            )[1],
        )
        monkeypatch.setattr(
            cpp,
            "read_full",
            lambda path, hdu, use_mmap: (
                observed.append(bool(use_mmap)),
                torch.zeros((8, 8), dtype=torch.float32),
            )[1],
        )

        out = torchsky.read(
            "synthetic_compressed.fits",
            hdu=1,
            mmap=True,
            policy="smart",
            cache_capacity=10,
            handle_cache_capacity=16,
        )

        assert isinstance(out, torch.Tensor)
        assert observed == [True]

    def test_read_mmap_auto_default_policy_uses_direct_mmap(self, monkeypatch):
        """Default policy should skip smart auto-mmap heuristics."""
        cpp = pytest.importorskip("torchsky.cpp")

        observed = []
        monkeypatch.setattr(
            cpp,
            "read_full",
            lambda path, hdu, use_mmap: (
                observed.append(bool(use_mmap)),
                torch.zeros((4, 4), dtype=torch.float32),
            )[1],
        )
        if hasattr(cpp, "read_full_cached"):
            monkeypatch.setattr(
                cpp,
                "read_full_cached",
                lambda path, hdu, use_mmap: (
                    observed.append(bool(use_mmap)),
                    torch.zeros((4, 4), dtype=torch.float32),
                )[1],
            )

        out = torchsky.read(
            "synthetic_direct.fits",
            hdu=1,
            mmap="auto",
            policy="default",
            cache_capacity=10,
            handle_cache_capacity=16,
        )

        assert isinstance(out, torch.Tensor)
        assert observed == [True]

    def test_read_rejects_invalid_mmap_mode(self):
        """Only bool or 'auto' mmap mode should be accepted."""
        with pytest.raises(ValueError, match="mmap must be bool or 'auto'"):
            torchsky.read("dummy.fits", mmap="sometimes")

    def test_read_hdu_auto_detects_compressed_image_extension(self):
        """`hdu='auto'` should resolve to the first payload HDU (fitsio-like behavior)."""
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            image = np.random.normal(size=(64, 64)).astype(np.float32)
            primary = fits.PrimaryHDU()
            compressed = fits.CompImageHDU(image, compression_type="RICE_1")
            fits.HDUList([primary, compressed]).writeto(f.name, overwrite=True)
            path = f.name

        try:
            empty_primary = torchsky.read(path, hdu=0, mmap="auto")
            auto_tensor = torchsky.read(path, hdu="auto", mmap="auto")
            none_tensor = torchsky.read(path, hdu=None, mmap="auto")

            assert isinstance(empty_primary, torch.Tensor)
            assert empty_primary.numel() == 0
            assert isinstance(auto_tensor, torch.Tensor)
            assert isinstance(none_tensor, torch.Tensor)
            assert tuple(auto_tensor.shape) == image.shape
            assert tuple(none_tensor.shape) == image.shape
        finally:
            if os.path.exists(path):
                os.unlink(path)
            torchsky.clear_file_cache()

    def test_get_header_auto_matches_detected_hdu(self):
        """`get_header(..., hdu='auto')` should return the detected payload header."""
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            image = np.random.normal(size=(32, 48)).astype(np.float32)
            primary = fits.PrimaryHDU()
            compressed = fits.CompImageHDU(image, compression_type="RICE_1")
            fits.HDUList([primary, compressed]).writeto(f.name, overwrite=True)
            path = f.name

        try:
            header = torchsky.get_header(path, hdu="auto")
            assert int(header.get("NAXIS", 0)) == 2
            assert int(header.get("NAXIS1", 0)) > 0
            assert int(header.get("NAXIS2", 0)) > 0
        finally:
            if os.path.exists(path):
                os.unlink(path)
            torchsky.clear_file_cache()


class TestCacheManager:
    """Test CacheManager functionality."""

    def test_get_cache_manager_singleton(self):
        import torchsky.cache
        from torchsky.cache import get_cache_manager

        original_manager = torchsky.cache._cache_manager
        torchsky.cache._cache_manager = None

        try:
            with patch.object(
                torchsky.cache.CacheManager, "configure_cpp_cache"
            ) as mock_configure:
                manager1 = get_cache_manager()
                assert manager1 is not None
                mock_configure.assert_called_once()

                mock_configure.reset_mock()
                manager2 = get_cache_manager()

                assert manager1 is manager2
                mock_configure.assert_not_called()
        finally:
            torchsky.cache._cache_manager = original_manager

    def test_configure_cpp_cache_success(self):
        import torchsky.cache

        manager = torchsky.cache.CacheManager(
            torchsky.cache.CacheConfig(max_files=10, max_memory_mb=100)
        )

        with patch("torchsky.cpp.configure_cache") as mock_configure:
            manager.configure_cpp_cache()

        mock_configure.assert_called_once_with(10, 100)

    def test_configure_cpp_cache_missing_cpp_function(self):
        import torchsky.cache

        manager = torchsky.cache.CacheManager(
            torchsky.cache.CacheConfig(max_files=10, max_memory_mb=100)
        )
        mock_cpp = MagicMock(spec=[])

        with patch.dict("sys.modules", {"torchsky.cpp": mock_cpp}):
            manager.configure_cpp_cache()


class TestCacheConfig:
    """Test CacheConfig functionality."""

    def test_default_initialization(self):
        config = CacheConfig()
        assert config.max_files == 100
        assert config.max_memory_mb == 1024
        assert config.disk_cache_gb == 10
        assert config.prefetch_enabled is True

    def test_custom_initialization(self):
        config = CacheConfig(
            max_files=50, max_memory_mb=512, disk_cache_gb=5, prefetch_enabled=False
        )
        assert config.max_files == 50
        assert config.max_memory_mb == 512
        assert config.disk_cache_gb == 5
        assert config.prefetch_enabled is False

    @patch("torchsky.cache.psutil", None)
    def test_for_environment_no_psutil(self):
        config = CacheConfig.for_environment()
        assert config.max_files == 100
        assert config.max_memory_mb == 1024
        assert config.disk_cache_gb == 5
        assert config.prefetch_enabled is False

    @patch("torchsky.cache.psutil")
    @patch.object(CacheConfig, "_is_hpc_environment", return_value=True)
    def test_for_environment_hpc(self, mock_hpc, mock_psutil):
        mock_memory = MagicMock()
        mock_memory.total = 100 * (1024**3)  # 100 GB
        mock_psutil.virtual_memory.return_value = mock_memory

        config = CacheConfig.for_environment()
        assert config.max_files == 1000
        assert config.max_memory_mb == int(100 * 1024 * 0.3)
        assert config.disk_cache_gb == 50
        assert config.prefetch_enabled is True

    @patch("torchsky.cache.psutil")
    @patch.object(CacheConfig, "_is_hpc_environment", return_value=False)
    @patch.object(CacheConfig, "_is_cloud_environment", return_value=True)
    def test_for_environment_cloud(self, mock_cloud, mock_hpc, mock_psutil):
        mock_memory = MagicMock()
        mock_memory.total = 16 * (1024**3)  # 16 GB
        mock_psutil.virtual_memory.return_value = mock_memory

        config = CacheConfig.for_environment()
        assert config.max_files == 500
        assert config.max_memory_mb == int(16 * 1024 * 0.2)
        assert config.disk_cache_gb == 20
        assert config.prefetch_enabled is True

    @patch("torchsky.cache.psutil")
    @patch.object(CacheConfig, "_is_hpc_environment", return_value=False)
    @patch.object(CacheConfig, "_is_cloud_environment", return_value=False)
    @patch.object(CacheConfig, "_is_gpu_environment", return_value=True)
    def test_for_environment_gpu(self, mock_gpu, mock_cloud, mock_hpc, mock_psutil):
        mock_memory = MagicMock()
        mock_memory.total = 32 * (1024**3)  # 32 GB
        mock_psutil.virtual_memory.return_value = mock_memory

        config = CacheConfig.for_environment()
        assert config.max_files == 200
        assert config.max_memory_mb == int(32 * 1024 * 0.4)
        assert config.disk_cache_gb == 30
        assert config.prefetch_enabled is True

    @patch("torchsky.cache.psutil")
    @patch.object(CacheConfig, "_is_hpc_environment", return_value=False)
    @patch.object(CacheConfig, "_is_cloud_environment", return_value=False)
    @patch.object(CacheConfig, "_is_gpu_environment", return_value=False)
    def test_for_environment_default(self, mock_gpu, mock_cloud, mock_hpc, mock_psutil):
        mock_memory = MagicMock()
        mock_memory.total = 8 * (1024**3)  # 8 GB
        mock_psutil.virtual_memory.return_value = mock_memory

        config = CacheConfig.for_environment()
        assert config.max_files == 100
        assert config.max_memory_mb == min(2048, int(8 * 1024 * 0.1))
        assert config.disk_cache_gb == 5
        assert config.prefetch_enabled is False

    @patch.dict(os.environ, {"SLURM_JOB_ID": "12345"})
    def test_is_hpc_environment_true(self):
        assert CacheConfig._is_hpc_environment() is True

    @patch.dict(os.environ, clear=True)
    def test_is_hpc_environment_false(self):
        assert CacheConfig._is_hpc_environment() is False

    @patch.dict(os.environ, {"AWS_EXECUTION_ENV": "AWS_Lambda"})
    def test_is_cloud_environment_true(self):
        assert CacheConfig._is_cloud_environment() is True

    @patch.dict(os.environ, clear=True)
    def test_is_cloud_environment_false(self):
        assert CacheConfig._is_cloud_environment() is False

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    def test_is_gpu_environment_true(self, mock_count, mock_available):
        assert CacheConfig._is_gpu_environment() is True

    @patch("torch.cuda.is_available", return_value=False)
    def test_is_gpu_environment_false_not_available(self, mock_available):
        assert CacheConfig._is_gpu_environment() is False

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=0)
    def test_is_gpu_environment_false_no_devices(self, mock_count, mock_available):
        assert CacheConfig._is_gpu_environment() is False


class TestCacheOptimization:
    """Test CacheOptimization functionality."""

    @patch("torchsky.cache.CacheManager.configure_cpp_cache")
    def test_optimize_for_dataset_small(self, mock_configure_cpp_cache):
        # Create a test configuration
        config = CacheConfig(disk_cache_gb=10)
        manager = torchsky.cache.CacheManager(config)

        # Mock get_cache_manager to return our specific manager
        with patch("torchsky.cache.get_cache_manager", return_value=manager):
            # Test a small dataset that fits easily in cache
            # 100 files, 10 MB each = ~1 GB total (< 10 GB limit)
            file_paths = ["file_{}.fits".format(i) for i in range(100)]
            torchsky.cache.optimize_for_dataset(file_paths, avg_file_size_mb=10.0)

            # Verification
            # Should enable aggressive caching for the entire dataset
            assert manager.config.max_files == 100
            assert manager.config.prefetch_enabled is True
            mock_configure_cpp_cache.assert_called_once()

    @patch("torchsky.cache.CacheManager.configure_cpp_cache")
    def test_optimize_for_dataset_large(self, mock_configure_cpp_cache):
        # Create a test configuration
        config = CacheConfig(disk_cache_gb=10)
        manager = torchsky.cache.CacheManager(config)

        # Mock get_cache_manager to return our specific manager
        with patch("torchsky.cache.get_cache_manager", return_value=manager):
            # Test a large dataset that exceeds cache limit
            # 200 files, 100 MB each = ~19.5 GB total (> 10 GB limit)
            file_paths = ["file_{}.fits".format(i) for i in range(200)]
            torchsky.cache.optimize_for_dataset(file_paths, avg_file_size_mb=100.0)

            # Verification
            # Should restrict max files to optimal_files
            # optimal_files = int(10 * 1024 / 100) = 102
            # min(102, 1000) = 102
            assert manager.config.max_files == 102
            # prefetch_enabled should not be forced to True in this branch
            # (assuming default was False, or at least it doesn't change it)
            mock_configure_cpp_cache.assert_called_once()

    @patch("torchsky.cache.CacheManager.configure_cpp_cache")
    def test_optimize_for_dataset_huge_many_files(self, mock_configure_cpp_cache):
        # Create a test configuration
        config = CacheConfig(disk_cache_gb=100)
        manager = torchsky.cache.CacheManager(config)

        # Mock get_cache_manager to return our specific manager
        with patch("torchsky.cache.get_cache_manager", return_value=manager):
            # Test a very large number of files that exceeds cache limit
            # optimal_files = int(100 * 1024 / 10) = 10240
            # max_files is capped at 1000
            file_paths = ["file_{}.fits".format(i) for i in range(20000)]
            torchsky.cache.optimize_for_dataset(file_paths, avg_file_size_mb=10.0)

            # Verification
            assert manager.config.max_files == 1000
            mock_configure_cpp_cache.assert_called_once()


def test_configure_for_environment():
    """Test configure_for_environment configures the cpp cache via the manager."""
    with patch("torchsky.cache.get_cache_manager") as mock_get_cache_manager:
        mock_manager = MagicMock()
        mock_get_cache_manager.return_value = mock_manager

        torchsky.cache.configure_for_environment()

        mock_get_cache_manager.assert_called_once()
        mock_manager.configure_cpp_cache.assert_called_once()


class TestCacheManagerFunctions:
    """Test cache manager module-level functions."""

    def test_configure_for_environment(self):
        from torchsky.cache import configure_for_environment

        with patch("torchsky.cache.get_cache_manager") as mock_get_cache_manager:
            mock_manager = MagicMock()
            mock_get_cache_manager.return_value = mock_manager

            configure_for_environment()

            mock_get_cache_manager.assert_called_once()
            mock_manager.configure_cpp_cache.assert_called_once()

    def test_get_cache_stats(self):
        import torchsky.cache
        from torchsky.cache import get_cache_manager, get_cache_stats

        original_manager = torchsky.cache._cache_manager
        torchsky.cache._cache_manager = None

        try:
            manager = get_cache_manager()
            manager._stats["hits"] = 10
            manager._stats["misses"] = 5
            manager._stats["evictions"] = 2
            manager._stats["bytes_read"] = 1024

            stats = get_cache_stats()

            assert stats["hits"] == 10
            assert stats["misses"] == 5
            assert stats["evictions"] == 2
            assert stats["bytes_read"] == 1024
            assert "cpp_cache_size" in stats
            assert "config" in stats
            assert "hit_rate" in stats
            assert stats["hit_rate"] == 10 / 15
        finally:
            torchsky.cache._cache_manager = original_manager

    @patch("torchsky.cache.CacheManager")
    def test_configure_cache(self, mock_manager_cls):
        import torchsky.cache
        from torchsky.cache import configure_cache

        original_manager = torchsky.cache._cache_manager
        mock_manager = MagicMock()
        mock_manager_cls.return_value = mock_manager

        try:
            configure_cache(max_files=42, max_memory_mb=1337, disk_cache_gb=7)

            assert torchsky.cache._cache_manager is mock_manager
            mock_manager_cls.assert_called_once()
            config_arg = mock_manager_cls.call_args[0][0]
            assert config_arg.max_files == 42
            assert config_arg.max_memory_mb == 1337
            assert config_arg.disk_cache_gb == 7
            mock_manager.configure_cpp_cache.assert_called_once()
        finally:
            torchsky.cache._cache_manager = original_manager

    def test_clear_cache(self):
        from torchsky.cache import get_cache_manager, clear_cache, get_cache_stats

        import torchsky.cache

        original_manager = torchsky.cache._cache_manager
        torchsky.cache._cache_manager = None

        try:
            # Set some state
            manager = get_cache_manager()
            manager._stats["hits"] = 10
            manager._stats["misses"] = 5

            # Verify state is set
            stats_before = get_cache_stats()
            assert stats_before["hits"] == 10
            assert stats_before["misses"] == 5

            # Clear cache
            clear_cache()

            # Verify state is reset
            stats_after = get_cache_stats()
            assert stats_after["hits"] == 0
            assert stats_after["misses"] == 0
        finally:
            torchsky.cache._cache_manager = original_manager



    @patch('torchsky.cache._detect_environment_type')
    @patch.object(torchsky.cache.CacheConfig, 'for_environment')
    def test_get_optimal_cache_config(self, mock_for_environment, mock_detect_env):
        from torchsky.cache import get_optimal_cache_config, CacheConfig

        # Set up mocks
        mock_detect_env.return_value = "custom_env"
        mock_config = CacheConfig(
            max_files=123,
            max_memory_mb=456,
            disk_cache_gb=789,
            prefetch_enabled=True,
        )
        mock_for_environment.return_value = mock_config

        # Call the function
        result = get_optimal_cache_config()

        # Verify
        mock_for_environment.assert_called_once()
        mock_detect_env.assert_called_once()
        assert result == {
            "max_files": 123,
            "max_memory_mb": 456,
            "disk_cache_gb": 789,
            "prefetch_enabled": True,
            "environment": "custom_env",
        }

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
