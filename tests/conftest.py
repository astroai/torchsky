from pathlib import Path
import os
import sys

# xdist + MKL/OpenMP: avoid fork-related deadlocks under pytest -n auto.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_configure(config) -> None:
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
    config.addinivalue_line(
        "markers", "needs_float64: skipped on MPS (no float64 support)"
    )


@pytest.fixture(autouse=True)
def _clear_torchfits_handles_after_test():
    yield
    try:
        import torchfits

        torchfits.clear_file_cache()
    except Exception:
        pass


NEEDS_FLOAT64_REASON = "needs float64; MPS has none"


def pytest_collection_modifyitems(config, items) -> None:
    """Skip ``needs_float64`` tests when the default compute device is MPS."""
    try:
        from torchsky.device import default_compute_device

        on_mps = default_compute_device().type == "mps"
    except Exception:
        return
    if not on_mps:
        return
    skip = pytest.mark.skip(reason=NEEDS_FLOAT64_REASON)
    for item in items:
        marker = item.get_closest_marker("needs_float64")
        if marker is None:
            continue
        # Bare marker: skip every param. With args: skip only params whose id
        # contains one of the named tokens (e.g. float64-only resampling kernels).
        callspec = getattr(item, "callspec", None)
        param_tokens = set(callspec.id.split("-")) if callspec is not None else set()
        if not marker.args or param_tokens & set(marker.args):
            item.add_marker(skip)


def _available_accelerators() -> list[str]:
    import torch

    found = []
    if torch.cuda.is_available():
        found.append("cuda")
    if torch.backends.mps.is_available():
        found.append("mps")
    return found


@pytest.fixture(params=["cpu", "cuda", "mps"])
def compute_device(request):
    """Run a test on CPU and on every available accelerator (skip if absent)."""
    import torch

    name = request.param
    if name != "cpu" and name not in _available_accelerators():
        pytest.skip(f"{name} unavailable")
    return torch.device(name)


@pytest.fixture
def default_device():
    """The library default compute device (CUDA/MPS via TORCHSKY_DEVICE or CPU)."""
    from torchsky.device import default_compute_device

    return default_compute_device()


def assert_close_on_device(actual, expected, **kwargs) -> None:
    """``torch.testing.assert_close`` with ``expected`` moved to ``actual``'s device.

    Library results live on the compute device (CUDA/MPS when selected) while test
    references are built on CPU; numerical tolerances are unchanged.
    """
    import torch

    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        expected = expected.to(actual.device)
    torch.testing.assert_close(actual, expected, **kwargs)
