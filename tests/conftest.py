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


@pytest.fixture(autouse=True)
def _clear_torchsky_native_io_handles_after_test():
    yield
    table_mod = sys.modules.get("torchsky.table")
    clear_cache = getattr(table_mod, "clear_cache", None)
    if clear_cache is not None:
        clear_cache()

    try:
        import torchfits
        torchfits.clear_file_cache()
    except Exception:
        pass
