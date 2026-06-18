from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def _clear_torchsky_native_io_handles_after_test():
    yield
    table_mod = sys.modules.get("torchsky.table")
    close_handles = getattr(table_mod, "_close_all_cached_handles", None)
    if close_handles is not None:
        close_handles()

    caches_mod = sys.modules.get("torchsky._io_engine.caches")
    clear_file_cache = getattr(caches_mod, "clear_file_cache", None)
    if clear_file_cache is not None:
        cpp_mod = sys.modules.get("torchsky.cpp")
        clear_file_cache(cpp_module=cpp_mod)
