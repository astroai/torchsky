from __future__ import annotations

import os
import sys
from pathlib import Path

# Centralized output directory for all benchmarks
DEFAULT_OUTPUT_DIR = Path("benchmarks_results")
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"


def ensure_src_on_path() -> None:
    """Make `import torchsky` resolve to the in-tree package (with native extension)."""
    src = str(SRC_ROOT)
    if src not in sys.path:
        sys.path.insert(0, src)


def benchmark_subprocess_env() -> dict[str, str]:
    """Environment for benchmark subprocesses: repo root + src on PYTHONPATH."""
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(SRC_ROOT), str(REPO_ROOT)]
    if existing:
        parts.append(existing)
    return {**os.environ, "PYTHONPATH": os.pathsep.join(parts)}
