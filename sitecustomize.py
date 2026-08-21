"""Worktree bootstrap: neutralize the main-checkout editable finder.

The shared pixi environment installs torchsky editable against
~/src/torchsky via a meta-path finder; when running from this worktree we
strip that finder and put this tree's ``src`` first so local edits load.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")

if os.path.basename(_HERE) == "torchsky-deep-field" or _HERE.startswith("/tmp/"):
    sys.meta_path = [
        f
        for f in sys.meta_path
        if type(f).__name__ != "ScikitBuildRedirectingFinder"
    ]
    while _SRC in sys.path:
        sys.path.remove(_SRC)
    sys.path.insert(0, _SRC)
