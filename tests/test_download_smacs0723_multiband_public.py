from __future__ import annotations
import pytest
pytest.importorskip('requests')

import sys
from pathlib import Path

import pytest

pytest.importorskip("requests")

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from download_smacs0723_multiband_public import _default_max_exposures_for_band  # noqa: E402


def test_default_public_exposure_caps_match_short_and_long_bands() -> None:
    assert _default_max_exposures_for_band("F090W") == 100
    assert _default_max_exposures_for_band("F277W") == 100
