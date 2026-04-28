import sys
from unittest.mock import MagicMock
sys.modules['torchsky.cpp'] = MagicMock()

import pytest

pytest.main(["tests/test_hdu.py"])
