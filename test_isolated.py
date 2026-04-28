import sys
from unittest.mock import MagicMock
sys.modules['torchsky.cpp'] = MagicMock()

import pytest
import os
import torch
from torchsky.hdu import TableHDU, Header, TensorHDU

pytest.main(["tests/test_hdu.py"])
