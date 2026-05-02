import sys
from unittest.mock import MagicMock
sys.modules['torchsky.cpp'] = MagicMock()
sys.path.insert(0, 'src')
import pytest
sys.exit(pytest.main(['tests/test_api.py']))
