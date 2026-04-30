import sys
from unittest.mock import MagicMock
sys.modules["torchsky.cpp"] = MagicMock()
import torchsky.table

# Just a quick check to ensure we haven't introduced syntax errors
print("torchsky.table is imported")
