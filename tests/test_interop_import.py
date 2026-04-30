import sys
from unittest.mock import patch
import pytest
import torchsky.interop


def test_to_arrow_import_error():
    # Use patch.dict to remove pyarrow from sys.modules
    with patch.dict(sys.modules, {"pyarrow": None}):
        with pytest.raises(
            ImportError, match="PyArrow is required for to_arrow conversion."
        ):
            torchsky.interop.to_arrow({})


def test_to_pandas_import_error():
    # Use patch.dict to remove pandas from sys.modules
    with patch.dict(sys.modules, {"pandas": None}):
        with pytest.raises(
            ImportError, match="Pandas is required for to_pandas conversion."
        ):
            torchsky.interop.to_pandas({})
