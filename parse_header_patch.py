with open("test_isolated_all.py", "w") as f:
    f.write("""
import sys
from unittest.mock import MagicMock

class MockCpp:
    pass

sys.modules['torchsky.cpp'] = MockCpp()

import pytest
if __name__ == "__main__":
    sys.exit(pytest.main(["tests/test_wcs_upstream_astropy.py", "tests/test_wcs_legacy.py", "tests/test_frame_stypes.py", "tests/test_io.py", "tests/test_release_smoke.py", "tests/test_security_eval.py", "-v"]))
""")
