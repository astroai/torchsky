filepath = "src/torchsky/wcs/core.py"
with open(filepath, "r") as f:
    content = f.read()

import_future = "from __future__ import annotations\n"
if import_future not in content:
    content = import_future + content

with open(filepath, "w") as f:
    f.write(content)
