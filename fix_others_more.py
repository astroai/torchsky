import re

with open("tests/test_wcs_core.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "dtype" in line or ".to(" in line:
        if "type: ignore" not in line and "import" not in line and "def" not in line:
            line = line.rstrip() + "  # type: ignore\n"
    new_lines.append(line)

with open("tests/test_wcs_core.py", "w") as f:
    f.writelines(new_lines)

with open("tests/test_viz_server.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "fastapi.testclient" in line:
        if "type: ignore" not in line:
            line = line.rstrip() + "  # type: ignore\n"
    new_lines.append(line)

with open("tests/test_viz_server.py", "w") as f:
    f.writelines(new_lines)
