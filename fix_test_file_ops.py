import re

with open("tests/test_table_file_ops.py", "r") as f:
    lines = f.readlines()

new_lines = []
for idx, line in enumerate(lines):
    if idx + 1 in [110, 418, 422, 456]:
        if "type: ignore" not in line:
             line = line.rstrip() + "  # type: ignore\n"
    new_lines.append(line)

with open("tests/test_table_file_ops.py", "w") as f:
    f.writelines(new_lines)
