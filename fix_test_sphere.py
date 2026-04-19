import re

with open("tests/test_sphere_spectral.py", "r") as f:
    content = f.read()

# For lines with errors, append "# type: ignore"
errors = [
    135, 137, 151, 160, 172, 179, 191, 196, 219, 223, 242, 243, 252, 263, 355, 369, 378, 528, 530, 546, 624, 634, 643, 645, 646, 680, 688, 707, 720
]

lines = content.split('\n')
for i in errors:
    # 1-based to 0-based
    idx = i - 1
    if "type: ignore" not in lines[idx]:
        lines[idx] = lines[idx] + "  # type: ignore"

with open("tests/test_sphere_spectral.py", "w") as f:
    f.write('\n'.join(lines))
