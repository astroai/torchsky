from pathlib import Path


def test_torchsky_does_not_import_torchfits_private_modules():
    source_root = Path(__file__).parents[1] / "src" / "torchsky"
    offenders = []
    for path in source_root.rglob("*.py"):
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if "torchfits._" in line:
                offenders.append(f"{path}:{line_number}: {line.strip()}")
    assert not offenders, "\n".join(offenders)
