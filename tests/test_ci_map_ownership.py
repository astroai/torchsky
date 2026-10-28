from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_every_map_test_file_has_a_ci_owner() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    missing = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests").glob("test_map_*.py")
        if path.relative_to(ROOT).as_posix() not in workflow
    )
    assert not missing, f"map tests missing from CI ownership: {missing}"
