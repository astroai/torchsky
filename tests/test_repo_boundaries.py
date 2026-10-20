"""Keep torchsky's repository boundary with torchsky-harness explicit.

Core implementation, bounded tests, documentation, and packaging metadata live
here. Benchmark runners, gate tasks, evidence bundles, campaigns, and comparator
harnesses live in the sibling torchsky-harness repository.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "torchsky"


def test_no_benchmark_or_campaign_tree() -> None:
    for name in ("benchmarks", "campaigns", "runs", "benchmark_results"):
        assert not (ROOT / name).exists(), (
            f"{name}/ belongs in torchsky-harness, not torchsky"
        )


def test_core_package_has_no_benchmark_modules() -> None:
    offenders = sorted(
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*.py")
        if path.name.startswith("bench_") or path.name.startswith("benchmark_")
    )
    assert not offenders, f"benchmark modules belong in torchsky-harness: {offenders}"


def test_benchmarks_doc_stays_a_redirect() -> None:
    text = (ROOT / "docs" / "benchmarks.md").read_text(encoding="utf-8")
    assert "torchsky-harness" in text
