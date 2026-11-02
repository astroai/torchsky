#!/usr/bin/env python3
"""Render the reference benchmark tables from the committed JSON snapshot.

    pixi run docs-bench-tables          # regenerate docs/evaluation/benchmarks.md
    pixi run docs-bench-tables-check    # fail if the page is stale

The JSON artifact is produced by torchsky-harness (``benchmarks/bench_docs_reference.py``,
run on CANFAR by the CPU and GPU campaigns). Tables are rendered
between the ``<!-- BENCH_TABLES -->`` marker in the template; prose outside the
marker stays hand-written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data" / "reference_benchmarks.json"
TEMPLATE = ROOT / "docs" / "evaluation" / "_benchmarks_template.md"
PAGE = ROOT / "docs" / "evaluation" / "benchmarks.md"
MARKER = "<!-- BENCH_TABLES -->"


def _environment_table(environment: dict[str, object]) -> list[str]:
    rows = [
        ("Machine", f"{environment.get('cpu')} ({environment.get('machine')})"),
        ("GPU", environment.get("gpu") or "not used"),
        ("Operating system", environment.get("platform")),
        ("PyTorch", environment.get("torch")),
        ("torchfits", environment.get("torchfits")),
        ("torchsky", environment.get("torchsky")),
        ("Threads", environment.get("threads")),
        ("Measured", environment.get("generated_at")),
        ("Commit", environment.get("git_sha") or "unknown"),
    ]
    lines = ["## Environment", "", "| Field | Value |", "|---|---|"]
    lines.extend(f"| {name} | `{value}` |" for name, value in rows)
    return lines


def _case_tables(cases: list[dict[str, object]]) -> list[str]:
    groups: dict[str, list[dict[str, object]]] = {}
    for case in cases:
        groups.setdefault(str(case["group"]), []).append(case)
    lines: list[str] = []
    for group, items in groups.items():
        lines.extend(
            [
                f"## {group}",
                "",
                "| Case | Parameters | Median | Throughput |",
                "|---|---|---|---|",
            ]
        )
        for case in items:
            throughput = f"{case['throughput']:g} {case['throughput_unit']}"
            lines.append(
                f"| `{case['case']}` | {case['parameters']} | "
                f"{float(case['median_s']):.4g} s | {throughput} |"
            )
        lines.append("")
    return lines


def render() -> str:
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    template = TEMPLATE.read_text(encoding="utf-8")
    if MARKER not in template:
        raise SystemExit(f"template is missing {MARKER}")
    environment = dict(payload["environment"])
    environment.setdefault("generated_at", payload.get("generated_at"))
    body = [*_environment_table(environment), ""]
    body.extend(_case_tables(payload["cases"]))
    text = template.replace(MARKER, "\n".join(body).rstrip())
    return text.rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    text = render()
    if args.check:
        if not PAGE.is_file() or PAGE.read_text(encoding="utf-8") != text:
            print(
                f"{PAGE.relative_to(ROOT)} is stale; run `pixi run docs-bench-tables`."
            )
            return 1
        print("Reference benchmark tables are current.")
        return 0
    PAGE.parent.mkdir(parents=True, exist_ok=True)
    PAGE.write_text(text, encoding="utf-8")
    print(f"wrote {PAGE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
