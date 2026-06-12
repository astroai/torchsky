#!/usr/bin/env python3
"""Archive a multi-domain benchmark scorecard into committed history."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCORECARDS_DIR = ROOT / "benchmarks" / "scorecards"
BENCH_ROOT_DEFAULT = ROOT / "benchmarks_results"

LARGE_N_THRESHOLD = 100_000
PREFERRED_METHOD_BY_FAMILY = {
    "smart": "torchsky",
    "specialized": "torchsky_specialized",
    "numpy": "torchsky_numpy",
}


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if v != v:
        return None
    return v


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def _metadata_dict(row: dict[str, Any]) -> dict[str, Any]:
    md = row.get("metadata")
    if isinstance(md, dict):
        return md
    if isinstance(md, str):
        txt = md.strip()
        if not txt:
            return {}
        try:
            parsed = ast.literal_eval(txt)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _extract_n_points(row: dict[str, Any]) -> int | None:
    n = _to_int(row.get("n_points"))
    if n is not None and n >= 0:
        return n
    case_id = str(row.get("case_id") or "")
    marker = "::n"
    i = case_id.find(marker)
    if i < 0:
        return None
    j = case_id.find("::", i + len(marker))
    if j < 0:
        return None
    return _to_int(case_id[i + len(marker) : j])


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except Exception:
        return "unknown"


def _parse_runs(spec: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid run mapping '{part}', expected domain=run_id")
        domain, run_id = part.split("=", 1)
        domain = domain.strip()
        run_id = run_id.strip()
        if not domain or not run_id:
            raise ValueError(f"Invalid run mapping '{part}'")
        out[domain] = run_id
    if not out:
        raise ValueError("At least one domain=run_id mapping is required")
    return out


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _compute_domain_scorecard(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[tuple[dict[str, Any], float]]] = (
        defaultdict(list)
    )
    for row in rows:
        if not bool(row.get("comparable", False)):
            continue
        if str(row.get("status")) != "OK":
            continue
        t = _to_float(row.get("time_s"))
        if t is None or t <= 0:
            continue
        grouped[
            (str(row.get("domain")), str(row.get("case_id")), str(row.get("family")))
        ].append((row, t))

    scorecard: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"wins": 0, "total": 0}
    )
    large_n_scorecard: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"wins": 0, "total": 0}
    )
    for (_domain, _case_id, family), grp in grouped.items():
        grp.sort(key=lambda x: x[1])
        tf_candidates = [r for (r, _t) in grp if str(r.get("library")) == "torchsky"]
        if not tf_candidates:
            continue

        preferred = PREFERRED_METHOD_BY_FAMILY.get(family)
        torch_row = None
        if preferred is not None:
            for r in tf_candidates:
                if str(r.get("method")) == preferred:
                    torch_row = r
                    break
        if torch_row is None:
            for r in tf_candidates:
                if str(r.get("method", "")).startswith("torchsky"):
                    torch_row = r
                    break
        if torch_row is None:
            torch_row = tf_candidates[0]

        domain = str(torch_row.get("domain"))
        key = (domain, family)
        scorecard[key]["total"] += 1
        if grp[0][0] is torch_row:
            scorecard[key]["wins"] += 1

        n_points = _extract_n_points(torch_row)
        if n_points is not None and n_points >= LARGE_N_THRESHOLD:
            large_n_scorecard[key]["total"] += 1
            if grp[0][0] is torch_row:
                large_n_scorecard[key]["wins"] += 1

    families: dict[str, dict[str, Any]] = {}
    for (domain, family), stats in sorted(scorecard.items()):
        total = int(stats["total"])
        wins = int(stats["wins"])
        families[f"{domain}:{family}"] = {
            "domain": domain,
            "family": family,
            "wins": wins,
            "total": total,
            "win_rate": (wins / total) if total else 0.0,
            "large_n_wins": int(large_n_scorecard.get((domain, family), {}).get("wins", 0)),
            "large_n_total": int(large_n_scorecard.get((domain, family), {}).get("total", 0)),
        }

    skipped = sum(1 for r in rows if str(r.get("status")) == "SKIPPED")
    return {
        "normalized_rows": len(rows),
        "skipped_rows": skipped,
        "families": families,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Benchmark scorecard: {payload['label']}",
        "",
        f"- Archived: `{payload['archived_at']}`",
        f"- Profile: `{payload['profile']}`",
        f"- Platform: `{payload['platform']}`",
        f"- Git SHA: `{payload['git_sha']}`",
        "",
        "## Summary",
        "",
        "| Domain | Run ID | Families first | Deficits |",
        "|---|---|---:|---:|",
    ]
    for domain, block in sorted(payload["domains"].items()):
        fams = block["scorecard"]["families"]
        wins = sum(v["wins"] for v in fams.values())
        total = sum(v["total"] for v in fams.values())
        lines.append(
            f"| {domain} | `{block['run_id']}` | {wins}/{total} | {block['deficit_rows']} |"
        )
    lines.extend(["", "## By family", ""])
    for domain, block in sorted(payload["domains"].items()):
        lines.append(f"### {domain} (`{block['run_id']}`)")
        lines.append("")
        for key, fam in sorted(block["scorecard"]["families"].items()):
            wr = 100.0 * fam["win_rate"]
            ln = ""
            if fam["large_n_total"]:
                ln = f"; large-N {fam['large_n_wins']}/{fam['large_n_total']}"
            lines.append(
                f"- `{fam['family']}`: {fam['wins']}/{fam['total']} ({wr:.1f}%){ln}"
            )
        if block["deficit_rows"] == 0:
            lines.append("- Deficits: none")
        else:
            lines.append(f"- Deficits: {block['deficit_rows']} rows")
        lines.append("")
    if payload.get("notes"):
        lines.extend(["## Notes", "", payload["notes"], ""])
    lines.append(
        "Local artifacts (gitignored): "
        + ", ".join(
            f"`benchmarks_results/{block['run_id']}/`"
            for block in payload["domains"].values()
        )
    )
    lines.append("")
    return "\n".join(lines)


def _update_index(index_path: Path, payload: dict[str, Any], stem: str) -> None:
    header = (
        "# Benchmark scorecard index\n\n"
        "Committed snapshots for tracking throughput leadership over time. "
        "Ephemeral run artifacts remain under `benchmarks_results/` (gitignored).\n\n"
        "Archive a new snapshot after a full domain sweep:\n\n"
        "```bash\n"
        "pixi run bench-archive-scorecard -- \\\n"
        "  --label main_domains \\\n"
        "  --runs fits=<fits_run_id>,fitstable=<fitstable_run_id>,wcs=<wcs_run_id>,sphere=<sphere_run_id>\n"
        "```\n\n"
        "| Date | Label | FITS smart | Table smart | WCS large-N | Sphere specialized | Deficits | Git SHA | Snapshot |\n"
        "|---|---|---:|---:|---:|---:|---:|---|---|\n"
    )

    def _fam(domain: str, family: str, field: str) -> str:
        block = payload["domains"].get(domain, {})
        fams = block.get("scorecard", {}).get("families", {})
        for fam in fams.values():
            if fam.get("family") == family:
                if field == "wins":
                    return f"{fam['wins']}/{fam['total']}"
                if field == "large_n":
                    return f"{fam['large_n_wins']}/{fam['large_n_total']}"
        return "-"

    total_deficits = sum(
        int(block.get("deficit_rows", 0)) for block in payload["domains"].values()
    )
    row = (
        f"| {payload['date']} | {payload['label']} "
        f"| {_fam('fits', 'smart', 'wins')} "
        f"| {_fam('fitstable', 'smart', 'wins')} "
        f"| {_fam('wcs', 'smart', 'large_n')} "
        f"| {_fam('sphere', 'specialized', 'wins')} "
        f"| {total_deficits} "
        f"| `{payload['git_sha'][:12]}` "
        f"| [{stem}]({stem}.md) |\n"
    )

    if index_path.exists():
        body = index_path.read_text(encoding="utf-8")
        if stem in body:
            return
        if "| Date | Label |" in body:
            lines = body.splitlines(keepends=True)
            out: list[str] = []
            inserted = False
            for line in lines:
                out.append(line)
                if not inserted and line.startswith("|") and "202" in line:
                    out.append(row)
                    inserted = True
            if not inserted:
                out.append(row)
            index_path.write_text("".join(out), encoding="utf-8")
            return

    index_path.write_text(header + row, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        required=True,
        help="Short label for this scorecard (e.g. main_domains)",
    )
    parser.add_argument(
        "--runs",
        required=True,
        help="Comma-separated domain=run_id mappings under --bench-root",
    )
    parser.add_argument(
        "--bench-root",
        type=Path,
        default=BENCH_ROOT_DEFAULT,
        help="Root directory containing per-run output folders",
    )
    parser.add_argument(
        "--profile",
        default="user",
        help="Benchmark profile recorded in the snapshot",
    )
    parser.add_argument(
        "--date",
        default="",
        help="YYYYMMDD date stamp (default: today UTC)",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Optional free-form notes stored in the snapshot",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print payload JSON without writing files",
    )
    args = parser.parse_args()

    try:
        runs = _parse_runs(args.runs)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    date = args.date or time.strftime("%Y%m%d")
    stem = f"{date}_{args.label}"
    domains: dict[str, Any] = {}

    for domain, run_id in runs.items():
        run_dir = args.bench_root / run_id
        results_path = run_dir / "results.csv"
        deficits_path = run_dir / "torchsky_deficits.csv"
        if not results_path.exists():
            print(f"Missing results: {results_path}", file=sys.stderr)
            return 1
        rows = _load_rows(results_path)
        deficit_rows = 0
        if deficits_path.exists():
            deficit_rows = max(0, len(_load_rows(deficits_path)))
        domains[domain] = {
            "run_id": run_id,
            "deficit_rows": deficit_rows,
            "scorecard": _compute_domain_scorecard(rows),
        }

    payload: dict[str, Any] = {
        "schema": "torchsky.benchmark_scorecard/v1",
        "label": args.label,
        "date": date,
        "archived_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "profile": args.profile,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_sha": _git_sha(),
        "bench_root": str(args.bench_root),
        "domains": domains,
        "notes": args.notes.strip(),
    }

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    SCORECARDS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = SCORECARDS_DIR / f"{stem}.json"
    md_path = SCORECARDS_DIR / f"{stem}.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    _update_index(SCORECARDS_DIR / "index.md", payload, stem)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Updated {SCORECARDS_DIR / 'index.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
