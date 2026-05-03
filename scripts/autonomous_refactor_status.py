#!/usr/bin/env python
"""Print the autonomous refactor launch status for scheduled Codex runs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo


START_DATE = date(2026, 5, 8)
TIMEZONE = "America/Vancouver"
TARGET_BRANCH = "codex/autonomous-refactor-2026-05"
LOG_PATH = Path("docs/reviews/autonomous_refactor_log.md")
UNCHECKED_RE = re.compile(r"^- \[ \] (?P<item>.+)$")


@dataclass(frozen=True)
class RefactorStatus:
    today: str
    timezone: str
    start_date: str
    can_start: bool
    target_branch: str
    current_branch: str
    git_status: str
    next_item: str | None
    log_path: str


def _run_git(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except OSError as exc:
        return f"git unavailable: {exc}"
    return result.stdout.strip()


def _next_unchecked_item(log_path: Path) -> str | None:
    if not log_path.exists():
        return None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = UNCHECKED_RE.match(line)
        if match:
            return match.group("item")
    return None


def build_status() -> RefactorStatus:
    today = datetime.now(ZoneInfo(TIMEZONE)).date()
    return RefactorStatus(
        today=today.isoformat(),
        timezone=TIMEZONE,
        start_date=START_DATE.isoformat(),
        can_start=today >= START_DATE,
        target_branch=TARGET_BRANCH,
        current_branch=_run_git(["branch", "--show-current"]),
        git_status=_run_git(["status", "--short", "--branch"]),
        next_item=_next_unchecked_item(LOG_PATH),
        log_path=str(LOG_PATH),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON status.")
    parser.add_argument(
        "--fail-before-start",
        action="store_true",
        help="Exit 2 if the May 8 launch gate has not opened.",
    )
    args = parser.parse_args(argv)

    status = build_status()
    if args.json:
        print(json.dumps(asdict(status), indent=2, sort_keys=True))
    else:
        print(f"today: {status.today} ({status.timezone})")
        print(f"start_date: {status.start_date}")
        print(f"can_start: {status.can_start}")
        print(f"target_branch: {status.target_branch}")
        print(f"current_branch: {status.current_branch}")
        print("git_status:")
        print(status.git_status or "<empty>")
        print(f"next_item: {status.next_item or '<none>'}")

    if args.fail_before_start and not status.can_start:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
