#!/usr/bin/env python3
import os
import stat
import subprocess
from pathlib import Path


HOOK_CONTENT = """#!/bin/bash
# Pre-push: ruff + compileall only. Full pytest is GitHub build-test
# (or optional `pixi run ci-local`). Do not put the test suite back here.
export PATH="$HOME/.pixi/bin:$PATH"
if ! command -v pixi >/dev/null 2>&1; then
  echo "pre-push: pixi not found on PATH"
  exit 1
fi
echo "pre-push: pixi run preflight-push"
pixi run preflight-push || {
  echo "preflight-push failed; push aborted"
  exit 1
}
exit 0
"""


def install_pre_push_hook():
    repo_root = Path(__file__).parent.parent
    hook_path = repo_root / ".git" / "hooks" / "pre-push"

    if not (repo_root / ".git").exists():
        print("Error: .git directory not found. Are you in the repo root?")
        return

    hook_path.parent.mkdir(parents=True, exist_ok=True)
    hook_path.write_text(HOOK_CONTENT)
    st = os.stat(hook_path)
    os.chmod(hook_path, st.st_mode | stat.S_IEXEC)
    print(f"Installed pre-push hook at {hook_path}")


if __name__ == "__main__":
    install_pre_push_hook()
    try:
        subprocess.run(["pre-commit", "install"], check=True)
        print("Pre-commit hooks installed.")
    except Exception as e:
        print(f"Failed to run pre-commit install: {e}")
