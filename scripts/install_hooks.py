#!/usr/bin/env python3
import os
import stat
import subprocess
from pathlib import Path

def install_pre_push_hook():
    repo_root = Path(__file__).parent.parent
    hook_path = repo_root / ".git" / "hooks" / "pre-push"

    if not (repo_root / ".git").exists():
        print("Error: .git directory not found. Are you in the repo root?")
        return

    hook_content = """#!/bin/bash
# Pre-push hook to run local CI gate
echo "🚀 Running local CI gate before push..."
pixi run ci-local || {
    echo "❌ Local CI failed. Push aborted. Fix errors or use --no-verify if you are absolutely sure."
    exit 1
}
echo "✅ Local CI passed. Proceeding with push."
exit 0
"""

    # Ensure hooks directory exists
    hook_path.parent.mkdir(parents=True, exist_ok=True)

    with open(hook_path, "w") as f:
        f.write(hook_content)

    # Make hook executable
    st = os.stat(hook_path)
    os.chmod(hook_path, st.st_mode | stat.S_IEXEC)

    print(f"✅ Installed pre-push hook at {hook_path}")

if __name__ == "__main__":
    install_pre_push_hook()
    # Also run pre-commit install
    try:
        subprocess.run(["pre-commit", "install"], check=True)
        print("✅ Pre-commit hooks installed.")
    except Exception as e:
        print(f"⚠️ Failed to run pre-commit install: {e}")
