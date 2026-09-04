# torchsky-dev

Use this skill when working in the **torchsky** repo (agents, CI, local installs).

## Interpreter and commands

1. **Pixi first:** `pixi run preflight-push` before push (git hook). Targeted tests for files you changed. `pixi run ci-local` is optional local GitHub-suite parity, not a push blocker. Never rely on bare `python3` / `python` on PATH for project work.
2. Python **3.12+** supported (**`<3.15`**); see `pyproject.toml` `requires-python`.

## Install surfaces

- **Tests (pip):** `pip install -e ".[test]"` after installing `torch` (matches CI).
- **Bench + legacy comparators (pip):** `pip install -e ".[bench-all]"` after `torch`.
- **Pixi:** environments in `pixi.toml` (`test`, `bench-all`, `sphere-bench`, …).

See **[AGENTS.md](../../../AGENTS.md)** for the full policy.
