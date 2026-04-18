# torchsky-dev

Use this skill when working in the **torchsky** repo (agents, CI, local installs).

## Interpreter and commands

1. **Pixi first:** `pixi run ci-local` before push; `pixi run preflight-push` for a quick check; `pixi run -e test -- pytest tests/test_foo.py` for narrow tests. Never rely on bare `python3` / `python` on PATH for project work.
2. **uv (no Pixi):** `uv sync --group dev` then `uv run --python 3.12 --group dev ruff check .` from repo root.
3. Python **3.12+** supported (**`<3.15`**); see `pyproject.toml` `requires-python`.

## Install surfaces

- **Tests (pip):** `pip install -e ".[test]"` after installing `torch` (matches CI).
- **Bench + legacy comparators (pip):** `pip install -e ".[bench-all]"` after `torch`.
- **Pixi:** environments in `pixi.toml` (`test`, `bench-all`, `sphere-bench`, …).

See **[AGENTS.md](../../../AGENTS.md)** for the full policy.
