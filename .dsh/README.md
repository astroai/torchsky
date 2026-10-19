# torchsky × DeepSeek Harness (dsh)

Repo-local [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)
configuration: agent skills in `.dsh/skills/` and a headless-oriented overlay
in `cordis.patch.yml`. dsh auto-discovers skills from `.dsh/skills/` at the
project root (top project priority); no config is needed for that.

> dsh is a **developer preview** — expect breaking changes. Requires a DeepSeek
> API key, or any OpenAI-compatible endpoint via `DEEPSEEK_BASE_URL`.

## Install

```bash
npm install -g @deepseek-ai/dsh
```

> Do not use `npx -y @deepseek-ai/dsh ...`: on npm ≥ 10/11, npm's arg parser
> swallows launcher flags (`--profile`, `--patch`, `--dump-config` — only
> `--help`/`--version` pass through). A global install avoids the shim
> entirely. `--patch` paths are resolved relative to the CLI package dir, so
> always pass an **absolute** path (use `$PWD/.dsh/cordis.patch.yml`).

## Run

Web UI (needs localhost — not available on CANFAR headless sessions):

```bash
dsh web --patch "$PWD/.dsh/cordis.patch.yml"
```

Headless one-shot (CANFAR-friendly, no web server):

```bash
export DEEPSEEK_API_KEY=sk-...
dsh --profile headless --patch "$PWD/.dsh/cordis.patch.yml" \
  "Review the changed WCS implementation, run pixi run preflight-push and targeted package tests, then summarize any failures"
```

Sanity check before booting the app (prints the composed plugin tree):

```bash
dsh web --patch "$PWD/.dsh/cordis.patch.yml" --dump-config
```

Environment knobs:

- `DEEPSEEK_API_KEY` — required.
- `DEEPSEEK_BASE_URL` — OpenAI-compatible proxy (e.g. local vLLM server).
- `DSH_MODEL` — model override (default `deepseek-v4-flash`).
- `DSH_PERMISSION_MODE=danger-full-access` — for one-shot headless runs on a
  disposable CANFAR VM where `ask` approvals would stall (default is
  `workspace-write` + ask). Never use on a shared machine.
- `DSH_TOOLS_MODE=code` — Code Mode (tools exposed as a typed TypeScript SDK).

## Python SDK (scripted pipelines)

```bash
pip install deepseek-harness-sdk
```

```python
from pathlib import Path
from deepseek_harness import DeepSeekHarness

with DeepSeekHarness(
    provider="deepseek-official",
    model="deepseek-v4-flash",
    cwd=str(Path.cwd()),
    session_root="/tmp/dsh-sessions",
    cordis="path/to/minimal.cordis.yml",  # full composition, e.g. the SDK's jsonrpc-agent example
) as h:
    r = h.run("Inspect the current torchsky API guidance and identify the targeted package checks for a map change")
    print(r.final_response)
```

The SDK needs a full `cordis.yml` composition (not this repo's patch file);
clone the dsh repo for `examples/jsonrpc-agent/minimal.cordis.yml` or write
your own.

## Skills

- `torchsky-dev` — pixi workflow, AGENTS.md design rules, common commands
- `science-analysis` — torch / torchfits / pyarrow analysis patterns
- `map-suite` — bounded core map-contract tests; scientific comparisons are in
  the sibling torchsky-harness repository

Scientific `benchmark-gates` and `canfar` skills are maintained in the
[`torchsky-harness`](https://github.com/astroai/torchsky-harness) checkout.

## Notes

- Sessions are append-only logs (provenance for science runs); reuse a session
  id to continue a durable conversation with persistent bash state.
- `cordis.patch.yml` overrides: 10-minute bash timeout (Pixi headroom)
  and durable full-text session search.
