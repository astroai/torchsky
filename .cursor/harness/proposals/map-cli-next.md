# `torchsky map` CLI: repair help and expose a stable plan record

## Audit result (2026-07-22)

The map command is already a substantial application layer: it expands globs
and MEF image HDUs, reads and writes through `torchfits`, plans union/reference
WCS grids with `plan_map_product`, has memory guards and automatic tiling, and
shares the target-product representation with the Python map/INR APIs.  It
does **not** need another command hierarchy before those behaviours are made
reliable and inspectable.

There is an immediate UX regression: `torchsky map --help` crashes under
Python 3.13 with:

```
ValueError: unsupported format character ')' (0x29)
```

`argparse` interpolates percent signs in `help=` strings; the literal `5%` in
`src/torchsky/cli/map_command.py` must be written as `5%%`.  This also prevents
the root `torchsky --help` path from rendering because argparse formats all
subcommand help strings recursively.

## Smallest high-impact implementation

1. Escape that percent literal and add a regression test that calls
   `build_parser().parse_args(["map", "--help"])` and expects `SystemExit(0)`.
   Capture stdout and assert that the help contains `--union-footprint` and the
   human-facing text `5%`, rather than the escaped implementation spelling.
2. In the same bounded CLI milestone, add `--plan-json PATH` (valid with
   `--dry-run` and normal execution) to write an atomic, reproducible plan
   record **before** compute.  The record should be a stdlib `json` mapping
   containing only public execution facts:
   - schema version, `method`, output path, expanded input paths and selected
     HDUs;
   - output shape, planned header/WCS mapping, union-footprint/reference
     settings, value semantics;
   - coadd kernel/stack/pixfrac/tile decision or INR model/band/fit settings;
   - device request and resolved device;
   - calibration/map-path arrays after glob expansion.

   It must not encode tensors, timestamps, random state, environment dumps, or
   opaque private objects.  Normal execution still writes its FITS output using
   `torchfits`; the JSON is only an application-layer provenance sidecar.  A
   failure to create the plan record should fail before fitting/coadding, so a
   job cannot silently produce an untracked map.

## Why this first

The existing `--dry-run` prints a compact stderr line, but is not machine
readable and loses the chosen WCS, MEF expansion, and resolved tile plan—the
facts most needed before a large map job and to reproduce it later.  A plan
sidecar builds directly on existing planning instead of reimplementing map or
WCS logic, keeps all FITS boundaries in `torchfits`, and is compatible with the
current positional form:

```
torchsky map mosaic.fits 'inputs/*.fits' --plan-json mosaic.plan.json --dry-run
```

No new runtime dependency is required (`json`, `os`, and `pathlib` only).

## Test plan

- CLI help: `torchsky map --help` and `torchsky --help` exit successfully;
  ensure the visible padding text remains `5%`.
- Coadd dry-run using small Torchfits-written FITS fixtures: plan JSON exists,
  includes the final expanded inputs, planned output shape/header, default
  `bilinear`/`weighted_mean`, and does **not** call `stack_map_product` or
  create the FITS output.
- Normal coadd: plan JSON and FITS output both exist; read the FITS output with
  `torchfits`/`torchsky.read` exactly as the existing CLI integration test.
- Bad/unwritable `--plan-json`: fail before `stack_map_product`/INR fit.
- INR dry-run: record resolved method/model/render settings but do not read
  field observations, resolve remote SVO metadata, or fit.
- Verification: focused `pixi run -e test -- pytest tests/test_cli_map.py
  tests/test_map_image_plane.py -q`, then `pixi run preflight-push` before
  push. Full local tests remain opt-in under current repository policy.

## Deferred deliberately

- Splitting `map` into `coadd`/`inr` subcommands would break the working
  positional syntax and is not required for usability yet.
- Auto background estimation (`--sky`) is currently a removed pipeline feature
  with an honest error message.  It needs a scientifically validated primitive,
  not a CLI toggle revival.
- Full external provenance standards / a database are premature; the sidecar
  is a stable, inspectable bridge for local and batch workflows.
