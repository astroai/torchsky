# `torchsky` CLI

The `torchsky` console entry point exposes command-line workflows for map and
visualization tasks.

## What This Surface Is For

Use the CLI for runnable workflows, smoke checks, demos, and automation. Use the
Python API when composing tensor-native pipelines inside another application.

## Commands

### `torchsky info`

Prints a read-only FITS HDU summary without materializing image/table data:

```bash
torchsky info image_or_table.fits
torchsky info image_or_table.fits --json
```

Use `--headers` when an automation needs per-HDU header cards in the output.

### `torchsky cat`

Prints a bounded sample from a FITS table HDU:

```bash
torchsky cat catalog.fits --hdu CAT --columns RA,DEC --rows 5
torchsky cat catalog.fits --hdu 1 --rows 5 --json
```

`--start-row` is 1-based, matching the Python table-read API.

### `torchsky viz`

Starts the optional browser viewer. Install the `viz` extra first:

```bash
pip install "torchsky[viz]"
torchsky viz --root /path/to/data --port 8822
```

### `torchsky map`

Runs map-oriented workflows from the CLI. Some INR/spectral-source paths require
the `map` extra for SVO filter profile access:

```bash
pip install "torchsky[map]"
torchsky map --help
```

Classical stacks accept scalar weights plus optional per-input weight and mask
maps:

```bash
torchsky map out.fits img_a.fits img_b.fits \
  --weights 1.0 0.8 \
  --photometric-scales 1.0 1.02 \
  --background-offsets 0.0 12.5 \
  --weight-maps img_a_wht.fits img_b_wht.fits \
  --mask-maps img_a_mask.fits img_b_mask.fits \
  --primary-beam-maps img_a_pb.fits img_b_pb.fits
```

INR renders use the same scalar weighting, weight-map, mask-map, and
background-offset flags. Pass `--reference` when the output should be rendered
on a specific FITS WCS/shape instead of the planned union grid.

Use `--dry-run` before large jobs to validate the expanded input list and print
the planned output shape/tiling decision without writing the map or fitting an
INR.

## Automation Guidance

- Prefer explicit input/output paths and non-interactive flags.
- Keep downloaded data and generated benchmark artifacts out of the repository.
- For repeatable development commands, prefer Pixi tasks documented in
  [benchmarks.md](benchmarks.md), [contributing.md](contributing.md), and
  [release.md](release.md).

## Related Guides

- [viz.md](viz.md)
- [map.md](map.md)
- [install.md](install.md)
