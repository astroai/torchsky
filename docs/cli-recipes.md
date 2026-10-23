# CLI recipes

Copy-pasteable shell workflows for the `torchsky` console entry point. Every
command here exists in the argparse tree; run `torchsky <command> --help` for
the authoritative defaults of your installed version.

Paths follow one convention throughout:

| Placeholder | Meaning |
|---|---|
| `/data/exposures/` | input FITS exposures (read through `torchfits`) |
| `/data/weights/`, `/data/masks/` | per-input weight and mask maps |
| `/data/catalogs/` | FITS tables |
| `/work/` | outputs, plans, and scratch files |

FITS decoding and writing belong to `torchfits`; the CLI is an application layer
over the `torchsky` Python API. For pipeline code, use the Python entry points in
[mapmaking](map.md) and the [map reference](reference/map.md).

## Check the environment

```bash
torchsky --help

# Map workflows that download SVO filter profiles need the optional extra.
pip install "torchsky[map]"

# The browser viewer needs FastAPI/uvicorn/Pillow/matplotlib.
pip install "torchsky[viz]"
```

## Plan before you compute

Expand globs, validate inputs, and print the planned output shape and tiling
decision without writing a map. Globs are expanded by the CLI, so quote them to
keep per-input flag counts aligned:

```bash
torchsky map /work/mosaic.fits "/data/exposures/r_*.fits" \
  --dry-run \
  --plan-json /work/mosaic.plan.json
```

`--plan-json` atomically records the expanded paths and HDUs, calibration
inputs, target WCS and shape, tiling decision, method settings, and device.
Review the plan before submitting a batch job, then replay it later.

## Weighted coadd with per-input maps

```bash
torchsky map /work/mosaic.fits \
  /data/exposures/r_0001.fits /data/exposures/r_0002.fits \
  --weights 1.0 0.8 \
  --weight-maps /data/weights/r_0001.wht.fits /data/weights/r_0002.wht.fits \
  --mask-maps /data/masks/r_0001.mask.fits /data/masks/r_0002.mask.fits \
  --photometric-scales 1.0 1.02 \
  --background-offsets 0.0 12.5 \
  --output-planes SCI WHT IVAR MASK NEXP
```

- `--weights` are per-input scalar inverse-variance weights. Without them, the
  coadd uses unit relative weights and does not produce an IVAR plane.
- `--weight-maps` multiply the scalar weights; `--mask-maps` mark contributing
  pixels (nonzero pixels contribute, as stated in `--help`).
- `--dq-maps` supply one detector DQ plane per input and `--dq-hdu` selects its
  HDU (index or `EXTNAME`, default `DQ`); pixels with any set bit are rejected
  by default. `--good-bits` instead reads each `--mask-maps` plane as a bitmask
  and keeps pixels whose remaining bits are zero.
- The count of per-input maps must match the expanded input list; the CLI exits
  with an error when it does not.
- `--output-planes` chooses the written extras. The current default is
  `SCI WHT IVAR MASK NEXP`; `CTX`, `DQ`, and `NCORR` are opt-in.
- Use `--overwrite` / `--no-overwrite` to control replacement of an existing
  output (overwriting is the default).

## Fix the output grid

By default the target WCS is the union footprint of all inputs. Render on an
existing image's grid instead:

```bash
torchsky map /work/mosaic.fits "/data/exposures/r_*.fits" \
  --reference /data/reference_grid.fits \
  --reference-hdu 0 \
  --value-semantics surface_brightness
```

`--value-semantics flux` multiplies by the target pixel area; the choice is
recorded in the product provenance. Use `--union-footprint` /
`--no-union-footprint` when you want the union or the first input grid, and
`--plan-reference N` to pick which input defines the grid when `--reference`
is omitted.

## Drizzle, stacking, and clipping

```bash
torchsky map /work/mosaic_drz.fits "/data/exposures/r_*.fits" \
  --kernel drizzle_square \
  --drizzle-pixfrac 0.8 \
  --value-semantics flux
```

```bash
torchsky map /work/mosaic_clipped.fits "/data/exposures/r_*.fits" \
  --stack sigma_clipped \
  --clip-sigma 4.0 \
  --clip-iterations 3
```

`--workflow` selects a coherent preset instead of spelling out every flag:
`native`, `drizzle`, `swarp`, or `montage` (the preset sets kernel, stacking,
and background policy; explicit flags override it). Available kernels are listed
by `torchsky map --help`; the `drizzle_*` kernels use the native forward-drizzle
path when available.

## Bound memory on large mosaics

```bash
torchsky map /work/mosaic_large.fits "/data/exposures/r_*.fits" \
  --auto-tile \
  --tile-pixels 2048 \
  --scratch-dir /work/scratch \
  --max-footprint-pixels 400000000
```

Use `--tile-above-megapixels` to tune when auto-tiling starts,
`--no-auto-tile` to disable it, and `--max-footprint-pixels` as a hard refusal
guard. `--scratch-dir` locates spillable reducers for large products.

## Render a continuous field

```bash
torchsky map /work/mosaic_field.fits "/data/exposures/r_*.fits" \
  --method field \
  --field-preset balanced \
  --background-mode per-exposure \
  --device cuda
```

`--field-preset` accepts `fast`, `balanced`, `spectral`, and
`operator_constrained`. `--field-amp float16|bfloat16` opts into CUDA autocast
with local gradient scaling; leave it at `none` unless you have measured the
tradeoff. Spectral field runs may need the `map` extra for SVO profiles, or an
explicit `--inr-transmission` / `--inr-transmission-dir` file source.

## One product per filter

```bash
torchsky map /work/coadds/ "/data/exposures/*.fits" --per-filter
```

With `--per-filter` (alias `--per-band`) the output positional is a directory,
inputs are grouped by their FITS `FILTER`, and one product is written per
filter. Without it, inputs from different filters are rejected. With
`--plan-json`, one plan per filter is written using the filter name as a suffix.

## Replay a plan

```bash
torchsky map /work/mosaic_v2.fits \
  --from-plan /work/mosaic.plan.json \
  --scratch-dir /work/scratch_v2
```

`--from-plan` replays a validated schema-v2 plan; only the output path and
scratch location may change. Inputs are omitted from the command line in this
mode.

## Estimate a sky model

```bash
torchsky sky estimate \
  /data/exposures/r_0001.fits \
  /work/sky_r_0001.fits \
  --method mesh \
  --statistic huber \
  --block-size 128 \
  --mask /data/masks/r_0001.mask.fits \
  --device cuda
```

- `--method mesh|polynomial`; mesh accepts `--mesh-filter median_3|gaussian_3|none`
  and `--interpolate bilinear|bicubic`; polynomial accepts `--order`.
- `--statistic median|huber|tukey` is the per-cell robust statistic.
- `--mask` is an optional FITS mask where true pixels are excluded.
- The sky model is written to the output FITS path with the input header.

## Match sky offsets across exposures

```bash
torchsky sky match \
  /data/exposures/r_0001.fits /data/exposures/r_0002.fits \
  --method mesh \
  --statistic median \
  --block-size 128
```

Without `--output`, the command prints one additive offset per input. With an
output pattern, it writes corrected copies (`{i}` is replaced by the input
index):

```bash
torchsky sky match \
  /data/exposures/r_0001.fits /data/exposures/r_0002.fits \
  --masks /data/masks/r_0001.mask.fits /data/masks/r_0002.mask.fits \
  --output "/work/corrected_{i}.fits"
```

Inputs must share a pixel grid. `--masks` must supply one mask per image.

## Browser viewer

```bash
torchsky viz \
  --root /data \
  --host 127.0.0.1 \
  --port 8822 \
  --no-browser
```

The viewer has **no authentication**. Only bind `--host 0.0.0.0` on a trusted
network or behind an authenticated reverse proxy; it can read any file under
`--root`. `--cache-mb` sizes the PNG tile cache, and `--log-level` accepts
`critical`, `error`, `warning`, `info`, or `debug`. For remote clusters, see
[visualization](viz.md).

## Legacy FITS inspection

`torchsky info` and `torchsky cat` remain as migration-only commands. Prefer
`torchfits` for new automation.

```bash
torchsky info /data/exposures/r_0001.fits --headers
torchsky info /data/exposures/r_0001.fits --json

torchsky cat /data/catalogs/catalog.fits \
  --hdu 1 \
  --columns RA,DEC,MAG_G \
  --rows 20 \
  --start-row 1
```

`info` summarizes HDUs without materializing data; `--headers` includes per-HDU
FITS headers and `--json` switches to machine-readable output. `cat` prints a
bounded table sample (JSON with `--json`).

## Related guides

- [CLI reference](cli.md)
- [Mapmaking](map.md) and [map reference](reference/map.md)
- [Visualization](viz.md)
- [I/O boundary](io.md) and [tables](table.md)
