# Legacy Astrometric Calibration Plan

## Motivation

The ELIXIR pipeline WCS on CFHT MegaCam exposures has limited astrometric
accuracy (~0.5–1.0"). This is sufficient for single-epoch detection but causes
visible misregistration when coadding multiple exposures. A full differentiable
astrometric solver is a major engineering effort; this document describes a
legacy (external-tool) pipeline that produces corrected WCS headers compatible
with both SWarp and torchsky.

## Design Choices

### Distortion model: TPV (SCAMP convention)

TPV is the natural choice because:

- **SWarp** consumes TPV natively (PV1_0..PV1_39, PV2_0..PV2_39 on a TAN
  projection or RA---TPV / DEC--TPV CTYPE).
- **torchsky** already supports TPV in both Python and fused C++ paths
  (`src/torchsky/wcs/tpv.py`, `wcs.cpp`), including pixel→world, world→pixel,
  and Jacobians.
- **SCAMP** (the de-facto standard for survey-level relative astrometry)
  produces TPV output by default.
- **MegaCam** optical distortion is well-parameterized by a degree-5 TPV
  polynomial (~30 terms per axis).

SIP is **not** supported by SWarp (confirmed from SWarp source: `fitswcs.c`
has no SIP code, no `A_ORDER`/`B_ORDER` parsing, and no `POLY_DEGREE`
config option). SWarp only handles TPV/PV distortion. Torchsky supports both
SIP and TPV, so the choice depends on the downstream tools. For the SWarp
compatibility path, use TPV (SCAMP convention).

### Per-CCD vs per-exposure

Each MegaCam exposure has 36 CCDs with independent CRVAL/CRPIX but a shared
CD matrix. The approach is:

1. **Per-exposure solution** — Run astrometry.net on a single reference CCD
   (nearest the field centre), then propagate the CD + distortion to all 36
   CCDs via their existing CRPIX offsets. This avoids solving 36×N CCDs.
2. **Per-observation solution** — Run SCAMP on the full MEF (SCAMP handles
   multi-extension FITS natively). This refines per-CCD CRPIX/CRVAL and a
   shared distortion model simultaneously.

Option (2) is preferable for final quality; option (1) is cheaper for testing.

### Reference Catalog

- **GAIA DR3** (via astrometry.net `--no-fits2fits --ra --dec --radius` or the
  `gaia-*` index files) provides ~0.1–0.2 mas absolute reference.
- Default astrometry.net index files cover the full sky at ~0.1"/pix
  (index-4219 through index-4223 for the MegaCam 0.187" scale).

## Pipeline

### Step 0: Environment setup (pixi only)

The `astrometry` Pixi feature provides SCAMP and SExtractor:

```bash
pixi run -e astrometry -- scamp ...
pixi run -e astrometry -- sex ...
```

| Conda package | Tool | Purpose |
|---|---|---|
| `astromatic-scamp` | `scamp` | Multi-CCD relative astrometric refinement |
| `astromatic-source-extractor` | `sex` | Source detection (SCAMP input) |
| `astropy` | — | FITS I/O in helper scripts |

**astrometry.net** (`solve-field`) is available via `pixi exec` (runs under
Rosetta on Apple Silicon, native on Linux):

```bash
# macOS arm64:
pixi exec --platform osx-64 --channel conda-forge -s astrometry -- solve-field --help

# Linux x86_64:
pixi exec --channel conda-forge -s astrometry -- solve-field --help
```

The orchestration script (`astrometric_calibrate.py`) handles platform
detection and invokes `solve-field` via `pixi exec` automatically.

### Step 1: Re-WCS each exposure with astrometry.net

For each exposure's reference CCD (nearest to field centre, typically the
chip that includes the pointing centre):

```bash
solve-field --no-plot \
  --scale-low 0.18 --scale-high 0.20 \
  --tweak-order 3 \
  --cpulimit 60 \
  exposure_ccd.fits \
  -o exposure_ccd.solved
```

Outputs:
- `exposure_ccd.solved` (new WCS in the FITS header)
- `exposure_ccd.wcs` (WCS as a separate header file)

The `--tweak-order 3` flag enables a SIP polynomial fit up to cubic terms,
which captures the linear component of MegaCam's distortion.

### Step 2: Propagate to all CCDs

The `.wcs` file contains a SIP+TAN solution. To propagate to the other 35
CCDs:

```bash
pixi run -e astrometry -- python scripts/apply_wcs_to_mef.py propagate \
    --mef exposure.fits.fz \
    --solved-wcs exposure.solved.wcs \
    --output exposure.propagated.fits
```

The script:
1. Reads the solved CD matrix and SIP/TPV distortion from the `.wcs` file.
2. For each CCD: keeps its CRPIX, applies the solved CD + distortion.
3. Computes CRVAL by evaluating the solved WCS at each CCD's CRPIX.
4. Optionally uses `--ref-hdu N` to extract the solved WCS from an HDU
   within the MEF itself (useful when `solve-field` has modified it in
   place).

Alternatively, to propagate from a reference HDU within the MEF:

```bash
# Solve one HDU in-place
solve-field --no-plots --tweak-order 3 \
    --scale-units arcsecperpix --scale-low 0.184 --scale-high 0.190 \
    exposure.fits.fz --extension 19

# Then propagate to all others
pixi run -e astrometry -- python scripts/apply_wcs_to_mef.py propagate \
    --mef exposure.fits.fz \
    --ref-hdu 19 \
    --output exposure.propagated.fits
```

### Step 3: SCAMP refinement (optional, for best quality)

With propagated WCS from step 2, run SExtractor on the MEF (SCAMP reads
SExtractor catalogs, not images directly):

```bash
# SExtractor: each extension becomes a separate catalog
sex   exposure.propagated.fits[1]  -c default.sex -CATALOG_NAME sex/exposure_1.cat
sex   exposure.propagated.fits[2]  -c default.sex -CATALOG_NAME sex/exposure_2.cat
...
sex   exposure.propagated.fits[36] -c default.sex -CATALOG_NAME sex/exposure_36.cat

# SCAMP: refine all catalogs jointly
scamp sex/exposure_*.cat \
  -c default.scamp \
  -ASTREF_CATALOG GAIA-DR3 \
  -POSITION_MAXERR 1.0 \
  -POSITION_TOPOUT_MAXERR_ARCSEC 1.0 \
  -DISTORT_DEGREES 5 \
  -CROSSID_RADIUS 45 \
  -SOLUTION_METHOD RELATIVE \
  -MOSAIC_TYPE FIX_FOCAL_PLANE
```

SCAMP outputs per-extension `.head` files named `<base>_<N>.head` where `N`
is the extension number (1-indexed). Merge them back into the MEF:

```bash
pixi run -e astrometry -- python scripts/apply_wcs_to_mef.py merge-scamp \
    --mef exposure.propagated.fits \
    --scamp-prefix scamp/exposure \
    --output exposure.calibrated.fits
```

This updates each extension's WCS with SCAMP's refined TPV coefficients.

### Step 4: Coadd with torchsky

After steps 1–3 (or step 1–2 for a lighter pipeline), the exposure FITS files
have corrected WCS in either SIP or TPV format. Feed them to the existing
torchsky `map` command — both SIP and TPV are detected and used automatically
by `torchsky.wcs.WCS` and the coadd resampling code (`wcs_has_distortion`
handles both).

```bash
python -c "from torchsky.cli import main; main()" map \
  mosaic.fits exposure1.fits exposure2.fits ... \
  --hdu 1 \
  --kernel lanczos3
```

## Key Constraints

1. **Per-CCD HDU numbering** — The ELIXIR pipeline numbers HDUs 1..36. The
   `--hdu` auto-expansion in `map_command.py` probes all TensorHDUs and
   expands them, which is compatible with both corrected and uncorrected WCS.

2. **CD vs PC matrix** — ELIXIR uses CDi_j. Both astrometry.net and SCAMP
   output CDi_j. No conversion needed.

3. **CTYPE convention** — ELIXIR uses RA---TAN / DEC--TAN. Astrometry.net
   appends SIP (RA---TAN---SIP). SCAMP produces RA---TPV / DEC--TPV (or
   RA---TAN + PV keywords). Torchsky handles all three automatically.

4. **Compressed FITS (.fits.fz)** — Both astrometry.net and SCAMP read
   compressed FITS (CFITSIO handles `.fits.fz` via imcopy or direct read).
   However, RICE-compressed tiles must be decompressed first:
   `funpack exposure.fits.fz` or `imcopy`.

5. **Memory** — For 10 exposures × 36 CCDs × 16 MPix each, the full pipeline
   needs ~20 GB of RAM during SCAMP processing. The torchsky coadd tiles at
   4096×4096 to stay under memory limits.

## Files to Create / Modify

| File | Purpose |
|---|---|
| `scripts/apply_wcs_to_mef.py` | Propagate a solved WCS to all MEF extensions |
| `scripts/astrometric_calibrate.py` | Orchestrate the full pipeline (astrometry.net → SCAMP → header merge) |
| `pixi.toml` | Add `[feature.astrometry]` environment |
| `AGENTS.md` | Document the astrometric calibration workflow |

## Open Questions

1. Does `solve-field` handle `.fits.fz` compressed MEF extensions directly?
   (Astropy does, but `solve-field` may need `funpack` first.)
2. What is the typical positional RMS after ELIXIR vs after astrometry.net vs
   after SCAMP?
3. Does SCAMP handle 36-CCD MEF as multiple extensions, or must we split into
   separate files? (SCAMP docs say yes, but test needed.)
4. Is `--tweak-order 3` sufficient for MegaCam, or is a higher-order SIP
   needed?
5. Does SCAMP output per-extension `.head` numbered by extension index or by
   input file position?
6. Should `--max-exposures` for the astrometric calibration script differ from
   the coadd script default (currently 10)?

## Out of Scope (This Phase)

- Differentiable astrometric fitting within torchsky
- On-the-fly distortion correction (currently a pre-processing step)
- Distortion solutions for non-CFHT instruments
- SCAMP integration in the `download_cfht_megapipe.py` run command
