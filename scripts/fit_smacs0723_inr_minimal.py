"""Fit fixed-WCS or bundle-adjusted INR on SMACS J0723 minicutout bands.

Prerequisite:
    pixi run -e default -- python scripts/build_smacs0723_minicutout.py --download

Run from the torchsky repo root:
    pixi run -e test -- python scripts/fit_smacs0723_inr_minimal.py
    pixi run -e test -- python scripts/fit_smacs0723_inr_minimal.py --bundle
    pixi run -e test -- python scripts/fit_smacs0723_inr_minimal.py --bundle --bands F277W F444W
    pixi run -e test -- python scripts/fit_smacs0723_inr_minimal.py --bands all --bundle

Outputs are saved under:
    data/smacs0723_minicutout/

Main FITS:
    smacs_minicut_hyperspectral_inr_<fixed_wcs|bundle_adjustment>_<BANDS>.fits

Quicklooks:
    quicklooks_inr/<BAND>_<fixed_wcs|bundle_adjustment>_observed_rendered_residual.png
    quicklooks_inr/pseudo_rgb_from_cube_<fixed_wcs|bundle_adjustment>_<BANDS>.png
    quicklooks_inr/center_pixel_spectrum_<fixed_wcs|bundle_adjustment>_<BANDS>.png
"""

from __future__ import annotations

# ============================================================
# 0. Imports
# ============================================================

# [COPY FROM YOUR CODE]
import os
from pathlib import Path

# [MY ADDITION]
# This replaces the hard-coded WANTED_BANDS setting with command-line parsing.
import argparse

# [COPY FROM YOUR CODE]
import torch
import matplotlib.pyplot as plt

# [COPY FROM YOUR CODE]
import torchsky
from torchsky.map import (
    FieldObservation,
    estimate_field_domain,
    fit_field_model,
    make_field_model,
    render_field_cube,
)

# [COPY FROM YOUR CODE]
# This is needed for the per-band observed/rendered/residual quicklook.
from torchsky.map import render_field_observation

# [COPY FROM YOUR CODE]
from torchsky.spectral import SpectralAxis, write_cube_fits


# ============================================================
# 1. Parse command-line options
# ============================================================

# [MY ADDITION]
# This is the main change requested:
#   --bundle enables bundle adjustment.
#   --bands controls which bands to fit without editing the file.
parser = argparse.ArgumentParser(
    description="Fit fixed-WCS or bundle-adjusted INR on SMACS J0723 minicutout bands."
)

# [MY ADDITION]
parser.add_argument(
    "--bands",
    nargs="+",
    default=["F277W", "F444W"],
    help=(
        "Bands to fit, e.g. --bands F277W F444W. "
        "Use --bands all to fit all recognized cutout bands. "
        "Default: F277W F444W."
    ),
)

# [MY ADDITION]
parser.add_argument(
    "--bundle",
    action="store_true",
    help="Enable bundle adjustment during INR fitting.",
)

# [MY ADDITION]
args = parser.parse_args()

# [MY ADDITION]
if len(args.bands) == 1 and args.bands[0].lower() == "all":
    WANTED_BANDS = None
else:
    WANTED_BANDS = {band.upper() for band in args.bands}

# [MY ADDITION]
# This prevents fixed-WCS and bundle-adjusted outputs from overwriting each other.
RUN_TAG = "bundle_adjustment" if args.bundle else "fixed_wcs"

# [MY ADDITION]
print("bundle adjustment:", "enabled" if args.bundle else "disabled")
print("wanted bands:", "all" if WANTED_BANDS is None else sorted(WANTED_BANDS))


# ============================================================
# 2. Locate data
# ============================================================

# [COPY FROM YOUR CODE]
# Run Jupyter/script from the repository root, or set TORCHSKY_DATA.
ROOT = Path.cwd()
data_root = Path(os.environ.get("TORCHSKY_DATA", ROOT / "data")).expanduser().resolve()
cut_dir = data_root / "smacs0723_minicutout"

# [COPY FROM YOUR CODE]
paths = sorted(cut_dir.glob("smacs0723_*_cutout_*.fits"))
if not paths:
    raise FileNotFoundError(
        f"No cutouts under {cut_dir}. Run:\n"
        "  pixi run -e default -- python scripts/build_smacs0723_minicutout.py --download"
    )


# ============================================================
# 3. Band metadata
# ============================================================

# [COPY FROM YOUR CODE]
# Approximate effective wavelengths / band support (Angstrom);
# matches benchmark smoke specs.
BAND_META = {
    "F090W": (9000.0, 8000.0, 10000.0),
    "F150W": (15000.0, 13000.0, 17000.0),
    "F200W": (20000.0, 17000.0, 23000.0),
    "F277W": (27700.0, 24000.0, 32000.0),
    "F356W": (35600.0, 31000.0, 40000.0),
    "F444W": (44400.0, 38000.0, 50000.0),
}


# ============================================================
# 4. Build FieldObservation objects
# ============================================================

# [COPY FROM YOUR CODE]
observations = []
ref_header = None

# [COPY FROM YOUR CODE]
for path in paths:
    band = path.stem.split("_")[1].upper()

    # [MY ADDITION]
    # This uses the command-line --bands option.
    if WANTED_BANDS is not None and band not in WANTED_BANDS:
        continue

    # [COPY FROM YOUR CODE]
    if band not in BAND_META:
        continue

    # [COPY FROM YOUR CODE]
    wave, smin, smax = BAND_META[band]
    data, hdr = torchsky.read(str(path), hdu=0, return_header=True)

    # [COPY FROM YOUR CODE]
    if data.ndim != 2:
        data = data.reshape(data.shape[-2], data.shape[-1])

    # [COPY FROM YOUR CODE]
    hdr_d = dict(hdr)

    # [COPY FROM YOUR CODE, WITH YOUR PREVIOUS SMALL FIX KEPT]
    # Keep the first selected band's header as the target WCS.
    if ref_header is None:
        ref_header = hdr_d

    # [COPY FROM YOUR CODE, WITH YOUR PREVIOUS SMALL FIX KEPT]
    # torch.as_tensor(...) is safer than data.to(...) if torchsky.read returns a NumPy-like array.
    observations.append(
        FieldObservation(
            image=torch.as_tensor(data, dtype=torch.float32),
            header=hdr_d,
            wavelength=float(wave),
            spectral_min=float(smin),
            spectral_max=float(smax),
            transmission_center=float(wave),
            band_name=band,
        )
    )

# [COPY FROM YOUR CODE]
if len(observations) < 2:
    raise RuntimeError("Need at least two band cutouts; check filenames and BAND_META.")

# [COPY FROM YOUR CODE]
print("loaded bands:", [o.band_name for o in observations])
print("input shapes:", [tuple(o.image.shape) for o in observations])


# ============================================================
# 5. Fit INR
# ============================================================

# [COPY FROM YOUR CODE]
domain = estimate_field_domain(observations)

# [COPY FROM YOUR CODE]
torch.manual_seed(13)

# [COPY FROM YOUR CODE]
model = make_field_model(
    "hashgrid_spectral_field",
    domain,
    hidden_dim=48,
    num_levels=4,
    log2_hashmap_size=14,
)

if args.bundle:
    # [MY ADDITION]
    # Import only when --bundle is used, so fixed-WCS mode does not depend on this API path.
    from torchsky.map import create_exposure_poses

    # [COPY FROM PREVIOUS BUNDLE ADJUSTMENT SNIPPET]
    # [MY ADAPTATION]
    # Previous idea:
    #   poses = create_exposure_poses(field_observations, anchor_first=True)
    #
    # In this file, the variable is called observations.
    poses = create_exposure_poses(
        observations,
        anchor_first=True,
    )

    # [COPY FROM PREVIOUS BUNDLE ADJUSTMENT SNIPPET]
    # [MY ADAPTATION]
    # The key bundle-adjustment change is:
    #   observation_exposure_poses=poses
    fit = fit_field_model(
        observations,
        model,
        observation_exposure_poses=poses,
        steps=80,
        lr=2e-3,
        loss_kind="charbonnier",
        num_spectral_samples=10,
    )

    # [MY ADDITION]
    print("bundle adjustment: enabled")
    print("anchor observation:", observations[0].band_name)

else:
    # [COPY FROM YOUR CODE]
    # Original fixed-WCS fitting path.
    fit = fit_field_model(
        observations,
        model,
        steps=80,
        lr=2e-3,
        loss_kind="charbonnier",
        num_spectral_samples=10,
    )

    # [MY ADDITION]
    print("bundle adjustment: disabled")

# [COPY FROM YOUR CODE]
print("bands", [o.band_name for o in observations])
print("loss", fit.initial_loss, "->", fit.final_loss)


# ============================================================
# 6. Save per-band INR quicklooks
# ============================================================

# [COPY FROM YOUR CODE]
# For each input band, it saves:
#   observed | rendered | residual
quicklook_dir = cut_dir / "quicklooks_inr"
quicklook_dir.mkdir(parents=True, exist_ok=True)

# [COPY FROM YOUR CODE]
for observation in observations:
    with torch.no_grad():
        observed = observation.image.detach().cpu()
        rendered = render_field_observation(model, observation).detach().cpu()

    residual = observed - rendered

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(observed, origin="lower", cmap="gray")
    axes[0].set_title(f"{observation.band_name} observed")

    axes[1].imshow(rendered, origin="lower", cmap="gray")
    axes[1].set_title(f"{observation.band_name} INR rendered")

    axes[2].imshow(residual, origin="lower", cmap="gray")
    axes[2].set_title(f"{observation.band_name} residual")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()

    # [MY ADDITION]
    # Add RUN_TAG so fixed-WCS and bundle-adjusted diagnostic figures do not overwrite each other.
    quicklook_path = quicklook_dir / (
        f"{observation.band_name}_{RUN_TAG}_observed_rendered_residual.png"
    )

    fig.savefig(quicklook_path, dpi=150)
    plt.close(fig)

    print("wrote", quicklook_path)


# ============================================================
# 7. Render hyperspectral cube
# ============================================================

# [COPY FROM YOUR CODE]
bands_loaded = [o.band_name for o in observations if o.band_name]
lo = min(BAND_META[b][1] for b in bands_loaded)
hi = max(BAND_META[b][2] for b in bands_loaded)

# [COPY FROM YOUR CODE]
# Reasonable wavelength grid across loaded bands.
waves = torch.linspace(float(lo), float(hi), 24, dtype=torch.float64)
cube_axis = SpectralAxis(values=waves, unit="Angstrom", type="WAVE")

# [COPY FROM YOUR CODE]
with torch.no_grad():
    cube = render_field_cube(
        model,
        target_header=ref_header,
        spectral_axis=cube_axis,
    )


# ============================================================
# 8. Write cube FITS
# ============================================================

# [COPY FROM YOUR CODE]
band_tag = "_".join(bands_loaded)

# [MY ADDITION]
# Add RUN_TAG so fixed-WCS and bundle-adjusted FITS files do not overwrite each other.
out_path = cut_dir / f"smacs_minicut_hyperspectral_inr_{RUN_TAG}_{band_tag}.fits"

# [COPY FROM YOUR CODE]
spatial_hdr = {
    k: v
    for k, v in ref_header.items()
    if not str(k).upper().startswith(
        ("NAXIS", "CTYPE3", "CRVAL3", "CDELT3", "CRPIX3", "CUNIT3")
    )
}

# [COPY FROM YOUR CODE]
spatial_hdr["NAXIS"] = 2
spatial_hdr["NAXIS1"] = ref_header["NAXIS1"]
spatial_hdr["NAXIS2"] = ref_header["NAXIS2"]

# [COPY FROM YOUR CODE]
write_cube_fits(
    str(out_path),
    cube.cpu(),
    spatial_header=spatial_hdr,
    spectral_axis=cube_axis,
    overwrite=True,
)

# [COPY FROM YOUR CODE]
print("wrote", out_path)


# ============================================================
# 9. Pseudo-RGB quicklook from cube slices
# ============================================================

# [COPY FROM YOUR CODE]
# Pseudo-RGB from three wavelength slices.
idx = [
    max(0, len(waves) // 6),
    len(waves) // 2,
    min(len(waves) - 1, (5 * len(waves)) // 6),
]


# [COPY FROM YOUR CODE]
def norm(x: torch.Tensor) -> torch.Tensor:
    x = x - x.min()
    return x / (x.max() + 1e-8)


# [COPY FROM YOUR CODE]
r, g, b = tuple(norm(cube[i]) for i in idx)
rgb = torch.stack([r, g, b], dim=0).permute(1, 2, 0).cpu().numpy()

# [COPY FROM YOUR CODE]
plt.figure(figsize=(4, 4))
plt.imshow(rgb, origin="lower")
plt.title("pseudo-RGB from cube slices")
plt.axis("off")
plt.tight_layout()

# [MY ADDITION]
# Add RUN_TAG so fixed-WCS and bundle-adjusted pseudo-RGB quicklooks do not overwrite each other.
rgb_path = quicklook_dir / f"pseudo_rgb_from_cube_{RUN_TAG}_{band_tag}.png"

# [COPY FROM YOUR CODE]
plt.savefig(rgb_path, dpi=150)
print("wrote", rgb_path)

# [COPY FROM YOUR CODE]
plt.show()


# ============================================================
# 10. Center-pixel spectrum quicklook
# ============================================================

# [COPY FROM YOUR CODE]
cy, cx = cube.shape[1] // 2, cube.shape[2] // 2

# [COPY FROM YOUR CODE]
plt.figure(figsize=(6, 3))
plt.plot(waves.numpy(), cube[:, cy, cx].cpu().numpy())
plt.xlabel("wavelength (Angstrom)")
plt.ylabel("model intensity (arb.)")
plt.title(f"spectrum at pixel ({cy}, {cx})")
plt.tight_layout()

# [MY ADDITION]
# Add RUN_TAG so fixed-WCS and bundle-adjusted spectrum quicklooks do not overwrite each other.
spectrum_path = quicklook_dir / f"center_pixel_spectrum_{RUN_TAG}_{band_tag}.png"

# [COPY FROM YOUR CODE]
plt.savefig(spectrum_path, dpi=150)
print("wrote", spectrum_path)

# [COPY FROM YOUR CODE]
plt.show()