"""Shared tiny field and fit for the exposure-fit tests: synthetic records, their conversion to FieldObservation, and a cached tiny fit."""

from functools import lru_cache

import numpy as np
import torch

from _exposure_fit_synth import SynthInstrument, make_field
from torchsky.map.exposure_fit import SkyFitConfig
from torchsky.map.field.types import FieldObservation

RA0, DEC0 = 150.0, 2.0


def tiny_field(seed=0):
    insts = [
        SynthInstrument(
            "A",
            0.10,
            24,
            6.5,
            0.07,
            1.5,
            space_grad=0.15,
            filters=[("A1", 1.0, 1.3), ("A2", 1.6, 2.1)],
        ),
        SynthInstrument(
            "B", 0.12, 20, 6.5, 0.10, 2.4, space_grad=0.15, filters=[("B1", 2.0, 2.8)]
        ),
    ]
    exps, *_ = make_field(
        seed=3 + seed,
        n_per_filter=4,
        n_held=1,
        insts=insts,
        shift_rms=0.03,
        centre_range=0.4,
        rot_choices=(0, 1, 2, 3),
        patch_px=12,
        truth_half=1.0,
    )
    return exps


def to_observation(e, box_half=0.8):
    """A FieldObservation equivalent to a synthetic record: a TAN header whose tangent plane at (RA0, DEC0) reproduces the record's X, Y."""
    ny, nx = e.d.shape
    J = np.asarray(e.extra["J"])  # (X, Y) per (col, row) [arcsec per pixel]
    cd = np.array([[-J[0, 0], -J[0, 1]], [J[1, 0], J[1, 1]]]) / 3600.0  # east = -X
    # pixel (zero-based) where X = Y = 0: solve  J p + (X0, Y0) = 0  with (X0, Y0) the position of pixel (0, 0)
    p = np.linalg.solve(J, -np.array([float(e.X[0, 0]), float(e.Y[0, 0])]))
    header = {
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": RA0,
        "CRVAL2": DEC0,
        "CRPIX1": p[0] + 1.0,
        "CRPIX2": p[1] + 1.0,
        "CD1_1": cd[0, 0],
        "CD1_2": cd[0, 1],
        "CD2_1": cd[1, 0],
        "CD2_2": cd[1, 1],
        "NAXIS1": nx,
        "NAXIS2": ny,
        "TSINST": "NIRCAM" if e.inst_name == "A" else "WFC3_IR",
        "TSNAME": e.name,
        "TSSPLIT": e.split,
        "TSGROUP": e.group,
        "TSBOXRA": RA0,
        "TSBOXDE": DEC0,
        "TSBOXHS": box_half,
        "TSDETNX": nx,
        "TSDETNY": ny,
        "DETECTOR": e.det_name,
    }
    b = e.bandpass
    return FieldObservation(
        image=e.d.clone(),
        header=header,
        mask=e.valid.clone(),
        err=e.sig.clone(),
        band_name=b.name,
        transmission_wavelength=torch.tensor(b.lam_um),
        transmission_throughput=torch.tensor(b.T),
        spectral_unit="um",
        value_unit="MJy/sr",
        value_semantics="surface_brightness",
    )


def tiny_observations(seed=0):
    return [to_observation(e) for e in tiny_field(seed)]


def tiny_config(steps=12, seed=0):
    return SkyFitConfig(
        sky_pitch=0.08,
        item_count=300,
        spectral_k=6,
        kernel_extent=1.0,
        reference_instrument=0,
        steps=steps,
        budget_minutes=1e6,
        patches_per_step=3,
        seed=seed,
        device="cpu",
    )


# private settings for the small field (what the old public "advanced" block held)
TINY_INTERNAL = dict(
    render_pitch=0.04,
    level_nh=8,
    region_px=13,
    n_lambda=3,
    k_hidden=16,
    warm_kernel_steps=30,
    inner_sky=1,
    val_every=2,
    held_steps=10,
    held_reg_steps=6,
    log_every=5,
    is_snr=10.0,
    reg_prepass_steps=8,
    reg_prepass_iters=1,
)


def tiny_fit(exposures, steps=12, seed=0):
    """Private fit of records with the small settings (the public fit_sky cannot take them: they are not public)."""
    from torchsky.map.exposure_fit._skyfit import _fit_exposures

    return _fit_exposures(
        exposures, tiny_config(steps, seed), internal=TINY_INTERNAL, sky_half_extent=1.8
    )


@lru_cache(maxsize=None)
def cached_fit(seed=0, steps=12):
    torch.set_num_threads(1)
    exps = tiny_field()
    return exps, tiny_fit(exps, steps, seed)
