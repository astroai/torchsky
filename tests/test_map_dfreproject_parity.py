"""Parity checks vs ``dfreproject.calculate_reprojection`` (DFT-style GPU path when available).

Uses the same weighted numerator/denominator pattern as ``benchmarks/bench_map_compat.py``.
``dfreproject`` is optional: install via ``torchsky[bench]`` (see ``pyproject.toml``).

Requires: ``torchsky.cpp``, ``dfreproject``, ``astropy``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torchsky.cpp")
pytest.importorskip("dfreproject")
from astropy.io import fits  # noqa: E402

from torchsky.map import (  # noqa: E402
    ImageObservation,
    coadd_image_plane,
    plan_image_plane_target,
)


def _tan_header(
    size: int,
    *,
    crval1: float,
    crval2: float,
    cdelt: float = 0.02,
) -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": size / 2.0,
        "CRPIX2": size / 2.0,
        "CRVAL1": crval1,
        "CRVAL2": crval2,
        "CDELT1": -cdelt,
        "CDELT2": cdelt,
    }


def test_torchsky_bilinear_close_to_dfreproject_on_shifted_grid() -> None:
    from dfreproject import calculate_reprojection

    size = 32
    src_hdr = _tan_header(size, crval1=180.0, crval2=0.0)
    tgt_hdr = _tan_header(size, crval1=180.05, crval2=0.02)
    yy, xx = np.mgrid[0:size, 0:size]
    blob = np.exp(
        -((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (2.0 * 4.0**2)
    ).astype(np.float32)
    weight = np.ones_like(blob, dtype=np.float32)
    src_ast = fits.Header(src_hdr)
    tgt_ast = fits.Header(tgt_hdr)

    numer = calculate_reprojection(
        (blob * weight, src_ast),
        tgt_ast,
        shape_out=(size, size),
        order="bilinear",
        conserve_flux=False,
        compute_jacobian=False,
        show_chunk_progress=False,
        show_log=False,
    )
    denom = calculate_reprojection(
        (weight, src_ast),
        tgt_ast,
        shape_out=(size, size),
        order="bilinear",
        conserve_flux=False,
        compute_jacobian=False,
        show_chunk_progress=False,
        show_log=False,
    )
    numer = np.asarray(numer, dtype=np.float32)
    denom = np.asarray(denom, dtype=np.float32)
    numer = np.nan_to_num(numer, nan=0.0, posinf=0.0, neginf=0.0)
    denom = np.nan_to_num(denom, nan=0.0, posinf=0.0, neginf=0.0)
    numer = np.where(denom > 0.0, numer, 0.0)
    df = np.divide(
        numer,
        np.clip(denom, 1e-8, None),
        out=np.zeros_like(numer),
        where=denom > 1e-8,
    ).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "src.fits"
        fits.writeto(path, blob, fits.Header(src_hdr))
        obs = [ImageObservation(path=str(path))]
        plan = plan_image_plane_target(obs, header_overrides=tgt_hdr)
        ts = coadd_image_plane(
            obs,
            target_header=plan.header,
            resampling_kernel="bilinear",
        )
        ts_np = ts.data.detach().cpu().numpy()

    mask = denom > 1e-6
    assert bool(mask.any())
    delta = np.abs(ts_np[mask] - df[mask])
    assert float(delta.max()) < 0.025
