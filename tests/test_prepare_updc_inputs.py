"""Tests for benchmarks/updc/prepare_inputs.py completion semantics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

BENCHMARKS_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

from updc.prepare_inputs import (  # noqa: E402
    ExposureState,
    TargetRenderSpec,
    _read_state,
    _replace_and_write,
)


def _native_ops_healthy() -> bool:
    try:
        import torch

        from torchsky import _cpp

        _cpp.healpix_ring2nest_torch_cpu(4, torch.arange(12))
        return True
    except Exception:
        return False


# The completion path dispatches through native WCS projection ops; skip
# cleanly where the extension ABI is currently broken (macOS arm64 + conda
# torch 2.13 at main; linux CI builds and runs these).
pytestmark = pytest.mark.skipif(
    not _native_ops_healthy(),
    reason="torchsky native extension non-functional on this platform",
)

N = 32


def _header() -> fits.Header:
    return fits.Header(
        {
            "NAXIS": 2,
            "NAXIS1": N,
            "NAXIS2": N,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRPIX1": (N + 1) / 2,
            "CRPIX2": (N + 1) / 2,
            "CRVAL1": 10.0,
            "CRVAL2": -30.0,
            "CD1_1": -3e-5,
            "CD1_2": 0.0,
            "CD2_1": 0.0,
            "CD2_2": 3e-5,
        }
    )


def _write_exposure(path: Path, data: np.ndarray, dq: np.ndarray) -> None:
    fits.PrimaryHDU().writeto(path, overwrite=True)
    with fits.open(path, mode="update") as hdul:
        hdul.append(fits.ImageHDU(data=data.astype(np.float32), name="SCI"))
        hdul.append(fits.ImageHDU(data=dq.astype(np.int32), name="DQ"))


def _make_state(tmp_path: Path, *, seed_tag: str) -> tuple[ExposureState, Path]:
    path = tmp_path / f"exposure_{seed_tag}.fits"
    rng = np.random.default_rng(0)
    data = rng.normal(0.0, 0.01, size=(N, N)).astype(np.float32)
    dq = np.zeros((N, N), dtype=np.int32)
    _write_exposure(path, data, dq)
    state = _read_state(path, dq_fix=False)
    # Pre-mark two statistical outliers so replacement has work to do.
    state.rejected[3, 3] = True
    state.rejected[10, 17] = True
    return state, path


def _target(tmp_path: Path) -> TargetRenderSpec:
    reference = tmp_path / "reference.fits"
    fits.PrimaryHDU().writeto(reference, overwrite=True)
    with fits.open(reference, mode="update") as hdul:
        hdul.append(fits.ImageHDU(data=np.zeros((N, N), np.float32), name="SCI"))
        hdul["SCI"].header.update(_header())
    with fits.open(reference) as hdul:
        return TargetRenderSpec.from_header(
            dict(hdul["SCI"].header), value_semantics="surface_brightness"
        )


def _flat_stats(target: TargetRenderSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    location = np.full(target.shape, 0.0, dtype=np.float32)
    scale = np.full(target.shape, 0.01, dtype=np.float32)
    usable = np.ones(target.shape, dtype=np.float32)
    return location, scale, usable


def test_seed_determinism(tmp_path: Path) -> None:
    state_a, source_a = _make_state(tmp_path, seed_tag="a")
    target = _target(tmp_path)
    stats = _flat_stats(target)

    out_one = tmp_path / "run1"
    rows_one = _replace_and_write(
        [_read_state(source_a, dq_fix=False)],
        target=target,
        location=stats[0],
        scale=stats[1],
        usable_sky=stats[2],
        output_dir=out_one,
        seed=1234,
        device=__import__("torch").device("cpu"),
        dq_fix=False,
    )
    out_two = tmp_path / "run2"
    rows_two = _replace_and_write(
        [_read_state(source_a, dq_fix=False)],
        target=target,
        location=stats[0],
        scale=stats[1],
        usable_sky=stats[2],
        output_dir=out_two,
        seed=1234,
        device=__import__("torch").device("cpu"),
        dq_fix=False,
    )
    data_one = fits.getdata(rows_one[0]["output"], "SCI")
    data_two = fits.getdata(rows_two[0]["output"], "SCI")
    assert np.array_equal(data_one, data_two)


def test_updcmask_marks_exactly_replaced_pixels(tmp_path: Path) -> None:
    torch = __import__("torch")
    state, source = _make_state(tmp_path, seed_tag="b")
    target = _target(tmp_path)
    location, scale, usable = _flat_stats(target)
    out_dir = tmp_path / "masked"

    rows = _replace_and_write(
        [state],
        target=target,
        location=location,
        scale=scale,
        usable_sky=usable,
        output_dir=out_dir,
        seed=7,
        device=torch.device("cpu"),
        dq_fix=False,
    )
    destination = Path(rows[0]["output"])
    completed = fits.getdata(destination, "SCI")
    original = fits.getdata(source, "SCI")
    replaced = completed != original
    mask = fits.getdata(destination, "UPDCMASK").astype(bool)
    assert np.array_equal(mask, replaced)
    assert mask.sum() >= 2  # the two pre-marked outliers were completed


def test_dq_fix_preserves_other_flags(tmp_path: Path) -> None:
    torch = __import__("torch")
    state, source = _make_state(tmp_path, seed_tag="c")
    # Pixel A: DO_NOT_USE bit only -> replaced under --dq-fix.
    # Pixel B: SATURATED bit (value 4) -> preserved verbatim under --dq-fix.
    do_not_use_pixel = (5, 5)
    saturated_pixel = (6, 6)
    state.dq[do_not_use_pixel] = 1
    state.dq[saturated_pixel] = 4

    target = _target(tmp_path)
    location, scale, usable = _flat_stats(target)
    rows = _replace_and_write(
        [state],
        target=target,
        location=location,
        scale=scale,
        usable_sky=usable,
        output_dir=tmp_path / "dqfix",
        seed=11,
        device=torch.device("cpu"),
        dq_fix=True,
    )
    destination = Path(rows[0]["output"])
    completed = fits.getdata(destination, "SCI")
    original = fits.getdata(source, "SCI")
    mask = fits.getdata(destination, "UPDCMASK").astype(bool)

    assert mask[do_not_use_pixel]
    assert not mask[saturated_pixel]  # other DQ flags are not replaceable
    assert completed[saturated_pixel] == original[saturated_pixel]


def test_read_state_valid_masks(tmp_path: Path) -> None:
    path = tmp_path / "dq_policies.fits"
    data = np.ones((4, 4), dtype=np.float32)
    data[1, 1] = np.nan
    dq = np.zeros((4, 4), dtype=np.int64)
    dq[2, 2] = 1  # DO_NOT_USE
    dq[3, 3] = 4  # non-DO_NOT_USE flag
    _write_exposure(path, data, dq)

    default = _read_state(path, dq_fix=False)
    assert not default.valid[1, 1] and not default.valid[2, 2] and not default.valid[3, 3]

    conservative = _read_state(path, dq_fix=True)
    # DO_NOT_USE excluded; the flagged-but-valid pixel stays in statistics.
    assert not conservative.valid[2, 2]
    assert conservative.valid[3, 3] and not np.isnan(conservative.data[3, 3])
