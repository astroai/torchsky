"""Spectral-axis handling for 3D IFU cube assembly.

The target spectral axis of ``coadd_cube`` must be honoured as given, including
non-uniform axes. These tests pin

* exact line localization on an irregular target axis,
* exact normalization for a constant spectrum on an irregular axis,
* interior accuracy for a regular target axis,
* the recorded error envelope for an irregular axis, which is currently an
  index-space interpolation, and
* the explicit rejection of non-increasing axes.
"""

from __future__ import annotations

import math

import pytest
import torch

from torchsky.map import Observation, ResampleConfig, Target, coadd_cube
from torchsky.spectral import SpectralAxis
from torchsky.wcs import WCS
from torchsky.wcs.gwcs_sampling import IFUSliceWCS

N_SLICE = 12
N_WAVE = 400
WAVE_START = 5000.0
WAVE_STEP = 2.0
TARGET_SIZE = 16
TARGET_CDELT_ARCSEC = 2.0

IRREGULAR_AXIS = torch.tensor(
    [5000.0, 5010.0, 5040.0, 5060.0, 5100.0, 5150.0, 5220.0], dtype=torch.float64
)
REGULAR_AXIS = torch.tensor(
    [5000.0, 5020.0, 5040.0, 5060.0, 5080.0, 5100.0], dtype=torch.float64
)
LOG_AXIS = torch.tensor(
    [5000.0 * math.pow(1.02, index) for index in range(8)], dtype=torch.float64
)


def _ifu_wcs() -> IFUSliceWCS:
    rows = torch.arange(N_WAVE, dtype=torch.float64)
    columns = torch.arange(N_SLICE, dtype=torch.float64)
    ra_map = (150.0 + (columns - (N_SLICE - 1) / 2.0) * TARGET_CDELT_ARCSEC / 3600.0)
    ra_map = ra_map.unsqueeze(0).expand(N_WAVE, N_SLICE).clone()
    dec_map = torch.full((N_WAVE, N_SLICE), 2.0, dtype=torch.float64)
    wave_map = (WAVE_START + WAVE_STEP * rows)
    wave_map = wave_map.unsqueeze(1).expand(N_WAVE, N_SLICE).clone()
    slice_map = columns.to(torch.int64).unsqueeze(0).expand(N_WAVE, N_SLICE).clone()
    return IFUSliceWCS(
        ra_map=ra_map,
        dec_map=dec_map,
        wavelength_map=wave_map,
        slice_id_map=slice_map,
    )


def _target() -> Target:
    header = {
        "NAXIS": 2,
        "NAXIS1": TARGET_SIZE,
        "NAXIS2": TARGET_SIZE,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": (TARGET_SIZE + 1) / 2.0,
        "CRPIX2": (TARGET_SIZE + 1) / 2.0,
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CDELT1": -TARGET_CDELT_ARCSEC / 3600.0,
        "CDELT2": TARGET_CDELT_ARCSEC / 3600.0,
    }
    return Target(wcs=WCS(header), shape=(TARGET_SIZE, TARGET_SIZE))


def _cube(
    spectrum,
    axis: torch.Tensor,
    *,
    spectral_kernel: str,
):
    wave_map = torch.arange(N_WAVE, dtype=torch.float64) * WAVE_STEP + WAVE_START
    data = spectrum(wave_map)
    data = data.unsqueeze(1).expand(N_WAVE, N_SLICE).clone().to(torch.float32)
    observation = Observation(
        data=data,
        ifu_wcs=_ifu_wcs(),
        weight=torch.ones_like(data),
    )
    return coadd_cube(
        [observation],
        _target(),
        wavelengths=axis,
        resample=ResampleConfig(spectral_kernel=spectral_kernel),
    )


def _interior_mask(weight: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros_like(weight, dtype=torch.bool)
    mask[1:-1, 2:-2, 2:-2] = True
    return mask & (weight > 0.0)


def _max_relative_error(values: torch.Tensor, truth: torch.Tensor) -> float:
    return float(((values - truth).abs() / truth.abs()).max())


@pytest.mark.parametrize("spectral_kernel", ["nearest", "drizzle", "linear"])
def test_irregular_axis_line_localization(spectral_kernel: str) -> None:
    """A monochromatic detector row must land in the bin that contains its wavelength."""
    for line_index, line_wave in enumerate(IRREGULAR_AXIS.tolist()):
        row = int(round((line_wave - WAVE_START) / WAVE_STEP))

        def spectrum(wave_map: torch.Tensor, row: int = row) -> torch.Tensor:
            data = torch.zeros_like(wave_map)
            data[(wave_map - wave_map[row]).abs() < 1.0e-6] = 7.0
            return data

        result = _cube(spectrum, IRREGULAR_AXIS, spectral_kernel=spectral_kernel)
        per_voxel = result.cube[:, 8, 8]
        assert per_voxel[line_index] > 0.0
        other = torch.cat([per_voxel[:line_index], per_voxel[line_index + 1 :]])
        assert float(other.abs().max()) == 0.0


@pytest.mark.parametrize("spectral_kernel", ["drizzle", "linear"])
def test_irregular_axis_off_grid_sample_spans_one_interval(spectral_kernel: str) -> None:
    row = int(round((5105.0 - WAVE_START) / WAVE_STEP))

    def spectrum(wave_map: torch.Tensor, row: int = row) -> torch.Tensor:
        data = torch.zeros_like(wave_map)
        data[(wave_map - wave_map[row]).abs() < 1.0e-6] = 7.0
        return data

    result = _cube(spectrum, IRREGULAR_AXIS, spectral_kernel=spectral_kernel)
    per_voxel = result.cube[:, 8, 8]
    nonzero = torch.nonzero(per_voxel > 0.0, as_tuple=False).flatten().tolist()
    assert nonzero == [4, 5]


@pytest.mark.parametrize("spectral_kernel", ["nearest", "drizzle", "linear"])
def test_irregular_axis_constant_spectrum_is_normalized(spectral_kernel: str) -> None:
    axis = IRREGULAR_AXIS
    result = _cube(
        lambda wave_map: torch.full_like(wave_map, 5.0),
        axis,
        spectral_kernel=spectral_kernel,
    )
    valid = result.weight > 0.0
    assert bool(valid.any())
    assert torch.allclose(
        result.cube[valid],
        torch.full_like(result.cube[valid], 5.0),
        rtol=0.0,
        atol=1.0e-5,
    )


def _sloped_spectrum(wave_map: torch.Tensor) -> torch.Tensor:
    return 2.0 + 0.01 * (wave_map - WAVE_START)


def _curved_spectrum(wave_map: torch.Tensor) -> torch.Tensor:
    return _sloped_spectrum(wave_map) + 2.0e-5 * (wave_map - WAVE_START) ** 2


def test_regular_axis_interpolates_a_smooth_spectrum() -> None:
    axis = REGULAR_AXIS
    for spectrum in (_sloped_spectrum, _curved_spectrum):
        truth = spectrum(axis).to(torch.float32).view(-1, 1, 1)
        for spectral_kernel in ("drizzle", "linear"):
            result = _cube(spectrum, axis, spectral_kernel=spectral_kernel)
            interior = _interior_mask(result.weight)
            assert bool(interior.any())
            error = _max_relative_error(
                result.cube[interior],
                truth.expand_as(result.cube)[interior],
            )
            assert error <= 2.5e-3


@pytest.mark.parametrize(
    "axis,envelope", [(IRREGULAR_AXIS, 0.10), (LOG_AXIS, 0.15)]
)
def test_irregular_axis_error_envelope(axis: torch.Tensor, envelope: float) -> None:
    """Irregular axes use an index-space kernel: pin its current error envelope.

    This is a recorded limitation, not a correctness claim. A future physical
    wavelength-space interpolation should reduce it; if it regresses past this
    envelope the gate fails.
    """
    result = _cube(_sloped_spectrum, axis, spectral_kernel="drizzle")
    interior = _interior_mask(result.weight)
    truth = _sloped_spectrum(axis).to(torch.float32)
    truth = truth.view(-1, 1, 1).expand_as(result.cube)
    error = _max_relative_error(result.cube[interior], truth[interior])
    assert error <= envelope


def test_non_increasing_axis_is_rejected() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        _cube(
            lambda wave_map: wave_map,
            torch.tensor([5220.0, 5150.0, 5100.0, 5060.0, 5040.0, 5010.0, 5000.0]),
            spectral_kernel="linear",
        )


def test_spectral_axis_reports_regularity_for_irregular_axis() -> None:
    assert not SpectralAxis(values=IRREGULAR_AXIS, unit="Angstrom", type="WAVE").is_regular()
    assert SpectralAxis(values=REGULAR_AXIS, unit="Angstrom", type="WAVE").is_regular()
