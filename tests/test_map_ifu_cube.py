from __future__ import annotations

import math

import pytest
import torch

from torchsky.map import (
    CubeTarget,
    DQPolicy,
    IFUObservation,
    ResampleConfig,
    coadd_cube,
    plan_ifu_cube_response,
)
from torchsky.wcs import WCS
from torchsky.wcs.gwcs_sampling import IFUSliceWCS


def _wcs(shape: tuple[int, int], scale_arcsec: float = 1.0) -> WCS:
    height, width = shape
    return WCS(
        {
            "NAXIS": 2,
            "NAXIS1": width,
            "NAXIS2": height,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRPIX1": (width + 1) / 2,
            "CRPIX2": (height + 1) / 2,
            "CRVAL1": 150.0,
            "CRVAL2": 2.0,
            "CDELT1": -scale_arcsec / 3600.0,
            "CDELT2": scale_arcsec / 3600.0,
        }
    )


def _transform(
    shape: tuple[int, int],
    *,
    wave_start: float = 5000.0,
    wave_step: float = 10.0,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> IFUSliceWCS:
    height, width = shape
    y, x = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    ra, dec = _wcs(shape).to(device).pixel_to_world(x, y)
    wavelength = wave_start + wave_step * y
    slice_id = torch.zeros_like(x, dtype=torch.int64)
    return IFUSliceWCS(ra, dec, wavelength, slice_id)


def _independent_identity_overlap_oracle(science: torch.Tensor) -> torch.Tensor:
    output = torch.zeros(
        (science.shape[0], science.shape[0], science.shape[1]), dtype=science.dtype
    )
    for row in range(science.shape[0]):
        for column in range(science.shape[1]):
            spatial_overlap = max(
                0.0, min(column + 0.5, column + 0.5) - max(column - 0.5, column - 0.5)
            )
            spectral_overlap = max(
                0.0, min(row + 1.0, row + 1.0) - max(float(row), float(row))
            )
            output[row, row, column] += (
                science[row, column] * spatial_overlap * spectral_overlap
            )
    return output


def test_physical_drizzle_matches_tiny_overlap_oracle_and_conserves_flux() -> None:
    science = torch.arange(1, 7, dtype=torch.float64).reshape(3, 2).requires_grad_()
    observation = IFUObservation(
        sci=science,
        detector_to_world=_transform((3, 2)),
        value_semantics="flux",
        provenance={"exposure": "tiny"},
    )
    target = CubeTarget(
        _wcs((3, 2)),
        torch.tensor([4995.0, 5005.0, 5015.0, 5025.0], dtype=torch.float64),
        shape=(3, 2),
        value_semantics="flux",
    )
    result = coadd_cube([observation], target, spectral_kernel="drizzle")
    oracle = _independent_identity_overlap_oracle(science.detach())

    torch.testing.assert_close(result.data, oracle, rtol=0.0, atol=2.0e-6)
    torch.testing.assert_close(result.data.sum(), science.sum(), rtol=0.0, atol=2.0e-6)
    assert result.provenance["kernel"] == "drizzle"
    assert result.response_plan is not None
    result.data.sum().backward()
    torch.testing.assert_close(
        science.grad, torch.ones_like(science), atol=2.0e-6, rtol=0.0
    )


def test_world_spectral_pixel_footprint_supports_disjoint_single_pixel_slice() -> None:
    spatial_wcs = _wcs((1, 1))
    vertices = torch.tensor(
        [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
        dtype=torch.float64,
    )
    ra, dec = spatial_wcs.pixel_to_world(vertices[:, 0], vertices[:, 1])
    footprint = torch.stack(
        (ra, dec, torch.tensor([4995.0, 4995.0, 5005.0, 5005.0])), dim=-1
    ).reshape(1, 1, 4, 3)
    transform = IFUSliceWCS(
        torch.tensor([[150.0]], dtype=torch.float64),
        torch.tensor([[2.0]], dtype=torch.float64),
        torch.tensor([[5000.0]], dtype=torch.float64),
        torch.zeros((1, 1), dtype=torch.int64),
    )
    observation = IFUObservation(
        torch.tensor([[3.0]], dtype=torch.float64),
        transform,
        detector_pixel_footprint=footprint,
        value_semantics="flux",
    )
    target = CubeTarget(
        spatial_wcs,
        torch.tensor([4995.0, 5005.0], dtype=torch.float64),
        shape=(1, 1),
        value_semantics="flux",
    )
    result = coadd_cube([observation], target, spectral_kernel="drizzle")
    torch.testing.assert_close(
        result.data, torch.tensor([[[3.0]]], dtype=torch.float64)
    )


def test_one_plan_propagates_variance_dq_context_covariance_and_queries() -> None:
    base = torch.arange(12, dtype=torch.float64).reshape(2, 6).mT
    science = base[::2]
    assert not science.is_contiguous()
    variance = torch.linspace(0.5, 1.0, 6, dtype=torch.float64).reshape(3, 2)
    dq = torch.zeros((3, 2), dtype=torch.int32)
    dq[1, 1] = 2
    observation = IFUObservation(
        science,
        _transform((3, 2)),
        dq=dq,
        variance_components={"poisson": variance, "read": 0.25 * variance},
        dq_policy=DQPolicy(reject_bits=0, unknown_bit_policy="preserve"),
    )
    target = CubeTarget(
        _wcs((3, 2)),
        torch.tensor([4995.0, 5005.0, 5015.0, 5025.0], dtype=torch.float64),
        shape=(3, 2),
    )
    result = coadd_cube(
        [observation],
        target,
        spectral_kernel="drizzle",
        response_chunk_size=1,
    )
    plan = result.response_plan
    source_variance = 1.25 * variance.reshape(-1)
    dense = torch.zeros((result.data.numel(), science.numel()), dtype=torch.float64)
    dense[plan.target_indices, plan.source_indices] += plan.coefficients
    expected_covariance = dense @ torch.diag(source_variance) @ dense.mT

    assert result.dq is not None and int(result.dq[1, 1, 1]) == 2
    assert result.context is not None and result.context.shape == (1, 3, 3, 2)
    assert result.inverse_variance is not None
    torch.testing.assert_close(
        result.covariance.diagonal().reshape(-1),
        torch.diagonal(expected_covariance),
    )
    query = torch.zeros_like(result.data)
    query[:, 1, 1] = 1.0
    uncertainty = result.aperture_uncertainty(
        torch.nn.functional.one_hot(torch.tensor(3), 6).reshape(3, 2),
        torch.ones(3, dtype=torch.float64),
    )
    expected = query.reshape(-1) @ expected_covariance @ query.reshape(-1)
    torch.testing.assert_close(uncertainty.variance, expected)
    torch.testing.assert_close(
        result.covariance.apply_covariance(query).reshape(-1),
        expected_covariance @ query.reshape(-1),
    )
    torch.testing.assert_close(
        plan.apply(science.reshape(-1), chunk_size=1), result.data
    )
    torch.testing.assert_close(
        plan.apply(science.reshape(-1), chunk_size=99), result.data
    )
    trial = torch.randn_like(result.data)
    torch.testing.assert_close(
        torch.sum(plan.apply(science.reshape(-1)) * trial),
        torch.sum(science.reshape(-1) * plan.adjoint(trial)),
    )


@pytest.mark.parametrize("kernel", ["emsm", "msm"])
def test_physical_distance_kernels_use_declared_roi_and_resolving_power(
    kernel: str,
) -> None:
    science = torch.tensor([[7.0]], dtype=torch.float64)
    transform = IFUSliceWCS(
        torch.tensor([[150.0]], dtype=torch.float64),
        torch.tensor([[2.0]], dtype=torch.float64),
        torch.tensor([[5000.0]], dtype=torch.float64),
        torch.zeros((1, 1), dtype=torch.int64),
    )
    observation = IFUObservation(science, transform, value_semantics="flux")
    target = CubeTarget(
        _wcs((9, 9)),
        torch.linspace(4975.0, 5025.0, 11, dtype=torch.float64),
        shape=(9, 9),
        value_semantics="flux",
    )
    result = coadd_cube(
        [observation],
        target,
        spectral_kernel=kernel,
        spatial_roi=lambda wavelength: wavelength.new_tensor(4.1),
        resolving_power=lambda wavelength: wavelength.new_tensor(500.0),
    )

    torch.testing.assert_close(result.data.sum(), science.sum())
    nonzero = torch.nonzero(result.data > 0, as_tuple=False)
    assert int(nonzero[:, 1].max() - nonzero[:, 1].min()) > 2
    assert int(nonzero[:, 2].max() - nonzero[:, 2].min()) > 2
    assert int(nonzero[:, 0].max() - nonzero[:, 0].min()) > 1


def test_synthetic_continuum_and_emission_line_flux_centroid_and_fwhm() -> None:
    shape = (41, 3)
    wavelengths = torch.arange(shape[0], dtype=torch.float64) * 2.0 + 4960.0
    sigma = 6.0
    line = 10.0 * torch.exp(-0.5 * ((wavelengths - 5000.0) / sigma).square())
    spectrum = 2.0 + line
    science = spectrum[:, None].expand(shape).clone()
    spatial_wcs = _wcs((3, 3))
    x = torch.arange(3, dtype=torch.float64)
    ra, dec = spatial_wcs.pixel_to_world(x, torch.ones_like(x))
    trace_offset = (
        torch.arange(shape[0], dtype=torch.float64) - (shape[0] - 1) / 2
    ) * (0.01 / 3600.0)
    transform = IFUSliceWCS(
        ra[None].expand(shape).clone(),
        dec[None].expand(shape).clone() + trace_offset[:, None],
        wavelengths[:, None].expand(shape).clone(),
        torch.zeros(shape, dtype=torch.int64),
    )
    observation = IFUObservation(science, transform)
    edges = torch.arange(4959.0, 5042.0, 2.0, dtype=torch.float64)
    target = CubeTarget(spatial_wcs, edges, shape=(3, 3))
    result = coadd_cube([observation], target, spectral_kernel="drizzle")
    extracted = result.data[:, 1, 1]
    centers = result.spectral_axis.values
    continuum = extracted.new_tensor(2.0)
    line_only = (extracted - continuum).clamp_min(0)
    centroid = torch.sum(centers * line_only) / line_only.sum()
    variance = torch.sum((centers - centroid).square() * line_only) / line_only.sum()
    fwhm = 2.355 * torch.sqrt(variance)

    torch.testing.assert_close(
        extracted[[0, -1]], continuum.expand(2), atol=2.0e-3, rtol=0.0
    )
    torch.testing.assert_close(
        centroid, centroid.new_tensor(5000.0), atol=0.05, rtol=0.0
    )
    torch.testing.assert_close(
        fwhm, fwhm.new_tensor(2.355 * sigma), atol=0.25, rtol=0.0
    )
    observed_flux = torch.sum(line_only * (edges[1:] - edges[:-1]))
    expected_flux = math.sqrt(2.0 * math.pi) * 10.0 * sigma
    torch.testing.assert_close(
        observed_flux, observed_flux.new_tensor(expected_flux), atol=0.4, rtol=0.0
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
    ],
)
def test_ifu_device_dtype_and_memory_bound(device: str) -> None:
    dtype = torch.float32 if device == "mps" else torch.float64
    transform = _transform((2, 2), dtype=dtype, device=device)
    observation = IFUObservation(
        torch.ones((2, 2), dtype=dtype, device=device), transform
    )
    target = CubeTarget(
        _wcs((2, 2)),
        torch.tensor([4995.0, 5005.0, 5015.0], dtype=dtype),
        shape=(2, 2),
    )
    response = plan_ifu_cube_response(
        [observation],
        target,
        device=device,
        chunk_size=2,
        footprint_tolerance=0.2 if device == "mps" else 0.02,
    )
    assert response.plan.coefficients.device.type == device
    assert response.plan.coefficients.dtype == dtype
    with pytest.raises(MemoryError, match="max_response_entries"):
        plan_ifu_cube_response(
            [observation],
            target,
            device=device,
            max_response_entries=1,
            footprint_tolerance=0.2 if device == "mps" else 0.02,
        )


def test_legacy_ifu_coadd_remains_available() -> None:
    transform = _transform((3, 2), dtype=torch.float32)
    legacy = IFUSliceWCS(
        transform.ra_map,
        transform.dec_map,
        transform.wavelength_map,
        transform.slice_id_map,
    )
    from torchsky.map import Observation, Target

    result = coadd_cube(
        [Observation(data=torch.ones((3, 2)), ifu_wcs=legacy)],
        Target(wcs=_wcs((3, 2)), shape=(3, 2)),
        wavelengths=torch.tensor([5000.0, 5010.0, 5020.0]),
        resample=ResampleConfig(spectral_kernel="linear"),
    )
    assert result.data.shape == (3, 3, 2)


def test_legacy_ifu_coadd_accepts_cpu_ifu_wcs_on_device(compute_device) -> None:
    """CPU-resident IFU maps must be moved to the compute device, not mixed with it."""
    from torchsky.map import Observation, Target

    transform = _transform((3, 2), dtype=torch.float32)
    legacy = IFUSliceWCS(
        transform.ra_map,
        transform.dec_map,
        transform.wavelength_map,
        transform.slice_id_map,
    )
    result = coadd_cube(
        [Observation(data=torch.ones((3, 2)), ifu_wcs=legacy)],
        Target(wcs=_wcs((3, 2)), shape=(3, 2)),
        wavelengths=torch.tensor([5000.0, 5010.0, 5020.0]),
        resample=ResampleConfig(spectral_kernel="linear"),
        device=compute_device,
    )
    assert result.data.device.type == compute_device.type
    assert result.data.shape == (3, 3, 2)
