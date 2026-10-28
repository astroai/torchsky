from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from torchsky.sphere import alm2lonlat, alm2lonlat_spin, alm2map, alm_index, alm_size, map2alm
from torchsky.sphere.sht import (
    CARGeometry,
    CARSampling,
    HEALPixGeometry,
    SHTAnalysisKind,
    SHTCapabilityError,
    SHTOperation,
    SHTRequest,
    SpinConvention,
    TorchHarmonicsBackend,
    adjoint_synthesize,
    analyze,
    car_synthesize,
    pseudo_analyze,
    synthesize,
)


def _car_header(
    nlat: int = 13,
    nlon: int = 24,
    *,
    sampling: str = "clenshaw-curtis",
    longitude_origin: float = 0.0,
    reverse_latitude: bool = False,
    reverse_longitude: bool = False,
    swap_axes: bool = False,
) -> dict[str, float | int | str]:
    latitude_step = 180.0 / (nlat - 1) if sampling == "clenshaw-curtis" else 180.0 / nlat
    latitude_crpix = (nlat + 1.0) / 2.0
    longitude_step = 360.0 / nlon
    if reverse_latitude:
        latitude_step = -latitude_step
    else:
        latitude_step = -latitude_step
    if reverse_latitude:
        latitude_step = abs(latitude_step)
    if reverse_longitude:
        longitude_step = -longitude_step
    if not swap_axes:
        return {
            "NAXIS1": nlon,
            "NAXIS2": nlat,
            "CTYPE1": "RA---CAR",
            "CTYPE2": "DEC--CAR",
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "CRVAL1": longitude_origin,
            "CRVAL2": 0.0,
            "CRPIX1": 1.0,
            "CRPIX2": latitude_crpix,
            "CDELT1": longitude_step,
            "CDELT2": latitude_step,
        }
    return {
        "NAXIS1": nlat,
        "NAXIS2": nlon,
        "CTYPE1": "DEC--CAR",
        "CTYPE2": "RA---CAR",
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CRVAL1": 0.0,
        "CRVAL2": longitude_origin,
        "CRPIX1": latitude_crpix,
        "CRPIX2": 1.0,
        "CDELT1": latitude_step,
        "CDELT2": longitude_step,
    }


def _geometry(
    nlat: int = 13,
    nlon: int = 24,
    *,
    sampling: str = "clenshaw-curtis",
    longitude_origin: float = 0.0,
    reverse_latitude: bool = False,
    reverse_longitude: bool = False,
    swap_axes: bool = False,
) -> CARGeometry:
    return CARGeometry.from_header(
        _car_header(
            nlat,
            nlon,
            sampling=sampling,
            longitude_origin=longitude_origin,
            reverse_latitude=reverse_latitude,
            reverse_longitude=reverse_longitude,
            swap_axes=swap_axes,
        ),
        spin_convention="iau",
    )


def _random_real_field_alm(lmax: int, *, spin: int = 0, seed: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    leading = (2,) if spin else ()
    real = torch.randn((*leading, alm_size(lmax)), dtype=torch.float64, generator=generator)
    imag = torch.randn((*leading, alm_size(lmax)), dtype=torch.float64, generator=generator)
    for ell in range(lmax + 1):
        imag[..., alm_index(ell, 0, lmax)] = 0.0
    alm = torch.complex(real, imag)
    if spin:
        for m in range(lmax + 1):
            for ell in range(m, min(spin, lmax + 1)):
                alm[..., alm_index(ell, m, lmax)] = 0.0
    return alm


def _packed_inner(left: torch.Tensor, right: torch.Tensor, lmax: int) -> torch.Tensor:
    gains = []
    for m in range(lmax + 1):
        gains.extend([1.0 if m == 0 else 2.0] * (lmax - m + 1))
    gain = torch.tensor(gains, dtype=left.real.dtype, device=left.device)
    return torch.sum(gain * torch.real(torch.conj(left) * right))


def test_geometry_and_request_contracts_are_immutable() -> None:
    geometry = _geometry()
    request = SHTRequest(geometry, 4, SHTOperation.SYNTHESIZE)
    with pytest.raises(FrozenInstanceError):
        setattr(geometry, "longitude_origin_radians", 1.0)
    with pytest.raises(FrozenInstanceError):
        setattr(request, "lmax", 5)


def test_car_geometry_recognizes_clenshaw_curtis_and_fejer1() -> None:
    cc = _geometry(nlat=13, nlon=24, sampling="clenshaw-curtis")
    f1 = _geometry(nlat=12, nlon=24, sampling="fejer-1")
    assert cc.sampling is CARSampling.CLENSHAW_CURTIS
    assert f1.sampling is CARSampling.FEJER1
    assert sum(cc.quadrature_weights) == pytest.approx(2.0, abs=2.0e-15)
    assert sum(f1.quadrature_weights) == pytest.approx(2.0, abs=2.0e-15)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"CTYPE1": "RA---TAN"}, "CTYPE"),
        ({"CDELT1": 10.0}, "full-360"),
        ({"PC1_2": 0.1}, "separable"),
        ({"CRVAL2": 1.0}, "latitude reference"),
        ({"A_ORDER": 2}, "distortion"),
        ({"CUNIT1": "rad"}, "degrees"),
    ],
)
def test_car_geometry_fails_closed(
    update: dict[str, float | int | str], message: str
) -> None:
    header = _car_header()
    header.update(update)
    with pytest.raises(ValueError, match=message):
        CARGeometry.from_header(header, spin_convention="iau")


def test_car_geometry_rejects_unrecognized_latitude_centers() -> None:
    header = _car_header()
    header["CRPIX2"] = float(header["CRPIX2"]) + 0.25
    with pytest.raises(ValueError, match="latitude centers"):
        CARGeometry.from_header(header, spin_convention="iau")


def test_car_geometry_handles_reversed_and_swapped_axes() -> None:
    canonical = _geometry(nlat=9, nlon=16)
    swapped = _geometry(nlat=9, nlon=16, swap_axes=True)
    reversed_geometry = _geometry(
        nlat=9,
        nlon=16,
        reverse_latitude=True,
        reverse_longitude=True,
    )
    alm = _random_real_field_alm(3)
    canonical_map = synthesize(alm, canonical, lmax=3).values
    swapped_map = synthesize(alm, swapped, lmax=3).values
    assert swapped.shape == (16, 9)
    torch.testing.assert_close(swapped_map.transpose(-2, -1), canonical_map)
    reversed_map = synthesize(alm, reversed_geometry, lmax=3).values
    lon_step = 360.0 / reversed_geometry.nlon
    lon = torch.arange(reversed_geometry.nlon, dtype=torch.float64) * -lon_step
    lat = torch.linspace(-90.0, 90.0, reversed_geometry.nlat, dtype=torch.float64)
    lon_grid = lon.unsqueeze(0).expand(reversed_geometry.nlat, -1)
    lat_grid = lat.unsqueeze(1).expand(-1, reversed_geometry.nlon)
    expected = alm2lonlat(alm, lon_grid, lat_grid, lmax=3)
    torch.testing.assert_close(reversed_map, expected, atol=2.0e-12, rtol=2.0e-12)


def test_scalar_analytic_ylm_and_longitude_origin() -> None:
    geometry = _geometry(nlat=11, nlon=20, longitude_origin=37.0)
    lmax = 4
    alm = torch.zeros(alm_size(lmax), dtype=torch.complex128)
    alm[alm_index(3, 2, lmax)] = torch.tensor(0.7 - 0.2j)
    output = synthesize(alm, geometry, lmax=lmax).values
    lon = 37.0 + torch.arange(geometry.nlon, dtype=torch.float64) * (360.0 / geometry.nlon)
    lat = torch.linspace(90.0, -90.0, geometry.nlat, dtype=torch.float64)
    expected = alm2lonlat(
        alm,
        lon.unsqueeze(0).expand(geometry.nlat, -1),
        lat.unsqueeze(1).expand(-1, geometry.nlon),
        lmax=lmax,
    )
    torch.testing.assert_close(output, expected, atol=3.0e-12, rtol=3.0e-12)
    torch.testing.assert_close(output[:, 0], expected[:, 0], atol=3.0e-12, rtol=3.0e-12)


@pytest.mark.parametrize("sampling,nlat", [("clenshaw-curtis", 13), ("fejer-1", 12)])
def test_scalar_quadrature_recovers_bandlimited_alms(sampling: str, nlat: int) -> None:
    geometry = _geometry(nlat=nlat, nlon=24, sampling=sampling)
    alm = _random_real_field_alm(5)
    map_values = synthesize(alm, geometry, lmax=5).values
    result = analyze(map_values, geometry, lmax=5)
    assert result.analysis_kind is SHTAnalysisKind.QUADRATURE
    torch.testing.assert_close(result.values, alm, atol=8.0e-12, rtol=8.0e-12)


def test_scalar_discrete_adjoint_is_exact_for_batches() -> None:
    geometry = _geometry(nlat=9, nlon=16, longitude_origin=19.0)
    lmax = 3
    alm = torch.stack((_random_real_field_alm(lmax, seed=2), _random_real_field_alm(lmax, seed=3)))
    map_values = torch.randn(2, *geometry.shape, dtype=torch.float64)
    synthesized = synthesize(alm, geometry, lmax=lmax).values
    adjoint = adjoint_synthesize(map_values, geometry, lmax=lmax).values
    for batch in range(2):
        lhs = torch.sum(synthesized[batch] * map_values[batch])
        rhs = _packed_inner(alm[batch], adjoint[batch], lmax)
        torch.testing.assert_close(lhs, rhs, atol=2.0e-11, rtol=2.0e-11)


def test_spin2_analytic_eb_and_u_sign_convention() -> None:
    iau = _geometry(nlat=9, nlon=16, longitude_origin=11.0)
    cosmo = CARGeometry.from_header(
        _car_header(nlat=9, nlon=16, longitude_origin=11.0),
        spin_convention="cosmo",
    )
    lmax = 3
    alm = torch.zeros((2, alm_size(lmax)), dtype=torch.complex128)
    alm[0, alm_index(2, 0, lmax)] = 0.4
    alm[1, alm_index(3, 0, lmax)] = -0.3
    iau_map = synthesize(alm, iau, lmax=lmax, spin=2).values
    cosmo_map = synthesize(alm, cosmo, lmax=lmax, spin=2).values
    lon = 11.0 + torch.arange(iau.nlon, dtype=torch.float64) * (360.0 / iau.nlon)
    lat = torch.linspace(90.0, -90.0, iau.nlat, dtype=torch.float64)
    expected = alm2lonlat_spin(
        (alm[0], alm[1]),
        2,
        lon.unsqueeze(0).expand(iau.nlat, -1),
        lat.unsqueeze(1).expand(-1, iau.nlon),
        lmax=lmax,
    )
    torch.testing.assert_close(iau_map, expected, atol=4.0e-12, rtol=4.0e-12)
    torch.testing.assert_close(cosmo_map[0], iau_map[0])
    torch.testing.assert_close(cosmo_map[1], -iau_map[1])


def test_spin2_discrete_adjoint_and_quadrature() -> None:
    geometry = _geometry(nlat=13, nlon=24, longitude_origin=23.0)
    lmax = 5
    alm = _random_real_field_alm(lmax, spin=2)
    map_values = torch.randn(2, *geometry.shape, dtype=torch.float64)
    synthesized = synthesize(alm, geometry, lmax=lmax, spin=2).values
    adjoint = adjoint_synthesize(map_values, geometry, lmax=lmax, spin=2).values
    lhs = torch.sum(synthesized * map_values)
    rhs = _packed_inner(alm[0], adjoint[0], lmax) + _packed_inner(alm[1], adjoint[1], lmax)
    torch.testing.assert_close(lhs, rhs, atol=3.0e-11, rtol=3.0e-11)
    recovered = analyze(synthesized, geometry, lmax=lmax, spin=2).values
    torch.testing.assert_close(recovered, alm, atol=2.0e-11, rtol=2.0e-11)


def test_pseudo_analyze_is_explicit_least_squares() -> None:
    geometry = _geometry(nlat=9, nlon=16)
    alm = _random_real_field_alm(3)
    map_values = synthesize(alm, geometry, lmax=3).values
    result = pseudo_analyze(map_values, geometry, lmax=3, tolerance=1.0e-11, max_iterations=30)
    assert result.analysis_kind is SHTAnalysisKind.LEAST_SQUARES
    assert result.converged is True
    assert result.iterations is not None and result.iterations > 0
    assert result.relative_residual is not None and result.relative_residual < 1.0e-10
    torch.testing.assert_close(result.values, alm, atol=2.0e-10, rtol=2.0e-10)


def test_car_scalar_gradcheck() -> None:
    geometry = _geometry(nlat=7, nlon=12)
    request = SHTRequest(geometry, 2, SHTOperation.SYNTHESIZE)
    alm = _random_real_field_alm(2).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda value: car_synthesize(value, request), (alm,), atol=1.0e-5, rtol=1.0e-5)


def test_car_spin_gradcheck() -> None:
    geometry = _geometry(nlat=7, nlon=12)
    request = SHTRequest(geometry, 2, SHTOperation.SYNTHESIZE, spin=2)
    alm = _random_real_field_alm(2, spin=2).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda value: car_synthesize(value, request), (alm,), atol=2.0e-5, rtol=2.0e-5)


def test_car_synthesis_compiles_fullgraph() -> None:
    geometry = _geometry(nlat=7, nlon=12)
    request = SHTRequest(geometry, 2, SHTOperation.SYNTHESIZE)
    alm = _random_real_field_alm(2)
    compiled = torch.compile(lambda value: car_synthesize(value, request), backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(alm), car_synthesize(alm, request))


def test_car_paths_do_not_use_pixel_by_alm_bases(monkeypatch: pytest.MonkeyPatch) -> None:
    geometry = _geometry(nlat=9, nlon=16)
    alm = _random_real_field_alm(3)

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("dense pixel-by-alm basis requested")

    monkeypatch.setattr("torchsky.sphere.spectral._ylm_basis", fail)
    monkeypatch.setattr("torchsky.sphere.spectral._scalar_alm2map_mat", fail)
    monkeypatch.setattr("torchsky.sphere.spectral._spin_ylm_basis_concat", fail)
    map_values = synthesize(alm, geometry, lmax=3, latitude_block_size=2).values
    recovered = analyze(map_values, geometry, lmax=3, latitude_block_size=2).values
    torch.testing.assert_close(recovered, alm, atol=1.0e-11, rtol=1.0e-11)
    spin_alm = _random_real_field_alm(3, spin=2)
    spin_map = synthesize(
        spin_alm, geometry, lmax=3, spin=2, latitude_block_size=2
    ).values
    spin_recovered = analyze(
        spin_map, geometry, lmax=3, spin=2, latitude_block_size=2
    ).values
    torch.testing.assert_close(spin_recovered, spin_alm, atol=2.0e-11, rtol=2.0e-11)


def test_native_healpix_adapter_preserves_legacy_defaults() -> None:
    nside = 4
    lmax = 3
    geometry = HEALPixGeometry(nside, spin_convention=SpinConvention.IAU)
    alm = _random_real_field_alm(lmax)
    legacy_map = alm2map(alm, nside, lmax=lmax)
    semantic_map = synthesize(alm, geometry, lmax=lmax).values
    torch.testing.assert_close(semantic_map, legacy_map, atol=0.0, rtol=0.0)
    legacy_alm = map2alm(legacy_map, nside=nside, lmax=lmax)
    semantic_alm = analyze(legacy_map, geometry, lmax=lmax).values
    torch.testing.assert_close(semantic_alm, legacy_alm, atol=0.0, rtol=0.0)


def test_torch_harmonics_adapter_fails_for_missing_semantics() -> None:
    geometry = _geometry(nlat=9, nlon=16)
    backend = TorchHarmonicsBackend()
    with pytest.raises(SHTCapabilityError, match="cannot provide adjoint_synthesize"):
        adjoint_synthesize(torch.zeros(geometry.shape), geometry, lmax=3, backend=backend)
    with pytest.raises(SHTCapabilityError, match="cannot provide pseudo_analyze"):
        pseudo_analyze(torch.zeros(geometry.shape), geometry, lmax=3, backend=backend)


def test_torch_harmonics_adapter_rejects_unsupported_solver_controls() -> None:
    geometry = _geometry(nlat=9, nlon=16)
    request = SHTRequest(geometry, 3, operation=SHTOperation.ANALYZE)
    backend = TorchHarmonicsBackend()
    values = torch.zeros(geometry.shape)
    with pytest.raises(SHTCapabilityError, match="does not implement tolerance"):
        backend.execute(values, request, tolerance=1.0e-10, max_iterations=0)
    with pytest.raises(SHTCapabilityError, match="does not implement tolerance"):
        backend.execute(values, request, tolerance=0.0, max_iterations=1)


def test_torch_harmonics_adapter_uses_optional_grid_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _geometry(nlat=9, nlon=16)

    class FakeGrid:
        def __init__(self, *, nlat: int, nlon: int) -> None:
            self.nlat = nlat
            self.nlon = nlon

    class FakeTransform:
        def __init__(self, grid: FakeGrid, *, lmax: int, mmax: int) -> None:
            self.grid = grid
            self.lmax = lmax
            self.mmax = mmax

        def to(self, **kwargs: object) -> FakeTransform:
            return self

    class FakeInverse(FakeTransform):
        def __call__(self, values: torch.Tensor) -> torch.Tensor:
            return torch.zeros(
                (*values.shape[:-2], self.grid.nlat, self.grid.nlon),
                dtype=values.real.dtype,
                device=values.device,
            )

    class FakeForward(FakeTransform):
        def __call__(self, values: torch.Tensor) -> torch.Tensor:
            return torch.zeros(
                (*values.shape[:-2], self.lmax, self.mmax),
                dtype=torch.complex128,
                device=values.device,
            )

    module = SimpleNamespace(
        EquiangularGrid=FakeGrid,
        InverseRealSHT=FakeInverse,
        RealSHT=FakeForward,
    )
    backend = TorchHarmonicsBackend()
    monkeypatch.setattr(backend, "_module", lambda: module)
    alm = torch.zeros(alm_size(3), dtype=torch.complex128)
    synthesized = synthesize(alm, geometry, lmax=3, backend=backend)
    analyzed = analyze(synthesized.values, geometry, lmax=3, backend=backend)
    assert synthesized.values.shape == geometry.shape
    assert analyzed.values.shape == alm.shape
    assert analyzed.backend == "torch-harmonics"


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_car_transform_device_contract(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    geometry = _geometry(nlat=7, nlon=12)
    alm = _random_real_field_alm(2).to(device=device, dtype=torch.complex64)
    map_values = synthesize(alm, geometry, lmax=2).values
    recovered = analyze(map_values, geometry, lmax=2).values
    assert map_values.device.type == device
    assert recovered.device.type == device
    torch.testing.assert_close(recovered, alm, atol=2.0e-5, rtol=2.0e-5)
    spin_alm = _random_real_field_alm(2, spin=2).to(
        device=device, dtype=torch.complex64
    )
    spin_map = synthesize(spin_alm, geometry, lmax=2, spin=2).values
    spin_recovered = analyze(spin_map, geometry, lmax=2, spin=2).values
    assert spin_map.device.type == device
    assert spin_recovered.device.type == device
    torch.testing.assert_close(spin_recovered, spin_alm, atol=3.0e-5, rtol=3.0e-5)
