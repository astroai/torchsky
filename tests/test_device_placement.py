"""Device-follows-input placement and float64-on-MPS contracts."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchsky.device import (
    default_compute_device,
    require_float64,
    resolve_device,
    resolve_fit_device,
)
from torchsky.map import Observation as ImageObservation
from torchsky.map.image_plane_coadd import coadd_image_plane

_ACCELERATORS = [
    name
    for name, ok in (
        ("cuda", torch.cuda.is_available()),
        ("mps", torch.backends.mps.is_available()),
    )
    if ok
]
needs_accelerator = pytest.mark.skipif(
    not _ACCELERATORS, reason="no CUDA or MPS accelerator available"
)
needs_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)


class _OnDevice:
    def __init__(self, device: str) -> None:
        self.device = torch.device(device)


def _header() -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": 8,
        "NAXIS2": 8,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 4.0,
        "CRPIX2": 4.0,
        "CRVAL1": 12.0,
        "CRVAL2": -3.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
    }


def _obs() -> ImageObservation:
    return ImageObservation(data=torch.ones(8, 8), header=_header())


def test_resolve_device_follows_tensor_even_when_default_differs(monkeypatch) -> None:
    monkeypatch.setenv("TORCHSKY_DEVICE", "meta")
    assert default_compute_device().type == "meta"
    assert resolve_device(None, torch.zeros(2)).type == "cpu"
    assert resolve_device(None, [_obs()]).type == "cpu"
    assert resolve_device(None, nn.Linear(1, 1)).type == "cpu"
    assert resolve_fit_device(nn.Linear(1, 1), None).type == "cpu"


def test_resolve_device_explicit_wins() -> None:
    assert resolve_device("cpu", _OnDevice("cuda:0")).type == "cpu"
    assert resolve_device(torch.device("meta"), torch.zeros(2)).type == "meta"


def test_resolve_device_path_inputs_use_default(monkeypatch) -> None:
    monkeypatch.setenv("TORCHSKY_DEVICE", "meta")
    assert resolve_device(None, [ImageObservation(path="a.fits")]).type == "meta"
    assert resolve_device(None).type == "meta"


def test_resolve_device_mixed_devices_raise() -> None:
    with pytest.raises(ValueError, match="different devices"):
        resolve_device(None, torch.zeros(1), _OnDevice("cuda:0"))
    # An explicit device resolves the conflict without moving anything.
    assert resolve_device("cpu", torch.zeros(1), _OnDevice("cuda:0")).type == "cpu"


@needs_accelerator
def test_cpu_observation_returns_cpu_with_accelerator_default(monkeypatch) -> None:
    monkeypatch.setenv("TORCHSKY_DEVICE", _ACCELERATORS[0])
    result = coadd_image_plane([_obs()], target_header=_header())
    assert result.data.device.type == "cpu"


@needs_accelerator
def test_explicit_device_wins_over_cpu_input() -> None:
    result = coadd_image_plane(
        [_obs()], target_header=_header(), device=_ACCELERATORS[0]
    )
    assert result.data.device.type == _ACCELERATORS[0]


@needs_accelerator
def test_mixed_observation_devices_raise() -> None:
    other = ImageObservation(
        data=torch.ones(8, 8, device=_ACCELERATORS[0]), header=_header()
    )
    with pytest.raises(ValueError, match="different devices"):
        coadd_image_plane([_obs(), other], target_header=_header())


def test_require_float64_semantics() -> None:
    require_float64("cpu", what="x")
    require_float64(None, what="x")
    with pytest.raises(NotImplementedError, match="x needs float64.*MPS"):
        require_float64("mps", what="x")


@needs_mps
def test_float64_entry_points_raise_on_mps() -> None:
    from torchsky.map.field.models_encoder import (
        SkyEncoderMLPField,
        SkyEncoderResMLPField,
    )
    from torchsky.map.field.models_implicit import (
        LearnedSpectralBasisField,
        SpectralBasisField,
    )
    from torchsky.map.image_plane_projection import _forward_project_weighted_to_target
    from torchsky.map.operators.geometry_conservative import (
        _exact_conservative_polygon_project,
    )
    from torchsky.map.operators.geometry_drizzle import (
        exact_point_drizzle_project,
        exact_square_drizzle_project,
    )
    from torchsky.map.periodic_affine import resample_linear_affine_periodic_bilinear_2d
    from torchsky.map.spectral_affine import resample_linear_affine_spectral_2d
    from torchsky.map.spectral_nonuniform import resample_nonuniform_spectral_2d

    m = torch.device("mps")
    img = torch.ones(4, 4, device=m)
    coords = torch.zeros(3, device=m)
    aff = dict(a00=1.0, a01=0.0, a10=0.0, a11=1.0, bx=0.0, by=0.0)
    proj = dict(
        value_semantics="surface_brightness",
        pixel_area_sr_map_fn=None,
        chunk_pixels=16,
    )
    calls = [
        (
            "drizzle",
            lambda: exact_square_drizzle_project(
                img, img, None, None, drizzle_pixfrac=1.0, **proj
            ),
        ),
        ("drizzle", lambda: exact_point_drizzle_project(img, img, None, None, **proj)),
        (
            "conservative",
            lambda: _exact_conservative_polygon_project(img, img, None, None, **proj),
        ),
        (
            "splat",
            lambda: _forward_project_weighted_to_target(
                img,
                img,
                None,
                None,
                resampling_kernel="splat_box",
                value_semantics="surface_brightness",
                drizzle_pixfrac=1.0,
            ),
        ),
        ("nonuniform", lambda: resample_nonuniform_spectral_2d(img, img, img)),
        ("affine spectral", lambda: resample_linear_affine_spectral_2d(img, **aff)),
        ("periodic", lambda: resample_linear_affine_periodic_bilinear_2d(img, **aff)),
        (
            "SpectralBasisField",
            lambda: SpectralBasisField.forward(None, coords, coords, 5000.0),
        ),
        (
            "LearnedSpectralBasisField",
            lambda: LearnedSpectralBasisField.forward(None, coords, coords, 5000.0),
        ),
        (
            "SkyEncoderMLPField",
            lambda: SkyEncoderMLPField.forward(None, coords, coords, 5000.0),
        ),
        (
            "SkyEncoderResMLPField",
            lambda: SkyEncoderResMLPField.forward(None, coords, coords, 5000.0),
        ),
    ]
    for label, call in calls:
        with pytest.raises(NotImplementedError, match="float64.*MPS"):
            call()


def test_infer_linear_affine_pixel_map_accepts_accelerator_wcs() -> None:
    import torch

    from torchsky.map.operators.resampling import infer_linear_affine_pixel_map
    from torchsky.wcs import WCS

    if not torch.cuda.is_available():
        # MPS WCS evaluation is float32 (~5e-4 px), below the linearity check.
        pytest.skip("needs CUDA")
    accel = "cuda"
    base = {
        "NAXIS": 2,
        "NAXIS1": 32,
        "NAXIS2": 32,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 16.0,
        "CRPIX2": 16.0,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CDELT1": -0.02,
        "CDELT2": 0.02,
    }
    target = dict(base, CRPIX1=18.0)
    cpu = infer_linear_affine_pixel_map(WCS(base), WCS(target), (32, 32))
    dev = infer_linear_affine_pixel_map(
        WCS(base).to(accel), WCS(target).to(accel), (32, 32)
    )
    tol = 1e-9
    for a, b in zip(cpu, dev, strict=True):
        assert abs(a - b) < tol


def test_synfast_returns_on_the_spectrum_device_and_is_seed_stable() -> None:
    import torch

    from torchsky import sphere

    ell = torch.arange(33, dtype=torch.float64)
    cl = 1.0 / ell.clamp(min=1.0) ** 2
    torch.manual_seed(3)
    cpu_alm = sphere.synalm(cl, lmax=32)
    assert cpu_alm.device.type == "cpu"
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    torch.manual_seed(3)
    gpu_alm = sphere.synalm(cl.cuda(), lmax=32)
    assert gpu_alm.device.type == "cuda"
    # The draw happens on the CPU, so a seed gives the same coefficients everywhere.
    torch.testing.assert_close(gpu_alm.cpu(), cpu_alm, rtol=0.0, atol=0.0)
    sky = sphere.synfast(cl.cuda(), nside=8, lmax=16)
    assert sky.device.type == "cuda"
    assert sphere.anafast(sky, lmax=16).device.type == "cuda"


def test_stacked_wcs_forms_report_device_mismatch_clearly() -> None:
    import torch

    from torchsky.wcs import WCS

    accel = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else None)
    )
    if accel is None:
        pytest.skip("no CUDA or MPS")
    header = {
        "NAXIS": 2,
        "NAXIS1": 64,
        "NAXIS2": 64,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 32.0,
        "CRPIX2": 32.0,
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CDELT1": -1e-4,
        "CDELT2": 1e-4,
    }
    wcs = WCS(header)
    pix = torch.rand(8, 2, device=accel) * 64
    # Differentiable forms keep WCS parameters in the graph: mismatch is an error.
    for call in (wcs.pixel_to_world_differentiable, wcs.world_to_pixel_differentiable):
        with pytest.raises(ValueError, match="wcs.to"):
            call(pix)
    # Plain forms follow the coordinates' device, stacked or not, and agree.
    ra, dec = wcs.pixel_to_world(pix[:, 0], pix[:, 1])
    stacked = wcs.pixel_to_world(pix)
    assert ra.device.type == accel and stacked.device.type == accel
    torch.testing.assert_close(stacked[:, 0], ra)
    torch.testing.assert_close(stacked[:, 1], dec)
    assert wcs.world_to_pixel(stacked).device.type == accel
    assert wcs.to(accel).pixel_to_world(pix).device.type == accel


@needs_mps
def test_blot_on_mps_samples_at_the_cpu_coordinates() -> None:
    """MPS blot maps destination pixels to coadd pixels in float64, like coadd does.

    The coadd is the coadd x pixel coordinate, so blot returns, per destination
    pixel, the coadd x position it sampled. At 0.04 arcsec pixels a float32
    world grid moved blot samples by more than a pixel.
    """
    from torchsky.map import blot

    def header(n: int, crval1: float, crval2: float, rot_deg: float) -> dict:
        import math

        cd = 0.04 / 3600.0
        c, s = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
        return {
            "NAXIS": 2,
            "NAXIS1": n,
            "NAXIS2": n,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRPIX1": n / 2 + 0.5,
            "CRPIX2": n / 2 + 0.5,
            "CRVAL1": crval1,
            "CRVAL2": crval2,
            "CD1_1": -cd * c,
            "CD1_2": cd * s,
            "CD2_1": cd * s,
            "CD2_2": cd * c,
        }

    coadd_header = header(64, 150.1, 2.2, 0.0)
    dest_header = header(40, 150.1 + 3e-5, 2.2 - 2e-5, 13.0)
    ramp = torch.arange(64.0).expand(64, 64).clone()

    on_cpu = blot(ramp, coadd_header, target=dest_header)
    on_mps = blot(ramp.to("mps"), coadd_header, target=dest_header, device="mps")

    assert isinstance(on_cpu, torch.Tensor) and isinstance(on_mps, torch.Tensor)
    assert on_mps.device.type == "mps"
    covered = on_cpu != 0
    assert bool(covered.any())
    error_pixels = (on_mps.cpu() - on_cpu)[covered].abs().max()
    assert float(error_pixels) < 1e-3


def _fine_header(
    n: int,
    crval: tuple[float, float],
    *,
    rot_deg: float = 0.0,
    projection: str = "TAN",
    sip: bool = False,
) -> dict:
    import math

    cd = 0.04 / 3600.0
    c, s = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
    suffix = "-SIP" if sip else ""
    header = {
        "NAXIS": 2,
        "NAXIS1": n,
        "NAXIS2": n,
        "CTYPE1": f"RA---{projection}{suffix}",
        "CTYPE2": f"DEC--{projection}{suffix}",
        "CRPIX1": n / 2 + 0.5,
        "CRPIX2": n / 2 + 0.5,
        "CRVAL1": crval[0],
        "CRVAL2": crval[1],
        "CD1_1": -cd * c,
        "CD1_2": cd * s,
        "CD2_1": cd * s,
        "CD2_2": cd * c,
    }
    if sip:
        header.update(
            {"A_ORDER": 2, "B_ORDER": 2, "A_2_0": 1e-5, "B_0_2": -1e-5, "A_1_1": 2e-6}
        )
    return header


@needs_mps
@pytest.mark.parametrize(
    ("source_kw", "target_kw"),
    [({}, {"projection": "SIN"}), ({"sip": True}, {})],
    ids=["tan-to-sin", "sip-to-tan"],
)
def test_coadd_on_mps_samples_world_grid_targets_at_cpu_coordinates(
    source_kw: dict, target_kw: dict
) -> None:
    """Targets that need a world grid map pixels in float64 on MPS too.

    A non-TAN target (no homography) or a SIP source (round-trip gate) took the
    float32 world grid on MPS: samples moved by more than a pixel at 0.04 arcsec
    and the SIP round-trip gate dropped covered pixels.
    """
    source = _fine_header(64, (150.1, 2.2), **source_kw)
    target = _fine_header(40, (150.1 + 3e-5, 2.2 - 2e-5), rot_deg=13.0, **target_kw)
    ramp = torch.arange(64.0).expand(64, 64).clone()  # value == source x pixel
    observations = [ImageObservation(data=ramp, header=source)]

    on_cpu = coadd_image_plane(observations, target_header=target, auto_tile=False)
    on_mps = coadd_image_plane(
        observations, target_header=target, auto_tile=False, device="mps"
    )

    assert on_mps.data.device.type == "mps"
    covered = on_cpu.weight > 0
    assert bool(covered.any())
    assert torch.equal(on_mps.weight.cpu() > 0, covered)
    error_pixels = (on_mps.data.cpu() - on_cpu.data)[covered].abs().max()
    assert float(error_pixels) < 1e-3
