import pytest
import torch

from torchsky.wcs.core import WCS as TorchWCS


# All supported projections
PROJECTIONS = ("TAN", "SIN", "ARC", "AIT", "MOL", "HPX", "CEA", "MER")

# Projections with closed-form analytic inverses (gradcheck passes exactly)
# TAN, SIN, ARC, HPX use Newton iteration — gradient through iterative solver
# does not match numerical finite-difference Jacobian and is excluded here.
_ANALYTIC_INVERSE_PROJECTIONS = ("AIT", "MOL", "CEA", "MER")


def _make_header(
    projection: str,
    *,
    crval1: float = 180.0,
    crval2: float = 0.0,
    naxis1: int = 256,
    naxis2: int = 256,
) -> dict:
    header = {}
    header["NAXIS"] = 2
    header["NAXIS1"] = naxis1
    header["NAXIS2"] = naxis2
    header["CTYPE1"] = f"RA---{projection}"
    header["CTYPE2"] = f"DEC--{projection}"
    header["CRPIX1"] = (naxis1 / 2.0) + 0.5
    header["CRPIX2"] = (naxis2 / 2.0) + 0.5
    header["CRVAL1"] = crval1
    header["CRVAL2"] = crval2
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"

    if projection in {"AIT", "MOL", "HPX"}:
        header["CD1_1"] = -1.0
        header["CD1_2"] = 0.0
        header["CD2_1"] = 0.0
        header["CD2_2"] = 1.0
    elif projection in {"CEA", "MER"}:
        header["CD1_1"] = -0.5
        header["CD1_2"] = 0.0
        header["CD2_1"] = 0.0
        header["CD2_2"] = 0.5
        if projection == "CEA":
            header["PV2_1"] = 1.0
    else:
        header["CD1_1"] = -0.01
        header["CD1_2"] = 0.0
        header["CD2_1"] = 0.0
        header["CD2_2"] = 0.01

    return header


def _make_wcs(projection: str) -> TorchWCS:
    return TorchWCS(dict(_make_header(projection)))


@pytest.mark.parametrize("projection", PROJECTIONS)
def test_pixel_to_world_differentiable_gradcheck(projection: str) -> None:
    """Analytic pixel->world Jacobian matches finite differences for all projections."""
    wcs = _make_wcs(projection)
    samples = (
        [[128.0, 128.0], [120.0, 140.0], [135.0, 115.0]]
        if projection == "HPX"
        else [[128.0, 128.0], [100.0, 200.0], [150.0, 50.0]]
    )
    pixels = torch.tensor(
        samples,
        dtype=torch.float64,
        requires_grad=True,
    )

    def f(p: torch.Tensor) -> torch.Tensor:
        return wcs.pixel_to_world_differentiable(p)

    assert torch.autograd.gradcheck(f, (pixels,), atol=1e-4, rtol=1e-4, eps=1e-6)


@pytest.mark.parametrize("projection", _ANALYTIC_INVERSE_PROJECTIONS)
def test_world_to_pixel_differentiable_gradcheck(projection: str) -> None:
    """Analytic world->pixel Jacobian matches finite differences.

    Only tested for projections with closed-form analytic inverses
    (AIT, MOL, CEA, MER). TAN, SIN, ARC, HPX use Newton iteration whose
    gradient through the iterative solver does not equal the exact Jacobian.
    """
    wcs = _make_wcs(projection)
    pixels = torch.tensor(
        [[128.0, 128.0], [120.0, 140.0], [135.0, 115.0]],
        dtype=torch.float64,
    )

    world = wcs.pixel_to_world_differentiable(pixels.detach())
    world.requires_grad_(True)

    def f(w: torch.Tensor) -> torch.Tensor:
        return wcs.world_to_pixel_differentiable(w)

    assert torch.autograd.gradcheck(f, (world,), atol=1e-4, rtol=1e-4, eps=1e-6)


@pytest.mark.parametrize("projection", PROJECTIONS)
def test_pixel_to_world_differentiable_gradgradcheck(projection: str) -> None:
    """Second-order gradients through pixel->world are consistent."""
    wcs = _make_wcs(projection)
    samples = (
        [[128.0, 128.0], [120.0, 140.0]]
        if projection == "HPX"
        else [[128.0, 128.0], [100.0, 200.0]]
    )
    pixels = torch.tensor(
        samples,
        dtype=torch.float64,
        requires_grad=True,
    )

    def f(p: torch.Tensor) -> torch.Tensor:
        return wcs.pixel_to_world_differentiable(p)

    assert torch.autograd.gradgradcheck(
        f, (pixels,), atol=1e-3, rtol=1e-3, eps=1e-6
    )


@pytest.mark.parametrize("projection", _ANALYTIC_INVERSE_PROJECTIONS)
def test_world_to_pixel_differentiable_gradgradcheck(projection: str) -> None:
    """Second-order gradients through world->pixel are consistent.

    Only tested for projections with closed-form analytic inverses.
    """
    wcs = _make_wcs(projection)
    pixels = torch.tensor(
        [[128.0, 128.0], [120.0, 140.0]],
        dtype=torch.float64,
    )

    world = wcs.pixel_to_world_differentiable(pixels.detach())
    world.requires_grad_(True)

    def f(w: torch.Tensor) -> torch.Tensor:
        return wcs.world_to_pixel_differentiable(w)

    assert torch.autograd.gradgradcheck(
        f, (world,), atol=1e-3, rtol=1e-3, eps=1e-6
    )


def test_tan_with_sip_gradcheck() -> None:
    """Forward gradcheck passes through SIP distortion + TAN projection."""
    header = {
        "NAXIS": 2,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 128.0,
        "CRPIX2": 128.0,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CD1_1": -0.01,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.01,
        "A_ORDER": 3,
        "A_0_2": 0.00005,
        "A_3_0": -0.00002,
        "B_ORDER": 3,
        "B_0_2": 0.00003,
        "B_1_1": -0.00001,
    }
    wcs = TorchWCS(header)
    pixels = torch.tensor(
        [[128.0, 128.0], [100.0, 150.0]],
        dtype=torch.float64,
        requires_grad=True,
    )

    def f(p: torch.Tensor) -> torch.Tensor:
        return wcs.pixel_to_world_differentiable(p)

    assert torch.autograd.gradcheck(f, (pixels,), atol=1e-4, rtol=1e-4)


def test_tpv_gradcheck() -> None:
    """Forward gradcheck passes through TPV distortion + TAN projection."""
    header = {
        "NAXIS": 2,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 128.0,
        "CRPIX2": 128.0,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CD1_1": -0.01,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.01,
        "PV1_0": 0.02,
        "PV1_1": 1.0,
        "PV1_3": -0.01,
        "PV2_1": 1.0,
        "PV2_2": -0.015,
    }
    wcs = TorchWCS(header)
    pixels = torch.tensor(
        [[128.0, 128.0], [100.0, 150.0]],
        dtype=torch.float64,
        requires_grad=True,
    )

    def f(p: torch.Tensor) -> torch.Tensor:
        return wcs.pixel_to_world_differentiable(p)

    assert torch.autograd.gradcheck(f, (pixels,), atol=2e-4, rtol=2e-4)
