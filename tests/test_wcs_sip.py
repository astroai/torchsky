import pytest
import torch

from torchsky.wcs.core import WCS as TorchWCS


def _sip_header() -> dict:
    header = {}
    header["NAXIS"] = 2
    header["NAXIS1"] = 1024
    header["NAXIS2"] = 1024
    header["CTYPE1"] = "RA---TAN-SIP"
    header["CTYPE2"] = "DEC--TAN-SIP"
    header["CRPIX1"] = 512.0
    header["CRPIX2"] = 512.0
    header["CRVAL1"] = 180.0
    header["CRVAL2"] = 5.0
    header["CD1_1"] = -2.8e-4
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 2.8e-4

    header["A_ORDER"] = 3
    header["B_ORDER"] = 3
    header["A_2_0"] = 3.0e-6
    header["A_1_1"] = -2.0e-6
    header["A_0_2"] = 1.0e-6
    header["B_2_0"] = -2.5e-6
    header["B_1_1"] = 2.0e-6
    header["B_0_2"] = 2.5e-6

    return header


def _high_order_sip_header(order: int) -> dict:
    """Dense SIP forward terms so ``distort`` uses the vectorized ``index_select`` path."""
    header = {}
    header["NAXIS"] = 2
    header["NAXIS1"] = 2048
    header["NAXIS2"] = 2048
    header["CTYPE1"] = "RA---TAN-SIP"
    header["CTYPE2"] = "DEC--TAN-SIP"
    header["CRPIX1"] = 1024.0
    header["CRPIX2"] = 1024.0
    header["CRVAL1"] = 180.0
    header["CRVAL2"] = 0.0
    header["CD1_1"] = -2.8e-5
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 2.8e-5
    header["A_ORDER"] = order
    header["B_ORDER"] = order
    scale = 1e-9
    for p in range(order + 1):
        for q in range(order + 1):
            if p + q > order:
                continue
            header[f"A_{p}_{q}"] = float((p + 1) * scale)
            header[f"B_{p}_{q}"] = float((q + 1) * scale)
    return header


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_sip_wcs_to_cuda_moves_sip_coeffs() -> None:
    twcs = TorchWCS(dict(_sip_header()))
    twcs_cuda = twcs.to("cuda")
    assert twcs_cuda.sip is not None
    assert twcs_cuda.sip._a_pq.device.type == "cuda"
    x = torch.tensor([400.0, 500.0], device="cuda", dtype=torch.float64)
    y = torch.tensor([410.0, 510.0], device="cuda", dtype=torch.float64)
    ra, dec = twcs_cuda.pixel_to_world(x, y, origin=0)
    assert ra.device == x.device
    assert dec.device == x.device
    assert torch.isfinite(ra).all() and torch.isfinite(dec).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_high_order_sip_cuda_pixels_without_explicit_wcs_to() -> None:
    """Regression: CUDA pixel coords + SIP index tensors on CPU must not mix in index_select."""
    order = 6
    twcs = TorchWCS(dict(_high_order_sip_header(order)))
    assert twcs.sip is not None
    n_a = len(twcs.sip.a_terms)
    n_b = len(twcs.sip.b_terms)
    assert n_a + n_b > 24
    x = torch.tensor([512.0], device="cuda", dtype=torch.float64)
    y = torch.tensor([512.0], device="cuda", dtype=torch.float64)
    ra, dec = twcs.pixel_to_world(x, y, origin=0)
    assert ra.device.type == "cuda"
    assert torch.isfinite(ra).all() and torch.isfinite(dec).all()


_STRONG_SIP = {
    # x_d = u + 1e-3 u^2 never goes below -250 px: targets further out on the
    # -x side have no real pre-image, so the Newton inverse cannot converge.
    "quadratic": {"A_ORDER": 2, "B_ORDER": 2, "A_2_0": 1.0e-3, "B_1_1": 2.0e-5},
    # Cubic forward (dense Newton path): the only real pre-image of those
    # targets is ~1e6 px away, far outside the solver's reach.
    "cubic": {
        "A_ORDER": 3,
        "B_ORDER": 3,
        "A_2_0": 1.0e-3,
        "A_3_0": 1.0e-9,
        "B_1_1": 2.0e-5,
    },
    # With AP/BP the solver seeds from the inverse polynomial (active-set path).
    "cubic+APBP": {
        "A_ORDER": 3,
        "B_ORDER": 3,
        "A_2_0": 1.0e-3,
        "A_3_0": 1.0e-9,
        "B_1_1": 2.0e-5,
        "AP_ORDER": 2,
        "BP_ORDER": 2,
        "AP_2_0": -1.0e-3,
        "BP_1_1": -2.0e-5,
    },
}


@pytest.mark.parametrize("variant", list(_STRONG_SIP))
def test_sip_inverse_fails_closed_where_newton_cannot_converge(variant: str) -> None:
    from astropy.wcs import WCS as AstropyWCS
    from astropy.wcs import NoConvergence

    linear = _sip_header()
    for key in [k for k in linear if k.startswith(("A_", "B_", "AP_", "BP_"))]:
        del linear[key]
    linear["CTYPE1"], linear["CTYPE2"] = "RA---TAN", "DEC--TAN"
    header = dict(linear)
    header["CTYPE1"], header["CTYPE2"] = "RA---TAN-SIP", "DEC--TAN-SIP"
    header.update(_STRONG_SIP[variant])

    # Sky positions whose undistorted (CD-only) pixel offset is u_d.
    u_d = torch.tensor([-400.0, -300.0, -100.0, 0.0, 150.0, 300.0], dtype=torch.float64)
    v_d = torch.tensor([20.0, -40.0, 60.0, 0.0, -80.0, 100.0], dtype=torch.float64)
    crpix = float(header["CRPIX1"]) - 1.0, float(header["CRPIX2"]) - 1.0
    ra, dec = TorchWCS(dict(linear)).pixel_to_world(u_d + crpix[0], v_d + crpix[1])
    unreachable = u_d < -250.0

    wcs = TorchWCS(dict(header))
    for x, y in (
        wcs.world_to_pixel(ra, dec),
        wcs.world_to_pixel(torch.stack((ra, dec), dim=-1)).unbind(-1),
    ):
        assert torch.isnan(x[unreachable]).all() and torch.isnan(y[unreachable]).all()
        assert torch.isfinite(x[~unreachable]).all()
        assert torch.isfinite(y[~unreachable]).all()

    oracle = AstropyWCS(header)
    world = torch.stack((ra, dec), dim=-1).numpy()
    with pytest.raises(NoConvergence):
        oracle.all_world2pix(world[unreachable.numpy()], 0, maxiter=50)
    # Reachable targets: astropy's fixed-point all_world2pix itself stalls on
    # this strong distortion, so check the solutions through its forward map.
    x, y = wcs.world_to_pixel(ra[~unreachable], dec[~unreachable])
    back = oracle.all_pix2world(torch.stack((x, y), dim=-1).numpy(), 0)
    torch.testing.assert_close(
        torch.from_numpy(back),
        torch.from_numpy(world[~unreachable.numpy()]),
        atol=1e-9,
        rtol=0.0,
    )
