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
