import numpy as np
import torch
from astropy.wcs import WCS as AstropyWCS
import torchsky.wcs as wcs


def create_high_order_sip_header(order=10):
    header = {  # type: ignore
        "NAXIS": 2,
        "CTYPE1": "RA---TAN-SIP",  # type: ignore
        "CTYPE2": "DEC--TAN-SIP",  # type: ignore
        "CRPIX1": 1024.0,
        "CRPIX2": 1024.0,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CD1_1": -0.01,  # type: ignore
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.01,
        "A_ORDER": order,
        "B_ORDER": order,
    }

    # Fill with some non-zero coefficients  # type: ignore
    for i in range(order + 1):
        for j in range(order + 1 - i):  # type: ignore
            if i + j > 1:
                # Decaying coefficients to avoid extreme singularity
                # coeff * 1024^10 should be reasonable
                # 1024^10 ~ 10^30. 1e-35 * 10^30 = 1e-5.  # type: ignore
                header[f"A_{i}_{j}"] = 1e-25 * (1e-1 ** (i + j - 2))  # type: ignore
                header[f"B_{i}_{j}"] = 1e-25 * (1e-1 ** (i + j - 2))  # type: ignore

    return header  # type: ignore


def test_sip_high_order_precision():
    order = 10
    header = create_high_order_sip_header(order)  # type: ignore

    # Astropy WCS
    awcs = AstropyWCS(header)  # type: ignore

    # TorchFits WCS
    twcs = wcs.WCS(header)  # type: ignore

    # Test on a large grid
    pixels = torch.cartesian_prod(
        torch.linspace(0, 2048, 10), torch.linspace(0, 2048, 10)
    ).to(torch.float64)

    # Forward: Pixel -> World  # type: ignore
    expected_world = awcs.all_pix2world(pixels.numpy(), 0)  # type: ignore
    actual_world = twcs.pixel_to_world(pixels, origin=0).numpy()  # type: ignore

    # We expect extreme precision (double precision)
    np.testing.assert_allclose(actual_world, expected_world, atol=1e-10)  # type: ignore
    print(f"SIP Order {order} Forward Transform Passed!")

    # Inverse: World -> Pixel (Newton-Raphson)  # type: ignore
    actual_pix = twcs.world_to_pixel(torch.from_numpy(expected_world), origin=0).numpy()  # type: ignore
    np.testing.assert_allclose(actual_pix, pixels.numpy(), atol=1e-5)  # type: ignore
    print(f"SIP Order {order} Inverse Transform Passed!")


def test_sip_dense_polynomial_handles_image_grid_shapes():
    header = create_high_order_sip_header(order=8)  # type: ignore
    awcs = AstropyWCS(header)  # type: ignore
    twcs = wcs.WCS(header)  # type: ignore

    xs = torch.linspace(0, 2048, 32, dtype=torch.float64)
    ys = torch.linspace(0, 2048, 24, dtype=torch.float64)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    expected_ra, expected_dec = awcs.all_pix2world(xx.numpy(), yy.numpy(), 0)  # type: ignore
    actual = twcs.pixel_to_world(xx, yy, origin=0)
    actual_ra = actual[0].numpy()  # type: ignore
    actual_dec = actual[1].numpy()  # type: ignore

    np.testing.assert_allclose(actual_ra, expected_ra, atol=1e-10)  # type: ignore
    np.testing.assert_allclose(actual_dec, expected_dec, atol=1e-10)  # type: ignore


def create_zpn_header():
    header = {  # type: ignore
        "NAXIS": 2,
        "CTYPE1": "RA---ZPN",  # type: ignore
        "CTYPE2": "DEC--ZPN",  # type: ignore
        "CRPIX1": 1024.0,
        "CRPIX2": 1024.0,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CD1_1": -0.01,  # type: ignore
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.01,
        # Standard ZPN PVs (Unity model: P(r) = r)
        "PV2_1": 1.0,  # C_1 (coeff of R^1)
        "PV2_2": 0.0,  # C_2 (coeff of R^2)
        "PV2_3": 0.0,  # C_3 (coeff of R^3)
    }
    return header  # type: ignore


def test_zpn_precision():
    header = create_zpn_header()  # type: ignore
    awcs = AstropyWCS(header)  # type: ignore
    twcs = wcs.WCS(header)  # type: ignore

    pixels = torch.cartesian_prod(
        torch.linspace(512, 1536, 5), torch.linspace(512, 1536, 5)
    ).to(torch.float64)

    expected_world = awcs.all_pix2world(pixels.numpy(), 0)  # type: ignore
    actual_world = twcs.pixel_to_world(pixels, origin=0).numpy()  # type: ignore

    # ZPN residuals
    np.testing.assert_allclose(actual_world, expected_world, atol=1e-9)  # type: ignore
    print("ZPN Forward Transform Passed!")

    # Inverse
    actual_pix = twcs.world_to_pixel(torch.from_numpy(expected_world), origin=0).numpy()  # type: ignore
    np.testing.assert_allclose(actual_pix, pixels.numpy(), atol=1e-5)  # type: ignore
    print("ZPN Inverse Transform Passed!")


if __name__ == "__main__":
    test_sip_high_order_precision()
    test_zpn_precision()
