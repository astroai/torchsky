import pytest
import torch

from torchsky.map.healpix_image_plane import (
    healpix_nside_from_map_len,
    sample_healpix_map_on_image_header,
    sample_healpix_map_on_tan_header,
)

def test_healpix_nside_from_map_len():
    # Valid nside
    assert healpix_nside_from_map_len(12) == 1
    assert healpix_nside_from_map_len(48) == 2
    assert healpix_nside_from_map_len(192) == 4
    assert healpix_nside_from_map_len(768) == 8

    # Invalid npix (not divisible by 12)
    with pytest.raises(ValueError, match="npix must be divisible by 12"):
        healpix_nside_from_map_len(10)

    # Invalid npix (divisible by 12 but not 12*nside**2)
    with pytest.raises(ValueError, match="npix is not of the form 12\\*nside\\*\\*2"):
        healpix_nside_from_map_len(24)  # 12 * 2 (sqrt(2) is not int)


def test_sample_healpix_map_on_image_header():
    nside = 2
    npix = 12 * nside**2
    h, w = 10, 10

    header = {
        "NAXIS": 2,
        "NAXIS1": w,
        "NAXIS2": h,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": 0.0,
        "CRVAL2": 0.0,
        "CRPIX1": w / 2,
        "CRPIX2": h / 2,
        "CDELT1": -0.1,
        "CDELT2": 0.1,
    }

    # Valid inputs
    healpix_map = torch.ones(npix, dtype=torch.float32)

    # Ring
    result_ring = sample_healpix_map_on_image_header(
        healpix_map, nside=nside, nest=False, target_header=header
    )
    assert result_ring.shape == (h, w)
    assert result_ring.dtype == torch.float32

    # Nest
    result_nest = sample_healpix_map_on_image_header(
        healpix_map, nside=nside, nest=True, target_header=header
    )
    assert result_nest.shape == (h, w)
    assert result_nest.dtype == torch.float32

    # Different device/dtype
    if torch.cuda.is_available():
        healpix_map_cuda = healpix_map.cuda().double()
        result_cuda = sample_healpix_map_on_image_header(
            healpix_map_cuda, nside=nside, nest=False, target_header=header
        )
        assert result_cuda.device.type == "cuda"
        assert result_cuda.dtype == torch.float64

    # Invalid ndim
    with pytest.raises(ValueError, match="healpix_map must be a 1D tensor of length 12\\*nside\\*\\*2"):
        sample_healpix_map_on_image_header(
            torch.ones((12, nside**2)), nside=nside, nest=False, target_header=header
        )

    # Invalid npix match
    with pytest.raises(ValueError, match="does not match nside=2"):
        sample_healpix_map_on_image_header(
            torch.ones(npix + 1), nside=nside, nest=False, target_header=header
        )

    # Invalid target_header NAXIS
    bad_header = header.copy()
    bad_header["NAXIS1"] = 0
    with pytest.raises(ValueError, match="target header must have positive NAXIS1/NAXIS2"):
        sample_healpix_map_on_image_header(
            healpix_map, nside=nside, nest=False, target_header=bad_header
        )

def test_sample_healpix_map_on_tan_header_alias():
    nside = 2
    npix = 12 * nside**2
    h, w = 10, 10

    header = {
        "NAXIS": 2,
        "NAXIS1": w,
        "NAXIS2": h,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": 0.0,
        "CRVAL2": 0.0,
        "CRPIX1": w / 2,
        "CRPIX2": h / 2,
        "CDELT1": -0.1,
        "CDELT2": 0.1,
    }

    healpix_map = torch.ones(npix, dtype=torch.float32)

    result = sample_healpix_map_on_tan_header(
        healpix_map, nside=nside, nest=False, target_header=header
    )
    assert result.shape == (h, w)

    # Check they point to the exact same function or exhibit identical behavior
    assert sample_healpix_map_on_tan_header is sample_healpix_map_on_image_header
