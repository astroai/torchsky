"""Outlier noise fallback is unit-explicit: ADU and electrons agree given gain."""

import pytest
import torch

from torchsky.map import Observation, OutlierConfig, detect_outliers


def _header(n=32):
    return {
        "NAXIS": 2,
        "NAXIS1": n,
        "NAXIS2": n,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": n / 2 + 0.5,
        "CRPIX2": n / 2 + 0.5,
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CDELT1": -0.001,
        "CDELT2": 0.001,
    }


def _electron_frames(seed=0, n=32):
    g = torch.Generator().manual_seed(seed)
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij"
    )
    sky = 400.0 + 3000.0 * torch.exp(-6 * (x**2 + y**2))
    frames = [
        torch.poisson(sky, generator=g) + 8.0 * torch.randn(n, n, generator=g)
        for _ in range(3)
    ]
    frames[0][10, 10] += 1500.0  # cosmic ray
    frames[1][20, 5] += 600.0
    return frames


def _masks(frames, gain):
    obs = [Observation(data=f, wcs=_header()) for f in frames]
    masks, _ = detect_outliers(
        obs,
        config=OutlierConfig(snr=(5.0, 3.0), scale=(1.2, 0.7)),
        gain=gain,
        device="cpu",
    )
    return torch.stack(masks)


@pytest.mark.parametrize("gain", (0.5, 1.7, 4.0))
def test_adu_with_gain_matches_electrons(gain):
    electrons = _electron_frames()
    adu = [f / gain for f in electrons]
    want = _masks(electrons, 1.0)
    got = _masks(adu, gain)
    assert want.any()
    assert torch.equal(want, got)
    assert bool(got[0, 10, 10]) and bool(got[1, 20, 5])


def test_scale_covariance_and_gain_none():
    electrons = _electron_frames(seed=1)
    # Pure rescaling with a covariant gain never changes the decision.
    want = _masks(electrons, 1.0)
    assert torch.equal(_masks([f * 1e3 for f in electrons], 1e-3), want)
    assert torch.equal(_masks([f * 1e-3 for f in electrons], 1e3), want)
    # Without a Poisson term the cosmic rays are still found.
    assert bool(_masks(electrons, None)[0, 10, 10])


def test_gain_validation():
    frames = _electron_frames()
    for bad in (0.0, -1.0, float("inf")):
        with pytest.raises(ValueError, match="gain"):
            _masks(frames, bad)
