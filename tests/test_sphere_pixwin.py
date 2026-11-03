"""HEALPix pixel windows computed from the pixel shapes."""

from __future__ import annotations

import math

import pytest
import torch

from torchsky.sphere import pixwin
from torchsky.sphere import _pixwin


def _brute_force_window(nside: int, level: int) -> torch.Tensor:
    """Pixel window from the SHT of every pixel's indicator map.

    Each parent pixel is the exact union of its NESTED children at
    ``nside * 2**level``, so this is an independent quadrature of the defining
    sum ``(4 pi / (2l+1)) (1/N) sum_p sum_m |w_lm(p)|^2``.
    """
    from torchsky.sphere.compat import Alm
    from torchsky.sphere.spectral import map2alm_batched

    fine = nside * 2**level
    npix = 12 * nside * nside
    lmax = 3 * nside - 1
    parent = torch.arange(12 * fine * fine) // 4**level
    maps = (parent[None, :] == torch.arange(npix)[:, None]).to(torch.float64)
    alm = map2alm_batched(maps, nside=fine, lmax=lmax, nest=True, iter=3)
    lm = [Alm.getlm(i, lmax) for i in range(alm.shape[-1])]
    ell = torch.tensor([pair[0] for pair in lm])
    m = torch.tensor([pair[1] for pair in lm])
    power = (alm.abs().square() * torch.where(m > 0, 2.0, 1.0)).sum(0).to(torch.float64)
    per_l = torch.zeros(lmax + 1, dtype=torch.float64).index_add_(0, ell, power)
    omega = 4.0 * math.pi / npix
    degree = torch.arange(lmax + 1, dtype=torch.float64)
    return torch.sqrt(4.0 * math.pi / (2.0 * degree + 1.0) * per_l / npix) / omega


@pytest.mark.parametrize("nside", [1, 2])
def test_pixwin_matches_brute_force_indicator_transform(nside: int) -> None:
    # The indicator quadrature converges as (nside/fine)^2; at 64x it is 6e-5.
    torch.testing.assert_close(
        pixwin(nside), _brute_force_window(nside, 6), rtol=0.0, atol=1e-4
    )


def test_pixwin_contract() -> None:
    temperature, polarization = pixwin(16, pol=True)
    assert temperature.shape == polarization.shape == (48,)
    assert temperature.dtype == torch.float64
    assert float(temperature[0]) == pytest.approx(1.0, abs=1e-14)
    assert bool((temperature[1:] < temperature[:-1]).all())
    assert polarization[:2].tolist() == [0.0, 0.0]
    # Spin-2 response at l is close to the spin-0 one at l - 1 (l(l+1) - 4 ~ (l-1)l).
    assert float(polarization[2]) == pytest.approx(float(temperature[1]), abs=1e-5)
    assert pixwin(16, lmax=64).shape == (65,)
    assert torch.equal(pixwin(16, lmax=10), temperature[:11])
    with pytest.raises(ValueError, match="4\\*nside"):
        pixwin(16, lmax=65)
    with pytest.raises(ValueError, match="power of two"):
        pixwin(12)
    with pytest.raises(ValueError, match="nside <= 8192"):
        pixwin(16384)
    with pytest.raises(ValueError, match="non-negative"):
        pixwin(16, lmax=-1)


def test_pixwin_converges_as_inverse_nside_squared() -> None:
    # 1 - w_l at fixed l / nside scales as nside^-2 for small pixels; the
    # computed windows follow it to the expected O(nside^-4) correction.
    w32 = pixwin(32)
    w64 = pixwin(64)
    ratio = (1.0 - w64[2]) / (1.0 - w32[2])
    assert float(ratio) == pytest.approx(0.25, rel=1e-3)


def test_pixwin_disk_cache_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TORCHSKY_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(_pixwin, "_DISK_CACHE_MIN_NSIDE", 4)
    _pixwin._memory_cache.pop(4, None)
    first = pixwin(4, pol=True)
    files = list((tmp_path / "pixwin").glob("pixwin_n4_*.npz"))
    assert len(files) == 1
    _pixwin._memory_cache.pop(4, None)
    second = pixwin(4, pol=True)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    files[0].write_bytes(b"damaged")
    _pixwin._memory_cache.pop(4, None)
    torch.testing.assert_close(pixwin(4, pol=True), first, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("nside", [2, 8, 32, 128, 256])
def test_pixwin_matches_healpy(nside: int) -> None:
    healpy = pytest.importorskip("healpy")
    try:
        ref_t, ref_p = healpy.pixwin(nside, pol=True)
    except Exception as error:  # healpy downloads its tables on first use
        pytest.skip(f"healpy pixel-window table unavailable: {error}")
    got_t, got_p = pixwin(nside, pol=True)
    # The HEALPix tables for nside >= 256 carry ~1e-5 errors of their own: the
    # computed windows keep the exact nside^-2 scaling there, the tables do not.
    atol_t = 1e-11 if nside <= 128 else 3e-5
    torch.testing.assert_close(
        got_t, torch.as_tensor(ref_t, dtype=torch.float64), rtol=0.0, atol=atol_t
    )
    # HEALPix's polarization frame convention differs at O(nside^-4).
    atol_p = max(3e-4 * (2.0 / nside) ** 4, atol_t)
    torch.testing.assert_close(
        got_p, torch.as_tensor(ref_p, dtype=torch.float64), rtol=0.0, atol=atol_p
    )
