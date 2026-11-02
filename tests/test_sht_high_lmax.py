"""Scalable ring-FFT SHT at lmax >= 2*nside (regression for dense-basis fallback)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

from torchsky.sphere import alm2map, alm_size, map2alm
from torchsky.sphere import spectral as sp


@pytest.mark.parametrize("nside,lmax", [(16, 16), (16, 32), (16, 47), (16, 100)])
def test_map2alm_ring_path_finite_for_lmax_beyond_3nside(nside: int, lmax: int) -> None:
    g = torch.Generator().manual_seed(1)
    m = torch.randn(12 * nside * nside, generator=g, dtype=torch.float64)
    alm = map2alm(m, nside=nside, lmax=lmax)
    assert alm.shape == (alm_size(lmax, lmax),)
    assert bool(torch.isfinite(alm).all())
    assert bool(torch.isfinite(alm2map(alm, nside=nside, lmax=lmax)).all())


def test_native_recurrence_seed_has_no_overflow_nan_at_degree_1100(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """coeff ~ 2**l overflowed for l >= 1024 (inf * 0 = NaN in the seed)."""
    nside, lmax = 16, 1100
    g = torch.Generator().manual_seed(2)
    m = torch.randn(12 * nside * nside, generator=g, dtype=torch.float64)
    alm = map2alm(m, nside=nside, lmax=lmax)
    assert bool(torch.isfinite(alm).all())
    synth = alm2map(alm, nside=nside, lmax=lmax)
    assert bool(torch.isfinite(synth).all())
    # Independent implementation (rescaled torch recurrence) must agree.
    monkeypatch.setattr(sp, "_SCALAR_RECURRENCE_CPP_ENABLE", False)
    monkeypatch.setattr(sp, "_SCALAR_INTERP_CPP_ENABLE", False)
    torch.testing.assert_close(
        map2alm(m, nside=nside, lmax=lmax), alm, rtol=1e-8, atol=1e-8
    )
    torch.testing.assert_close(
        alm2map(alm, nside=nside, lmax=lmax), synth, rtol=1e-8, atol=1e-8
    )


@pytest.mark.parametrize("nside,lmax", [(8, 23), (16, 95)])
def test_torch_legendre_scan_matches_native(
    nside: int, lmax: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    g = torch.Generator().manual_seed(3)
    m = torch.randn(12 * nside * nside, generator=g, dtype=torch.float64)
    ref_alm = map2alm(m, nside=nside, lmax=lmax)
    ref_map = alm2map(ref_alm, nside=nside, lmax=lmax)
    monkeypatch.setattr(sp, "_SCALAR_RECURRENCE_CPP_ENABLE", False)
    monkeypatch.setattr(sp, "_SCALAR_INTERP_CPP_ENABLE", False)
    alm = map2alm(m, nside=nside, lmax=lmax)
    torch.testing.assert_close(alm, ref_alm, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(
        alm2map(ref_alm, nside=nside, lmax=lmax), ref_map, rtol=1e-9, atol=1e-9
    )


def test_torch_legendre_scan_survives_float64_underflow_range() -> None:
    """lambda_mm underflows float64 for m*ln(1/sin) > 745; the scan rescales."""
    theta = torch.tensor([0.05, 0.4, 1.2, 1.5707963267948966], dtype=torch.float64)
    lmax = 1200
    got = {}
    for ell, k, lam in sp._scalar_legendre_scan(
        theta, m0=1000, m1=1001, lmax=lmax, device=torch.device("cpu")
    ):
        got[ell] = lam[0].clone()
    assert all(bool(torch.isfinite(v).all()) for v in got.values())
    # Orthonormality sanity at the equator ring: sum over l of lam^2 is bounded.
    assert float(got[lmax][3].abs()) < 5.0


_HEALPY_SCRIPT = textwrap.dedent(
    """
    import sys, numpy as np, healpy as hp
    nside, lmax = int(sys.argv[1]), int(sys.argv[2])
    m = np.random.default_rng(0).standard_normal(12 * nside * nside)
    np.savez(sys.argv[3], m=m, a=hp.map2alm(m, lmax=lmax, iter=0),
             ai=hp.map2alm(m, lmax=lmax, iter=2))
    """
)


@pytest.mark.parametrize("nside,mult", [(16, 1), (16, 2), (16, 3)])
def test_map2alm_matches_healpy_subprocess(nside: int, mult: int, tmp_path) -> None:
    """healpy runs in a child process (it can segfault when loaded after torch)."""
    lmax = 3 * nside - 1 if mult == 3 else mult * nside
    child = os.environ.get("TORCHSKY_TEST_HEALPY_PYTHON", sys.executable)
    probe = subprocess.run(
        [child, "-c", "import healpy"], capture_output=True, check=False
    )
    if probe.returncode != 0:
        pytest.skip("healpy not importable in a child process")
    out = tmp_path / "ref.npz"
    subprocess.run(
        [child, "-c", _HEALPY_SCRIPT, str(nside), str(lmax), str(out)],
        check=True,
    )
    ref = np.load(out)
    m = torch.from_numpy(ref["m"])
    for key, it in (("a", 0), ("ai", 2)):
        got = map2alm(m, nside=nside, lmax=lmax, iter=it).numpy()
        np.testing.assert_allclose(got, ref[key], rtol=1e-9, atol=1e-9)


def test_native_recurrence_exponent_scaling_matches_scan_beyond_underflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """lmax > ~1900 with mmax = lmax: sin**m underflows float64 on most rings.

    The native kernel carries a per-(m, ring) 2**256 exponent; the torch scan is
    the independent reference (it never prunes rings).
    """
    nside, lmax = 4, 2600
    g = torch.Generator().manual_seed(11)
    m = torch.randn(12 * nside * nside, generator=g, dtype=torch.float64)
    alm = map2alm(m, nside=nside, lmax=lmax)
    synth = alm2map(alm, nside=nside, lmax=lmax)
    assert bool(torch.isfinite(alm).all()) and bool(torch.isfinite(synth).all())
    monkeypatch.setattr(sp, "_SCALAR_RECURRENCE_CPP_ENABLE", False)
    monkeypatch.setattr(sp, "_SCALAR_INTERP_CPP_ENABLE", False)
    ref_alm = map2alm(m, nside=nside, lmax=lmax)
    ref_synth = alm2map(alm, nside=nside, lmax=lmax)
    assert float((alm - ref_alm).abs().max() / ref_alm.abs().max()) < 1e-10
    assert float((synth - ref_synth).abs().max() / ref_synth.abs().max()) < 1e-10


_HEALPY_HIGH_SCRIPT = textwrap.dedent(
    """
    import sys, numpy as np, healpy as hp
    nside, lmax = int(sys.argv[1]), int(sys.argv[2])
    m = np.random.default_rng(0).standard_normal(12 * nside * nside)
    a = hp.map2alm(m, lmax=lmax, iter=0)
    np.savez(sys.argv[3], m=m, a=a, s=hp.alm2map(a, nside, lmax=lmax))
    """
)


@pytest.mark.parametrize("nside", [512, 1024])
def test_scalar_sht_at_three_nside_matches_healpy(nside: int, tmp_path) -> None:
    """lmax = 3*nside - 1 (the maximal HEALPix bandlimit) against healpy."""
    lmax = 3 * nside - 1
    child = os.environ.get("TORCHSKY_TEST_HEALPY_PYTHON", sys.executable)
    probe = subprocess.run(
        [child, "-c", "import healpy"], capture_output=True, check=False
    )
    if probe.returncode != 0:
        pytest.skip("healpy not importable in a child process")
    out = tmp_path / "ref.npz"
    subprocess.run(
        [child, "-c", _HEALPY_HIGH_SCRIPT, str(nside), str(lmax), str(out)],
        check=True,
    )
    ref = np.load(out)
    m = torch.from_numpy(ref["m"])
    alm = map2alm(m, nside=nside, lmax=lmax).numpy()
    assert np.abs(alm - ref["a"]).max() / np.abs(ref["a"]).max() < 1e-10
    synth = alm2map(torch.from_numpy(ref["a"]), nside=nside, lmax=lmax).numpy()
    assert np.abs(synth - ref["s"]).max() / np.abs(ref["s"]).max() < 1e-9


_HEALPY_POL_SCRIPT = textwrap.dedent(
    """
    import sys, numpy as np, healpy as hp
    nside, lmax = int(sys.argv[1]), int(sys.argv[2])
    rng = np.random.default_rng(0)
    m = rng.standard_normal((3, 12 * nside * nside))
    a = hp.map2alm(m, lmax=lmax, iter=0, pol=True)
    np.savez(sys.argv[3], m=m, a=a, s=np.array(hp.alm2map(a, nside, lmax=lmax, pol=True)))
    """
)


def test_spin_transform_at_three_nside_matches_healpy(tmp_path) -> None:
    """Spin-2 (E/B/Q/U) recurrence keeps its scaled seeds at lmax = 3*nside - 1."""
    nside = 1024
    lmax = 3 * nside - 1
    child = os.environ.get("TORCHSKY_TEST_HEALPY_PYTHON", sys.executable)
    probe = subprocess.run(
        [child, "-c", "import healpy"], capture_output=True, check=False
    )
    if probe.returncode != 0:
        pytest.skip("healpy not importable in a child process")
    out = tmp_path / "ref.npz"
    subprocess.run(
        [child, "-c", _HEALPY_POL_SCRIPT, str(nside), str(lmax), str(out)],
        check=True,
    )
    ref = np.load(out)
    alm = map2alm(torch.from_numpy(ref["m"]), nside=nside, lmax=lmax, pol=True).numpy()
    for i in range(3):
        assert np.abs(alm[i] - ref["a"][i]).max() / np.abs(ref["a"][i]).max() < 1e-9
    synth = alm2map(torch.from_numpy(ref["a"]), nside=nside, lmax=lmax, pol=True)
    for i in range(3):
        err = np.abs(synth[i].numpy() - ref["s"][i]).max() / np.abs(ref["s"][i]).max()
        assert err < 1e-9
