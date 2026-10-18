import math

import numpy as np
import pytest
import torch

from torchsky._native import cpp as _native_cpp
from torchfits import write as write_fits
from torchsky.sphere import compat
from torchsky.sphere import healpix as hp
import torchsky.sphere.spectral as spectral_mod
import torchsky.sphere.harmonics as harmonics_mod
from torchsky.sphere.spectral import (
    alm2lonlat,
    alm2lonlat_spin,
    alm2map,
    alm2map_batched,
    alm2map_der1,
    alm2map_spin,
    alm_index,
    alm_size,
    anafast,
    gaussian_beam,
    map2alm,
    map2alm_batched,
    map2alm_lsq,
    map2alm_spin,
    rotate_alm,
    smoothalm,
    smoothmap,
    synalm,
    synfast,
    _ring_layout_for_nside,
)


def test_harmonics_alias_matches_spectral_module() -> None:
    assert harmonics_mod.__all__ == spectral_mod.__all__
    assert harmonics_mod.map2alm is spectral_mod.map2alm
    assert harmonics_mod.alm2map is spectral_mod.alm2map


def test_compile_helper_falls_back_after_specialization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_calls = 0

    def _broken_compile(fn):
        def _broken(*args, **kwargs):
            nonlocal compile_calls
            compile_calls += 1
            raise RuntimeError("stale compiler cache")

        return _broken

    monkeypatch.setattr(spectral_mod, "_COMPILE_OK", True)
    monkeypatch.setattr(torch, "compile", _broken_compile)
    wrapped = spectral_mod._maybe_compile(lambda value: value.square())
    value = torch.tensor([2.0])
    with pytest.warns(RuntimeWarning, match="using eager execution"):
        torch.testing.assert_close(wrapped(value), torch.tensor([4.0]))
    torch.testing.assert_close(wrapped(value), torch.tensor([4.0]))
    assert compile_calls == 1


try:
    import healpy as _hp  # type: ignore
except Exception:  # pragma: no cover
    _hp = None


def _has_cpp_ring_fourier() -> bool:
    return (
        _native_cpp is not None
        and hasattr(_native_cpp, "_healpix_ring_fourier_modes_cpu")
        and hasattr(_native_cpp, "_healpix_ring_fourier_synthesis_cpu")
    )


def _has_cpp_spin_concat() -> bool:
    return (
        _native_cpp is not None
        and hasattr(_native_cpp, "_healpix_spin_interpolate_concat_cpu")
        and hasattr(_native_cpp, "_healpix_spin_integrate_concat_cpu")
    )


def _has_cpp_spin_ring_fused() -> bool:
    return (
        _native_cpp is not None
        and hasattr(_native_cpp, "_healpix_spin_map2alm_ring_concat_cpu")
        and hasattr(_native_cpp, "_healpix_spin_ring_finalize_cpu")
    )


def _has_cpp_spin_recurrence() -> bool:
    return (
        _native_cpp is not None
        and hasattr(_native_cpp, "_healpix_spin_interpolate_recurrence_cpu")
        and hasattr(_native_cpp, "_healpix_spin_integrate_recurrence_cpu")
    )


def _random_alm(lmax: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    arr = np.zeros(alm_size(lmax), dtype=np.complex128)
    for m in range(lmax + 1):
        for ell in range(m, lmax + 1):
            scale = 1.0 / ((ell + 1) ** 2)
            if m == 0:
                val = scale * rng.normal()
            else:
                val = scale * (rng.normal() + 1j * rng.normal())
            arr[alm_index(ell, m, lmax)] = val
    return torch.from_numpy(arr)


def _random_spin_alms(
    lmax: int, spin: int = 2, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    a_e = np.zeros(alm_size(lmax), dtype=np.complex128)
    a_b = np.zeros(alm_size(lmax), dtype=np.complex128)
    for m in range(lmax + 1):
        for ell in range(m, lmax + 1):
            if ell < spin:
                continue
            scale = 1.0 / ((ell + 1) ** 2)
            idx = alm_index(ell, m, lmax)
            if m == 0:
                a_e[idx] = scale * rng.normal()
                a_b[idx] = scale * rng.normal()
            else:
                a_e[idx] = scale * (rng.normal() + 1j * rng.normal())
                a_b[idx] = scale * (rng.normal() + 1j * rng.normal())
    return torch.from_numpy(a_e), torch.from_numpy(a_b)


def test_alm_size_and_index_ordering() -> None:
    lmax = 5
    assert alm_size(lmax) == 21

    idx = []
    for m in range(lmax + 1):
        for ell in range(m, lmax + 1):
            idx.append(alm_index(ell, m, lmax))
    assert idx == list(range(alm_size(lmax)))


def test_clear_spectral_caches_resets_lookup_state() -> None:
    spectral_mod.clear_spectral_caches()
    spectral_mod._alm_ell_array(4, 4)

    assert spectral_mod._ALM_ELL_CACHE

    spectral_mod.clear_spectral_caches()

    assert not spectral_mod._ALM_ELL_CACHE
    assert spectral_mod._SCALAR_MAP2ALM_MAT_CACHE_TOTAL_BYTES == 0
    assert spectral_mod._SPIN_YLM_CONCAT_CACHE_TOTAL_BYTES == 0


def test_map2alm_alm2map_roundtrip_low_l() -> None:
    nside = 16
    lmax = 8
    alm_true = _random_alm(lmax, seed=12)

    m = alm2map(alm_true, nside=nside, lmax=lmax)
    alm_rec = map2alm(m, nside=nside, lmax=lmax)

    num = torch.linalg.norm(alm_rec - alm_true)
    den = torch.linalg.norm(alm_true).clamp_min(1e-15)
    rel = float((num / den).item())
    assert rel < 1.5e-1


def test_map2alm_refuses_unsupported_devices() -> None:
    nside = 4
    npix = hp.nside2npix(nside)
    m = torch.zeros(npix, dtype=torch.float64, device="meta")
    with pytest.raises(ValueError, match="supports cpu, cuda, and mps"):
        map2alm(m, nside=nside, lmax=4)
    alm = torch.zeros(alm_size(4), dtype=torch.complex128, device="meta")
    with pytest.raises(ValueError, match="supports cpu, cuda, and mps"):
        alm2map(alm, nside=nside, lmax=4)


def test_map2alm_stays_on_accelerator() -> None:
    devices: list[str] = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    if not devices:
        pytest.skip("no CUDA or MPS")
    nside = 4
    npix = hp.nside2npix(nside)
    for device in devices:
        dtype = torch.float32 if device == "mps" else torch.float64
        m = torch.ones(npix, dtype=dtype, device=device)
        alm = map2alm(m, nside=nside, lmax=4)
        assert alm.device.type == device
        rec = alm2map(alm, nside=nside, lmax=4)
        assert rec.device.type == device
        assert torch.isfinite(alm).all()
        assert torch.isfinite(rec).all()


def test_map2alm_pixel_weights_preserve_constant_monopole() -> None:
    nside = 8
    m = torch.ones(hp.nside2npix(nside), dtype=torch.float64)
    alm0 = map2alm(m, nside=nside, lmax=4)
    alm_w = map2alm(m, nside=nside, lmax=4, use_weights=True)
    torch.testing.assert_close(alm_w[0].real, alm0[0].real, atol=1e-10, rtol=0.0)
    assert torch.isfinite(alm_w).all()


def test_map2alm_gal_cut_zeros_equatorial_strip() -> None:
    nside = 8
    npix = hp.nside2npix(nside)
    m = torch.ones(npix, dtype=torch.float64)
    cut = 20.0
    alm = map2alm(m, nside=nside, lmax=4, gal_cut=cut)
    pix = torch.arange(npix, dtype=torch.int64)
    _lon, lat = hp.pix2ang(nside, pix, nest=False, lonlat=True)
    masked = m.clone()
    masked[torch.abs(lat) < cut] = 0.0
    alm_ref = map2alm(masked, nside=nside, lmax=4)
    torch.testing.assert_close(alm, alm_ref, atol=0.0, rtol=0.0)


def test_anafast_gal_cut_and_weights_run() -> None:
    nside = 8
    m = torch.randn(hp.nside2npix(nside), dtype=torch.float64)
    cl = anafast(m, nside=nside, lmax=6, gal_cut=10.0, use_weights=True, pol=False)
    assert cl.shape == (7,)
    assert torch.isfinite(cl).all()




def test_map2alm_ring_weights_reduce_p2_leakage() -> None:
    nside = 8
    lmax = 4
    pix = torch.arange(hp.nside2npix(nside), dtype=torch.int64)
    theta, _ = hp.pix2ang(nside, pix, nest=False)
    z = torch.cos(theta)
    p2 = 0.5 * (3.0 * z * z - 1.0)
    exact_a20 = math.sqrt(4.0 * math.pi / 5.0)
    i20 = alm_index(2, 0, lmax)
    a0 = map2alm(p2, nside=nside, lmax=lmax, iter=0)
    aw = map2alm(p2, nside=nside, lmax=lmax, use_weights=True, iter=0)
    err0 = abs(complex(a0[i20].item()) - exact_a20)
    errw = abs(complex(aw[i20].item()) - exact_a20)
    assert errw < err0


def test_map2alm_rejects_simultaneous_weight_flags() -> None:
    m = torch.ones(hp.nside2npix(8), dtype=torch.float64)
    with pytest.raises(ValueError, match="mutually exclusive"):
        map2alm(m, nside=8, lmax=4, use_weights=True, use_pixel_weights=True)


def test_map2alm_pixel_weights_require_datapath() -> None:
    m = torch.ones(hp.nside2npix(8), dtype=torch.float64)
    with pytest.raises(FileNotFoundError, match="datapath"):
        map2alm(m, nside=8, lmax=4, use_pixel_weights=True)


def test_map2alm_pixel_weight_fits_is_one_plus_correction(tmp_path) -> None:
    nside = 8
    npix = hp.nside2npix(nside)
    corr = 0.1
    weights = torch.full((npix,), corr, dtype=torch.float64)
    dest = tmp_path / "full_weights"
    dest.mkdir()
    write_fits(
        str(dest / "healpix_full_weights_nside_0008.fits"),
        weights,
        overwrite=True,
    )
    m = torch.randn(npix, dtype=torch.float64)
    alm_w = map2alm(m, nside=nside, lmax=4, use_pixel_weights=True, datapath=tmp_path)
    alm_ref = map2alm(m * (1.0 + corr), nside=nside, lmax=4)
    torch.testing.assert_close(alm_w, alm_ref, atol=0.0, rtol=0.0)


def test_anafast_dipole_dominates_l1() -> None:
    nside = 32
    npix = hp.nside2npix(nside)
    pix = torch.arange(npix, dtype=torch.int64)
    _, _, z = hp.pix2vec(nside, pix, nest=False)

    cl = anafast(z, nside=nside, lmax=12)
    assert cl.shape == (13,)  # type: ignore
    assert float(cl[1]) > 0.0
    leakage = torch.sum(torch.abs(cl[2:]))  # type: ignore
    assert float(leakage / cl[1]) < 0.12


def test_compat_spectral_wrappers() -> None:
    nside = 8
    lmax = 6
    alm = _random_alm(lmax, seed=3)
    m = compat.alm2map(alm, nside=nside, lmax=lmax)
    rec = compat.map2alm(m, nside=nside, lmax=lmax)
    cl = compat.anafast(m, nside=nside, lmax=lmax)

    assert m.shape == (hp.nside2npix(nside),)
    assert rec.shape == alm.shape
    assert cl.shape == (lmax + 1,)  # type: ignore










def test_synalm_zero_cls_returns_zero() -> None:
    lmax = 8
    cl = torch.zeros((lmax + 1,), dtype=torch.float64)
    alm = synalm(cl, lmax=lmax, mmax=lmax)
    assert getattr(alm, "shape", None) == (alm_size(lmax),)
    torch.testing.assert_close(alm, torch.zeros_like(alm), atol=0.0, rtol=0.0)


def test_synfast_zero_cls_returns_zero_map() -> None:
    nside = 8
    lmax = 6
    cl = torch.zeros((lmax + 1,), dtype=torch.float64)
    m = synfast(cl, nside=nside, lmax=lmax, mmax=lmax, pol=True)
    assert m.shape == (hp.nside2npix(nside),)  # type: ignore
    torch.testing.assert_close(m, torch.zeros_like(m), atol=0.0, rtol=0.0)  # type: ignore


def test_synfast_pol_and_alm_output_shapes() -> None:
    nside = 8
    lmax = 6
    z = torch.zeros((lmax + 1,), dtype=torch.float64)
    cls4 = [torch.ones_like(z), z.clone(), torch.ones_like(z), z.clone()]
    maps, alms = synfast(
        cls4,
        nside=nside,
        lmax=lmax,
        mmax=lmax,
        alm=True,
        pol=True,
        new=False,  # type: ignore
    )
    assert getattr(maps, "shape", None) == (3, hp.nside2npix(nside))
    assert getattr(alms, "shape", None) == (3, alm_size(lmax))


def test_synfast_zero_with_pixwin_is_zero() -> None:
    nside = 8
    lmax = 6
    cl = torch.zeros((lmax + 1,), dtype=torch.float64)
    m = synfast(cl, nside=nside, lmax=lmax, mmax=lmax, pixwin=True, pol=False)
    torch.testing.assert_close(m, torch.zeros_like(m), atol=0.0, rtol=0.0)  # type: ignore


def test_map2alm_lsq_scalar_residual_improves() -> None:
    nside = 16
    lmax = 8
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(404)
    m = torch.from_numpy(rng.normal(size=npix)).to(torch.float64)

    alm0 = map2alm(m, nside=nside, lmax=lmax, mmax=lmax, pol=False)
    rec0 = alm2map(alm0, nside=nside, lmax=lmax, mmax=lmax, pol=False)
    rel0 = float(torch.linalg.norm(m - rec0) / torch.linalg.norm(m).clamp_min(1e-15))

    alm_lsq, rel_res, n_iter = map2alm_lsq(
        m,
        lmax=lmax,
        mmax=lmax,
        nside=nside,
        pol=False,
        tol=1e-14,
        maxiter=6,
    )
    assert getattr(alm_lsq, "shape", None) == alm0.shape
    assert rel_res <= rel0 + 1e-12
    assert 0 <= n_iter <= 6


def test_map2alm_lsq_maxiter_zero_matches_single_pass() -> None:
    nside = 8
    lmax = 6
    npix = hp.nside2npix(nside)
    m = torch.full((npix,), 1.0, dtype=torch.float64)

    alm_ref = map2alm(m, nside=nside, lmax=lmax, mmax=lmax, pol=False)
    alm_lsq, rel_res, n_iter = map2alm_lsq(
        m,
        lmax=lmax,
        mmax=lmax,
        nside=nside,
        pol=False,
        tol=1e-30,
        maxiter=0,
    )
    rec = alm2map(alm_lsq, nside=nside, lmax=lmax, mmax=lmax, pol=False)
    rel_ref = float(torch.linalg.norm(m - rec) / torch.linalg.norm(m).clamp_min(1e-15))
    torch.testing.assert_close(alm_lsq, alm_ref, atol=1e-12, rtol=0.0)
    assert rel_res == pytest.approx(rel_ref, abs=1e-12)
    assert n_iter == 0


def test_map2alm_lsq_pol_shapes() -> None:
    nside = 8
    lmax = 6
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(99)
    maps = torch.from_numpy(rng.normal(size=(3, npix))).to(torch.float64)
    alm, rel_res, n_iter = map2alm_lsq(
        maps,
        lmax=lmax,
        mmax=lmax,
        nside=nside,
        pol=True,
        tol=1e-10,
        maxiter=4,
    )
    assert getattr(alm, "shape", None) == (3, alm_size(lmax))
    assert isinstance(rel_res, float)
    assert math.isfinite(rel_res)
    assert 0 <= n_iter <= 4






def test_alm2map_infer_lmax() -> None:
    lmax = 4
    nside = 8
    alm = _random_alm(lmax, seed=9)
    m = alm2map(alm, nside=nside)
    assert getattr(m, "shape", None) == (hp.nside2npix(nside),)


@pytest.mark.parametrize("nest", [False, True])
def test_alm2map_scalar_cpp_direct_parity(
    monkeypatch: pytest.MonkeyPatch, nest: bool
) -> None:
    if _native_cpp is None or not hasattr(
        _native_cpp, "_healpix_scalar_alm2map_direct_cpu"
    ):
        pytest.skip("cpp scalar alm2map direct kernel unavailable")

    nside = 16
    lmax = 8
    alm = _random_alm(lmax, seed=71).to(torch.complex128)

    monkeypatch.setenv("TORCHSKY_SCALAR_ALM2MAP_CPP", "0")
    ref = alm2map(alm, nside=nside, lmax=lmax, nest=nest)
    monkeypatch.setenv("TORCHSKY_SCALAR_ALM2MAP_CPP", "1")
    got = alm2map(alm, nside=nside, lmax=lmax, nest=nest)
    torch.testing.assert_close(got, ref, atol=1e-10, rtol=1e-10)


def test_alm2map_scalar_ring_torch_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    nside = 16
    lmax = 8
    alm = _random_alm(lmax, seed=73).to(torch.complex128)

    monkeypatch.setenv("TORCHSKY_SCALAR_ALM2MAP_RING_TORCH", "0")
    ref = alm2map(alm, nside=nside, lmax=lmax, nest=False)

    monkeypatch.setenv("TORCHSKY_SCALAR_ALM2MAP_CPP", "0")
    monkeypatch.setenv("TORCHSKY_SCALAR_ALM2MAP_RING_TORCH", "force")
    got = alm2map(alm, nside=nside, lmax=lmax, nest=False)
    torch.testing.assert_close(got, ref, atol=1e-10, rtol=1e-10)


def test_alm2map_scalar_ring_torch_autograd(monkeypatch: pytest.MonkeyPatch) -> None:
    nside = 8
    lmax = 6
    nalm = alm_size(lmax)
    alm = torch.randn((nalm,), dtype=torch.complex128, requires_grad=True)
    monkeypatch.setenv("TORCHSKY_SCALAR_ALM2MAP_CPP", "0")
    monkeypatch.setenv("TORCHSKY_SCALAR_ALM2MAP_RING_TORCH", "force")
    m = alm2map(alm, nside=nside, lmax=lmax)
    loss = m.square().sum()
    loss.backward()
    assert alm.grad is not None
    assert torch.isfinite(alm.grad).all()
    assert float(torch.linalg.norm(alm.grad)) > 0.0


def test_alm2map_scalar_ring_torch_float32_finite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nside = 32
    lmax = 32
    nalm = alm_size(lmax)
    rng = np.random.default_rng(91)
    alm_np = (rng.normal(size=nalm) + 1j * rng.normal(size=nalm)).astype(np.complex64)
    alm = torch.from_numpy(alm_np)

    monkeypatch.setenv("TORCHSKY_SCALAR_ALM2MAP_CPP", "0")
    monkeypatch.setenv("TORCHSKY_SCALAR_ALM2MAP_RING_TORCH", "force")
    m = alm2map(alm, nside=nside, lmax=lmax, nest=False)
    assert torch.isfinite(m).all()


def test_map2alm_scalar_ring_torch_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    nside = 16
    lmax = 8
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(92)
    m = torch.from_numpy(rng.normal(size=npix)).to(torch.float64)

    monkeypatch.setenv("TORCHSKY_SCALAR_MAP2ALM_RING_TORCH", "0")
    ref = map2alm(m, nside=nside, lmax=lmax, mmax=lmax, nest=False)

    monkeypatch.setenv("TORCHSKY_SCALAR_MAP2ALM_RING_TORCH", "force")
    got = map2alm(m, nside=nside, lmax=lmax, mmax=lmax, nest=False)
    torch.testing.assert_close(got, ref, atol=1e-9, rtol=1e-9)


def test_map2alm_scalar_ring_torch_autograd(monkeypatch: pytest.MonkeyPatch) -> None:
    nside = 8
    lmax = 6
    npix = hp.nside2npix(nside)
    m = torch.randn((npix,), dtype=torch.float64, requires_grad=True)

    monkeypatch.setenv("TORCHSKY_SCALAR_MAP2ALM_RING_TORCH", "force")
    alm = map2alm(m, nside=nside, lmax=lmax, mmax=lmax, nest=False)
    loss = (alm.real.square() + alm.imag.square()).sum()
    loss.backward()
    assert m.grad is not None
    assert torch.isfinite(m.grad).all()
    assert float(torch.linalg.norm(m.grad)) > 0.0


def test_map2alm_scalar_ring_torch_float32_finite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nside = 32
    lmax = 32
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(93)
    m = torch.from_numpy(rng.normal(size=npix).astype(np.float32))

    monkeypatch.setenv("TORCHSKY_SCALAR_MAP2ALM_RING_TORCH", "force")
    alm = map2alm(m, nside=nside, lmax=lmax, mmax=lmax, nest=False)
    assert torch.isfinite(alm.real).all()
    assert torch.isfinite(alm.imag).all()


def test_alm2map_scalar_cpp_flag_keeps_autograd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nside = 8
    lmax = 6
    nalm = alm_size(lmax)
    alm = torch.randn((nalm,), dtype=torch.complex128, requires_grad=True)
    monkeypatch.setenv("TORCHSKY_SCALAR_ALM2MAP_CPP", "1")
    m = alm2map(alm, nside=nside, lmax=lmax)
    loss = m.square().sum()
    loss.backward()
    assert alm.grad is not None
    assert torch.isfinite(alm.grad).all()


def test_anafast_cross_constant_map() -> None:
    nside = 16
    npix = hp.nside2npix(nside)
    m1 = torch.full((npix,), 2.0, dtype=torch.float64)
    m2 = torch.full((npix,), -3.0, dtype=torch.float64)
    cl = anafast(m1, map2=m2, nside=nside, lmax=6)

    assert cl.shape == (7,)  # type: ignore
    assert float(cl[0]) < 0.0
    tail = torch.max(torch.abs(cl[1:]))  # type: ignore
    assert float(tail / cl[0].abs().clamp_min(1e-15)) < 0.1


def test_gaussian_beam_monotonic() -> None:
    b = gaussian_beam(math.radians(30.0 / 60.0), lmax=32)
    assert getattr(b, "shape", None) == (33,)
    assert float(b[0]) == pytest.approx(1.0)
    assert torch.all(b[1:] <= b[:-1])




def test_smoothalm_reduces_high_l_power() -> None:
    lmax = 24
    alm = _random_alm(lmax, seed=21)
    b = gaussian_beam(math.radians(40.0 / 60.0), lmax=lmax)
    sm = smoothalm(alm, lmax=lmax, beam=b)
    assert torch.linalg.norm(sm).item() <= torch.linalg.norm(alm).item() + 1e-12
    idx_hi = alm_index(lmax, min(8, lmax), lmax)
    assert abs(sm[idx_hi]) <= abs(alm[idx_hi]) + 1e-12


def test_smoothalm_pol_component_beams() -> None:
    lmax = 10
    alms = torch.stack(
        [
            _random_alm(lmax, seed=11),
            _random_alm(lmax, seed=12),
            _random_alm(lmax, seed=13),
        ],
        dim=0,
    )
    beam = torch.stack(
        [
            torch.linspace(1.0, 0.8, lmax + 1),
            torch.linspace(1.0, 0.5, lmax + 1),
            torch.linspace(1.0, 0.2, lmax + 1),
        ],
        dim=1,
    )
    sm = smoothalm(alms, lmax=lmax, beam=beam, pol=True)
    assert getattr(sm, "shape", None) == alms.shape
    idx = alm_index(lmax, min(lmax, 3), lmax)
    ell = lmax
    torch.testing.assert_close(
        sm[0, idx],
        alms[0, idx] * beam[ell, 0].to(torch.complex128),
        atol=1e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        sm[1, idx],
        alms[1, idx] * beam[ell, 1].to(torch.complex128),
        atol=1e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        sm[2, idx],
        alms[2, idx] * beam[ell, 2].to(torch.complex128),
        atol=1e-12,
        rtol=0.0,
    )


def test_smoothmap_variance_decreases() -> None:
    nside = 16
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(11)
    m = torch.from_numpy(rng.normal(size=npix)).to(torch.float64)
    sm = smoothmap(m, nside=nside, lmax=24, fwhm_rad=math.radians(30.0 / 60.0))
    assert sm.shape == m.shape
    assert float(torch.var(sm)) < float(torch.var(m))


@pytest.mark.skipif(_hp is None, reason="healpy not available")
def test_pol_map2alm_alm2map_and_anafast_paths() -> None:
    nside = 16
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(123)
    maps = torch.from_numpy(rng.normal(size=(3, npix))).to(torch.float64)
    lmax = 10

    alm_tf = map2alm(maps, nside=nside, lmax=lmax, pol=True)
    alm_hp = torch.from_numpy(
        _hp.map2alm(maps.numpy(), lmax=lmax, mmax=lmax, iter=0, pol=True)  # type: ignore
    ).to(torch.complex128)
    assert (
        getattr(alm_tf, "shape", None)
        == getattr(alm_hp, "shape", None)
        == (3, alm_size(lmax))
    )
    rel_alm = torch.linalg.norm(
        torch.as_tensor(alm_tf) - torch.as_tensor(alm_hp)
    ) / torch.linalg.norm(alm_hp).clamp_min(1e-15)
    assert float(rel_alm) < 2.5e-1

    maps_tf = alm2map(alm_tf, nside=nside, lmax=lmax, pol=True)
    maps_hp = torch.from_numpy(
        _hp.alm2map(alm_tf.numpy(), nside=nside, lmax=lmax, mmax=lmax, pol=True)  # type: ignore
    ).to(torch.float64)
    rel_map = torch.linalg.norm(
        torch.as_tensor(maps_tf) - torch.as_tensor(maps_hp)
    ) / torch.linalg.norm(maps_hp).clamp_min(1e-15)
    assert float(rel_map) < 2.5e-1

    cl_tf = anafast(maps, nside=nside, lmax=lmax, pol=True)
    cl_hp = torch.from_numpy(
        _hp.anafast(maps.numpy(), lmax=lmax, mmax=lmax, pol=True)  # type: ignore
    ).to(torch.float64)
    assert cl_tf.shape == cl_hp.shape == (6, lmax + 1)  # type: ignore
    rel_cl = torch.linalg.norm(cl_tf - cl_hp) / torch.linalg.norm(cl_hp).clamp_min(  # type: ignore
        1e-15
    )
    assert float(rel_cl) < 3.0e-1


def test_smoothmap_pol_torch_shapes() -> None:
    nside = 8
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(777)
    maps = torch.from_numpy(rng.normal(size=(3, npix))).to(torch.float64)
    sm = smoothmap(
        maps,
        nside=nside,
        lmax=12,
        sigma=math.radians(20.0 / 60.0),
        pol=True,
    )
    assert getattr(sm, "shape", None) == getattr(maps, "shape", None)






@pytest.mark.parametrize("spin", [1, 2, 3])
def test_spin_torch_roundtrip_without_healpy_backend(spin: int) -> None:
    nside = 16
    lmax = 8
    a_e, a_b = _random_spin_alms(lmax, spin=spin, seed=17)
    qu = alm2map_spin((a_e, a_b), nside=nside, spin=spin, lmax=lmax)
    e_rec, b_rec = map2alm_spin(qu, spin=spin, nside=nside, lmax=lmax)

    rel_e = torch.linalg.norm(e_rec - a_e) / torch.linalg.norm(a_e).clamp_min(1e-15)
    rel_b = torch.linalg.norm(b_rec - a_b) / torch.linalg.norm(a_b).clamp_min(1e-15)
    assert float(rel_e) < 3e-3
    assert float(rel_b) < 3e-3


@pytest.mark.skipif(_hp is None, reason="healpy not available")
def test_compat_spin_and_smoothing() -> None:
    nside = 8
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(13)
    m = torch.from_numpy(rng.normal(size=npix)).to(torch.float64)
    sm = compat.smoothmap(m, nside=nside, lmax=12, fwhm_rad=math.radians(20.0 / 60.0))
    assert getattr(sm, "shape", None) == getattr(m, "shape", None)
    assert torch.isfinite(sm).all()

    qu = torch.from_numpy(rng.normal(size=(2, npix))).to(torch.float64)
    a1, a2 = compat.map2alm_spin(qu, spin=2, nside=nside, lmax=8)
    rec = compat.alm2map_spin((a1, a2), nside=nside, spin=2, lmax=8)
    assert getattr(rec, "shape", None) == qu.shape


@pytest.mark.parametrize("spin", [1, 2])
def test_spin_autograd_fallback_paths(spin: int) -> None:
    nside = 8
    lmax = 6
    npix = hp.nside2npix(nside)

    qu = torch.randn((2, npix), dtype=torch.float64, requires_grad=True)
    ae, ab = map2alm_spin(qu, spin=spin, nside=nside, lmax=lmax)
    loss_map2 = (
        ae.real.square() + ae.imag.square() + ab.real.square() + ab.imag.square()
    ).sum()
    loss_map2.backward()
    assert qu.grad is not None
    assert torch.isfinite(qu.grad).all()
    assert float(torch.linalg.norm(qu.grad)) > 0.0

    nalm = alm_size(lmax)
    ae_in = torch.randn((nalm,), dtype=torch.complex128, requires_grad=True)
    ab_in = torch.randn((nalm,), dtype=torch.complex128, requires_grad=True)
    qu_rec = alm2map_spin((ae_in, ab_in), nside=nside, spin=spin, lmax=lmax)
    loss_alm2 = qu_rec.square().sum()
    loss_alm2.backward()
    assert ae_in.grad is not None and ab_in.grad is not None
    assert torch.isfinite(ae_in.grad).all() and torch.isfinite(ab_in.grad).all()
    assert float(torch.linalg.norm(ae_in.grad)) > 0.0
    assert float(torch.linalg.norm(ab_in.grad)) > 0.0


def test_map2alm_spin_ring_torch_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    nside = 16
    lmax = 12
    spin = 2
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(91)
    qu = torch.from_numpy(rng.normal(size=(2, npix))).to(torch.float64)

    monkeypatch.setenv("TORCHSKY_SPIN_MAP2ALM_RING_TORCH", "0")
    a0, b0 = map2alm_spin(qu, spin=spin, nside=nside, lmax=lmax)
    monkeypatch.setenv("TORCHSKY_SPIN_MAP2ALM_RING_TORCH", "force")
    a1, b1 = map2alm_spin(qu, spin=spin, nside=nside, lmax=lmax)

    torch.testing.assert_close(a1, a0, atol=1e-10, rtol=0.0)
    torch.testing.assert_close(b1, b0, atol=1e-10, rtol=0.0)


def test_map2alm_spin_ring_torch_parity_nested(monkeypatch: pytest.MonkeyPatch) -> None:
    nside = 16
    lmax = 12
    spin = 2
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(1231)
    qu_ring = torch.from_numpy(rng.normal(size=(2, npix))).to(torch.float64)
    qu_nest = torch.stack(
        [hp.reorder(qu_ring[0], r2n=True), hp.reorder(qu_ring[1], r2n=True)], dim=0
    )

    monkeypatch.setenv("TORCHSKY_SPIN_MAP2ALM_RING_TORCH", "0")
    a0, b0 = map2alm_spin(qu_nest, spin=spin, nside=nside, lmax=lmax, nest=True)
    monkeypatch.setenv("TORCHSKY_SPIN_MAP2ALM_RING_TORCH", "force")
    a1, b1 = map2alm_spin(qu_nest, spin=spin, nside=nside, lmax=lmax, nest=True)

    torch.testing.assert_close(a1, a0, atol=1e-10, rtol=0.0)
    torch.testing.assert_close(b1, b0, atol=1e-10, rtol=0.0)


@pytest.mark.skipif(
    not _has_cpp_ring_fourier(), reason="C++ ring Fourier kernels not available"
)
def test_map2alm_spin_ring_fourier_cpp_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    nside = 16
    lmax = 12
    spin = 2
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(191)
    qu = torch.from_numpy(rng.normal(size=(2, npix))).to(torch.float64)

    monkeypatch.setenv("TORCHSKY_SPIN_MAP2ALM_RING_TORCH", "force")
    monkeypatch.setenv("TORCHSKY_RING_FOURIER_CPP", "0")
    a0, b0 = map2alm_spin(qu, spin=spin, nside=nside, lmax=lmax)
    monkeypatch.setenv("TORCHSKY_RING_FOURIER_CPP", "1")
    a1, b1 = map2alm_spin(qu, spin=spin, nside=nside, lmax=lmax)
    torch.testing.assert_close(a1, a0, atol=1e-10, rtol=0.0)
    torch.testing.assert_close(b1, b0, atol=1e-10, rtol=0.0)


@pytest.mark.skipif(
    not _has_cpp_spin_concat(), reason="C++ spin concat kernels not available"
)
def test_map2alm_spin_concat_cpp_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    nside = 16
    lmax = 12
    spin = 2
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(211)
    qu = torch.from_numpy(rng.normal(size=(2, npix))).to(torch.float64)

    monkeypatch.setenv("TORCHSKY_SPIN_MAP2ALM_RING_TORCH", "force")
    monkeypatch.setenv("TORCHSKY_SPIN_MAP2ALM_RECURRENCE_CPP", "0")
    monkeypatch.setenv("TORCHSKY_SPIN_MAP2ALM_RING_CONCAT_CPP", "0")
    monkeypatch.setenv("TORCHSKY_SPIN_RING_INTEGRATE_CONCAT_CPP", "0")
    a0, b0 = map2alm_spin(qu, spin=spin, nside=nside, lmax=lmax)
    monkeypatch.setenv("TORCHSKY_SPIN_RING_INTEGRATE_CONCAT_CPP", "1")
    a1, b1 = map2alm_spin(qu, spin=spin, nside=nside, lmax=lmax)
    torch.testing.assert_close(a1, a0, atol=1e-10, rtol=0.0)
    torch.testing.assert_close(b1, b0, atol=1e-10, rtol=0.0)


@pytest.mark.skipif(
    not _has_cpp_spin_ring_fused(), reason="C++ fused spin ring kernels not available"
)
def test_map2alm_spin_ring_fused_cpp_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    nside = 16
    lmax = 12
    spin = 2
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(313)
    qu = torch.from_numpy(rng.normal(size=(2, npix))).to(torch.float64)

    monkeypatch.setenv("TORCHSKY_SPIN_MAP2ALM_RING_TORCH", "force")
    monkeypatch.setenv("TORCHSKY_SPIN_MAP2ALM_RECURRENCE_CPP", "0")
    monkeypatch.setenv("TORCHSKY_RING_FOURIER_CPP", "1")
    monkeypatch.setenv("TORCHSKY_SPIN_MAP2ALM_RING_CONCAT_CPP", "0")
    a0, b0 = map2alm_spin(qu, spin=spin, nside=nside, lmax=lmax)
    monkeypatch.setenv("TORCHSKY_SPIN_MAP2ALM_RING_CONCAT_CPP", "1")
    a1, b1 = map2alm_spin(qu, spin=spin, nside=nside, lmax=lmax)
    torch.testing.assert_close(a1, a0, atol=1e-10, rtol=0.0)
    torch.testing.assert_close(b1, b0, atol=1e-10, rtol=0.0)


def test_map2alm_spin_ring_torch_autograd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TORCHSKY_SPIN_MAP2ALM_RING_TORCH", "force")
    nside = 8
    lmax = 6
    spin = 2
    npix = hp.nside2npix(nside)
    qu = torch.randn((2, npix), dtype=torch.float64, requires_grad=True)
    ae, ab = map2alm_spin(qu, spin=spin, nside=nside, lmax=lmax)
    loss = (
        ae.real.square() + ae.imag.square() + ab.real.square() + ab.imag.square()
    ).sum()
    loss.backward()
    assert qu.grad is not None
    assert torch.isfinite(qu.grad).all()
    assert float(torch.linalg.norm(qu.grad)) > 0.0


def test_alm2map_spin_ring_torch_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    nside = 16
    lmax = 12
    spin = 2
    a_e, a_b = _random_spin_alms(lmax, spin=spin, seed=123)

    monkeypatch.setenv("TORCHSKY_SPIN_ALM2MAP_RING_TORCH", "0")
    qu0 = alm2map_spin((a_e, a_b), nside=nside, spin=spin, lmax=lmax)
    monkeypatch.setenv("TORCHSKY_SPIN_ALM2MAP_RING_TORCH", "force")
    qu1 = alm2map_spin((a_e, a_b), nside=nside, spin=spin, lmax=lmax)
    torch.testing.assert_close(qu1, qu0, atol=1e-10, rtol=0.0)


def test_alm2map_spin_ring_torch_parity_nested(monkeypatch: pytest.MonkeyPatch) -> None:
    nside = 16
    lmax = 12
    spin = 2
    a_e, a_b = _random_spin_alms(lmax, spin=spin, seed=1321)

    monkeypatch.setenv("TORCHSKY_SPIN_ALM2MAP_RING_TORCH", "0")
    qu0 = alm2map_spin((a_e, a_b), nside=nside, spin=spin, lmax=lmax, nest=True)
    monkeypatch.setenv("TORCHSKY_SPIN_ALM2MAP_RING_TORCH", "force")
    qu1 = alm2map_spin((a_e, a_b), nside=nside, spin=spin, lmax=lmax, nest=True)
    torch.testing.assert_close(qu1, qu0, atol=1e-10, rtol=0.0)


@pytest.mark.skipif(
    not _has_cpp_ring_fourier(), reason="C++ ring Fourier kernels not available"
)
def test_alm2map_spin_ring_fourier_cpp_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    nside = 16
    lmax = 12
    spin = 2
    a_e, a_b = _random_spin_alms(lmax, spin=spin, seed=231)

    monkeypatch.setenv("TORCHSKY_SPIN_ALM2MAP_RING_TORCH", "force")
    monkeypatch.setenv("TORCHSKY_RING_FOURIER_CPP", "0")
    qu0 = alm2map_spin((a_e, a_b), nside=nside, spin=spin, lmax=lmax)
    monkeypatch.setenv("TORCHSKY_RING_FOURIER_CPP", "1")
    qu1 = alm2map_spin((a_e, a_b), nside=nside, spin=spin, lmax=lmax)
    torch.testing.assert_close(qu1, qu0, atol=1e-10, rtol=0.0)


@pytest.mark.skipif(
    not _has_cpp_spin_concat(), reason="C++ spin concat kernels not available"
)
def test_alm2map_spin_concat_cpp_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    nside = 16
    lmax = 12
    spin = 2
    a_e, a_b = _random_spin_alms(lmax, spin=spin, seed=307)

    monkeypatch.setenv("TORCHSKY_SPIN_ALM2MAP_RING_TORCH", "force")
    monkeypatch.setenv("TORCHSKY_SPIN_ALM2MAP_RECURRENCE_CPP", "0")
    monkeypatch.setenv("TORCHSKY_SPIN_RING_INTERP_CONCAT_CPP", "0")
    qu0 = alm2map_spin((a_e, a_b), nside=nside, spin=spin, lmax=lmax)
    monkeypatch.setenv("TORCHSKY_SPIN_RING_INTERP_CONCAT_CPP", "1")
    qu1 = alm2map_spin((a_e, a_b), nside=nside, spin=spin, lmax=lmax)
    torch.testing.assert_close(qu1, qu0, atol=1e-10, rtol=0.0)


@pytest.mark.skipif(
    not _has_cpp_spin_ring_fused(), reason="C++ fused spin ring kernels not available"
)
def test_alm2map_spin_ring_finalize_cpp_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    nside = 16
    lmax = 12
    spin = 2
    a_e, a_b = _random_spin_alms(lmax, spin=spin, seed=353)

    monkeypatch.setenv("TORCHSKY_SPIN_ALM2MAP_RING_TORCH", "force")
    monkeypatch.setenv("TORCHSKY_SPIN_ALM2MAP_RECURRENCE_CPP", "0")
    monkeypatch.setenv("TORCHSKY_RING_FOURIER_CPP", "1")
    monkeypatch.setenv("TORCHSKY_SPIN_RING_FINALIZE_CPP", "0")
    qu0 = alm2map_spin((a_e, a_b), nside=nside, spin=spin, lmax=lmax)
    monkeypatch.setenv("TORCHSKY_SPIN_RING_FINALIZE_CPP", "1")
    qu1 = alm2map_spin((a_e, a_b), nside=nside, spin=spin, lmax=lmax)
    torch.testing.assert_close(qu1, qu0, atol=1e-10, rtol=0.0)


def test_alm2map_spin_ring_torch_autograd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TORCHSKY_SPIN_ALM2MAP_RING_TORCH", "force")
    nside = 8
    lmax = 6
    spin = 2
    nalm = alm_size(lmax)
    ae_in = torch.randn((nalm,), dtype=torch.complex128, requires_grad=True)
    ab_in = torch.randn((nalm,), dtype=torch.complex128, requires_grad=True)
    qu_rec = alm2map_spin((ae_in, ab_in), nside=nside, spin=spin, lmax=lmax)
    loss = qu_rec.square().sum()
    loss.backward()
    assert ae_in.grad is not None and ab_in.grad is not None
    assert torch.isfinite(ae_in.grad).all() and torch.isfinite(ab_in.grad).all()
    assert float(torch.linalg.norm(ae_in.grad)) > 0.0
    assert float(torch.linalg.norm(ab_in.grad)) > 0.0


def test_map2alm_batched_scalar_matches_rowwise() -> None:
    nside = 4
    lmax = 8
    npix = hp.nside2npix(nside)
    bsz = 5
    maps = torch.randn(bsz, npix, dtype=torch.float64)
    out_b = map2alm_batched(maps, nside=nside, lmax=lmax)
    ref = torch.stack(
        [map2alm(maps[i], nside=nside, lmax=lmax) for i in range(bsz)], dim=0
    )
    torch.testing.assert_close(out_b, ref, atol=0.0, rtol=0.0)


def test_map2alm_batched_pol_matches_rowwise() -> None:
    nside = 2
    lmax = 3
    npix = hp.nside2npix(nside)
    bsz = 2
    maps = torch.randn(bsz, 3, npix, dtype=torch.float64)
    out_b = map2alm_batched(maps, nside=nside, lmax=lmax, pol=True)
    ref = torch.stack(
        [map2alm(maps[i], nside=nside, lmax=lmax, pol=True) for i in range(bsz)],
        dim=0,
    )
    torch.testing.assert_close(out_b, ref, atol=0.0, rtol=0.0)


def test_alm2map_batched_scalar_matches_rowwise() -> None:
    nside = 4
    lmax = 8
    nalm = alm_size(lmax)
    bsz = 4
    alms = torch.randn(bsz, nalm, dtype=torch.complex128)
    out_b = alm2map_batched(alms, nside, lmax=lmax)
    ref = torch.stack([alm2map(alms[i], nside, lmax=lmax) for i in range(bsz)], dim=0)
    torch.testing.assert_close(out_b, ref, atol=0.0, rtol=0.0)


def test_alm2map_ring_torch_gradcheck(monkeypatch: pytest.MonkeyPatch) -> None:
    """Analytic alm->map Jacobian matches finite differences (ring torch path)."""
    nside = 4
    lmax = 3
    nalm = alm_size(lmax)
    alm = torch.randn(nalm, dtype=torch.complex128, requires_grad=True)

    monkeypatch.setenv("TORCHSKY_SCALAR_ALM2MAP_RING_TORCH", "force")

    def f(a: torch.Tensor) -> torch.Tensor:
        return alm2map(a, nside=nside, lmax=lmax)

    assert torch.autograd.gradcheck(f, (alm,), atol=1e-4, rtol=1e-4)


def test_map2alm_ring_torch_gradcheck(monkeypatch: pytest.MonkeyPatch) -> None:
    """Analytic map->alm Jacobian matches finite differences (ring torch path)."""
    nside = 4
    lmax = 3
    npix = hp.nside2npix(nside)
    m = torch.randn(npix, dtype=torch.float64, requires_grad=True)

    monkeypatch.setenv("TORCHSKY_SCALAR_MAP2ALM_RING_TORCH", "force")

    def f(x: torch.Tensor) -> torch.Tensor:
        return map2alm(x, nside=nside, lmax=lmax, mmax=lmax)

    assert torch.autograd.gradcheck(f, (m,), atol=1e-4, rtol=1e-4)


def test_alm2map_map2alm_roundtrip_gradcheck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full map->alm->map roundtrip preserves gradients (both paths forced)."""
    nside = 4
    lmax = 3
    npix = hp.nside2npix(nside)
    m = torch.randn(npix, dtype=torch.float64, requires_grad=True)

    monkeypatch.setenv("TORCHSKY_SCALAR_MAP2ALM_RING_TORCH", "force")
    monkeypatch.setenv("TORCHSKY_SCALAR_ALM2MAP_RING_TORCH", "force")

    def f(x: torch.Tensor) -> torch.Tensor:
        a = map2alm(x, nside=nside, lmax=lmax, mmax=lmax)
        return alm2map(a, nside=nside, lmax=lmax)

    assert torch.autograd.gradcheck(f, (m,), atol=2e-4, rtol=2e-4)


def test_validated_spin_profile_defaults_match_docs() -> None:
    """Bench replay profile defaults (see docs/benchmarks.md): RING Fourier + spin concat on."""
    import torchsky.sphere.spectral as spectral_mod

    assert spectral_mod._RING_FOURIER_CPP_ENABLE is True
    assert spectral_mod._RING_FOURIER_MODES_CPP_ENABLE is True
    assert spectral_mod._RING_FOURIER_SYNTH_CPP_ENABLE is True
    assert spectral_mod._SPIN_MAP2ALM_RING_CONCAT_CPP_ENABLE is True
    assert spectral_mod._SPIN_RING_FINALIZE_CPP_ENABLE is True
    assert spectral_mod._SPIN_RING_INTERP_CONCAT_CPP_ENABLE is True
    assert spectral_mod._SPIN_RING_INTEGRATE_CONCAT_CPP_ENABLE is True
    assert spectral_mod._SCALAR_RECURRENCE_CPP_ENABLE is True
    assert spectral_mod._SCALAR_INTERP_CPP_ENABLE is True


def _force_spin_ring(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TORCHSKY_SPIN_MAP2ALM_RING_TORCH", "force")
    monkeypatch.setenv("TORCHSKY_SPIN_ALM2MAP_RING_TORCH", "force")
    monkeypatch.setenv("TORCHSKY_SPIN_MAP2ALM_RECURRENCE_CPP", "1")
    monkeypatch.setenv("TORCHSKY_SPIN_ALM2MAP_RECURRENCE_CPP", "1")


@pytest.mark.skipif(
    not _has_cpp_spin_recurrence(), reason="C++ spin recurrence kernels not available"
)
@pytest.mark.parametrize("spin", [1, 2])
def test_ishioka_matches_wigner_spin_sht(
    spin: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ishioka lowering must match the Wigner recurrence, not only healpy."""
    nside = 8
    lmax = 12
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(401)
    qu = torch.from_numpy(rng.normal(size=(2, npix))).to(torch.float64)
    a_e, a_b = _random_spin_alms(lmax, spin=spin, seed=402)
    _force_spin_ring(monkeypatch)

    monkeypatch.setenv("TORCHSKY_SPIN_ISHIOKA", "1")
    ae_i, ab_i = map2alm_spin(qu, spin=spin, nside=nside, lmax=lmax)
    qu_i = alm2map_spin((a_e, a_b), nside=nside, spin=spin, lmax=lmax)
    monkeypatch.setenv("TORCHSKY_SPIN_ISHIOKA", "0")
    ae_w, ab_w = map2alm_spin(qu, spin=spin, nside=nside, lmax=lmax)
    qu_w = alm2map_spin((a_e, a_b), nside=nside, spin=spin, lmax=lmax)

    torch.testing.assert_close(ae_i, ae_w, atol=1e-10, rtol=0.0)
    torch.testing.assert_close(ab_i, ab_w, atol=1e-10, rtol=0.0)
    torch.testing.assert_close(qu_i, qu_w, atol=1e-10, rtol=0.0)


@pytest.mark.skipif(
    not _has_cpp_spin_recurrence(), reason="C++ spin recurrence kernels not available"
)
@pytest.mark.parametrize("spin", [1, 2])
def test_ishioka_spin_interpolate_integrate_adjoint(
    spin: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    nside = 8
    lmax = 12
    nalm = alm_size(lmax)
    rng = np.random.default_rng(411)
    plus = torch.from_numpy(
        (rng.normal(size=nalm) + 1j * rng.normal(size=nalm)).astype(np.complex128)
    )
    minus = torch.from_numpy(
        (rng.normal(size=nalm) + 1j * rng.normal(size=nalm)).astype(np.complex128)
    )
    _, _, theta, _ = _ring_layout_for_nside(nside)
    nrings = int(theta.numel())
    s_plus = torch.from_numpy(
        (
            rng.normal(size=(lmax + 1, nrings))
            + 1j * rng.normal(size=(lmax + 1, nrings))
        ).astype(np.complex128)
    )
    s_minus = torch.from_numpy(
        (
            rng.normal(size=(lmax + 1, nrings))
            + 1j * rng.normal(size=(lmax + 1, nrings))
        ).astype(np.complex128)
    )
    pix_w = (4.0 * np.pi) / float(hp.nside2npix(nside))
    w = torch.tensor([pix_w], dtype=torch.float64)
    monkeypatch.setenv("TORCHSKY_SPIN_ISHIOKA", "1")
    assert _native_cpp is not None
    s_modes = _native_cpp._healpix_spin_interpolate_recurrence_cpu(
        torch.stack([plus, minus]), theta, lmax, lmax, spin
    )
    alms = _native_cpp._healpix_spin_integrate_recurrence_cpu(
        s_plus, s_minus, theta, w, lmax, lmax, spin
    )
    lhs = pix_w * (
        (s_modes[0].real * s_plus.real + s_modes[0].imag * s_plus.imag).sum()
        + (s_modes[1].real * s_minus.real + s_modes[1].imag * s_minus.imag).sum()
    )
    rhs = (plus.real * alms[0].real + plus.imag * alms[0].imag).sum() + (
        minus.real * alms[1].real + minus.imag * alms[1].imag
    ).sum()
    denom = torch.maximum(rhs.abs(), torch.tensor(1e-15, dtype=torch.float64))
    assert float((lhs - rhs).abs() / denom) < 1e-12


@pytest.mark.parametrize("spin", [1, 2])
def test_spin_sht_small_nside_is_finite(
    spin: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mmax+1 can be smaller than the C++ m-pool; must not abort."""
    nside = 2
    lmax = 3
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(421)
    qu = torch.from_numpy(rng.normal(size=(2, npix))).to(torch.float64)
    a_e, a_b = _random_spin_alms(lmax, spin=spin, seed=422)
    _force_spin_ring(monkeypatch)
    ae, ab = map2alm_spin(qu, spin=spin, nside=nside, lmax=lmax)
    rec = alm2map_spin((a_e, a_b), nside=nside, spin=spin, lmax=lmax)
    assert torch.isfinite(ae).all() and torch.isfinite(ab).all()
    assert torch.isfinite(rec).all()






def test_rotate_alm_identity_and_z_phase() -> None:
    lmax = 4
    nalm = alm_size(lmax)
    rng = torch.Generator().manual_seed(0)
    alm = torch.complex(
        torch.randn(nalm, generator=rng, dtype=torch.float64),
        torch.randn(nalm, generator=rng, dtype=torch.float64),
    )
    torch.testing.assert_close(
        rotate_alm(alm, 0.0, 0.0, 0.0, lmax=lmax), alm, atol=1e-12, rtol=0.0
    )
    rotated = rotate_alm(alm, 0.0, 0.0, math.pi, lmax=lmax)
    for m in range(lmax + 1):
        phase = -1.0 if (m & 1) else 1.0
        for ell in range(m, lmax + 1):
            idx = alm_index(ell, m, lmax)
            torch.testing.assert_close(
                rotated[idx], alm[idx] * phase, atol=1e-10, rtol=0.0
            )


def test_rotate_alm_and_alm2lonlat_stay_on_accelerator() -> None:
    devices: list[str] = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    if not devices:
        pytest.skip("no CUDA or MPS")
    lmax = 4
    nalm = alm_size(lmax)
    rng = torch.Generator().manual_seed(1)
    alm_cpu = torch.complex(
        torch.randn(nalm, generator=rng, dtype=torch.float64),
        torch.randn(nalm, generator=rng, dtype=torch.float64),
    )
    ident_cpu = rotate_alm(alm_cpu, 0.0, 0.0, 0.0, lmax=lmax)
    torch.testing.assert_close(ident_cpu, alm_cpu, atol=1e-12, rtol=0.0)
    rotated_cpu = rotate_alm(alm_cpu, 0.3, 0.5, -0.2, lmax=lmax)
    y00 = 1.0 / math.sqrt(4.0 * math.pi)
    for device in devices:
        cdtype = torch.complex64 if device == "mps" else torch.complex128
        rdtype = torch.float32 if device == "mps" else torch.float64
        alm = alm_cpu.to(device=device, dtype=cdtype)
        ident = rotate_alm(alm, 0.0, 0.0, 0.0, lmax=lmax)
        assert ident.device.type == device
        atol = 2.0e-5 if device == "mps" else 1.0e-12
        torch.testing.assert_close(
            ident.cpu().to(torch.complex128), ident_cpu, atol=atol, rtol=0.0
        )
        rotated = rotate_alm(alm, 0.3, 0.5, -0.2, lmax=lmax)
        assert rotated.device.type == device
        rot_atol = 2.0e-4 if device == "mps" else 1.0e-10
        torch.testing.assert_close(
            rotated.cpu().to(torch.complex128), rotated_cpu, atol=rot_atol, rtol=0.0
        )
        mono = torch.zeros(nalm, dtype=cdtype, device=device)
        mono[0] = 1.0
        val = alm2lonlat(mono, 40.0, -10.0, lmax=lmax)
        assert val.device.type == device
        torch.testing.assert_close(
            val.cpu().to(torch.float64),
            torch.tensor(y00, dtype=torch.float64),
            atol=2.0e-5 if device == "mps" else 1.0e-12,
            rtol=0.0,
        )
        zeros = torch.zeros(nalm, dtype=cdtype, device=device)
        spin = alm2lonlat_spin((zeros, zeros), 2, 10.0, 20.0, lmax=lmax)
        assert spin.device.type == device
        assert spin.dtype == rdtype
        torch.testing.assert_close(spin, torch.zeros_like(spin), atol=0.0, rtol=0.0)




def test_alm2map_der1_matches_spin1_sqrt_ell() -> None:
    nside = 8
    lmax = 6
    rng = torch.Generator().manual_seed(3)
    nalm = alm_size(lmax)
    alm = torch.complex(
        torch.randn(nalm, generator=rng, dtype=torch.float64),
        torch.randn(nalm, generator=rng, dtype=torch.float64),
    )
    # m=0 coeffs must be real for a real map.
    for ell in range(lmax + 1):
        idx = alm_index(ell, 0, lmax)
        alm[idx] = complex(float(alm[idx].real), 0.0)

    m, d_theta, d_phi = alm2map_der1(alm, nside, lmax=lmax)
    torch.testing.assert_close(m, alm2map(alm, nside, lmax=lmax), atol=0.0, rtol=0.0)

    ell = torch.empty(nalm, dtype=torch.float64)
    for mm in range(lmax + 1):
        for ell_i in range(mm, lmax + 1):
            ell[alm_index(ell_i, mm, lmax)] = float(ell_i)
    alm_e = alm * torch.sqrt(ell * (ell + 1.0))
    alm_b = torch.zeros_like(alm_e)
    qu = alm2map_spin((alm_e, alm_b), nside, 1, lmax=lmax)
    torch.testing.assert_close(d_theta, qu[0], atol=1e-12, rtol=0.0)
    torch.testing.assert_close(d_phi, qu[1], atol=1e-12, rtol=0.0)
