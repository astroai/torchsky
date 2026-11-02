import numpy as np
import pytest
import torch

from torchsky.sphere.healpix import (
    ang2pix_nested,
    ang2pix_ring,
    pix2ang_ring,
    ud_grade,
)


def _random_lonlat(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0.0, 360.0, n)
    dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n)))
    return ra, dec


def test_ud_grade_custom_badval_is_respected() -> None:
    nside_in = 8
    nside_out = 4
    npix_in = 12 * nside_in * nside_in
    custom_bad = -12345.0

    m = np.ones(npix_in, dtype=np.float64)
    m[:4] = custom_bad  # one degraded output pixel gets only bad inputs

    out = (
        ud_grade(
            torch.from_numpy(m),
            nside_out=nside_out,
            order_in="NEST",
            order_out="NEST",
            pess=False,
            badval=custom_bad,
        )
        .cpu()
        .numpy()
    )
    assert out[0] == custom_bad
    np.testing.assert_allclose(out[1:], 1.0, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_cpu_small_sample() -> None:
    nside = 256
    n = 20000
    ra, dec = _random_lonlat(n, seed=2718)
    ring = np.random.default_rng(2719).integers(
        0, 12 * nside * nside, size=n, dtype=np.int64
    )

    ra_cpu = torch.from_numpy(ra).to(torch.float64)
    dec_cpu = torch.from_numpy(dec).to(torch.float64)
    ring_cpu = torch.from_numpy(ring)

    ra_gpu = ra_cpu.cuda()
    dec_gpu = dec_cpu.cuda()
    ring_gpu = ring_cpu.cuda()

    pix_r_cpu = ang2pix_ring(nside, ra_cpu, dec_cpu)
    pix_r_gpu = ang2pix_ring(nside, ra_gpu, dec_gpu).cpu()
    torch.testing.assert_close(pix_r_gpu, pix_r_cpu)

    pix_n_cpu = ang2pix_nested(nside, ra_cpu, dec_cpu)
    pix_n_gpu = ang2pix_nested(nside, ra_gpu, dec_gpu).cpu()
    torch.testing.assert_close(pix_n_gpu, pix_n_cpu)

    ra_r_cpu, dec_r_cpu = pix2ang_ring(nside, ring_cpu)
    ra_r_gpu, dec_r_gpu = pix2ang_ring(nside, ring_gpu)
    torch.testing.assert_close(ra_r_gpu.cpu(), ra_r_cpu, atol=1e-10, rtol=0.0)
    torch.testing.assert_close(dec_r_gpu.cpu(), dec_r_cpu, atol=1e-10, rtol=0.0)


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS not available",
)
def test_mps_matches_cpu_small_sample() -> None:
    nside = 256
    n = 20000
    ra, dec = _random_lonlat(n, seed=4242)
    ring = np.random.default_rng(4243).integers(
        0, 12 * nside * nside, size=n, dtype=np.int64
    )

    ra_cpu = torch.from_numpy(ra).to(torch.float64)
    dec_cpu = torch.from_numpy(dec).to(torch.float64)
    ring_cpu = torch.from_numpy(ring)

    ra_mps = ra_cpu.to(torch.float32).to("mps")
    dec_mps = dec_cpu.to(torch.float32).to("mps")
    ring_mps = ring_cpu.to("mps")

    pix_r_cpu = ang2pix_ring(nside, ra_cpu, dec_cpu)
    pix_r_mps = ang2pix_ring(nside, ra_mps, dec_mps).cpu()
    torch.testing.assert_close(pix_r_mps, pix_r_cpu)

    pix_n_cpu = ang2pix_nested(nside, ra_cpu, dec_cpu)
    pix_n_mps = ang2pix_nested(nside, ra_mps, dec_mps).cpu()
    torch.testing.assert_close(pix_n_mps, pix_n_cpu)

    ra_r_cpu, dec_r_cpu = pix2ang_ring(nside, ring_cpu)
    ra_r_mps, dec_r_mps = pix2ang_ring(nside, ring_mps)
    torch.testing.assert_close(
        ra_r_mps.cpu(), ra_r_cpu.to(torch.float32), atol=1e-4, rtol=0.0
    )
    torch.testing.assert_close(
        dec_r_mps.cpu(), dec_r_cpu.to(torch.float32), atol=1e-4, rtol=0.0
    )
