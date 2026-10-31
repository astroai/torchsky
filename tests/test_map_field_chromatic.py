from __future__ import annotations

import pytest
import torch
from torch import nn

from torchsky.filter_profiles import photon_quadrature
from torchsky.map.field.chromatic import (
    ChromaticPSFField,
    LogLambdaRBFBasis,
    SpectralSkyField,
    spectral_coefficients_from_bands,
)


def test_chromatic_psf_basis_is_normalised_and_zero_sum() -> None:
    torch.manual_seed(0)
    psf = ChromaticPSFField(num_channels=2, num_detectors=3, pixel_scale_arcsec=0.04, size=21)
    lam = torch.tensor([0.9, 2.0, 4.4])
    b = psf.basis(lam, torch.tensor([0, 0, 1]))
    assert b.shape == (3, 4, 21, 21)
    assert torch.allclose(b[:, 0].sum((-2, -1)), torch.ones(3), atol=1e-5)
    assert bool((b[:, 0] >= 0).all())
    assert torch.allclose(b[:, 1:].sum((-2, -1)), torch.zeros(3, 3), atol=1e-6)
    w = psf.weights(torch.tensor(2.0), torch.tensor(0), torch.tensor(1), torch.rand(10) * 2 - 1, torch.rand(10) * 2 - 1)
    assert w.shape == (10, 4) and torch.allclose(w[:, 0], torch.ones(10))
    # zero-initialised weight net: no spatial variation before training
    assert torch.allclose(w[:, 1:], torch.zeros(10, 3))


def test_chromatic_psf_learns_diffraction_scaling() -> None:
    torch.manual_seed(0)
    psf = ChromaticPSFField(num_channels=1, num_detectors=1, pixel_scale_arcsec=0.04, size=31, num_kernels=1)
    lam = torch.tensor([1.0, 2.0, 4.0])
    r = torch.arange(31.0) - 15
    rr = r[None, :] ** 2 + r[:, None] ** 2
    sig = 0.42 * lam * 1e-6 / 6.5 * 206264.8 / 0.04  # Gaussian approx. of the Airy core, in pixels
    target = torch.exp(-0.5 * rr[None] / sig[:, None, None] ** 2)
    target = target / target.sum((-2, -1), keepdim=True)
    opt = torch.optim.Adam(psf.parameters(), lr=3e-3)
    for _ in range(400):
        opt.zero_grad()
        loss = (psf.basis(lam, torch.zeros(3))[:, 0] - target).pow(2).sum()
        loss.backward()
        opt.step()
    # an unseen wavelength in between should have an intermediate core: encircled energy
    # inside 1.5 px decreases monotonically with wavelength
    inner = rr <= 1.5**2
    ee = [float(psf.basis(torch.tensor([x]), torch.tensor([0]))[0, 0].detach()[inner].sum()) for x in (2.0, 3.0, 4.0)]
    assert ee[0] > ee[1] > ee[2]
    assert float(psf.centring_penalty(lam, torch.zeros(3))) < 0.05  # centroids within ~0.1 px


def test_spectral_sky_reproduces_band_images_and_is_differentiable() -> None:
    wl = torch.linspace(0.6, 5.5, 4000, dtype=torch.float64)
    centres, widths = [0.9, 1.5, 2.0, 2.8, 3.6, 4.4], [0.2, 0.3, 0.45, 0.6, 0.75, 1.0]
    nodes, weights = [], []
    for c, w in zip(centres, widths):
        t = ((wl > c - w / 2) & (wl < c + w / 2)).double()
        n, q = photon_quadrature(wl, t, 6)
        nodes.append(n)
        weights.append(q)
    nodes_um, node_w = torch.stack(nodes).float(), torch.stack(weights).float()
    basis = LogLambdaRBFBasis(10, 0.7, 5.3)
    bands = torch.rand(6, 16, 16) + torch.linspace(1, 3, 6)[:, None, None]

    class Grid(nn.Module):
        def __init__(self, a: torch.Tensor) -> None:
            super().__init__()
            self.a = nn.Parameter(a)

        def forward(self) -> torch.Tensor:
            return self.a

    a0 = spectral_coefficients_from_bands(bands, nodes_um, node_w, basis)
    sky = SpectralSkyField(Grid(a0), basis, nodes_um, node_w, spectral_smoothness=1e-3)
    sn, s = sky.at_nodes()
    assert sn.shape == (6, 6, 16, 16)
    assert float(((s - bands) / bands).abs().max()) < 5e-3
    (s.sum() + sky.penalty()).backward()
    assert sky.coefficients.a.grad is not None
    assert sky.spectral_cube(torch.tensor([1.0, 3.0])).shape == (2, 16, 16)


def test_validation_errors() -> None:
    with pytest.raises(ValueError):
        ChromaticPSFField(num_channels=1, num_detectors=1, pixel_scale_arcsec=0.04, size=30)
    with pytest.raises(ValueError):
        LogLambdaRBFBasis(1, 0.7, 5.0)
    with pytest.raises(ValueError):
        SpectralSkyField(nn.Identity(), LogLambdaRBFBasis(4, 0.7, 5.0), torch.ones(2, 3), torch.ones(3, 2))
