"""Fast-lane smoke: learned-PSF forward/adjoint/renderer on CPU (no heavy deps)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from torchsky.map.field.types import PSFSpec  # noqa: E402
from torchsky.map.operators import (  # noqa: E402
    LowRankChromaticPSF,
    apply_learned_psf_spec_adjoint,
    apply_psf_spec,
)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(5)
    model = LowRankChromaticPSF(
        grid_side=2,
        spectral_degree=1,
        base_sigma_pix=0.8,
        extent_pix=48.0,
        kernel_size=5,
    )
    h = w = 24
    grid = torch.linspace(2, 46, 2)
    u = torch.stack(torch.meshgrid(grid, grid, indexing="ij"), dim=-1).reshape(-1, 2)
    spec = PSFSpec.learned(model, base_sigma_pix=0.8, cell_u_pix=u, cell_lam_norm=0.5)
    x = torch.randn(h, w)
    y = apply_psf_spec(x, spec)
    xa = torch.randn(h, w)
    lhs = float((apply_psf_spec(xa, spec) * y).sum())
    rhs = float((xa * apply_learned_psf_spec_adjoint(y, spec)).sum())
    assert abs(lhs - rhs) / abs(lhs) < 1e-10
    sx = torch.arange(w, dtype=torch.float64).unsqueeze(0).expand(h, w) + 12.0
    sy = torch.arange(h, dtype=torch.float64).unsqueeze(1).expand(h, w) + 12.0
    out = model(
        torch.randn(h, w),
        spatial_x=sx,
        spatial_y=sy,
        wavelength=torch.tensor(15000.0),
    )
    assert out.shape == (h, w)
    print("fast lane: learned-PSF forward/adjoint/renderer OK")


if __name__ == "__main__":
    main()
