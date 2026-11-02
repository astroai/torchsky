"""Joint temperature and polarization (T/E/B) inverse from flat cutouts.

One latent set of T/E/B spherical-harmonic coefficients (lmax=3) explains two
scalar temperature cutouts and two spin-2 (Q/U) polarization cutouts. A
TT/EE/BB/TE cross-spectrum prior couples the components, and the normal
equations are solved with diagonally preconditioned CG. The spin-2 operators
map E/B alm to Q/U images and back with the exact adjoint. Reports temperature
and Q/U fit RMSE and the E/B power of the truth vs the fit (small cutouts fit the maps
without uniquely determining each mode). Runs on CPU (the
solver hooks take no device). Runtime about 3 s.

    pixi run -e test -- python examples/polarization_teb.py [--out DIR]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from torchsky.map import TargetRenderSpec
from torchsky.map.solvers import (
    build_spherical_harmonic_solver_hooks,
    build_spherical_spin_harmonic_solver_hooks,
    build_state_subspace_operator,
    build_teb_cls_prior_operator,
    compose_bound_operators,
    make_diagonal_preconditioner,
)

LMAX = 3
NALM = (LMAX + 1) * (LMAX + 2) // 2


def tan_header(crval1: float, crval2: float) -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": 72,
        "NAXIS2": 72,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 36.5,
        "CRPIX2": 36.5,
        "CRVAL1": crval1,
        "CRVAL2": crval2,
        "CDELT1": -0.12,
        "CDELT2": 0.12,
    }


def specs(centres, label):
    return tuple(
        TargetRenderSpec.from_header(tan_header(*c), band_label=label) for c in centres
    )


def rmse(pred, obs) -> list[float]:
    return [
        round(math.sqrt(float(((p - o) ** 2).mean())), 5) for p, o in zip(pred, obs)
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or Path(tempfile.mkdtemp(prefix="torchsky-teb-"))
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(47)

    truth = torch.zeros((3, NALM), dtype=torch.complex64)  # rows: T, E, B
    truth[0, 0] = complex(math.sqrt(4.0 * math.pi) * 0.35, 0.0)
    truth[0, 1] = complex(0.08, 0.0)
    truth[0, 2] = complex(-0.04, 0.02)
    truth[1, 2] = complex(0.06, 0.02)
    truth[1, 5] = complex(-0.03, 0.01)
    truth[2, 4] = complex(0.04, -0.01)
    truth[2, 7] = complex(-0.02, 0.01)

    temp_block = build_state_subspace_operator(
        truth,
        build_spherical_harmonic_solver_hooks(lmax=LMAX).bind_targets(
            specs([(210.0, 54.0), (214.0, 52.0)], "temp")
        ),
        selector=0,
    )
    pol_block = build_state_subspace_operator(
        truth,
        build_spherical_spin_harmonic_solver_hooks(spin=2, lmax=LMAX).bind_targets(
            specs([(210.0, 54.0), (208.5, 55.0)], "pol")
        ),
        selector=[1, 2],
    )
    prior = build_teb_cls_prior_operator(
        truth,
        lmax=LMAX,
        cls_tt=torch.tensor([10.0, 2.0, 0.8, 0.3], dtype=torch.float64),
        cls_ee=torch.tensor([5.0, 1.2, 0.5, 0.2], dtype=torch.float64),
        cls_bb=torch.tensor([5.0, 1.2, 0.5, 0.2], dtype=torch.float64),
        cls_te=torch.tensor([0.0, 0.4, 0.15, 0.05], dtype=torch.float64),
        apply_noise=lambda t: t * 0.08,
    )
    ops = compose_bound_operators(temp_block, pol_block, prior)

    obs_t = tuple(c + 0.01 * torch.randn_like(c) for c in temp_block.apply(truth))
    obs_p = tuple(c + 0.005 * torch.randn_like(c) for c in pol_block.apply(truth))
    data = (*obs_t, *obs_p, torch.zeros_like(truth))
    diag = ops.estimate_normal_operator_diagonal(
        data, initial_probe=torch.zeros_like(truth), damping=1e-4
    )
    fit = ops.solve_normal_equation_cg(
        data,
        initial_solution=torch.zeros_like(truth),
        iterations=10,
        damping=1e-4,
        preconditioner=make_diagonal_preconditioner(diag),
    )

    rec_t, rec_p = temp_block.apply(fit.solution), pol_block.apply(fit.solution)
    rt, rp = rmse(rec_t, obs_t), rmse(rec_p, obs_p)

    def power(a: torch.Tensor) -> float:
        return float((a.abs() ** 2).sum())

    print(
        f"lmax={LMAX} nalm={NALM}: temperature cutouts={len(obs_t)}, Q/U cutouts={len(obs_p)}"
    )
    print(f"temperature RMSE {rt} (noise 0.01); polarization RMSE {rp} (noise 0.005)")
    print(
        f"E power truth/fit = {power(truth[1]):.4f}/{power(fit.solution[1]):.4f}; "
        f"B power truth/fit = {power(truth[2]):.4f}/{power(fit.solution[2]):.4f}"
    )
    print(
        f"CG residual {fit.residual_norm_history[0]:.3g} -> {fit.residual_norm_history[-1]:.3g}"
    )
    assert max(rt) < 0.02 and max(rp) < 0.015

    fig, axes = plt.subplots(2, 3, figsize=(8, 5), constrained_layout=True)
    # polarization cutout 0 has components (Q, U) stacked on the leading axis.
    obs0, rec0 = obs_p[0], rec_p[0]
    panels = [
        ("T observed", obs_t[0]),
        ("T fit", rec_t[0]),
        ("T residual", obs_t[0] - rec_t[0]),
        ("Q observed", obs0[0]),
        ("Q fit", rec0[0]),
        ("Q residual", obs0[0] - rec0[0]),
    ]
    for ax, (title, im) in zip(axes.ravel(), panels):
        ax.imshow(im, origin="lower", cmap="RdBu" if "resid" in title else "viridis")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.savefig(out / "polarization_teb.png", dpi=100)
    print(f"figure: {out / 'polarization_teb.png'}")


if __name__ == "__main__":
    main()
