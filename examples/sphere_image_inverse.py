"""Fit an all-sky harmonic model to flat cutouts with priors and constraints.

A low-order spherical-harmonic state (alm, lmax=3) is observed through three
TAN cutouts. The same image-to-alm operator and its exact adjoint drive a
preconditioned conjugate-gradient solve of the normal equations in three
configurations: (1) data only, (2) data + C_ell prior with a Gaussian beam,
(3) mixed constraints (data + C_ell prior + coefficient anchor + degree
smoothness + dense harmonic covariance). Reports the fit RMSE and
the RMSE on a fourth, unfitted far-away cutout, which shows that three small
cutouts fix the rendered image but not every mode, so extrapolation depends on
the prior. Runs on CPU (the solver hooks take no device).
Runtime about 3 s.

    pixi run -e test -- python examples/sphere_image_inverse.py [--out DIR]
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
    build_harmonic_cl_prior_operator,
    build_harmonic_degree_prior_operator,
    build_harmonic_dense_prior_operator,
    build_spherical_harmonic_solver_hooks,
    build_state_identity_operator,
    compose_bound_operators,
    harmonic_dense_covariance,
    make_diagonal_preconditioner,
)
from torchsky.sphere import gaussian_beam

LMAX = 3
NALM = (LMAX + 1) * (LMAX + 2) // 2
NOISE = 0.01


def tan_header(crval1: float, crval2: float) -> dict:
    return {"NAXIS": 2, "NAXIS1": 72, "NAXIS2": 72, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
            "CRPIX1": 36.5, "CRPIX2": 36.5, "CRVAL1": crval1, "CRVAL2": crval2,
            "CDELT1": -0.15, "CDELT2": 0.15}


def solve(operator_set, data, truth, iterations: int):
    diag = operator_set.estimate_normal_operator_diagonal(
        data, initial_probe=torch.zeros_like(truth), damping=1e-4)
    return operator_set.solve_normal_equation_cg(
        data, initial_solution=torch.zeros_like(truth), iterations=iterations, damping=1e-4,
        preconditioner=make_diagonal_preconditioner(diag))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or Path(tempfile.mkdtemp(prefix="torchsky-inverse-"))
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(43)

    truth = torch.zeros(NALM, dtype=torch.complex64)
    truth[0] = complex(math.sqrt(4.0 * math.pi) * 0.30, 0.0)
    truth[1] = complex(0.10, 0.0)
    truth[2] = complex(-0.06, 0.02)
    truth[4] = complex(0.03, -0.01)
    truth[7] = complex(-0.02, 0.01)

    targets = tuple(TargetRenderSpec.from_header(tan_header(*c), band_label="bg")
                    for c in [(210.0, 54.0), (214.0, 52.5), (205.5, 55.5)])
    image_block = build_spherical_harmonic_solver_hooks(lmax=LMAX).bind_targets(targets)
    clean = image_block.apply(truth)
    observed = tuple(c + NOISE * torch.randn_like(c) for c in clean)
    zero = torch.zeros_like(truth)

    # A fourth, far-away cutout is never fitted; it measures extrapolation.
    held = TargetRenderSpec.from_header(tan_header(150.0, 20.0), band_label="bg")
    held_op = build_spherical_harmonic_solver_hooks(lmax=LMAX).bind_target(held)
    held_truth = held_op.apply(truth)

    def report(name, fit, op_set):
        rec = image_block.apply(fit.solution)
        rmse = [math.sqrt(float(((r - o) ** 2).mean())) for r, o in zip(rec, observed)]
        hold = math.sqrt(float(((held_op.apply(fit.solution) - held_truth) ** 2).mean()))
        print(f"{name:34s} fit RMSE {[round(v, 4) for v in rmse]}  held-out RMSE {hold:.4f}  "
              f"CG residual {fit.residual_norm_history[0]:.3g} -> {fit.residual_norm_history[-1]:.3g}")
        return rec, hold

    # 1. Data only.
    fit1 = solve(image_block, observed, truth, 12)
    rec1, d1 = report("data only", fit1, image_block)

    # 2. Data + C_ell prior (beam-transferred).
    cl_block = build_harmonic_cl_prior_operator(
        truth, lmax=LMAX, cls=torch.tensor([10.0, 1.5, 0.4, 0.15], dtype=torch.float64),
        beam_transfer=gaussian_beam(math.radians(1.0), LMAX), apply_noise=lambda t: t * 0.1)
    set2 = compose_bound_operators(image_block, cl_block)
    fit2 = solve(set2, (*observed, zero), truth, 10)
    _, d2 = report("data + C_ell prior", fit2, set2)

    # 3. Mixed constraints.
    anchor = truth.clone()
    anchor[2] += complex(0.02, -0.01)
    anchor_block = build_state_identity_operator(truth, apply_noise=lambda t: t * 0.2)
    smooth_block = build_harmonic_degree_prior_operator(
        truth, lmax=LMAX, power=2.0, include_monopole=False, apply_noise=lambda t: t * 0.05)
    dense_block = build_harmonic_dense_prior_operator(
        truth, harmonic_dense_covariance(lmax=LMAX, ell_correlation_length=1.2, diagonal_jitter=1e-3,
                                         amplitude=1.0, dtype=torch.complex128),
        apply_noise=lambda t: t * 0.05)
    set3 = compose_bound_operators(image_block, cl_block, anchor_block, smooth_block, dense_block)
    fit3 = solve(set3, (*observed, zero, anchor, zero, zero), truth, 12)
    rec3, d3 = report("mixed (C_ell+anchor+degree+dense)", fit3, set3)
    print(f"lmax={LMAX} nalm={NALM} cutouts={len(targets)}; the cutouts fix the rendered image, "
          f"not every alm, so extrapolation depends on the prior")
    assert max(d1, d2, d3) < 5.0

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.1), constrained_layout=True)
    for ax, (title, im) in zip(axes, [("observed", observed[0]), ("mixed-constraint fit", rec3[0]),
                                      ("residual", observed[0] - rec3[0])]):
        h = ax.imshow(im, origin="lower", cmap="viridis" if title != "residual" else "RdBu")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(h, ax=ax, shrink=0.8)
    fig.savefig(out / "sphere_image_inverse.png", dpi=100)
    print(f"figure: {out / 'sphere_image_inverse.png'}")


if __name__ == "__main__":
    main()
