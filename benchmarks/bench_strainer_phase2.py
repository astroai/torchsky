#!/usr/bin/env python3
"""Phase 2 Strainer benchmark: multi-noise pretrain + early training loss + frozen trunk.

Compares baseline fit, round-robin synthetic pretrain with fresh decoder, and frozen-trunk fine-tuning
on a tiny synthetic ``strainer_sky_encoder_mlp`` observation. Emits JSON on stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from torchsky.map import (
    FieldDomain,
    FieldObservation,
    estimate_field_domain,
    fit_field_model,
    make_field_model,
    pretrain_strainer_trunk_synthetic,
    render_field_observation,
)


def _tan_header(
    size: int,
    *,
    crval1: float,
    crval2: float,
    scale_deg: float,
) -> dict[str, float | int | str]:
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": size / 2.0,
        "CRPIX2": size / 2.0,
        "CRVAL1": crval1,
        "CRVAL2": crval2,
        "CD1_1": -scale_deg,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": scale_deg,
    }


def _build_observation(
    *, truth_seed: int, size: int = 14
) -> tuple[FieldObservation, FieldDomain]:
    """Build a fixed synthetic observation. Use a dedicated ``truth_seed`` so fit models can use other seeds."""
    torch.manual_seed(truth_seed)
    truth_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=900.0,
        coord_mode="sphere_xyz",
    )
    truth = make_field_model("strainer_sky_encoder_mlp", truth_domain)
    header = _tan_header(size, crval1=150.0, crval2=2.0, scale_deg=1.0 / 3600.0)
    template = FieldObservation(
        image=torch.zeros((size, size), dtype=torch.float32),
        header=header,
        wavelength=5600.0,
    )
    image = render_field_observation(truth, template, num_spectral_samples=1).detach()
    obs = FieldObservation(
        image=image,
        header=header,
        weight=torch.ones_like(image),
        wavelength=5600.0,
    )
    fit_domain = estimate_field_domain([obs], coord_mode="sphere_xyz")
    return obs, fit_domain


def _summarize_fit(fit: Any, *, early_steps: tuple[int, ...]) -> dict[str, float]:
    out: dict[str, float] = {
        "initial_training_loss": float(fit.initial_loss),
        "final_training_loss": float(fit.final_loss),
    }
    for s in early_steps:
        key = f"training_loss_after_step_{s}"
        out[key] = float(fit.training_loss_after_step(s))
    return out


def _resolve_early_steps(
    requested: tuple[int, ...], *, fit_steps: int
) -> tuple[tuple[int, ...], tuple[int, ...], bool]:
    """Return (effective_steps, skipped_steps, fallback_used).

    If every requested step is out of range, effective steps default to
    ``(1, fit_steps)`` (or ``(1,)`` when ``fit_steps == 1``).
    """
    if fit_steps < 1:
        raise ValueError("fit_steps must be >= 1")
    skipped = tuple(sorted(set(s for s in requested if s < 1 or s > fit_steps)))
    in_range = tuple(sorted(set(s for s in requested if 1 <= s <= fit_steps)))
    if in_range:
        return in_range, skipped, False
    fallback = (1, fit_steps) if fit_steps > 1 else (1,)
    return fallback, skipped, True


def run_scenario(
    *,
    name: str,
    obs: FieldObservation,
    fit_domain: FieldDomain,
    seed: int,
    fit_steps: int,
    lr: float,
    synthetic_pretrain_steps: int,
    freeze_encoder: bool,
    freeze_body: bool,
    early_steps: tuple[int, ...],
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = make_field_model("strainer_sky_encoder_mlp", fit_domain)
    if synthetic_pretrain_steps > 0:
        pretrain_strainer_trunk_synthetic(
            model,
            steps=synthetic_pretrain_steps,
            lr=lr,
            noise_kinds=("gaussian", "uniform", "spectrum"),
            batch_size=2048,
            seed=seed + 11,
        )
        model.reset_decoder_head()
    fit = fit_field_model(
        [obs],
        model,
        steps=fit_steps,
        lr=lr,
        num_spectral_samples=1,
        freeze_field_encoder=freeze_encoder,
        freeze_field_body=freeze_body,
    )
    row: dict[str, Any] = {
        "name": name,
        "synthetic_pretrain_steps": synthetic_pretrain_steps,
        "freeze_field_encoder": freeze_encoder,
        "freeze_field_body": freeze_body,
        **_summarize_fit(fit, early_steps=early_steps),
    }
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fit-steps", type=int, default=28)
    parser.add_argument("--lr", type=float, default=2.5e-3)
    parser.add_argument("--synthetic-pretrain-steps", type=int, default=120)
    parser.add_argument(
        "--early-steps",
        type=str,
        default="1,5,10,20",
        help="Comma-separated 1-based step indices to report; values above --fit-steps are omitted (listed in early_steps_skipped)",
    )
    args = parser.parse_args()
    early_requested = tuple(
        int(x.strip()) for x in args.early_steps.split(",") if x.strip()
    )
    early, early_skipped, early_fallback = _resolve_early_steps(
        early_requested, fit_steps=args.fit_steps
    )

    obs, fit_domain = _build_observation(truth_seed=args.seed)
    scenarios = [
        ("baseline_no_pretrain", 0, False, False),
        ("pretrain_reset_unfrozen", args.synthetic_pretrain_steps, False, False),
        ("pretrain_reset_frozen_trunk", args.synthetic_pretrain_steps, True, True),
    ]
    rows = []
    for idx, (name, sp, fe, fb) in enumerate(scenarios):
        # Offset so fit-model init never matches the truth field (same seed would give ~0 loss).
        model_init_seed = args.seed + 10_000 + idx * 997
        rows.append(
            run_scenario(
                name=name,
                obs=obs,
                fit_domain=fit_domain,
                seed=model_init_seed,
                fit_steps=args.fit_steps,
                lr=args.lr,
                synthetic_pretrain_steps=sp,
                freeze_encoder=fe,
                freeze_body=fb,
                early_steps=early,
            )
        )

    payload: dict[str, Any] = {
        "suite": "strainer_phase2",
        "fit_steps": int(args.fit_steps),
        "early_steps": list(early),
        "rows": rows,
    }
    if early_skipped:
        payload["early_steps_skipped"] = list(early_skipped)
    if early_fallback or early_skipped:
        payload["early_steps_requested"] = list(early_requested)
    if early_fallback:
        payload["early_steps_fallback"] = True
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
