#!/usr/bin/env python3
"""Offline synthetic pretraining for Strainer-style sky-encoder fields (inr_next.md).

Writes a trunk checkpoint for ``load_strainer_trunk_checkpoint``. After loading into a
new model, call ``reset_decoder_head()`` then ``fit_field_model`` (optionally with
``freeze_field_encoder`` / ``freeze_field_body`` for head-only fine-tuning).

Use ``--noise-kinds`` for Phase-2-style round-robin targets (gaussian, uniform, spectrum).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from torchsky.map.field import (
        FieldDomain,
        make_field_model,
        pretrain_strainer_trunk_synthetic,
        save_strainer_trunk_checkpoint,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Output .pt path")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--noise-kind",
        choices=("uniform", "gaussian", "spectrum"),
        default="gaussian",
        help="Used when --noise-kinds is not set",
    )
    parser.add_argument(
        "--noise-kinds",
        type=str,
        default=None,
        metavar="LIST",
        help="Comma-separated round-robin kinds per step, e.g. gaussian,uniform,spectrum",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--center-ra", type=float, default=150.0)
    parser.add_argument("--center-dec", type=float, default=2.0)
    parser.add_argument("--spatial-scale-deg", type=float, default=0.05)
    parser.add_argument("--wavelength-center", type=float, default=5600.0)
    parser.add_argument("--wavelength-scale", type=float, default=900.0)
    parser.add_argument("--coord-mode", choices=("local_tangent", "sphere_xyz"), default="sphere_xyz")
    args = parser.parse_args()

    domain = FieldDomain(
        center_ra_deg=args.center_ra,
        center_dec_deg=args.center_dec,
        spatial_scale_deg=args.spatial_scale_deg,
        wavelength_center=args.wavelength_center,
        wavelength_scale=args.wavelength_scale,
        coord_mode=args.coord_mode,
    )
    torch.manual_seed(args.seed)
    model = make_field_model("strainer_sky_encoder_mlp", domain)
    if args.noise_kinds:
        kinds = tuple(k.strip() for k in args.noise_kinds.split(",") if k.strip())
        for k in kinds:
            if k not in ("uniform", "gaussian", "spectrum"):
                raise SystemExit(f"unknown noise kind {k!r} in --noise-kinds")
        pretrain_strainer_trunk_synthetic(
            model,
            steps=args.steps,
            lr=args.lr,
            noise_kinds=kinds,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    else:
        pretrain_strainer_trunk_synthetic(
            model,
            steps=args.steps,
            lr=args.lr,
            noise_kind=args.noise_kind,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_strainer_trunk_checkpoint(
        model,
        str(args.out),
        meta={
            "steps": args.steps,
            "noise_kind": args.noise_kind,
            "noise_kinds": args.noise_kinds,
            "coord_mode": args.coord_mode,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
