from __future__ import annotations

import argparse
import sys

import torch

from ..device import default_compute_device
from torchfits import read, write
from ..map.sky import CrossExposureSkyCorrector, MeshSkyEstimator, PolynomialSkyModel


def _resolve_estimator(args: argparse.Namespace):
    if args.method == "mesh":
        return MeshSkyEstimator(
            block_size=int(args.block_size),
            statistic=str(args.statistic),
            mesh_filter=args.mesh_filter,
            interpolate=str(args.interpolate),
        )
    return PolynomialSkyModel(
        order=int(args.order),
        statistic=str(args.statistic),
    )


def _run_estimate(args: argparse.Namespace) -> int:
    read_dev = str(args.device or default_compute_device())
    img, hdr = read(str(args.image), hdu=int(args.hdu), return_header=True, device=read_dev)
    mask = None
    if args.mask is not None:
        mask = read(str(args.mask), hdu=int(args.mask_hdu), device=read_dev)
        mask = mask.to(dtype=torch.bool)

    est = _resolve_estimator(args)
    sky = est(img, mask) if mask is not None else est(img)

    write(str(args.output), sky, header=dict(hdr), overwrite=True)
    print(
        f"wrote {args.output} shape={tuple(sky.shape)} "
        f"method={args.method} statistic={args.statistic}",
        file=sys.stderr,
    )
    return 0


def _run_match(args: argparse.Namespace) -> int:
    read_dev = str(args.device or default_compute_device())
    images = []
    hdrs = []
    for p in args.images:
        img, h = read(str(p), hdu=int(args.hdu), return_header=True, device=read_dev)
        images.append(img)
        hdrs.append(dict(h))

    masks = None
    if args.masks:
        if len(args.masks) != len(images):
            raise SystemExit("--masks count must match the number of input images")
        masks = []
        for p in args.masks:
            m = read(str(p), hdu=int(args.mask_hdu), device=read_dev)
            masks.append(m.to(dtype=torch.bool))

    est = _resolve_estimator(args)
    corrector = CrossExposureSkyCorrector(estimator=est)

    if args.output:
        corrected = corrector.correct(images, masks)
        for i, (corr, hdr) in enumerate(zip(corrected, hdrs)):
            write(str(args.output).replace("{i}", str(i)), corr, header=hdr, overwrite=True)
    else:
        offsets = corrector.fit_offsets(images, masks)
        for i, off in enumerate(offsets):
            print(f"  offset[{i}] = {float(off):.6f}")
    return 0


def register_sky_parsers(subparsers: argparse.Action) -> None:
    p = subparsers.add_parser(
        "sky",
        help="Sky background estimation and multi-exposure matching.",
    )
    sub = p.add_subparsers(dest="sky_command", required=True)

    est = sub.add_parser("estimate", help="Estimate 2D sky background for an image.")
    est.add_argument("image", help="Input FITS image path.")
    est.add_argument("output", help="Output FITS path for sky model.")
    est.add_argument("--hdu", default=0, type=int, help="Image HDU index.")
    est.add_argument("--mask", default=None, help="Optional mask FITS (True = exclude).")
    est.add_argument("--mask-hdu", default=0, type=int, help="Mask HDU index.")
    est.add_argument(
        "--method",
        default="mesh",
        choices=("mesh", "polynomial"),
        help="Sky model type (default: mesh).",
    )
    est.add_argument(
        "--statistic",
        default="huber",
        choices=("median", "huber", "tukey"),
        help="Per-cell/per-pixel robust statistic (default: huber).",
    )
    est.add_argument(
        "--block-size",
        default=128,
        type=int,
        help="Mesh block size in pixels (mesh method, default: 128).",
    )
    est.add_argument(
        "--mesh-filter",
        default="median_3",
        choices=("median_3", "gaussian_3", "none"),
        help="Mesh post-filter (mesh method, default: median_3).",
    )
    est.add_argument(
        "--interpolate",
        default="bicubic",
        choices=("bilinear", "bicubic"),
        help="Mesh-to-image interpolation (mesh method, default: bicubic).",
    )
    est.add_argument(
        "--order",
        default=3,
        type=int,
        help="Polynomial degree (polynomial method, default: 3).",
    )
    est.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help="Compute device (e.g. cuda, cpu).",
    )
    est.set_defaults(handler=_run_estimate)

    match = sub.add_parser(
        "match",
        help="Compute additive sky offsets across overlapping exposures.",
    )
    match.add_argument("images", nargs="+", help="Input FITS images (same pixel grid).")
    match.add_argument("--hdu", default=0, type=int, help="Image HDU index.")
    match.add_argument("--masks", default=None, nargs="*", help="Optional per-image masks.")
    match.add_argument("--mask-hdu", default=0, type=int, help="Mask HDU index.")
    match.add_argument("--output", default=None, help="Output pattern with {i} (e.g. sky_{i}.fits).")
    match.add_argument(
        "--method",
        default="mesh",
        choices=("mesh", "polynomial"),
        help="Sky model type for per-image estimation.",
    )
    match.add_argument(
        "--statistic",
        default="median",
        choices=("median", "huber", "tukey"),
        help="Per-mesh statistic (default: median).",
    )
    match.add_argument(
        "--block-size",
        default=128,
        type=int,
        help="Mesh block size for per-image sky (default: 128).",
    )
    match.add_argument(
        "--order",
        default=3,
        type=int,
        help="Polynomial degree (polynomial method, default: 3).",
    )
    match.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help="Compute device (e.g. cuda, cpu).",
    )
    match.add_argument(
        "--mesh-filter",
        default="median_3",
        choices=("median_3", "gaussian_3", "none"),
        help="Mesh post-filter (mesh method, default: median_3).",
    )
    match.add_argument(
        "--interpolate",
        default="bicubic",
        choices=("bilinear", "bicubic"),
        help="Mesh-to-image interpolation (mesh method, default: bicubic).",
    )
    match.set_defaults(handler=_run_match)
