"""Command-line entry points for torchsky."""

from __future__ import annotations

import argparse

from .map import ImageObservation, ImagePlaneMapMaker


def _map_command(args: argparse.Namespace) -> int:
    observations = [
        ImageObservation(path=path, hdu=args.hdu, weight=weight)
        for path, weight in zip(args.inputs, args.weights)
    ]
    result = ImagePlaneMapMaker().run(observations, output_path=args.output)
    print(
        f"wrote {args.output} shape={tuple(result.data.shape)} "
        f"mean_weight={float(result.weight.mean().item()):.4f}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="torchsky")
    subparsers = parser.add_subparsers(dest="command", required=True)

    map_parser = subparsers.add_parser("map", help="Image-plane-first mapmaking")
    map_parser.add_argument("output", help="Output FITS path")
    map_parser.add_argument("inputs", nargs="+", help="Input FITS images")
    map_parser.add_argument("--hdu", default=0, type=int, help="Input HDU index")
    map_parser.add_argument(
        "--weights",
        default=None,
        nargs="*",
        type=float,
        help="Optional per-input scalar weights",
    )
    map_parser.set_defaults(handler=_map_command)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "map" and args.weights is None:
        args.weights = [1.0] * len(args.inputs)
    if len(args.weights) != len(args.inputs):
        raise SystemExit("--weights must match the number of inputs")
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
