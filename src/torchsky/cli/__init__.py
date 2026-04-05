"""Command-line entry points for torchsky."""

from __future__ import annotations

import argparse

from .map_command import register_map_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="torchsky")
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_map_parser(subparsers)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
