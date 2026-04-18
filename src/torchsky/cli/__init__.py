"""Command-line entry points for torchsky."""

from __future__ import annotations

import argparse

from .map_command import register_map_parser


def _register_viz_parser(subparsers: argparse.Action) -> None:
    p = subparsers.add_parser(
        "viz",
        help="Start the torchsky web visualization server.",
        description=(
            "Browser-based FITS / sky / validation viewer. Works on localhost and remote clusters "
            "(CANFAR, JupyterHub) via HTTPS.\n\n"
            "Security: there is no authentication. Only expose --host 0.0.0.0 on trusted networks "
            "or behind a reverse proxy. The server can read any file under --root and loads full "
            "images into memory for HiPS/validation (see --max-image-pixels and registration caps)."
        ),
    )
    p.add_argument(
        "--root", default=".", metavar="DIR",
        help="Directory to browse (default: current directory).",
    )
    p.add_argument(
        "--port", type=int, default=8822, metavar="PORT",
        help="HTTP port to listen on (default: 8822).",
    )
    p.add_argument(
        "--host", default="127.0.0.1", metavar="HOST",
        help="Bind address (default: 127.0.0.1). Use 0.0.0.0 for remote access.",
    )
    p.add_argument(
        "--no-browser", action="store_true",
        help="Do not open a browser window automatically.",
    )
    p.add_argument(
        "--cache-mb", type=int, default=512, metavar="MB",
        help="PNG tile LRU cache size in MiB (default: 512).",
    )
    p.add_argument(
        "--log-level", default="warning",
        choices=["critical", "error", "warning", "info", "debug"],
        help="uvicorn log level (default: warning).",
    )
    p.add_argument(
        "--max-registered-surveys", type=int, default=64, metavar="N",
        help="Max HiPS surveys kept in memory; oldest dropped when exceeded (default: 64).",
    )
    p.add_argument(
        "--max-validation-pairs", type=int, default=32, metavar="N",
        help="Max validation registrations; oldest dropped when exceeded (default: 32).",
    )
    p.add_argument(
        "--max-image-pixels", type=int, default=None, metavar="N",
        help="Reject HiPS/validation images with H*W above this (default: 512*1024*1024).",
    )
    p.add_argument(
        "--expose-health-root", action="store_true",
        help="Include absolute --root path in /api/health JSON (off by default).",
    )
    p.add_argument(
        "--cors-origin", action="append", default=None, dest="cors_origins",
        metavar="URL",
        help="Allowed CORS origin (repeatable). Default: all origins, no credentials.",
    )
    p.add_argument(
        "--cors-credentials", action="store_true",
        help="Allow credentialed CORS (only with explicit --cors-origin list).",
    )
    p.set_defaults(handler=_viz_handler)


def _viz_handler(args: argparse.Namespace) -> int:
    try:
        from torchsky.viz import serve
    except ImportError:
        import sys
        print(
            "torchsky[viz] extra not installed.\n"
            "Install with: pip install 'torchsky[viz]'",
            file=sys.stderr,
        )
        return 1
    cors = list(args.cors_origins) if args.cors_origins else None
    serve(
        root=args.root,
        port=args.port,
        host=args.host,
        open_browser=not args.no_browser,
        cache_mb=args.cache_mb,
        log_level=args.log_level,
        max_registered_surveys=args.max_registered_surveys,
        max_validation_pairs=args.max_validation_pairs,
        max_image_pixels=args.max_image_pixels,
        expose_root_in_health=args.expose_health_root,
        cors_allow_origins=cors,
        cors_allow_credentials=args.cors_credentials,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="torchsky")
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_map_parser(subparsers)
    _register_viz_parser(subparsers)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
