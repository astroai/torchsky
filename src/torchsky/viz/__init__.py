"""
torchsky.viz — browser-based astronomy visualization server.

All computation runs server-side via the torchsky API.
The browser receives PNG tiles, JSON geometries, and chart data.

Usage::

    # CLI
    torchsky viz --root /path/to/data --port 8822

    # Python
    import torchsky.viz as viz
    viz.serve(root="/path/to/data", port=8822)
"""

from __future__ import annotations

import threading
import webbrowser
from pathlib import Path


def serve(
    root: str | Path = ".",
    port: int = 8822,
    host: str = "127.0.0.1",
    open_browser: bool = True,
    cache_mb: int = 512,
    log_level: str = "warning",
    *,
    max_registered_surveys: int = 64,
    max_validation_pairs: int = 32,
    max_image_pixels: int | None = None,
    expose_root_in_health: bool = False,
    cors_allow_origins: list[str] | None = None,
    cors_allow_credentials: bool = False,
) -> None:
    """
    Start the torchsky viz server (blocking).

    Args:
        root:         Directory to browse. Defaults to current directory.
        port:         HTTP port. Default 8822.
        host:         Bind address. Default 127.0.0.1 (localhost only).
        open_browser: Open the browser automatically.
        cache_mb:     Tile cache size in MiB.
        log_level:    uvicorn log level.
        max_registered_surveys: Cap HiPS registrations (FIFO eviction).
        max_validation_pairs: Cap validation pairs (FIFO eviction).
        max_image_pixels: Max H*W per image for HiPS/validation (default 512M).
        expose_root_in_health: Put absolute root path in ``/api/health`` JSON.
        cors_allow_origins: Explicit CORS origins; ``None`` defaults to none.
        cors_allow_credentials: Enable credentialed CORS (disallowed with ``*`` origins).
    """
    try:
        import uvicorn
    except ImportError as e:
        raise ImportError(
            "torchsky[viz] extra required: pip install torchsky[viz]"
        ) from e

    from .server import _DEFAULT_MAX_IMAGE_PIXELS, create_app

    app = create_app(
        root_dir=str(root),
        cache_mb=cache_mb,
        max_registered_surveys=max_registered_surveys,
        max_validation_pairs=max_validation_pairs,
        max_image_pixels=max_image_pixels
        if max_image_pixels is not None
        else _DEFAULT_MAX_IMAGE_PIXELS,
        expose_root_in_health=expose_root_in_health,
        cors_allow_origins=cors_allow_origins,
        cors_allow_credentials=cors_allow_credentials,
    )

    url = f"http://{host}:{port}"
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    print(f"\n  torchsky viz  →  {url}\n  root: {Path(root).resolve()}\n")
    uvicorn.run(app, host=host, port=port, log_level=log_level)


def serve_background(
    root: str | Path = ".",
    port: int = 8822,
    host: str = "127.0.0.1",
    cache_mb: int = 512,
    **kwargs: object,
) -> threading.Thread:
    """
    Start the server in a daemon thread (non-blocking, for notebooks).

    Returns the thread so callers can check liveness.
    """
    kw = dict(root=root, port=port, host=host, open_browser=False, cache_mb=cache_mb)
    kw.update(kwargs)
    t = threading.Thread(
        target=serve,
        kwargs=kw,
        daemon=True,
        name="torchsky-viz",
    )
    t.start()
    return t


def show_url(port: int = 8822, host: str = "127.0.0.1") -> str:
    """Return (and print) the correct URL, auto-detecting Jupyter environment."""
    base = f"http://{host}:{port}"
    try:
        # JupyterHub server-proxy detection
        import os

        hub_base = os.environ.get("JUPYTERHUB_SERVICE_PREFIX")
        if hub_base:
            base = f"{hub_base.rstrip('/')}/proxy/{port}"
    except Exception:
        pass
    print(f"  torchsky viz  →  {base}")
    return base


__all__ = ["serve", "serve_background", "show_url"]
