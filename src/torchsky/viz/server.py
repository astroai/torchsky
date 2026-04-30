"""
FastAPI application factory for torchsky.viz.

Operator notes: the app trusts ``--root`` as the only filesystem boundary (see path helpers in
routers). There is no built-in auth. Prefer binding to 127.0.0.1 unless a reverse proxy enforces
access control. HiPS and validation routes load whole images into RAM; use registration and
pixel limits to reduce accidental large-memory use on shared hosts.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .cache import LRUTileCache
from .routers import fits, hips, moc, validate, catalog

_STATIC_DIR = Path(__file__).parent / "static"

_DEFAULT_MAX_IMAGE_PIXELS = 512 * 1024 * 1024


def create_app(
    root_dir: str = ".",
    cache_mb: int = 512,
    *,
    max_registered_surveys: int = 64,
    max_validation_pairs: int = 32,
    max_image_pixels: int = _DEFAULT_MAX_IMAGE_PIXELS,
    expose_root_in_health: bool = False,
    cors_allow_origins: list[str] | None = None,
    cors_allow_credentials: bool = False,
) -> FastAPI:
    """
    Build and return the torchsky.viz FastAPI application.

    Args:
        root_dir:  Directory to browse. Must exist.
        cache_mb:  PNG tile LRU cache size in MiB.
        max_registered_surveys: Evict oldest HiPS registrations when exceeded.
        max_validation_pairs: Evict oldest validation pairs when exceeded.
        max_image_pixels: Reject HiPS/validation image HDUs larger than this (H*W).
        expose_root_in_health: If True, include absolute ``root`` in ``/api/health``.
        cors_allow_origins: Allowed CORS origins; ``None`` means ``[]``.
        cors_allow_credentials: Browser credentialed requests; forced off if origins is ``*``.
    """
    import torchsky

    root = Path(root_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"root_dir does not exist: {root}")

    app = FastAPI(
        title="torchsky viz",
        version=torchsky.__version__,
        description="Browser-based astronomy visualization powered by torchsky.",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    allow_origins = list(cors_allow_origins) if cors_allow_origins else []
    allow_cred = bool(cors_allow_credentials)
    if "*" in allow_origins:
        allow_cred = False

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_cred,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.root_dir = root
    app.state.tile_cache = LRUTileCache(max_mb=cache_mb)
    app.state.max_registered_surveys = max(1, int(max_registered_surveys))
    app.state.max_validation_pairs = max(1, int(max_validation_pairs))
    app.state.max_image_pixels = max(1, int(max_image_pixels))
    app.state.expose_root_in_health = bool(expose_root_in_health)

    app.include_router(fits.router, prefix="/api/fits", tags=["fits"])
    app.include_router(hips.router, prefix="/api/hips", tags=["hips"])
    app.include_router(moc.router, prefix="/api/moc", tags=["moc"])
    app.include_router(validate.router, prefix="/api/validate", tags=["validate"])
    app.include_router(catalog.router, prefix="/api/catalog", tags=["catalog"])

    @app.get("/api/health")
    async def health():
        out = {
            "status": "ok",
            "cache_mb": app.state.tile_cache.size_mb,
            "version": torchsky.__version__,
        }
        if app.state.expose_root_in_health:
            out["root"] = str(root)
        return out

    if _STATIC_DIR.exists():
        app.mount(
            "/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static"
        )

    return app
