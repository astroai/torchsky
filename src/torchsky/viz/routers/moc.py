"""
/api/moc/* — MOC file endpoints.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from ..renderers.moc_geojson import moc_to_geojson
from ._safe_path import resolve_under_root

router = APIRouter()


def _resolve(root: Path, path: str) -> Path:
    return resolve_under_root(root, path)


def _moc_geojson_sync(request: Request, path: str) -> JSONResponse:
    from torchsky.sphere.moc import MOC

    root: Path = request.app.state.root_dir
    abs_path = str(_resolve(root, path))

    try:
        moc = MOC.read_fits(abs_path)
        geojson = moc_to_geojson(moc)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return JSONResponse(geojson)


@router.get("/geojson")
async def moc_geojson(request: Request, path: str) -> JSONResponse:
    """Return MOC as a GeoJSON FeatureCollection of polygon rings."""
    from .._blocking import run_in_thread

    return await run_in_thread(_moc_geojson_sync, request, path)


def _moc_stats_sync(request: Request, path: str) -> JSONResponse:
    from torchsky.sphere.moc import MOC

    root: Path = request.app.state.root_dir
    abs_path = str(_resolve(root, path))

    try:
        moc = MOC.read_fits(abs_path)
        stats = {
            "area_sq_deg": round(moc.area_sq_deg, 4),
            "area_sr": round(moc.area, 6),
            "sky_fraction": round(moc.area / (4 * 3.14159265358979), 6),
            "max_order": moc.max_order,
            "n_cells": int(moc.uniq.numel()),
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return JSONResponse(stats)


@router.get("/stats")
async def moc_stats(request: Request, path: str) -> JSONResponse:
    """Return MOC statistics: area, max_order, n_cells."""
    from .._blocking import run_in_thread

    return await run_in_thread(_moc_stats_sync, request, path)


@router.get("/fits")
async def moc_fits(request: Request, path: str) -> Response:
    """Serve raw MOC FITS bytes — for Aladin Lite's native MOC loading."""
    root: Path = request.app.state.root_dir
    abs_path = _resolve(root, path)

    if not abs_path.exists():
        raise HTTPException(status_code=404)

    content = abs_path.read_bytes()
    return Response(
        content=content,
        media_type="application/fits",
        headers={"Content-Disposition": f'attachment; filename="{abs_path.name}"'},
    )
