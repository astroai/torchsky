"""
/api/hips/* — On-the-fly HiPS tile server.

Converts any torchsky-readable FITS image into HiPS tiles
consumable by Aladin Lite v3.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, PlainTextResponse

from ..renderers.hips_tile import (
    render_hips_tile,
    render_preview,
    compute_hips_order,
    TILE_SIZE,
)
from ..renderers.image import zscale_limits
from ._safe_path import resolve_under_root

router = APIRouter()


@dataclass
class SurveyInfo:
    path: str
    hdu: int
    stretch: str
    cmap: str
    vmin: Optional[float]
    vmax: Optional[float]
    image_shape: tuple[int, int]
    max_order: int
    wcs: object  # torchsky.WCS
    image: object  # cached Tensor (ref, not copied)


def _survey_id(path: str, hdu: int, stretch: str, cmap: str) -> str:
    key = f"{path}:{hdu}:{stretch}:{cmap}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _get_surveys(request: Request) -> dict[str, SurveyInfo]:
    if not hasattr(request.app.state, "surveys"):
        request.app.state.surveys = {}
    return request.app.state.surveys


def _evict_surveys_fifo(
    request: Request, surveys: dict[str, SurveyInfo], sid: str
) -> None:
    """When over cap, drop oldest registrations (FIFO by insertion order)."""
    cap = int(getattr(request.app.state, "max_registered_surveys", 64))
    if sid in surveys:
        return
    while len(surveys) >= cap and surveys:
        oldest = next(iter(surveys))
        del surveys[oldest]


def _register_survey_sync(
    request: Request,
    path: str,
    hdu: int,
    stretch: str,
    cmap: str,
    vmin: Optional[float],
    vmax: Optional[float],
) -> JSONResponse:
    """
    Register a FITS file as a HiPS survey. Returns the survey ID.

    The survey ID is used as the prefix for all subsequent tile requests.
    """
    import torchsky

    root: Path = request.app.state.root_dir
    abs_path = str(resolve_under_root(root, path))

    sid = _survey_id(abs_path, hdu, stretch, cmap)
    surveys = _get_surveys(request)

    if sid in surveys:
        s = surveys[sid]
        return JSONResponse(
            {
                "id": sid,
                "max_order": s.max_order,
                "shape": list(s.image_shape),
                "vmin": s.vmin,
                "vmax": s.vmax,
            }
        )

    try:
        tensor, header = torchsky.read_image(abs_path, hdu=hdu, config=torchsky.ReadImageConfig(return_header=True))
        h, w = int(tensor.shape[-2]), int(tensor.shape[-1])
        n_pix = h * w
        max_px = int(getattr(request.app.state, "max_image_pixels", 536_870_912))
        if n_pix > max_px:
            raise HTTPException(
                status_code=413,
                detail=f"Image has {n_pix} pixels; max allowed is {max_px} (--max-image-pixels).",
            )

        wcs = torchsky.WCS(dict(header))
        max_order = compute_hips_order(wcs, (h, w))

        if vmin is None or vmax is None:
            sample = tensor.flatten().float()
            sample = sample[sample.isfinite()]
            _v1, _v2 = zscale_limits(sample) if sample.numel() > 0 else (0.0, 1.0)
            if vmin is None:
                vmin = _v1
            if vmax is None:
                vmax = _v2

        _evict_surveys_fifo(request, surveys, sid)
        info = SurveyInfo(
            path=abs_path,
            hdu=hdu,
            stretch=stretch,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            image_shape=(h, w),
            max_order=max_order,
            wcs=wcs,
            image=tensor,
        )
        surveys[sid] = info

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return JSONResponse(
        {
            "id": sid,
            "max_order": max_order,
            "shape": [h, w],
            "vmin": vmin,
            "vmax": vmax,
        }
    )


@router.get("/register")
async def register_survey(
    request: Request,
    path: str,
    hdu: int = 0,
    stretch: str = "zscale",
    cmap: str = "gray",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> JSONResponse:
    from .._blocking import run_in_thread

    return await run_in_thread(
        _register_survey_sync,
        request,
        path,
        hdu,
        stretch,
        cmap,
        vmin,
        vmax,
    )


@router.get("/list")
async def list_surveys(request: Request) -> JSONResponse:
    surveys = _get_surveys(request)
    return JSONResponse(
        [
            {
                "id": sid,
                "path": s.path,
                "hdu": s.hdu,
                "stretch": s.stretch,
                "cmap": s.cmap,
                "shape": list(s.image_shape),
                "max_order": s.max_order,
            }
            for sid, s in surveys.items()
        ]
    )


@router.get("/{sid}/properties")
async def hips_properties(sid: str, request: Request) -> PlainTextResponse:
    """Return the HiPS properties file for Aladin Lite."""
    surveys = _get_surveys(request)
    if sid not in surveys:
        raise HTTPException(status_code=404, detail=f"Survey {sid} not registered")
    s = surveys[sid]
    name = Path(s.path).stem
    base_url = str(request.base_url).rstrip("/") + f"/api/hips/{sid}"

    props = f"""# HiPS properties — generated by torchsky.viz
creator_did          = ivo://torchsky/{sid}
obs_title            = {name} (HDU {s.hdu})
hips_service_url     = {base_url}
hips_status          = public master
hips_type            = image
hips_tile_format     = png
hips_tile_width      = {TILE_SIZE}
hips_frame           = equatorial
hips_order           = {s.max_order}
hips_order_min       = 0
hips_pixel_cut       = {s.vmin:.6g} {s.vmax:.6g}
dataproduct_type     = image
"""
    return PlainTextResponse(props, media_type="text/plain")


# Tile URL pattern: /api/hips/{sid}/Norder{n}/Dir{d}/Npix{p}.png
_TILE_RE = re.compile(r"^Norder(\d+)/Dir(\d+)/Npix(\d+)\.png$")


def _hips_preview_png_sync(s: SurveyInfo) -> bytes:
    return render_preview(
        s.image,
        max_pixels=1024,
        stretch=s.stretch,
        cmap=s.cmap,
        vmin=s.vmin,
        vmax=s.vmax,
    )


def _hips_tile_png_sync(s: SurveyInfo, order: int, pix: int) -> bytes | None:
    return render_hips_tile(
        s.image,
        s.wcs,
        order=order,
        pix=pix,
        stretch=s.stretch,
        cmap=s.cmap,
        vmin=s.vmin,
        vmax=s.vmax,
    )


@router.get("/{sid}/{tile_path:path}")
async def hips_tile(sid: str, tile_path: str, request: Request) -> Response:
    """
    Serve one HiPS tile as PNG.

    URL format: /api/hips/{sid}/Norder{n}/Dir{d}/Npix{p}.png
    HiPS ``Dir`` follows IVOA grouping: ``Dir = Npix // 10000`` (integer division).
    """
    from .._blocking import run_in_thread

    surveys = _get_surveys(request)
    if sid not in surveys:
        raise HTTPException(status_code=404, detail=f"Survey {sid} not registered")

    m = _TILE_RE.match(tile_path)
    if not m:
        if tile_path == "preview.png":
            s = surveys[sid]
            try:
                png = await run_in_thread(_hips_preview_png_sync, s)
                return Response(
                    content=png,
                    media_type="image/png",
                    headers={"Cache-Control": "public, max-age=300"},
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e)) from e
        raise HTTPException(
            status_code=400, detail=f"Unrecognized tile URL: {tile_path}"
        )

    order = int(m.group(1))
    dir_url = int(m.group(2))
    pix = int(m.group(3))
    expected_dir = pix // 10000
    if dir_url != expected_dir:
        raise HTTPException(
            status_code=400,
            detail=f"HiPS Dir mismatch: for Npix={pix} expected Dir{expected_dir}, got Dir{dir_url}",
        )

    s = surveys[sid]
    cache = request.app.state.tile_cache
    cache_key = (sid, order, pix)

    cached = cache.get(cache_key)
    if cached is not None:
        return Response(
            content=cached,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600", "X-Cache": "HIT"},
        )

    try:
        png = await run_in_thread(_hips_tile_png_sync, s, order, pix)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    if png is None:
        from ..renderers.image import to_png
        import torch

        blank = to_png(
            torch.zeros(1, 1),
            cmap="gray",
            alpha_mask=torch.zeros(1, 1, dtype=torch.bool),
        )
        return Response(
            content=blank,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    cache.put(cache_key, png)
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600", "X-Cache": "MISS"},
    )
