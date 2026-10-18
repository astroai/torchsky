"""
/api/validate/* — Model validation diagnostics.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from ..renderers.validation import (
    residual_panel_pngs,
    pull_histogram,
    chi2_map_png,
    chi2_stats,
)
from ._safe_path import resolve_under_root

router = APIRouter()


class ValidationPair(BaseModel):
    truth_path: str
    model_path: str
    sigma_path: Optional[str] = None
    truth_hdu: int = 0
    model_hdu: int = 0
    sigma_hdu: int = 0


@dataclass
class ValidationReg:
    truth: object  # Tensor
    model: object
    sigma: Optional[object]
    pair: ValidationPair


def _pairs(request: Request) -> dict[str, ValidationReg]:
    if not hasattr(request.app.state, "val_pairs"):
        request.app.state.val_pairs = {}
    return request.app.state.val_pairs


def _evict_validation_fifo(
    request: Request, pairs: dict[str, ValidationReg], pid: str
) -> None:
    cap = int(getattr(request.app.state, "max_validation_pairs", 32))
    if pid in pairs:
        return
    while len(pairs) >= cap and pairs:
        oldest = next(iter(pairs))
        del pairs[oldest]


def _tensor_pixel_count(t: object) -> int:
    import torch

    ten = torch.as_tensor(t)
    if ten.dim() < 2:
        return int(ten.numel())
    return int(ten.shape[-1] * ten.shape[-2])


def _register_pair_sync(request: Request, pair: ValidationPair) -> JSONResponse:
    """Register a truth/model/(optional sigma) FITS pair. Returns pair ID."""
    from torchfits import read_tensor as read_image

    root: Path = request.app.state.root_dir
    max_px = int(getattr(request.app.state, "max_image_pixels", 512 * 1024 * 1024))

    def load(rel: str, hdu: int) -> object:
        abs_p = resolve_under_root(root, rel)
        return read_image(str(abs_p), hdu=hdu)

    try:
        truth = load(pair.truth_path, pair.truth_hdu).float()
        model = load(pair.model_path, pair.model_hdu).float()
        sigma = (
            load(pair.sigma_path, pair.sigma_hdu).float() if pair.sigma_path else None
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    for label, ten in ("truth", truth), ("model", model), ("sigma", sigma):
        if ten is None:
            continue
        n = _tensor_pixel_count(ten)
        if n > max_px:
            raise HTTPException(
                status_code=413,
                detail=f"{label} image has {n} pixels; max allowed is {max_px}.",
            )

    pid = hashlib.sha256(
        f"{pair.truth_path}:{pair.truth_hdu}:{pair.model_path}:{pair.model_hdu}:{pair.sigma_path}:{pair.sigma_hdu}".encode()
    ).hexdigest()
    pairs = _pairs(request)
    _evict_validation_fifo(request, pairs, pid)
    pairs[pid] = ValidationReg(truth=truth, model=model, sigma=sigma, pair=pair)
    return JSONResponse({"id": pid})


@router.post("/register")
async def register_pair(pair: ValidationPair, request: Request) -> JSONResponse:
    from .._blocking import run_in_thread

    return await run_in_thread(_register_pair_sync, request, pair)


def _png_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode()
    return f"data:image/png;base64,{b64}"


def _panel_pngs_sync(
    pid: str,
    request: Request,
    stretch: str,
    pull_range: float,
) -> JSONResponse:
    pairs = _pairs(request)
    if pid not in pairs:
        raise HTTPException(status_code=404, detail="Pair not registered")
    r = pairs[pid]
    try:
        pngs = residual_panel_pngs(
            r.truth,
            r.model,
            r.sigma,
            stretch=stretch,
            pull_sigma_range=pull_range,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return JSONResponse({k: _png_data_url(v) for k, v in pngs.items()})


@router.get("/{pid}/panel_pngs")
async def panel_pngs(
    pid: str,
    request: Request,
    stretch: str = "zscale",
    pull_range: float = 5.0,
) -> JSONResponse:
    """Return truth/model/residual/pull as base64-encoded PNGs."""
    from .._blocking import run_in_thread

    return await run_in_thread(_panel_pngs_sync, pid, request, stretch, pull_range)


def _pull_hist_sync(pid: str, request: Request) -> JSONResponse:
    pairs = _pairs(request)
    if pid not in pairs:
        raise HTTPException(status_code=404, detail="Pair not registered")
    r = pairs[pid]
    try:
        result = pull_histogram(r.truth, r.model, r.sigma)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return JSONResponse(result)


@router.get("/{pid}/pull_histogram")
async def val_pull_histogram(pid: str, request: Request) -> JSONResponse:
    from .._blocking import run_in_thread

    return await run_in_thread(_pull_hist_sync, pid, request)


def _chi2_stats_sync(pid: str, request: Request) -> JSONResponse:
    pairs = _pairs(request)
    if pid not in pairs:
        raise HTTPException(status_code=404, detail="Pair not registered")
    r = pairs[pid]
    if r.sigma is None:
        raise HTTPException(status_code=422, detail="sigma required for chi2")
    try:
        stats = chi2_stats(r.truth, r.model, r.sigma)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return JSONResponse(stats)


@router.get("/{pid}/chi2_stats")
async def val_chi2_stats(pid: str, request: Request) -> JSONResponse:
    from .._blocking import run_in_thread

    return await run_in_thread(_chi2_stats_sync, pid, request)


def _chi2_map_sync(pid: str, request: Request) -> Response:
    pairs = _pairs(request)
    if pid not in pairs:
        raise HTTPException(status_code=404, detail="Pair not registered")
    r = pairs[pid]
    if r.sigma is None:
        raise HTTPException(status_code=422, detail="sigma required for chi2 map")
    try:
        png = chi2_map_png(r.truth, r.model, r.sigma)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return Response(content=png, media_type="image/png")


@router.get("/{pid}/chi2_map")
async def val_chi2_map(pid: str, request: Request) -> Response:
    from .._blocking import run_in_thread

    return await run_in_thread(_chi2_map_sync, pid, request)


def _dr_report_sync(pid: str, request: Request) -> JSONResponse:
    from ..renderers.image import detect_dynamic_range

    pairs = _pairs(request)
    if pid not in pairs:
        raise HTTPException(status_code=404, detail="Pair not registered")
    r = pairs[pid]
    residual = r.model - r.truth
    return JSONResponse(
        {
            "truth": detect_dynamic_range(r.truth),
            "model": detect_dynamic_range(r.model),
            "residual": detect_dynamic_range(residual),
        }
    )


@router.get("/{pid}/dr_report")
async def val_dr_report(pid: str, request: Request) -> JSONResponse:
    """Return dynamic-range report for truth, model, and residual."""
    from .._blocking import run_in_thread

    return await run_in_thread(_dr_report_sync, pid, request)
