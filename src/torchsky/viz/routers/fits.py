"""
/api/fits/* — FITS file browser and metadata endpoints.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ._safe_path import resolve_under_root

router = APIRouter()


def _resolve_path(root: Path, rel_path: str) -> Path:
    """Resolve a relative path within root, refusing path traversal."""
    return resolve_under_root(root, rel_path)


def _is_fits(name: str) -> bool:
    n = name.lower()
    return any(
        n.endswith(ext) for ext in (".fits", ".fit", ".fz", ".fits.gz", ".fit.gz")
    )


def _file_tree_sync(request: Request, path: str) -> JSONResponse:
    """Return directory tree of FITS files under root (or a subdirectory)."""
    root: Path = request.app.state.root_dir
    base = _resolve_path(root, path) if path else root

    if not base.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    items = []
    if base.is_file():
        items = [
            {
                "name": base.name,
                "path": str(base.relative_to(root)),
                "type": "file",
                "size": base.stat().st_size,
            }
        ]
    else:
        for entry in sorted(base.iterdir()):
            if entry.is_dir() and not entry.name.startswith("."):
                items.append(
                    {
                        "name": entry.name,
                        "path": str(entry.relative_to(root)),
                        "type": "dir",
                    }
                )
            elif entry.is_file() and _is_fits(entry.name):
                items.append(
                    {
                        "name": entry.name,
                        "path": str(entry.relative_to(root)),
                        "type": "file",
                        "size": entry.stat().st_size,
                    }
                )

    return JSONResponse(
        {"items": items, "base": str(base.relative_to(root)) if base != root else ""}
    )


@router.get("/tree")
async def file_tree(request: Request, path: str = "") -> JSONResponse:
    from .._blocking import run_in_thread

    return await run_in_thread(_file_tree_sync, request, path)


def _fits_info_sync(request: Request, path: str, hdu: int) -> JSONResponse:
    """Return FITS image metadata: shape, BITPIX, WCS summary, HDU list."""
    import torchsky

    root: Path = request.app.state.root_dir
    abs_path = str(_resolve_path(root, path))

    try:
        hdul = torchsky.open(abs_path)
        hdu_list = [
            {
                "index": i,
                "name": getattr(h, "name", f"HDU{i}"),
                "type": type(h).__name__,
            }
            for i, h in enumerate(hdul)
        ]
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Could not read FITS HDU listing: {e}",
        ) from e

    try:
        tensor, header = torchsky.read_image(abs_path, hdu=hdu, return_header=True)
        shape = list(tensor.shape)
        bitpix = int(header.get("BITPIX", 0))
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    wcs_info = {}
    try:
        wcs = torchsky.WCS(dict(header))
        wcs_info = {
            "ctype1": wcs.wcs_params.get("CTYPE1", ""),
            "ctype2": wcs.wcs_params.get("CTYPE2", ""),
            "crval1": wcs.wcs_params.get("CRVAL1", 0.0),
            "crval2": wcs.wcs_params.get("CRVAL2", 0.0),
            "crpix1": wcs.wcs_params.get("CRPIX1", 0.0),
            "crpix2": wcs.wcs_params.get("CRPIX2", 0.0),
            "proj_code": wcs._proj_code,
        }
    except Exception:
        pass

    return JSONResponse(
        {
            "path": path,
            "hdu": hdu,
            "shape": shape,
            "bitpix": bitpix,
            "wcs": wcs_info,
            "hdu_list": hdu_list,
            "n_hdus": len(hdu_list),
        }
    )


@router.get("/info")
async def fits_info(request: Request, path: str, hdu: int = 0) -> JSONResponse:
    from .._blocking import run_in_thread

    return await run_in_thread(_fits_info_sync, request, path, hdu)


def _fits_histogram_sync(
    request: Request,
    path: str,
    hdu: int,
    nbins: int,
) -> JSONResponse:
    """Return image histogram, ZScale limits, and dynamic range report."""
    import torchsky
    from ..renderers.image import compute_histogram

    root: Path = request.app.state.root_dir
    abs_path = str(_resolve_path(root, path))

    try:
        tensor = torchsky.read_image(abs_path, hdu=hdu)
        flat = tensor.float().flatten()
        if flat.numel() > 1_000_000:
            flat = flat[:: flat.numel() // 1_000_000]
        result = compute_histogram(flat.reshape(1, -1), n_bins=nbins)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return JSONResponse(result)


@router.get("/histogram")
async def fits_histogram(
    request: Request,
    path: str,
    hdu: int = 0,
    nbins: int = 256,
) -> JSONResponse:
    from .._blocking import run_in_thread

    return await run_in_thread(_fits_histogram_sync, request, path, hdu, nbins)


def _fits_header_sync(request: Request, path: str, hdu: int) -> JSONResponse:
    """Return the full FITS header as a key-value dict."""
    import torchsky

    root: Path = request.app.state.root_dir
    abs_path = str(_resolve_path(root, path))

    try:
        header = torchsky.get_header(abs_path, hdu)
        hdr_dict = {}
        for k, v in header.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                hdr_dict[k] = v
            else:
                hdr_dict[k] = str(v)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return JSONResponse({"header": hdr_dict, "path": path, "hdu": hdu})


@router.get("/header")
async def fits_header(request: Request, path: str, hdu: int = 0) -> JSONResponse:
    from .._blocking import run_in_thread

    return await run_in_thread(_fits_header_sync, request, path, hdu)
