"""
/api/catalog/* — Catalog table overlay endpoints.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ._safe_path import resolve_under_root

router = APIRouter()


def _resolve(root: Path, path: str) -> Path:
    return resolve_under_root(root, path)


def _resolve_ra_dec_keys(data: object, ra_col: str, dec_col: str) -> tuple[str, str]:
    """Pick RA/Dec column names using the same candidate lists as the viz UI."""
    ra_candidates = [ra_col, "RA", "ra", "Alpha", "ALPHA", "alpha", "RAJ2000"]
    dec_candidates = [dec_col, "DEC", "dec", "Delta", "DELTA", "delta", "DEJ2000"]
    ra_key = next((k for k in ra_candidates if k in data), None)
    dec_key = next((k for k in dec_candidates if k in data), None)
    if ra_key is None or dec_key is None:
        available = list(data.keys()) if hasattr(data, "keys") else []
        raise HTTPException(
            status_code=422,
            detail=f"Could not find RA/Dec columns. Available: {available[:20]}",
        )
    return ra_key, dec_key


def _catalog_load_sync(
    request: Request,
    path: str,
    ra_col: str,
    dec_col: str,
    hdu: int,
    max_rows: int,
) -> JSONResponse:
    from torchfits.table import read_torch as read_table

    root: Path = request.app.state.root_dir
    abs_path = str(_resolve(root, path))

    try:
        data = read_table(abs_path, hdu=hdu)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    ra_key, dec_key = _resolve_ra_dec_keys(data, ra_col, dec_col)

    import torch

    ra_t = torch.as_tensor(data[ra_key])
    dec_t = torch.as_tensor(data[dec_key])

    n = ra_t.shape[0]
    if n > max_rows:
        step = max(1, n // max_rows)
        ra_t = ra_t[::step]
        dec_t = dec_t[::step]

    valid = torch.isfinite(ra_t) & torch.isfinite(dec_t)
    ra_t = ra_t[valid]
    dec_t = dec_t[valid]

    return JSONResponse(
        {
            "ra": ra_t.tolist(),
            "dec": dec_t.tolist(),
            "n_total": n,
            "n_returned": int(ra_t.numel()),
            "ra_col": ra_key,
            "dec_col": dec_key,
        }
    )


@router.get("/load")
async def catalog_load(
    request: Request,
    path: str,
    ra_col: str = "RA",
    dec_col: str = "DEC",
    hdu: int = 1,
    max_rows: int = 50_000,
) -> JSONResponse:
    """
    Load a FITS catalog table and return (ra, dec) arrays.

    Automatically tries common RA/Dec column name variants.
    """
    from .._blocking import run_in_thread

    return await run_in_thread(
        _catalog_load_sync,
        request,
        path,
        ra_col,
        dec_col,
        hdu,
        max_rows,
    )


def _catalog_filter_moc_sync(
    request: Request,
    catalog_path: str,
    moc_path: str,
    ra_col: str,
    dec_col: str,
    hdu: int,
) -> JSONResponse:
    from torchfits.table import read_torch as read_table
    from ...sphere.moc import MOC

    root: Path = request.app.state.root_dir
    cat_abs = str(_resolve(root, catalog_path))
    moc_abs = str(_resolve(root, moc_path))

    try:
        data = read_table(cat_abs, hdu=hdu)
        moc = MOC.read_fits(moc_abs)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    ra_key, dec_key = _resolve_ra_dec_keys(data, ra_col, dec_col)

    import torch

    ra_t = torch.as_tensor(data[ra_key], dtype=torch.float64)
    dec_t = torch.as_tensor(data[dec_key], dtype=torch.float64)
    ra_filt, dec_filt = moc.filter_catalog(ra_t, dec_t)

    return JSONResponse(
        {
            "ra": ra_filt.tolist(),
            "dec": dec_filt.tolist(),
            "n_total": int(ra_t.numel()),
            "n_in_moc": int(ra_filt.numel()),
        }
    )


@router.get("/filter_moc")
async def catalog_filter_moc(
    request: Request,
    catalog_path: str,
    moc_path: str,
    ra_col: str = "RA",
    dec_col: str = "DEC",
    hdu: int = 1,
) -> JSONResponse:
    """Filter a catalog to points inside a MOC footprint."""
    from .._blocking import run_in_thread

    return await run_in_thread(
        _catalog_filter_moc_sync,
        request,
        catalog_path,
        moc_path,
        ra_col,
        dec_col,
        hdu,
    )
