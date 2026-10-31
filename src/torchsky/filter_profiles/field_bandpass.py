"""Resolve field :class:`~torchsky.map.field.FieldObservation` spectral kwargs.

Default path uses the `SVO Filter Profile Service <https://svo2.cab.inta-csic.es/theory/fps/>`_
via ``astroquery.svo_fps`` (optional dependency).
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Literal

import torch

FieldSpectralSource = Literal["svo", "builtin", "file", "gaussian"]

_GAUSS_FWHM_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


def field_observation_spectral_kwargs(spec: dict[str, Any]) -> dict[str, Any]:
    """Map a resolved bandpass dict to :class:`FieldObservation` keyword args."""
    out: dict[str, Any] = {
        "wavelength": float(spec["wavelength"]),
        "spectral_min": float(spec["spectral_min"]),
        "spectral_max": float(spec["spectral_max"]),
        "transmission_center": float(spec["transmission_center"]),
        "transmission_sigma": float(spec["transmission_sigma"]),
        "band_name": str(spec["band_name"]),
    }
    tw = spec.get("transmission_wavelength")
    tt = spec.get("transmission_throughput")
    if tw is not None or tt is not None:
        if tw is None or tt is None:
            raise ValueError(
                "transmission_wavelength and transmission_throughput must both be present in spec"
            )
        out["transmission_wavelength"] = torch.as_tensor(tw, dtype=torch.float32)
        out["transmission_throughput"] = torch.as_tensor(tt, dtype=torch.float32)
    return out


def _prepare_curve_for_field(
    wavelength: torch.Tensor,
    transmission: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort, clip, merge duplicate wavelengths (mean throughput) for tabulated bandpass."""
    w = torch.as_tensor(wavelength, dtype=torch.float64).reshape(-1)
    t = torch.clamp(
        torch.as_tensor(transmission, dtype=torch.float64).reshape(-1), min=0.0
    )
    if w.shape != t.shape:
        raise ValueError("wavelength and transmission must have the same shape")
    if w.numel() < 2:
        raise ValueError("tabulated transmission requires at least two samples")
    order = torch.argsort(w)
    w, t = w[order], t[order]
    uw, inv = torch.unique(w, return_inverse=True)
    if uw.numel() < 2:
        raise ValueError(
            "tabulated transmission requires at least two distinct wavelengths"
        )
    ts = torch.bincount(inv, weights=t, minlength=uw.numel())
    cnt = torch.bincount(inv, minlength=uw.numel())
    ta = ts / torch.clamp(cnt, min=1)
    return uw.to(torch.float32), ta.to(torch.float32)


def bandpass_gaussian(
    *,
    band_name: str,
    lambda_eff: float,
    lambda_min: float,
    lambda_max: float,
    sigma: float | None = None,
) -> dict[str, Any]:
    """Gaussian transmission envelope on ``[lambda_min, lambda_max]`` (same units, e.g. Å)."""
    lo = float(lambda_min)
    hi = float(lambda_max)
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        raise ValueError(
            "lambda_min/lambda_max must be finite with lambda_max > lambda_min"
        )
    leff = float(lambda_eff)
    if not math.isfinite(leff):
        raise ValueError("lambda_eff must be finite")
    if sigma is None:
        ts = (hi - lo) / _GAUSS_FWHM_SIGMA
    else:
        ts = float(sigma)
        if not math.isfinite(ts) or ts <= 0.0:
            raise ValueError("sigma must be finite and positive when given")
    tc = leff
    return {
        "wavelength": leff,
        "spectral_min": lo,
        "spectral_max": hi,
        "transmission_center": tc,
        "transmission_sigma": ts,
        "band_name": str(band_name).upper(),
    }


def bandpass_from_transmission_file(
    path: Path | str,
    *,
    band_name: str,
    wavelength_unit: Literal["angstrom", "nm"] = "angstrom",
) -> dict[str, Any]:
    """Load two-column ``wavelength transmission`` text; summarize to FieldObservation kwargs.

    Lines starting with ``#`` and blank lines are ignored. First two columns are parsed as
    floats (comma or whitespace separated).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"transmission file not found: {p}")
    rows: list[tuple[float, float]] = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "#" in s:
            s = s.split("#", 1)[0].strip()
        parts = re.split(r"[,\s]+", s)
        nums = [x for x in parts if x]
        if len(nums) < 2:
            continue
        try:
            rows.append((float(nums[0]), float(nums[1])))
        except ValueError:
            continue
    if len(rows) < 3:
        raise ValueError(f"{p}: need at least 3 wavelength/transmission samples")
    wl = torch.tensor([a for a, _ in rows], dtype=torch.float64)
    tr = torch.clamp(torch.tensor([b for _, b in rows], dtype=torch.float64), min=0.0)
    if str(wavelength_unit).lower() == "nm":
        wl = wl * 10.0
    return _summarize_transmission_curve(wl, tr, band_name=str(band_name).upper())


def _summarize_transmission_curve(
    wavelength: torch.Tensor,
    transmission: torch.Tensor,
    *,
    band_name: str,
) -> dict[str, Any]:
    t = torch.as_tensor(transmission, dtype=torch.float64)
    w = torch.as_tensor(wavelength, dtype=torch.float64)
    if w.shape != t.shape:
        raise ValueError("wavelength and transmission must have the same shape")
    peak = float(torch.max(t))
    if not math.isfinite(peak) or peak <= 0.0:
        raise ValueError("transmission must have a positive peak")
    mask = t > peak * 1.0e-3
    if not torch.any(mask):
        mask = t > 0.0
    w_eff = w[mask]
    t_eff = t[mask]
    order = torch.argsort(w_eff)
    w_eff = w_eff[order]
    t_eff = t_eff[order]
    num = float(torch.sum(w_eff * t_eff))
    den = float(torch.sum(t_eff))
    lam_eff = num / den if den > 0.0 else float(torch.mean(w_eff))
    lam_min = float(w_eff.min())
    lam_max = float(w_eff.max())
    half = 0.5 * peak
    above = t_eff >= half
    fwhm: float | None = None
    if torch.any(above):
        idx = torch.where(above)[0]
        fwhm = float(w_eff[idx.max()] - w_eff[idx.min()])
    sigma = (
        (fwhm / _GAUSS_FWHM_SIGMA)
        if fwhm and fwhm > 0.0
        else (lam_max - lam_min) / _GAUSS_FWHM_SIGMA
    )
    if not math.isfinite(sigma) or sigma <= 0.0:
        sigma = max((lam_max - lam_min) / _GAUSS_FWHM_SIGMA, 1.0)
    pivot = lam_eff
    w32, t32 = _prepare_curve_for_field(w, t)
    return {
        "wavelength": lam_eff,
        "spectral_min": lam_min,
        "spectral_max": lam_max,
        "transmission_center": pivot,
        "transmission_sigma": sigma,
        "band_name": str(band_name).upper(),
        "transmission_wavelength": w32,
        "transmission_throughput": t32,
    }


def _pick_svo_row(
    table: Any,
    *,
    band_token: str,
    prefer: Literal["classic", "2025", "any"],
) -> Any:
    band_u = str(band_token).strip().upper()
    ids = [str(r["filterID"]) for r in table]
    matches = [i for i, fid in enumerate(ids) if fid.upper().endswith(f".{band_u}")]
    if not matches:
        raise ValueError(
            f"No SVO filter with filterID ending in '.{band_u}' for this facility/instrument query"
        )

    def score(i: int) -> tuple[int, str]:
        fid = ids[i]
        u = fid.upper()
        if prefer == "2025":
            has25 = "2025" in u
            return (0 if has25 else 1, fid)
        if prefer == "classic":
            has25 = "2025" in u
            return (0 if not has25 else 1, fid)
        return (0, fid)

    best_i = min(matches, key=score)
    return table[best_i]


def _sanitize_filename(name: str) -> str:
    """Sanitize filter ID or facility/instrument name for use in cache filename."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(name).strip())


def _svo_cache_dir(custom_path: Path | str | None = None) -> Path:
    """Resolve directory for persistent SVO filter cache."""
    if custom_path is not None:
        p = Path(custom_path).expanduser().resolve()
    elif os.environ.get("TORCHSKY_SVO_CACHE"):
        p = Path(os.environ["TORCHSKY_SVO_CACHE"]).expanduser().resolve()
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg).expanduser().resolve() if xdg else Path.home() / ".cache"
        p = base / "torchsky" / "svo"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _atomic_write_json(path: Path, data: Any) -> None:
    """Atomically write JSON data via a temporary file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _load_cached_transmission(
    cache_dir: Path, filter_id: str
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return a cached curve, or ``None`` when missing/truncated/inconsistent.

    A damaged entry must trigger a refetch rather than a hard failure, so shape
    consistency is validated here instead of surfacing from ``_prepare_curve_for_field``.
    """
    cache_file = cache_dir / f"filter_{_sanitize_filename(filter_id)}.json"
    if not cache_file.is_file():
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        wl = torch.as_tensor(data["wavelength"], dtype=torch.float64).reshape(-1)
        tr = torch.as_tensor(data["transmission"], dtype=torch.float64).reshape(-1)
    except Exception:
        return None
    if wl.numel() < 2 or wl.shape != tr.shape:
        return None
    return wl, tr


def _save_cached_transmission(
    cache_dir: Path, filter_id: str, wl: torch.Tensor, tr: torch.Tensor
) -> None:
    cache_file = cache_dir / f"filter_{_sanitize_filename(filter_id)}.json"
    data = {
        "filter_id": filter_id,
        "wavelength": wl.detach().cpu().tolist(),
        "transmission": tr.detach().cpu().tolist(),
    }
    _atomic_write_json(cache_file, data)


def _load_cached_filter_list(
    cache_dir: Path, facility: str, instrument: str
) -> list[dict[str, Any]] | None:
    cache_file = (
        cache_dir
        / f"index_{_sanitize_filename(facility)}_{_sanitize_filename(instrument)}.json"
    )
    if not cache_file.is_file():
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        return None
    return None


def _save_cached_filter_list(
    cache_dir: Path, facility: str, instrument: str, table: Any
) -> None:
    cache_file = (
        cache_dir
        / f"index_{_sanitize_filename(facility)}_{_sanitize_filename(instrument)}.json"
    )
    rows: list[dict[str, Any]] = []
    for r in table:
        row_dict: dict[str, Any] = {"filterID": str(r["filterID"])}
        for k in (
            "WavelengthEff",
            "WavelengthMin",
            "WavelengthMax",
            "WavelengthPivot",
            "FWHM",
        ):
            try:
                val = r[k]
                if val is not None:
                    row_dict[k] = float(val)
            except (KeyError, IndexError, TypeError, ValueError):
                pass
        rows.append(row_dict)
    _atomic_write_json(cache_file, rows)


def _svo_fps() -> Any:
    """Import astroquery's SVO FPS client, failing with an install hint."""
    try:
        from astroquery.svo_fps import SvoFps  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "astroquery is required for --field-spectral-source=svo when the filter "
            "profile is not cached. Install with: pip install 'astroquery>=0.4.6'"
        ) from exc
    return SvoFps


def _svo_transmission(
    cache_dir: Path | None, filter_id: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one SVO transmission curve, preferring the on-disk cache."""
    if cache_dir is not None:
        cached = _load_cached_transmission(cache_dir, filter_id)
        if cached is not None:
            return cached
    data = _svo_fps().get_transmission_data(filter_id)
    wl = torch.as_tensor(_column_values(data["Wavelength"]), dtype=torch.float64)
    tr = torch.as_tensor(_column_values(data["Transmission"]), dtype=torch.float64)
    if cache_dir is not None:
        _save_cached_transmission(cache_dir, filter_id, wl, tr)
    return wl, tr


def _column_values(column: Any) -> Any:
    """Return an astropy column/Quantity as raw values."""
    return column.value if hasattr(column, "value") else column


def bandpass_from_svo(
    *,
    band_name: str,
    facility: str,
    instrument: str,
    filter_id: str | None,
    prefer: Literal["classic", "2025", "any"],
    cache_dir: Path | str | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Resolve an SVO filter profile and summarize it for a field observation."""
    bn = str(band_name).strip().upper()
    cdir = _svo_cache_dir(cache_dir) if use_cache else None

    if filter_id:
        fid = str(filter_id).strip()
    else:
        rows = (
            _load_cached_filter_list(cdir, facility=facility, instrument=instrument)
            if cdir is not None
            else None
        )
        if rows is None:
            rows = _svo_fps().get_filter_list(
                facility=facility, instrument=instrument
            )
            if cdir is not None:
                _save_cached_filter_list(
                    cdir, facility=facility, instrument=instrument, table=rows
                )
        fid = str(_pick_svo_row(rows, band_token=band_name, prefer=prefer)["filterID"])

    wl, tr = _svo_transmission(cdir, fid)
    spec = _summarize_transmission_curve(wl, tr, band_name=bn)
    spec["svo_filter_id"] = fid
    return spec


def resolve_field_bandpass(
    *,
    source: FieldSpectralSource,
    band_name: str,
    builtin_table: dict[str, tuple[float, float, float, float]],
    svo_facility: str,
    svo_instrument: str,
    svo_filter_id: str | None,
    svo_prefer: Literal["classic", "2025", "any"],
    svo_cache_dir: Path | str | None = None,
    svo_use_cache: bool = True,
    transmission_path: Path | None,
    transmission_dir: Path | None,
    transmission_unit: Literal["angstrom", "nm"],
    gaussian_lambda_eff: float | None,
    gaussian_lambda_min: float | None,
    gaussian_lambda_max: float | None,
    gaussian_sigma: float | None,
) -> dict[str, Any]:
    """Resolve spectral kwargs for one band name (e.g. ``F277W``)."""
    src = str(source)
    if src not in {"svo", "builtin", "file", "gaussian"}:
        raise ValueError(f"unknown field spectral source {source!r}")
    bn = str(band_name).strip().upper()
    if src == "builtin":
        if bn not in builtin_table:
            raise ValueError(
                f"Unknown filter/band {bn!r} for builtin field table; "
                "use --field-spectral-source=svo, file, or gaussian"
            )
        smin, smax, tc, ts = builtin_table[bn]
        return {
            "wavelength": float(tc),
            "spectral_min": float(smin),
            "spectral_max": float(smax),
            "transmission_center": float(tc),
            "transmission_sigma": float(ts),
            "band_name": bn,
        }
    if src == "gaussian":
        if (
            gaussian_lambda_eff is None
            or gaussian_lambda_min is None
            or gaussian_lambda_max is None
        ):
            raise ValueError(
                "gaussian spectral source requires --field-gaussian-lambda-eff, "
                "--field-gaussian-lambda-min, and --field-gaussian-lambda-max"
            )
        return bandpass_gaussian(
            band_name=bn,
            lambda_eff=float(gaussian_lambda_eff),
            lambda_min=float(gaussian_lambda_min),
            lambda_max=float(gaussian_lambda_max),
            sigma=gaussian_sigma,
        )
    if src == "file":
        path: Path | None = transmission_path
        if path is None and transmission_dir is not None:
            d = Path(transmission_dir)
            for name in (f"{bn}.txt", f"{bn}.dat", f"{bn}.csv"):
                cand = d / name
                if cand.is_file():
                    path = cand
                    break
        if path is None:
            raise ValueError(
                "file spectral source requires --field-transmission or a file "
                f"{bn}.txt|dat|csv under --field-transmission-dir"
            )
        return bandpass_from_transmission_file(
            path, band_name=bn, wavelength_unit=transmission_unit
        )
    if src == "svo":
        return bandpass_from_svo(
            band_name=bn,
            facility=str(svo_facility).strip(),
            instrument=str(svo_instrument).strip(),
            filter_id=svo_filter_id,
            prefer=svo_prefer,
            cache_dir=svo_cache_dir,
            use_cache=svo_use_cache,
        )
    raise AssertionError("unreachable")

def photon_quadrature(
    wavelength: torch.Tensor,
    throughput: torch.Tensor,
    num_nodes: int,
    *,
    flux_convention: Literal["fnu", "flambda"] = "fnu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equal-weight quadrature nodes for a photon-counting band average.

    The band average of a spectrum ``g`` seen by a photon counter is
    ``<g> = int g(l) T(l) k(l) dl / int T(l) k(l) dl`` with ``k = 1/l`` for
    ``f_nu`` (JWST, AB magnitudes) and ``k = l`` for ``f_lambda``. The nodes are the
    midpoints of ``num_nodes`` equal-mass bins of that measure, so
    ``<g> ~= mean_q g(l_q)``; they concentrate where the band transmits.
    Returns ``(nodes, weights)`` in the units of ``wavelength``; weights sum to one.
    """
    if num_nodes < 1:
        raise ValueError("num_nodes must be positive")
    w = torch.as_tensor(wavelength, dtype=torch.float64).flatten()
    t = torch.as_tensor(throughput, dtype=torch.float64).flatten().clamp(min=0.0)
    if w.shape != t.shape or w.numel() < 2:
        raise ValueError("wavelength and throughput must be 1-D curves of equal length >= 2")
    order = torch.argsort(w)
    w, t = w[order], t[order]
    if not bool((w > 0).all()):
        raise ValueError("wavelengths must be positive")
    if flux_convention == "fnu":
        density = t / w
    elif flux_convention == "flambda":
        density = t * w
    else:
        raise ValueError("flux_convention must be 'fnu' or 'flambda'")
    seg = 0.5 * (density[1:] + density[:-1]) * (w[1:] - w[:-1])
    cdf = torch.cat([torch.zeros(1, dtype=w.dtype), torch.cumsum(seg, 0)])
    if not float(cdf[-1]) > 0.0:
        raise ValueError("throughput must have positive integral")
    cdf = cdf / cdf[-1]
    q = (torch.arange(num_nodes, dtype=w.dtype) + 0.5) / num_nodes
    idx = torch.searchsorted(cdf, q).clamp(1, cdf.numel() - 1)
    frac = (q - cdf[idx - 1]) / (cdf[idx] - cdf[idx - 1]).clamp(min=1e-300)
    nodes = w[idx - 1] + frac * (w[idx] - w[idx - 1])
    weights = torch.full((num_nodes,), 1.0 / num_nodes, dtype=w.dtype)
    return nodes, weights
