"""Resolve INR :class:`~torchsky.map.field.FieldObservation` spectral kwargs.

Default path uses the `SVO Filter Profile Service <https://svo2.cab.inta-csic.es/theory/fps/>`_
via ``astroquery.svo_fps`` (optional dependency).
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Literal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy
else:
    import numpy

import torch

InrSpectralSource = Literal["svo", "builtin", "file", "gaussian"]

_GAUSS_FWHM_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


def field_observation_spectral_kwargs(spec: dict[str, Any]) -> dict[str, Any]:
    """Map a resolved bandpass dict to :class:`FieldObservation` keyword args."""
    import numpy as np
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
    import numpy as np
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


def bandpass_from_svo(
    *,
    band_name: str,
    facility: str,
    instrument: str,
    filter_id: str | None,
    prefer: Literal["classic", "2025", "any"],
) -> dict[str, Any]:
    import numpy as np
    try:
        from astroquery.svo_fps import SvoFps  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "astroquery is required for --inr-spectral-source=svo. "
            "Install with: pip install 'astroquery>=0.4.6'"
        ) from exc

    bn = str(band_name).strip().upper()
    if filter_id:
        fid = str(filter_id).strip()
        data = SvoFps.get_transmission_data(fid)
        wl = torch.as_tensor(
            data["Wavelength"].value
            if hasattr(data["Wavelength"], "value")
            else data["Wavelength"],
            dtype=torch.float64,
        )
        tr = torch.as_tensor(
            data["Transmission"].value
            if hasattr(data["Transmission"], "value")
            else data["Transmission"],
            dtype=torch.float64,
        )
        spec = _summarize_transmission_curve(wl, tr, band_name=bn)
        spec["svo_filter_id"] = fid
        return spec

    fl = SvoFps.get_filter_list(facility=facility, instrument=instrument)
    row = _pick_svo_row(fl, band_token=band_name, prefer=prefer)
    fid_out = str(row["filterID"])
    data = SvoFps.get_transmission_data(fid_out)
    wl = torch.as_tensor(
        data["Wavelength"].value
        if hasattr(data["Wavelength"], "value")
        else data["Wavelength"],
        dtype=torch.float64,
    )
    tr = torch.as_tensor(
        data["Transmission"].value
        if hasattr(data["Transmission"], "value")
        else data["Transmission"],
        dtype=torch.float64,
    )
    spec = _summarize_transmission_curve(wl, tr, band_name=bn)
    spec["svo_filter_id"] = fid_out
    return spec


def resolve_inr_bandpass(
    *,
    source: InrSpectralSource,
    band_name: str,
    builtin_table: dict[str, tuple[float, float, float, float]],
    svo_facility: str,
    svo_instrument: str,
    svo_filter_id: str | None,
    svo_prefer: Literal["classic", "2025", "any"],
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
        raise ValueError(f"unknown INR spectral source {source!r}")
    bn = str(band_name).strip().upper()
    if src == "builtin":
        if bn not in builtin_table:
            raise ValueError(
                f"Unknown filter/band {bn!r} for builtin INR table; "
                "use --inr-spectral-source=svo, file, or gaussian"
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
                "gaussian spectral source requires --inr-gaussian-lambda-eff, "
                "--inr-gaussian-lambda-min, and --inr-gaussian-lambda-max"
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
                "file spectral source requires --inr-transmission or a file "
                f"{bn}.txt|dat|csv under --inr-transmission-dir"
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
        )
    raise AssertionError("unreachable")
