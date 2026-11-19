"""Conversion of ``FieldObservation`` objects to the fit's private ``Exposure`` records.

Bookkeeping cards read from ``observation.header`` (all written by ``load_exposures``; a hand-built observation must carry them):

* ``TSINST`` instrument name (``NIRCAM``, ``WFC3_IR``, ``ACS_WFC``), ``DETECTOR`` detector name;
* ``TSNAME`` unique exposure name; ``TSSPLIT`` ``"train"`` (default) or ``"held"``; ``TSGROUP`` dither group (optional);
* ``TSBOXRA``, ``TSBOXDE``, ``TSBOXHS`` centre [deg] and half side [arcsec] of the sky box the cut was made for;
* ``TSDETNX``, ``TSDETNY`` size of the whole detector chip [pixel] (the cut's own size is the image shape).

Positions: the header solution (``torchsky.wcs.WCS``) of the cut is evaluated at every pixel and projected onto the tangent plane at the box
centre, X to the west and Y to the north in arcsec, exactly as the prototype did.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from ...instrument import instrument_description
from ...wcs import WCS
from ._bandpass import Filter, finish_filter
from ._exposures import Exposure, mean_jacobian, rotation_of

_SCALE = 0.01  # arcsec per pixel of the projection plane (the prototype's value)
_REQUIRED = ("TSINST", "TSBOXRA", "TSBOXDE", "TSDETNX", "TSDETNY")


def _card(obs, key, default=None):
    h = obs.header
    if key in h and h[key] not in (None, ""):
        return h[key]
    if default is None:
        raise ValueError(
            f"observation {obs.band_name!r}: header card {key} is missing (written by load_exposures)"
        )
    return default


def tangent_positions(header: dict, shape: tuple[int, int], ra: float, dec: float):
    """X (west) and Y (north) [arcsec] of every pixel of a cut, float64 arrays of ``shape``."""
    ny, nx = shape
    size = 8
    thead = {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CRVAL1": ra,
        "CRVAL2": dec,
        "CRPIX1": size / 2 + 0.5,
        "CRPIX2": size / 2 + 0.5,
        "CD1_1": -_SCALE / 3600.0,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": _SCALE / 3600.0,
    }
    wd, wt = WCS(header=header), WCS(header=thead)
    rr, cc = np.mgrid[0:ny, 0:nx]
    pc = torch.tensor(np.stack([cc.ravel(), rr.ravel()], 1), dtype=torch.float64)
    tp = wt.world_to_pixel_differentiable(
        wd.pixel_to_world_differentiable(pc, origin=0), origin=0
    ).numpy()
    c = (size - 1) / 2
    return ((tp[:, 0] - c) * _SCALE).reshape(ny, nx), ((tp[:, 1] - c) * _SCALE).reshape(
        ny, nx
    )


def _bandpass(obs, n_lambda: int) -> Filter:
    if obs.transmission_wavelength is None or obs.transmission_throughput is None:
        raise ValueError(
            f"observation {obs.band_name!r} has no filter curve (transmission_wavelength / transmission_throughput); "
            "give the instrument description filter curves (InstrumentDescription.with_filter_curves)"
        )
    if obs.spectral_unit not in ("um", "micron", "micrometre"):
        raise ValueError(
            f"observation {obs.band_name!r}: spectral_unit must be 'um' for the filter curve, got {obs.spectral_unit!r}"
        )
    lam = obs.transmission_wavelength.detach().cpu().double().numpy()
    T = obs.transmission_throughput.detach().cpu().double().numpy()
    return finish_filter(
        Filter(str(obs.band_name), str(_card(obs, "TSINST")), lam, T), n_lambda
    )


def observation_to_record(
    obs,
    *,
    filt: int,
    inst: int,
    chan: int,
    det: int,
    bandpass: Filter,
    name: str | None = None,
) -> Exposure:
    """One private record from one observation (indices are those of the fit's lists)."""
    for k in _REQUIRED:
        _card(obs, k)
    if obs.err is None or obs.mask is None:
        raise ValueError(f"observation {obs.band_name!r} needs err and mask")
    desc = instrument_description(str(_card(obs, "TSINST")))
    ra, dec = float(_card(obs, "TSBOXRA")), float(_card(obs, "TSBOXDE"))
    X, Y = tangent_positions(obs.header, tuple(obs.image.shape), ra, dec)
    valid = obs.mask.detach().cpu().bool()
    img = obs.image.detach().cpu().float()
    err = obs.err.detach().cpu().float()
    d = np.nan_to_num(img.numpy())
    sig = np.where(
        np.isfinite(err.numpy()) & (err.numpy() > 0), err.numpy(), 1.0
    ).astype(np.float32)
    J = mean_jacobian(X, Y)
    vn = valid.numpy()
    if not vn.any():
        raise ValueError("observation has no valid pixel")
    nm = name or str(_card(obs, "TSNAME", f"{obs.band_name}_{id(obs)}"))
    return Exposure(
        name=nm,
        filt=filt,
        inst=inst,
        chan=chan,
        det=det,
        det_shape=(int(_card(obs, "TSDETNY")), int(_card(obs, "TSDETNX"))),
        cut0=(int(round(obs.detector_y0_pix)), int(round(obs.detector_x0_pix))),
        d=torch.from_numpy(d),
        sig=torch.from_numpy(sig),
        valid=valid,
        X=torch.from_numpy(X.astype(np.float32)),
        Y=torch.from_numpy(Y.astype(np.float32)),
        R=rotation_of(J),
        split=str(_card(obs, "TSSPLIT", "train")),
        group=str(_card(obs, "TSGROUP", "x")) if "TSGROUP" in obs.header else "",
        shift0=(0.0, 0.0),
        stripes="stripes" in desc.detector_terms,
        bkg0=float(np.median(img.numpy()[vn])),
        extra={
            "J": J.tolist(),
            "detector": str(_card(obs, "DETECTOR", "")),
            "filter": str(obs.band_name),
            "diameter_m": desc.diameter_m,
            "instrument": desc.name,
        },
        bandpass=bandpass,
        inst_name=desc.name,
        det_name=str(_card(obs, "DETECTOR", desc.name)),
        chan_name=desc.channel_of(str(_card(obs, "DETECTOR", ""))),
    )


def observations_to_exposures(
    observations: Sequence, *, n_lambda: int
) -> tuple[list[Exposure], float | None]:
    """Records for a list of observations and the common box half side [arcsec] (None when the cards differ or are absent).

    Index rules follow the prototype's loader: instruments sorted by name; detectors and channels in order of first appearance; filters
    sorted by the digits of the name for one instrument, by pivot wavelength for several.
    """
    obs = list(observations)
    if not obs:
        raise ValueError("fit_sky needs at least one observation")
    insts = sorted({str(_card(o, "TSINST")) for o in obs})
    curves: dict[tuple[str, str], Filter] = {}
    for o in obs:
        key = (str(_card(o, "TSINST")), str(o.band_name))
        if key not in curves:
            curves[key] = _bandpass(o, n_lambda)
    names = {}
    for ins, f in curves:
        if names.setdefault(f, ins) != ins:
            raise ValueError(
                f"filter {f} is used by two instruments ({names[f]}, {ins}); filters are keyed by name"
            )
    if len(insts) <= 1:
        order = sorted(
            names,
            key=lambda f: (
                (0, int("".join(c for c in f if c.isdigit())))
                if any(c.isdigit() for c in f)
                else (1, f)
            ),
        )
    else:
        order = [
            c.name
            for c in sorted(
                (curves[(names[f], f)] for f in names), key=lambda c: (c.pivot, c.name)
            )
        ]
    dets: list[str] = []
    chans: list[str] = []
    recs = []
    seen: set[str] = set()
    for i, o in enumerate(obs):
        desc = instrument_description(str(_card(o, "TSINST")))
        det = str(_card(o, "DETECTOR", desc.name))
        chan = desc.channel_of(det if desc.name == "NIRCAM" else "")
        if det not in dets:
            dets.append(det)
        if chan not in chans:
            chans.append(chan)
        nm = str(o.header.get("TSNAME", f"{o.band_name}_{i}"))
        if nm in seen:
            raise ValueError(f"two observations share the name {nm!r}")
        seen.add(nm)
        recs.append(
            observation_to_record(
                o,
                filt=order.index(str(o.band_name)),
                inst=insts.index(desc.name),
                chan=chans.index(chan),
                det=dets.index(det),
                bandpass=curves[(desc.name, str(o.band_name))],
                name=nm,
            )
        )
    halves = {float(o.header["TSBOXHS"]) for o in obs if "TSBOXHS" in o.header}
    return recs, (
        halves.pop()
        if len(halves) == 1 and all("TSBOXHS" in o.header for o in obs)
        else None
    )
