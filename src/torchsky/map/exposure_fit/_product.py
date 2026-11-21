"""``SkyFit.product``: the sky image and its standard deviation, from the moved uncertainty code (``_uncertainty.py``, numerics unchanged).

This module is glue: it converts the training observations back to the fit's private records, builds the Gauss-Newton normal operator of
``_uncertainty.py``, runs one shared solve for all requested bands, reads each band out, and packages the arrays with their header and the
stated approximations. Nothing here fits anything; the fit is not changed.

Conventions:

* Grid. The sky nodes are at ``(i - (N - 1) / 2) * pitch`` from the box centre, column index along X (west) and row index along Y (north), the same
  tangent plane as the fit's header positions. The default product grid is the node grid.
* Resolution. The product of a band is the continuous sky convolved with a unit-integral circular Gaussian. ``resolution`` is its FWHM [arcsec];
  the moved code works with the Gaussian standard deviation ``s``, ``FWHM = 2 sqrt(2 ln 2) s``. ``"native"`` is the median over the covered nodes
  of the supported resolution of the band (smallest ``s`` from which the noise per element stops rising faster than ``alpha = 2``; see
  ``_uncertainty.supported_resolution``), so each band has its own native resolution from the one shared solve.
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
import torch

from . import _uncertainty as U
from ._observations import observation_to_record
from ._skyfit import _module_state

if TYPE_CHECKING:
    from ..field.types import FieldObservation

FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
_CLOCK = time.perf_counter  # tests replace it to cut a solve at a chosen moment
_MAXITER = 60  # conjugate-gradient iterations per solve (the prototype's default; tests may raise it)

ZERO_LEVEL_CONVENTION = (
    "median rule: for each filter, the mean over the training exposures of the fitted background constant equals the mean of "
    "their data medians; the sky zero level is tied to the data medians and its uncertainty is not in std"
)


@dataclasses.dataclass(frozen=True)
class SkyProduct:
    """The sky of one or several bands with its standard deviation, on a stated grid and resolution (``SkyFit.product`` returns one).

    The product of a band is the fitted sky of that band convolved with a circular Gaussian of the stated full width at half maximum, sampled on
    the grid of ``header``. ``std`` is the standard deviation of that same quantity from the linearised (Gauss-Newton) posterior with the response
    held fixed; see ``notes`` for what it does not include.

    Attributes:
        image: ``[B, H, W]`` float32 tensor, surface brightness in ``unit``; row index along north (Y), column index along west (X).
        std: ``[B, H, W]`` float32 standard deviation of ``image``, same units; 0 where not ``valid``.
        valid: ``[H, W]`` bool; True where the sky is covered by training pixels and ``std`` is finite and positive.
        bands: the ``B`` filter names, in the order of ``image``.
        resolution_fwhm_arcsec: per band, the full width at half maximum [arcsec] of the Gaussian that ``image`` and ``std`` are smoothed with.
        unit: the unit of ``image`` and ``std`` (``"MJy/sr"`` for data from ``load_exposures``).
        header: a tangent-plane (``RA---TAN`` / ``DEC--TAN``) WCS as a plain dict of cards (``NAXIS1``, ``NAXIS2``, ``CRVAL``, ``CRPIX``, ``CD``,
            ``BUNIT``), evaluated by ``torchsky.wcs.WCS``. ``CRVAL`` is the centre of the sky box; ``CD1_1`` is negative (east to the left).
        pitch_arcsec: the grid spacing [arcsec].
        zero_level_convention: a short statement of the rule that fixes the sky zero level (median rule); its uncertainty is not in ``std``.
        polished: True when ``image`` is the fit plus one Gauss-Newton step (the estimator ``std`` describes). False when the step was not
            requested or did not run: ``image`` is then the optimiser's own sky, whose error is NOT ``std`` (its noise is about half of ``std`` and it
            carries a bias; the pulls of such a sky are not calibrated).
        method: how ``std`` was obtained: ``"draws"`` (posterior draws of the linearised posterior) or ``"circulant"`` (Fourier approximation, uniform
            coverage). It is ``"circulant"`` also when draws were asked for but the time budget allowed none.
        n_samples: number of posterior draws that completed (0 for the circulant method).
        truncated: True when the time budget (``max_minutes``) cut a solve.
        residual: relative residual of the conjugate-gradient solve that was cut when ``truncated``; otherwise that of the Gauss-Newton step
            (``nan`` when the step was not run).
        n_passes: number of applications of the normal operator (each is one forward and one adjoint render of every training pixel).
        notes: the stated approximations, one sentence each.
    """

    image: torch.Tensor
    std: torch.Tensor
    valid: torch.Tensor
    bands: tuple
    resolution_fwhm_arcsec: tuple
    unit: str
    header: dict
    pitch_arcsec: float
    zero_level_convention: str
    polished: bool
    method: str
    n_samples: int
    truncated: bool
    residual: float
    n_passes: int
    notes: tuple


# ----------------------------------------------------------------------------- arguments
def _check_arguments(
    method, polish, samples, batch_pixels, max_minutes, resolution, pitch_arcsec, region
):
    if method not in ("circulant", "draws"):
        raise ValueError(f"method must be 'circulant' or 'draws', not {method!r}")
    if not isinstance(polish, (bool, np.bool_)):
        raise ValueError(f"polish must be True or False, not {polish!r}")
    if method == "draws" and (
        isinstance(samples, bool) or not isinstance(samples, int) or samples < 1
    ):
        raise ValueError(
            f"samples must be an integer of at least 1 for method='draws', not {samples!r}"
        )
    if (
        isinstance(batch_pixels, bool)
        or not isinstance(batch_pixels, int)
        or batch_pixels < 1
    ):
        raise ValueError(
            f"batch_pixels must be a positive integer, not {batch_pixels!r}"
        )
    if max_minutes is not None and not (
        isinstance(max_minutes, (int, float))
        and math.isfinite(max_minutes)
        and max_minutes > 0
    ):
        raise ValueError(f"max_minutes must be None or positive, not {max_minutes!r}")
    if isinstance(resolution, str):
        if resolution != "native":
            raise ValueError(
                f"resolution must be 'native' or a FWHM in arcsec, not {resolution!r}"
            )
    elif not (
        isinstance(resolution, (int, float))
        and not isinstance(resolution, bool)
        and math.isfinite(resolution)
        and resolution > 0
    ):
        raise ValueError(
            f"resolution must be 'native' or a positive FWHM in arcsec, not {resolution!r}"
        )
    if pitch_arcsec is not None and not (
        isinstance(pitch_arcsec, (int, float))
        and not isinstance(pitch_arcsec, bool)
        and math.isfinite(pitch_arcsec)
        and pitch_arcsec > 0
    ):
        raise ValueError(
            f"pitch_arcsec must be None or a positive number, not {pitch_arcsec!r}"
        )
    if region is not None:
        ok = (
            isinstance(region, (tuple, list))
            and len(region) == 4
            and all(
                isinstance(v, (int, float))
                and math.isfinite(v)
                and not isinstance(v, bool)
                for v in region
            )
            and region[0] < region[1]
            and region[2] < region[3]
        )
        if not ok:
            raise ValueError(
                "region must be None or (x_min, x_max, y_min, y_max) in arcsec from the box centre (X west, Y north) with x_min < x_max and y_min < y_max, "
                f"not {region!r}"
            )


def resolve_bands(fset_names: Sequence[str], bands) -> list[tuple[int, str]]:
    """Filter indices and names for ``bands``: a name, an index, ``"all"``, or a sequence of names and indices."""
    names = list(fset_names)
    if not isinstance(bands, (str, int, np.integer, list, tuple)):
        raise ValueError(
            f"bands is a filter name, an index, 'all' or a sequence of them, not {bands!r}; the fit has {names}"
        )
    items = [bands] if isinstance(bands, (str, int, np.integer)) else list(bands)
    if isinstance(bands, str) and bands == "all":
        items = list(names)
    if not items:
        raise ValueError("bands is empty")
    out: list[tuple[int, str]] = []
    for b in items:
        if isinstance(b, str):
            if b not in names:
                raise ValueError(f"unknown band {b!r}; the fit has {names}")
            i = names.index(b)
        elif isinstance(b, (int, np.integer)) and not isinstance(b, bool):
            i = int(b)
            if not 0 <= i < len(names):
                raise ValueError(
                    f"band index {i} is outside 0..{len(names) - 1}; the fit has {names}"
                )
        else:
            raise ValueError(
                f"a band is a filter name or an index, not {b!r}; the fit has {names}"
            )
        if any(i == j for j, _ in out):
            raise ValueError(f"band {names[i]!r} is listed twice")
        out.append((i, names[i]))
    return out


# ----------------------------------------------------------------------------- the training records
def training_records(fit, observations) -> tuple[list, str]:
    """The fit's training exposures as private records, rebuilt from ``observations`` (matched by card ``TSNAME``; held-out ones are ignored).

    Every training exposure of the fit must be present, once. The data are checked against what the fit saw (median error and background
    reference per exposure): a mismatch means these are not the exposures the fit was made from.
    """
    names = fit.exposure_names
    op = fit.operator
    meta = op.meta
    by_name: dict[str, FieldObservation] = {}
    for o in observations:
        n = str(o.header.get("TSNAME", ""))
        if n not in names:
            raise ValueError(
                f"observation {n!r} (card TSNAME) was not in the list given to fit_sky"
            )
        if n in by_name:
            raise ValueError(f"two observations share the name {n!r}")
        by_name[n] = o
    centre = fit._structure.get("box_centre_deg")
    recs, units = [], set()
    for row, n in enumerate(names):
        if not bool(op.train_mask[row]):
            continue
        if n not in by_name:
            raise ValueError(
                f"training exposure {n!r} of the fit is missing from observations: product needs every training exposure the fit was made from"
            )
        o = by_name[n]
        if str(o.header.get("TSSPLIT", "train")) != "train":
            raise ValueError(
                f"observation {n!r} is marked held-out but was a training exposure of the fit"
            )
        if centre is not None:
            ra, de = float(o.header["TSBOXRA"]), float(o.header["TSBOXDE"])
            if abs(ra - centre[0]) > 1e-9 or abs(de - centre[1]) > 1e-9:
                raise ValueError(
                    f"observation {n!r}: sky box centre ({ra}, {de}) differs from the fit's {tuple(centre)}"
                )
        rec = observation_to_record(
            o,
            filt=meta["filt"][row],
            inst=meta["inst"][row],
            chan=meta["chan"][row],
            det=meta["det"][row],
            bandpass=op.fset.filters[meta["filt"][row]],
            name=n,
        )
        if abs(float(rec.sig[rec.valid].median()) - float(op.e_sigbar[row])) > 1e-5 * (
            1.0 + abs(float(op.e_sigbar[row]))
        ) or abs(rec.bkg0 - float(op.bkg0[row])) > 1e-5 * (
            1.0 + abs(float(op.bkg0[row]))
        ):
            raise ValueError(
                f"observation {n!r} does not match the exposure the fit was made from (median error or median value differ)"
            )
        if getattr(o, "value_unit", None):
            units.add(str(o.value_unit))
        dev = op.dev
        recs.append(
            dataclasses.replace(
                rec,
                eid=row,
                cid=row,
                split="train",
                extra=dict(rec.extra),
                patches=[],
                d=rec.d.to(dev),
                sig=rec.sig.to(dev),
                valid=rec.valid.to(dev),
                X=rec.X.to(dev),
                Y=rec.Y.to(dev),
            )
        )
    if len(units) > 1:
        raise ValueError(f"the observations carry different units: {sorted(units)}")
    return recs, (units.pop() if units else "MJy/sr")


# ----------------------------------------------------------------------------- grids and header
def output_axes(N: int, pitch: float, pitch_arcsec, region):
    """(x_out, y_out) in arcsec (float64 tensors) of the product grid; (None, None) for the node grid."""
    if pitch_arcsec is None and region is None:
        return None, None
    half = (N - 1) / 2 * pitch
    p = float(pitch if pitch_arcsec is None else pitch_arcsec)
    x0, x1, y0, y1 = (-half, half, -half, half) if region is None else region
    nx = int(math.floor((x1 - x0) / p + 1e-9)) + 1
    ny = int(math.floor((y1 - y0) / p + 1e-9)) + 1
    return (
        x0 + p * torch.arange(nx, dtype=torch.float64),
        y0 + p * torch.arange(ny, dtype=torch.float64),
    )


def tangent_header(
    shape: tuple[int, int],
    pitch: float,
    x0: float,
    y0: float,
    ra: float,
    dec: float,
    unit: str,
) -> dict:
    """Tangent-plane WCS of a product grid: X (west) along columns from ``x0``, Y (north) along rows from ``y0`` [arcsec], tangent point
    (``ra``, ``dec``) [deg]. The same sign convention as the exposure positions: east to the left (``CD1_1 < 0``), north up (``CD2_2 > 0``)."""
    ny, nx = shape
    return {
        "NAXIS": 2,
        "NAXIS1": int(nx),
        "NAXIS2": int(ny),
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CRVAL1": float(ra),
        "CRVAL2": float(dec),
        "CRPIX1": float(-x0 / pitch + 1.0),
        "CRPIX2": float(-y0 / pitch + 1.0),
        "CD1_1": -float(pitch) / 3600.0,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": float(pitch) / 3600.0,
        "BUNIT": unit,
    }


# ----------------------------------------------------------------------------- the computation
def compute(
    op,
    exposures,
    bands: Sequence[int],
    *,
    pitch_arcsec=None,
    resolution="native",
    region=None,
    method="circulant",
    polish=True,
    samples=8,
    batch_pixels=16000,
    max_minutes=None,
    log: Callable[[str], None] | None = None,
    t_start: float | None = None,
) -> dict:
    """The arrays of the product from a field operator and private training records (float64, CPU). ``SkyFit.product`` wraps this.

    ``bands`` are filter indices. Returns a dict with ``image`` and ``std`` of shape ``(B, H, W)``, ``valid`` ``(H, W)``, the per-band ``s`` (Gaussian
    standard deviation [arcsec]), the shared-solve record and the grid axes. Must be called inside ``_module_state`` of the fit.
    """
    log = log or U._silent
    t0 = _CLOCK() if t_start is None else t_start
    sky = op.sky
    pitch, N = sky.pitch, sky.N
    plan: dict[str, Any] = dict(
        method=method,
        polish=bool(polish),
        n_samples=samples,
        maxiter=_MAXITER,
        n_exact=0,
        n_bands=len(bands),
    )
    uop = U.SkyNormalOperator(
        op, exposures, op.cfg, rho=10.0, log=log, batch_px=batch_pixels, plan=plan
    )
    pl = U.plan_passes(uop.K, **plan)
    mon = None
    if max_minutes is not None:
        mon = U.Monitor(
            uop,
            log=log,
            max_minutes=max_minutes,
            t0=t0,
            planned_apps=pl["setup"] + pl["polish"] + pl["draws"],
            reserve=60.0 + 30.0 * len(bands),
            clock=_CLOCK,
        )
    shared = U.solve_shared(
        uop,
        n_samples=samples,
        method=method,
        polish=bool(polish),
        tol=1e-4,
        maxiter=_MAXITER,
        log=log,
        monitor=mon,
    )
    x_out, y_out = output_axes(N, pitch, pitch_arcsec, region)
    images, stds, s_used, readouts = [], [], [], []
    valid_node = None
    for b in bands:
        r = U.band_readout(
            op,
            uop,
            shared,
            int(b),
            resolution="auto"
            if isinstance(resolution, str)
            else float(resolution) / FWHM_PER_SIGMA,
            log=log,
            monitor=mon,
            rho=10.0,
            tol=1e-4,
        )
        if not math.isfinite(r["resolution"]):
            raise ValueError(
                f"the supported resolution of band {int(b)} could not be determined (the noise never settles on the scale ladder); give an explicit resolution"
            )
        readouts.append(r)
        s_used.append(float(r["resolution"]))
        node_std = r["sigma"]
        vn = r["covered"] & torch.isfinite(node_std) & (node_std > 0)
        valid_node = vn if valid_node is None else (valid_node & vn)
        if x_out is None:
            images.append(r["image"])
            stds.append(node_std)
        else:
            phi = r["phi"].double()
            A = (
                shared["A_pol"] if shared["A_pol"] is not None else uop.A_fit.cpu()
            ).double()
            x_nodes = (torch.arange(N, dtype=torch.float64) - (N - 1) / 2) * pitch
            Mx = U.tent_gauss_1d(x_out, x_nodes, pitch, s_used[-1])
            My = U.tent_gauss_1d(y_out, x_nodes, pitch, s_used[-1])
            images.append(My @ U.band_of(A, phi) @ Mx.T)
            Ix = U.tent_gauss_1d(x_out, x_nodes, pitch, 0.0)
            Iy = U.tent_gauss_1d(y_out, x_nodes, pitch, 0.0)
            stds.append(Iy @ node_std.double() @ Ix.T)
    # bands is non-empty (resolve_bands refuses an empty list)
    assert valid_node is not None
    if x_out is None:
        valid = valid_node.clone()
    else:
        x_nodes = (torch.arange(N, dtype=torch.float64) - (N - 1) / 2) * pitch
        Ix = U.tent_gauss_1d(x_out, x_nodes, pitch, 0.0)
        Iy = U.tent_gauss_1d(y_out, x_nodes, pitch, 0.0)
        valid = (Iy @ valid_node.double() @ Ix.T) > 1.0 - 1e-9
    image = torch.stack([i.double() for i in images])
    std = torch.stack([s.double() for s in stds]) * valid.double()
    n_done = len(shared["recs"])
    pol = shared["polish"]
    polished = bool(
        shared["A_pol"] is not None and pol is not None and pol["iterations"] > 0
    )
    return {
        "image": image,
        "std": std,
        "valid": valid,
        "s": s_used,
        "polished": polished,
        "method": "draws" if shared["samples"] is not None else "circulant",
        "n_samples": n_done if shared["samples"] is not None else 0,
        "truncated": bool(shared["truncated"]),
        # a solve cut before its first iteration has residual 1 (the zero iterate); the moved code leaves nan there
        "residual": 1.0
        if shared["truncated"] and not math.isfinite(shared["residual"])
        else float(shared["residual"]),
        "n_passes": int(uop.n_apply),
        "n_forward": int(uop.n_forward),
        "seconds": _CLOCK() - t0,
        "x_out": x_out,
        "y_out": y_out,
        "shared": shared,
        "readouts": readouts,
        "operator": uop,
    }


def _notes(res: dict, method_asked: str, custom_grid: bool, polish_asked: bool):
    n = [
        "The fitted response (kernel, throughput), the registration (affine terms), the per-exposure gains and the noise terms (scales and floors) are "
        "held at their fitted values; their uncertainty is not in std.",
        "The six background terms of every training exposure are marginalised. The sky modes that are exactly degenerate with them (a constant, a "
        "gradient and a quadratic over the field) are tied to the background convention (the training mean of each term is held), so they are not in std.",
        "The zero level of the sky is the median rule of the fit (one number per filter); its uncertainty is not in std.",
        "The sky of the fit is a set of positive Gaussian items; std is the posterior of the free sky grid at the items' raster (the items are not "
        "linearised), with the Huber weights of the fit and a weak prior on each grid value; the product may be negative.",
    ]
    if res["method"] == "circulant":
        n.append(
            "std is the circulant (Fourier) approximation: it assumes uniform coverage (scaled by the local relative coverage), and ignores chip gaps, "
            "edges and the background terms; it agrees with posterior draws to about 1 per cent at the supported resolution on the known-truth problem."
        )
        if method_asked == "draws":
            n.append(
                "Posterior draws were asked for but the time budget allowed none: std is the circulant approximation."
            )
    else:
        n.append(
            f"std is from {res['n_samples']} posterior draws of 60 conjugate-gradient iterations, averaged over a Gaussian window of three times the "
            "product width; its sampling error is about 1/sqrt(2 n) in variance and the truncated solves leave the variance a few per cent low."
        )
    if res["polished"]:
        n.append(
            "image is the fit plus one Gauss-Newton step with the response held (the estimator std describes); positivity is not imposed."
        )
    else:
        n.append(
            "image is the optimiser's own sky, not the Gauss-Newton step: its error is not std (its noise is about half of std and it is biased); do not "
            "quote std for it."
        )
    if polish_asked and not res["polished"]:
        n.append(
            "The Gauss-Newton step was asked for but the time budget did not allow it to run."
        )
    if res["truncated"]:
        n.append(
            f"The time budget cut a solve (relative residual {res['residual']:.2e}); a cut Gauss-Newton step is the last iterate, a cut draw is discarded."
        )
    if custom_grid:
        n.append(
            "On a grid other than the node grid, image is evaluated exactly at the output positions, but std and valid are interpolated linearly from the "
            "node grid (valid needs all four surrounding nodes valid)."
        )
    return tuple(n)


def make_product(
    fit,
    observations,
    bands,
    pitch_arcsec=None,
    resolution="native",
    region=None,
    *,
    method="circulant",
    polish=True,
    samples=8,
    batch_pixels=16000,
    max_minutes=None,
    log=None,
) -> SkyProduct:
    """Implementation of :meth:`SkyFit.product`."""
    t0 = _CLOCK()
    _check_arguments(
        method,
        polish,
        samples,
        batch_pixels,
        max_minutes,
        resolution,
        pitch_arcsec,
        region,
    )
    op = fit.operator
    chosen = resolve_bands(op.fset.names, bands)
    recs, unit = training_records(fit, list(observations))
    cfg = op.cfg
    with _module_state(domain=fit._domain(), item_chunk=cfg.item_chunk):
        res = compute(
            op,
            recs,
            [i for i, _ in chosen],
            pitch_arcsec=pitch_arcsec,
            resolution=resolution,
            region=region,
            method=method,
            polish=polish,
            samples=samples,
            batch_pixels=batch_pixels,
            max_minutes=max_minutes,
            log=log,
            t_start=t0,
        )
    pitch = float(op.sky.pitch if pitch_arcsec is None else pitch_arcsec)
    N = op.sky.N
    x_out, y_out = res["x_out"], res["y_out"]
    if x_out is None:
        x0 = y0 = -(N - 1) / 2 * op.sky.pitch
        shape = (N, N)
    else:
        x0, y0 = float(x_out[0]), float(y_out[0])
        shape = (len(y_out), len(x_out))
    centre = fit._structure.get("box_centre_deg")
    notes = list(_notes(res, method, x_out is not None, bool(polish)))
    if centre is None:
        notes.append(
            "The sky position of the box centre is not known to this fit (it was made from records without the cards TSBOXRA and TSBOXDE): CRVAL is 0, 0."
        )
    header = tangent_header(
        shape,
        pitch,
        x0,
        y0,
        0.0 if centre is None else centre[0],
        0.0 if centre is None else centre[1],
        unit,
    )
    return SkyProduct(
        image=res["image"].float(),
        std=res["std"].float(),
        valid=res["valid"].clone(),
        bands=tuple(n for _, n in chosen),
        resolution_fwhm_arcsec=tuple(s * FWHM_PER_SIGMA for s in res["s"]),
        unit=unit,
        header=header,
        pitch_arcsec=pitch,
        zero_level_convention=ZERO_LEVEL_CONVENTION,
        polished=res["polished"],
        method=res["method"],
        n_samples=res["n_samples"],
        truncated=res["truncated"],
        residual=res["residual"],
        n_passes=res["n_passes"],
        notes=tuple(notes),
    )


# ----------------------------------------------------------------------------- half difference
def half_difference(product_a: SkyProduct, product_b: SkyProduct) -> dict:
    """Pull statistics of two products made from disjoint halves of the exposures (the empirical check of ``std``; no new fit).

    Fit the same field twice, once on each of two disjoint halves of the training exposures (for example alternate dither groups), make a product
    of each with the same bands, grid and an explicit ``resolution``, and pass both here. For each band the pull is
    ``(image_a - image_b) / sqrt(std_a**2 + std_b**2)`` at the pixels valid in both. If ``std`` is right, its mean is 0 and its rms is 1; real data
    are asked for an rms of 0.9 to 1.2 per band. Biases common to the two halves (the zero level, the held response) cancel, so this checks the noise
    part of ``std`` only.

    Args:
        product_a, product_b: two :class:`SkyProduct` with equal bands, grid (header), unit, and resolution (to 1e-6 arcsec). Pass an explicit
            ``resolution`` to both: the native resolution differs between halves.

    Returns:
        A dict with ``bands``; per band (lists in that order) ``n_pixels``, ``mean`` and ``rms`` of the pull, and ``sigma_ratio`` (the rms of
        ``(image_a - image_b)`` over the rms of ``sqrt(std_a**2 + std_b**2)``: the implied scale of ``std``); and ``pull`` (``[B, H, W]``, NaN where
        not valid in both). Neighbouring pixels are correlated over about the resolution width, so the statistical error of each rms is larger than
        ``1 / sqrt(2 n_pixels)``.

    Raises:
        ValueError: the products are not on the same grid, bands, unit and resolution, or have no pixel valid in both.
    """
    a, b = product_a, product_b
    if a.bands != b.bands:
        raise ValueError(f"the products have different bands: {a.bands} and {b.bands}")
    if a.image.shape != b.image.shape or a.header != b.header:
        raise ValueError("the products are not on the same grid (header)")
    if a.unit != b.unit:
        raise ValueError(f"the products have different units: {a.unit} and {b.unit}")
    if any(
        abs(x - y) > 1e-6
        for x, y in zip(a.resolution_fwhm_arcsec, b.resolution_fwhm_arcsec)
    ):
        raise ValueError(
            "the products have different resolutions "
            f"({a.resolution_fwhm_arcsec} and {b.resolution_fwhm_arcsec}); make both with the same explicit resolution="
        )
    both = a.valid & b.valid
    if not bool(both.any()):
        raise ValueError("no pixel is valid in both products")
    d = a.image.double() - b.image.double()
    s = torch.sqrt(a.std.double() ** 2 + b.std.double() ** 2)
    pull = torch.full_like(d, float("nan"))
    pull[:, both] = d[:, both] / s[:, both]
    out = {
        "bands": a.bands,
        "n_pixels": [],
        "mean": [],
        "rms": [],
        "sigma_ratio": [],
    }
    for k in range(d.shape[0]):
        z = pull[k][both]
        out["n_pixels"].append(int(z.numel()))
        out["mean"].append(float(z.mean()))
        out["rms"].append(float(z.pow(2).mean().sqrt()))
        out["sigma_ratio"].append(
            float(d[k][both].pow(2).mean().sqrt() / s[k][both].pow(2).mean().sqrt())
        )
    out["pull"] = pull.float()
    return out
