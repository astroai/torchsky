"""``SkyFit``, ``fit_sky`` and the private ``_fit_exposures``.

``_fit_exposures`` runs, on a list of private exposure records, the same sequence of steps as the prototype's command line (kernel warm start,
optional coarse registration, registration pre-pass, joint fit, held-out nuisance fit) with the moved code, unchanged. ``fit_sky`` converts
:class:`~torchsky.map.field.types.FieldObservation` objects to records (``_observations.py``) and calls it. This module prepares inputs and
packages outputs; it does no arithmetic of its own.

Module-level state of the moved code (the spectral domain of the sky basis and the rasteriser chunk size are plain module variables there) is
set at the start of every call and restored at the end, so that two fits with different settings cannot disturb each other.

Saving does not reuse ``map/field/artifacts.py``: that module's manifest requires a registered field-model kind (foam / implicit models with
validated constructor arguments) and a ``FieldDomain``, neither of which describes an item sky with its response. The layout is the same in
spirit: ``manifest.json`` (plain JSON) and ``state.pt`` (tensors only, read back with ``weights_only=True``), written atomically.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from . import _items
from ._bandpass import Filter, FilterSet, finish_filter
from ._config import _INTERNAL_SETTINGS, SkyFitConfig, _namespace
from ._exposures import Exposure, build_meta, make_patches
from ._fit import (
    filt_info_of,
    fit,
    heldout_stage,
    make_sky,
    noise_floor_values,
    register_prepass,
    warm_kernel,
)
from ._items import RBFBasis
from ._register import coarse_register
from ._response import FieldOperator
from ._spectral import spectral_setup

if TYPE_CHECKING:
    from ..field.types import FieldObservation
    from ._placeholders import ReferenceCatalogue
    from ._product import SkyProduct
    from ._readers import KernelStamp

_FORMAT = "torchsky.map.exposure_fit/1"


# ----------------------------------------------------------------------------- helpers
def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@contextlib.contextmanager
def _module_state(domain=None, item_chunk=None):
    """Set (and restore on exit) the moved code's module variables: the spectral domain [um] and the rasteriser chunk size."""
    old_dom, old_chunk = list(_items.SPECTRAL_DOMAIN), _items.ITEM_CHUNK
    try:
        if domain is not None:
            _items.set_spectral_domain(*domain)
        if item_chunk is not None and item_chunk > 0:
            _items.ITEM_CHUNK = int(item_chunk)
        yield
    finally:
        _items.set_spectral_domain(*old_dom)
        _items.ITEM_CHUNK = old_chunk


def _prepare(exposures: Sequence[Exposure], dev: torch.device) -> list[Exposure]:
    """Shallow copies of the exposures with the bookkeeping ids set and the pixel tensors on ``dev``; the caller's objects are not changed."""
    if not len(exposures):
        raise ValueError("fit_sky needs at least one exposure")
    out = []
    for i, e in enumerate(exposures):
        if e.eid not in (-1, i):
            raise ValueError(
                f"exposure {e.name!r}: eid {e.eid} does not equal its position {i} in the list (leave eid at -1 or number the list 0..n-1)"
            )
        if e.tile != 0 or e.host is not None:
            raise ValueError(
                f"exposure {e.name!r}: several tiles are not supported by fit_sky in part 1 (tile {e.tile})"
            )
        if e.split not in ("train", "held"):
            raise ValueError(
                f"exposure {e.name!r}: split must be 'train' or 'held', not {e.split!r}"
            )
        c = dataclasses.replace(
            e,
            eid=i,
            cid=i,
            extra=dict(e.extra),
            patches=list(e.patches),
            d=e.d.to(dev),
            sig=e.sig.to(dev),
            valid=e.valid.to(dev),
            X=e.X.to(dev),
            Y=e.Y.to(dev),
        )
        out.append(c)
    if not any(e.split == "train" for e in out):
        raise ValueError("fit_sky needs at least one training exposure")
    return out


def _filters_of(exps: Sequence[Exposure]) -> FilterSet:
    """The FilterSet (filter index -> curve) behind the exposures' integer ``filt``; copies, so the caller's curves are never modified."""
    nf = max(e.filt for e in exps) + 1
    flt: list[Filter | None] = [None] * nf
    for e in exps:
        if e.bandpass is None:
            raise ValueError(
                f"exposure {e.name!r}: no bandpass (the filter curve is needed to build the response)"
            )
        cur = flt[e.filt]
        if cur is None:
            flt[e.filt] = e.bandpass
        elif cur is not e.bandpass and (
            cur.name != e.bandpass.name
            or not np.array_equal(cur.lam_um, e.bandpass.lam_um)
            or not np.array_equal(cur.T, e.bandpass.T)
        ):
            raise ValueError(
                f"exposures with filter index {e.filt} carry different bandpasses ({cur.name!r} and {e.bandpass.name!r})"
            )
    missing = [i for i, f in enumerate(flt) if f is None]
    if missing:
        raise ValueError(
            f"filter indices {missing} have no exposure: number the filters 0..n-1 without gaps"
        )
    return FilterSet([copy.deepcopy(f) for f in flt])


def _names_by_index(exps: Sequence[Exposure], attr: str, prefix: str) -> list[str]:
    n = (
        max(
            getattr(
                e, {"inst_name": "inst", "det_name": "det", "chan_name": "chan"}[attr]
            )
            for e in exps
        )
        + 1
    )
    out = [f"{prefix}{i}" for i in range(n)]
    for e in exps:
        v = getattr(e, attr)
        if v:
            out[
                getattr(
                    e,
                    {"inst_name": "inst", "det_name": "det", "chan_name": "chan"}[attr],
                )
            ] = v
    return out


def _meta_of(exps: Sequence[Exposure]) -> tuple[dict, list[str]]:
    insts = _names_by_index(exps, "inst_name", "instrument")
    dets = _names_by_index(exps, "det_name", "detector")
    chans = _names_by_index(exps, "chan_name", "channel")
    return build_meta(exps, dets, chans, insts), insts


def _sky_half_extent(exps: Sequence[Exposure], internal: dict) -> float:
    """Half side of the sky domain [arcsec] when the caller gives none: the largest absolute header position of a valid pixel, minus the data
    margin plus the sky margin (the prototype took the scoring box instead; ``fit_sky`` passes the box when the observations carry it)."""
    h = 0.0
    for e in exps:
        v = e.valid
        if bool(v.any()):
            h = max(h, float(e.X[v].abs().max()), float(e.Y[v].abs().max()))
    return h - internal["data_margin"] + internal["sky_margin"]


def _json_safe(x: Any) -> Any:
    """JSON-safe copy of x (nested dicts / lists of None, bool, int, finite float, str); a dict stays a dict, a list or tuple becomes a list."""
    if x is None or isinstance(x, (bool, int, str)):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return _json_safe(float(x))
    if isinstance(x, np.ndarray):
        return _json_safe(x.tolist())
    if torch.is_tensor(x):
        return _json_safe(x.detach().cpu().tolist())
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return str(x)


# ----------------------------------------------------------------------------- SkyFit
class SkyFit(nn.Module):
    """A fitted sky and response, from :func:`fit_sky`.

    The module holds the moved ``FieldOperator`` (private): its ``sky`` submodule is the item sky (positions, log widths, angles, log spectral
    coefficients and the spectral basis); the rest of its parameters are the response (kernel network, kernel and throughput codes, noise
    scales) and the per-exposure terms (background, affine, gain, kernel code, stripes), one row per exposure in the order of the list given
    to ``fit_sky``. ``state_dict()`` has the operator's keys under the prefix ``_operator.``.

    Attributes:
        config: the :class:`SkyFitConfig` of the fit.
        report: small JSON-safe dict: step count and timing, the loss log, learned noise floor, warm-start, registration and spectral-setup
            records.
    """

    def __init__(
        self,
        operator: FieldOperator,
        config: SkyFitConfig,
        structure: dict,
        report: dict,
        stage: dict | None,
    ):
        super().__init__()
        self._operator = operator
        self.config = config
        self.report = report
        self._structure = structure  # JSON-safe description needed to rebuild the operator (filters, meta, names, resolved options)
        self._stage = stage  # held-out stage: {"rows": LongTensor, "snaps": {0|1|"all": {term: tensor}}} or None

    @property
    def operator(self) -> FieldOperator:
        """The private operator (used by the tests and the prediction code)."""
        return self._operator

    @property
    def sky(self) -> nn.Module:
        """The item sky: ``n_items`` Gaussian items with position, two widths, angle and ``spectral_k`` coefficients each, and the spectral basis."""
        return self._operator.sky

    @property
    def response(self) -> "_Response":
        """Read-only view of the fitted response: the kernel network and codes, and :meth:`_Response.noise_terms`."""
        return _Response(self)

    @property
    def exposure_names(self) -> list[str]:
        return list(self._structure["exposure_names"])

    @property
    def exposure_terms(self) -> dict:
        """Per-exposure terms, ``{exposure name: {"split", "affine", "gain", "background", "kernel_code"}}``.

        ``affine`` holds the six physical numbers (four matrix entries, two shifts in arcsec), ``gain`` the log-throughput random effect,
        ``background`` the six Legendre coefficients, ``kernel_code`` the per-exposure kernel code. Held-out exposures show the terms fitted
        after the main fit.
        """
        op = self._operator
        out = {}
        with torch.no_grad():
            aff = (op.affine * op.aff_scale).cpu().tolist()
            for i, name in enumerate(self.exposure_names):
                out[name] = {
                    "split": "train" if bool(op.train_mask[i]) else "held",
                    "affine": aff[i],
                    "gain": float(op.u[i]),
                    "background": op.bkg[i].cpu().tolist(),
                    "kernel_code": op.codes.z[i].cpu().tolist(),
                }
        return out

    def _domain(self):
        return self._structure["spectral_domain"]

    @property
    def bands(self) -> tuple:
        """Filter names of the fit, in the order of the fit's filter index (what ``product`` and the response readers take)."""
        return tuple(self._operator.fset.names)

    @property
    def instruments(self) -> tuple:
        """Instrument names of the fit, in the order of the fit's instrument index."""
        return tuple(self._structure["instrument_names"])

    def product(
        self,
        bands,
        pitch_arcsec=None,
        resolution="native",
        region=None,
        *,
        observations,
        method="circulant",
        polish=True,
        samples=8,
        batch_pixels=16000,
        max_minutes=None,
        log=None,
    ) -> "SkyProduct":
        """The sky image of one or several bands with its standard deviation (a :class:`SkyProduct`).

        One solve of the Gauss-Newton normal operator serves all bands. The normal operator is built from the training pixels, so the training
        exposures must be given again (the fit does not keep them, which keeps it small and makes a loaded fit work the same way).

        Args:
            bands: a filter name (see :attr:`bands`), a filter index, ``"all"``, or a sequence of names and indices.
            pitch_arcsec: spacing of the output grid [arcsec]; None takes the sky node pitch of the fit.
            resolution: ``"native"`` (default), or the full width at half maximum [arcsec] of the Gaussian the sky is smoothed with, the same for all
                bands. ``"native"`` is, per band, the median over covered nodes of the supported resolution: the smallest Gaussian width from which
                the noise of a resolution element stops rising faster than twice as fast as the resolution is lost (alpha = 2). Finer than that, the
                number depends on the weak prior, not on the data.
            region: ``(x_min, x_max, y_min, y_max)`` [arcsec] from the box centre (X to the west, Y to the north); None takes the whole sky grid.
            observations: the training observations the fit was made from, matched by card ``TSNAME`` (held-out ones are ignored). They must be the
                same data: the median error and background reference of every exposure are checked.
            method: ``"circulant"`` (cheap Fourier approximation of the standard deviation, about 13 passes over the training pixels) or ``"draws"``
                (posterior draws, about 61 passes each).
            polish: True (default) makes ``image`` the fit plus one Gauss-Newton step (about 61 passes), the estimator ``std`` describes. False
                returns the optimiser's own sky, whose error is not ``std``.
            samples: number of posterior draws (``method="draws"``).
            batch_pixels: pixel budget of one batched render (memory only; 16000 suits an 11 GB GPU).
            max_minutes: time budget from the start of the call; None for none. When the next pass would end after it, the solve stops cleanly and
                ``SkyProduct.truncated`` is True (a cut step is its last iterate, a cut draw is dropped). The set-up (K + 2 passes) is not interrupted.
            log: callable that receives one line of text at a time; None keeps the call silent.

        Raises:
            ValueError: an argument is wrong, a band is unknown, or ``observations`` do not match the fit.
        """
        from ._product import make_product

        return make_product(
            self,
            observations,
            bands,
            pitch_arcsec,
            resolution,
            region,
            method=method,
            polish=polish,
            samples=samples,
            batch_pixels=batch_pixels,
            max_minutes=max_minutes,
            log=log,
        )

    # ------------------------------------------------------------------ saving and loading
    def save(self, path: str | os.PathLike) -> Path:
        """Write the fit to the directory ``path``: ``manifest.json`` (config, structure, report) and ``state.pt`` (tensors only).

        The manifest is plain JSON; ``state.pt`` holds only tensors and is read back with ``weights_only=True``.
        """
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        manifest = {
            "format": _FORMAT,
            "config": self.config.to_dict(),
            "structure": self._structure,
            "report": self.report,
        }
        state = {
            "operator": {
                k: v.detach().cpu() for k, v in self.operator.state_dict().items()
            }
        }
        if self._stage is not None:
            state["stage_rows"] = self._stage["rows"].detach().cpu()
            state["stage_snaps"] = {
                str(k): {t: v.detach().cpu() for t, v in s.items()}
                for k, s in self._stage["snaps"].items()
            }
        tmp_json, tmp_pt = p / "manifest.json.tmp", p / "state.pt.tmp"
        tmp_json.write_text(json.dumps(manifest, allow_nan=False, indent=1))
        torch.save(state, tmp_pt)
        os.replace(tmp_pt, p / "state.pt")
        os.replace(tmp_json, p / "manifest.json")
        return p

    @classmethod
    def load(cls, path: str | os.PathLike, device: str | None = None) -> "SkyFit":
        """Read a fit written by :meth:`save`. ``device`` overrides the device of the saved configuration."""
        p = Path(path)
        manifest = json.loads((p / "manifest.json").read_text())
        if manifest.get("format") != _FORMAT:
            raise ValueError(
                f"{p}: not a {_FORMAT} artifact (format {manifest.get('format')!r})"
            )
        state = torch.load(p / "state.pt", map_location="cpu", weights_only=True)
        config = SkyFitConfig.from_dict(manifest["config"])
        st = manifest["structure"]
        dev = _resolve_device(device or config.device)
        cfg = _namespace(config, st.get("internal"))
        cfg.__dict__.update(st["resolved_options"])
        filters = [
            finish_filter(
                Filter(
                    f["name"],
                    f["instrument"],
                    np.asarray(f["lam_um"], np.float64),
                    np.asarray(f["T"], np.float64),
                ),
                st["Q"],
            )
            for f in st["filters"]
        ]
        fset = FilterSet(filters)
        meta = dict(st["meta"])
        meta["R"] = np.asarray(meta["R"], np.float64)
        with _module_state(item_chunk=cfg.item_chunk):
            spectral_setup(
                fset, cfg, log=lambda s: None
            )  # sets the spectral domain and the curve grid again (deterministic)
            if [float(v) for v in _items.SPECTRAL_DOMAIN] != [
                float(v) for v in st["spectral_domain"]
            ] or cfg.spec_k != st["resolved_options"]["spec_k"]:
                raise ValueError(
                    "the spectral setup of the saved fit could not be reproduced"
                )
            basis = RBFBasis(cfg.spec_k).to(dev)
            sky = _items.ItemSky(
                torch.zeros(cfg.spec_k, st["N"], st["N"]),
                cfg.sky_pitch,
                basis,
                cfg.item_count,
                floor=cfg.item_floor,
                max_width=cfg.item_max_width,
                n_levels=cfg.item_levels,
                n_views=1,
                seed=cfg.seed,
            ).to(dev)
            sky.register_buffer("sig_b", torch.zeros(len(fset)).to(dev))
            op = FieldOperator(cfg, fset, meta, sky, dev).to(dev)
            op.load_state_dict(state["operator"], strict=True)
        stage = None
        if "stage_rows" in state:
            snaps = {
                (int(k) if k in ("0", "1") else k): {t: v.to(dev) for t, v in s.items()}
                for k, s in state["stage_snaps"].items()
            }
            stage = {"rows": state["stage_rows"].to(dev), "snaps": snaps}
        return cls(op, config, st, manifest["report"], stage)


class _Response:
    """Read-only view of the response of a :class:`SkyFit` (it shares the fit's tensors; it has no state of its own).

    Attributes:
        kernel_network: the kernel network (``nn.Module``); ``codes``: the kernel and throughput codes (``nn.Module``).
    """

    def __init__(self, fit: SkyFit):
        self._fit = fit
        self.kernel_network = fit._operator.net
        self.codes = fit._operator.codes

    def kernel(
        self,
        instrument: str,
        band: str,
        exposure: str | None = None,
        *,
        pitch_arcsec: float | None = None,
        half_size_arcsec: float | None = None,
    ) -> "KernelStamp":
        """The fitted kernel (point-spread function) of an instrument and band, as a :class:`KernelStamp` (``stamp [n, n]``, ``pitch_arcsec``).

        The kernel is the bandpass-weighted mean of the kernels at the filter's wavelength nodes, each normalised to unit integral inside the
        normalisation radius ``config.kernel_norm_radius`` (a smooth window, 1 inside 0.85 of the radius and 0 beyond it; no window when the radius
        is 0), on the model's own quadrature. So the stamp summed over that window times the pixel area is 1 to the quadrature accuracy. The unit is
        1 / arcsec^2. It is in the detector-aligned frame (columns along X, rows along Y of the detector), with the blur-gauge dilation applied, as
        the renderer uses it.

        Args:
            instrument: instrument name (see ``SkyFit.instruments``). band: filter name (``SkyFit.bands``).
            exposure: name of an exposure of that instrument and band to include its kernel code (a training or held-out one); None gives the
                mean kernel of the band (code at the pivot wavelength, no per-exposure term).
            pitch_arcsec: grid spacing [arcsec]; None takes the render pitch of the instrument's channel.
            half_size_arcsec: half side of the stamp [arcsec]; None takes the normalisation radius, or the kernel extent when that is smaller or the radius is 0.
        """
        from ._readers import kernel_stamp

        return kernel_stamp(
            self._fit,
            instrument,
            band,
            exposure,
            pitch_arcsec=pitch_arcsec,
            half_size_arcsec=half_size_arcsec,
        )

    def throughput(self, instrument: str, band: str):
        """Throughput of an instrument and band: ``(wavelengths, curve)``, both float32 tensors of shape ``[Q]`` (``Q`` nodes of the filter, in
        micrometres).

        ``curve`` is the published transmission at the node times ``exp(gamma)``, with ``gamma`` the learned log-throughput of an exposure with
        zero gain random effect, after the gauge: the reference instrument's exposures have a transmission-weighted mean of the log-throughput
        (including their gains) equal to zero at every node, so its curve equals the published one up to the small gains. Dimensionless.
        """
        from ._readers import throughput_curve

        return throughput_curve(self._fit, instrument, band)

    def noise_terms(self) -> dict:
        """Learned noise terms (dimensionless): ``{"floor": {channel: fractional floor}, "scale": {"detector/filter": factor on the pipeline error}}``.

        The noise variance of a pixel is ``(pipeline error * scale)**2 + (floor * model value)**2``."""
        import torch.nn.functional as F

        op, st = self._fit._operator, self._fit._structure
        with torch.no_grad():
            floor = F.softplus(op.floor_raw).cpu().tolist()
            h = (3 * torch.tanh(op.hs / 3)).exp().cpu()
        chans, dets, filts = (
            st["channel_names"],
            st["detector_names"],
            [f["name"] for f in st["filters"]],
        )
        return {
            "floor": {chans[c]: float(floor[c]) for c in range(len(chans))},
            "scale": {
                f"{dets[d]}/{filts[f]}": float(h[d, f])
                for d in range(len(dets))
                for f in range(len(filts))
            },
        }


# ----------------------------------------------------------------------------- fitting
def _fit_exposures(
    exposures: Sequence[Exposure],
    config: SkyFitConfig,
    *,
    internal: Mapping | None = None,
    sky_half_extent: float | None = None,
    box_centre: tuple[float, float] | None = None,
    log: Callable[[str], None] | None = None,
) -> SkyFit:
    """Fit an item sky and the response to private exposure records (what ``fit_sky`` calls after converting its observations).

    Args:
        exposures: private ``Exposure`` records of one sky patch (``split`` is ``"train"`` or ``"held"``; each carries its filter curve). The
            caller's objects are not modified.
        config: the public configuration.
        internal: overrides of single entries of the private settings (``_config._INTERNAL_SETTINGS``); for tests and the bit-identity check.
        sky_half_extent: half side of the square sky domain [arcsec]; None derives it from the data.
        box_centre: (RA, Dec) [deg] of the tangent point of the sky box, kept for the header of ``SkyFit.product``; None when unknown.
        log: receives one line of text at a time; None keeps the fit silent.
    """
    log = log or (lambda s: None)
    cfg = _namespace(config, internal)
    internal_used = {
        k: getattr(cfg, k) for k in _INTERNAL_SETTINGS
    }  # every private setting in force: a saved fit does not depend on later defaults
    dev = _resolve_device(cfg.device)
    exps = _prepare(exposures, dev)
    fset = _filters_of(exps)
    meta, insts = _meta_of(exps)
    half = (
        float(sky_half_extent)
        if sky_half_extent is not None
        else _sky_half_extent(exps, vars(cfg))
    )
    t_start = time.perf_counter()
    with _module_state(item_chunk=cfg.item_chunk):
        torch.manual_seed(cfg.seed)
        spec_rec = spectral_setup(
            fset, cfg, log=log
        )  # spectral domain, curve grid and K from the filters
        sky, N = make_sky(exps, fset, cfg, half, dev)
        model = FieldOperator(cfg, fset, meta, sky, dev).to(dev)
        for e in exps:
            if not e.patches:  # prebuilt patches are kept (the prototype's synthetic fields set theirs)
                pix = float(abs(np.linalg.det(np.asarray(e.extra["J"]))) ** 0.5)
                make_patches(e, max(16, int(round(cfg.patch_arcsec / pix))))
        winfo = warm_kernel(model, cfg, filt_info_of(exps, fset, insts), log=log)
        coarse = coarse_register(model, exps, cfg, log=log, ts=None, inst_names=insts)
        reg = register_prepass(model, exps, cfg, log=log)
        if coarse:
            reg = dict(reg, coarse=coarse)
        rec = fit(model, exps, cfg, log=log)
        held = [e.cid for e in exps if e.split == "held"]
        stage = heldout_stage(model, exps, held, cfg, log=log)
        domain = [float(v) for v in _items.SPECTRAL_DOMAIN]
    stage_state = None
    if stage["rows"] is not None:
        stage_state = {"rows": stage["rows"], "snaps": stage["snaps"]}
    with torch.no_grad():
        floors = noise_floor_values(model)
    report = _json_safe(
        {
            "n_steps": rec.get("n_steps"),
            "seconds": rec.get("seconds"),
            "s_per_step": rec.get("s_per_step"),
            "loss_log": rec.get("steps"),
            "dilation_max": rec.get("dilation_max"),
            "dilation_final": rec.get("dilation_final"),
            "noise_floor_final": floors,
            "noise_at_bound_final": rec.get("noise_at_bound_final"),
            "spectral_setup": spec_rec,
            "warm_kernel": winfo,
            "registration_prepass": reg,
            "tau": model.tau.table(),
            "n_train": sum(e.split == "train" for e in exps),
            "n_held": len(held),
            "total_seconds": time.perf_counter() - t_start,
            **(
                {"kernel_centroid_gauge": rec["kernel_centroid_gauge"]}
                if "kernel_centroid_gauge" in rec
                else {}
            ),  # only when the kernel-centroid gauge is on (internal setting)
        }
    )
    chans = _names_by_index(exps, "chan_name", "channel")
    structure = _json_safe(
        {
            "filters": [
                {
                    "name": f.name,
                    "instrument": f.instrument,
                    "lam_um": f.lam_um,
                    "T": f.T,
                }
                for f in fset.filters
            ],
            "Q": fset.Q,
            "meta": {k: v for k, v in meta.items()},
            "N": N,
            "spectral_domain": domain,
            "exposure_names": [e.name for e in exps],
            "instrument_names": insts,
            "channel_names": chans,
            "detector_names": list(meta["det_names"]),
            "internal": internal_used,
            "box_centre_deg": None
            if box_centre is None
            else [float(v) for v in box_centre],
            "resolved_options": {
                "spec_k": cfg.spec_k,
                "gauge_ref_inst": cfg.gauge_ref_inst,
                "prior_pixels": cfg.prior_pixels,
            },
        }
    )
    return SkyFit(model, config, structure, report, stage_state)


def fit_sky(
    observations: "Sequence[FieldObservation]",
    config: SkyFitConfig,
    *,
    catalogue: "ReferenceCatalogue | None" = None,
    init: "SkyFit | None" = None,
    log: Callable[[str], None] | None = None,
) -> SkyFit:
    """Fit an item sky and the instrument response to exposures of one sky patch.

    Args:
        observations: one :class:`~torchsky.map.field.types.FieldObservation` per exposure and chip, as :func:`load_exposures` returns them
            (values in MJy/sr, ``err``, ``mask``, a header with a tangent-plane solution, the filter curve, and the bookkeeping cards listed in
            the guide). The caller's objects are not modified. An observation whose header card ``TSSPLIT`` is ``"held"`` is not used by the
            fit; its own terms are fitted afterwards so that :func:`predict_exposure` can score it.
        config: a :class:`SkyFitConfig`.
        catalogue: reserved for a reference catalogue (a later piece of the move); must be None.
        init: reserved for warm-starting from an earlier fit (a later piece); must be None.
        log: callable that receives one line of text at a time; None keeps the fit silent.

    Returns:
        The fitted :class:`SkyFit`.

    Raises:
        NotImplementedError: ``catalogue`` or ``init`` is not None.
        ValueError: an observation cannot be converted (missing bookkeeping card, filter curve, unit, or mask), or the list is unusable.
    """
    if catalogue is not None:
        raise NotImplementedError(
            "a reference catalogue is not used yet (a later piece of the move); pass catalogue=None"
        )
    if init is not None:
        raise NotImplementedError(
            "warm start from an earlier fit (init=) is not implemented yet; pass init=None"
        )
    from ._observations import observations_to_exposures

    exposures, box_half = observations_to_exposures(
        observations, n_lambda=_INTERNAL_SETTINGS["n_lambda"]
    )
    half = (
        None
        if box_half is None
        else box_half
        - _INTERNAL_SETTINGS["data_margin"]
        + _INTERNAL_SETTINGS["sky_margin"]
    )
    h0 = observations[0].header
    centre = (
        (float(h0["TSBOXRA"]), float(h0["TSBOXDE"]))
        if "TSBOXRA" in h0 and "TSBOXDE" in h0
        else None
    )
    return _fit_exposures(
        exposures, config, sky_half_extent=half, box_centre=centre, log=log
    )
