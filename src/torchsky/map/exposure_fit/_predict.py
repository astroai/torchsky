"""``predict_exposure``: the prototype's cross-fitted held-out prediction of one exposure (moved code, unchanged)."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, NamedTuple

import torch

from ._exposures import Exposure
from ._fit import make_heldout_predictor
from ._skyfit import SkyFit, _json_safe, _module_state

if TYPE_CHECKING:
    from ..field.types import FieldObservation


class ExposurePrediction(NamedTuple):
    """Held-out prediction of one exposure (all arrays over the exposure's whole cut, shape ``(ny, nx)``).

    Attributes:
        model: predicted pixel values in the units of the data (NaN where the model could not be evaluated).
        sigma: the learned noise standard deviation at every pixel (pipeline error scaled by the learned noise scale, plus the learned
            fractional floor times the model).
        terms: the per-exposure terms fitted on the exposure: ``affine`` (the six physical numbers: four matrix, two shift in arcsec),
            ``gain`` (log-throughput random effect), ``background`` (six quadratic-background coefficients), ``kernel_code``.
        chi2: score record against the pipeline error (``chi2``, ``n``, ``chi2_bright``, ``n_bright``, ``chi2_model_noise``, and with
            cross-fitting ``all_pixel_old``).
    """

    model: torch.Tensor
    sigma: torch.Tensor
    terms: dict
    chi2: dict


def predict_exposure(
    fit: SkyFit, observation: "FieldObservation", *, cross_fit: bool = True
) -> ExposurePrediction:
    """Predict a held-out exposure of ``fit``.

    ``observation`` is either one that carried header card ``TSSPLIT = "held"`` in the list given to :func:`fit_sky` (matched by ``TSNAME``; its
    terms were fitted by ``fit_sky``), or any other exposure of an instrument and band the fit knows, whose name (``TSNAME``) is not in the fit.
    For a new exposure the per-exposure terms (background, affine, gain, kernel code, stripes) are fitted here, with the sky, the response and the
    random-effect scales held at the fit's values, by the same procedure ``fit_sky`` uses for held-out exposures (the fit itself is not changed; the
    global random state is restored). With ``cross_fit=True`` (the
    prototype's default) every pixel is predicted by terms that did not see it: the terms were fitted twice, each time on one half of the
    interleaved 4 x 4 pixel blocks, and each half is predicted by the fit on the other. ``cross_fit=False`` uses the all-pixel fit (optimistic).

    Raises:
        ValueError: the observation was a training exposure of the fit, its instrument, detector or band is not in the fit, or it lacks the
            bookkeeping cards (see :func:`fit_sky`).
    """
    from ._observations import observation_to_record

    names = fit.exposure_names
    name = str(observation.header.get("TSNAME", ""))
    if name not in names:
        return _predict_unseen(fit, observation, name, cross_fit)
    meta, k = fit.operator.meta, names.index(name)
    rec = observation_to_record(
        observation,
        filt=meta["filt"][k],
        inst=meta["inst"][k],
        chan=meta["chan"][k],
        det=meta["det"][k],
        bandpass=fit.operator.fset.filters[meta["filt"][k]],
        name=name,
    )
    return _predict_record(fit, rec, cross_fit=cross_fit)


def _predict_record(
    fit: SkyFit, exposure: Exposure, *, cross_fit: bool = True
) -> ExposurePrediction:
    """The prediction for a private record (matched by name to a held-out exposure of the fit)."""
    if fit._stage is None:
        raise ValueError("the fit has no held-out exposures")
    names = fit.exposure_names
    if exposure.name not in names:
        raise ValueError(
            f"exposure {exposure.name!r} was not in the list given to fit_sky"
        )
    eid = names.index(exposure.name)
    if not bool((fit._stage["rows"] == eid).any()):
        raise ValueError(
            f"exposure {exposure.name!r} was a training exposure of the fit; predict_exposure is for held-out exposures"
        )
    model = fit.operator
    dev = model.dev
    e = dataclasses.replace(
        exposure,
        eid=eid,
        cid=eid,
        extra=dict(exposure.extra),
        d=exposure.d.to(dev),
        sig=exposure.sig.to(dev),
        valid=exposure.valid.to(dev),
        X=exposure.X.to(dev),
        Y=exposure.Y.to(dev),
    )
    snaps = fit._stage["snaps"] if cross_fit else {"all": fit._stage["snaps"]["all"]}
    cfg = model.cfg
    with (
        _module_state(domain=fit._domain(), item_chunk=cfg.item_chunk),
        torch.no_grad(),
    ):
        P = {k: v.detach() for k, v in model.prepare().items()}
        pred = make_heldout_predictor(model, cfg, P, fit._stage["rows"], snaps)
        m, pv, S, rec = pred(e)
        sigma = model.sigma(model.et(e.eid), S, e.sig)
        terms = {
            "affine": (model.affine[eid] * model.aff_scale).cpu().tolist(),
            "gain": float(model.u[eid]),
            "background": model.bkg[eid].cpu().tolist(),
            "kernel_code": model.codes.z[eid].cpu().tolist(),
        }
    return ExposurePrediction(model=m, sigma=sigma, terms=terms, chi2=_json_safe(rec))


def _predict_unseen(
    fit: SkyFit, observation: "FieldObservation", name: str, cross_fit: bool
) -> ExposurePrediction:
    """Prediction of an exposure that was not in the fit's list: one row is added to a copy of the operator, its terms are fitted, and the old
    route predicts it. The fit's own tensors are not touched; the sky and the response are shared, frozen."""
    import numpy as np

    from ...instrument import instrument_description
    from ._exposures import build_meta, make_patches
    from ._fit import heldout_stage
    from ._observations import observation_to_record
    from ._response import FieldOperator

    op, st = fit.operator, fit._structure
    if not name:
        raise ValueError("the observation has no TSNAME card")
    desc = instrument_description(str(observation.header.get("TSINST", "")))
    det = str(observation.header.get("DETECTOR", desc.name))
    chan = desc.channel_of(det if desc.name == "NIRCAM" else "")
    band = str(observation.band_name)
    for what, val, known in (
        ("instrument", desc.name, list(st["instrument_names"])),
        ("detector", det, list(st["detector_names"])),
        ("channel", chan, list(st["channel_names"])),
        ("band", band, list(op.fset.names)),
    ):
        if val not in known:
            raise ValueError(
                f"the {what} {val!r} of observation {name!r} is not in the fit (it has {known}); a new exposure needs a response the fit learned"
            )
    centre = st.get("box_centre_deg")
    if centre is not None and (
        abs(float(observation.header["TSBOXRA"]) - centre[0]) > 1e-9
        or abs(float(observation.header["TSBOXDE"]) - centre[1]) > 1e-9
    ):
        raise ValueError(
            f"observation {name!r}: sky box centre differs from the fit's {tuple(centre)}"
        )
    ifilt, iinst = (
        list(op.fset.names).index(band),
        list(st["instrument_names"]).index(desc.name),
    )
    rec = observation_to_record(
        observation,
        filt=ifilt,
        inst=iinst,
        chan=list(st["channel_names"]).index(chan),
        det=list(st["detector_names"]).index(det),
        bandpass=op.fset.filters[ifilt],
        name=name,
    )
    E = op.E
    new = build_meta(
        [dataclasses.replace(rec, split="held")],
        list(st["detector_names"]),
        list(st["channel_names"]),
        list(st["instrument_names"]),
    )
    meta = dict(op.meta)
    for k in (
        "diameter_m_e",
        "filt",
        "inst",
        "chan",
        "det",
        "train",
        "sigbar",
        "det_shape",
        "shift0",
        "stripes",
        "bkg0",
        "cut",
        "cuts",
    ):
        meta[k] = list(op.meta[k]) + list(new[k])
    meta["R"] = np.concatenate([np.asarray(op.meta["R"]), new["R"]], 0)
    dev, cfg = op.dev, op.cfg
    stage = None
    with (
        _module_state(domain=fit._domain(), item_chunk=cfg.item_chunk),
        torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []),
    ):
        torch.manual_seed(int(fit.config.seed))
        op2 = FieldOperator(cfg, op.fset, meta, op.sky, dev).to(dev)
        old = op.state_dict()
        new_state = op2.state_dict()
        for k, v in old.items():
            tgt = new_state[k]
            if tgt.shape == v.shape:
                new_state[k] = v.detach().clone()
            elif (
                tgt.shape[0] == E + 1
                and v.shape[0] == E
                and tgt.shape[1:] == v.shape[1:]
            ):
                tgt = tgt.clone()
                tgt[:E] = v.detach()
                new_state[k] = tgt
            else:
                raise ValueError(
                    f"cannot extend {k}: {tuple(v.shape)} to {tuple(tgt.shape)}"
                )
        op2.load_state_dict(new_state, strict=True)
        r2 = dataclasses.replace(
            rec,
            eid=E,
            cid=0,
            split="held",
            extra=dict(rec.extra),
            patches=[],
            d=rec.d.to(dev),
            sig=rec.sig.to(dev),
            valid=rec.valid.to(dev),
            X=rec.X.to(dev),
            Y=rec.Y.to(dev),
        )
        pix = float(abs(np.linalg.det(np.asarray(r2.extra["J"]))) ** 0.5)
        make_patches(r2, max(16, int(round(cfg.patch_arcsec / pix))))
        stage = heldout_stage(op2, [r2], [0], cfg, log=lambda s: None)
        tmp = SkyFit(
            op2,
            fit.config,
            dict(st, exposure_names=list(st["exposure_names"]) + [name]),
            fit.report,
            {"rows": stage["rows"], "snaps": stage["snaps"]},
        )
        return _predict_record(tmp, rec, cross_fit=cross_fit)
