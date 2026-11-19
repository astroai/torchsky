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

    ``observation`` is one that carried header card ``TSSPLIT = "held"`` in the list given to :func:`fit_sky`; it is matched by ``TSNAME``.
    The per-exposure terms of a held-out exposure were fitted by ``fit_sky`` on the fitted sky and response. With ``cross_fit=True`` (the
    prototype's default) every pixel is predicted by terms that did not see it: the terms were fitted twice, each time on one half of the
    interleaved 4 x 4 pixel blocks, and each half is predicted by the fit on the other. ``cross_fit=False`` uses the all-pixel fit (optimistic).
    A new exposure that was not in the list has no fitted terms yet and raises ``ValueError``.
    """
    from ._observations import observation_to_record

    names = fit.exposure_names
    name = str(observation.header.get("TSNAME", ""))
    if name not in names:
        raise ValueError(
            f"observation {name!r} (card TSNAME) was not in the list given to fit_sky (a new exposure has no fitted terms yet)"
        )
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
