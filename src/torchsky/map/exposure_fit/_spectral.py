"""Spectral domain and filter-curve grid of the fit (moved unchanged from the prototype's instruments.py, ``spectral_setup``).

Instrument facts (diameters, detectors, data-quality policy) are not kept here: they live in :mod:`torchsky.instrument`.
"""

from __future__ import annotations

import math

import numpy as np

# reference spectral domain of the sky and filter-curve input grid (the constants used before 18 Nov 2026; sky.LAM_LO / LAM_HI, bandpass.CURVE_*)
REF_LAM = (0.70, 5.30)
REF_CURVE = (0.55, 5.6, 48)
NODE_MARGIN = (
    0.02  # the derived spectral domain covers every quadrature node x (1 +- 2 %)
)


# ----------------------------------------------------------------------------- spectral domain from the loaded bandpasses (audit finding 4)
def spectral_setup(fset, cfg, log=print):
    """Spectral domain of the sky basis and of the kernel / throughput wavelength coordinate, filter-curve input grid and number of spectral
    coefficients K, from the loaded filters.

    --spectral-domain union (default): domain = hull(REF_LAM, [min node / (1 + m), max node x (1 + m)]) over every filter's quadrature nodes,
    m = 2 %; curve grid = hull(REF_CURVE range, union of the curves' supports T >= 1e-3 max), its point count grown at the reference log density
    (48 points over ln(5.6 / 0.55)). With every node inside 0.70-5.30 um and every support inside 0.55-5.6 um (all NIRCam wide filters of the
    SMACS field; the synthetic fields) both are exactly the old constants. 'fixed': the old constants whatever is loaded.
    --spec-k-rule fixed (default): K = --spec-k. 'span': K = 1 + round((--spec-k - 1) ln(hi / lo) / ln(5.30 / 0.70)), the same bump spacing in
    ln(lambda) as 12 bumps over the reference domain (0.184 in ln lambda, 0.080 dex; bump width = spacing), so K = 12 for the reference domain.
    Sets sky.SPECTRAL_DOMAIN, the filters' curve vectors, and cfg.spec_k; returns a record for the log and metrics."""
    from . import _items as _sky

    mode = getattr(cfg, "spectral_domain", "union")
    nodes = np.concatenate([np.asarray(f.nodes, np.float64) for f in fset.filters])
    sup = []
    for f in fset.filters:
        keep = f.T >= 1e-3 * f.T.max()
        sup.append((float(f.lam_um[keep].min()), float(f.lam_um[keep].max())))
    s_lo, s_hi = min(s[0] for s in sup), max(s[1] for s in sup)
    if mode == "fixed":
        lo, hi = REF_LAM
        c_lo, c_hi, c_n = REF_CURVE
    else:
        lo = min(REF_LAM[0], float(nodes.min()) / (1 + NODE_MARGIN))
        hi = max(REF_LAM[1], float(nodes.max()) * (1 + NODE_MARGIN))
        c_lo, c_hi = min(REF_CURVE[0], s_lo), max(REF_CURVE[1], s_hi)
        if (c_lo, c_hi) == REF_CURVE[:2]:
            c_n = REF_CURVE[2]
        else:
            c_n = 1 + int(
                math.ceil(
                    (REF_CURVE[2] - 1)
                    * math.log(c_hi / c_lo)
                    / math.log(REF_CURVE[1] / REF_CURVE[0])
                )
            )
    k_req = int(cfg.spec_k)
    rule = getattr(cfg, "spec_k_rule", "fixed")
    if rule == "span":
        k = 1 + int(
            round((k_req - 1) * math.log(hi / lo) / math.log(REF_LAM[1] / REF_LAM[0]))
        )
    else:
        k = k_req
    _sky.set_spectral_domain(lo, hi)
    changed_grid = fset.set_curve_grid(c_lo, c_hi, c_n)
    cfg.spec_k = k
    rec = {
        "mode": mode,
        "domain_um": [lo, hi],
        "reference_domain_um": list(REF_LAM),
        "node_range_um": [float(nodes.min()), float(nodes.max())],
        "curve_grid_um": [c_lo, c_hi, c_n],
        "curve_support_um": [s_lo, s_hi],
        "curve_grid_changed": changed_grid,
        "spec_k_rule": rule,
        "spec_k_requested": k_req,
        "spec_k": k,
        "ln_spacing": math.log(hi / lo) / max(k - 1, 1),
        "outside_old_domain": [
            f.name
            for f in fset.filters
            if f.nodes.min() < REF_LAM[0] or f.nodes.max() > REF_LAM[1]
        ],
    }
    log(
        f"spectral domain {lo:.4f}-{hi:.4f} um ({mode}; nodes {nodes.min():.4f}-{nodes.max():.4f} um), K = {k} ({rule}, bump spacing "
        f"{rec['ln_spacing']:.3f} in ln lambda), filter-curve grid {c_lo:.4f}-{c_hi:.4f} um x {c_n}"
        + (
            f"; filters with nodes outside the old 0.70-5.30 um domain: {rec['outside_old_domain']}"
            if rec["outside_old_domain"]
            else ""
        )
    )
    return rec
