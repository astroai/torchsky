"""Crossmatch Gaia DR3, 2MASS and AllWISE in the field of the open cluster M67.

Science question: which Gaia sources have a near-infrared (2MASS) and a
mid-infrared (AllWISE) counterpart, and what do the matched colours say about
the stars? A 0.25 degree cone around M67 (RA 132.825, Dec +11.8) is fetched
from the Gaia archive and IRSA TAP services (about 2000 / 1300 / 2800 rows,
cached after the first run). Gaia positions are propagated to the 2MASS
epoch (J2000) with their proper motions, then matched with
``torchsky.catalogs.crossmatch_sky`` using a fixed radius (``best1`` and
``mutual``), with per-source catalogue errors (``skyerr``), and with
``crossmatch_many(mode="group")`` for the three-way group. A brute-force
O(N*M) great-circle search on a subset must reproduce the indexed ``best1``
result exactly. The figure shows the Gaia-2MASS colour-magnitude diagram, where
the M67 main sequence and turn-off appear, and the J-K versus K-W1 colour plane
of the three-way matches.

Data: Gaia DR3 (ESA/Gaia/DPAC, CC BY-SA 3.0 IGO), 2MASS Point Source Catalog
(UMass/IPAC-Caltech, NASA/NSF) and AllWISE Source Catalog (NASA/IPAC), all via
public TAP services. Runtime about 2 s on CPU once the cones are cached.

    pixi run -e test -- python examples/crossmatch_gaia_2mass_allwise.py [--device cuda] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from _sample_data import skip_if_unavailable
from _sample_tap import fetch_cone

from torchsky.catalogs import crossmatch_many, crossmatch_sky
from torchsky.sphere import great_circle_distance

CENTER = (132.825, 11.8)
RADIUS_DEG = 0.25
EPOCH_2MASS = 2000.0
MATCH_ARCSEC = 2.0
SIGMA_FACTOR = 3.0
N_BRUTE = 300


def column(table, name: str, dtype=torch.float64) -> torch.Tensor:
    return torch.tensor(table[name].to_numpy(zero_copy_only=False).copy(), dtype=dtype)


def propagate(ra, dec, pmra, pmdec, years):
    """Linear proper-motion shift (mas/yr inputs, degrees out); NaN motions are zero."""
    pmra, pmdec = torch.nan_to_num(pmra), torch.nan_to_num(pmdec)
    dec_new = dec + pmdec * years / 3.6e6
    ra_new = ra + pmra * years / 3.6e6 / dec.deg2rad().cos()
    return ra_new % 360.0, dec_new


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    dev = args.device

    cols = {
        "gaia_dr3": (
            "source_id", "ra", "dec", "ra_error", "dec_error", "parallax",
            "pmra", "pmdec", "phot_g_mean_mag", "bp_rp",
        ),
        "twomass": ("designation", "ra", "dec", "err_maj", "j_m", "h_m", "k_m"),
        "allwise": ("designation", "ra", "dec", "sigra", "sigdec", "w1mpro", "w2mpro"),
    }
    with skip_if_unavailable():
        gaia = fetch_cone("gaia_dr3", *CENTER, RADIUS_DEG, columns=cols["gaia_dr3"])
        tmass = fetch_cone("twomass", *CENTER, RADIUS_DEG, columns=cols["twomass"])
        wise = fetch_cone("allwise", *CENTER, RADIUS_DEG, columns=cols["allwise"])
    print(
        f"cone r={RADIUS_DEG} deg at {CENTER}: Gaia {gaia.num_rows}, "
        f"2MASS {tmass.num_rows}, AllWISE {wise.num_rows} rows"
    )

    g_ra, g_dec = propagate(
        column(gaia, "ra"), column(gaia, "dec"),
        column(gaia, "pmra"), column(gaia, "pmdec"),
        EPOCH_2MASS - 2016.0,
    )
    t_ra, t_dec = column(tmass, "ra"), column(tmass, "dec")
    w_ra, w_dec = column(wise, "ra"), column(wise, "dec")
    g_ra, g_dec, t_ra, t_dec, w_ra, w_dec = (
        x.to(dev) for x in (g_ra, g_dec, t_ra, t_dec, w_ra, w_dec)
    )

    def timed(fn):
        t0 = time.perf_counter()
        out = fn()
        return out, 1e3 * (time.perf_counter() - t0)

    # Fixed-radius matches Gaia -> 2MASS.
    best, ms_best = timed(
        lambda: crossmatch_sky(g_ra, g_dec, t_ra, t_dec, radius_arcsec=MATCH_ARCSEC, find="best1", device=dev)
    )
    mutual, ms_mut = timed(
        lambda: crossmatch_sky(g_ra, g_dec, t_ra, t_dec, radius_arcsec=MATCH_ARCSEC, find="mutual", device=dev)
    )
    print(
        f"Gaia x 2MASS within {MATCH_ARCSEC} arcsec: best1 {best.left_index.numel()} pairs "
        f"({ms_best:.0f} ms), mutual {mutual.left_index.numel()} pairs ({ms_mut:.0f} ms), "
        f"median separation {float(best.separation_arcsec.median()):.3f} arcsec"
    )

    # Per-source catalogue errors: pair kept when sep <= k (err_gaia + err_2mass).
    g_err = (torch.hypot(column(gaia, "ra_error"), column(gaia, "dec_error")) / 1e3).to(dev)
    t_err = torch.nan_to_num(column(tmass, "err_maj")).to(dev)
    w_err = torch.hypot(
        torch.nan_to_num(column(wise, "sigra")), torch.nan_to_num(column(wise, "sigdec"))
    ).to(dev)
    skyerr, ms_err = timed(
        lambda: crossmatch_sky(
            g_ra, g_dec, t_ra, t_dec, find="best1", device=dev,
            left_err_arcsec=SIGMA_FACTOR * g_err, right_err_arcsec=SIGMA_FACTOR * t_err,
        )
    )
    print(
        f"skyerr ({SIGMA_FACTOR:.0f} x catalogue errors): {skyerr.left_index.numel()} pairs "
        f"({ms_err:.0f} ms), {int(torch.isin(skyerr.left_index, best.left_index).sum())} shared with the fixed radius"
    )

    skyerr_w, ms_w = timed(
        lambda: crossmatch_sky(
            g_ra, g_dec, w_ra, w_dec, find="best1", device=dev,
            left_err_arcsec=SIGMA_FACTOR * g_err, right_err_arcsec=SIGMA_FACTOR * w_err,
        )
    )
    print(
        f"Gaia x AllWISE skyerr: {skyerr_w.left_index.numel()} pairs ({ms_w:.0f} ms), "
        f"median separation {float(skyerr_w.separation_arcsec.median()):.3f} arcsec"
    )

    # Brute-force O(N*M) check of best1 on a subset of Gaia rows.
    n_sub = min(N_BRUTE, g_ra.numel())
    sep = great_circle_distance(
        g_ra[:n_sub, None], g_dec[:n_sub, None], t_ra[None, :], t_dec[None, :]
    ) * 3600.0
    near_sep, near_idx = sep.min(dim=1)
    keep = near_sep <= MATCH_ARCSEC
    brute = {int(i): int(j) for i, j in zip(torch.nonzero(keep).flatten(), near_idx[keep])}
    fast = {
        int(i): int(j)
        for i, j in zip(best.left_index.cpu(), best.right_index.cpu())
        if int(i) < n_sub
    }
    print(f"brute-force check on {n_sub} Gaia rows: {len(brute)} matches, identical = {brute == fast}")
    assert brute == fast

    # Three-way friends-of-friends group.
    many, ms_many = timed(
        lambda: crossmatch_many(
            [(g_ra, g_dec), (t_ra, t_dec), (w_ra, w_dec)], 2.5, mode="group"
        )
    )
    counts = torch.bincount(many.n_members.cpu(), minlength=4)
    print(
        f"crossmatch_many group (2.5 arcsec, {ms_many:.0f} ms): {many.index.shape[0]} groups; "
        f"groups with 1/2/3 catalogues: {int(counts[1])}/{int(counts[2])}/{int(counts[3])}"
    )

    # Figure: CMD from mutual matches and the infrared colour plane of triples.
    g_mag = column(gaia, "phot_g_mean_mag")
    k_mag, j_mag = column(tmass, "k_m"), column(tmass, "j_m")
    w1 = column(wise, "w1mpro")
    gi, ti = mutual.left_index.cpu(), mutual.right_index.cpu()
    idx = many.index.cpu()
    triple = (idx >= 0).all(dim=1)
    gi3, ti3, wi3 = (idx[triple, c] for c in range(3))
    plx = column(gaia, "parallax")
    member = (plx > 0.9) & (plx < 1.3)  # M67 parallax is about 1.15 mas

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    ax = axes[0]
    ax.scatter(g_mag[gi] - k_mag[ti], g_mag[gi], s=3, c="0.65", label="Gaia x 2MASS")
    m = member[gi]
    ax.scatter((g_mag[gi] - k_mag[ti])[m], g_mag[gi][m], s=4, c="C3", label="parallax 0.9-1.3 mas")
    ax.invert_yaxis()
    ax.set(xlabel="G - Ks [mag]", ylabel="G [mag]", title="Colour-magnitude diagram")
    ax.legend(markerscale=3, fontsize=8)
    ax = axes[1]
    ax.scatter(j_mag[ti3] - k_mag[ti3], k_mag[ti3] - w1[wi3], s=4, c="C0")
    ax.set(xlabel="J - Ks [mag]", ylabel="Ks - W1 [mag]", xlim=(-0.2, 1.4), ylim=(-0.4, 0.8),
           title=f"Gaia x 2MASS x AllWISE ({int(triple.sum())} stars)")
    out = args.out or Path(tempfile.mkdtemp(prefix="torchsky-gaia-2mass-allwise-"))
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "crossmatch_gaia_2mass_allwise.png", dpi=110)
    print(f"figure: {out / 'crossmatch_gaia_2mass_allwise.png'}; device={dev}")


if __name__ == "__main__":
    main()
