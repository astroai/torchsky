"""Gaia DR3 proper motions of the Pleiades in the Galactic frame.

Science question: which stars in a 1 degree cone around the Pleiades move with
the cluster, and how does the cluster move in Galactic coordinates? The 2745
Gaia DR3 sources with G < 16 and a proper motion are loaded into one
``torchsky.coords.CatalogState`` at the Gaia epoch J2016.0 (TCB) and rotated to
the Galactic frame with ``to_frame("GALACTIC")``, which turns the
proper-motion vector together with the position. Members are the stars with
a cluster probability above 0.5 in a two-component Gaussian mixture of
proper motion and parallax fitted by EM. Each member is then propagated to
J2000.0 twice: ``propagate_angular`` (proper motion only) and, for stars with a
Gaia radial velocity, ``propagate_physical`` (rigorous space motion), and the
two are compared. Checks: the rotation preserves the proper-motion modulus,
the Galactic-to-ICRS round trip is exact, and the cluster mean matches the
published Gaia value. The astropy ``SkyCoord.apply_space_motion`` parity check
of the same pipeline lives in ``tests/test_gaia_pleiades_motion_parity.py``.
The figure shows the vector-point diagram in (mu_l*, mu_b) and the members on
the sky with their motion relative to the cluster mean.

Data: Gaia DR3 (ESA/Gaia/DPAC, CC BY-SA 3.0 IGO; Gaia Collaboration 2023),
queried from the IRSA TAP mirror and pinned by sha256 (270 kB CSV). Runtime
about 2 s on CPU once the sample is cached.

    pixi run -e test -- python examples/gaia_pleiades_motion.py [--device cuda] [--out DIR]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.csv as pacsv
import torch

from _sample_data import fetch_sample, skip_if_unavailable

from torchsky.coords import CatalogState

GAIA_EPOCH = 2016.0
TARGET_EPOCH = 2000.0
K_KM_S = 4.740470463533348  # km/s per (mas/yr / mas): 1 au/yr in km/s
# Gaia Collaboration, Babusiaux et al. 2018 (A&A 616, A10), Pleiades.
PUBLISHED = {"pmra": 19.997, "pmdec": -45.548, "parallax": 7.364}


def column(table, name: str, device: str) -> torch.Tensor:
    values = table[name].to_numpy(zero_copy_only=False).astype("float64")
    return torch.from_numpy(values).to(device)


def mixture_members(x: torch.Tensor, n_iter: int = 200) -> torch.Tensor:
    """Membership probability from a two-component Gaussian mixture fitted by EM.

    ``x`` is ``(N, 3)``: (mu_l*, mu_b, parallax). One full-covariance Gaussian
    models the cluster, one the field. The cluster starts at the densest
    2 mas/yr proper-motion cell among stars with parallax > 3 mas (closer than
    330 pc, where the field is sparse); the field starts from all stars.
    """
    near = x[:, 2] > 3.0
    cells = (x[near, :2] / 2.0).round()
    uniq, counts = torch.unique(cells, dim=0, return_counts=True)
    seed = torch.cat([2.0 * uniq[counts.argmax()], x[near, 2].median().reshape(1)])
    mean = torch.stack([seed, x.mean(0)])
    eye = torch.eye(3, dtype=x.dtype, device=x.device)
    cov = torch.stack([eye * x.new_tensor([4.0, 4.0, 0.25]), torch.cov(x.T)])
    weight = x.new_tensor([0.1, 0.9])
    for _ in range(n_iter):
        dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
        log_r = dist.log_prob(x[:, None, :]) + weight.log()
        resp = torch.softmax(log_r, dim=1)
        nk = resp.sum(0)
        new_mean = (resp.T @ x) / nk[:, None]
        diff = x[:, None, :] - new_mean[None]
        cov = torch.einsum("nk,nki,nkj->kij", resp, diff, diff) / nk[:, None, None]
        cov = cov + 1e-6 * eye
        shift = float((new_mean - mean).abs().max())
        mean, weight = new_mean, nk / x.shape[0]
        if shift < 1e-8:
            break
    return resp[:, 0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    dev = args.device

    with skip_if_unavailable():
        path = fetch_sample("gaia_pleiades_cone")
    table = pacsv.read_csv(path)
    ra, dec = column(table, "ra", dev), column(table, "dec", dev)
    pmra, pmdec = column(table, "pmra", dev), column(table, "pmdec", dev)
    plx, rv = column(table, "parallax", dev), column(table, "radial_velocity", dev)
    g_mag = column(table, "phot_g_mean_mag", dev)
    print(
        f"Gaia DR3 cone r=1 deg around the Pleiades: {table.num_rows} stars with G < 16"
    )

    icrs = CatalogState(
        lon=ra,
        lat=dec,
        frame="ICRS",
        epoch=GAIA_EPOCH,
        pm_lon_coslat_mas_yr=pmra,
        pm_lat_mas_yr=pmdec,
        parallax_mas=plx,
    )
    gal = icrs.to_frame("GALACTIC")
    pm_l, pm_b = gal.pm_lon_coslat_mas_yr, gal.pm_lat_mas_yr
    modulus_err = float(
        (torch.hypot(pm_l, pm_b) - torch.hypot(pmra, pmdec)).abs().max()
    )
    back = gal.to_frame("ICRS")
    pos_err_uas = (
        float(
            torch.maximum(
                ((back.lon - ra + 180.0) % 360.0 - 180.0).abs() * dec.deg2rad().cos(),
                (back.lat - dec).abs(),
            ).max()
        )
        * 3.6e9
    )
    print(
        f"ICRS -> Galactic: |mu| preserved to {modulus_err:.1e} mas/yr, "
        f"round-trip position error {pos_err_uas:.1e} uas"
    )
    assert modulus_err < 1e-9 and pos_err_uas < 1.0

    p_member = mixture_members(torch.stack([pm_l, pm_b, plx], dim=1))
    member = p_member > 0.5
    n_mem = int(member.sum())
    mean_l, mean_b = float(pm_l[member].mean()), float(pm_b[member].mean())
    mean_ra, mean_dec = float(pmra[member].mean()), float(pmdec[member].mean())
    mean_plx = float(plx[member].mean())
    print(
        f"members: {n_mem} of {table.num_rows}; mean parallax {mean_plx:.3f} mas "
        f"({1000.0 / mean_plx:.1f} pc)"
    )
    print(
        f"cluster proper motion: ICRS (mu_a*, mu_d) = ({mean_ra:.2f}, {mean_dec:.2f}) mas/yr; "
        f"Galactic (mu_l*, mu_b) = ({mean_l:.2f}, {mean_b:.2f}) mas/yr"
    )
    print(
        f"published (Gaia DR2, Babusiaux et al. 2018): ({PUBLISHED['pmra']:.2f}, "
        f"{PUBLISHED['pmdec']:.2f}) mas/yr, parallax {PUBLISHED['parallax']:.3f} mas"
    )
    assert (
        abs(mean_ra - PUBLISHED["pmra"]) < 0.5
        and abs(mean_dec - PUBLISHED["pmdec"]) < 0.5
    )
    assert abs(mean_plx - PUBLISHED["parallax"]) < 0.1

    # Tangential velocity of the cluster and the spread of member motions.
    v_tan = K_KM_S * math.hypot(mean_l, mean_b) / mean_plx
    d_l, d_b = pm_l[member] - mean_l, pm_b[member] - mean_b
    sigma_pm = float(torch.sqrt(0.5 * (d_l.square() + d_b.square()).mean()))
    print(
        f"tangential velocity {v_tan:.1f} km/s; member proper-motion rms per axis "
        f"{sigma_pm:.2f} mas/yr = {K_KM_S * sigma_pm / mean_plx:.2f} km/s "
        "(internal motion, binaries and projection, not deconvolved)"
    )

    # Epoch propagation J2016.0 -> J2000.0: angular model for every member,
    # rigorous space motion where Gaia measured a radial velocity.
    mem = member
    sub = CatalogState(
        lon=ra[mem],
        lat=dec[mem],
        frame="ICRS",
        epoch=GAIA_EPOCH,
        pm_lon_coslat_mas_yr=pmra[mem],
        pm_lat_mas_yr=pmdec[mem],
        parallax_mas=plx[mem],
        radial_velocity_km_s=rv[mem],
    )
    angular = sub.propagate_angular(TARGET_EPOCH)
    shift = (
        torch.hypot(
            ((angular.lon - sub.lon + 180.0) % 360.0 - 180.0) * sub.lat.deg2rad().cos(),
            angular.lat - sub.lat,
        )
        * 3600.0
    )
    has_rv = torch.isfinite(rv[mem])
    full = CatalogState(
        lon=ra[mem][has_rv],
        lat=dec[mem][has_rv],
        frame="ICRS",
        epoch=GAIA_EPOCH,
        pm_lon_coslat_mas_yr=pmra[mem][has_rv],
        pm_lat_mas_yr=pmdec[mem][has_rv],
        parallax_mas=plx[mem][has_rv],
        radial_velocity_km_s=rv[mem][has_rv],
    )
    physical = full.propagate_physical(TARGET_EPOCH)
    gap_uas = (
        torch.hypot(
            (physical.lon - angular.lon[has_rv]) * physical.lat.deg2rad().cos(),
            physical.lat - angular.lat[has_rv],
        )
        * 3.6e9
    )
    print(
        f"J2016.0 -> J2000.0: members move {float(shift.median()):.3f} arcsec (median); "
        f"angular vs rigorous space motion for {int(has_rv.sum())} members with RV: "
        f"max {float(gap_uas.max()):.2f} uas"
    )
    assert float(gap_uas.max()) < 50.0  # perspective acceleration is tiny at 136 pc

    out = args.out or Path(tempfile.mkdtemp(prefix="torchsky-gaia-pleiades-"))
    out.mkdir(parents=True, exist_ok=True)
    cpu = lambda t: t.detach().cpu()  # noqa: E731
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    ax = axes[0]
    ax.scatter(cpu(pm_l[~member]), cpu(pm_b[~member]), s=2, c="0.6", label="field")
    ax.scatter(
        cpu(pm_l[member]), cpu(pm_b[member]), s=4, c="C3", label=f"members ({n_mem})"
    )
    ax.set(
        xlim=(-60, 60),
        ylim=(-60, 40),
        aspect="equal",
        xlabel=r"$\mu_{l*}$ [mas/yr]",
        ylabel=r"$\mu_b$ [mas/yr]",
        title="Vector-point diagram, Galactic frame",
    )
    ax.legend(markerscale=4, fontsize=8, loc="lower left")
    ax = axes[1]
    size = (17.0 - cpu(g_mag[member])) * 3.0
    ax.scatter(cpu(gal.lon[member]), cpu(gal.lat[member]), s=size, c="C3", alpha=0.6)
    ax.quiver(
        cpu(gal.lon[member]),
        cpu(gal.lat[member]),
        cpu(d_l),
        cpu(d_b),
        angles="xy",
        scale=40.0,
        width=0.003,
        color="k",
    )
    ax.set(
        xlabel="l [deg]",
        ylabel="b [deg]",
        aspect="equal",
        title="Members, motion relative to the cluster mean",
    )
    ax.invert_xaxis()
    target = out / "gaia_pleiades_motion.png"
    fig.savefig(target, dpi=110)
    print(f"figure: {target}; device={dev}")


if __name__ == "__main__":
    main()
