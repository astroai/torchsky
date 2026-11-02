"""Celestial frames, proper motion, and observed alt/az in torchsky.coords.

Converts ICRS to Galactic, FK5 and the mean ecliptic of date (with exact
round-trip checks), propagates the full space motion of Barnard's star from
Gaia epoch J2016.0 to J2050.0, and computes observed alt/az for a catalogue
at a Mauna Kea site. All inputs are literals. Runtime about 2 s on CPU.

    pixi run -e test -- python examples/frames_and_motion.py [--device cuda]
"""

from __future__ import annotations

import argparse

import torch

from torchsky.coords import (
    CatalogState,
    EarthOrientation,
    Site,
    convert_celestial,
    icrs_to_ecliptic_of_date,
    observed_altaz,
    propagate_proper_motion,
    propagate_space_motion,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    dev = ap.parse_args().device
    f64 = dict(dtype=torch.float64, device=dev)

    # 1. Frame conversions: Sgr A*, the Crab, the LMC centre.
    ra = torch.tensor([266.41683, 83.63308, 80.89417], **f64)
    dec = torch.tensor([-29.00781, 22.01450, -69.75611], **f64)
    glon, glat = convert_celestial(ra, dec, from_frame="ICRS", to_frame="Galactic")
    ra2, dec2 = convert_celestial(glon, glat, from_frame="Galactic", to_frame="ICRS")
    err_gal = float(torch.maximum((ra2 - ra).abs(), (dec2 - dec).abs()).max()) * 3600e3
    assert err_gal < 1e-3, err_gal
    fk5 = convert_celestial(ra, dec, from_frame="ICRS", to_frame="FK5", to_equinox=2000.0)
    fk5_off = float((fk5[0] - ra).abs().max()) * 3600e3
    lam, beta = icrs_to_ecliptic_of_date(ra, dec, torch.tensor(60000.0, **f64))
    print("source      l [deg]   b [deg]   ecl.lon  ecl.lat")
    for i, name in enumerate(["Sgr A*", "Crab", "LMC"]):
        print(f"{name:8s} {glon[i]:9.4f} {glat[i]:9.4f} {lam[i]:9.4f} {beta[i]:8.4f}")
    print(f"Galactic round trip: {err_gal:.1e} mas; ICRS->FK5(J2000) shift: {fk5_off:.1f} mas")

    # 2. Space motion of Barnard's star, J2016.0 -> J2050.0.
    star = [torch.tensor([v], **f64) for v in
            (269.45402, 4.66828, -797.84, 10326.93, 546.98, -110.51)]
    out = propagate_space_motion(*star, source_epoch_jyear=2016.0, target_epoch_jyear=2050.0)
    ra_n, dec_n = out[0], out[1]
    d_dec_arcsec = float(dec_n - star[1]) * 3600.0
    flat_ra, flat_dec = propagate_proper_motion(
        *star[:4], source_epoch_jyear=2016.0, target_epoch_jyear=2050.0)
    diff_mas = float(((flat_dec - dec_n) ** 2 + ((flat_ra - ra_n) * torch.cos(dec_n.deg2rad())) ** 2).sqrt()) * 3600e3
    assert abs(d_dec_arcsec - 10.327 * 34.0) < 2.0
    print(f"Barnard's star moves {d_dec_arcsec:.1f} arcsec in Dec over 34 yr; "
          f"rigorous vs linear proper motion differ by {diff_mas:.1f} mas")

    # 3. Observed alt/az from Mauna Kea at one UTC epoch.
    site = Site(lon_deg=torch.tensor(-155.4681, **f64), lat_deg=torch.tensor(19.8207, **f64),
                height_m=4205.0)
    eop = EarthOrientation(dut1_s=torch.tensor(0.0, **f64))
    state = CatalogState(lon=ra, lat=dec, frame="ICRS", epoch=2000.0)
    az, alt = observed_altaz(state, obstime_mjd_utc=torch.tensor(60000.3, **f64),
                             site=site, eop=eop, refraction=False)
    for name, a, e in zip(["Sgr A*", "Crab", "LMC"], az, alt):
        print(f"{name:8s} az={float(a):7.2f} alt={float(e):7.2f}")
    assert torch.isfinite(alt).all()
    print(f"device={dev}")


if __name__ == "__main__":
    main()
