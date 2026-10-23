from __future__ import annotations

import math

import torch

from torchsky.map import (
    HealpixResult,
    Observation,
    ResampleConfig,
    Target,
    coadd,
    coadd_cube,
    coadd_healpix,
    resample_healpix,
)
from torchsky.sphere import BMOC, MOC, HealSparseMap, sample_randoms
from torchsky.wcs import (
    WCS,
    IFUSliceWCS,
    apply_dcr,
    apply_ephemeris_offset,
    dcr_offset,
)


def _make_tan_wcs(
    crval: tuple[float, float] = (150.0, 2.0),
    cdelt: float = 1.0 / 3600.0,
    shape: tuple[int, int] = (32, 32),
) -> WCS:
    ny, nx = shape
    return WCS(
        {
            "NAXIS": 2,
            "NAXIS1": nx,
            "NAXIS2": ny,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRPIX1": nx / 2.0 + 0.5,
            "CRPIX2": ny / 2.0 + 0.5,
            "CRVAL1": crval[0],
            "CRVAL2": crval[1],
            "CD1_1": -cdelt,
            "CD1_2": 0.0,
            "CD2_1": 0.0,
            "CD2_2": cdelt,
        }
    )


def test_wcs_to_header_roundtrip_and_sip() -> None:
    header = {
        "NAXIS": 2,
        "NAXIS1": 64,
        "NAXIS2": 48,
        "CTYPE1": "RA---TAN-SIP",
        "CTYPE2": "DEC--TAN-SIP",
        "CRPIX1": 32.0,
        "CRPIX2": 24.0,
        "CRVAL1": 53.12,
        "CRVAL2": -27.81,
        "CD1_1": -5.0e-5,
        "CD1_2": 1.0e-6,
        "CD2_1": 1.0e-6,
        "CD2_2": 5.0e-5,
        "A_ORDER": 2,
        "A_2_0": 1.2e-7,
        "A_0_2": -0.8e-7,
        "B_ORDER": 2,
        "B_1_1": 1.5e-7,
    }
    w1 = WCS(header)
    assert w1.pixel_shape == (64, 48)
    assert w1.array_shape == (48, 64)

    exported = w1.to_header(relax=True, include_naxis=True)
    assert exported["CTYPE1"] == "RA---TAN-SIP"
    assert exported["A_ORDER"] == 2
    assert math.isclose(float(exported["A_2_0"]), 1.2e-7, rel_tol=1e-12)

    head_str = w1.to_head_string(relax=True)
    assert "CTYPE1" in head_str
    assert all(len(line) == 80 for line in head_str.splitlines())
    assert head_str.strip().endswith("END")

    w2 = WCS(exported)
    x = torch.tensor([10.0, 25.0, 50.0], dtype=torch.float64)
    y = torch.tensor([8.0, 20.0, 40.0], dtype=torch.float64)
    ra1, dec1 = w1.pixel_to_world(x, y)
    ra2, dec2 = w2.pixel_to_world(x, y)
    assert torch.allclose(ra1, ra2, atol=1e-11)
    assert torch.allclose(dec1, dec2, atol=1e-11)


def test_wcs_geometry_helpers() -> None:
    scale = 0.5 / 3600.0
    angle = math.radians(25.0)
    header = {
        "NAXIS": 2,
        "NAXIS1": 40,
        "NAXIS2": 30,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 20.5,
        "CRPIX2": 15.5,
        "CRVAL1": 180.0,
        "CRVAL2": 10.0,
        "CD1_1": -scale * math.cos(angle),
        "CD1_2": -scale * math.sin(angle),
        "CD2_1": -scale * math.sin(angle),
        "CD2_2": scale * math.cos(angle),
    }
    w = WCS(header)

    sx, sy = w.pixel_scales()
    assert math.isclose(sx, scale, rel_tol=1e-10)
    assert math.isclose(sy, scale, rel_tol=1e-10)
    assert math.isclose(w.pixel_area(), scale * scale, rel_tol=1e-10)
    assert math.isclose(w.position_angle(), 25.0, abs_tol=1e-8)

    fp = w.footprint(center=True)
    assert fp.shape == (4, 2)

    pam_deg2 = w.pixel_area_map(unit="deg2")
    pam_arcsec2 = w.pixel_area_map(unit="arcsec2")
    assert pam_deg2.shape == (30, 40)
    assert torch.allclose(pam_arcsec2, torch.full_like(pam_arcsec2, 0.25), rtol=1e-4)


def test_ifu_slice_wcs_piecewise_disjoint() -> None:
    w_s0 = _make_tan_wcs(crval=(150.0, 2.0), shape=(16, 16))
    w_s1 = _make_tan_wcs(crval=(150.02, 2.0), shape=(16, 16))

    mask0 = torch.zeros((16, 16), dtype=torch.bool)
    mask0[:, :7] = True
    mask1 = torch.zeros((16, 16), dtype=torch.bool)
    mask1[:, 9:] = True

    yg = torch.arange(16, dtype=torch.float64).unsqueeze(1).expand(16, 16)
    wave0 = 5000.0 + 10.0 * yg
    wave1 = 6000.0 + 10.0 * yg

    ifu = IFUSliceWCS.from_slices(
        [
            {"mask": mask0, "wcs": w_s0, "wavelength": wave0},
            {"mask": mask1, "wcs": w_s1, "wavelength": wave1},
        ]
    )
    assert ifu.detector_shape == (16, 16)
    assert ifu.valid_mask.sum().item() == int(mask0.sum() + mask1.sum())

    # Evaluate across slice 0, dead gap (x=7.5), and slice 1
    x = torch.tensor([2.0, 7.5, 12.0], dtype=torch.float64)
    y = torch.tensor([5.0, 5.0, 5.0], dtype=torch.float64)
    ra, dec, wave = ifu.pixel_to_world(x, y)

    assert torch.isfinite(ra[0]) and torch.isfinite(wave[0])
    assert torch.isnan(ra[1]) and torch.isnan(dec[1]) and torch.isnan(wave[1])
    assert torch.isfinite(ra[2]) and torch.isfinite(wave[2])
    assert math.isclose(float(wave[0].item()), 5050.0, abs_tol=1e-6)
    assert math.isclose(float(wave[2].item()), 6050.0, abs_tol=1e-6)


def test_dcr_and_ephemeris_offsets() -> None:
    wave = torch.tensor([4000.0, 5500.0, 8000.0], dtype=torch.float64)
    zenith = torch.tensor(45.0, dtype=torch.float64)
    d_east_deg, d_north_deg = dcr_offset(
        wave, zenith, parallactic_deg=0.0, ref_wavelength_angstrom=5500.0
    )
    assert torch.allclose(d_east_deg, torch.zeros_like(d_east_deg), atol=1e-12)
    assert math.isclose(float(d_north_deg[1].item()), 0.0, abs_tol=1e-10)
    assert float(d_north_deg[0].item()) > 0.0
    assert float(d_north_deg[2].item()) < 0.0

    ra = torch.tensor([150.0, 150.0, 150.0], dtype=torch.float64)
    dec = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
    ra_dcr, dec_dcr = apply_dcr(
        ra,
        dec,
        wavelength_angstrom=wave,
        zenith_deg=45.0,
        parallactic_deg=0.0,
        ref_wavelength_angstrom=5500.0,
    )
    assert torch.allclose(ra_dcr, ra, atol=1e-12)
    assert float(dec_dcr[0].item()) > float(dec[0].item())

    # Moving object with dra_dt = +3600 arcsec/hr (= 1 deg/hr in cos(dec)-scaled RA) at dec=0
    ra0 = torch.tensor([100.0], dtype=torch.float64)
    dec0 = torch.tensor([0.0], dtype=torch.float64)
    ra_track, dec_track = apply_ephemeris_offset(
        ra0,
        dec0,
        mjd_obs=60000.0 + 1.0 / 24.0,
        reference_mjd=60000.0,
        dra_cosdec_arcsec_per_hr=3600.0,
        ddec_arcsec_per_hr=-1800.0,
    )
    assert math.isclose(float(ra_track.item()), 99.0, abs_tol=1e-9)
    assert math.isclose(float(dec_track.item()), 0.5, abs_tol=1e-9)


def test_bmoc_and_moc_from_wcs_and_randoms() -> None:
    bmoc = BMOC.from_cone(150.0, 2.0, 1.5, max_order=6)
    assert not bmoc.is_empty
    assert bmoc.area_sq_deg >= bmoc.interior_area_sq_deg > 0.0
    assert (bmoc.interior_moc() & bmoc.boundary_moc()).is_empty
    assert (bmoc.interior_moc() | bmoc.boundary_moc()) == bmoc.to_moc()

    w = _make_tan_wcs(crval=(150.0, 2.0), cdelt=0.02, shape=(32, 32))
    moc_w = MOC.from_wcs(w, max_order=8)
    bmoc_w = BMOC.from_wcs(w, max_order=8)
    assert not moc_w.is_empty
    assert not bmoc_w.is_empty
    assert bool(moc_w.contains(torch.tensor([150.0]), torch.tensor([2.0])).item())
    assert bool(bmoc_w.contains(torch.tensor([150.0]), torch.tensor([2.0])).item())

    gen = torch.Generator().manual_seed(42)
    ra_rnd, dec_rnd = moc_w.sample_randoms(500, generator=gen)
    assert ra_rnd.shape == (500,)
    assert dec_rnd.shape == (500,)
    assert torch.all(moc_w.contains(ra_rnd, dec_rnd))

    # Polygon BMOC
    poly_ra = torch.tensor([149.5, 150.5, 150.5, 149.5], dtype=torch.float64)
    poly_dec = torch.tensor([1.5, 1.5, 2.5, 2.5], dtype=torch.float64)
    bmoc_poly = BMOC.from_polygon(poly_ra, poly_dec, max_order=8)
    assert not bmoc_poly.is_empty
    assert bmoc_poly.interior_area_sq_deg > 0.0


def test_healsparse_wide_columns_and_from_observations() -> None:
    pix = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
    cols = {
        "flux": torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
        "weight": torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32),
        "fracdet": torch.tensor([0.5, 1.0, 0.8, 0.2], dtype=torch.float32),
    }
    hsp = HealSparseMap.from_columns(pix, cols, nside_sparse=64, nside_coverage=8, nest=True)
    assert hsp.column_names == ("flux", "weight", "fracdet")

    w_map = hsp["weight"]
    assert w_map.column_names == ("weight",)
    q_val, q_valid = w_map.lookup_with_valid(pix)
    assert torch.all(q_valid)
    assert torch.allclose(q_val, cols["weight"])

    sub_map = hsp[["flux", "fracdet"]]
    assert sub_map.column_names == ("flux", "fracdet")

    # Weighted randoms from HealSparseMap
    gen = torch.Generator().manual_seed(123)
    ra_r, dec_r = sample_randoms(200, hsp, weights="fracdet", generator=gen)
    assert ra_r.shape == (200,)
    _, valid_r = hsp.lookup_radec(ra_r, dec_r)
    assert torch.all(valid_r)

    # Build from Observation list
    w = _make_tan_wcs(crval=(150.0, 2.0), cdelt=0.05, shape=(24, 24))
    obs = Observation(
        data=torch.ones((24, 24), dtype=torch.float32),
        wcs=w,
        weight=torch.full((24, 24), 4.0, dtype=torch.float32),
    )
    cov_map = HealSparseMap.from_observations([obs], nside_sparse=128, nside_coverage=16)
    assert cov_map.column_names == ("weight", "nexp", "fracdet")
    assert cov_map.valid_pixels.numel() > 0


def test_coadd_healpix_and_resample_healpix() -> None:
    w = _make_tan_wcs(crval=(150.0, 2.0), cdelt=0.05, shape=(32, 32))
    obs = Observation(
        data=torch.full((32, 32), 7.5, dtype=torch.float32),
        wcs=w,
        weight=torch.ones((32, 32), dtype=torch.float32),
    )
    hp_res = coadd_healpix([obs], nside=64, nest=True, sparse=False)
    assert isinstance(hp_res, HealpixResult)
    cov = hp_res.weight > 0
    assert int(cov.sum().item()) > 0
    assert torch.allclose(hp_res.image[cov], torch.full_like(hp_res.image[cov], 7.5), atol=1e-4)

    hp_sparse = coadd_healpix([obs], nside=64, nest=True, sparse=True, nside_coverage=16)
    assert hp_sparse.sparse_map is not None
    assert hp_sparse.sparse_map.valid_pixels.numel() == int(cov.sum().item())

    target = Target(wcs=_make_tan_wcs(crval=(150.0, 2.0), cdelt=0.04, shape=(16, 16)), shape=(16, 16))
    cutout = resample_healpix(hp_res, target)
    assert cutout.shape == (16, 16)
    assert torch.allclose(cutout.image, torch.full_like(cutout.image, 7.5), atol=1e-3)


def test_coadd_cube_3d_ifu_drizzle_and_emsm() -> None:
    w = _make_tan_wcs(crval=(150.0, 2.0), cdelt=1.0 / 3600.0, shape=(12, 12))
    yg = torch.arange(12, dtype=torch.float32).unsqueeze(1).expand(12, 12)
    wave_map = 5000.0 + 20.0 * yg  # 5000 .. 5220 Angstrom
    # Synthetic linear spectrum F(lambda) = 2.0 + 0.01 * (lambda - 5000)
    data = 2.0 + 0.01 * (wave_map - 5000.0)
    obs = Observation(
        data=data,
        wcs=w,
        weight=torch.ones_like(data),
        wavelength_map=wave_map,
    )
    target = Target(wcs=w, shape=(12, 12))
    wave_grid = torch.tensor([5020.0, 5060.0, 5100.0, 5140.0, 5180.0], dtype=torch.float32)

    cube_drz = coadd_cube(
        [obs],
        target,
        wavelengths=wave_grid,
        resample=ResampleConfig(spectral_kernel="drizzle", spectral_pixfrac=1.0),
    )
    assert cube_drz.cube.shape == (5, 12, 12)
    assert int((cube_drz.weight > 0).sum().item()) > 0

    cube_emsm = coadd_cube(
        [obs],
        target,
        wavelengths=wave_grid,
        resample=ResampleConfig(spectral_kernel="emsm", emsm_scale=1.2),
    )
    assert cube_emsm.cube.shape == (5, 12, 12)
    valid = cube_emsm.weight > 1e-3
    assert int(valid.sum().item()) > 0


def test_coadd_spatial_tiling_matches_untiled() -> None:
    w1 = _make_tan_wcs(crval=(150.0, 2.0), cdelt=1.0 / 3600.0, shape=(32, 32))
    w2 = _make_tan_wcs(crval=(150.002, 2.001), cdelt=1.0 / 3600.0, shape=(32, 32))

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 32, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, 32, dtype=torch.float32),
        indexing="ij",
    )
    img1 = torch.exp(-(xx**2 + yy**2) * 4.0) + 1.0
    img2 = torch.exp(-((xx - 0.1) ** 2 + (yy + 0.1) ** 2) * 4.0) + 1.0

    obs1 = Observation(data=img1, wcs=w1, weight=torch.full_like(img1, 2.0))
    obs2 = Observation(data=img2, wcs=w2, weight=torch.full_like(img2, 3.0))
    target = Target(wcs=w1, shape=(32, 32))

    res_full = coadd([obs1, obs2], target)
    res_tiled = coadd([obs1, obs2], target, tile_size=12)

    assert torch.allclose(res_full.image, res_tiled.image, atol=1e-5)
    assert torch.allclose(res_full.weight, res_tiled.weight, atol=1e-5)
    assert torch.equal(res_full.context, res_tiled.context)


def test_clean_api_aliases_across_wcs_sphere_catalogs_map() -> None:
    import torchsky.catalogs as catalogs
    import torchsky.map as tmap
    import torchsky.sphere as sphere

    w = _make_tan_wcs()
    ra, dec = w.pix2world(torch.tensor([31.5]), torch.tensor([31.5]))
    x, y = w.world2pix(ra, dec)
    assert torch.allclose(x, torch.tensor([31.5], dtype=x.dtype), atol=1e-6)
    assert torch.allclose(y, torch.tensor([31.5], dtype=y.dtype), atol=1e-6)

    jac_fwd = w.jacobian(torch.tensor([31.5]), torch.tensor([31.5]))
    jac_inv = w.jacobian(ra, dec, inverse=True)
    assert jac_fwd.shape == (1, 2, 2)
    assert jac_inv.shape == (1, 2, 2)

    w_aff = w.with_affine(torch.eye(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64))
    assert isinstance(w_aff, WCS)

    poly = sphere.Polygon(
        torch.tensor([10.0, 11.0, 11.0, 10.0]),
        torch.tensor([0.0, 0.0, 1.0, 1.0]),
    )
    assert poly.contains(10.5, 0.5)
    pix_poly = sphere.query_polygon(
        64,
        torch.tensor([10.0, 11.0, 11.0, 10.0]),
        torch.tensor([0.0, 0.0, 1.0, 1.0]),
    )
    assert pix_poly.numel() > 0
    pix_cone = sphere.query_cone(64, 10.5, 0.5, 1.0)
    assert pix_cone.numel() > 0

    hp_map = torch.ones(sphere.nside2npix(16), dtype=torch.float64) * 3.5
    sampled = sphere.sample_map(hp_map, torch.tensor([10.0, 20.0]), torch.tensor([0.0, 5.0]))
    assert torch.allclose(sampled, torch.tensor([3.5, 3.5], dtype=sampled.dtype))

    cat1 = catalogs.Catalog(lon=torch.tensor([10.0, 20.0]), lat=torch.tensor([0.0, 5.0]))
    cat2 = catalogs.Catalog(lon=torch.tensor([10.0001, 50.0]), lat=torch.tensor([0.0, -5.0]))
    assert cat1.state.lon.shape == (2,)
    xm1 = catalogs.crossmatch(cat1, cat2, 5.0)
    xm2 = catalogs.crossmatch(cat1.lon, cat1.lat, cat2.lon, cat2.lat, radius_arcsec=5.0)
    assert torch.equal(xm1.left_index, xm2.left_index)
    assert tmap.CubeResult is tmap.CubeResult
    assert tmap.MultibandResult is tmap.MultibandResult
    assert tmap.ApertureUncertainty is tmap.ApertureUncertainty
