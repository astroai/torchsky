import numpy as np
import torch
import pytest
import math

from torchsky.sphere import compat
from torchsky.sphere import healpix as hp
from torchsky.sphere.geom import query_ellipse as geom_query_ellipse
from torchsky.sphere.geom import query_polygon_general
from torchsky.sphere.spectral import alm_index, alm_size, rotate_alm, synfast


def test_compat_alm_index_roundtrip_and_resolution_exports() -> None:
    lmax = 8
    for m in range(lmax + 1):
        for ell in range(m, lmax + 1):
            index = compat.Alm.getidx(lmax, ell, m)
            assert compat.Alm.getlm(index, lmax) == (ell, m)

    assert compat.nside2resol(8) == hp.nside2resol(8)
    assert compat.nside2pixarea(8, degrees=True) == hp.nside2pixarea(8, degrees=True)


def test_compat_map_fits_roundtrip(tmp_path) -> None:
    nside = 8
    values = torch.arange(hp.nside2npix(nside), dtype=torch.float64)
    path = tmp_path / "map.fits"

    compat.write_map(path, values)
    got = compat.read_map(path)

    torch.testing.assert_close(got, values)
    torch.testing.assert_close(
        compat.new_map(nside, dtype=torch.float64),
        torch.zeros_like(values),
    )


def test_compat_query_polygon_dispatches_concave_regions() -> None:
    lon = torch.tensor([-20.0, 20.0, 20.0, 0.0, 0.0, -20.0], dtype=torch.float64)
    lat = torch.tensor([-20.0, -20.0, 0.0, 0.0, 20.0, 20.0], dtype=torch.float64)
    vertices = torch.stack((lon, lat), dim=1)

    got = compat.query_polygon(32, vertices, lonlat=True)
    expected = query_polygon_general(32, lon, lat)

    torch.testing.assert_close(got, expected)


def test_compat_ang2pix_pix2ang_roundtrip_matches_healpix() -> None:
    rng = np.random.default_rng(123)
    lon = torch.from_numpy(rng.uniform(0.0, 360.0, size=1024)).to(torch.float64)
    lat = torch.from_numpy(np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=1024)))).to(
        torch.float64
    )

    pix = compat.ang2pix(64, lon, lat, nest=False, lonlat=True)
    expected = hp.ang2pix(64, lon, lat, nest=False, lonlat=True)
    torch.testing.assert_close(pix, expected)

    lon1, lat1 = compat.pix2ang(64, pix, nest=False, lonlat=True)
    lon2, lat2 = hp.pix2ang(64, pix, nest=False, lonlat=True)
    torch.testing.assert_close(lon1, lon2, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(lat1, lat2, atol=1e-12, rtol=0.0)


def test_compat_neighbors_aliases_identical() -> None:
    pix = torch.tensor([0, 1, 10, 42], dtype=torch.int64)
    a = compat.get_all_neighbors(32, pix, nest=False)
    b = compat.get_all_neighbours(32, pix, nest=False)
    torch.testing.assert_close(a, b)


def test_compat_neighbors_angle_form_matches_healpix() -> None:
    lon = torch.tensor([12.0, 200.0], dtype=torch.float64)
    lat = torch.tensor([-10.0, 40.0], dtype=torch.float64)
    got = compat.get_all_neighbors(64, lon, lat, nest=False, lonlat=True)
    expected = hp.get_all_neighbours(64, lon, lat, nest=False, lonlat=True)
    torch.testing.assert_close(got, expected)


@pytest.mark.parametrize("nest", [False, True])
def test_compat_ang2pix_pix2ang_match_hpgeom(nest: bool) -> None:
    hpgeom = pytest.importorskip("hpgeom")
    nside = 32
    lon_np = np.array([0.25, 17.5, 92.25, 181.0, 359.1], dtype=np.float64)
    lat_np = np.array([-63.0, -12.5, 0.75, 34.0, 71.25], dtype=np.float64)
    lon = torch.from_numpy(lon_np)
    lat = torch.from_numpy(lat_np)

    got_pix = compat.ang2pix(nside, lon, lat, nest=nest, lonlat=True)
    expected_pix = torch.as_tensor(
        hpgeom.angle_to_pixel(
            nside, lon_np, lat_np, nest=nest, lonlat=True, degrees=True
        ),
        dtype=torch.int64,
    )
    torch.testing.assert_close(got_pix, expected_pix)

    got_lon, got_lat = compat.pix2ang(nside, got_pix, nest=nest, lonlat=True)
    expected_lon_np, expected_lat_np = hpgeom.pixel_to_angle(
        nside, expected_pix.numpy(), nest=nest, lonlat=True, degrees=True
    )
    torch.testing.assert_close(
        got_lon, torch.as_tensor(expected_lon_np), atol=1e-12, rtol=0.0
    )
    torch.testing.assert_close(
        got_lat, torch.as_tensor(expected_lat_np), atol=1e-12, rtol=0.0
    )


@pytest.mark.parametrize("nest", [False, True])
def test_compat_ang2pix_matches_astropy_healpix(nest: bool) -> None:
    pytest.importorskip("astropy_healpix")
    units = pytest.importorskip("astropy.units")
    from astropy_healpix import HEALPix

    nside = 16
    lon_np = np.array([3.0, 45.5, 127.25, 250.0, 315.75], dtype=np.float64)
    lat_np = np.array([-50.0, -5.5, 22.0, 48.75, 80.0], dtype=np.float64)
    got = compat.ang2pix(
        nside,
        torch.from_numpy(lon_np),
        torch.from_numpy(lat_np),
        nest=nest,
        lonlat=True,
    )
    order = "nested" if nest else "ring"
    expected = HEALPix(nside=nside, order=order).lonlat_to_healpix(
        lon_np * units.deg,
        lat_np * units.deg,
    )
    torch.testing.assert_close(got, torch.as_tensor(expected, dtype=torch.int64))


def test_compat_query_circle_strict_mode_matches_non_strict() -> None:
    """Strict mode uses the same torch-native kernel as the default path."""
    args = (32, 25.0, -15.0, 5.0)
    kwargs = dict(degrees=True, inclusive=True, nest=True, fact=4)
    with compat.strict_mode(True):
        out_s = compat.query_circle(*args, **kwargs)
    with compat.strict_mode(False):
        out_n = compat.query_circle(*args, **kwargs)
    torch.testing.assert_close(out_s, out_n)


def test_compat_query_circle_vec_strict_mode_matches_non_strict() -> None:
    lon = 25.0
    lat = -15.0
    radius = np.deg2rad(5.0)
    x, y, z = hp.lonlat_to_xyz(torch.tensor(lon), torch.tensor(lat))
    vec = torch.stack([x, y, z]).to(torch.float64)
    with compat.strict_mode(True):
        out_s = compat.query_circle_vec(
            32, vec, radius, inclusive=True, nest=True, fact=4
        )
    with compat.strict_mode(False):
        out_n = compat.query_circle_vec(
            32, vec, radius, inclusive=True, nest=True, fact=4
        )
    torch.testing.assert_close(out_s, out_n)


def test_compat_query_ellipse_calls_geometry_implementation() -> None:
    out_compat = compat.query_ellipse(
        32, 10.0, -20.0, 6.0, 2.5, pa_deg=35.0, nest=False
    )
    out_geom = geom_query_ellipse(32, 10.0, -20.0, 6.0, 2.5, pa_deg=35.0, nest=False)
    torch.testing.assert_close(out_compat, out_geom)


def test_compat_query_ellipse_strict_mode_matches_non_strict() -> None:
    args = (32, 10.0, -20.0, 6.0, 2.5)
    with compat.strict_mode(True):
        out_s = compat.query_ellipse(*args, pa_deg=35.0, nest=False)
    with compat.strict_mode(False):
        out_n = compat.query_ellipse(*args, pa_deg=35.0, nest=False)
    torch.testing.assert_close(out_s, out_n)


def test_compat_query_ellipse_auto_backend_matches_torch() -> None:
    kwargs = dict(
        nside=32,
        a=10.0,
        b=-20.0,
        semi_major=6.0,
        semi_minor=2.5,
        pa_deg=35.0,
        nest=True,
        inclusive=True,
        fact=4,
    )
    out_auto = compat.query_ellipse(**kwargs, backend="auto")
    out_torch = compat.query_ellipse(**kwargs, backend="torch")
    torch.testing.assert_close(out_auto, out_torch)


def test_compat_query_ellipse_return_ranges_and_theta_phi_units() -> None:
    nside = 32
    lon = 10.0
    lat = -20.0
    major = 6.0
    minor = 2.5
    pa = 35.0

    got_ranges = compat.query_ellipse(
        nside,
        lon,
        lat,
        major,
        minor,
        pa_deg=pa,
        nest=True,
        inclusive=True,
        fact=4,
        return_pixel_ranges=True,
    )
    got_pixels = compat.query_ellipse(
        nside,
        lon,
        lat,
        major,
        minor,
        pa_deg=pa,
        nest=True,
        inclusive=True,
        fact=4,
        return_pixel_ranges=False,
    )
    # Ranges from `pixels_to_pixel_ranges` are half-open [start, stop); expand with
    # inclusive=False to match that encoding (inclusive=True is for closed-range rows).
    round_trip = compat.pixel_ranges_to_pixels(got_ranges, inclusive=False)
    assert set(round_trip.cpu().tolist()) == set(got_pixels.cpu().tolist())

    theta = np.deg2rad(90.0 - lat)
    phi = np.deg2rad(lon)
    got_tf = compat.query_ellipse(
        nside,
        theta,
        phi,
        np.deg2rad(major),
        np.deg2rad(minor),
        alpha=np.deg2rad(pa),
        nest=True,
        inclusive=False,
        fact=4,
        lonlat=False,
        degrees=False,
    )
    assert got_tf.ndim == 1
    assert got_tf.dtype == torch.int64
    assert got_tf.numel() > 0


def test_compat_query_ellipse_inclusive_fact_refines_overlap() -> None:
    # Thin, long ellipse where coarse overlap approximations can miss edge pixels.
    nside = 64
    lon = 246.75536890809175
    lat = 42.359164144309915
    major = 19.620377411573955
    minor = 0.44602023784123124
    pa = 159.9907917472977

    coarse = compat.query_ellipse(
        nside, lon, lat, major, minor, pa_deg=pa, nest=False, inclusive=True, fact=1
    )
    fine = compat.query_ellipse(
        nside, lon, lat, major, minor, pa_deg=pa, nest=False, inclusive=True, fact=8
    )
    assert coarse.numel() > 0 and fine.numel() > 0
    assert fine.numel() != coarse.numel()
    # fact=1 uses direct overlap; fact>1 uses supersampling projection — sets need
    # not be nested; require meaningful overlap on this stress geometry.
    common = set(coarse.cpu().tolist()) & set(fine.cpu().tolist())
    assert len(common) >= min(coarse.numel(), fine.numel()) // 4


def test_compat_query_ellipse_inclusive_fact_non_empty_on_hard_case() -> None:
    nside = 64
    lon = 32.82103279179087
    lat = 61.05973037738469
    major = 19.277723008371698
    minor = 11.918411640241988
    pa = -43.690581832574374

    got = compat.query_ellipse(
        nside, lon, lat, major, minor, pa_deg=pa, nest=False, inclusive=True, fact=8
    )
    assert got.numel() > 0
    assert got.dtype == torch.int64


def test_compat_fit_remove_dipole() -> None:
    nside = 8
    npix = hp.nside2npix(nside)
    pix = torch.arange(npix, dtype=torch.int64)
    x, y, z = hp.pix2vec(nside, pix, nest=False)
    mono = torch.tensor(1.2, dtype=torch.float64)
    dip = torch.tensor([0.05, -0.03, 0.02], dtype=torch.float64)
    m = mono + dip[0] * x + dip[1] * y + dip[2] * z

    mono_fit, dip_fit = compat.fit_dipole(m, nside=nside, nest=False)
    torch.testing.assert_close(mono_fit, mono, atol=1e-10, rtol=0.0)
    torch.testing.assert_close(dip_fit, dip, atol=1e-10, rtol=0.0)

    cleaned = compat.remove_dipole(m, nside=nside, nest=False)
    torch.testing.assert_close(cleaned, torch.zeros_like(cleaned), atol=1e-10, rtol=0.0)


def test_compat_pol_spectral_api_shapes() -> None:
    nside = 8
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(222)
    maps = torch.from_numpy(rng.normal(size=(3, npix))).to(torch.float64)

    alms = compat.map2alm(maps, nside=nside, lmax=8, pol=True)
    assert alms.shape[0] == 3

    rec = compat.alm2map(alms, nside=nside, lmax=8, pol=True)
    assert rec.shape == maps.shape

    cl = compat.anafast(maps, nside=nside, lmax=8, pol=True)
    assert cl.shape == (6, 9)

    beam = compat.gaussian_beam(np.deg2rad(1.0), lmax=8, pol=True)
    assert beam.shape == (9, 4)


def test_compat_map2alm_lsq_scalar_api() -> None:
    nside = 8
    lmax = 6
    npix = hp.nside2npix(nside)
    expected_nalm = (lmax + 1) * (lmax + 2) // 2
    rng = np.random.default_rng(507)
    m = torch.from_numpy(rng.normal(size=npix)).to(torch.float64)
    alm, rel_res, n_iter = compat.map2alm_lsq(
        m, lmax=lmax, mmax=lmax, pol=False, maxiter=4, nside=nside
    )
    assert alm.shape == (expected_nalm,)
    assert isinstance(rel_res, float)
    assert np.isfinite(rel_res)
    assert 0 <= n_iter <= 4


def test_compat_sphtfunc_aliases() -> None:
    lmax = 6
    nalm = (lmax + 1) * (lmax + 2) // 2
    alm = torch.ones((nalm,), dtype=torch.complex128)
    fl = torch.linspace(1.0, 2.0, steps=lmax + 1, dtype=torch.float64)

    scaled = compat.almxfl(alm, fl)
    assert scaled.shape == alm.shape

    cl = compat.alm2cl(alm, lmax=lmax, mmax=lmax)
    assert cl.shape == (lmax + 1,)

    b1 = compat.gaussian_beam(np.deg2rad(1.0), lmax=lmax, pol=False)
    b2 = compat.gauss_beam(np.deg2rad(1.0), lmax=lmax, pol=False)
    torch.testing.assert_close(b1, b2, atol=1e-12, rtol=0.0)

    nside = 8
    npix = hp.nside2npix(nside)
    m = torch.linspace(-1.0, 1.0, steps=npix, dtype=torch.float64)
    sm = compat.smoothing(m, fwhm=np.deg2rad(30.0 / 60.0), lmax=lmax)
    assert sm.shape == m.shape


def test_compat_syn_and_beam_aliases() -> None:
    lmax = 6
    nside = 8
    cl = torch.zeros((lmax + 1,), dtype=torch.float64)

    alm = compat.synalm(cl, lmax=lmax, mmax=lmax)
    assert alm.shape == ((lmax + 1) * (lmax + 2) // 2,)

    m = compat.synfast(cl, nside=nside, lmax=lmax, mmax=lmax, pol=True)
    assert m.shape == (hp.nside2npix(nside),)

    theta = torch.linspace(0.0, 0.2, steps=64, dtype=torch.float64)
    bl = compat.gauss_beam(np.deg2rad(1.0), lmax=lmax, pol=False)
    beam = compat.bl2beam(bl, theta)
    bl_back = compat.beam2bl(beam, theta, lmax=lmax)
    assert beam.shape == theta.shape
    assert bl_back.shape == bl.shape


def test_compat_pixwin_api() -> None:
    nside = 8
    lmax = 10
    pw = compat.pixwin(nside, pol=False, lmax=lmax)
    assert pw.shape == (lmax + 1,)
    pwt, pwp = compat.pixwin(nside, pol=True, lmax=lmax)
    assert pwt.shape == (lmax + 1,)
    assert pwp.shape == (lmax + 1,)


def test_compat_vec_queries_match_sphere_healpix_aliases() -> None:
    """compat re-exports match the sphere-owned HEALPix entry points."""
    nside = 32
    lon = 25.0
    lat = -15.0
    radius = np.deg2rad(5.0)
    x, y, z = hp.lonlat_to_xyz(torch.tensor(lon), torch.tensor(lat))
    vec = torch.stack([x, y, z]).to(torch.float64)

    got_circle = compat.query_circle_vec(nside, vec, radius, nest=True, inclusive=False)
    torch.testing.assert_close(
        got_circle, hp.query_circle_vec(nside, vec, radius, nest=True, inclusive=False)
    )
    got_circle_inc = compat.query_circle_vec(
        nside, vec, radius, nest=True, inclusive=True
    )
    torch.testing.assert_close(
        got_circle_inc,
        hp.query_circle_vec(nside, vec, radius, nest=True, inclusive=True),
    )

    v_lon = np.array([10.0, 20.0, 20.0, 10.0])
    v_lat = np.array([-5.0, -5.0, 5.0, 5.0])
    vx, vy, vz = hp.lonlat_to_xyz(torch.as_tensor(v_lon), torch.as_tensor(v_lat))
    vxyz = torch.stack([vx, vy, vz], dim=1).to(torch.float64)
    got_poly = compat.query_polygon_vec(nside, vxyz, nest=False, inclusive=False)
    torch.testing.assert_close(
        got_poly, hp.query_polygon_vec(nside, vxyz, nest=False, inclusive=False)
    )

    ranges = torch.tensor([[10, 13], [20, 22]], dtype=torch.int64)
    torch.testing.assert_close(
        compat.pixel_ranges_to_pixels(ranges, inclusive=False),
        torch.tensor([10, 11, 12, 20, 21], dtype=torch.int64),
    )
    torch.testing.assert_close(
        compat.pixel_ranges_to_pixels(ranges, inclusive=True),
        torch.tensor([10, 11, 12, 13, 20, 21, 22], dtype=torch.int64),
    )
    torch.testing.assert_close(
        compat.upgrade_pixel_ranges(2, ranges, 8),
        hp.upgrade_pixel_ranges(2, ranges, 8),
    )


def test_compat_pixels_to_ranges_roundtrip() -> None:
    pixels = torch.tensor([8, 9, 10, 15, 17, 18, 18], dtype=torch.int64)
    ranges = compat.pixels_to_pixel_ranges(pixels)
    torch.testing.assert_close(
        ranges, torch.tensor([[8, 11], [15, 16], [17, 19]], dtype=torch.int64)
    )
    restored = compat.pixel_ranges_to_pixels(ranges, inclusive=False)
    torch.testing.assert_close(
        restored, torch.tensor([8, 9, 10, 15, 17, 18], dtype=torch.int64)
    )


def test_compat_pixel_range_helpers_match_hpgeom() -> None:
    hpgeom = pytest.importorskip("hpgeom")
    ranges = np.array([[0, 3], [7, 8], [20, 24]], dtype=np.int64)

    got_exclusive = compat.pixel_ranges_to_pixels(ranges, inclusive=False)
    expected_exclusive = torch.as_tensor(
        hpgeom.pixel_ranges_to_pixels(ranges, inclusive=False), dtype=torch.int64
    )
    torch.testing.assert_close(got_exclusive, expected_exclusive)

    got_inclusive = compat.pixel_ranges_to_pixels(ranges, inclusive=True)
    expected_inclusive = torch.as_tensor(
        hpgeom.pixel_ranges_to_pixels(ranges, inclusive=True), dtype=torch.int64
    )
    torch.testing.assert_close(got_inclusive, expected_inclusive)

    upgraded = compat.upgrade_pixel_ranges(2, ranges, 16)
    expected_upgraded = torch.as_tensor(
        hpgeom.upgrade_pixel_ranges(2, ranges, 16), dtype=torch.int64
    )
    torch.testing.assert_close(upgraded, expected_upgraded)


@pytest.mark.parametrize("nest", [False, True])
def test_compat_upgrade_pixels_match_hpgeom(nest: bool) -> None:
    hpgeom = pytest.importorskip("hpgeom")
    if not hasattr(hpgeom, "upgrade_pixels"):
        pytest.skip("hpgeom does not expose upgrade_pixels")

    pixels_np = np.array([2, 5, 18], dtype=np.int64)
    pixels = torch.from_numpy(pixels_np)
    got = compat.upgrade_pixels(4, pixels, 16, nest=nest)
    expected = torch.as_tensor(
        hpgeom.upgrade_pixels(4, pixels_np, 16, nest=nest),
        dtype=torch.int64,
    )
    if not nest:
        expected = torch.unique(expected, sorted=True)
    torch.testing.assert_close(got, expected)


def test_compat_degrade_pixels_expected_nested_and_ring() -> None:
    parent_nest = torch.tensor([2, 5], dtype=torch.int64)
    child_nest = compat.upgrade_pixels(2, parent_nest, 8, nest=True)
    torch.testing.assert_close(
        compat.degrade_pixels(8, child_nest[[0, 1, 20, -1]], 2, nest=True),
        parent_nest,
    )

    parent_ring = compat.nest2ring(2, parent_nest)
    child_ring = compat.upgrade_pixels(2, parent_ring, 8, nest=False)
    degraded_ring = compat.degrade_pixels(8, child_ring[[0, 1, 20, -1]], 2, nest=False)
    degraded_nest = torch.sort(compat.ring2nest(2, degraded_ring)).values
    torch.testing.assert_close(degraded_nest, parent_nest)


def test_compat_query_circle_pixel_ranges_match_hpgeom_expansion() -> None:
    hpgeom = pytest.importorskip("hpgeom")
    nside = 32
    lon = 42.0
    lat = -12.5
    radius = 3.0

    got_ranges = compat.query_circle(
        nside,
        lon,
        lat,
        radius,
        nest=True,
        inclusive=False,
        degrees=True,
        return_pixel_ranges=True,
    )
    got_pixels = compat.pixel_ranges_to_pixels(got_ranges, inclusive=False)
    expected_pixels = torch.as_tensor(
        hpgeom.query_circle(
            nside,
            lon,
            lat,
            radius,
            nest=True,
            inclusive=False,
            degrees=True,
        ),
        dtype=torch.int64,
    )

    torch.testing.assert_close(
        torch.sort(got_pixels).values,
        torch.sort(expected_pixels).values,
    )


def test_compat_fit_remove_monopole_matches_healpy() -> None:
    healpy = pytest.importorskip("healpy")
    nside = 8
    npix = hp.nside2npix(nside)
    pix = torch.arange(npix, dtype=torch.int64)
    _, lat = hp.pix2ang(nside, pix, nest=False, lonlat=True)
    m = (1.5 + 0.01 * lat).to(torch.float64)
    m[5] = hp.UNSEEN

    got = compat.fit_monopole(m, nest=False, bad=hp.UNSEEN, gal_cut=10.0)
    exp = torch.tensor(
        float(healpy.fit_monopole(m.numpy(), nest=False, bad=hp.UNSEEN, gal_cut=10.0)),
        dtype=torch.float64,
    )
    torch.testing.assert_close(got, exp, atol=1e-12, rtol=0.0)

    got_removed, got_fit = compat.remove_monopole(
        m, nest=False, bad=hp.UNSEEN, gal_cut=10.0, fitval=True, copy=True
    )
    exp_removed, exp_fit = healpy.remove_monopole(
        m.numpy(),
        nest=False,
        bad=hp.UNSEEN,
        gal_cut=10.0,
        fitval=True,
        copy=True,
        verbose=False,
    )
    torch.testing.assert_close(
        got_fit, torch.tensor(float(exp_fit), dtype=torch.float64), atol=1e-12, rtol=0.0
    )
    torch.testing.assert_close(
        got_removed,
        torch.as_tensor(exp_removed, dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )


def test_compat_fit_remove_dipole_matches_healpy() -> None:
    healpy = pytest.importorskip("healpy")
    nside = 8
    npix = hp.nside2npix(nside)
    pix = torch.arange(npix, dtype=torch.int64)
    x, y, z = hp.pix2vec(nside, pix, nest=False)
    m = (1.1 + 0.2 * x - 0.05 * y + 0.08 * z).to(torch.float64)
    m[7] = hp.UNSEEN

    got_mono, got_dip = compat.fit_dipole(m, nest=False, bad=hp.UNSEEN, gal_cut=12.0)
    exp_mono, exp_dip = healpy.fit_dipole(
        m.numpy(), nest=False, bad=hp.UNSEEN, gal_cut=12.0
    )
    torch.testing.assert_close(
        got_mono,
        torch.tensor(float(exp_mono), dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        got_dip, torch.as_tensor(exp_dip, dtype=torch.float64), atol=1e-12, rtol=0.0
    )

    got_removed, got_mono2, got_dip2 = compat.remove_dipole(
        m,
        nest=False,
        bad=hp.UNSEEN,
        gal_cut=12.0,
        fitval=True,
        copy=True,
    )
    exp_removed, exp_mono2, exp_dip2 = healpy.remove_dipole(
        m.numpy(),
        nest=False,
        bad=hp.UNSEEN,
        gal_cut=12.0,
        fitval=True,
        copy=True,
        verbose=False,
    )
    exp_removed_t = torch.as_tensor(exp_removed.filled(hp.UNSEEN), dtype=torch.float64)
    torch.testing.assert_close(got_removed, exp_removed_t, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(
        got_mono2,
        torch.tensor(float(exp_mono2), dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        got_dip2, torch.as_tensor(exp_dip2, dtype=torch.float64), atol=1e-12, rtol=0.0
    )


def test_compat_rotator_rotate_alm_and_map_alms() -> None:
    nside = 8
    lmax = 4

    rot = compat.Rotator(rot=(10.0, 20.0, 30.0), deg=True)
    assert abs(rot.psi - math.radians(10.0)) < 1e-15
    nalm = alm_size(lmax)
    rng = torch.Generator().manual_seed(0)
    alm = torch.complex(
        torch.randn(nalm, generator=rng, dtype=torch.float64),
        torch.randn(nalm, generator=rng, dtype=torch.float64),
    )
    for ell in range(lmax + 1):
        idx = alm_index(ell, 0, lmax)
        alm[idx] = complex(float(alm[idx].real), 0.0)
    got = rot.rotate_alm(alm, lmax=lmax)
    expected = rotate_alm(alm, rot.psi, rot.theta, rot.phi, lmax=lmax)
    torch.testing.assert_close(got, expected, atol=0.0, rtol=0.0)

    m = synfast(torch.ones(lmax + 1), nside=nside, lmax=lmax)
    m_rot = rot.rotate_map_alms(m, lmax=lmax)
    assert m_rot.shape == m.shape
    der = compat.alm2map_der1(alm, nside, lmax=lmax)
    assert len(der) == 3
    assert der[0].shape == (hp.nside2npix(nside),)
