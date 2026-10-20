import math

import numpy as np
import pytest
import torch
import torchsky.sphere.healpix as healpix_mod

from torchsky.sphere.healpix import (
    ang2pix,
    ang2vec,
    ang2pix_ring,
    angular_distance_deg,
    boundaries,
    vec2ang,
    degrade_pixels,
    degrade_pixel_ranges,
    get_all_neighbours,
    get_interp_val,
    get_interp_weights,
    max_pixrad,
    max_pixel_radius,
    pixel_resolution_to_nside,
    query_box,
    query_circle,
    isnpixok,
    isnsideok,
    nside2order,
    lonlat_to_xyz,
    get_map_size,
    get_nside,
    mask_bad,
    nest_children,
    nest_parent,
    nest2ring,
    nside2npix,
    nside2pixarea,
    nside2resol,
    npix2nside,
    order2nside,
    pixel_ranges_to_pixels,
    pixels_to_pixel_ranges,
    pix2ang,
    pix2ang_ring,
    pix2vec,
    spherical_fourier_features,
    query_circle_vec,
    query_disc,
    query_polygon,
    query_polygon_vec,
    query_strip,
    reorder,
    ring2nest,
    upgrade_pixels,
    upgrade_pixel_ranges,
    ud_grade,
    vec2pix,
    xyz_to_lonlat,
)


def test_healpix_module_is_sphere_owned() -> None:
    assert healpix_mod.ang2pix is ang2pix
    assert healpix_mod.query_disc is query_disc


def _ra_delta_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a - b + 180.0, 360.0) - 180.0


def test_lonlat_xyz_roundtrip() -> None:
    rng = np.random.default_rng(123)
    ra = torch.from_numpy(rng.uniform(0.0, 360.0, size=10000)).to(torch.float64)
    dec = torch.from_numpy(
        np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=10000)))
    ).to(torch.float64)

    x, y, z = lonlat_to_xyz(ra, dec)
    ra2, dec2 = xyz_to_lonlat(x, y, z)

    dra = torch.abs(_ra_delta_deg(ra2, ra))
    ddec = torch.abs(dec2 - dec)
    torch.testing.assert_close(dra, torch.zeros_like(dra), atol=1e-12, rtol=0.0)
    torch.testing.assert_close(ddec, torch.zeros_like(ddec), atol=1e-12, rtol=0.0)


def test_xyz_to_lonlat_rejects_zero_vectors() -> None:
    with pytest.raises(ValueError, match="positive norm"):
        xyz_to_lonlat(torch.tensor([1.0, 0.0]), torch.zeros(2), torch.zeros(2))


def test_angular_distance_known_points() -> None:
    d0 = angular_distance_deg(
        torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
    )
    d1 = angular_distance_deg(
        torch.tensor(0.0), torch.tensor(0.0), torch.tensor(90.0), torch.tensor(0.0)
    )
    d2 = angular_distance_deg(
        torch.tensor(0.0), torch.tensor(0.0), torch.tensor(180.0), torch.tensor(0.0)
    )
    torch.testing.assert_close(
        d0, torch.tensor(0.0, dtype=torch.float64), atol=1e-12, rtol=0.0
    )
    torch.testing.assert_close(
        d1, torch.tensor(90.0, dtype=torch.float64), atol=1e-12, rtol=0.0
    )
    torch.testing.assert_close(
        d2, torch.tensor(180.0, dtype=torch.float64), atol=1e-12, rtol=0.0
    )


def test_spherical_fourier_features_shape_and_grad() -> None:
    ra = torch.linspace(0.0, 359.0, steps=257, dtype=torch.float64, requires_grad=True)
    dec = torch.linspace(
        -80.0, 80.0, steps=257, dtype=torch.float64, requires_grad=True
    )
    feats = spherical_fourier_features(
        ra,
        dec,
        num_frequencies=5,
        min_frequency=0.5,
        base=2.0,
        include_xyz=True,
        include_lonlat=True,
    )
    assert feats.shape == (257, 53)

    loss = feats.square().mean()
    loss.backward()
    assert ra.grad is not None
    assert dec.grad is not None
    assert torch.isfinite(ra.grad).all()
    assert torch.isfinite(dec.grad).all()


def test_nest_parent_children_roundtrip() -> None:
    pix = torch.tensor([0, 1, 7, 15, 31, 127, 1024, 1234567], dtype=torch.int64)
    children = nest_children(pix, levels=2)
    assert children.shape == (pix.numel(), 16)

    parent = nest_parent(children, levels=2)
    expected = pix.unsqueeze(-1).expand_as(parent)
    torch.testing.assert_close(parent, expected)


def test_nest_children_offsets() -> None:
    pix = torch.tensor([11, 42], dtype=torch.int64)
    got = nest_children(pix, levels=1)
    expected = torch.tensor([[44, 45, 46, 47], [168, 169, 170, 171]], dtype=torch.int64)
    torch.testing.assert_close(got, expected)


def test_nside_scalar_helpers() -> None:
    assert nside2npix(8) == 768
    assert order2nside(5) == 32
    assert nside2order(32) == 5
    assert npix2nside(12288) == 32
    assert bool(isnsideok(32))
    assert not bool(isnsideok(12))
    assert bool(isnpixok(12288))
    assert not bool(isnpixok(12345))
    assert bool(isnsideok(1 << 29))
    assert not bool(isnsideok(1 << 30))
    with pytest.raises(ValueError, match=r"2\*\*29"):
        order2nside(30)

    area = nside2pixarea(16)
    area_deg = nside2pixarea(16, degrees=True)
    resol = nside2resol(16)
    resol_arcmin = nside2resol(16, arcmin=True)

    assert area > 0.0
    assert area_deg > 0.0
    assert resol > 0.0
    assert resol_arcmin > 0.0
    assert max_pixrad(16) == max_pixel_radius(16)
    np.testing.assert_allclose(
        area_deg, area * (180.0 / np.pi) ** 2, rtol=0.0, atol=1e-14
    )
    np.testing.assert_allclose(
        resol_arcmin, resol * 60.0 * 180.0 / np.pi, rtol=0.0, atol=1e-12
    )


def test_map_helpers_and_ud_grade_shapes() -> None:
    nside = 8
    npix = 12 * nside * nside
    m = torch.arange(npix, dtype=torch.float64)
    assert get_nside(m) == nside
    assert get_map_size(m) == npix

    bad = m.clone()
    bad[3] = -1.6375e30
    mb = mask_bad(bad)
    assert bool(mb[3].item())
    assert not bool(mb[4].item())

    up = ud_grade(m, nside_out=16, order_in="RING", order_out="RING")
    dn = ud_grade(m, nside_out=4, order_in="RING", order_out="RING")
    assert up.shape == (12 * 16 * 16,)
    assert dn.shape == (12 * 4 * 4,)

    mm = torch.stack([m, m + 1.0], dim=0)
    out = ud_grade(mm, nside_out=4, order_in="RING", order_out="NEST")
    assert out.shape == (2, 12 * 4 * 4)


def test_reorder_last_axis_preserves_leading_shapes_and_layouts() -> None:
    nside = 2
    npix = nside2npix(nside)
    base = torch.arange(3 * 2 * npix, dtype=torch.float64).reshape(3, 2, npix)
    maps = base.transpose(0, 1)

    nested = reorder(maps, r2n=True)
    restored = reorder(nested, n2r=True)

    assert nested.shape == maps.shape
    torch.testing.assert_close(restored, maps)
    empty = reorder(torch.empty((0, 2, npix)), r2n=True)
    assert empty.shape == (0, 2, npix)


def test_ang2pix_pix2ang_compat_wrappers() -> None:
    rng = np.random.default_rng(99)
    ra = torch.from_numpy(rng.uniform(0.0, 360.0, size=4096)).to(torch.float64)
    dec = torch.from_numpy(np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=4096)))).to(
        torch.float64
    )

    theta = torch.deg2rad(90.0 - dec)
    phi = torch.deg2rad(ra)

    pix_a = ang2pix(64, theta, phi, nest=False, lonlat=False)
    pix_b = ang2pix_ring(64, ra, dec)
    torch.testing.assert_close(pix_a, pix_b)

    theta2, phi2 = pix2ang(64, pix_a, nest=False, lonlat=False)
    ra2, dec2 = pix2ang_ring(64, pix_a)
    np.testing.assert_allclose(
        theta2.cpu().numpy(),
        np.deg2rad(90.0 - dec2.cpu().numpy()),
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        phi2.cpu().numpy(), np.deg2rad(ra2.cpu().numpy()), atol=1e-12, rtol=0.0
    )


def test_ang2pix_fixed_wrap_poles_and_transition_boundary() -> None:
    lon = torch.tensor([0.0, 90.0, 180.0, 270.0, 360.0, -90.0], dtype=torch.float64)
    north = torch.full_like(lon, 90.0)
    south = torch.full_like(lon, -90.0)
    transition = torch.full_like(lon, math.degrees(math.asin(2.0 / 3.0)))

    torch.testing.assert_close(
        ang2pix(1, lon, north, lonlat=True),
        torch.tensor([0, 1, 2, 3, 0, 3], dtype=torch.int64),
    )
    torch.testing.assert_close(
        ang2pix(1, lon, south, lonlat=True),
        torch.tensor([8, 9, 10, 11, 8, 11], dtype=torch.int64),
    )
    torch.testing.assert_close(
        ang2pix(2, lon, transition, lonlat=True),
        torch.tensor([4, 6, 8, 10, 4, 10], dtype=torch.int64),
    )
    torch.testing.assert_close(
        ang2pix(2, lon, transition, nest=True, lonlat=True),
        torch.tensor([2, 6, 10, 14, 2, 14], dtype=torch.int64),
    )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_lonlat_conversions_reject_nonfinite_values(bad: float) -> None:
    finite = torch.tensor([0.0], dtype=torch.float64)
    invalid = torch.tensor([bad], dtype=torch.float64)

    with pytest.raises(ValueError, match="finite"):
        ang2pix(1, invalid, finite, lonlat=True)
    with pytest.raises(ValueError, match="finite"):
        ang2pix(1, finite, invalid, nest=True, lonlat=True)
    with pytest.raises(ValueError, match="finite"):
        lonlat_to_xyz(invalid, finite)
    with pytest.raises(ValueError, match="finite"):
        get_interp_weights(1, invalid, finite, lonlat=True)


@pytest.mark.parametrize("latitude", [-90.0001, 90.0001])
def test_lonlat_conversions_reject_latitude_outside_domain(latitude: float) -> None:
    lon = torch.tensor([0.0], dtype=torch.float64)
    lat = torch.tensor([latitude], dtype=torch.float64)

    with pytest.raises(ValueError, match=r"\[-90, 90\]"):
        ang2pix(1, lon, lat, lonlat=True)
    with pytest.raises(ValueError, match=r"\[-90, 90\]"):
        lonlat_to_xyz(lon, lat)
    with pytest.raises(ValueError, match=r"\[-90, 90\]"):
        get_interp_weights(1, lon, lat, lonlat=True)


def test_cpp_torch_bindings_core_healpix_match_python() -> None:
    cpp = pytest.importorskip("torchsky._cpp")
    required = [
        "healpix_ring2nest_torch_cpu",
        "healpix_nest2ring_torch_cpu",
        "healpix_ang2pix_ring_torch_cpu",
        "healpix_ang2pix_nested_torch_cpu",
        "healpix_pix2ang_ring_torch_cpu",
        "healpix_pix2ang_nested_torch_cpu",
    ]
    if not all(hasattr(cpp, name) for name in required):
        pytest.skip("torch-native HEALPix C++ bindings unavailable")

    nside = 64
    rng = np.random.default_rng(321)
    ra = torch.from_numpy(rng.uniform(0.0, 360.0, size=1024)).to(torch.float64)
    dec = torch.from_numpy(np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=1024)))).to(
        torch.float64
    )

    pix_ring_cpp = cpp.healpix_ang2pix_ring_torch_cpu(nside, ra, dec)
    pix_ring_py = ang2pix_ring(nside, ra, dec)
    torch.testing.assert_close(pix_ring_cpp, pix_ring_py)

    pix_nest_cpp = cpp.healpix_ang2pix_nested_torch_cpu(nside, ra, dec)
    pix_nest_py = ang2pix(
        nside, torch.deg2rad(90.0 - dec), torch.deg2rad(ra), nest=True, lonlat=False
    )
    torch.testing.assert_close(pix_nest_cpp, pix_nest_py)

    pix_nest_from_ring_cpp = cpp.healpix_ring2nest_torch_cpu(nside, pix_ring_cpp)
    pix_nest_from_ring_py = ring2nest(nside, pix_ring_cpp)
    torch.testing.assert_close(pix_nest_from_ring_cpp, pix_nest_from_ring_py)
    torch.testing.assert_close(
        cpp.healpix_nest2ring_torch_cpu(nside, pix_nest_cpp),
        nest2ring(nside, pix_nest_cpp),
    )

    ra_r_cpp, dec_r_cpp = cpp.healpix_pix2ang_ring_torch_cpu(nside, pix_ring_cpp)
    ra_r_py, dec_r_py = pix2ang_ring(nside, pix_ring_cpp)
    torch.testing.assert_close(ra_r_cpp, ra_r_py, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(dec_r_cpp, dec_r_py, atol=1e-12, rtol=0.0)

    ra_n_cpp, dec_n_cpp = cpp.healpix_pix2ang_nested_torch_cpu(nside, pix_nest_cpp)
    ra_n_py, dec_n_py = pix2ang(nside, pix_nest_cpp, nest=True, lonlat=True)
    torch.testing.assert_close(ra_n_cpp, ra_n_py, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(dec_n_cpp, dec_n_py, atol=1e-12, rtol=0.0)


def test_cpp_torch_bindings_interp_val_match_python() -> None:
    cpp = pytest.importorskip("torchsky._cpp")
    required = [
        "healpix_get_interp_val_ring_torch_cpu",
        "healpix_get_interp_val_nested_torch_cpu",
    ]
    if not all(hasattr(cpp, name) for name in required):
        pytest.skip("torch-native HEALPix interp C++ bindings unavailable")

    nside = 32
    npix = nside2npix(nside)
    rng = np.random.default_rng(987)
    lon = torch.from_numpy(rng.uniform(0.0, 360.0, size=256)).to(torch.float64)
    lat = torch.from_numpy(np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=256)))).to(
        torch.float64
    )
    m_ring = torch.from_numpy(rng.normal(size=npix)).to(torch.float64)
    m_nest = m_ring[ring2nest(nside, torch.arange(npix, dtype=torch.int64))]

    got_r = cpp.healpix_get_interp_val_ring_torch_cpu(nside, m_ring, lon, lat)
    exp_r = get_interp_val(m_ring, lon, lat, nest=False, lonlat=True)
    torch.testing.assert_close(got_r, exp_r, atol=1e-12, rtol=1e-12)

    got_n = cpp.healpix_get_interp_val_nested_torch_cpu(nside, m_nest, lon, lat)
    exp_n = get_interp_val(m_nest, lon, lat, nest=True, lonlat=True)
    torch.testing.assert_close(got_n, exp_n, atol=1e-12, rtol=1e-12)

    m_stack = torch.stack([m_ring, m_ring * 0.5 + 0.2], dim=0)
    got_stack = cpp.healpix_get_interp_val_ring_torch_cpu(nside, m_stack, lon, lat)
    exp_stack = get_interp_val(m_stack, lon, lat, nest=False, lonlat=True)
    torch.testing.assert_close(got_stack, exp_stack, atol=1e-12, rtol=1e-12)


def test_interp_coordinates_retain_gradients_on_cpu() -> None:
    nside = 8
    lon = torch.tensor([12.3, 87.6], dtype=torch.float64, requires_grad=True)
    lat = torch.tensor([-23.4, 31.2], dtype=torch.float64, requires_grad=True)

    _, weights = get_interp_weights(nside, lon, lat, nest=True, lonlat=True)
    coeff = torch.arange(1, 5, dtype=weights.dtype).unsqueeze(1)
    (weights * coeff).sum().backward()
    assert lon.grad is not None and torch.isfinite(lon.grad).all()
    assert lat.grad is not None and torch.isfinite(lat.grad).all()

    lon2 = lon.detach().clone().requires_grad_(True)
    lat2 = lat.detach().clone().requires_grad_(True)
    values = torch.arange(nside2npix(nside), dtype=torch.float64)
    get_interp_val(values, lon2, lat2, nest=True, lonlat=True).sum().backward()
    assert lon2.grad is not None and torch.isfinite(lon2.grad).all()
    assert lat2.grad is not None and torch.isfinite(lat2.grad).all()


def test_vec2pix_pix2vec_center_roundtrip() -> None:
    nside = 8
    npix = nside2npix(nside)
    for nest in (False, True):
        pix = torch.arange(npix, dtype=torch.int64)
        x, y, z = pix2vec(nside, pix, nest=nest)
        back = vec2pix(nside, x, y, z, nest=nest)
        torch.testing.assert_close(back, pix)


def test_ang2vec_vec2ang_roundtrip() -> None:
    rng = np.random.default_rng(13)
    lon = torch.from_numpy(rng.uniform(0.0, 360.0, size=4096)).to(torch.float64)
    lat = torch.from_numpy(np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=4096)))).to(
        torch.float64
    )
    vec = ang2vec(lon, lat, lonlat=True)
    lon2, lat2 = vec2ang(vec, lonlat=True)

    dlon = torch.abs(_ra_delta_deg(lon2, lon))
    dlat = torch.abs(lat2 - lat)
    torch.testing.assert_close(dlon, torch.zeros_like(dlon), atol=1e-12, rtol=0.0)
    torch.testing.assert_close(dlat, torch.zeros_like(dlat), atol=1e-12, rtol=0.0)


def test_vec2ang_preserves_leading_shape() -> None:
    lon = torch.tensor([[0.0, 45.0], [180.0, 270.0]], dtype=torch.float64)
    lat = torch.tensor([[0.0, 10.0], [-20.0, 30.0]], dtype=torch.float64)
    vectors = ang2vec(lon, lat, lonlat=True)
    lon_out, lat_out = vec2ang(vectors, lonlat=True)
    assert lon_out.shape == lon.shape
    assert lat_out.shape == lat.shape
    torch.testing.assert_close(_ra_delta_deg(lon_out, lon), torch.zeros_like(lon))
    torch.testing.assert_close(lat_out, lat)


def test_boundaries_shapes_and_unit_norm() -> None:
    b_scalar = boundaries(8, 1, step=4, nest=False)
    assert b_scalar.shape == (3, 16)

    pix = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
    b_vec = boundaries(8, pix, step=4, nest=False)
    assert b_vec.shape == (5, 3, 16)

    norms = torch.sqrt((b_vec * b_vec).sum(dim=1))
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-12, rtol=0.0)


def test_resolution_helpers_and_query_shapes() -> None:
    nside = pixel_resolution_to_nside(nside2resol(64))
    assert nside == 64
    assert max_pixel_radius(64, degrees=True) > 0.0

    q1 = query_circle(32, 10.0, -20.0, 5.0, degrees=True, nest=False)
    assert q1.ndim == 1
    assert q1.dtype == torch.int64

    q2 = query_box(32, 350.0, -10.0, 15.0, 10.0, nest=True)
    assert q2.ndim == 1
    assert q2.dtype == torch.int64


def test_query_circle_rejects_vector_centers() -> None:
    with pytest.raises(ValueError, match="scalar center"):
        query_circle(8, torch.tensor([10.0, 20.0]), torch.tensor([0.0, 1.0]), 2.0)


def test_query_vec_helpers_self_consistent() -> None:
    from torchsky.sphere import compat

    nside = 32
    lon = torch.tensor(14.0, dtype=torch.float64)
    lat = torch.tensor(-11.0, dtype=torch.float64)
    x, y, z = lonlat_to_xyz(lon, lat)
    vec = torch.stack([x, y, z]).to(torch.float64)

    got = query_circle_vec(nside, vec, np.deg2rad(4.0), inclusive=False, nest=True)
    torch.testing.assert_close(
        got,
        compat.query_circle_vec(
            nside, vec, np.deg2rad(4.0), inclusive=False, nest=True
        ),
    )
    got_inc = query_circle_vec(nside, vec, np.deg2rad(4.0), inclusive=True, nest=True)
    torch.testing.assert_close(
        got_inc,
        compat.query_circle_vec(nside, vec, np.deg2rad(4.0), inclusive=True, nest=True),
    )

    verts = torch.tensor(
        [[0.8, 0.1, 0.6], [0.7, 0.3, 0.65], [0.6, 0.2, 0.75], [0.75, 0.0, 0.66]],
        dtype=torch.float64,
    )
    verts = verts / torch.linalg.norm(verts, dim=1, keepdim=True)
    got_poly = query_polygon_vec(nside, verts, inclusive=False, nest=False)
    torch.testing.assert_close(
        got_poly,
        compat.query_polygon_vec(nside, verts, inclusive=False, nest=False),
    )


def test_query_disc_cpp_matches_python_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpp = pytest.importorskip("torchsky._cpp")
    if not hasattr(cpp, "healpix_query_disc_torch_cpu"):
        pytest.skip("torch-native HEALPix query_disc C++ binding unavailable")

    nside = 64
    lon = 13.7
    lat = -21.2
    radius_deg = 6.0

    got_ring = query_circle(
        nside, lon, lat, radius_deg, degrees=True, nest=False, inclusive=True
    )
    got_nest = query_circle(
        nside, lon, lat, radius_deg, degrees=True, nest=True, inclusive=False
    )

    monkeypatch.setattr(healpix_mod._ops, "has_fixed_dispatcher", lambda: False)
    exp_ring = query_circle(
        nside, lon, lat, radius_deg, degrees=True, nest=False, inclusive=True
    )
    exp_nest = query_circle(
        nside, lon, lat, radius_deg, degrees=True, nest=True, inclusive=False
    )

    torch.testing.assert_close(torch.sort(got_ring).values, torch.sort(exp_ring).values)
    torch.testing.assert_close(torch.sort(got_nest).values, torch.sort(exp_nest).values)


@pytest.mark.parametrize("nside", [1, 2, 4])
@pytest.mark.parametrize("radius", [0.0, 0.2, math.pi / 2.0, math.pi])
def test_query_disc_matches_bounded_bruteforce_center_oracle(
    nside: int, radius: float
) -> None:
    vec = torch.tensor([2.0, -3.0, 4.0], dtype=torch.float64)
    unit = vec / torch.linalg.vector_norm(vec)
    pix = torch.arange(nside2npix(nside), dtype=torch.int64)
    x, y, z = pix2vec(nside, pix, nest=False)
    dots = x * unit[0] + y * unit[1] + z * unit[2]
    expected = pix[dots >= math.cos(radius)]

    actual = query_disc(nside, vec, radius, nest=False)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("radius", [-1.0, math.pi + 1.0e-12, math.nan, math.inf])
def test_query_disc_rejects_invalid_radius(radius: float) -> None:
    with pytest.raises(ValueError, match=r"finite and in \[0, pi\]"):
        query_disc(1, [1.0, 0.0, 0.0], radius)


@pytest.mark.parametrize(
    "vec",
    [
        [0.0, 0.0, 0.0],
        [math.nan, 0.0, 1.0],
        [math.inf, 0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ],
)
def test_query_disc_rejects_invalid_vector(vec: list[float]) -> None:
    with pytest.raises(ValueError):
        query_disc(1, vec, 0.1)


def test_native_healpix_contracts_reject_invalid_domains() -> None:
    cpp = pytest.importorskip("torchsky._cpp")
    required = [
        "healpix_ang2pix_ring_torch_cpu",
        "healpix_ring2nest_torch_cpu",
        "healpix_query_disc_torch_cpu",
    ]
    if not all(hasattr(cpp, name) for name in required):
        pytest.skip("torch-native HEALPix bindings unavailable")

    empty = torch.empty(0, dtype=torch.int64)
    with pytest.raises(RuntimeError, match=r"2\*\*29"):
        cpp.healpix_ring2nest_torch_cpu(1 << 30, empty)

    finite = torch.tensor([0.0], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="finite"):
        cpp.healpix_ang2pix_ring_torch_cpu(
            1, torch.tensor([math.nan], dtype=torch.float64), finite
        )
    with pytest.raises(RuntimeError, match=r"\[-90, 90\]"):
        cpp.healpix_ang2pix_ring_torch_cpu(
            1, finite, torch.tensor([91.0], dtype=torch.float64)
        )

    good_vec = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    for cos_lim in (math.nan, math.inf, -1.0001, 1.0001):
        with pytest.raises(RuntimeError, match="cos_lim"):
            cpp.healpix_query_disc_torch_cpu(1, good_vec, cos_lim, False)
    with pytest.raises(RuntimeError, match="allocation size overflow"):
        cpp.healpix_query_disc_torch_cpu(1 << 29, good_vec, -1.0, False)
    for vec in (
        torch.tensor([math.nan, 0.0, 1.0], dtype=torch.float64),
        torch.tensor([math.inf, 0.0, 1.0], dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
    ):
        with pytest.raises(RuntimeError):
            cpp.healpix_query_disc_torch_cpu(1, vec, 0.5, False)


def test_query_disc_fact_inclusive_false_is_noop() -> None:
    vec = [1.0, 0.0, 0.0]
    radius = 0.15
    base = query_disc(8, vec, radius, inclusive=False, fact=4, nest=False)
    other = query_disc(8, vec, radius, inclusive=False, fact=8, nest=False)
    torch.testing.assert_close(torch.sort(base).values, torch.sort(other).values)


def test_query_disc_inclusive_fact_supersample_and_nest_power_of_two() -> None:
    nside = 8
    vec = [1.0, 0.0, 0.0]
    radius = 0.25
    # fact!=4 supersamples at nside*fact then degrades to parents.
    fact = 8
    hi = query_disc(
        nside * fact, vec, radius, inclusive=True, fact=4, nest=True
    )
    parent = torch.unique(torch.div(hi, fact * fact, rounding_mode="floor"))
    got = query_disc(nside, vec, radius, inclusive=True, fact=fact, nest=True)
    torch.testing.assert_close(torch.sort(got).values, torch.sort(parent).values)

    with pytest.raises(ValueError, match="power of two"):
        query_disc(nside, vec, radius, inclusive=True, fact=3, nest=True)
    with pytest.raises(ValueError, match="positive integer"):
        query_disc(nside, vec, radius, inclusive=True, fact=0, nest=False)


def test_query_disc_inclusive_fact_ring_via_nest_parent() -> None:
    nside = 8
    vec = [0.0, 0.0, 1.0]
    radius = 0.2
    fact = 2
    nest_pix = query_disc(nside, vec, radius, inclusive=True, fact=fact, nest=True)
    ring_pix = query_disc(nside, vec, radius, inclusive=True, fact=fact, nest=False)
    expected = torch.sort(nest2ring(nside, nest_pix)).values
    torch.testing.assert_close(torch.sort(ring_pix).values, expected)


def test_query_polygon_handles_nonconvex() -> None:
    nside = 8
    # Chevron in lon/lat: the (8, 12) vertex indents the equator-to-pole quad.
    verts = torch.tensor(
        [
            [0.0, 0.0],
            [40.0, 0.0],
            [8.0, 12.0],
            [0.0, 30.0],
        ],
        dtype=torch.float64,
    )
    pix = query_polygon(nside, verts, lonlat=True)
    assert pix.ndim == 1
    assert pix.dtype == torch.int64


def test_query_polygon_accepts_convex_quad() -> None:
    verts = torch.tensor(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ],
        dtype=torch.float64,
    )
    pix = query_polygon(8, verts, lonlat=True)
    assert pix.ndim == 1
    assert pix.numel() > 0


def test_query_disc_cuda_vec_matches_cpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cpp = pytest.importorskip("torchsky._cpp")
    if not hasattr(cpp, "healpix_query_disc_torch_cpu"):
        pytest.skip("torch-native HEALPix query_disc C++ binding unavailable")

    nside = 32
    vec = torch.tensor([0.1, -0.2, 0.97], dtype=torch.float64)
    vec = vec / vec.norm()
    radius = 0.08
    on_cpu = query_disc(nside, vec, radius, nest=False)
    on_gpu = query_disc(nside, vec.cuda(), radius, nest=False)
    torch.testing.assert_close(
        torch.sort(on_cpu).values, torch.sort(on_gpu.cpu()).values
    )


def test_query_disc_mps_vec_matches_cpu() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    cpp = pytest.importorskip("torchsky._cpp")
    if not hasattr(cpp, "healpix_query_disc_torch_cpu"):
        pytest.skip("torch-native HEALPix query_disc C++ binding unavailable")

    nside = 32
    vec = torch.tensor([0.1, -0.2, 0.97], dtype=torch.float32)
    vec = vec / vec.norm()
    radius = 0.08
    on_cpu = query_disc(nside, vec.cpu().double(), radius, nest=False)
    on_mps = query_disc(nside, vec.to("mps"), radius, nest=False)
    torch.testing.assert_close(
        torch.sort(on_cpu).values, torch.sort(on_mps.cpu()).values
    )


def test_query_strip_ring_prune_matches_bruteforce() -> None:
    nside = 4
    t1, t2 = 0.4, 1.2
    got = query_strip(nside, t1, t2, nest=False)
    npix = nside2npix(nside)
    pix = torch.arange(npix, dtype=torch.int64)
    lon, lat = pix2ang(nside, pix, nest=False, lonlat=True)
    theta = torch.deg2rad(90.0 - lat)
    lo = min(t1, t2)
    hi = max(t1, t2)
    exp = pix[(theta >= lo) & (theta <= hi)]
    torch.testing.assert_close(torch.sort(got).values, torch.sort(exp).values)


def test_pixel_ranges_helpers_expected_values() -> None:
    ranges = torch.tensor([[10, 13], [20, 22]], dtype=torch.int64)
    got_excl = pixel_ranges_to_pixels(ranges, inclusive=False)
    got_incl = pixel_ranges_to_pixels(ranges, inclusive=True)
    torch.testing.assert_close(
        got_excl, torch.tensor([10, 11, 12, 20, 21], dtype=torch.int64)
    )
    torch.testing.assert_close(
        got_incl, torch.tensor([10, 11, 12, 13, 20, 21, 22], dtype=torch.int64)
    )

    up = upgrade_pixel_ranges(2, ranges, 8)
    assert up.ndim == 2 and up.shape[1] == 2
    assert up.dtype == torch.int64
    assert up.numel() >= ranges.numel()

    pix = torch.tensor([7, 8, 9, 15, 16, 18], dtype=torch.int64)
    got_ranges = pixels_to_pixel_ranges(pix)
    torch.testing.assert_close(
        got_ranges, torch.tensor([[7, 10], [15, 17], [18, 19]], dtype=torch.int64)
    )


def test_degrade_pixel_ranges_matches_explicit_parent_pixels() -> None:
    ranges = torch.tensor([[0, 3], [6, 10], [20, 21]], dtype=torch.int64)
    degraded = degrade_pixel_ranges(8, ranges, 2)
    child_pixels = pixel_ranges_to_pixels(ranges, inclusive=False)
    parent_pixels = torch.unique(torch.div(child_pixels, 16, rounding_mode="floor"))
    expected = pixels_to_pixel_ranges(parent_pixels)

    torch.testing.assert_close(degraded, expected)


def test_degrade_pixel_ranges_canonicalizes_same_resolution() -> None:
    ranges = torch.tensor([[5, 8], [7, 10], [12, 13]], dtype=torch.int64)
    torch.testing.assert_close(
        degrade_pixel_ranges(4, ranges, 4),
        torch.tensor([[5, 10], [12, 13]], dtype=torch.int64),
    )


def test_upgrade_degrade_pixels_nested_expected_values() -> None:
    parents = torch.tensor([2, 2, 5], dtype=torch.int64)
    upgraded = upgrade_pixels(2, parents, 8, nest=True)
    expected_children = torch.cat(
        [
            torch.arange(2 * 16, 3 * 16, dtype=torch.int64),
            torch.arange(5 * 16, 6 * 16, dtype=torch.int64),
        ]
    )
    torch.testing.assert_close(upgraded, expected_children)

    touched = expected_children[torch.tensor([0, 1, 20, -1])]
    degraded = degrade_pixels(8, touched, 2, nest=True)
    torch.testing.assert_close(degraded, torch.tensor([2, 5], dtype=torch.int64))


def test_upgrade_degrade_pixels_ring_matches_nested_roundtrip() -> None:
    parent_nest = torch.tensor([2, 5], dtype=torch.int64)
    parent_ring = nest2ring(2, parent_nest)

    child_ring = upgrade_pixels(2, parent_ring, 8, nest=False)
    child_nest = torch.sort(ring2nest(8, child_ring)).values
    expected_child_nest = upgrade_pixels(2, parent_nest, 8, nest=True)
    torch.testing.assert_close(child_nest, expected_child_nest)

    degraded_ring = degrade_pixels(8, child_ring[torch.tensor([0, 3, 19, -1])], 2)
    degraded_nest = torch.sort(ring2nest(2, degraded_ring)).values
    torch.testing.assert_close(degraded_nest, parent_nest)


def test_upgrade_degrade_pixels_validate_resolution_direction() -> None:
    pixels = torch.tensor([0, 1], dtype=torch.int64)
    with pytest.raises(ValueError, match="nside_upgrade"):
        upgrade_pixels(8, pixels, 2, nest=True)
    with pytest.raises(ValueError, match="nside_degrade"):
        degrade_pixels(2, pixels, 8, nest=True)
    with pytest.raises(ValueError, match="outside"):
        degrade_pixels(2, torch.tensor([48], dtype=torch.int64), 1, nest=True)


def test_neighbors_and_interp_shapes() -> None:
    nside = 32
    pix = torch.tensor([0, 1, 2, 10, 100], dtype=torch.int64)
    neigh = get_all_neighbours(nside, pix, nest=False)
    assert neigh.shape == (8, pix.numel())
    assert neigh.dtype == torch.int64

    ra = torch.tensor([10.0, 45.0, 180.0], dtype=torch.float64)
    dec = torch.tensor([-20.0, 0.0, 35.0], dtype=torch.float64)
    ip, w = get_interp_weights(nside, ra, dec, nest=False, lonlat=True)
    assert ip.shape == (4, 3)
    assert w.shape == (4, 3)
    torch.testing.assert_close(
        w.sum(dim=0), torch.ones(3, dtype=torch.float64), atol=1e-12, rtol=0.0
    )

    m = torch.arange(12 * nside * nside, dtype=torch.float64)
    vals = get_interp_val(m, ra, dec, nest=False, lonlat=True)
    assert vals.shape == (3,)
    assert vals.dtype == torch.float64


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool])
def test_get_interp_val_promotes_nonfloating_maps(dtype: torch.dtype) -> None:
    nside = 2
    values = torch.arange(nside2npix(nside)).remainder(2).to(dtype)
    sampled = get_interp_val(values, 10.0, -20.0, lonlat=True)
    assert sampled.shape == ()
    assert sampled.dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_lonlat_xyz_cuda_matches_cpu() -> None:
    rng = np.random.default_rng(7)
    ra_cpu = torch.from_numpy(rng.uniform(0.0, 360.0, size=4096)).to(torch.float64)
    dec_cpu = torch.from_numpy(
        np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=4096)))
    ).to(torch.float64)
    ra_gpu = ra_cpu.cuda()
    dec_gpu = dec_cpu.cuda()

    x_cpu, y_cpu, z_cpu = lonlat_to_xyz(ra_cpu, dec_cpu)
    x_gpu, y_gpu, z_gpu = lonlat_to_xyz(ra_gpu, dec_gpu)
    torch.testing.assert_close(x_gpu.cpu(), x_cpu, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(y_gpu.cpu(), y_cpu, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(z_gpu.cpu(), z_cpu, atol=1e-12, rtol=0.0)


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS not available",
)
def test_lonlat_xyz_mps_matches_cpu() -> None:
    rng = np.random.default_rng(8)
    ra_cpu = torch.from_numpy(rng.uniform(0.0, 360.0, size=4096)).to(torch.float64)
    dec_cpu = torch.from_numpy(
        np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=4096)))
    ).to(torch.float64)
    ra_mps = ra_cpu.to(torch.float32).to("mps")
    dec_mps = dec_cpu.to(torch.float32).to("mps")

    x_cpu, y_cpu, z_cpu = lonlat_to_xyz(ra_cpu, dec_cpu)
    x_mps, y_mps, z_mps = lonlat_to_xyz(ra_mps, dec_mps)
    torch.testing.assert_close(
        x_mps.cpu(), x_cpu.to(torch.float32), atol=5e-6, rtol=0.0
    )
    torch.testing.assert_close(
        y_mps.cpu(), y_cpu.to(torch.float32), atol=5e-6, rtol=0.0
    )
    torch.testing.assert_close(
        z_mps.cpu(), z_cpu.to(torch.float32), atol=5e-6, rtol=0.0
    )
