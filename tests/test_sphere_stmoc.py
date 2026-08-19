import pytest
import torch

from torchsky.sphere import MOC, STMOC, healpix as hp
from torchsky.sphere import mjd_to_time_us, nest2uniq, time_us_to_mjd


def _pixel_lonlat(nside: int, pix: int) -> tuple[float, float]:
    lon, lat = hp.pix2ang(nside, pix, nest=True, lonlat=True)
    return float(lon.item()), float(lat.item())


def test_mjd_time_us_roundtrip_identity() -> None:
    mjd = torch.tensor([0.0, 51544.5, 59000.0, 59000.5], dtype=torch.float64)
    back = time_us_to_mjd(mjd_to_time_us(mjd))
    torch.testing.assert_close(back, mjd, atol=1e-12, rtol=0.0)


def test_stmoc_empty_contains_false() -> None:
    stmoc = STMOC.empty(time_depth=8, spatial_depth=4)
    assert stmoc.n_segments == 0
    assert not bool(stmoc.contains(59000.0, 0.0, 0.0).item())


def test_from_spatial_inside_outside_time_and_space() -> None:
    nside = 16
    moc = MOC(nest2uniq(nside, torch.tensor([0], dtype=torch.int64)))
    lon_in, lat_in = _pixel_lonlat(nside, 0)
    lon_out, lat_out = _pixel_lonlat(nside, 100)
    stmoc = STMOC.from_spatial_coverages(59000.0, 59001.0, moc, time_depth=40)
    assert bool(stmoc.contains(59000.5, lon_in, lat_in).item())
    assert not bool(stmoc.contains(58999.0, lon_in, lat_in).item())
    assert not bool(stmoc.contains(59002.0, lon_in, lat_in).item())
    assert not bool(stmoc.contains(59000.5, lon_out, lat_out).item())


def test_from_times_positions_contains_construction_points() -> None:
    nside = 8
    lon0, lat0 = _pixel_lonlat(nside, 3)
    lon1, lat1 = _pixel_lonlat(nside, 20)
    t = torch.tensor([59000.1, 59000.9], dtype=torch.float64)
    lon = torch.tensor([lon0, lon1], dtype=torch.float64)
    lat = torch.tensor([lat0, lat1], dtype=torch.float64)
    stmoc = STMOC.from_times_positions(t, lon, lat, time_depth=32, spatial_depth=3)
    mask = stmoc.contains(t, lon, lat)
    assert bool(mask.all().item())
    assert not bool(stmoc.contains(58000.0, lon0, lat0).item())


def test_query_by_time_spatial_union() -> None:
    nside = 8
    a = MOC(nest2uniq(nside, torch.tensor([0], dtype=torch.int64)))
    b = MOC(nest2uniq(nside, torch.tensor([10], dtype=torch.int64)))
    stmoc = STMOC.from_spatial_coverages(
        torch.tensor([59000.0, 59010.0]),
        torch.tensor([59001.0, 59011.0]),
        (a, b),
        time_depth=40,
    )
    lon_a, lat_a = _pixel_lonlat(nside, 0)
    lon_b, lat_b = _pixel_lonlat(nside, 10)
    window = stmoc.query_by_time(59000.0, 59000.5)
    assert bool(window.contains(lon_a, lat_a).item())
    assert not bool(window.contains(lon_b, lat_b).item())
    both = stmoc.query_by_time(59000.0, 59011.0)
    assert bool(both.contains(lon_a, lat_a).item())
    assert bool(both.contains(lon_b, lat_b).item())


def test_union_and_intersection() -> None:
    nside = 8
    a = MOC(nest2uniq(nside, torch.tensor([0], dtype=torch.int64)))
    b = MOC(nest2uniq(nside, torch.tensor([0, 10], dtype=torch.int64)))
    left = STMOC.from_spatial_coverages(59000.0, 59002.0, a, time_depth=40)
    right = STMOC.from_spatial_coverages(59001.0, 59003.0, b, time_depth=40)
    lon_a, lat_a = _pixel_lonlat(nside, 0)
    lon_b, lat_b = _pixel_lonlat(nside, 10)
    u = left.union(right)
    assert bool(u.contains(59000.5, lon_a, lat_a).item())
    assert bool(u.contains(59002.5, lon_b, lat_b).item())
    inter = left.intersection(right)
    assert bool(inter.contains(59001.5, lon_a, lat_a).item())
    assert not bool(inter.contains(59001.5, lon_b, lat_b).item())
    assert not bool(inter.contains(59000.5, lon_a, lat_a).item())


def test_overlapping_intervals_are_canonicalized() -> None:
    nside = 8
    a = MOC(nest2uniq(nside, torch.tensor([0], dtype=torch.int64)))
    b = MOC(nest2uniq(nside, torch.tensor([10], dtype=torch.int64)))
    stmoc = STMOC.from_spatial_coverages(
        torch.tensor([59000.0, 59000.5]),
        torch.tensor([59001.0, 59001.5]),
        (a, b),
        time_depth=40,
    )
    lon_a, lat_a = _pixel_lonlat(nside, 0)
    lon_b, lat_b = _pixel_lonlat(nside, 10)
    assert bool(stmoc.contains(59000.25, lon_a, lat_a).item())
    assert not bool(stmoc.contains(59000.25, lon_b, lat_b).item())
    assert bool(stmoc.contains(59000.75, lon_a, lat_a).item())
    assert bool(stmoc.contains(59000.75, lon_b, lat_b).item())
    starts = stmoc.time_start_us.tolist()
    ends = stmoc.time_end_us.tolist()
    assert all(ends[i] <= starts[i + 1] for i in range(len(starts) - 1))


def test_fits_roundtrip(tmp_path) -> None:
    nside = 8
    moc = MOC(nest2uniq(nside, torch.tensor([0, 3], dtype=torch.int64)))
    stmoc = STMOC.from_spatial_coverages(59000.0, 59001.0, moc, time_depth=40)
    path = tmp_path / "stmoc.fits"
    stmoc.write_fits(str(path), overwrite=True)
    loaded = STMOC.read_fits(str(path))
    lon, lat = _pixel_lonlat(nside, 0)
    assert loaded.time_depth == stmoc.time_depth
    assert loaded.spatial_depth == stmoc.spatial_depth
    assert loaded.n_segments == stmoc.n_segments
    assert bool(loaded.contains(59000.5, lon, lat).item())
    torch.testing.assert_close(loaded.time_start_us, stmoc.time_start_us)
    torch.testing.assert_close(loaded.time_end_us, stmoc.time_end_us)


def test_empty_fits_roundtrip(tmp_path) -> None:
    stmoc = STMOC.empty(time_depth=4, spatial_depth=2)
    path = tmp_path / "empty_stmoc.fits"
    stmoc.write_fits(str(path), overwrite=True)
    loaded = STMOC.read_fits(str(path))
    assert loaded.n_segments == 0
    assert loaded.time_depth == 4
    assert loaded.spatial_depth == 2


def test_contains_requires_broadcastable_shapes() -> None:
    moc = MOC(nest2uniq(4, torch.tensor([0], dtype=torch.int64)))
    stmoc = STMOC.from_spatial_coverages(59000.0, 59001.0, moc, time_depth=40)
    with pytest.raises(ValueError, match="broadcast"):
        stmoc.contains(
            torch.zeros(2),
            torch.zeros(3),
            torch.zeros(4),
        )


def test_stmoc_ascii_roundtrip() -> None:
    nside = 8
    moc = MOC(nest2uniq(nside, torch.tensor([0, 3], dtype=torch.int64)))
    stmoc = STMOC.from_spatial_coverages(59000.0, 59001.0, moc, time_depth=40)
    text = stmoc.to_ascii()
    assert text.startswith("t40/")
    loaded = STMOC.from_ascii(text)
    lon, lat = _pixel_lonlat(nside, 0)
    assert bool(loaded.contains(59000.5, lon, lat).item())
    assert loaded.to_ascii() == text


def test_stmoc_empty_ascii_roundtrip() -> None:
    stmoc = STMOC.empty(time_depth=4, spatial_depth=2)
    assert stmoc.to_ascii() == ""
    loaded = STMOC.from_ascii("")
    assert loaded.n_segments == 0
