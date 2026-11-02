import pytest
import torch

from torchsky.sphere import MOC, STMOC, healpix as hp
from torchsky.sphere import mjd_to_time_us, nest2uniq, time_us_to_mjd


def _pixel_lonlat(nside: int, pix: int) -> tuple[float, float]:
    lon, lat = hp.pix2ang(nside, pix, nest=True, lonlat=True)
    return float(lon.item()), float(lat.item())


def _brute_contains(
    stmoc: STMOC,
    time_mjd: torch.Tensor,
    lon_deg: torch.Tensor,
    lat_deg: torch.Tensor,
) -> torch.Tensor:
    time_mjd, lon_deg, lat_deg = torch.broadcast_tensors(time_mjd, lon_deg, lat_deg)
    time_us = mjd_to_time_us(time_mjd.reshape(-1))
    lon = lon_deg.reshape(-1)
    lat = lat_deg.reshape(-1)
    result = torch.zeros(time_us.numel(), dtype=torch.bool, device=time_us.device)
    for start, end, moc in zip(
        stmoc.time_start_us.tolist(),
        stmoc.time_end_us.tolist(),
        stmoc.space,
        strict=True,
    ):
        hit = (time_us >= start) & (time_us < end)
        indices = torch.nonzero(hit, as_tuple=False).reshape(-1)
        if indices.numel() > 0:
            result[indices] |= moc.contains(lon[indices], lat[indices])
    return result.reshape(time_mjd.shape)


def _device_or_skip(device_type: str) -> torch.device:
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device_type == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    return torch.device(device_type)


def test_mjd_time_us_roundtrip_identity() -> None:
    mjd = torch.tensor([0.0, 51544.5, 59000.0, 59000.5], dtype=torch.float64)
    back = time_us_to_mjd(mjd_to_time_us(mjd))
    torch.testing.assert_close(back, mjd, atol=1e-12, rtol=0.0)
    assert int(mjd_to_time_us(59000.5).item()) == int(mjd_to_time_us(mjd[-1]).item())


def test_stmoc_empty_contains_false() -> None:
    stmoc = STMOC.empty(time_depth=8, spatial_depth=4)
    assert stmoc.n_segments == 0
    assert not bool(stmoc.contains(59000.0, 0.0, 0.0).item())


def test_overlapping_empty_spatial_segments_contain_nothing() -> None:
    empty = MOC(torch.empty(0, dtype=torch.int64))
    base = int(mjd_to_time_us(59000.0).item())
    stmoc = STMOC(
        torch.tensor([base, base + 1_000_000]),
        torch.tensor([base + 3_000_000, base + 4_000_000]),
        (empty, empty),
        61,
        0,
    )
    time_mjd = time_us_to_mjd(base + torch.arange(80, dtype=torch.int64) * 50_000)
    assert not bool(stmoc.contains(time_mjd, 0.0, 0.0).any())


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


def test_difference() -> None:
    nside = 8
    a = MOC(nest2uniq(nside, torch.tensor([0], dtype=torch.int64)))
    b = MOC(nest2uniq(nside, torch.tensor([0, 10], dtype=torch.int64)))
    left = STMOC.from_spatial_coverages(59000.0, 59002.0, a, time_depth=40)
    right = STMOC.from_spatial_coverages(59001.0, 59003.0, b, time_depth=40)
    lon_a, lat_a = _pixel_lonlat(nside, 0)
    lon_b, lat_b = _pixel_lonlat(nside, 10)
    diff = left.difference(right)
    # Only the non-overlapping early interval keeps pixel 0.
    assert bool(diff.contains(59000.5, lon_a, lat_a).item())
    assert not bool(diff.contains(59001.5, lon_a, lat_a).item())
    assert not bool(diff.contains(59002.5, lon_b, lat_b).item())
    # Spatial-only subtract when times fully overlap.
    wide = STMOC.from_spatial_coverages(59000.0, 59002.0, b, time_depth=40)
    narrow = STMOC.from_spatial_coverages(59000.0, 59002.0, a, time_depth=40)
    spatial_diff = wide.difference(narrow)
    assert not bool(spatial_diff.contains(59001.0, lon_a, lat_a).item())
    assert bool(spatial_diff.contains(59001.0, lon_b, lat_b).item())


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


def test_contains_randomized_brute_force_parity_with_broadcasting() -> None:
    generator = torch.Generator().manual_seed(7104)
    nside = 8
    base = int(mjd_to_time_us(59000.0).item())
    starts = base + torch.arange(24, dtype=torch.int64) * 4_000_000
    ends = starts + torch.randint(
        1_000_000, 3_500_000, (24,), generator=generator, dtype=torch.int64
    )
    pixels = torch.randint(0, 12 * nside * nside, (24,), generator=generator)
    spaces = tuple(
        MOC(nest2uniq(nside, torch.tensor([int(pixel)]))) for pixel in pixels
    )
    permutation = torch.randperm(24, generator=generator)
    stmoc = STMOC(
        starts[permutation],
        ends[permutation],
        tuple(spaces[int(index)] for index in permutation),
        61,
        3,
    )
    query_us = base + torch.randint(
        -2_000_000, 98_000_000, (37, 1), generator=generator, dtype=torch.int64
    )
    query_pixels = torch.randint(0, 12 * nside * nside, (1, 19), generator=generator)
    lon, lat = hp.pix2ang(nside, query_pixels, nest=True, lonlat=True)
    time_mjd = time_us_to_mjd(query_us)
    expected = _brute_contains(stmoc, time_mjd, lon, lat)
    torch.testing.assert_close(stmoc.contains(time_mjd, lon, lat), expected)


def test_contains_half_open_boundaries_and_query_overlap_boundaries() -> None:
    nside = 8
    a = MOC(nest2uniq(nside, torch.tensor([0], dtype=torch.int64)))
    b = MOC(nest2uniq(nside, torch.tensor([100], dtype=torch.int64)))
    base = int(mjd_to_time_us(59000.0).item())
    step = 86_400_000_000
    stmoc = STMOC(
        torch.tensor([base, base + step]),
        torch.tensor([base + step, base + 2 * step]),
        (a, b),
        61,
        3,
    )
    lon_a, lat_a = _pixel_lonlat(nside, 0)
    boundary_times = time_us_to_mjd(
        torch.tensor([base, base + step // 2, base + step], dtype=torch.int64)
    )
    assert stmoc.contains(boundary_times, lon_a, lat_a).tolist() == [True, True, False]
    first = stmoc.query_by_time(
        float(time_us_to_mjd(torch.tensor(base)).item()),
        float(time_us_to_mjd(torch.tensor(base + step)).item()),
    )
    second = stmoc.query_by_time(
        float(time_us_to_mjd(torch.tensor(base + step)).item()),
        float(time_us_to_mjd(torch.tensor(base + 2 * step)).item()),
    )
    assert first == a
    assert second == b


def test_overlapping_unsorted_segments_and_set_algebra_match_brute_force() -> None:
    generator = torch.Generator().manual_seed(98)
    nside = 8
    base = int(mjd_to_time_us(59000.0).item())
    step = 10_000_000
    pixels_a = (2, 11, 31, 52)
    pixels_b = (7, 11, 44)
    spaces_a = tuple(
        MOC(nest2uniq(nside, torch.tensor([pixel], dtype=torch.int64)))
        for pixel in pixels_a
    )
    spaces_b = tuple(
        MOC(nest2uniq(nside, torch.tensor([pixel], dtype=torch.int64)))
        for pixel in pixels_b
    )
    a = STMOC(
        base + torch.tensor([20, 0, 10, 5]) * step,
        base + torch.tensor([35, 18, 30, 14]) * step,
        spaces_a,
        61,
        3,
    )
    b = STMOC(
        base + torch.tensor([12, 28, 3]) * step,
        base + torch.tensor([22, 40, 9]) * step,
        spaces_b,
        61,
        3,
    )
    query_us = base + torch.randint(
        0, 410_000_000, (300,), generator=generator, dtype=torch.int64
    )
    query_pixels = torch.randint(0, 12 * nside * nside, (300,), generator=generator)
    lon, lat = hp.pix2ang(nside, query_pixels, nest=True, lonlat=True)
    time_mjd = time_us_to_mjd(query_us)
    in_a = _brute_contains(a, time_mjd, lon, lat)
    in_b = _brute_contains(b, time_mjd, lon, lat)
    torch.testing.assert_close(a.contains(time_mjd, lon, lat), in_a)
    torch.testing.assert_close(b.contains(time_mjd, lon, lat), in_b)
    torch.testing.assert_close(a.union(b).contains(time_mjd, lon, lat), in_a | in_b)
    torch.testing.assert_close(
        a.intersection(b).contains(time_mjd, lon, lat), in_a & in_b
    )
    torch.testing.assert_close(
        a.difference(b).contains(time_mjd, lon, lat), in_a & ~in_b
    )
    window = a.query_by_time(
        float(time_us_to_mjd(torch.tensor(base + 9 * step)).item()),
        float(time_us_to_mjd(torch.tensor(base + 10 * step)).item()),
    )
    assert window == spaces_a[1].union(spaces_a[3])


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps"])
def test_contains_noncontiguous_inputs_preserve_device_and_dtype(
    device_type: str,
) -> None:
    device = _device_or_skip(device_type)
    nside = 8
    moc = MOC(nest2uniq(nside, torch.tensor([0], dtype=torch.int64)))
    stmoc = STMOC.from_spatial_coverages(59000.0, 59001.0, moc, time_depth=40)
    lon_in, lat_in = _pixel_lonlat(nside, 0)
    dtype = torch.float32 if device_type == "mps" else torch.float64
    time_mjd = torch.full((16,), 59000.5, dtype=dtype, device=device)[::2]
    lon = torch.full((16,), lon_in, dtype=dtype, device=device)[::2]
    lat = torch.full((16,), lat_in, dtype=dtype, device=device)[::2]
    assert not time_mjd.is_contiguous()
    result = stmoc.contains(time_mjd, lon, lat)
    assert result.device.type == device.type
    assert result.dtype == torch.bool
    assert bool(result.all())


def test_large_segment_and_query_candidate_scaling(monkeypatch) -> None:
    n_segments = 4096
    n_queries = 16_384
    nside = 8
    moc = MOC(nest2uniq(nside, torch.tensor([0], dtype=torch.int64)))
    calls = 0
    original_contains = moc.contains

    def counted_contains(lon_deg, lat_deg):
        nonlocal calls
        calls += 1
        return original_contains(lon_deg, lat_deg)

    monkeypatch.setattr(moc, "contains", counted_contains)
    base = int(mjd_to_time_us(59000.0).item())
    step = 2_000_000
    starts = base + torch.arange(n_segments, dtype=torch.int64) * step
    stmoc = STMOC(
        starts,
        starts + step,
        tuple(moc for _ in range(n_segments)),
        61,
        3,
    )
    query_segments = torch.arange(n_queries - 1, -1, -1) % n_segments
    time_mjd = time_us_to_mjd(starts[query_segments] + step // 2)
    lon_in, lat_in = _pixel_lonlat(nside, 0)
    result = stmoc.contains(time_mjd, lon_in, lat_in)
    assert bool(result.all())
    assert calls == 1


def test_large_temporal_boolean_sweep_scaling() -> None:
    n_segments = 5000
    nside = 8
    moc = MOC(nest2uniq(nside, torch.tensor([0], dtype=torch.int64)))
    base = int(mjd_to_time_us(59000.0).item())
    step = 2_000_000
    starts = base + torch.arange(n_segments, dtype=torch.int64) * 2 * step
    left = STMOC(
        starts,
        starts + step,
        tuple(moc for _ in range(n_segments)),
        61,
        3,
    )
    right = STMOC(
        starts + step,
        starts + 2 * step,
        tuple(moc for _ in range(n_segments)),
        61,
        3,
    )
    combined = left.union(right)
    assert combined.n_segments == 1
    assert combined.time_start_us.tolist() == [base]
    assert combined.time_end_us.tolist() == [base + n_segments * 2 * step]


def test_large_narrow_time_query_selects_only_overlaps() -> None:
    n_segments = 50_000
    nside = 8
    moc = MOC(nest2uniq(nside, torch.tensor([0], dtype=torch.int64)))
    base = int(mjd_to_time_us(59000.0).item())
    step = 2_000_000
    starts = base + torch.arange(n_segments, dtype=torch.int64) * step
    stmoc = STMOC(
        starts,
        starts + step,
        tuple(moc for _ in range(n_segments)),
        61,
        3,
    )
    target = 31_337
    result = stmoc.query_by_time(
        float(time_us_to_mjd(starts[target] + 1).item()),
        float(time_us_to_mjd(starts[target] + step - 1).item()),
    )
    assert result == moc
