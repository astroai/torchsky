"""Tests for the full-sky plotting surface: gridding and figure conventions."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from torchsky.sphere import healpix  # noqa: E402
from torchsky.sphere.core import lonlat_to_unit_xyz  # noqa: E402
from torchsky.sphere.moc import MOC, nest2uniq  # noqa: E402
from torchsky.viz import allsky  # noqa: E402


def _footprint(nside: int = 16, *, lon: float = 90.0, lat: float = 0.0, radius: float = 30.0):
    """A MOC disc, built from nested pixels as the uniq encoding requires."""
    pixels = healpix.query_disc(
        nside, lonlat_to_unit_xyz(lon, lat), math.radians(radius), nest=True
    )
    return MOC(nest2uniq(nside, pixels)), pixels


@pytest.fixture(scope="module")
def toy_sky():
    nside = 16
    pix = torch.arange(healpix.nside2npix(nside))
    lon, lat = healpix.pix2ang(nside, pix, nest=False, lonlat=True)
    values = 0.5 * torch.exp(-((lat / 15.0) ** 2)) + 1e-3
    return values, lon, lat, nside


def _spot_grid(
    *, spot_lon: float = 90.0, resolution: tuple[int, int] = (72, 36)
) -> allsky.SkyGrid:
    width, height = resolution
    lon = (torch.arange(width, dtype=torch.float64) + 0.5) * (360.0 / width) - 180.0
    lat = (torch.arange(height, dtype=torch.float64) + 0.5) * (180.0 / height) - 90.0
    grid_lon, grid_lat = torch.meshgrid(lon, lat, indexing="xy")
    delta = torch.remainder(grid_lon - spot_lon + 180.0, 360.0) - 180.0
    data = torch.exp(-(delta**2) / 20.0 - (grid_lat**2) / 20.0)
    return allsky.SkyGrid(
        lon_edges=torch.linspace(-180.0, 180.0, width + 1, dtype=torch.float64),
        lat_edges=torch.linspace(-90.0, 90.0, height + 1, dtype=torch.float64),
        data=data,
        counts=torch.ones((height, width), dtype=torch.int64),
        statistic="nearest",
        sampling="resampled",
    )


def test_sky_grid_bins_points_and_keeps_empty_cells_empty() -> None:
    lon = torch.tensor([-179.9, -179.5, 0.0, 179.9])
    lat = torch.tensor([-89.9, -89.5, 0.0, 89.9])
    values = torch.tensor([1.0, 3.0, 5.0, 7.0])

    grid = allsky.sky_grid(lon, lat, values, resolution=(36, 18))

    assert grid.data.shape == (18, 36)
    assert grid.sampling == "binned"
    assert int(grid.counts[0, 0]) == 2
    torch.testing.assert_close(
        grid.data[0, 0], torch.tensor(2.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        grid.data[9, 18], torch.tensor(5.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        grid.data[17, 35], torch.tensor(7.0, dtype=torch.float64)
    )
    assert torch.isnan(grid.data[5, 5])


def test_sky_grid_drops_non_finite_samples() -> None:
    lon = torch.tensor([0.0, 0.0, 10.0])
    lat = torch.tensor([0.0, 0.0, 0.0])
    values = torch.tensor([2.0, float("nan"), float("nan")])

    grid = allsky.sky_grid(lon, lat, values, resolution=(36, 18))

    torch.testing.assert_close(
        grid.data[9, 18], torch.tensor(2.0, dtype=torch.float64)
    )
    assert torch.isnan(grid.data[9, 19])
    assert int(grid.counts.sum()) == 1


def test_dense_map_is_resampled_without_gaps(toy_sky) -> None:
    values, _, _, nside = toy_sky
    grid = allsky.healpix_sky_grid(values, nside=nside)

    assert grid.sampling == "resampled"
    assert grid.resolution == allsky._auto_resolution(nside)
    assert not bool(torch.isnan(grid.data).any())


def test_dense_resample_does_not_scan_every_pixel(monkeypatch, toy_sky) -> None:
    """The dense path costs O(grid), so it must not walk the map pixel by pixel."""
    values, _, _, nside = toy_sky

    def _boom(*args, **kwargs):
        raise AssertionError("dense resampling must not scan every pixel")

    monkeypatch.setattr(healpix, "pix2ang", _boom)
    grid = allsky.healpix_sky_grid(values, nside=nside, resolution=(32, 16))
    assert grid.sampling == "resampled"


def test_sparse_map_is_binned_and_can_force_resample_error() -> None:
    nside = 8
    pixels = torch.tensor([0, 1, 5])
    values = torch.tensor([1.0, 2.0, 3.0])

    grid = allsky.healpix_sky_grid(
        values, nside=nside, pixels=pixels, resolution=(64, 32)
    )
    assert grid.sampling == "binned"
    assert int(grid.counts.sum()) == 3
    assert bool(torch.isnan(grid.data).any())

    with pytest.raises(ValueError, match="sparse maps must be binned"):
        allsky.healpix_sky_grid(
            values, nside=nside, pixels=pixels, resample=True
        )


def test_healpix_grid_rejects_wrong_shapes(toy_sky) -> None:
    values, _, _, nside = toy_sky
    with pytest.raises(ValueError, match="needs"):
        allsky.healpix_sky_grid(values[:-1], nside=nside)
    with pytest.raises(ValueError, match="select a band"):
        allsky.healpix_sky_grid(values.reshape(2, -1), nside=nside)
    with pytest.raises(ValueError, match="nside is required"):
        allsky.healpix_sky_grid(values)
    with pytest.raises(ValueError, match="disagree"):
        allsky.healpix_sky_grid(
            values[:4], nside=nside, pixels=torch.arange(5)
        )


def test_plot_uses_astronomy_ticks_and_colorbar(toy_sky) -> None:
    values, _, _, nside = toy_sky

    ax = allsky.plot_healpix_sky(
        values,
        nside=nside,
        projection="mollweide",
        theme="dark",
        unit="counts s$^{-1}$ deg$^{-2}$",
        title="Toy sky",
        subtitle="Synthetic band",
    )

    assert type(ax).__name__ == "MollweideAxes"
    assert len(ax.collections) == 1
    # The meridian tick labels are suppressed in favour of the RA row below.
    assert all(tick.get_text() == "" for tick in ax.get_xticklabels())
    # RA increases to the left: the leftmost meridian is 10h, the rightmost 14h.
    ra_labels = [text.get_text() for text in ax.texts]
    assert ra_labels[0] == "10$^h$"
    assert ra_labels[-1] == "14$^h$"
    assert [tick.get_text() for tick in ax.get_yticklabels()] == [
        "\u221260\u00b0",
        "\u221230\u00b0",
        "0\u00b0",
        "30\u00b0",
        "60\u00b0",
    ]
    colorbar_ax = ax.figure.axes[1]
    captions = [text.get_text() for text in colorbar_ax.texts]
    assert "counts s$^{-1}$ deg$^{-2}$" in captions
    assert "ASINH" in captions
    assert any("Toy sky" == text.get_text() for text in ax.figure.texts)
    assert any("nearest HEALPix pixel" in text.get_text() for text in ax.figure.texts)


def test_mesh_is_mirrored_for_the_ra_convention() -> None:
    grid = _spot_grid(spot_lon=90.0)
    width = grid.data.shape[1]

    flipped = allsky.plot_sky_grid(
        grid, projection="aitoff", ra_increases_left=True, colorbar=False
    )
    data = np.ma.filled(np.asarray(flipped.collections[0].get_array()), 0.0)
    assert int(np.argmax(data[data.shape[0] // 2])) < width // 2

    direct = allsky.plot_sky_grid(
        grid, projection="aitoff", ra_increases_left=False, colorbar=False
    )
    data = np.ma.filled(np.asarray(direct.collections[0].get_array()), 0.0)
    assert int(np.argmax(data[data.shape[0] // 2])) > width // 2


def test_log_stretch_masks_non_positive_and_rejects_all_non_positive() -> None:
    grid = _spot_grid()
    values = grid.data.clone()
    values[0, :] = 0.0
    grid = allsky.SkyGrid(
        lon_edges=grid.lon_edges,
        lat_edges=grid.lat_edges,
        data=values,
        counts=grid.counts,
    )

    ax = allsky.plot_sky_grid(grid, projection="mollweide", stretch="log")
    mesh = np.asarray(ax.collections[0].get_array())
    assert np.isnan(np.asarray(np.ma.filled(mesh, np.nan))[0]).all()

    zeros = allsky.SkyGrid(
        lon_edges=grid.lon_edges,
        lat_edges=grid.lat_edges,
        data=torch.zeros_like(values),
        counts=grid.counts,
    )
    with pytest.raises(ValueError, match="positive"):
        allsky.plot_sky_grid(zeros, stretch="log")


def test_moc_and_catalog_overlays_get_a_legend() -> None:
    grid = _spot_grid()
    moc, _ = _footprint()

    ax = allsky.plot_sky_grid(
        grid,
        projection="hammer",
        moc=moc,
        moc_label="Coverage",
        points=(torch.tensor([80.0, 100.0]), torch.tensor([0.0, 5.0])),
        point_label="Sources",
    )

    legend = ax.figure.legends[0]
    texts = [text.get_text() for text in legend.get_texts()]
    assert "Coverage" in texts
    assert "Sources" in texts
    # A proxy handle carries the entry; no auto-generated artist labels leak in.
    assert not any(text.startswith("_") for text in texts)


def test_footprint_outline_is_a_single_closed_loop() -> None:
    moc, _ = _footprint()
    loops = moc.boundary()
    assert len(loops) == 1
    loop = loops[0]
    torch.testing.assert_close(loop[0], loop[-1], atol=1e-9, rtol=0)


def test_polar_projection_uses_a_polar_axes_with_hours_around_the_rim() -> None:
    grid = _spot_grid()
    ax = allsky.plot_sky_grid(grid, projection="polar", colorbar=False)

    assert type(ax).__name__ == "PolarAxes"
    # RA=0 at the top, increasing counter-clockwise for the left-handed default.
    assert math.isclose(ax.get_theta_offset(), math.pi / 2.0, abs_tol=1e-9)
    assert ax.get_theta_direction() == 1
    labels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert labels[0] == "0$^h$" and labels[-1] == "22$^h$"
    # Five declination rings, three labels: 60N, equator, 60S.
    assert [tick.get_text() for tick in ax.get_yticklabels()] == [
        "60\u00b0",
        "",
        "0\u00b0",
        "",
        "\u221260\u00b0",
    ]
    assert len(ax.get_yticks()) == 5


def test_polar_footprint_never_wraps_across_the_24h_seam() -> None:
    grid = _spot_grid()
    moc, _ = _footprint(nside=16, lon=10.0, lat=40.0, radius=25.0)
    ax = allsky.plot_sky_grid(
        grid, projection="polar", colorbar=False, moc=moc, moc_label="Coverage"
    )

    assert ax.lines, "the footprint outline should be drawn"
    for line in ax.lines:
        theta, _radius = line.get_data()
        if len(theta) < 2:
            continue
        steps = np.abs(np.diff(np.asarray(theta, dtype=float))) % (2.0 * np.pi)
        # A loop that straddles the seam must be split, not drawn the long way.
        assert float(steps.max(initial=0.0)) <= np.pi


def test_ra_labels_sit_below_the_map_and_follow_the_meridians(toy_sky) -> None:
    values, _, _, nside = toy_sky
    ax = allsky.plot_healpix_sky(
        values, nside=nside, projection="mollweide", colorbar=False
    )

    labels = ax.texts
    assert len(labels) == 11
    for text in labels:
        x_frac, y_frac = text.get_position()
        assert y_frac < 0.0, "RA labels belong under the map, not on the equator"
        hours = float(text.get_text().split("$")[0])
        # RA grows to the left: the label under display fraction x is 12h at the
        # centre and falls to 0h/24h at the edges.
        assert math.isclose((12.0 - 24.0 * x_frac) % 24.0, hours, abs_tol=1e-6)
    # The outermost meridians are labelled at the edges of the projection.
    xs = sorted(text.get_position()[0] for text in labels)
    assert xs[0] < 0.1 and xs[-1] > 0.9


def test_legend_clears_the_colourbar_tick_labels() -> None:
    grid = _spot_grid()
    moc, _ = _footprint()
    ax = allsky.plot_sky_grid(
        grid,
        projection="polar",
        moc=moc,
        moc_label="Coverage",
        points=(torch.tensor([80.0]), torch.tensor([0.0])),
        point_label="Sources",
        unit="counts",
    )
    fig = ax.figure
    fig.canvas.draw()

    legend = fig.legends[0].get_window_extent()
    tick_tops = [
        label.get_window_extent().y0 for label in fig.axes[1].get_xticklabels()
    ]
    assert legend.y1 < min(tick_tops), "legend overlaps the colour bar ticks"
    assert legend.y0 > 0.0


def test_polar_raster_averages_the_cap_instead_of_converging_columns() -> None:
    grid = _spot_grid()
    theta, radius, raster = allsky._polar_raster(grid)

    assert raster.shape == grid.data.shape
    assert float(radius[0]) == 0.0 and float(radius[-1]) == 180.0
    assert float(theta[0]) == -180.0 and float(theta[-1]) == 180.0
    # Every azimuth of the innermost ring is the pole, so they must agree; one
    # value per column is what keeps the centre from drawing as a starburst.
    first = raster[0]
    assert bool(torch.allclose(first, first[0].expand_as(first)))


def test_polar_mesh_draws_the_resampled_raster() -> None:
    grid = _spot_grid()
    ax = allsky.plot_sky_grid(grid, projection="polar", colorbar=False)

    mesh = np.asarray(np.ma.filled(ax.collections[0].get_array(), np.nan))
    assert mesh.shape == grid.data.shape
    assert np.allclose(mesh[0], mesh[0, 0])


def test_legend_gets_its_own_band_under_every_projection() -> None:
    grid = _spot_grid()
    moc, _ = _footprint()
    for projection in ("mollweide", "hammer", "aitoff"):
        ax = allsky.plot_sky_grid(
            grid,
            projection=projection,
            moc=moc,
            moc_label="Coverage",
            points=(torch.tensor([80.0]), torch.tensor([0.0])),
            point_label="Sources",
            unit="counts",
        )
        fig = ax.figure
        fig.canvas.draw()
        legend = fig.legends[0].get_window_extent()
        assert legend.y1 <= ax.bbox.y0 + 1.0, (
            f"{projection} legend overlaps the map"
        )


def test_colourbar_tick_labels_are_evenly_spaced_and_do_not_overlap() -> None:
    grid = _spot_grid()
    ax = allsky.plot_sky_grid(
        grid, projection="mollweide", stretch="asinh", unit="counts"
    )
    fig = ax.figure
    fig.canvas.draw()
    ticks = [label for label in fig.axes[1].get_xticklabels() if label.get_text()]
    texts = [label.get_text() for label in ticks]

    assert len(texts) == 5
    # Two significant digits, no exponent soup on a slide-sized bar.
    assert all(len(text.replace("0.", "").lstrip("0")) <= 3 for text in texts)
    boxes = sorted((label.get_window_extent() for label in ticks), key=lambda b: b.x0)
    for left, right in zip(boxes, boxes[1:], strict=False):
        assert left.x1 < right.x0


def _rescaled_grid(offset: float, span: float) -> allsky.SkyGrid:
    grid = _spot_grid()
    return allsky.SkyGrid(
        lon_edges=grid.lon_edges,
        lat_edges=grid.lat_edges,
        data=offset + span * grid.data,
        counts=grid.counts,
    )


def test_log_colourbar_uses_decade_ticks_when_the_range_allows() -> None:
    grid = _rescaled_grid(0.01, 0.99)  # two decades, 1e-2 .. 1e0
    ax = allsky.plot_sky_grid(
        grid, projection="hammer", stretch="log", unit="counts", vmin=0.01, vmax=1.0
    )
    fig = ax.figure
    fig.canvas.draw()
    texts = [
        label.get_text()
        for label in fig.axes[1].get_xticklabels()
        if label.get_text()
    ]

    assert len(texts) == 3
    assert all("10^" in text for text in texts)


def test_log_colourbar_falls_back_to_even_fractions_inside_a_narrow_range() -> None:
    grid = _rescaled_grid(4.0, 5.0)  # no decade falls inside 4 .. 9
    ax = allsky.plot_sky_grid(
        grid, projection="hammer", stretch="log", unit="counts", vmin=4.0, vmax=9.0
    )
    fig = ax.figure
    fig.canvas.draw()
    texts = [
        label.get_text()
        for label in fig.axes[1].get_xticklabels()
        if label.get_text()
    ]

    # A single decade tick would leave the bar almost unlabelled, so the ticks
    # move to evenly spaced positions along it.
    assert len(texts) == 5
    positions = [float(text) for text in texts]
    assert positions == sorted(positions)
    assert 4.0 <= positions[0] < positions[-1] <= 9.0


def test_drawing_only_a_footprint_skips_the_mesh() -> None:
    ax = allsky.plot_sky_grid(None, projection="mollweide", colorbar=False)
    assert len(ax.collections) == 0
    assert type(ax).__name__ == "MollweideAxes"


def test_unknown_projection_is_rejected(toy_sky) -> None:
    values, _, _, nside = toy_sky
    with pytest.raises(ValueError, match="unknown projection"):
        allsky.plot_healpix_sky(values, nside=nside, projection="gnomonic")


@pytest.mark.parametrize("theme", ["dark", "paper"])
def test_saved_figure_uses_theme_background(tmp_path: Path, theme: str) -> None:
    grid = _spot_grid()
    path = tmp_path / f"sky_{theme}.png"

    allsky.plot_sky_grid(
        grid,
        projection="mollweide",
        theme=theme,
        title="Saved figure",
        save=path,
        dpi=110,
    )

    assert path.is_file()
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    corner = rgb[4, 4]
    if theme == "dark":
        assert corner.max() < 40
    else:
        assert corner.min() > 200
    dpi = Image.open(path).info.get("dpi")
    assert dpi is not None and abs(float(dpi[0]) - 110.0) < 1.5


def test_plotting_requires_matplotlib(monkeypatch, toy_sky) -> None:
    values, _, _, nside = toy_sky
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(ImportError, match="Matplotlib is required"):
        allsky.plot_healpix_sky(values, nside=nside)
