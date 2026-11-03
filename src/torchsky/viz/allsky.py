"""Full-sky plotting for HEALPix maps, footprints and catalogs.

The pipeline is tensor-native and linear in the number of samples: pixel centres
are converted to ``(lon, lat)`` in bounded chunks, binned onto a regular
longitude/latitude grid with ``index_add_``, and drawn once with ``pcolormesh``
on a matplotlib all-sky projection. Nothing here depends on healpy.

Longitude convention: the grid is stored with geographic longitude in
``[-180, 180]`` and right ascension is the longitude, so RA grows to the left by
default -- the convention every astronomy figure uses.

Projections:

* ``mollweide``, ``aitoff`` and ``hammer`` are full-sky, 2:1 maps.
* ``polar`` is a north-pole view on a matplotlib polar axes: colatitude is the
  radius, so the whole sphere fits in one disc and declination reads as the
  distance from the centre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
from torch import Tensor

from . import theme as _theme
from .palettes import matplotlib_cmap
from ..device import _to

SkyProjection = Literal["mollweide", "aitoff", "hammer", "polar"]
Stretch = Literal["linear", "log", "asinh"]
Statistic = Literal["mean", "sum"]

#: Projections drawn as a 2:1 rectangle. ``polar`` is square and handled apart.
_GEO_ASPECT: dict[str, float] = {"mollweide": 2.0, "aitoff": 2.0, "hammer": 2.0}
_SKY_PROJECTIONS = frozenset({*_GEO_ASPECT, "polar"})
_CHUNK = 1 << 22
_LIMIT_SAMPLE = 1 << 20

# Layout, in inches, measured from the figure edges.
_TITLE_IN = 1.02
_SUBTITLE_IN = 0.30
_FOOTNOTE_IN = 0.52
_CBAR_H_IN = 0.14
_CBAR_GAP_IN = 0.62
_LEGEND_IN = 0.40
_MARGIN_IN = 0.45


@dataclass(frozen=True)
class SkyGrid:
    """A regular longitude/latitude grid of values covering the whole sphere.

    ``lon_edges`` span ``[-180, 180]`` degrees and ``lat_edges`` span
    ``[-90, 90]``; ``data`` is ``(len(lat_edges) - 1, len(lon_edges) - 1)`` with
    ``NaN`` in cells that received no samples.
    """

    lon_edges: Tensor
    lat_edges: Tensor
    data: Tensor
    counts: Tensor
    statistic: str = "mean"
    sampling: str = "binned"

    @property
    def resolution(self) -> tuple[int, int]:
        """``(width, height)`` of the grid in cells."""
        return (int(self.data.shape[1]), int(self.data.shape[0]))

    @property
    def n_samples(self) -> int:
        """Number of samples that landed on the grid."""
        return int(self.counts.sum().item())


@dataclass(frozen=True)
class _Layout:
    """Figure rectangles for one all-sky figure, in figure fractions."""

    map_rect: list[float]
    cbar_rect: list[float] | None
    legend_top: float
    title_y: float
    title_gap: float
    footnote_y: float


def _cell_centres(width: int, height: int) -> tuple[Tensor, Tensor]:
    """Return ``(lon, lat)`` cell centres in degrees, shape ``(height, width)``."""
    lon = (torch.arange(width, dtype=torch.float64) + 0.5) * (360.0 / width) - 180.0
    lat = (torch.arange(height, dtype=torch.float64) + 0.5) * (180.0 / height) - 90.0
    grid_lon, grid_lat = torch.meshgrid(lon, lat, indexing="xy")
    return grid_lon, grid_lat


def _empty_buffers(
    width: int, height: int, *, device: torch.device
) -> tuple[Tensor, Tensor]:
    return (
        torch.zeros(height * width, dtype=torch.float64, device=device),
        torch.zeros(height * width, dtype=torch.int64, device=device),
    )


def _accumulate(
    buf_sum: Tensor,
    buf_count: Tensor,
    lon_deg: Tensor,
    lat_deg: Tensor,
    values: Tensor,
    *,
    width: int,
    height: int,
) -> None:
    """Bin one chunk of samples into the flat accumulators."""
    lon = lon_deg.reshape(-1)
    lat = lat_deg.reshape(-1)
    val = _to(values.reshape(-1), device=buf_sum.device, dtype=torch.float64)
    valid = torch.isfinite(lon) & torch.isfinite(lat) & torch.isfinite(val)
    if not bool(valid.any()):
        return
    lon = lon[valid]
    lat = lat[valid]
    val = val[valid]
    col = torch.floor(torch.remainder(lon + 180.0, 360.0) * (width / 360.0))
    row = torch.floor((lat.clamp(-90.0, 90.0) + 90.0) * (height / 180.0))
    flat = row.long().clamp_(0, height - 1) * width + col.long().clamp_(0, width - 1)
    buf_sum.index_add_(0, flat, val)
    buf_count.index_add_(0, flat, torch.ones_like(flat))


def _finalize(
    buf_sum: Tensor,
    buf_count: Tensor,
    *,
    width: int,
    height: int,
    statistic: str,
    sampling: str = "binned",
) -> SkyGrid:
    counts = buf_count.reshape(height, width)
    empty = counts == 0
    if statistic == "mean":
        data = buf_sum.reshape(height, width) / counts.clamp_min(1).double()
    else:
        data = buf_sum.reshape(height, width).clone()
    data[empty] = float("nan")
    return SkyGrid(
        lon_edges=torch.linspace(-180.0, 180.0, width + 1, dtype=torch.float64),
        lat_edges=torch.linspace(-90.0, 90.0, height + 1, dtype=torch.float64),
        data=data,
        counts=counts,
        statistic=statistic,
        sampling=sampling,
    )


def _validate_resolution(resolution: tuple[int, int]) -> tuple[int, int]:
    width, height = int(resolution[0]), int(resolution[1])
    if width < 4 or height < 2:
        raise ValueError("resolution must be at least (4, 2)")
    return width, height


def _auto_resolution(nside: int) -> tuple[int, int]:
    """Pick a grid whose cells are about one HEALPix pixel across.

    A 58.6°/nside pixel means roughly ``6.1 * nside`` cells around the equator;
    using a finer grid would leave most cells empty and the map see-through.
    """
    width = int(round(6.1 * float(nside)))
    width = max(64, min(2048, width + (width % 2)))
    return (width, max(32, width // 2))


def _validate_statistic(statistic: str) -> str:
    if statistic not in ("mean", "sum"):
        raise ValueError(f"unknown statistic {statistic!r}; use 'mean' or 'sum'")
    return statistic


def sky_grid(
    lon: Tensor,
    lat: Tensor,
    values: Tensor,
    *,
    resolution: tuple[int, int] = (1440, 720),
    statistic: Statistic = "mean",
) -> SkyGrid:
    """Bin point samples ``(lon, lat, values)`` onto a full-sky grid.

    Longitude is in degrees and may be given in any range (it wraps); latitude
    must be in ``[-90, 90]``. Non-finite samples are dropped rather than
    poisoning their cell.
    """
    width, height = _validate_resolution(resolution)
    stat = _validate_statistic(statistic)
    lon_t = torch.as_tensor(lon)
    lat_t = torch.as_tensor(lat)
    val_t = torch.as_tensor(values)
    lon_t, lat_t, val_t = torch.broadcast_tensors(lon_t, lat_t, val_t)
    device = val_t.device
    buf_sum, buf_count = _empty_buffers(width, height, device=device)
    total = lon_t.numel()
    for start in range(0, total, _CHUNK):
        stop = min(start + _CHUNK, total)
        _accumulate(
            buf_sum,
            buf_count,
            lon_t[start:stop],
            lat_t[start:stop],
            val_t[start:stop],
            width=width,
            height=height,
        )
    return _finalize(buf_sum, buf_count, width=width, height=height, statistic=stat)


def _resample_dense(
    vals: Tensor, *, nside: int, nest: bool, width: int, height: int
) -> Tensor:
    """Nearest-pixel resampling: each grid cell takes its containing pixel."""
    from ..sphere import healpix as _healpix

    lon, lat = _cell_centres(width, height)
    lon = lon.reshape(-1)
    lat = lat.reshape(-1)
    out = torch.empty(lon.numel(), dtype=torch.float64, device=vals.device)
    for start in range(0, lon.numel(), _CHUNK):
        stop = min(start + _CHUNK, lon.numel())
        pix = _healpix.ang2pix(
            nside,
            lon[start:stop].to(vals.device),
            lat[start:stop].to(vals.device),
            nest=nest,
            lonlat=True,
        )
        out[start:stop] = vals[pix.long()].double()
    return out.reshape(height, width)


def healpix_sky_grid(
    values: Tensor,
    *,
    nside: int | None = None,
    nest: bool = False,
    pixels: Tensor | None = None,
    resolution: tuple[int, int] | None = None,
    statistic: Statistic = "mean",
    resample: bool | None = None,
) -> SkyGrid:
    """Turn a dense or sparse HEALPix map into a full-sky grid.

    Pass ``pixels`` alongside ``values`` for a sparse map; otherwise ``values``
    must hold one sample per pixel of ``nside``.

    A dense map is nearest-pixel resampled: each grid cell shows the pixel
    containing its centre, so the figure has no empty cells and the cost depends
    on the grid rather than on the map size. Pass ``resample=False`` to
    box-average instead (``statistic`` selects ``mean`` or ``sum``), which
    anti-aliases a map much finer than the grid. Sparse maps are always binned:
    there is no value to resample at unobserved pixels, so their holes are real.

    ``resolution=None`` picks a grid about one pixel wide at the input nside.
    """
    from ..sphere import healpix as _healpix

    if nside is None:
        raise ValueError("nside is required to convert HEALPix pixels")
    nside = int(nside)
    width, height = _validate_resolution(
        resolution if resolution is not None else _auto_resolution(nside)
    )
    stat = _validate_statistic(statistic)
    vals = torch.as_tensor(values).detach()
    if vals.ndim > 1:
        raise ValueError(
            f"expected one value per pixel, got shape {tuple(vals.shape)}; "
            "select a band first"
        )
    npix = int(_healpix.nside2npix(nside))
    if pixels is None:
        if vals.numel() != npix:
            raise ValueError(
                f"dense map for nside={nside} needs {npix} pixels, got {vals.numel()}"
            )
        pix_all = None
        total = npix
    else:
        pix_all = torch.as_tensor(pixels, dtype=torch.int64).reshape(-1)
        if pix_all.numel() != vals.numel():
            raise ValueError(
                f"pixels ({pix_all.numel()}) and values ({vals.numel()}) disagree"
            )
        total = pix_all.numel()

    use_resample = resample if resample is not None else (pix_all is None)
    if use_resample:
        if pix_all is not None:
            raise ValueError("sparse maps must be binned; drop resample=True")
        data = _resample_dense(vals, nside=nside, nest=nest, width=width, height=height)
        return SkyGrid(
            lon_edges=torch.linspace(-180.0, 180.0, width + 1, dtype=torch.float64),
            lat_edges=torch.linspace(-90.0, 90.0, height + 1, dtype=torch.float64),
            data=data,
            counts=torch.ones((height, width), dtype=torch.int64),
            statistic="nearest",
            sampling="resampled",
        )

    device = vals.device
    buf_sum, buf_count = _empty_buffers(width, height, device=device)
    for start in range(0, total, _CHUNK):
        stop = min(start + _CHUNK, total)
        if pix_all is None:
            base = torch.arange(start, stop, dtype=torch.int64, device=device)
            chunk = vals[start:stop]
        else:
            base = pix_all[start:stop].to(device=device)
            chunk = vals[start:stop]
        lon, lat = _healpix.pix2ang(nside, base, nest=nest, lonlat=True)
        _accumulate(
            buf_sum,
            buf_count,
            lon,
            lat,
            chunk,
            width=width,
            height=height,
        )
    return _finalize(buf_sum, buf_count, width=width, height=height, statistic=stat)


# ---------------------------------------------------------------------------
# Tick labels
# ---------------------------------------------------------------------------


def _ra_label(ra_deg: float) -> str:
    """Format an RA angle as hours, with a real superscript ``h``."""
    hours = float(ra_deg % 360.0) / 15.0
    if math.isclose(hours, 0.0, abs_tol=1e-9):
        return "0$^h$"
    return f"{hours:g}$^h$"


def _ra_tick_labels(ticks_deg: Tensor, *, ra_increases_left: bool) -> list[str]:
    """Label display longitudes with the right ascension they carry."""
    labels = []
    for value in ticks_deg.tolist():
        ra = (-value) if ra_increases_left else value
        labels.append(_ra_label(ra))
    return labels


def _dec_label(dec_deg: float) -> str:
    sign = "\u2212" if dec_deg < 0 else ""
    return f"{sign}{abs(dec_deg):g}\u00b0"


def _dec_tick_labels(ticks_deg: Tensor) -> list[str]:
    return [_dec_label(value) for value in ticks_deg.tolist()]


# ---------------------------------------------------------------------------
# Layout, ticks and chrome
# ---------------------------------------------------------------------------


def _auto_figsize(projection: str) -> tuple[float, float]:
    return (8.2, 8.2) if projection == "polar" else (13.0, 7.0)


def _layout(
    figsize: tuple[float, float],
    *,
    aspect: float,
    title: bool,
    subtitle: bool,
    colorbar: bool,
    legend_below: bool,
    footnote: bool,
    top_extra: float = 0.0,
) -> _Layout:
    """Place the map, colorbar and legend from inches, not magic fractions.

    Stacked from the bottom edge: footnote, optional legend, colorbar, map,
    then the title block. Every height is a real measurement, so the colorbar
    hugs the map instead of floating in leftover space.
    """
    width, height = float(figsize[0]), float(figsize[1])
    if title or subtitle:
        top = _TITLE_IN + (_SUBTITLE_IN if subtitle else 0.0)
    else:
        top = 0.30
    top += top_extra
    bottom = _FOOTNOTE_IN if footnote else 0.20
    if legend_below:
        bottom += _LEGEND_IN
    if colorbar:
        bottom += _CBAR_H_IN + _CBAR_GAP_IN
    avail_w = max(width - 2.0 * _MARGIN_IN, 1.0)
    avail_h = max(height - top - bottom, 1.0)
    map_w = min(avail_w, avail_h * aspect)
    map_h = map_w / aspect
    x0 = (width - map_w) / 2.0
    map_rect = [x0 / width, bottom / height, map_w / width, map_h / height]

    cbar_rect = None
    if colorbar:
        cbar_w = map_w * 0.62
        cbar_bottom = bottom - _CBAR_GAP_IN - _CBAR_H_IN
        cbar_rect = [
            (width - cbar_w) / 2.0 / width,
            cbar_bottom / height,
            cbar_w / width,
            _CBAR_H_IN / height,
        ]

    headroom = 0.70 + top_extra if subtitle else 0.36 + top_extra
    # A polar disc carries its RA labels outside the rim; keep the legend clear
    # of the colorbar's own tick labels rather than trusting a fixed reserve.
    if legend_below and cbar_rect is not None:
        legend_top = (cbar_rect[1] * height - 0.32) / height
    elif legend_below:
        legend_top = (map_rect[1] * height - 0.40) / height
    else:
        legend_top = map_rect[1] + map_rect[3]
    return _Layout(
        map_rect=map_rect,
        cbar_rect=cbar_rect,
        legend_top=legend_top,
        title_y=(bottom + map_h + headroom) / height,
        title_gap=0.28 / height,
        footnote_y=0.06 / height,
    )


def _oriented(
    grid: SkyGrid, data: Tensor, *, ra_increases_left: bool
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(x_edges_deg, y_edges_deg, data)`` in plot orientation.

    ``data`` is passed separately so display-time masking (log stretch) survives
    the mirror; the flip must apply to the values actually being drawn.
    """
    if not ra_increases_left:
        return grid.lon_edges, grid.lat_edges, data
    return -grid.lon_edges.flip(0), grid.lat_edges, data.flip(1)


def _polar_raster(
    sky: SkyGrid,
    *,
    data: Tensor | None = None,
    radial: int | None = None,
    azimuthal: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Resample a lon/lat grid onto ``(azimuth, radius)`` edges plus values.

    ``data`` defaults to ``sky.data``; pass the display copy to keep a stretch's
    masking (log drops non-positive cells).

    Drawing the longitude/latitude mesh straight into polar axes collapses every
    longitude column onto the centre, so the quads degenerate into slivers and
    the renderer lets the axes background through between them -- a visibly
    striped starburst at the pole. Sampling the grid once per polar cell has no
    degenerate quads, and it costs the same order as the grid itself.
    """
    height, width = (int(size) for size in sky.data.shape)
    n_radius = int(radial) if radial is not None else height
    n_azimuth = int(azimuthal) if azimuthal is not None else width
    radius_edges = torch.linspace(0.0, 180.0, n_radius + 1, dtype=torch.float64)
    azimuth_edges = torch.linspace(-180.0, 180.0, n_azimuth + 1, dtype=torch.float64)
    radius = 0.5 * (radius_edges[:-1] + radius_edges[1:])
    azimuth = 0.5 * (azimuth_edges[:-1] + azimuth_edges[1:])
    device = sky.data.device
    source = sky.data if data is None else data
    row = torch.clamp(((90.0 - radius) + 90.0) * (height / 180.0), 0, height - 1)
    col = torch.clamp((azimuth + 180.0) * (width / 360.0), 0, width - 1)
    sampled = source[
        row.long().to(device).unsqueeze(1), col.long().to(device).unsqueeze(0)
    ]
    # The innermost ring has no radial extent to speak of, so its columns are
    # all the pole; drawing them individually leaves hairline seams there. One
    # cap average is both the physically right value and artifact-free.
    cap = sampled[0][torch.isfinite(sampled[0])]
    if cap.numel() > 0:
        sampled[0, :] = cap.mean().to(sampled.dtype)
    return azimuth_edges, radius_edges, sampled


def _split_wrap(x_deg: Tensor, y_deg: Tensor) -> list[tuple[Tensor, Tensor]]:
    """Split a polyline where it jumps across the longitude seam."""
    if x_deg.numel() < 2:
        return [(x_deg, y_deg)]
    jumps = torch.nonzero((x_deg[1:] - x_deg[:-1]).abs() > 180.0).flatten() + 1
    if jumps.numel() == 0:
        return [(x_deg, y_deg)]
    xs = torch.tensor_split(x_deg, jumps.tolist())
    ys = torch.tensor_split(y_deg, jumps.tolist())
    return [(a, b) for a, b in zip(xs, ys, strict=True) if a.numel() >= 2]


def _limits(
    data: Tensor,
    vmin: float | None,
    vmax: float | None,
    percentile: tuple[float, float],
) -> tuple[float, float]:
    finite = data[torch.isfinite(data)].double()
    if finite.numel() == 0:
        return 0.0, 1.0
    if vmin is None or vmax is None:
        stride = max(1, finite.numel() // _LIMIT_SAMPLE)
        sample = finite[::stride]
        lo_p, hi_p = float(percentile[0]), float(percentile[1])
        if not 0.0 <= lo_p < hi_p <= 100.0:
            raise ValueError("percentile must satisfy 0 <= lo < hi <= 100")
        q = torch.quantile(sample, torch.tensor([lo_p / 100.0, hi_p / 100.0]).double())
        auto_min, auto_max = float(q[0]), float(q[1])
    else:
        auto_min, auto_max = 0.0, 1.0
    lo = float(vmin) if vmin is not None else auto_min
    hi = float(vmax) if vmax is not None else auto_max
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError("vmin/vmax must be finite")
    if hi <= lo:
        scale = max(abs(lo), 1.0)
        lo, hi = lo - 0.5 * scale, hi + 0.5 * scale
    return lo, hi


def _extend_for(
    data: Tensor, lo: float, hi: float, colorbar_extend: bool | None
) -> str:
    """Whether the display range clips the data (drives colorbar arrows)."""
    finite = data[torch.isfinite(data)]
    if colorbar_extend is False or finite.numel() == 0:
        return "neither"
    clipped_lo = lo > float(finite.min())
    clipped_hi = hi < float(finite.max())
    if colorbar_extend is True:
        return "both"
    if clipped_lo and clipped_hi:
        return "both"
    if clipped_lo:
        return "min"
    if clipped_hi:
        return "max"
    return "neither"


def _norm(stretch: str, vmin: float, vmax: float) -> Any:
    from matplotlib.colors import AsinhNorm, LogNorm, Normalize

    if stretch == "linear":
        return Normalize(vmin=vmin, vmax=vmax)
    if stretch == "log":
        if vmin <= 0.0:
            raise ValueError("log stretch needs a positive vmin")
        return LogNorm(vmin=vmin, vmax=vmax)
    if stretch == "asinh":
        # 5% of the display range is the Lupton-style softening width: enough to
        # lift faint structure without flattening the bright end into paste.
        width = max((vmax - vmin) / 20.0, 1e-12)
        return AsinhNorm(vmin=vmin, vmax=vmax, linear_width=width)
    raise ValueError(f"unknown stretch {stretch!r}; use 'linear', 'log' or 'asinh'")


def _style_geo_ticks(
    ax: Any,
    *,
    ra_increases_left: bool,
    tokens: _theme.Theme,
    fontsize: float,
) -> None:
    """Meridian ticks stay (they draw the graticule); their labels move below."""
    from matplotlib.ticker import NullFormatter

    ra_ticks = torch.arange(-150.0, 180.0, 30.0, dtype=torch.float64)
    ax.set_xticks(torch.deg2rad(ra_ticks).numpy())
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.tick_params(axis="x", length=0)

    dec_ticks = torch.tensor([-60.0, -30.0, 0.0, 30.0, 60.0], dtype=torch.float64)
    ax.set_yticks(torch.deg2rad(dec_ticks).numpy())
    ax.set_yticklabels(_dec_tick_labels(dec_ticks))

    family = _theme.mono_family()
    halo = _theme.halo_effect(tokens.axes_bg)
    # Place by axes fraction: the map rect is exactly the projected extent, so
    # this lines up with the meridians without depending on the projection's
    # own data transform (which scales longitude, not the axes box).
    for x, label in zip(
        ra_ticks, _ra_tick_labels(ra_ticks, ra_increases_left=ra_increases_left)
    ):
        x_frac = (float(x) + 180.0) / 360.0
        ax.text(
            x_frac,
            -0.022,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            color=tokens.ink_muted,
            fontsize=fontsize,
            fontfamily=family,
            clip_on=False,
            path_effects=halo,
        )
    ax.tick_params(axis="y", length=3, pad=2)


def _style_polar_ticks(
    ax: Any,
    *,
    ra_increases_left: bool,
    tokens: _theme.Theme,
    fontsize: float,
) -> None:
    """RA reads around the rim; declination reads inward from the rim."""
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(1 if ra_increases_left else -1)

    ra_deg = torch.arange(0.0, 360.0, 30.0, dtype=torch.float64)
    ax.set_xticks(torch.deg2rad(ra_deg).numpy())
    ax.set_xticklabels([_ra_label(value) for value in ra_deg.tolist()])

    # Five declination rings, but only three labels: the rings carry the grid,
    # and fewer labels keep the data readable where they cross it.
    colat = [30.0, 60.0, 90.0, 120.0, 150.0]
    labelled = {30.0, 90.0, 150.0}
    labels = [_dec_label(90.0 - value) if value in labelled else "" for value in colat]
    ax.set_rgrids(colat, labels=labels)
    ax.set_rlim(0.0, 180.0)
    ax.set_rlabel_position(135.0)
    ax.tick_params(axis="x", labelsize=fontsize, pad=3, length=3)
    ax.tick_params(axis="y", length=0, pad=1)

    family = _theme.mono_family()
    halo = _theme.halo_effect(tokens.axes_bg, linewidth=2.2)
    # Declination rings cut straight across the map, so their labels need the
    # same background halo as the RA row to stay legible over bright data.
    for label in ax.get_yticklabels():
        label.set_color(tokens.ink)
        label.set_fontfamily(family)
        label.set_fontsize(fontsize)
        # Matplotlib rotates ring labels to follow their radial; horizontal text
        # is both easier to read and impossible to mistake for a stray glyph.
        label.set_rotation(0)
        label.set_rotation_mode("anchor")
        label.set_path_effects(halo)
    for label in ax.get_xticklabels():
        label.set_color(tokens.ink_muted)
        label.set_fontfamily(family)
        label.set_fontsize(fontsize)
        label.set_path_effects(halo)


def _two_significant(value: float, _pos: float | None = None) -> str:
    """Label a colour-bar tick to two significant digits, without exponents."""
    if value == 0.0 or not math.isfinite(value):
        return "0"
    decimals = max(0, 1 - int(math.floor(math.log10(abs(value)))))
    return f"{value:.{decimals}f}"


def _decade_ticks(vmin: float, vmax: float, *, wanted: int = 4) -> list[float]:
    """Decade values inside ``[vmin, vmax]``, thinning a very wide range.

    A narrow range that straddles no decade returns fewer than two ticks; the
    caller falls back to evenly spaced bar fractions.
    """
    lo = int(math.floor(math.log10(vmin)))
    hi = int(math.floor(math.log10(vmax)))
    step = max(1, int(math.ceil((hi - lo + 1) / float(wanted))))
    return [
        10.0**power for power in range(lo, hi + 1, step) if vmin <= 10.0**power <= vmax
    ]


def _even_fractions(norm: Any, count: int = 5) -> list[float]:
    """Data values at evenly spaced positions along a normalised colour bar."""
    fractions = torch.linspace(0.12, 0.88, count, dtype=torch.float64)
    return [float(value) for value in norm.inverse(fractions.numpy())]


def _draw_colorbar(
    fig: Any,
    mesh: Any,
    rect: list[float],
    *,
    theme: str,
    tokens: _theme.Theme,
    unit: str | None,
    stretch: str,
    extend: str,
) -> Any:
    """A slim bar under the map, captioned above with unit and stretch."""
    cax = fig.add_axes(rect)
    cbar = fig.colorbar(
        mesh,
        cax=cax,
        orientation="horizontal",
        extend=extend,
        extendfrac=0.022 if extend != "neither" else None,
    )
    from matplotlib.ticker import (
        FuncFormatter,
        LogFormatterSciNotation,
        NullFormatter,
        NullLocator,
    )

    # A nonlinear stretch pushes equal data values unevenly along the bar, so a
    # data-space locator crowds labels at one end. Pin the ticks instead: asinh
    # to evenly spaced bar fractions, log to decades inside the range.
    # ``set_ticks`` installs a fixed locator, which ``update_ticks`` honours.
    decades: list[float] = []
    if stretch == "log":
        decades = _decade_ticks(float(mesh.norm.vmin), float(mesh.norm.vmax))
    if len(decades) >= 2:
        cbar.set_ticks(decades)
        cbar.formatter = LogFormatterSciNotation(labelOnlyBase=True)
    else:
        # Every stretch, log included when it straddles no decade, reads best
        # with labels at even positions along the bar.
        cbar.set_ticks(_even_fractions(mesh.norm))
        cbar.formatter = FuncFormatter(_two_significant)
    # A log-scaled colour-bar axis keeps its minor ticker, and ``update_ticks``
    # reinstates it after the fact; every minor tick would otherwise print its
    # own power and smear the labelling under the bar.
    cbar.minorlocator = NullLocator()
    cbar.minorformatter = NullFormatter()
    cbar.update_ticks()
    # Theme after the ticks exist: ``update_ticks`` builds fresh label artists and
    # they would otherwise come out in matplotlib's default ink.
    _theme.style_colorbar(cbar, theme)
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.tick_params(axis="x", length=2.5, pad=3)

    family = _theme.font_family()
    # Caption above the ends rather than beside them: flanking would crowd the
    # tick labels on a narrow bar.
    if unit:
        cbar.ax.text(
            1.0,
            1.75,
            unit,
            transform=cbar.ax.transAxes,
            ha="right",
            va="bottom",
            color=tokens.ink,
            fontsize=_theme.TYPE.label,
            fontfamily=family,
        )
    cbar.ax.text(
        0.0,
        1.75,
        stretch.upper(),
        transform=cbar.ax.transAxes,
        ha="left",
        va="bottom",
        color=tokens.ink_muted,
        fontsize=_theme.TYPE.footnote,
        fontfamily=_theme.mono_family(),
    )
    return cbar


def _draw_legend(
    fig: Any,
    handles: Sequence[Any],
    *,
    layout: _Layout,
    tokens: _theme.Theme,
) -> None:
    """One centred row in the band reserved under the map.

    The band is the only place a legend cannot land on the projection's rim, the
    graticule or the title block, whatever the figure size.
    """
    labels = [artist.get_label() for artist in handles]
    if not any(label and not label.startswith("_") for label in labels):
        return
    rect = layout.map_rect
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(rect[0] + rect[2] / 2.0, layout.legend_top),
        ncol=max(1, min(3, len(handles))),
        frameon=False,
        fontsize=_theme.TYPE.legend,
        labelcolor=tokens.ink,
        handlelength=1.6,
        columnspacing=1.8,
        borderpad=0.0,
    )


def _draw_moc(
    ax: Any,
    moc: Any,
    *,
    theme: str,
    label: str | None,
    ra_increases_left: bool,
    polar: bool,
) -> Any | None:
    from matplotlib.lines import Line2D

    tokens = _theme.theme_tokens(theme)
    loops = moc.boundary()
    if not loops:
        return None
    # A proxy handle carries the legend entry, so the legend does not depend on
    # which boundary loop happens to be drawn last.
    handle = Line2D(
        [],
        [],
        color=tokens.accent,
        linewidth=1.4,
        label=label if label else "_nolegend_",
    )
    for loop in loops:
        lon = loop[:, 0].double()
        lat = loop[:, 1].double()
        if polar:
            # Split at the 24h seam, otherwise a closed loop draws the long way
            # round the disc as a spiral.
            theta_deg = lon % 360.0
            radius = 90.0 - lat
            for seg_t, seg_r in _split_wrap(theta_deg, radius):
                ax.plot(
                    torch.deg2rad(seg_t).numpy(),
                    seg_r.numpy(),
                    color=tokens.accent,
                    linewidth=1.4,
                    alpha=0.95,
                    label="_nolegend_",
                    zorder=5,
                )
            continue
        x = -lon if ra_increases_left else lon
        for seg_x, seg_y in _split_wrap(x, lat):
            ax.plot(
                torch.deg2rad(seg_x).numpy(),
                torch.deg2rad(seg_y).numpy(),
                color=tokens.accent,
                linewidth=1.4,
                alpha=0.95,
                solid_joinstyle="round",
                label="_nolegend_",
                zorder=5,
            )
    return handle


def _draw_points(
    ax: Any,
    points: tuple[Tensor, Tensor],
    *,
    theme: str,
    label: str | None,
    color: str | None,
    size: float,
    ra_increases_left: bool,
    polar: bool,
) -> Any:
    tokens = _theme.theme_tokens(theme)
    lon = torch.as_tensor(points[0]).double().reshape(-1)
    lat = torch.as_tensor(points[1]).double().reshape(-1)
    if lon.numel() != lat.numel():
        raise ValueError("points must provide matching lon and lat arrays")
    if polar:
        xs = torch.deg2rad(lon % 360.0).numpy()
        ys = (90.0 - lat).numpy()
    else:
        x = -lon if ra_increases_left else lon
        xs = torch.deg2rad(x).numpy()
        ys = torch.deg2rad(lat).numpy()
    return ax.scatter(
        xs,
        ys,
        s=float(size),
        marker="o",
        facecolors="none",
        edgecolors=color or tokens.accent_alt,
        linewidths=0.9,
        alpha=0.95,
        label=label,
        zorder=6,
    )


def plot_sky_grid(
    sky: SkyGrid | None = None,
    *,
    projection: SkyProjection = "mollweide",
    ra_increases_left: bool = True,
    cmap: str = "viridis",
    stretch: Stretch = "asinh",
    vmin: float | None = None,
    vmax: float | None = None,
    percentile: tuple[float, float] = (1.0, 99.0),
    unit: str | None = None,
    cbar_label: str | None = None,
    colorbar_extend: bool | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    theme: str = "dark",
    grid: bool = True,
    colorbar: bool = True,
    footnote: bool = True,
    figsize: tuple[float, float] | None = None,
    dpi: int = 200,
    ax: Any = None,
    save: str | Path | None = None,
    moc: Any | None = None,
    moc_label: str | None = "footprint",
    points: tuple[Tensor, Tensor] | None = None,
    point_label: str | None = None,
    point_color: str | None = None,
    point_size: float = 18.0,
) -> Any:
    """Draw a full-sky grid on an all-sky projection.

    Args:
        sky: grid from :func:`healpix_sky_grid`/:func:`sky_grid`, or ``None``
            to draw only the graticule and overlays.
        projection: ``mollweide``, ``aitoff``, ``hammer`` (full-sky 2:1) or
            ``polar`` (north-pole disc, colatitude as radius).
        ra_increases_left: flip longitude so RA grows to the left (default);
            for ``polar`` it sets the clockwise/counter-clockwise direction.
        cmap: any matplotlib colormap name, or a bundled tile palette name.
        stretch: ``asinh`` (default, keeps faint structure visible across a
            high dynamic range), ``linear`` or ``log`` normalization.
        vmin: lower display limit; ``percentile`` decides when omitted.
        vmax: upper display limit; ``percentile`` decides when omitted.
        unit: colorbar unit; short units of the plotted quantity.
        cbar_label: overrides the colorbar caption entirely.
        colorbar_extend: force colorbar arrow ends; ``None`` derives them from
            whether the display range actually clips the data.
        theme: ``dark`` or ``paper``.
        grid: draw the graticule (also controls the RA/Dec tick labels).
        colorbar: draw a horizontal colorbar under the map.
        footnote: print projection, grid size and sampling under the figure.
        figsize: figure size in inches; defaults to 13x7, or 8.2x8.2 for polar.
        moc: optional :class:`torchsky.sphere.moc.MOC` outline overlay.
        points: optional ``(lon, lat)`` catalog overlay.
        save: path to write a PNG/PDF/SVG; the figure is themed for saving.

    Returns:
        The matplotlib axes.
    """
    if projection not in _SKY_PROJECTIONS:
        raise ValueError(
            f"unknown projection {projection!r}; choose from {sorted(_SKY_PROJECTIONS)}"
        )
    stretch_key = str(stretch).lower()
    palette = matplotlib_cmap(cmap)
    plt = _theme.pyplot()
    polar = projection == "polar"

    handles: list[Any] = []
    created = ax is None
    if created:
        figsize = (
            (figsize[0], figsize[1])
            if figsize is not None
            else _auto_figsize(projection)
        )
        layout = _layout(
            figsize,
            aspect=1.0 if polar else _GEO_ASPECT[projection],
            title=bool(title),
            subtitle=bool(subtitle),
            colorbar=colorbar,
            # An overlay legend always gets its own band under the map; pinned to
            # the map corner it collides with the projection's rim and graticule.
            legend_below=bool(moc is not None or points is not None),
            footnote=footnote,
            top_extra=0.42 if polar else 0.0,
        )
        fig = plt.figure(figsize=figsize, dpi=dpi)
        _theme.apply_figure_theme(fig, theme)
        if polar:
            ax = fig.add_axes(layout.map_rect, projection="polar")
        else:
            ax = fig.add_axes(layout.map_rect, projection=projection)
        _theme.title_block(
            fig, title, subtitle, theme, y=layout.title_y, gap=layout.title_gap
        )
    else:
        fig = ax.figure
        layout = None
        if title:
            tokens = _theme.theme_tokens(theme)
            ax.set_title(
                title,
                color=tokens.ink,
                fontsize=_theme.TYPE.title * 0.8,
                fontfamily=_theme.font_family(),
            )

    tokens = _theme.theme_tokens(theme)
    _theme.apply_axes_theme(ax, theme, grid=grid)
    # Give the globe/projection outline some definition against the background.
    ax.patch.set_edgecolor(tokens.spine)
    ax.patch.set_linewidth(1.0)

    mesh = None
    extend = "neither"
    if sky is not None:
        if not isinstance(sky, SkyGrid):
            raise TypeError("sky must be a SkyGrid or None")
        palette = palette.with_extremes(bad=tokens.axes_bg)
        data = sky.data.clone()
        lo, hi = _limits(data, vmin, vmax, percentile)
        if stretch_key == "log":
            positive = data[torch.isfinite(data) & (data > 0)]
            if positive.numel() == 0:
                raise ValueError("log stretch needs positive values")
            if vmin is None and lo <= 0.0:
                # The grid's low percentile can be exactly zero. Falling back to
                # the smallest positive float would span hundreds of decades and
                # overflow the colour bar, so apply the same percentile rule to
                # the values a log stretch can actually display.
                stride = max(1, positive.numel() // _LIMIT_SAMPLE)
                lo = float(
                    torch.quantile(
                        positive[::stride].double(), float(percentile[0]) / 100.0
                    )
                )
            data[data <= 0] = float("nan")
        norm = _norm(stretch_key, lo, hi)
        extend = _extend_for(data, lo, hi, colorbar_extend)
        if polar:
            theta_edges, radius_edges, raster = _polar_raster(sky, data=data)
            mesh = ax.pcolormesh(
                torch.deg2rad(theta_edges).double().numpy(),
                radius_edges.double().numpy(),
                raster.numpy(),
                shading="flat",
                cmap=palette,
                norm=norm,
                rasterized=True,
            )
        else:
            x_edges, y_edges, oriented = _oriented(
                sky, data, ra_increases_left=ra_increases_left
            )
            mesh = ax.pcolormesh(
                torch.deg2rad(x_edges).double().numpy(),
                torch.deg2rad(y_edges).double().numpy(),
                oriented.numpy(),
                shading="flat",
                cmap=palette,
                norm=norm,
                rasterized=True,
            )

    if polar:
        _style_polar_ticks(
            ax,
            ra_increases_left=ra_increases_left,
            tokens=tokens,
            fontsize=_theme.TYPE.tick,
        )
    else:
        _style_geo_ticks(
            ax,
            ra_increases_left=ra_increases_left,
            tokens=tokens,
            fontsize=_theme.TYPE.tick,
        )
    if not grid:
        ax.tick_params(length=0)
    _theme.apply_axes_theme(ax, theme, grid=grid)

    if moc is not None:
        artist = _draw_moc(
            ax,
            moc,
            theme=theme,
            label=moc_label,
            ra_increases_left=ra_increases_left,
            polar=polar,
        )
        if artist is not None:
            handles.append(artist)
    if points is not None:
        handles.append(
            _draw_points(
                ax,
                points,
                theme=theme,
                label=point_label,
                color=point_color,
                size=point_size,
                ra_increases_left=ra_increases_left,
                polar=polar,
            )
        )

    if mesh is not None and colorbar:
        if layout is not None and layout.cbar_rect is not None:
            _draw_colorbar(
                fig,
                mesh,
                layout.cbar_rect,
                theme=theme,
                tokens=tokens,
                unit=cbar_label if cbar_label is not None else unit,
                stretch=stretch_key,
                extend=extend,
            )
        else:
            cbar = fig.colorbar(
                mesh, ax=ax, orientation="horizontal", fraction=0.05, pad=0.08
            )
            _theme.style_colorbar(
                cbar, theme, label=cbar_label if cbar_label is not None else unit
            )

    if handles:
        if layout is not None:
            _draw_legend(fig, handles, layout=layout, tokens=tokens)
        elif any(artist.get_label() for artist in handles):
            ax.legend(
                loc="lower right",
                frameon=False,
                fontsize=_theme.TYPE.legend,
                labelcolor=tokens.ink,
                handlelength=1.4,
            )

    if sky is not None and footnote and created and layout is not None:
        sampling = (
            "nearest HEALPix pixel"
            if sky.sampling == "resampled"
            else f"{sky.n_samples:,} samples"
        )
        note = (
            f"{projection}  \u00b7  {sky.resolution[0]}\u00d7{sky.resolution[1]} grid"
            f"  \u00b7  {sampling}"
        )
        fig.text(
            layout.map_rect[0] + layout.map_rect[2],
            layout.footnote_y,
            note,
            ha="right",
            va="bottom",
            color=tokens.ink_muted,
            fontsize=_theme.TYPE.footnote,
            fontfamily=_theme.mono_family(),
        )

    if save is not None:
        _theme.save_figure(fig, save, dpi=dpi, theme=theme)
    return ax


def plot_healpix_sky(
    values: Tensor | None = None,
    *,
    nside: int | None = None,
    nest: bool = False,
    pixels: Tensor | None = None,
    resolution: tuple[int, int] | None = None,
    statistic: Statistic = "mean",
    **plot_kwargs: Any,
) -> Any:
    """Plot a dense or sparse HEALPix map as a full-sky figure.

    ``values=None`` draws only the graticule and any overlays, which is the
    usual way to show a survey footprint on a blank sky. All remaining keyword
    arguments go to :func:`plot_sky_grid`.
    """
    grid = None
    if values is not None:
        grid = healpix_sky_grid(
            values,
            nside=nside,
            nest=nest,
            pixels=pixels,
            resolution=resolution,
            statistic=statistic,
        )
    return plot_sky_grid(grid, **plot_kwargs)


__all__ = [
    "SkyGrid",
    "healpix_sky_grid",
    "plot_healpix_sky",
    "plot_sky_grid",
    "sky_grid",
]
