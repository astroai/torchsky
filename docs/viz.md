# Visualization

`torchsky.viz` has two surfaces:

- **Figures** for talks, papers and notebooks: full-sky maps from HEALPix maps
  or scattered samples, and spectral-cube quick-looks. Everything is drawn with
  matplotlib, which is imported lazily, so importing `torchsky.viz` stays cheap.
- **A local browser viewer** for decoded images, catalogs and MOC/HiPS-style
  products.

FITS decoding at the application boundary is provided by `torchfits`.

## Install

```bash
pip install "torchsky[viz]"
```

The `viz` extra brings `fastapi`/`uvicorn`/`pillow` for the server plus
`matplotlib` for figures. The server runs without matplotlib; figures run
without the server dependencies.

## All-sky figures

```python
# fragment: needs matplotlib
import torch
from torchsky.sphere import healpix
from torchsky.viz import plot_healpix_sky

nside = 256
values = torch.rand(healpix.nside2npix(nside))

plot_healpix_sky(
    values,
    nside=nside,
    title="Surface brightness",
    subtitle="Synthetic band · nside 256",
    unit="counts s$^{-1}$ deg$^{-2}$",
    projection="mollweide",
    save="sky.png",
)
```

The dense map is nearest-pixel resampled onto a longitude/latitude grid whose
size follows `nside`, so the figure costs O(grid) rather than O(npix) and has no
empty cells. Pass `resolution=(width, height)` to change it, or
`resample=False` to box-average a map that is much finer than the grid.

Sparse maps keep their real holes:

```python
# fragment: needs matplotlib and your own arrays
plot_healpix_sky(values, nside=nside, pixels=pixels, cmap="magma")
```

Scattered samples — catalog rows, source positions, simulation output — go
through `sky_grid` directly. Longitude wraps, latitude must be in `[-90, 90]`,
and non-finite samples are dropped:

```python
# fragment: needs matplotlib and your own arrays
from torchsky.viz import plot_sky_grid, sky_grid

grid = sky_grid(ra_deg, dec_deg, density, resolution=(1440, 720))
plot_sky_grid(grid, projection="aitoff", unit="sources deg$^{-2}$")
```

### Projections

| `projection` | View |
| --- | --- |
| `mollweide` (default), `aitoff`, `hammer` | full sky, 2:1 frame |
| `polar` | north pole at the centre, colatitude as radius |

The polar view resamples the grid onto an azimuth/radius raster before drawing.
A longitude/latitude mesh collapses every column onto the pole, and the
degenerate quads leak the background between them; the raster has no such
quads and its innermost ring carries the cap average.

Right ascension increases to the left by default (`ra_increases_left=True`),
which is the convention every astronomy figure uses; for `polar` the same flag
sets the clockwise/counter-clockwise direction.

### Footprints and catalogs

```python
# fragment: needs matplotlib and your own arrays
import math
from torchsky.sphere import healpix
from torchsky.sphere.core import lonlat_to_unit_xyz
from torchsky.sphere.moc import MOC, nest2uniq

pixels = healpix.query_disc(nside, lonlat_to_unit_xyz(80.0, -5.0), math.radians(60.0), nest=True)
moc = MOC(nest2uniq(nside, pixels))  # uniq is built from nested pixel indices

plot_healpix_sky(
    values,
    nside=nside,
    moc=moc,
    moc_label="Spectral coverage",
    points=(ra_deg, dec_deg),
    point_label="Catalog sources",
)
```

`values=None` draws the graticule and overlays on a blank sky, which is the
usual way to show a survey footprint on its own.

### Look and feel

`theme="dark"` (default) or `"paper"`; `cmap` accepts any matplotlib colormap
or one of the bundled tile palettes (`viridis`, `magma`, `thermal`, `gray`,
`rdbu_r`, `seismic`); `stretch` is `asinh` (default), `linear` or `log`.

The figure lays itself out in inches — title block, map, right-ascension row,
colour bar, legend band, footnote — so nothing overlaps at any figure size. An
overlay legend is a single centred row in its own band under the map, never
pinned to the projection's rim. Colorbars grow arrow ends automatically when the
display range clips the data, and their ticks stay readable:

- `asinh`/`linear` bars are labelled at evenly spaced positions along the bar
  (a data-space locator crowds labels at one end of an `asinh` stretch),
- `log` bars take decade ticks, falling back to the same even spacing when the
  range straddles no decade,
- every label is rounded to two significant digits,
- a `log` stretch whose low percentile is zero derives its floor from the same
  percentile of the positive values, so the bar spans a real dynamic range
  instead of hundreds of decades.

`save=` writes PNG/PDF/SVG with the themed background, padding included.

`examples/allsky_figures.py` renders the four-figure gallery used in talks:

```bash
python examples/allsky_figures.py --outdir allsky_figures --nside 256
```

## Cube quick-looks

```python
# fragment: needs matplotlib and your own arrays
import torchsky.viz as viz

viz.plot_pseudo_rgb(cube, spectral_axis, title="Pseudo-RGB", save="rgb.png")
viz.plot_pixel_spectrum(cube, spectral_axis, y=120, x=64, save="pixel.png")
```

`plot_pseudo_rgb` splits the spectral axis into three equal-count bands so each
channel carries the same amount of data, and shares one display range across
channels so colour ratios stay meaningful. Both helpers theme their artists
explicitly rather than through global `rcParams`, so a figure keeps its look
whenever it is saved later.

## Serve A Directory

```bash
torchsky viz --root /path/to/data --port 8822
```

```python
# fragment: needs matplotlib and your own arrays
from torchsky.viz import serve

serve(root="/path/to/data", port=8822)
```

## Operational Notes

- Treat the server as a local/development viewer unless you put it behind your
  own authentication and reverse proxy.
- Keep roots narrow; do not serve an entire home directory by accident.
- The served UI is the committed `static/index.html`; there is no npm build
  step or bundler required.
- Large products should use explicit tiling and bounded caches. FITS
  handle/decode caches belong to `torchfits`.

## Related Guides

- [CLI](cli.md)
- [Concepts](concepts.md#file-boundary)
- [Coadds and resampling](map.md)
