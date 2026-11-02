# Spherical harmonic transforms
<!-- TODO(0.5.0 docs): intro -->


`torchsky.sphere` analyzes and synthesizes HEALPix maps in spherical harmonic
space on CPU, CUDA, and MPS. The same namespace carries power spectra, beam and
pixel windows, spin-2 transforms, and explicit CAR/HEALPix SHT contracts, so a
mapmaking or field-fitting pipeline never needs healpy at runtime.

## Choosing an entry point

| Goal | Call | Returns |
|---|---|---|
| Analyze a map into coefficients | [`map2alm`](../reference/sphere.md#torchsky.sphere.map2alm) | `(nalm,)` or `(3, nalm)` for polarization |
| Synthesize a map from coefficients | [`alm2map`](../reference/sphere.md#torchsky.sphere.alm2map) | `(npix,)` or `(3, npix)` |
| Auto/cross power spectrum | [`anafast`](../reference/sphere.md#torchsky.sphere.anafast) | `(lmax+1,)` |
| Spectrum from existing coefficients | [`alm2cl`](../reference/sphere.md#torchsky.sphere.alm2cl) | `(lmax+1,)` or stacked spectra |
| Random realization from C_ell | [`synfast`](../reference/sphere.md#torchsky.sphere.synfast) / [`synalm`](../reference/sphere.md#torchsky.sphere.synalm) | map(s) / alm |
| Smooth in harmonic space | [`smoothmap`](../reference/sphere.md#torchsky.sphere.smoothmap) / [`smoothalm`](../reference/sphere.md#torchsky.sphere.smoothalm) | map(s) / alm |
| Apply an ell-dependent filter | [`almxfl`](../reference/sphere.md#torchsky.sphere.almxfl) | filtered alm |
| Spin-2 Q/U analysis and synthesis | [`map2alm_spin`](../reference/sphere.md#torchsky.sphere.map2alm_spin) / [`alm2map_spin`](../reference/sphere.md#torchsky.sphere.alm2map_spin) | `(alm_p, alm_m)` / `(2, npix)` |
| Least-squares analysis | [`map2alm_lsq`](../reference/sphere.md#torchsky.sphere.map2alm_lsq) | `(alm, rel_res, n_iter)` |
| Explicit geometry/backend contract | [`SHTRequest`](../reference/sphere.md#torchsky.sphere.SHTRequest), [`synthesize`](../reference/sphere.md#torchsky.sphere.synthesize), [`analyze`](../reference/sphere.md#torchsky.sphere.analyze) | capability-checked results |

The healpy-shaped aliases live in
[`torchsky.sphere.compat`](../reference/sphere-compat.md); use the
`torchsky.sphere` names in new code.

## Coefficient ordering

Alm arrays use healpy ordering: packed `m >= 0` blocks, complex dtype, with
`m = 0` real by convention. `lmax` defaults to `3 * nside - 1` and `mmax`
defaults to `lmax`. The flat index helpers are the contract for anything that
indexes coefficients directly:

<!-- docs: run -->
```python
import torch
import torchsky.sphere as sphere

lmax = 8
nalm = sphere.alm_size(lmax, lmax)
alm = torch.zeros(nalm, dtype=torch.complex128)

assert sphere.alm_index(0, 0, lmax, lmax) == 0
assert sphere.alm_index(1, 0, lmax, lmax) == 1
assert sphere.alm_index(1, 1, lmax, lmax) == lmax + 1
assert nalm == (lmax + 1) * (lmax + 2) // 2
```

## Analyze and synthesize

`map2alm` accepts a HEALPix map shaped `(npix,)`, or `(3, npix)` with
`pol=True` in `(T, E, B)` order. `alm2map` is the inverse. Both dispatch on the
tensor device and keep the map on it; CUDA and MPS use the ATen ring path.

<!-- docs: run -->
```python
import torch
import torchsky.sphere as sphere

torch.manual_seed(0)
nside, lmax = 16, 8
ell = torch.arange(lmax + 1, dtype=torch.float64)
cl = torch.exp(-ell * (ell + 1) * 0.05**2)

sky = sphere.synfast(cl, nside=nside, lmax=lmax, mmax=lmax, pol=False)
alm = sphere.map2alm(sky, nside=nside, lmax=lmax, mmax=lmax)
recon = sphere.alm2map(alm, nside=nside, lmax=lmax, mmax=lmax)

assert sky.shape == (12 * nside**2,)
assert alm.shape == (sphere.alm_size(lmax, lmax),)
assert recon.shape == sky.shape
print(sky.shape, alm.shape, float(recon.std()))
```

Analysis uses quadrature weights only when asked. `use_weights=True` applies
computed RING quadrature with mean 1; `use_pixel_weights=True` loads HEALPix
full-pixel weight FITS through `torchfits` from `datapath` or
`TORCHSKY_HEALPIX_DATAPATH` and multiplies by `1 + w`. The two flags are
mutually exclusive, and missing weight files raise rather than silently
approximating. `gal_cut` zeros pixels below the given latitude in degrees.

`iter` runs the same iterative refinement as healpy: analyze, synthesize, and
correct the residual. It costs one extra transform pair per iteration.

## Power spectra

`anafast` is the convenience path from maps to C_ell; `alm2cl` works on
coefficients you already have, including cross-spectra between two maps or
between T/E/B components.

<!-- docs: run -->
```python
import torch
import torchsky.sphere as sphere

torch.manual_seed(1)
nside, lmax = 16, 8
cl_in = torch.ones(lmax + 1, dtype=torch.float64)
sky = sphere.synfast(cl_in, nside=nside, lmax=lmax, pol=False)

cl_map = sphere.anafast(sky, nside=nside, lmax=lmax, pol=False)
alm = sphere.map2alm(sky, nside=nside, lmax=lmax)
cl_alm = sphere.alm2cl(alm, lmax=lmax)

assert cl_map.shape == (lmax + 1,)
assert torch.allclose(cl_map, cl_alm, rtol=1e-10, atol=1e-12)
print(cl_map[:4])
```

Polarized input is `(3, npix)` and returns spectra in healpy's stacked ordering;
pass `alm=True` to also receive the analyzed coefficients. For a cross-spectrum,
call `anafast(map1, map2, ...)` or `alm2cl(alm1, alm2)`.

## Beams, filters, and smoothing

`gaussian_beam(fwhm_rad, lmax)` returns the harmonic beam `B_ell`. Multiply
coefficients with `almxfl`, or smooth end-to-end with `smoothmap`. The pixel
window `pixwin(nside, lmax=...)` is embedded package data, not a healpy or FITS
dependency.

<!-- docs: run -->
```python
import torch
import torchsky.sphere as sphere

torch.manual_seed(2)
nside, lmax = 16, 8
ell = torch.arange(lmax + 1, dtype=torch.float64)
sky = sphere.synfast(torch.exp(-ell), nside=nside, lmax=lmax, pol=False)
alm = sphere.map2alm(sky, nside=nside, lmax=lmax)

fwhm_rad = float(torch.deg2rad(torch.tensor(2.0)))
beam = sphere.gaussian_beam(fwhm_rad, lmax)
filtered = sphere.almxfl(alm, beam)
smoothed = sphere.alm2map(filtered, nside=nside, lmax=lmax)

assert beam.shape == (lmax + 1,)
assert filtered.shape == alm.shape
assert float(smoothed.std()) < float(sky.std())

window = sphere.pixwin(nside, lmax=lmax)
assert window.shape == (lmax + 1,)

smoothed_map = sphere.smoothmap(
    sky, nside=nside, lmax=lmax, mmax=lmax, fwhm_rad=fwhm_rad
)
assert smoothed_map.shape == sky.shape
print(float(window[0]), float(smoothed_map.std()))
```

Use `smoothalm` when you want to keep a coefficient-space pipeline;
`smoothmap` analyzes, filters, and synthesizes in one call. `smoothmap` also
accepts `beam=` or `beam_window=` when the transfer function is measured rather
than Gaussian.

## Spin transforms

Spin-2 (Q/U) analysis takes a `(2, npix)` tensor and returns the two coefficient
sets. Synthesis takes that pair back to maps. Only integer, non-negative spins
are accepted.

<!-- docs: run -->
```python
import torch
import torchsky.sphere as sphere

torch.manual_seed(3)
nside, lmax = 16, 8
npix = 12 * nside**2
qu = torch.randn(2, npix, dtype=torch.float64)

alm_p, alm_m = sphere.map2alm_spin(qu, spin=2, nside=nside, lmax=lmax)
qu_back = sphere.alm2map_spin(
    (alm_p, alm_m), nside=nside, spin=2, lmax=lmax
)

assert alm_p.shape == alm_m.shape
assert qu_back.shape == qu.shape
print(float(alm_p.abs().max()))
```

## Bandlimit, precision, and devices

`lmax` up to `3 * nside - 1` is supported at every `nside` (checked against
healpy at `nside` 512, 1024 and 2048, `lmax = 6143`). The scalar Legendre
recurrence carries a per-(m, ring) power-of-`2**256` exponent, so the sectoral
seed `sin(theta)**m` never underflows float64 (it did for `lmax` above about
1900 when `mmax` was large); rings in the classically forbidden region
`m > lmax * sin(theta) + margin` are skipped with a WKB-derived margin (the
dropped weight is below `e**-90`). The torch path used on other devices applies
the same scaling in chunks.

MPS has no float64: the Legendre recurrence of the spin-weighted transforms
(`map2alm_spin`, `alm2map_spin`, and the `pol=True` paths) runs on the host in
float64 and returns float32/complex64 tensors on the MPS device.

## Least-squares analysis

`map2alm_lsq(map_values, lmax, mmax=None, *, nside=None, pol=True, tol=1e-10,
maxiter=20, nest=False)` iterates until the map-domain residual meets `tol`.
It returns `(alm, rel_res, n_iter)`, which is useful when quadrature accuracy is
the bottleneck or when you need the residual as a diagnostic. The default
`pol=True` expects `(3, npix)` input; pass `pol=False` for a scalar map.

## Explicit SHT contracts

The low-level SHT surface separates geometry, operation, and backend instead of
assuming HEALPix everywhere. `SHTRequest` binds a geometry
([`HEALPixGeometry`](../reference/sphere.md#torchsky.sphere.HEALPixGeometry) or
[`CARGeometry`](../reference/sphere.md#torchsky.sphere.CARGeometry)), a bandlimit,
and a convention; `synthesize`, `adjoint_synthesize`, `analyze`, and
`pseudo_analyze` dispatch through a backend that reports
`SHTBackendCapabilities`. Unsupported combinations fail explicitly: for example
`CARGeometry` accepts only separable, unrotated, full-longitude CAR grids with
Clenshaw-Curtis or Fejer-1 latitude centers.

This is the entry point for matrix-free and differentiable workflows that cannot
be expressed as a HEALPix map, and for checking that an optional backend supports
the operation before running it.

## Execution and memory

The scalar path uses ring FFTs plus Legendre recurrences, with an optional fused
CPU kernel and a pure-PyTorch fallback. The spin path keeps the map on the
device and uses in-process FFTs on CPU. Tuning knobs:

| Variable | Default | Effect |
|---|---|---|
| `TORCHSKY_SCALAR_MAX_CACHE_BYTES` | `512 MiB` | Basis-matrix cache budget |
| `TORCHSKY_SCALAR_RING_AUTO_MIN_BYTES` | `64 MiB` | Map size above which the ring path is chosen |
| `TORCHSKY_SPIN_RING_AUTO_MIN_BYTES` | `32 MiB` | Spin ring-path threshold |
| `TORCHSKY_SPIN_THREADS` | unset | Cap the CPU spin `m`-pool |
| `TORCHSKY_SHT_PROFILE` | unset | Print spin timing diagnostics to stderr |

Memory grows with `nside` and `lmax`; the cache knobs bound the reusable basis
matrices rather than the maps or coefficients. Comparative throughput evidence
is maintained in `torchsky-harness`, not here.

## Boundaries and gotchas

- `pol=True` requires `(3, npix)` maps and `(3, nalm)` alms in `(T, E, B)`
  order; the shapes are validated, not inferred.
- `rotate_alm` requires `mmax == lmax` and uses the healpy zyz Euler convention.
- `map2alm_spin` and `alm2map_spin` accept non-negative integer spins only.
- `pixwin` uses bundled tables for standard `nside` values and rejects unsupported
  sizes instead of approximating.
- `TORCHSKY_DISABLE_CPP=1` keeps identical semantics with the pure-PyTorch
  fallback; do not quote native performance numbers from that run.

## See also

- [Sphere and HEALPix guide](index.md)
- [`torchsky.sphere` reference](../reference/sphere.md)
- [healpy compatibility reference](../reference/sphere-compat.md)
- [Multi-order coverage and sparse maps](moc.md)
- [Scientific evaluation ownership](../performance/benchmarks.md)
