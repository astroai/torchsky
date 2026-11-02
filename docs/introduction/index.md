# What torchsky is
<!-- TODO(0.5.0 docs): intro -->


## What torchsky is

torchsky is the tensor-native compute layer for sky data. It takes tensors and
plain header mappings and returns tensors, on whatever device you ask for. The
long-term target is a differentiable model of the sky observed through many
instruments:

```text
observations ~ instrument/noise law(scene, kinematics, context)
```

Physical components, learned components, and hybrids of the two can be fitted
jointly across wavelength, instrument, and modality. The pieces you use today are
the geometric primitives underneath that picture.

| Pillar | Purpose | Namespaces |
|---|---|---|
| World | Coordinates, WCS, frames, spherical geometry, HEALPix, MOC, sparse maps, harmonics | `torchsky.wcs`, `torchsky.coords`, `torchsky.sphere` |
| Scene | Observations, operators, coadds, catalogs, spectra, fields, fitting | `torchsky.map`, `torchsky.catalogs`, `torchsky.scene` and `torchsky.field` (experimental) |
| Viz | Browser and notebook visualization | `torchsky.viz` |
