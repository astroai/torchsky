# Sparse maps
<!-- TODO(0.5.0 docs): intro -->


## Sparse HEALPix maps

A sparse map stores values only for a sorted pixel list. `from_pixels` takes
explicit indices, `from_dense` extracts the valid pixels of a dense array, and
`from_moc` realizes coverage without building the dense mask first.

<!-- docs: run -->
```python
import torch
from torchsky.sphere import SparseHealpixMap

nside = 16
dense = torch.randn(12 * nside**2, dtype=torch.float64)
pixels = torch.arange(100, dtype=torch.int64)
sparse = SparseHealpixMap.from_pixels(
    pixels=pixels,
    values=dense[:100],
    nside_sparse=nside,
    sentinel=0.0,
)

assert sparse.valid_pixels.numel() == 100
assert int(sparse.coverage_mask.sum()) == 100
assert sparse.get_values(torch.tensor([0, 99, 5000])).shape == (3,)

nearest = sparse.interpolate(
    torch.tensor([0.0]), torch.tensor([0.0]), method="nearest"
)
assert nearest.shape == (1,)

coarse = sparse.ud_grade(8)
assert coarse.nside == 8

roundtrip = sparse.to_dense(fill_value=0.0)
assert torch.equal(roundtrip[:100], dense[:100])
assert float(roundtrip[roundtrip.numel() - 1]) == 0.0
print(coarse.valid_pixels.numel())
```

`get_values` returns the sentinel for uncovered pixels and preserves the leading
value dimensions, so multi-band sparse maps work the same way. `interpolate`
supports `"nearest"` and `"bilinear"`; bilinear weights drop uncovered
neighbours and renormalize. `ud_grade` uses `power`, `pess`, or `reduction`
semantics when downgrading, and preserves the coverage hierarchy when the output
resolution allows it.

`coverage_mask` is a dense boolean mask at `nside_coverage` when one is set, or
at `nside` otherwise. Use `valid_pixels` for the stored indices and
`get_covered_pixels()` for the covered set, including pixels reached through a
coverage map.

## Coverage to values and back

`MOC.from_sparse_map`, `sparse.to_moc()`, and `MOC.to_sparse_map` convert between
topology and values at an explicit resolution. The conversion never expands a
high-order MOC into dense pixels; it realizes only the requested resolution.

<!-- docs: run -->
```python
import torch
from torchsky.sphere import MOC, SparseHealpixMap

nside = 16
sparse = SparseHealpixMap.from_pixels(
    pixels=torch.arange(64, dtype=torch.int64),
    values=torch.ones(64, dtype=torch.float64),
    nside_sparse=nside,
    sentinel=0.0,
)

moc = MOC.from_sparse_map(sparse)
assert moc.area > 0.0

coverage = moc.to_sparse_map(nside=16, nside_coverage=2)
assert coverage.valid_pixels.numel() > 0
assert coverage.nside_coverage == 2
print(moc.max_order, coverage.valid_pixels.numel())
```

## Multi-order value maps

`MultiOrderHealpixMap` attaches tensor values to sorted, disjoint UNIQ cells —
the shape of probabilistic sky maps from gravitational-wave or neutrino
localization, where resolution follows density. `values_are_densities=True`
declares values per steradian, so `integral(footprint)` multiplies by cell area
and by the covered fraction of each cell. Values remain differentiable.

<!-- docs: run -->
```python
import torch
from torchsky.sphere import MOC, MultiOrderHealpixMap, nest2uniq

uniq = nest2uniq(torch.tensor([1, 2]), torch.tensor([0, 4]))
values = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
multi = MultiOrderHealpixMap(uniq, values, values_are_densities=True)

footprint = MOC(nest2uniq(2, torch.tensor([0, 4])))
total = multi.integral(footprint)
assert float(total) > 0.0
total.backward()

assert multi.uniq.numel() == 2
assert values.grad is not None
print(float(total), values.grad.tolist())
```

`from_sparse_map` and `to_sparse_map` exchange fixed-resolution maps;
`from_columns`/`to_columns` exchange `UNIQ` plus value columns. Optional
host-side adapters for `mhealpy` and `cdshealpix` detach and copy to CPU; the
tensor path has no peer dependency.
