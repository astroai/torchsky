"""Bounded power-diagram primitives for experimental sky-field models.

The implementation is split by role: geometry and candidate construction
(``foam_geometry``), basis rendering and least-squares solvers
(``foam_solvers``), bounded pixel operators (``foam_operators``), and the
``SpectralPowerFoam`` field/adaptation wrapper (``foam_field``). Historical
imports from :mod:`torchsky.map.field.foam` are unchanged.
"""

from __future__ import annotations

from .foam_common import PowerValueSemantics as PowerValueSemantics

from .foam_geometry import (
    SparseCechGraph,
    TileDiskCSR,
    PolygonDiskCSR,
    power_distance,
    bounded_power_ownership,
    soft_bounded_power_weights,
    smooth_bounded_power_weights,
    cech_adjacency,
    sparse_cech_graph,
    tile_disk_candidate_csr,
    polygon_disk_candidate_csr,
    power_face_planes,
)
from .foam_solvers import (
    spectral_basis_whitening_transform,
    PowerPixelOperator,
    render_power_spectral_samples,
    render_power_spectral_basis,
    render_power_spectral_affine_basis,
    solve_spectral_cell_coefficients,
    solve_power_pixel_least_squares,
    solve_power_pixel_affine_least_squares,
    solve_spectral_power_pixel_affine_least_squares,
)
from .foam_operators import (
    bounded_power_pixel_operator,
    batched_bounded_power_pixel_operator,
    sparse_bounded_power_pixel_operator,
)
from .foam_field import (
    PersistentFoamSiteInitialization,
    FoamAdaptationResult,
    persistent_foam_site_initialization,
    SpectralPowerFoam,
    adapt_spectral_power_foam,
)

__all__ = [
    "PowerValueSemantics",
    "PersistentFoamSiteInitialization",
    "FoamAdaptationResult",
    "persistent_foam_site_initialization",
    "spectral_basis_whitening_transform",
    "SparseCechGraph",
    "TileDiskCSR",
    "PolygonDiskCSR",
    "PowerPixelOperator",
    "render_power_spectral_samples",
    "render_power_spectral_basis",
    "render_power_spectral_affine_basis",
    "solve_spectral_cell_coefficients",
    "solve_power_pixel_least_squares",
    "solve_power_pixel_affine_least_squares",
    "solve_spectral_power_pixel_affine_least_squares",
    "power_distance",
    "bounded_power_ownership",
    "soft_bounded_power_weights",
    "smooth_bounded_power_weights",
    "SpectralPowerFoam",
    "adapt_spectral_power_foam",
    "cech_adjacency",
    "sparse_cech_graph",
    "tile_disk_candidate_csr",
    "polygon_disk_candidate_csr",
    "power_face_planes",
    "bounded_power_pixel_operator",
    "batched_bounded_power_pixel_operator",
    "sparse_bounded_power_pixel_operator",
]
