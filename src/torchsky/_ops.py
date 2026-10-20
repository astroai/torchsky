from __future__ import annotations

from typing import Any

import torch

from ._native import cpp as _cpp

if _cpp is not None:
    from . import _fixed_dispatch as _fixed_dispatch
    from . import _spectral_dispatch as _spectral_dispatch
    from . import _wcs_dispatch as _wcs_dispatch


def has_wcs_dispatcher() -> bool:
    if _cpp is None:
        return False
    namespace = torch.ops.torchsky
    return hasattr(namespace, "wcs_pixel_to_world") and hasattr(
        namespace, "wcs_world_to_pixel"
    )


def has_fixed_dispatcher() -> bool:
    return _cpp is not None and hasattr(torch.ops.torchsky, "healpix_ring2nest")


def has_spectral_dispatcher() -> bool:
    return _cpp is not None and hasattr(
        torch.ops.torchsky, "spectral_scalar_integrate_recurrence"
    )


def _fixed(name: str, *args: Any) -> Any:
    if not has_fixed_dispatcher():
        raise RuntimeError("torchsky fixed-shape dispatcher operators are unavailable")
    return getattr(torch.ops.torchsky, name)(*args)


def _spectral(name: str, *args: Any) -> Any:
    if not has_spectral_dispatcher():
        raise RuntimeError("torchsky spectral dispatcher operators are unavailable")
    return getattr(torch.ops.torchsky, name)(*args)


def spectral_scalar_alm2map_direct(*args: Any) -> torch.Tensor:
    return _spectral("spectral_scalar_alm2map_direct", *args)


def spectral_ring_fourier_modes(*args: Any) -> torch.Tensor:
    return _spectral("spectral_ring_fourier_modes", *args)


def spectral_ring_fourier_modes_spin_conj(
    *args: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _spectral("spectral_ring_fourier_modes_spin_conj", *args)


def spectral_ring_fourier_synthesis(*args: Any) -> torch.Tensor:
    return _spectral("spectral_ring_fourier_synthesis", *args)


def spectral_spin_ring_finalize(*args: Any) -> torch.Tensor:
    return _spectral("spectral_spin_ring_finalize", *args)


def spectral_spin_map2alm_ring_concat(
    *args: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _spectral("spectral_spin_map2alm_ring_concat", *args)


def spectral_spin_interpolate_concat(
    *args: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _spectral("spectral_spin_interpolate_concat", *args)


def spectral_spin_integrate_concat(
    *args: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _spectral("spectral_spin_integrate_concat", *args)


def spectral_spin_integrate_recurrence(*args: Any) -> torch.Tensor:
    return _spectral("spectral_spin_integrate_recurrence", *args)


def spectral_spin_interpolate_recurrence(*args: Any) -> torch.Tensor:
    return _spectral("spectral_spin_interpolate_recurrence", *args)


def spectral_spin_map2alm_ring_recurrence(*args: Any) -> torch.Tensor:
    return _spectral("spectral_spin_map2alm_ring_recurrence", *args)


def spectral_spin_alm2map_ring_recurrence(*args: Any) -> torch.Tensor:
    return _spectral("spectral_spin_alm2map_ring_recurrence", *args)


def spectral_scalar_integrate_recurrence(*args: Any) -> torch.Tensor:
    return _spectral("spectral_scalar_integrate_recurrence", *args)


def spectral_scalar_interpolate_recurrence(*args: Any) -> torch.Tensor:
    return _spectral("spectral_scalar_interpolate_recurrence", *args)


def _spectral_spin_map2alm_from_basis(
    *args: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _spectral("_spectral_spin_map2alm_from_basis", *args)


def _spectral_spin_alm2map_from_basis(*args: Any) -> torch.Tensor:
    return _spectral("_spectral_spin_alm2map_from_basis", *args)


def sphere_planar_point_in_polygon(*args: Any) -> torch.Tensor:
    return _fixed("sphere_planar_point_in_polygon", *args)


def sphere_arc_intersections_matrix(*args: Any) -> torch.Tensor:
    return _fixed("sphere_arc_intersections_matrix", *args)


def healpix_ring2nest(*args: Any) -> torch.Tensor:
    return _fixed("healpix_ring2nest", *args)


def healpix_nest2ring(*args: Any) -> torch.Tensor:
    return _fixed("healpix_nest2ring", *args)


def healpix_ang2pix_ring(*args: Any) -> torch.Tensor:
    return _fixed("healpix_ang2pix_ring", *args)


def healpix_ang2pix_nested(*args: Any) -> torch.Tensor:
    return _fixed("healpix_ang2pix_nested", *args)


def healpix_pix2ang_ring(*args: Any) -> tuple[torch.Tensor, torch.Tensor]:
    return _fixed("healpix_pix2ang_ring", *args)


def healpix_pix2ang_nested(*args: Any) -> tuple[torch.Tensor, torch.Tensor]:
    return _fixed("healpix_pix2ang_nested", *args)


def healpix_neighbors_ring(*args: Any) -> torch.Tensor:
    return _fixed("healpix_neighbors_ring", *args)


def healpix_neighbors_nested(*args: Any) -> torch.Tensor:
    return _fixed("healpix_neighbors_nested", *args)


def healpix_get_interp_weights_ring(*args: Any) -> tuple[torch.Tensor, torch.Tensor]:
    return _fixed("healpix_get_interp_weights_ring", *args)


def healpix_get_interp_weights_nested(*args: Any) -> tuple[torch.Tensor, torch.Tensor]:
    return _fixed("healpix_get_interp_weights_nested", *args)


def healpix_get_interp_val_ring(*args: Any) -> torch.Tensor:
    return _fixed("healpix_get_interp_val_ring", *args)


def healpix_get_interp_val_nested(*args: Any) -> torch.Tensor:
    return _fixed("healpix_get_interp_val_nested", *args)


def healpix_query_disc(*args: Any) -> torch.Tensor:
    return _fixed("healpix_query_disc", *args)


def sparse_get_values_blocks(*args: Any) -> torch.Tensor:
    return _fixed("sparse_get_values_blocks", *args)


def wcs_pixel_to_world(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not has_wcs_dispatcher():
        raise RuntimeError("torchsky WCS dispatcher operators are unavailable")
    return torch.ops.torchsky.wcs_pixel_to_world(*args, **kwargs)


def wcs_world_to_pixel(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not has_wcs_dispatcher():
        raise RuntimeError("torchsky WCS dispatcher operators are unavailable")
    return torch.ops.torchsky.wcs_world_to_pixel(*args, **kwargs)
