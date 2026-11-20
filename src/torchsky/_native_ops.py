"""torch.library operators backed by the torch-free native core.

The C++ extension (`_cpp_ext`) does not include or link libtorch: kernels see
raw CPU buffers through DLPack and return owned buffers as torch tensors (zero
copy both ways). The operator schemas are defined here, under the
``torchsky`` namespace, with their CPU implementations; fake-tensor shapes and
autograd formulas are registered by ``_fixed_dispatch`` and
``_spectral_dispatch``. ``torch.ops.torchsky.<name>`` therefore behaves as
before (``torch.compile``, fake tensors, autograd), and one wheel serves every
supported PyTorch version.

Two operators are compositions of BLAS calls and live in Python:
``_spectral_spin_map2alm_from_basis`` and ``_spectral_spin_alm2map_from_basis``.
"""

from __future__ import annotations

from typing import Any

import torch

from . import _cpp_ext as _ext

_SCHEMAS: dict[str, str] = {
    "sphere_planar_point_in_polygon": "(Tensor points_xy, Tensor polygon_xy, float eps=1e-15) -> Tensor",
    "sphere_arc_intersections_matrix": "(Tensor a0, Tensor a1, Tensor b0, Tensor b1, float tol_rad=1e-10) -> Tensor",
    "healpix_ring2nest": "(int nside, Tensor pix_ring) -> Tensor",
    "healpix_nest2ring": "(int nside, Tensor pix_nest) -> Tensor",
    "healpix_ang2pix_ring": "(int nside, Tensor lon_deg, Tensor lat_deg) -> Tensor",
    "healpix_ang2pix_nested": "(int nside, Tensor lon_deg, Tensor lat_deg) -> Tensor",
    "healpix_pix2ang_ring": "(int nside, Tensor pix_ring) -> (Tensor lon_deg, Tensor lat_deg)",
    "healpix_pix2ang_nested": "(int nside, Tensor pix_nest) -> (Tensor lon_deg, Tensor lat_deg)",
    "healpix_neighbors_ring": "(int nside, Tensor pix_ring) -> Tensor",
    "healpix_neighbors_nested": "(int nside, Tensor pix_nest) -> Tensor",
    "healpix_get_interp_weights_ring": "(int nside, Tensor lon_deg, Tensor lat_deg) -> (Tensor pixels, Tensor weights)",
    "healpix_get_interp_weights_nested": "(int nside, Tensor lon_deg, Tensor lat_deg) -> (Tensor pixels, Tensor weights)",
    "healpix_get_interp_val_ring": "(int nside, Tensor maps, Tensor lon_deg, Tensor lat_deg) -> Tensor",
    "healpix_get_interp_val_nested": "(int nside, Tensor maps, Tensor lon_deg, Tensor lat_deg) -> Tensor",
    "healpix_query_disc": "(int nside, Tensor vec, float cos_lim, bool nest=False) -> Tensor",
    "sparse_get_values_blocks": "(Tensor query, Tensor block_index, Tensor block_values, Tensor fill) -> Tensor",
    "spectral_scalar_alm2map_direct": "(Tensor alm, Tensor cos_theta, Tensor phi, int lmax, int mmax) -> Tensor",
    "spectral_ring_fourier_modes": "(Tensor rows, Tensor starts, Tensor lengths, int mmax) -> Tensor",
    "spectral_ring_fourier_modes_spin_conj": "(Tensor row, Tensor starts, Tensor lengths, int mmax) -> (Tensor s_plus, Tensor s_minus)",
    "spectral_ring_fourier_synthesis": "(Tensor modes, Tensor starts, Tensor lengths, int npix) -> Tensor",
    "spectral_spin_ring_finalize": "(Tensor s_plus, Tensor s_minus, Tensor starts, Tensor lengths, int npix) -> Tensor",
    "spectral_spin_map2alm_ring_concat": "(Tensor p_plus, Tensor starts, Tensor lengths, Tensor phase0_neg, Tensor y_plus, Tensor y_minus, int lmax, int mmax, float pixel_weight) -> (Tensor c_plus, Tensor c_minus)",
    "spectral_spin_interpolate_concat": "(Tensor coeff_plus, Tensor coeff_minus, Tensor y_plus, Tensor y_minus, int lmax, int mmax) -> (Tensor s_plus, Tensor s_minus)",
    "spectral_spin_integrate_concat": "(Tensor s_plus, Tensor s_minus, Tensor y_plus, Tensor y_minus, int lmax, int mmax, float pixel_weight) -> (Tensor coeff_plus, Tensor coeff_minus)",
    "spectral_spin_integrate_recurrence": "(Tensor s_plus, Tensor s_minus, Tensor theta, Tensor weight, int lmax, int mmax, int spin) -> Tensor",
    "spectral_spin_interpolate_recurrence": "(Tensor alms, Tensor theta, int lmax, int mmax, int spin) -> Tensor",
    "spectral_spin_map2alm_ring_recurrence": "(Tensor maps, Tensor starts, Tensor lengths, Tensor theta, Tensor phase0_neg, int lmax, int mmax, int spin, float pixel_weight) -> Tensor",
    "spectral_spin_alm2map_ring_recurrence": "(Tensor alm_plus, Tensor starts, Tensor lengths, Tensor theta, Tensor phase0_pos, int lmax, int mmax, int spin, Tensor alm_minus, int npix) -> Tensor",
    "spectral_scalar_integrate_recurrence": "(Tensor modes, Tensor theta, Tensor weight, int lmax, int mmax) -> Tensor",
    "spectral_scalar_interpolate_recurrence": "(Tensor alm, Tensor theta, int lmax, int mmax) -> Tensor",
    "moc_uniq_to_ranges": "(Tensor uniq, int max_order) -> (Tensor, Tensor)",
    "moc_ranges_to_uniq": "(Tensor starts, Tensor ends, int max_order) -> Tensor",
    "moc_combine_ranges": "(Tensor a_starts, Tensor a_ends, Tensor b_starts, Tensor b_ends, int op) -> (Tensor, Tensor)",
    "moc_complement_ranges": "(Tensor starts, Tensor ends, int total) -> (Tensor, Tensor)",
    "moc_degrade_uniq": "(Tensor uniq, int order) -> Tensor",
    "sphere_query_disc_nest_ranges": "(int nside, Tensor vec, float cos_lim) -> Tensor",
    "drizzle_square_accumulate": (
        "(Tensor quad_x, Tensor quad_y, Tensor numer, Tensor weight, int width, int height, "
        "Tensor(a!) out_numer, Tensor(b!) out_weight, Tensor(c!) out_variance_weight, "
        "bool with_variance) -> int"
    ),
}

_PYTHON_SCHEMAS: dict[str, str] = {
    "_spectral_spin_map2alm_from_basis": "(Tensor q, Tensor u, Tensor ycp_t, Tensor ycm_t, float pixel_weight) -> (Tensor coeff_plus, Tensor coeff_minus)",
    "_spectral_spin_alm2map_from_basis": "(Tensor coeff_plus, Tensor coeff_minus, Tensor y_plus, Tensor y_minus, Tensor y_plus_m0, Tensor y_minus_m0) -> Tensor",
}

# Mutated outputs are written in place by the kernel: they must already be
# contiguous, dense CPU buffers.
_MUTATED = {"drizzle_square_accumulate": (6, 7, 8)}


def _buffer(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise RuntimeError(
                f"torchsky native kernels run on CPU tensors, got {value.device}"
            )
        return value.detach().resolve_conj().resolve_neg().contiguous()
    return value


def _defaults(name: str) -> list[Any]:
    """Schema default per argument (``_NO_DEFAULT`` where there is none)."""
    schema = getattr(torch.ops.torchsky, name).default._schema
    return [
        a.default_value if a.has_default_value() else _NO_DEFAULT
        for a in schema.arguments
    ]


_NO_DEFAULT = object()


def _native(name: str):
    kernel = getattr(_ext, name)
    mutated = _MUTATED.get(name, ())
    defaults: list[Any] = []

    def implementation(*args: Any) -> Any:
        if not defaults:
            defaults.extend(_defaults(name))
        if len(args) < len(defaults):
            # Python kernels receive only the arguments the caller passed.
            tail = defaults[len(args) :]
            if any(d is _NO_DEFAULT for d in tail):
                raise RuntimeError(f"torchsky::{name} is missing required arguments")
            args = (*args, *tail)
        for index in mutated:
            out = args[index]
            if not out.is_contiguous() or out.device.type != "cpu":
                raise RuntimeError(
                    f"torchsky::{name} output {index} must be a contiguous CPU tensor"
                )
        converted = [
            arg.detach() if i in mutated else _buffer(arg) for i, arg in enumerate(args)
        ]
        return kernel(torch.get_num_threads(), *converted)

    implementation.__name__ = name
    return implementation


def _spin_map2alm_from_basis(
    q: torch.Tensor,
    u: torch.Tensor,
    ycp_t: torch.Tensor,
    ycm_t: torch.Tensor,
    pixel_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(q + i u) @ ycp * w`` and ``conj(q + i u) @ ycm * w``."""
    for name, value, dtype, ndim in (
        ("q", q, torch.float64, 1),
        ("u", u, torch.float64, 1),
        ("ycp_t", ycp_t, torch.complex128, 2),
        ("ycm_t", ycm_t, torch.complex128, 2),
    ):
        if value.dtype != dtype or value.dim() != ndim:
            raise RuntimeError(f"{name} must be a {ndim}-D {dtype} tensor")
    if q.numel() != u.numel():
        raise RuntimeError("q/u length mismatch")
    if ycp_t.shape != ycm_t.shape:
        raise RuntimeError("basis tensors shape mismatch")
    if ycp_t.shape[0] != q.numel():
        raise RuntimeError("basis first dimension must match q/u length")
    p_plus = torch.complex(q, u)
    return (p_plus @ ycp_t) * pixel_weight, (
        torch.conj_physical(p_plus) @ ycm_t
    ) * pixel_weight


def _spin_alm2map_from_basis(
    coeff_plus: torch.Tensor,
    coeff_minus: torch.Tensor,
    y_plus: torch.Tensor,
    y_minus: torch.Tensor,
    y_plus_m0: torch.Tensor,
    y_minus_m0: torch.Tensor,
) -> torch.Tensor:
    """Spin-2 synthesis from basis matrices; returns stacked ``[q, u]`` (float64)."""
    if coeff_plus.numel() != coeff_minus.numel():
        raise RuntimeError("coeff_plus/coeff_minus length mismatch")
    if y_plus.shape != y_minus.shape or y_plus_m0.shape != y_minus_m0.shape:
        raise RuntimeError("basis tensors shape mismatch")
    nalm = coeff_plus.numel()
    if y_plus.shape[0] != nalm:
        raise RuntimeError("basis row count must match coefficient length")
    l0 = y_plus_m0.shape[0]
    if l0 > nalm:
        raise RuntimeError("m0 basis row count cannot exceed coefficient length")
    p_plus_pos = coeff_plus @ y_plus
    p_minus_pos = coeff_minus @ y_minus
    if l0 == nalm:
        p_plus, p_minus = p_plus_pos, p_minus_pos
    else:
        p_plus_m0 = coeff_plus[:l0] @ y_plus_m0
        p_minus_m0 = coeff_minus[:l0] @ y_minus_m0
        # conj_physical: a lazy conj view mis-traces under AOT functionalization.
        p_plus = p_plus_pos + torch.conj_physical(p_minus_pos - p_minus_m0)
        p_minus = p_minus_pos + torch.conj_physical(p_plus_pos - p_plus_m0)
    q = -0.5 * (p_plus + p_minus).real
    u = -0.5 * (p_plus - p_minus).imag
    return torch.stack((q.to(torch.float64), u.to(torch.float64)), 0)


_LIBRARY = torch.library.Library("torchsky", "DEF")
for _name, _schema in _SCHEMAS.items():
    _LIBRARY.define(_name + _schema)
    _LIBRARY.impl(_name, _native(_name), "CPU")
for _name, _schema in _PYTHON_SCHEMAS.items():
    _LIBRARY.define(_name + _schema)
_LIBRARY.impl("_spectral_spin_map2alm_from_basis", _spin_map2alm_from_basis, "CPU")
_LIBRARY.impl("_spectral_spin_alm2map_from_basis", _spin_alm2map_from_basis, "CPU")

__all__: list[str] = []
