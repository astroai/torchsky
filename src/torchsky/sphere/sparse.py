from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Any, Protocol, TYPE_CHECKING, runtime_checkable

import torch
from torch import Tensor

from torchsky import _ops

from . import healpix as _healpix
from ..device import _as_tensor, _to

if TYPE_CHECKING:
    from .geom import PixelizedRegion
    from .moc import MOC
    from .randoms import PreparedRandomSampler


@runtime_checkable
class HealSparseHostProtocol(Protocol):
    """Typing protocol for ``healsparse``-style host maps.

    Accepted by :meth:`SparseHealpixMap.from_healsparse`; not meant to be
    subclassed or instantiated.
    """

    nside_sparse: int
    nside_coverage: int
    sentinel: Any
    metadata: Any

    @property
    def valid_pixels(self) -> Any: ...

    def get_values_pix(self, pixels: Any, **kwargs: Any) -> Any: ...


_CPP_BLOCK_DTYPES = frozenset(
    {
        torch.float64,
        torch.float32,
        torch.int64,
        torch.int32,
        torch.bool,
    }
)

# Dense per-coverage-cell lookup blocks store one slot per fine pixel of every
# covered coverage cell, so their size is ``(n_covered + 1) * ratio**2`` and is
# set by ``nside / nside_coverage``, not by the number of stored pixels. They
# are built only while that slot count is at most this multiple of the stored
# pixel count, which keeps the lookup memory O(n_pixels). Sparser maps answer
# ``get_values`` by binary search on the sorted pixel list; both paths return
# identical values (see tests/test_sphere_sparse.py).
_LOOKUP_BLOCK_MAX_SLOTS_PER_PIXEL = 8


def _coerce_torch_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    import numpy as np

    if dtype in {np.float64, np.dtype("float64")}:
        return torch.float64
    if dtype in {np.float32, np.dtype("float32")}:
        return torch.float32
    if dtype in {np.int64, np.dtype("int64")}:
        return torch.int64
    if dtype in {np.int32, np.dtype("int32")}:
        return torch.int32
    if dtype in {np.bool_, np.dtype("bool")}:
        return torch.bool
    return torch.as_tensor([], dtype=dtype).dtype


def _canonicalize_pixels_values(
    pixels: Tensor, values: Tensor
) -> tuple[Tensor, Tensor]:
    """Sort sparse pixels and keep the last value supplied for duplicates."""
    if pixels.numel() == 0:
        return pixels, values
    order = torch.argsort(pixels, stable=True)
    pixels_sorted = pixels.index_select(0, order)
    values_sorted = values.index_select(-1, order.to(values.device))
    keep = torch.cat(
        [
            pixels_sorted[1:] != pixels_sorted[:-1],
            torch.tensor([True], device=pixels_sorted.device),
        ]
    )
    keep_idx = torch.nonzero(keep, as_tuple=False).flatten()
    return pixels_sorted.index_select(0, keep_idx), values_sorted.index_select(
        -1, keep_idx.to(values.device)
    )


def _validate_sparse_map_inputs(
    nside: int,
    pixels: Tensor,
    values: Tensor,
    column_names: tuple[str, ...] | None,
) -> None:
    npix = _healpix.nside2npix(nside)
    if pixels.dtype != torch.int64:
        raise ValueError("pixels must be int64")
    if pixels.ndim != 1:
        raise ValueError("pixels must be 1D")
    if values.ndim == 0:
        raise ValueError("values must have at least one dimension")
    if values.shape[-1] != pixels.shape[0]:
        raise ValueError("Size of last dimension of values must match pixels")
    if pixels.numel() > 0 and bool(torch.any((pixels < 0) | (pixels >= npix))):
        raise ValueError(f"pixels must be in [0, {npix}) for nside={nside}")
    if column_names is not None:
        if not column_names or any(not name for name in column_names):
            raise ValueError("column_names must contain non-empty names")
        if len(set(column_names)) != len(column_names):
            raise ValueError("column_names must be unique")
        if values.ndim == 1:
            if len(column_names) != 1:
                raise ValueError(
                    f"1D values require a single column name, got {column_names}"
                )
        elif values.ndim == 2:
            if values.shape[0] != len(column_names):
                raise ValueError(
                    f"column_names length ({len(column_names)}) must match leading dimension of values ({values.shape})"
                )
        else:
            raise ValueError("column_names are only supported for 1D or 2D values")


def _validate_nside_coverage(nside: int, nside_coverage: int) -> None:
    _healpix.nside2npix(nside_coverage)
    if nside_coverage > nside or nside % nside_coverage != 0:
        raise ValueError("nside_coverage must be a power-of-two divisor of nside")


def _copy_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    return dict(metadata) if metadata is not None else None


def _healsparse_bit_shift(nside: int, nside_coverage: int) -> int:
    ratio = int(nside) // int(nside_coverage)
    shift = 0
    while ratio > 1:
        if ratio % 2 != 0:
            raise ValueError("nside_coverage must be a power-of-two divisor of nside")
        ratio //= 2
        shift += 1
    return 2 * shift


def _encode_healsparse_images(
    pixels: Tensor,
    values: Tensor,
    nside: int,
    nest: bool,
    nside_coverage: int,
    fill_value: float,
) -> tuple[Tensor, Tensor]:
    """COV int64 map + SPARSE 1D image (sentinel block 0) per HealSparse FITS spec."""
    pix = pixels.detach().cpu().reshape(-1).to(torch.int64)
    val = values.detach().cpu().reshape(-1)
    if not nest:
        pix = _healpix.ring2nest(nside, pix)
    nfine = (nside // nside_coverage) ** 2
    npix_cov = _healpix.nside2npix(nside_coverage)
    bit_shift = _healsparse_bit_shift(nside, nside_coverage)
    cov_map = -torch.arange(npix_cov, dtype=torch.int64) * int(nfine)
    if pix.numel() == 0:
        sparse = torch.full(
            (int(nfine),),
            fill_value,
            dtype=val.dtype,
        )
        return cov_map, sparse
    ip_cov = torch.bitwise_right_shift(pix, bit_shift)
    cov_pix = torch.unique(ip_cov, sorted=True)
    for i, cp in enumerate(cov_pix.tolist()):
        cov_map[int(cp)] = (i + 1) * int(nfine) - int(cp) * int(nfine)
    n_blocks = int(cov_pix.numel()) + 1
    sparse = torch.full((n_blocks * int(nfine),), fill_value, dtype=val.dtype)
    index = pix + cov_map[ip_cov]
    sparse[index] = val
    return cov_map, sparse


def _decode_healsparse_images(
    cov_map: Tensor,
    sparse: Tensor,
    nside: int,
    nside_coverage: int,
    fill_value: float,
) -> tuple[Tensor, Tensor]:
    cov = cov_map.detach().cpu().reshape(-1).to(torch.int64)
    data = sparse.detach().cpu().reshape(-1)
    nfine = (nside // nside_coverage) ** 2
    npix_cov = _healpix.nside2npix(nside_coverage)
    if cov.numel() != npix_cov:
        raise ValueError(
            f"COV length {cov.numel()} != 12 nside_coverage^2 ({npix_cov})"
        )
    starts = cov + torch.arange(npix_cov, dtype=torch.int64) * int(nfine)
    occupied = torch.nonzero(starts != 0, as_tuple=False).flatten()
    if occupied.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.int64),
            torch.empty((0,), dtype=data.dtype),
        )
    local = torch.arange(int(nfine), dtype=torch.int64)
    block_idx = starts[occupied].unsqueeze(1) + local.unsqueeze(0)
    block_vals = data.index_select(0, block_idx.reshape(-1)).reshape(
        occupied.numel(), int(nfine)
    )
    pix = occupied.unsqueeze(1) * int(nfine) + local.unsqueeze(0)
    if data.dtype == torch.bool:
        valid = block_vals
    elif data.is_floating_point():
        valid = ~_healpix.mask_bad(block_vals, badval=fill_value) & torch.isfinite(
            block_vals
        )
    else:
        sent = torch.as_tensor(fill_value, dtype=data.dtype)
        valid = block_vals != sent
    return pix[valid], block_vals[valid]


@dataclass
class SparseHealpixMap:
    """
    Sparse HEALPix map supporting both simple (pixel list) and
    hierarchical (coverage map + blocks) representations.
    """

    nside: int
    nest: bool
    pixels: Tensor  # [n_sparse_pixels] int64 indices
    values: Tensor  # [..., n_sparse_pixels]
    fill_value: float = float(_healpix.UNSEEN)
    metadata: dict[str, Any] | None = None
    column_names: tuple[str, ...] | None = None

    # Hierarchical extensions (optional)
    nside_coverage: int | None = None
    coverage_map: Tensor | None = None
    _block_index: Tensor | None = field(default=None, repr=False, compare=False)
    _block_values: Tensor | None = field(default=None, repr=False, compare=False)
    _fill_tensor: Tensor | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.column_names is not None:
            self.column_names = tuple(str(c) for c in self.column_names)
        _validate_sparse_map_inputs(
            self.nside, self.pixels, self.values, self.column_names
        )
        if self.pixels.device != self.values.device:
            self.pixels = self.pixels.to(device=self.values.device)
        self.pixels, self.values = _canonicalize_pixels_values(self.pixels, self.values)
        if self.nside_coverage is not None:
            self.nside_coverage = int(self.nside_coverage)
            _validate_nside_coverage(int(self.nside), self.nside_coverage)
            if self._block_values is None:
                self._build_coverage_map()
            elif self.coverage_map is None:
                npix_cov = _healpix.nside2npix(self.nside_coverage)
                cov = torch.full(
                    (npix_cov,), -1, dtype=torch.int64, device=self.pixels.device
                )
                if self._block_index is not None:
                    cov[self._block_index > 0] = 0
                self.coverage_map = cov
            if self._fill_tensor is None:
                self._fill_tensor = _as_tensor(
                    self.fill_value, dtype=self.values.dtype, device=self.values.device
                )

    @property
    def nside_sparse(self) -> int:
        return self.nside

    @property
    def sentinel(self) -> float:
        return self.fill_value

    @property
    def columns(self) -> tuple[str, ...] | None:
        return self.column_names

    @property
    def dtype_names(self) -> tuple[str, ...] | None:
        return self.column_names

    @property
    def pixels_per_cov(self) -> int:
        if self.nside_coverage is None:
            return 1
        ratio = self.nside // self.nside_coverage
        return ratio * ratio

    def get_column(self, name: str | int) -> "SparseHealpixMap":
        """Extract a single named or indexed column as a 1D ``SparseHealpixMap``."""
        if isinstance(name, str):
            if self.column_names is None or name not in self.column_names:
                raise KeyError(
                    f"Unknown column {name!r}; available: {self.column_names}"
                )
            idx = self.column_names.index(name)
            col_names: tuple[str, ...] | None = (name,)
        else:
            idx = int(name)
            if self.values.ndim < 2 or idx < 0 or idx >= self.values.shape[0]:
                raise IndexError(
                    f"Column index {idx} out of range for shape {self.values.shape}"
                )
            col_names = (
                (self.column_names[idx],)
                if self.column_names is not None and idx < len(self.column_names)
                else None
            )
        sub_blocks = (
            self._block_values[idx]
            if self._block_values is not None and self._block_values.ndim >= 3
            else self._block_values
            if self.values.ndim == 1
            else None
        )
        return SparseHealpixMap(
            nside=self.nside,
            nest=self.nest,
            pixels=self.pixels,
            values=self.values if self.values.ndim == 1 else self.values[idx],
            fill_value=self.fill_value,
            metadata=_copy_metadata(self.metadata),
            column_names=col_names,
            nside_coverage=self.nside_coverage,
            coverage_map=self.coverage_map,
            _block_index=self._block_index,
            _block_values=sub_blocks,
            _fill_tensor=self._fill_tensor,
        )

    def __getitem__(
        self, key: str | int | Tensor | list[str] | tuple[str, ...]
    ) -> SparseHealpixMap | Tensor:
        if isinstance(key, str):
            return self.get_column(key)
        if isinstance(key, (list, tuple)) and all(isinstance(k, str) for k in key):
            if self.column_names is None:
                raise KeyError("SparseHealpixMap has no column_names")
            indices = [self.column_names.index(k) for k in key]
            idx_t = torch.tensor(indices, dtype=torch.int64, device=self.values.device)
            sub_blocks = (
                self._block_values.index_select(0, idx_t)
                if self._block_values is not None and self._block_values.ndim >= 3
                else None
            )
            return SparseHealpixMap(
                nside=self.nside,
                nest=self.nest,
                pixels=self.pixels,
                values=self.values.index_select(0, idx_t),
                fill_value=self.fill_value,
                metadata=_copy_metadata(self.metadata),
                column_names=tuple(key),
                nside_coverage=self.nside_coverage,
                coverage_map=self.coverage_map,
                _block_index=self._block_index,
                _block_values=sub_blocks,
                _fill_tensor=self._fill_tensor,
            )
        return self.get_values(
            _as_tensor(key, dtype=torch.int64, device=self.pixels.device)
        )

    @classmethod
    def from_columns(
        cls,
        pixels: Tensor,
        columns: dict[str, Tensor],
        nside_sparse: int,
        *,
        nest: bool = True,
        nside_coverage: int | None = None,
        sentinel: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "SparseHealpixMap":
        """Construct a multi-column ``SparseHealpixMap`` from a dict of ``{name: tensor}``."""
        if not columns:
            raise ValueError("columns dictionary must not be empty")
        names = tuple(str(k) for k in columns.keys())
        pix = torch.as_tensor(pixels, dtype=torch.int64).reshape(-1)
        col_tensors = [
            torch.as_tensor(v, device=pix.device).reshape(-1) for v in columns.values()
        ]
        out_dtype = col_tensors[0].dtype
        for ct in col_tensors[1:]:
            out_dtype = torch.promote_types(out_dtype, ct.dtype)
        stacked = torch.stack([t.to(dtype=out_dtype) for t in col_tensors], dim=0)
        fv = float(sentinel) if sentinel is not None else float(_healpix.UNSEEN)
        return cls(
            nside=int(nside_sparse),
            nest=nest,
            pixels=pix,
            values=stacked,
            fill_value=fv,
            metadata=metadata,
            column_names=names,
            nside_coverage=nside_coverage,
        )

    @classmethod
    def from_dense(
        cls,
        dense: Tensor,
        nside: int | None = None,
        nest: bool = True,
        valid_mask: Tensor | None = None,
        fill_value: float | Any = None,
        coverage_mode: str = "any",
        nside_coverage: int | None = None,
        metadata: dict[str, Any] | None = None,
        column_names: tuple[str, ...] | list[str] | None = None,
    ) -> "SparseHealpixMap":
        vals = torch.as_tensor(dense)
        if vals.ndim == 0:
            raise ValueError("dense must have at least one dimension")
        if nside is None:
            npix = vals.shape[-1]
            ns = _healpix.npix2nside(npix)
        else:
            ns = int(nside)
            npix = _healpix.nside2npix(ns)

        if valid_mask is not None:
            valid = _as_tensor(valid_mask, dtype=torch.bool, device=vals.device)
            if valid.shape != (npix,):
                raise ValueError("valid_mask must have shape (npix,)")
        else:
            if fill_value is None:
                fill_value = float(_healpix.UNSEEN)
            if coverage_mode == "any":
                valid = (
                    torch.any(
                        ~_healpix.mask_bad(vals, badval=fill_value)
                        & torch.isfinite(vals),
                        dim=tuple(range(vals.ndim - 1)),
                    )
                    if vals.ndim > 1
                    else (
                        ~_healpix.mask_bad(vals, badval=fill_value)
                        & torch.isfinite(vals)
                    )
                )
            else:  # all
                valid = (
                    torch.all(
                        ~_healpix.mask_bad(vals, badval=fill_value)
                        & torch.isfinite(vals),
                        dim=tuple(range(vals.ndim - 1)),
                    )
                    if vals.ndim > 1
                    else (
                        ~_healpix.mask_bad(vals, badval=fill_value)
                        & torch.isfinite(vals)
                    )
                )

        pixels = torch.nonzero(valid, as_tuple=False).reshape(-1).to(torch.int64)
        sparse_vals = (
            vals.index_select(-1, pixels) if pixels.numel() > 0 else vals[..., :0]
        )
        obj = cls(
            nside=ns,
            nest=nest,
            pixels=pixels,
            values=sparse_vals,
            fill_value=float(fill_value)
            if fill_value is not None
            else float(_healpix.UNSEEN),
            nside_coverage=nside_coverage,
            metadata=metadata,
            column_names=tuple(column_names) if column_names is not None else None,
        )
        return obj

    @classmethod
    def from_pixels(
        cls,
        pixels: Tensor,
        values: Tensor,
        nside_sparse: int,
        nest: bool = True,
        nside_coverage: int | None = None,
        sentinel: float | None = None,
        metadata: dict[str, Any] | None = None,
        column_names: tuple[str, ...] | list[str] | None = None,
    ) -> "SparseHealpixMap":
        fv = float(sentinel) if sentinel is not None else float(_healpix.UNSEEN)
        pix = torch.as_tensor(pixels, dtype=torch.int64)
        vals = torch.as_tensor(values)

        obj = cls(
            nside=nside_sparse,
            nest=nest,
            pixels=pix,
            values=vals,
            fill_value=fv,
            nside_coverage=nside_coverage,
            metadata=metadata,
            column_names=tuple(column_names) if column_names is not None else None,
        )
        return obj

    @classmethod
    def make_empty(
        cls,
        nside_coverage: int,
        nside_map: int,
        dtype: Any = torch.float64,
        sentinel: float | None = None,
        metadata: dict[str, Any] | None = None,
        column_names: tuple[str, ...] | list[str] | None = None,
    ) -> "SparseHealpixMap":
        """Create an empty SparseHealpixMap with specified configuration."""
        fv = float(sentinel) if sentinel is not None else float(_healpix.UNSEEN)
        dtype = _coerce_torch_dtype(dtype)
        c_names = tuple(column_names) if column_names is not None else None
        val_shape = (len(c_names), 0) if c_names is not None else (0,)

        return cls(
            nside=nside_map,
            nest=True,  # HealSparse always uses nesting internally
            pixels=torch.empty((0,), dtype=torch.int64),
            values=torch.empty(val_shape, dtype=dtype),
            fill_value=fv,
            nside_coverage=nside_coverage,
            metadata=metadata,
            column_names=c_names,
        )

    @classmethod
    def convert_healpix_map(
        cls,
        dense: Tensor,
        nside_coverage: int,
        nest: bool = True,
        sentinel: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "SparseHealpixMap":
        fv = float(sentinel) if sentinel is not None else float(_healpix.UNSEEN)
        return cls.from_dense(
            dense,
            nest=nest,
            fill_value=fv,
            nside_coverage=nside_coverage,
            metadata=metadata,
        )

    @classmethod
    def from_wcs(
        cls,
        wcs: Any,
        nside: int,
        *,
        shape: tuple[int, int] | None = None,
        weight: Tensor | None = None,
        mask: Tensor | None = None,
        nside_coverage: int | None = None,
        nest: bool = True,
    ) -> "SparseHealpixMap":
        """Build a ``SparseHealpixMap`` coverage/weight map from a 2D WCS and optional weight/mask."""
        from .moc import MOC

        nside_i = int(nside)
        order = nside_i.bit_length() - 1
        if nside_coverage is None:
            nside_coverage = min(32, nside_i)
        valid_mask = None
        if mask is not None:
            valid_mask = ~torch.as_tensor(mask, dtype=torch.bool)
        if weight is not None:
            w_t = torch.as_tensor(weight)
            if shape is None and w_t.ndim >= 2:
                shape = (int(w_t.shape[-2]), int(w_t.shape[-1]))
            w_ok = w_t > 0
            valid_mask = w_ok if valid_mask is None else (valid_mask & w_ok)

        moc = MOC.from_wcs(wcs, shape=shape, max_order=order, mask=valid_mask)
        if weight is None:
            return cls.from_moc(
                moc,
                nside=nside_i,
                nside_coverage=nside_coverage,
                value=1.0,
                sentinel=0.0,
                dtype=torch.float32,
                nest=nest,
            )
        reg = moc.to_pixelized_region(nside_i, nest=nest)
        pix = reg.pixels.to(dtype=torch.int64)
        if pix.numel() == 0:
            return cls.make_empty(
                nside_coverage, nside_i, dtype=torch.float32, sentinel=0.0
            )
        lon_p, lat_p = _healpix.pix2ang(nside_i, pix, nest=nest, lonlat=True)
        w_t = torch.as_tensor(weight)
        H, W = int(w_t.shape[-2]), int(w_t.shape[-1])
        x_p, y_p = wcs.world_to_pixel(lon_p, lat_p)
        ix = torch.round(x_p).to(torch.int64)
        iy = torch.round(y_p).to(torch.int64)
        in_b = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        vals = torch.zeros(pix.numel(), dtype=w_t.dtype, device=w_t.device)
        if bool(in_b.any()):
            vals[in_b] = w_t[iy[in_b], ix[in_b]]
        keep = vals > 0
        return cls.from_pixels(
            pix[keep],
            vals[keep],
            nside_sparse=nside_i,
            nest=nest,
            nside_coverage=nside_coverage,
            sentinel=0.0,
        )

    @classmethod
    def from_observations(
        cls,
        observations: Any,
        nside: int | None = None,
        *,
        nside_sparse: int | None = None,
        nside_coverage: int | None = None,
        nest: bool = True,
        columns: tuple[str, ...] | list[str] | None = ("weight", "nexp", "fracdet"),
    ) -> "SparseHealpixMap":
        """Build a survey property ``SparseHealpixMap`` (weight, nexp, fracdet) from observations."""
        from .moc import MOC

        nside_val = nside if nside is not None else nside_sparse
        if nside_val is None:
            raise ValueError("Either nside or nside_sparse must be provided")
        nside_i = int(nside_val)
        order = nside_i.bit_length() - 1
        if nside_coverage is None:
            nside_coverage = min(32, nside_i)
        moc = MOC.from_observations(observations, max_order=order)
        reg = moc.to_pixelized_region(nside_i, nest=nest)
        pix = reg.pixels.to(dtype=torch.int64)
        if pix.numel() == 0:
            return cls.make_empty(
                nside_coverage,
                nside_i,
                dtype=torch.float32,
                sentinel=0.0,
                column_names=columns,
            )
        # Sub-pixel 2x2 oversampling for fracdet
        pix_nest = pix if nest else _healpix.ring2nest(nside_i, pix)
        children = _healpix.nest_children(pix_nest, levels=1)  # (N, 4)
        lon_c, lat_c = _healpix.pix2ang(
            nside_i * 2, children.reshape(-1), nest=True, lonlat=True
        )
        lon_c = lon_c.reshape(pix.numel(), 4)
        lat_c = lat_c.reshape(pix.numel(), 4)
        lon_p, lat_p = _healpix.pix2ang(nside_i, pix, nest=nest, lonlat=True)

        w_sum = torch.zeros(pix.numel(), dtype=torch.float32, device=pix.device)
        nexp_sum = torch.zeros(pix.numel(), dtype=torch.float32, device=pix.device)
        sub_covered = torch.zeros((pix.numel(), 4), dtype=torch.bool, device=pix.device)

        for obs in observations:
            wcs = obs.wcs if hasattr(obs, "wcs") else obs
            if hasattr(obs, "data"):
                H, W = int(obs.data.shape[-2]), int(obs.data.shape[-1])
            else:
                arr_s = getattr(wcs, "array_shape", None)
                if arr_s is None:
                    raise ValueError("Observation or WCS must define image shape")
                H, W = int(arr_s[-2]), int(arr_s[-1])
            x_p, y_p = wcs.world_to_pixel(lon_p, lat_p)
            ix = torch.round(x_p).to(torch.int64)
            iy = torch.round(y_p).to(torch.int64)
            in_b = (
                torch.isfinite(x_p)
                & torch.isfinite(y_p)
                & (x_p >= -0.5)
                & (x_p < W - 0.5)
                & (y_p >= -0.5)
                & (y_p < H - 0.5)
            )
            if getattr(obs, "mask", None) is not None and bool(in_b.any()):
                m_t = _as_tensor(obs.mask, dtype=torch.bool, device=pix.device)
                in_b_idx = torch.nonzero(in_b, as_tuple=False).flatten()
                bad = m_t[iy[in_b_idx].clamp(0, H - 1), ix[in_b_idx].clamp(0, W - 1)]
                in_b[in_b_idx] = ~bad
            w_plane = getattr(obs, "weight_map", None)
            w_scalar = 1.0
            raw_w = getattr(obs, "weight", None)
            if isinstance(raw_w, Tensor) and raw_w.ndim >= 2:
                w_plane = raw_w
            elif raw_w is not None:
                w_scalar = float(raw_w)
            if w_plane is not None and bool(in_b.any()):
                w_t = (
                    _as_tensor(w_plane, dtype=torch.float32, device=pix.device)
                    * w_scalar
                )
                in_b_idx = torch.nonzero(in_b, as_tuple=False).flatten()
                w_val = w_t[iy[in_b_idx].clamp(0, H - 1), ix[in_b_idx].clamp(0, W - 1)]
                pos = w_val > 0
                w_sum[in_b_idx[pos]] += w_val[pos]
                nexp_sum[in_b_idx[pos]] += 1.0
            elif bool(in_b.any()):
                w_sum[in_b] += float(w_scalar)
                nexp_sum[in_b] += 1.0

            xc, yc = wcs.world_to_pixel(lon_c.reshape(-1), lat_c.reshape(-1))
            in_c = (
                torch.isfinite(xc)
                & torch.isfinite(yc)
                & (xc >= -0.5)
                & (xc < W - 0.5)
                & (yc >= -0.5)
                & (yc < H - 0.5)
            ).reshape(pix.numel(), 4)
            sub_covered |= in_c

        fracdet = sub_covered.to(torch.float32).mean(dim=-1)
        keep = (w_sum > 0) | (fracdet > 0)
        pix_k = pix[keep]
        if columns is None:
            return cls.from_pixels(
                pix_k,
                w_sum[keep],
                nside_sparse=nside_i,
                nest=nest,
                nside_coverage=nside_coverage,
                sentinel=0.0,
            )
        col_map: dict[str, Tensor] = {}
        for c in columns:
            if c == "weight":
                col_map[c] = w_sum[keep]
            elif c == "nexp":
                col_map[c] = nexp_sum[keep]
            elif c == "fracdet":
                col_map[c] = fracdet[keep]
            else:
                raise ValueError(f"Unsupported observation property column {c!r}")
        return cls.from_columns(
            pix_k,
            col_map,
            nside_sparse=nside_i,
            nest=nest,
            nside_coverage=nside_coverage,
            sentinel=0.0,
        )

    def sample_randoms(
        self,
        n: int,
        *,
        weights: Tensor | str | None = None,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Draw ``n`` points uniformly in solid angle (or weighted) inside this sparse map."""
        from .randoms import sample_randoms

        return sample_randoms(
            n,
            self,
            weights=weights,
            generator=generator,
            device=device,
        )

    def prepare_random_sampler(
        self,
        *,
        weights: SparseHealpixMap | Tensor | str | None = None,
        device: torch.device | str | None = None,
    ) -> PreparedRandomSampler:
        """Snapshot immutable, device-bound state for repeated random draws.

        Pixels, ordering conversion, and resolved weights are captured during
        preparation. Later mutation of this map or weight tensors does not
        update the sampler; call this method again to observe changes.
        """
        from .randoms import _prepare_sparse_random_sampler

        return _prepare_sparse_random_sampler(self, weights=weights, device=device)

    def _coverage_pixels(self) -> Tensor:
        if self.nside_coverage is None:
            return self.pixels
        _validate_nside_coverage(int(self.nside), int(self.nside_coverage))
        pix = self.pixels
        if pix.numel() > 0 and not self.nest:
            pix = _healpix.ring2nest(self.nside, pix)
        ratio = self.nside // self.nside_coverage
        return torch.div(pix, ratio * ratio, rounding_mode="floor")

    def _resampled_nside_coverage(self, nside_out: int) -> int | None:
        if self.nside_coverage is None:
            return None
        return min(int(self.nside_coverage), int(nside_out))

    def _build_coverage_map(self):
        if self.nside_coverage is None:
            self._block_index = None
            self._block_values = None
            return
        npix_cov = _healpix.nside2npix(self.nside_coverage)
        self.coverage_map = torch.full(
            (npix_cov,), -1, dtype=torch.int64, device=self.pixels.device
        )
        if self.pixels.numel() == 0:
            self._build_lookup_blocks()
            return

        cov_pixels = self._coverage_pixels()
        order = torch.argsort(cov_pixels, stable=True)
        cov_sorted = cov_pixels.index_select(0, order)
        first = torch.cat(
            [
                torch.tensor([True], dtype=torch.bool, device=cov_sorted.device),
                cov_sorted[1:] != cov_sorted[:-1],
            ]
        )
        self.coverage_map[cov_sorted[first]] = order[first]
        self._build_lookup_blocks()

    def _build_lookup_blocks(self) -> None:
        """Dense per-coverage-cell value blocks for O(1) ``get_values``.

        Block 0 is the fill/sentinel row so uncovered cells gather without a
        missing-index branch (healsparse stores the same layout as a flat
        offset table). Blocks are skipped (``_block_values is None``) when they
        would hold more than ``_LOOKUP_BLOCK_MAX_SLOTS_PER_PIXEL`` slots per
        stored pixel; ``get_values`` then binary-searches the sorted pixels.
        """
        self._block_index = None
        self._block_values = None
        if self.nside_coverage is None:
            self._fill_tensor = None
            return
        self._fill_tensor = _as_tensor(
            self.fill_value, dtype=self.values.dtype, device=self.values.device
        )
        n_pixels = int(self.pixels.numel())
        if n_pixels == 0:
            return
        npix_cov = _healpix.nside2npix(self.nside_coverage)
        ratio2 = int(self.pixels_per_cov)
        device = self.pixels.device
        lead = tuple(self.values.shape[:-1])
        lead_n = 1
        for size in lead:
            lead_n *= int(size)
        nest_pix = (
            self.pixels if self.nest else _healpix.ring2nest(self.nside, self.pixels)
        )
        cov = torch.div(nest_pix, ratio2, rounding_mode="floor")
        off = nest_pix - cov * ratio2
        unique_cov, inverse = torch.unique(cov, return_inverse=True)
        n_covered = int(unique_cov.numel())
        if (n_covered + 1) * ratio2 > _LOOKUP_BLOCK_MAX_SLOTS_PER_PIXEL * n_pixels:
            return
        index = torch.zeros((npix_cov,), dtype=torch.int64, device=device)
        index[unique_cov] = torch.arange(
            1, n_covered + 1, device=device, dtype=torch.int64
        )
        blocks = torch.full(
            (lead_n, n_covered + 1, ratio2),
            self.fill_value,
            dtype=self.values.dtype,
            device=self.values.device,
        )
        vals = self.values.reshape(lead_n, self.pixels.numel())
        blocks[:, inverse + 1, off] = vals
        self._block_index = index
        self._block_values = blocks.reshape(lead + (n_covered + 1, ratio2))

    def get_covered_pixels(self) -> Tensor:
        return self.pixels

    def to_moc(self) -> "MOC":
        """Create a MOC from this sparse map's coverage."""
        from .moc import MOC

        return MOC.from_sparse_map(self)

    @classmethod
    def from_healsparse(cls, map_obj: HealSparseHostProtocol) -> "SparseHealpixMap":
        """Import a host-side ``healsparse.HealSparseMap`` without dense expansion."""
        import numpy as np

        valid_pixels = map_obj.valid_pixels
        if callable(valid_pixels):
            valid_pixels = valid_pixels()
        pixels_np = np.asarray(valid_pixels, dtype=np.int64)
        values_np = np.asarray(map_obj.get_values_pix(pixels_np, nest=True))
        if values_np.dtype.fields is not None:
            raise ValueError("structured HealSparse maps require selecting a field")
        return cls.from_pixels(
            torch.as_tensor(pixels_np, dtype=torch.int64),
            torch.as_tensor(values_np),
            int(map_obj.nside_sparse),
            nest=True,
            nside_coverage=int(map_obj.nside_coverage),
            sentinel=map_obj.sentinel,
            metadata=dict(map_obj.metadata) if map_obj.metadata is not None else None,
        )

    def to_healsparse(self) -> Any:
        """Export to ``healsparse.HealSparseMap`` when healsparse is installed."""
        try:
            import healsparse
        except ImportError as exc:
            raise ImportError("healsparse is required for to_healsparse()") from exc

        if self.values.ndim != 1:
            raise ValueError("HealSparse interchange requires one-dimensional values")
        nside_coverage = self.nside_coverage
        if nside_coverage is None:
            nside_coverage = max(1, int(self.nside) // 16)
        metadata = (
            {str(key).upper(): value for key, value in self.metadata.items()}
            if self.metadata is not None
            else None
        )
        result = healsparse.HealSparseMap.make_empty(
            int(nside_coverage),
            int(self.nside),
            self.values.detach().cpu().numpy().dtype,
            sentinel=self.fill_value,
            metadata=metadata,
        )
        result.update_values_pix(
            self.pixels.detach().cpu().numpy(),
            self.values.detach().cpu().numpy(),
            nest=bool(self.nest),
        )
        return result

    def to_pixelized_region(
        self,
        nside: int | None = None,
        *,
        nest: bool | None = None,
    ) -> "PixelizedRegion":
        """Expose sparse-map coverage as a ``PixelizedRegion``.

        Same-resolution conversions only change HEALPix ordering. Different
        resolutions go through MOC coverage normalization, avoiding dense masks.
        """
        from .geom import PixelizedRegion

        nside_i = int(self.nside if nside is None else nside)
        nest_out = bool(self.nest if nest is None else nest)
        if nside_i == int(self.nside):
            pix = self.pixels
            if pix.numel() > 0 and nest_out != bool(self.nest):
                pix = (
                    _healpix.ring2nest(nside_i, pix)
                    if nest_out
                    else _healpix.nest2ring(nside_i, pix)
                )
            return PixelizedRegion(nside=nside_i, nest=nest_out, pixels=pix)
        return self.to_moc().to_pixelized_region(nside_i, nest=nest_out)

    @property
    def valid_pixels(self) -> Tensor:
        """Valid pixel indices. External HealSparse hosts may still expose a method."""
        return self.pixels

    def get_values(self, pixels: Tensor) -> Tensor:
        query_shape = tuple(torch.as_tensor(pixels).shape)
        if (
            isinstance(pixels, Tensor)
            and pixels.dtype == torch.int64
            and pixels.device == self.pixels.device
        ):
            pix_flat = pixels.reshape(-1)
        else:
            pix_flat = _as_tensor(
                pixels, dtype=torch.int64, device=self.pixels.device
            ).reshape(-1)
        flat_shape = tuple(self.values.shape[:-1]) + (pix_flat.numel(),)
        out_shape = tuple(self.values.shape[:-1]) + query_shape
        if self.pixels.numel() == 0:
            return torch.full(
                out_shape,
                self.fill_value,
                dtype=self.values.dtype,
                device=self.values.device,
            )
        if (
            self.nside_coverage is not None
            and self._block_index is not None
            and self._block_values is not None
        ):
            return self._get_values_blocks(pix_flat).reshape(out_shape)
        fill = torch.full(
            flat_shape,
            self.fill_value,
            dtype=self.values.dtype,
            device=self.values.device,
        )
        idx, ok = self._global_to_local(pix_flat)
        safe_idx = torch.where(ok, idx, torch.zeros_like(idx))
        gathered = self.values.index_select(-1, safe_idx)
        return torch.where(ok, gathered, fill).reshape(out_shape)

    def _get_values_blocks(self, pix_flat: Tensor) -> Tensor:
        query = pix_flat
        if not self.nest:
            npix = int(_healpix.nside2npix(self.nside))
            in_pix = (pix_flat >= 0) & (pix_flat < npix)
            safe = torch.where(in_pix, pix_flat, torch.zeros_like(pix_flat))
            query = torch.where(in_pix, _healpix.ring2nest(self.nside, safe), pix_flat)
        blocks = self._block_values
        index = self._block_index
        fill_t = self._fill_tensor
        if (
            _ops.has_fixed_dispatcher()
            and blocks is not None
            and index is not None
            and fill_t is not None
            and query.is_cpu
            and index.is_cpu
            and blocks.is_cpu
            and fill_t.is_cpu
            and blocks.dtype in _CPP_BLOCK_DTYPES
            and not blocks.requires_grad
        ):
            return _ops.sparse_get_values_blocks(query, index, blocks, fill_t)
        ratio2 = int(self.pixels_per_cov)
        out_shape = tuple(self.values.shape[:-1]) + (pix_flat.numel(),)
        fill = torch.full(
            out_shape,
            self.fill_value,
            dtype=self.values.dtype,
            device=self.values.device,
        )
        if index is None or blocks is None:
            return fill
        cov = torch.div(query, ratio2, rounding_mode="floor")
        off = query - cov * ratio2
        in_range = (cov >= 0) & (cov < index.numel()) & (off >= 0) & (off < ratio2)
        safe_cov = torch.where(in_range, cov, torch.zeros_like(cov))
        bid = index.index_select(0, safe_cov)
        if int(blocks.shape[-2]) == 0:
            return fill
        safe_bid = torch.where(in_range, bid, torch.zeros_like(bid))
        safe_off = torch.where(in_range, off, torch.zeros_like(off))
        lead_n = 1
        for size in self.values.shape[:-1]:
            lead_n *= int(size)
        packed = blocks.reshape(lead_n, *blocks.shape[-2:])
        gathered = packed[:, safe_bid, safe_off].reshape(fill.shape)
        return torch.where(in_range, gathered, fill)

    def lookup_with_valid(self, pixels: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(values, valid_mask)`` for queried pixel indices."""
        vals = self.get_values(pixels)
        _, ok = self._global_to_local(pixels)
        return vals, ok

    def get_values_pos(
        self,
        lon_or_theta: Tensor | float,
        lat_or_phi: Tensor | float,
        *,
        lonlat: bool = True,
        valid_mask: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Look up sparse map values at sky coordinates."""
        pix = _healpix.ang2pix(
            self.nside,
            lon_or_theta,
            lat_or_phi,
            nest=self.nest,
            lonlat=lonlat,
        )
        vals = self.get_values(pix)
        if not valid_mask:
            return vals
        _, ok = self._global_to_local(pix)
        return vals, ok

    def lookup_radec(
        self, ra_deg: Tensor | float, dec_deg: Tensor | float
    ) -> tuple[Tensor, Tensor]:
        """Return ``(values, valid_mask)`` at ``(ra_deg, dec_deg)``."""
        res = self.get_values_pos(ra_deg, dec_deg, lonlat=True, valid_mask=True)
        assert isinstance(res, tuple)
        return res

    def update_values_pix(self, pixels: Tensor, values: Tensor) -> None:
        """Update existing or add new pixels to the sparse map."""
        pix_new = _as_tensor(pixels, dtype=torch.int64, device=self.pixels.device)
        val_new = torch.as_tensor(values, device=self.values.device)

        if self.pixels.numel() == 0:
            self.pixels, self.values = _canonicalize_pixels_values(pix_new, val_new)
            self._build_coverage_map()
            return

        combined_pix = torch.cat([self.pixels, pix_new])
        combined_val = torch.cat([self.values, val_new], dim=-1)
        self.pixels, self.values = _canonicalize_pixels_values(
            combined_pix, combined_val
        )
        self._build_coverage_map()

    @classmethod
    def from_moc(
        cls,
        moc: "MOC",
        nside: int,
        nside_coverage: int | None = None,
        *,
        value: Any = True,
        sentinel: float | None = False,
        dtype: Any | None = None,
        metadata: dict[str, Any] | None = None,
        nest: bool = True,
    ) -> "SparseHealpixMap":
        """Create a sparse map from MOC coverage without materializing a dense mask."""
        nside_i = int(nside)
        region = moc.to_pixelized_region(nside_i, nest=True)
        pix = region.pixels.to(dtype=torch.int64)
        if not nest and pix.numel() > 0:
            pix = _healpix.nest2ring(nside_i, pix)
        if dtype is None:
            value_t = torch.as_tensor(value, device=pix.device)
            dtype_t = value_t.dtype
        else:
            dtype_t = _coerce_torch_dtype(dtype)
            value_t = _as_tensor(value, dtype=dtype_t, device=pix.device)
        if value_t.ndim != 0:
            raise ValueError("value must be a scalar")
        values = torch.full(
            (pix.numel(),),
            value_t.item(),
            dtype=dtype_t,
            device=pix.device,
        )
        return cls.from_pixels(
            pix,
            values,
            nside_sparse=nside_i,
            nest=nest,
            nside_coverage=nside_coverage,
            sentinel=sentinel,
            metadata=metadata,
        )

    @classmethod
    def from_region(
        cls,
        region: object,
        nside: int,
        *,
        nest: bool = True,
        nside_coverage: int | None = None,
        value: Any = True,
        sentinel: float | None = False,
        dtype: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "SparseHealpixMap":
        """Create a sparse coverage map from a spherical region.

        ``PixelizedRegion`` inputs preserve exact pixels at matching resolution
        and otherwise use MOC flattening at the requested resolution. Exact
        regions are pixelized through their ``pixelize`` or ``query_pixels``
        method via :meth:`torchsky.sphere.moc.MOC.from_region`.
        """
        from .geom import PixelizedRegion
        from .moc import MOC

        nside_i = int(nside)
        _healpix.nside2npix(nside_i)
        if nside_i <= 0 or (nside_i & (nside_i - 1)) != 0:
            raise ValueError("nside must be a positive power of two")

        if isinstance(region, PixelizedRegion) and int(region.nside) == nside_i:
            pix = region.pixels.to(dtype=torch.int64)
            if pix.numel() > 0 and bool(region.nest) != nest:
                pix = (
                    _healpix.ring2nest(nside_i, pix)
                    if nest
                    else _healpix.nest2ring(nside_i, pix)
                )
            if dtype is None:
                value_t = torch.as_tensor(value, device=pix.device)
                dtype_t = value_t.dtype
            else:
                dtype_t = _coerce_torch_dtype(dtype)
                value_t = _as_tensor(value, dtype=dtype_t, device=pix.device)
            if value_t.ndim != 0:
                raise ValueError("value must be a scalar")
            values = torch.full(
                (pix.numel(),),
                value_t.item(),
                dtype=dtype_t,
                device=pix.device,
            )
            return cls.from_pixels(
                pix,
                values,
                nside_sparse=nside_i,
                nest=nest,
                nside_coverage=nside_coverage,
                sentinel=sentinel,
                metadata=metadata,
            )

        order = nside_i.bit_length() - 1
        moc = MOC.from_region(region, max_order=order)
        return cls.from_moc(
            moc,
            nside=nside_i,
            nside_coverage=nside_coverage,
            value=value,
            sentinel=sentinel,
            dtype=dtype,
            metadata=metadata,
            nest=nest,
        )

    def to_dense(self, fill_value: float | None = None) -> Tensor:
        fv = self.fill_value if fill_value is None else float(fill_value)
        npix = _healpix.nside2npix(self.nside)
        out_shape = tuple(self.values.shape[:-1]) + (npix,)

        # Promote dtype if fill_value can't be represented (e.g. UNSEEN in bool map)
        out_dtype = self.values.dtype
        if out_dtype == torch.bool and fv != 0 and fv != 1:
            out_dtype = torch.float64

        out = torch.full(out_shape, fv, dtype=out_dtype, device=self.pixels.device)
        if self.pixels.numel() > 0:
            out.index_copy_(-1, self.pixels, self.values.to(out_dtype))
        return out

    @property
    def coverage_mask(self) -> Tensor:
        """
        Returns a boolean mask of the coverage.
        If nside_coverage is set, returns mask at that resolution (matches healsparse).
        """
        if self.nside_coverage is not None:
            npix_cov = _healpix.nside2npix(self.nside_coverage)
            mask = torch.zeros((npix_cov,), dtype=torch.bool, device=self.pixels.device)
            if self.pixels.numel() > 0:
                cov_pixels = self._coverage_pixels()
                mask[cov_pixels] = True
            return mask

        npix = _healpix.nside2npix(self.nside)
        mask = torch.zeros((npix,), dtype=torch.bool, device=self.pixels.device)
        if self.pixels.numel() > 0:
            mask[self.pixels] = True
        return mask

    def _global_to_local(self, pix: Tensor) -> tuple[Tensor, Tensor]:
        pix_t = _as_tensor(pix, dtype=torch.int64, device=self.pixels.device)
        if self.pixels.numel() == 0:
            return torch.zeros_like(pix_t), torch.zeros_like(pix_t, dtype=torch.bool)

        idx = torch.searchsorted(self.pixels, pix_t)
        in_bounds = idx < self.pixels.numel()
        safe_idx = torch.where(in_bounds, idx, torch.zeros_like(idx))
        val_at_idx = self.pixels.index_select(0, safe_idx)
        ok = in_bounds & (val_at_idx == pix_t)
        return idx, ok

    def interpolate(
        self,
        lon_deg: Tensor | float,
        lat_deg: Tensor | float,
        *,
        method: Literal["nearest", "bilinear"] = "bilinear",
    ) -> Tensor:
        work_dtype = (
            torch.float32 if self.values.device.type == "mps" else torch.float64
        )
        lon_t = _as_tensor(lon_deg, dtype=work_dtype, device=self.values.device)
        lat_t = _as_tensor(lat_deg, dtype=work_dtype, device=self.values.device)
        lon_t, lat_t = torch.broadcast_tensors(lon_t, lat_t)
        shape = lon_t.shape
        lon_f = lon_t.reshape(-1)
        lat_f = lat_t.reshape(-1)

        if method == "nearest":
            pix = _healpix.ang2pix(
                self.nside, lon_f, lat_f, nest=self.nest, lonlat=True
            )
            return self.get_values(pix).reshape(*self.values.shape[:-1], *shape)

        if method != "bilinear":
            raise ValueError("method must be one of {'nearest', 'bilinear'}")

        pix4, w4 = _healpix.get_interp_weights(
            self.nside, lon_f, lat_f, nest=self.nest, lonlat=True
        )
        pix4 = pix4.to(self.pixels.device).reshape(4, -1)
        w4 = _to(
            w4,
            self.values.device,
            dtype=self.values.real.dtype
            if self.values.is_complex()
            else self.values.dtype,
        ).reshape(4, -1)

        out_shape = tuple(self.values.shape[:-1]) + (lon_f.numel(),)
        accum = torch.zeros(
            out_shape, dtype=self.values.dtype, device=self.values.device
        )
        wsum = torch.zeros(out_shape, dtype=w4.dtype, device=self.values.device)

        if self.pixels.numel() > 0:
            stored_fill = (
                _as_tensor(
                    self.fill_value,
                    dtype=self.values.dtype,
                    device=self.values.device,
                )
                if self.values.is_floating_point()
                else None
            )
            for k in range(4):
                idx, ok = self._global_to_local(pix4[k])
                safe_idx = torch.where(ok, idx, torch.zeros_like(idx))
                vals = self.values.index_select(-1, safe_idx)
                wk = w4[k]
                wshape = [1] * (vals.ndim - 1) + [wk.shape[-1]]
                wk_b = wk.reshape(wshape).expand_as(vals)
                valid_val = ok.reshape(wshape).expand_as(vals)
                if stored_fill is not None:
                    valid_val = valid_val & torch.isfinite(vals) & (vals != stored_fill)
                safe_vals = torch.where(valid_val, vals, torch.zeros_like(vals))
                accum = accum + safe_vals * wk_b
                wsum = wsum + torch.where(valid_val, wk_b, torch.zeros_like(wk_b))

        out = torch.full(
            out_shape,
            float(self.fill_value),
            dtype=self.values.dtype,
            device=self.values.device,
        )
        nz = wsum > 0
        safe_wsum = torch.where(nz, wsum, torch.ones_like(wsum))
        filled = accum / safe_wsum
        out = torch.where(nz, filled, out)
        return out.reshape(*self.values.shape[:-1], *shape)

    def _ud_grade_from_coverage_blocks(
        self,
        nside_out: int,
        *,
        ratio: int,
        pess: bool,
        power: float | None,
        fv: Any,
        out_nside_coverage: int | None,
    ) -> SparseHealpixMap | None:
        """Degrade using fill-block layout. None means use the pixel-list path.

        Floating maps only, and only when coverage nside is unchanged so each
        coverage cell stays a contiguous NEST range. Bool/int occupancy lives
        in the pixel list; a fill-block cell cannot tell valid False/0 from an
        uncovered child.
        """
        blocks = self._block_values
        index = self._block_index
        if (
            blocks is None
            or index is None
            or self.nside_coverage is None
            or out_nside_coverage != self.nside_coverage
            or not self.values.is_floating_point()
        ):
            return None
        child_mult = int(ratio) * int(ratio)
        ratio2 = int(self.pixels_per_cov)
        if child_mult <= 1 or ratio2 % child_mult != 0:
            return None
        new_ratio2 = ratio2 // child_mult
        lead = tuple(self.values.shape[:-1])
        n_blocks = int(blocks.shape[-2])
        grouped = blocks.reshape(lead + (n_blocks, new_ratio2, child_mult))
        scale = (
            ((float(nside_out) / float(self.nside)) ** float(power))
            if power is not None
            else 1.0
        )
        stored_fill = _as_tensor(
            self.fill_value, dtype=grouped.dtype, device=grouped.device
        )
        goods = torch.isfinite(grouped) & (grouped != stored_fill)
        sum_vals = (grouped * goods.to(dtype=grouped.dtype)).sum(dim=-1)
        nhit = goods.to(dtype=torch.float64).sum(dim=-1)
        nhit_f = nhit / scale if power is not None else nhit
        out_val = torch.zeros_like(sum_vals)
        nz = nhit_f != 0
        out_val[..., nz] = sum_vals[..., nz] / nhit_f[..., nz].to(dtype=sum_vals.dtype)
        badout = (nhit != float(child_mult)) if pess else (nhit == 0.0)
        out_val[..., badout] = fv
        cov_ids = torch.nonzero(index, as_tuple=False).flatten()
        if cov_ids.numel() == 0:
            return SparseHealpixMap(
                nside=nside_out,
                nest=self.nest,
                pixels=self.pixels.new_empty((0,)),
                values=self.values[..., :0],
                fill_value=fv,
                nside_coverage=out_nside_coverage,
                metadata=_copy_metadata(self.metadata),
            )
        bids = index.index_select(0, cov_ids)
        data = out_val.index_select(-2, bids)
        bad = badout.index_select(-2, bids)
        off = torch.arange(new_ratio2, dtype=torch.int64, device=index.device)
        parent_nest = cov_ids.unsqueeze(1) * new_ratio2 + off.unsqueeze(0)
        if len(lead) == 0:
            keep = ~bad
        else:
            keep = torch.any(~bad, dim=tuple(range(len(lead))))
        if not bool(keep.any().item()):
            return SparseHealpixMap(
                nside=nside_out,
                nest=self.nest,
                pixels=self.pixels.new_empty((0,)),
                values=self.values[..., :0],
                fill_value=fv,
                nside_coverage=out_nside_coverage,
                metadata=_copy_metadata(self.metadata),
            )
        out_pix_n = parent_nest[keep]
        out_sel = data[..., keep]
        out_pix = out_pix_n if self.nest else _healpix.nest2ring(nside_out, out_pix_n)
        cell_alive = keep.any(dim=-1)
        data_alive = data[..., cell_alive, :]
        cov_alive = cov_ids[cell_alive]
        npix_cov = int(index.numel())
        new_index = torch.zeros((npix_cov,), dtype=torch.int64, device=index.device)
        n_alive = int(cov_alive.numel())
        new_index[cov_alive] = torch.arange(
            1, n_alive + 1, device=index.device, dtype=torch.int64
        )
        fill_row = torch.full(
            lead + (1, new_ratio2),
            fv,
            dtype=data.dtype,
            device=data.device,
        )
        new_blocks = torch.cat((fill_row, data_alive), dim=-2)
        cov_map = torch.full((npix_cov,), -1, dtype=torch.int64, device=index.device)
        cov_map[cov_alive] = 0
        return SparseHealpixMap(
            nside=nside_out,
            nest=self.nest,
            pixels=out_pix,
            values=out_sel,
            fill_value=fv,
            nside_coverage=out_nside_coverage,
            metadata=_copy_metadata(self.metadata),
            coverage_map=cov_map,
            _block_index=new_index,
            _block_values=new_blocks,
            _fill_tensor=_as_tensor(fv, dtype=data.dtype, device=data.device),
        )

    def ud_grade(
        self,
        nside_out: int,
        *,
        pess: bool = False,
        power: float | None = None,
        fill_value: float | None = None,
        reduction: str | None = None,
    ) -> "SparseHealpixMap":
        if reduction == "mean":
            pess = False

        nside_out = int(nside_out)
        _healpix.nside2npix(nside_out)
        fv = self.fill_value if fill_value is None else float(fill_value)
        out_nside_coverage = self._resampled_nside_coverage(nside_out)

        if nside_out == int(self.nside):
            if fill_value is None:
                return self
            return SparseHealpixMap(
                nside=self.nside,
                nest=self.nest,
                pixels=self.pixels,
                values=self.values,
                fill_value=fv,
                nside_coverage=out_nside_coverage,
                metadata=_copy_metadata(self.metadata),
            )

        nside_in = int(self.nside)
        up = nside_out > nside_in
        ratio_num = nside_out if up else nside_in
        ratio_den = nside_in if up else nside_out

        if (
            ratio_num % ratio_den == 0
            and (ratio_num // ratio_den & (ratio_num // ratio_den - 1)) == 0
        ):
            ratio = ratio_num // ratio_den
            if self.pixels.numel() == 0:
                return SparseHealpixMap(
                    nside=nside_out,
                    nest=self.nest,
                    pixels=self.pixels.new_empty((0,)),
                    values=self.values[..., :0],
                    fill_value=fv,
                    nside_coverage=out_nside_coverage,
                    metadata=_copy_metadata(self.metadata),
                )

            if not up:
                blocked = self._ud_grade_from_coverage_blocks(
                    nside_out,
                    ratio=ratio,
                    pess=pess,
                    power=power,
                    fv=fv,
                    out_nside_coverage=out_nside_coverage,
                )
                if blocked is not None:
                    return blocked

            pix_n = (
                self.pixels if self.nest else _healpix.ring2nest(nside_in, self.pixels)
            )
            val = self.values
            scale = (
                ((float(nside_out) / float(nside_in)) ** float(power))
                if power is not None
                else 1.0
            )

            if up:
                child_mult = ratio * ratio
                offs = torch.arange(child_mult, dtype=torch.int64, device=pix_n.device)
                out_pix_n = (
                    pix_n.unsqueeze(1) * child_mult + offs.unsqueeze(0)
                ).reshape(-1)
                out_val = val.repeat_interleave(child_mult, dim=-1)
                if power is not None:
                    out_val = out_val * scale
                out_pix = (
                    out_pix_n if self.nest else _healpix.nest2ring(nside_out, out_pix_n)
                )
                return SparseHealpixMap(
                    nside=nside_out,
                    nest=self.nest,
                    pixels=out_pix,
                    values=out_val,
                    fill_value=fv,
                    nside_coverage=out_nside_coverage,
                    metadata=_copy_metadata(self.metadata),
                )
            else:
                parent_mult = ratio * ratio
                order = torch.argsort(pix_n)
                pix_sorted = pix_n.index_select(0, order)
                val_sorted = val.index_select(-1, order)
                parents = torch.div(pix_sorted, parent_mult, rounding_mode="floor")
                uniq, counts = torch.unique_consecutive(parents, return_counts=True)
                group_ids = torch.repeat_interleave(
                    torch.arange(
                        uniq.numel(), dtype=torch.int64, device=parents.device
                    ),
                    counts,
                )
                sum_vals = torch.zeros(
                    (*val_sorted.shape[:-1], uniq.numel()),
                    dtype=torch.float64
                    if val_sorted.dtype == torch.bool
                    else val_sorted.dtype,
                    device=val_sorted.device,
                )
                if (
                    val_sorted.is_floating_point()
                    and self.nside_coverage is not None
                    and out_nside_coverage == self.nside_coverage
                ):
                    # Same rule as _ud_grade_from_coverage_blocks, used when the
                    # map is too sparse to carry lookup blocks: a stored child
                    # counts unless it is non-finite or exactly the stored fill.
                    stored_fill = _as_tensor(
                        self.fill_value, dtype=val_sorted.dtype, device=val.device
                    )
                    goods = torch.isfinite(val_sorted) & (val_sorted != stored_fill)
                elif val_sorted.is_floating_point() or val_sorted.is_complex():
                    base = val_sorted.real if val_sorted.is_complex() else val_sorted
                    goods = torch.isfinite(val_sorted) & (
                        ~torch.isclose(
                            base,
                            _as_tensor(fv, dtype=base.dtype, device=base.device),
                        )
                    )
                else:
                    goods = torch.ones_like(val_sorted, dtype=torch.bool)
                    if (
                        not val_sorted.is_floating_point()
                        and not val_sorted.is_complex()
                    ):
                        # For discrete types, we still want to skip sentinel if possible
                        # but often sentinel is not clearly defined for bool.
                        # HealSparse bool maps use False as background.
                        pass

                sum_vals.index_add_(
                    -1,
                    group_ids,
                    (val_sorted * goods.to(dtype=val_sorted.dtype))
                    if val_sorted.dtype != torch.bool
                    else goods.to(dtype=torch.float64),
                )
                nhit = torch.zeros(
                    (*val_sorted.shape[:-1], uniq.numel()),
                    dtype=torch.float64,
                    device=val_sorted.device,
                )
                nhit.index_add_(-1, group_ids, goods.to(dtype=torch.float64))
                nhit_f = nhit
                if power is not None:
                    nhit_f = nhit_f / scale
                if val_sorted.dtype == torch.bool:
                    # For bool, ud_grade (down) usually means "any child is True"
                    out_val = sum_vals > 0
                else:
                    out_val = torch.zeros_like(sum_vals)
                    nz = nhit_f != 0
                    out_val[..., nz] = sum_vals[..., nz] / nhit_f[..., nz].to(
                        dtype=sum_vals.dtype
                    )
                badout = (nhit != float(parent_mult)) if pess else (nhit == 0.0)
                out_val = out_val.clone()
                if out_val.dtype == torch.bool:
                    out_val[..., badout] = False
                else:
                    out_val[..., badout] = float(fv)
                if out_val.ndim == 1:
                    keep = ~badout
                else:
                    keep = torch.any(~badout, dim=tuple(range(out_val.ndim - 1)))
                out_pix_n = uniq[keep]
                out_val = out_val[..., keep]
                out_pix = (
                    out_pix_n if self.nest else _healpix.nest2ring(nside_out, out_pix_n)
                )
                return SparseHealpixMap(
                    nside=nside_out,
                    nest=self.nest,
                    pixels=out_pix,
                    values=out_val,
                    fill_value=fv,
                    nside_coverage=out_nside_coverage,
                    metadata=_copy_metadata(self.metadata),
                )

        dense = self.to_dense()
        order = "NEST" if self.nest else "RING"
        dense_out = _healpix.ud_grade(
            dense,
            nside_out,
            pess=pess,
            badval=fv,
            order_in=order,
            order_out=order,
            power=power,
        )
        valid = (
            torch.any(
                ~_healpix.mask_bad(dense_out, badval=fv) & torch.isfinite(dense_out),
                dim=tuple(range(dense_out.ndim - 1)),
            )
            if dense_out.ndim > 1
            else (~_healpix.mask_bad(dense_out, badval=fv) & torch.isfinite(dense_out))
        )
        return SparseHealpixMap.from_dense(
            dense_out,
            nside=nside_out,
            nest=self.nest,
            valid_mask=valid,
            fill_value=fv,
            nside_coverage=out_nside_coverage,
            metadata=_copy_metadata(self.metadata),
        )

    def write_fits(self, filename: str, overwrite: bool = False) -> None:
        """Write a HealSparse FITS file via torchfits (COV + SPARSE images).

        Maps with a 1-D value vector write ``PIXTYPE=HEALSPARSE`` COV+SPARSE
        images. Coverage nside is taken from ``nside_coverage`` when set,
        otherwise ``max(1, nside // 16)`` (pixel-list maps). Multidimensional
        values are rejected because HealSparse image layout is one-dimensional.
        """
        if self.values.ndim != 1:
            raise ValueError("FITS output only supports 1D sparse-map values")

        if self.nside_coverage is not None:
            self._write_healsparse_fits(filename, overwrite=overwrite)
            return
        nside_cov = max(1, int(self.nside) // 16)
        while nside_cov > 1 and (int(self.nside) % nside_cov) != 0:
            nside_cov //= 2
        _validate_nside_coverage(int(self.nside), nside_cov)
        self._write_healsparse_fits(
            filename, overwrite=overwrite, nside_coverage=nside_cov
        )

    def _write_healsparse_fits(
        self,
        filename: str,
        overwrite: bool = False,
        nside_coverage: int | None = None,
    ) -> None:
        from torchfits import insert_hdu, write

        resolved_coverage = (
            self.nside_coverage if nside_coverage is None else nside_coverage
        )
        if resolved_coverage is None:
            raise ValueError(
                "nside_coverage is required when the map has no hierarchical coverage"
            )
        nside_cov = int(resolved_coverage)
        cov_map, sparse = _encode_healsparse_images(
            self.pixels,
            self.values,
            int(self.nside),
            bool(self.nest),
            nside_cov,
            False if self.values.dtype == torch.bool else float(self.fill_value),
        )
        if self.values.dtype == torch.bool:
            sparse = sparse.to(torch.uint8)
        cov_header = {
            "EXTNAME": "COV",
            "PIXTYPE": "HEALSPARSE",
            "NSIDE": nside_cov,
        }
        sparse_header = {
            "EXTNAME": "SPARSE",
            "PIXTYPE": "HEALSPARSE",
            "SENTINEL": self.fill_value,
            "NSIDE": int(self.nside),
            "NSIDECOV": nside_cov,
            "ORDERING": "NEST",
        }
        if self.values.dtype == torch.bool:
            sparse_header["BOOLMAP"] = True
            sparse_header["SENTINEL"] = False
        reserved = {
            "NSIDE",
            "ORDERING",
            "SENTINEL",
            "EXTNAME",
            "NSIDE_COV",
            "NSIDECOV",
            "PIXTYPE",
            "BOOLMAP",
            "SIMPLE",
            "BITPIX",
            "NAXIS",
            "NAXIS1",
            "NAXIS2",
            "PCOUNT",
            "GCOUNT",
            "XTENSION",
        }
        if self.metadata:
            for k, v in self.metadata.items():
                k_upper = str(k).upper()
                if k_upper not in reserved:
                    sparse_header[k_upper] = v
        write(filename, cov_map, header=cov_header, overwrite=overwrite)
        insert_hdu(filename, sparse, index=1, header=sparse_header)

    @classmethod
    def _read_healsparse_fits(cls, filename: str) -> "SparseHealpixMap":
        from torchfits import read_header as get_header, read_tensor as read_image

        hdr_cov = get_header(filename, hdu=0)
        hdr = get_header(filename, hdu=1)
        cov = torch.as_tensor(read_image(filename, hdu=0, mmap=False))
        sparse = torch.as_tensor(read_image(filename, hdu=1, mmap=False))
        nside_cov = int(hdr_cov.get("NSIDE", hdr.get("NSIDECOV", 0)))
        nside = int(hdr.get("NSIDE", 0))
        fv_raw = hdr.get("SENTINEL", float(_healpix.UNSEEN))
        if isinstance(fv_raw, str) and fv_raw.strip().upper() in {"T", "TRUE"}:
            fv = 1.0
        elif isinstance(fv_raw, str) and fv_raw.strip().upper() in {"F", "FALSE"}:
            fv = 0.0
        elif fv_raw is None:
            fv = float(_healpix.UNSEEN)
        else:
            fv = float(fv_raw)
        if str(hdr.get("BOOLMAP", "")).strip().upper() in {"T", "TRUE", "1"}:
            sparse = sparse.to(torch.bool)
            fv = 0.0
        pixels, values = _decode_healsparse_images(cov, sparse, nside, nside_cov, fv)
        reserved = {
            "SIMPLE",
            "BITPIX",
            "NAXIS",
            "NAXIS1",
            "NAXIS2",
            "EXTEND",
            "COMMENT",
            "HISTORY",
            "XTENSION",
            "PCOUNT",
            "GCOUNT",
            "EXTNAME",
            "PIXTYPE",
            "NSIDE",
            "ORDERING",
            "SENTINEL",
            "NSIDE_COV",
            "NSIDECOV",
            "BOOLMAP",
        }
        metadata = {}
        for k, v in hdr.items():
            k_upper = str(k).upper()
            if k_upper in reserved or k_upper.startswith("NAXIS"):
                continue
            metadata[k_upper] = v
        return cls.from_pixels(
            pixels,
            values,
            nside,
            nest=True,
            nside_coverage=nside_cov,
            sentinel=fv,
            metadata=metadata or None,
        )

    @classmethod
    def read_fits(cls, filename: str) -> "SparseHealpixMap":
        """Read HealSparse COV+SPARSE images or a PIXEL/VALUE table."""
        from torchfits import read_header as get_header
        from torchfits.table import read_torch as read_table

        hdr0 = get_header(filename, hdu=0)
        pixtype = str(hdr0.get("PIXTYPE", "")).strip().upper()
        ext0 = str(hdr0.get("EXTNAME", "")).strip().upper()
        if pixtype == "HEALSPARSE" or ext0 == "COV":
            return cls._read_healsparse_fits(filename)

        hdu = _sparse_table_hdu(filename)
        hdr = get_header(filename, hdu=hdu)
        table = read_table(filename, hdu=hdu)
        nside = int(hdr.get("NSIDE", 0))
        nest = str(hdr.get("ORDERING", "NEST")).strip() == "NEST"
        fv_raw = hdr.get("SENTINEL", float(_healpix.UNSEEN))
        fv = float(fv_raw) if fv_raw is not None else float(_healpix.UNSEEN)
        ns_cov = hdr.get("NSIDECOV", hdr.get("NSIDE_COV"))
        legacy_hierarch = hdr.get("HIERARCH")
        legacy_nside_cov = isinstance(
            legacy_hierarch, str
        ) and legacy_hierarch.lstrip().startswith("NSIDE_COV")
        if ns_cov is None and legacy_nside_cov:
            _, _, ns_cov = legacy_hierarch.partition("=")
            ns_cov = ns_cov.strip()
        ns_cov_i = int(ns_cov) if ns_cov is not None else None

        standard_keys = {
            "SIMPLE",
            "BITPIX",
            "NAXIS",
            "NAXIS1",
            "NAXIS2",
            "EXTEND",
            "COMMENT",
            "HISTORY",
            "XTENSION",
            "PCOUNT",
            "GCOUNT",
            "TFIELDS",
            "EXTNAME",
            "NSIDE",
            "ORDERING",
            "SENTINEL",
            "NSIDE_COV",
            "NSIDECOV",
            "TFORM1",
            "TFORM2",
            "TTYPE1",
            "TTYPE2",
            "TUNIT1",
            "TUNIT2",
        }
        metadata = {}
        for k, v in hdr.items():
            k_upper = str(k).upper()
            if (
                k_upper not in standard_keys
                and not (k_upper == "HIERARCH" and legacy_nside_cov)
                and not k_upper.startswith("TTYPE")
                and not k_upper.startswith("TFORM")
                and not k_upper.startswith("TUNIT")
            ):
                val = v
                if isinstance(val, str):
                    val_s = val.strip()
                    try:
                        if "." in val_s or "E" in val_s.upper():
                            val = float(val_s)
                        else:
                            val = int(val_s)
                    except ValueError:
                        pass
                metadata[k_upper] = val

        pix_key = next((k for k in ("PIXEL", "pixel") if k in table), None)
        val_key = next((k for k in ("VALUE", "value") if k in table), None)
        if pix_key is None or val_key is None:
            keys = list(table.keys()) if hasattr(table, "keys") else []
            raise ValueError(f"SPARSE table missing PIXEL/VALUE columns; have {keys!r}")
        pixels = torch.as_tensor(table[pix_key], dtype=torch.int64).reshape(-1)
        values = torch.as_tensor(table[val_key]).reshape(-1)
        return cls.from_pixels(
            pixels,
            values,
            nside,
            nest=nest,
            nside_coverage=ns_cov_i,
            sentinel=fv,
            metadata=metadata,
        )


def _sparse_table_hdu(path: str) -> str:
    """Return the EXTNAME of the HealSparse ``SPARSE`` binary table extension, checking it exists."""
    import torchfits

    try:
        torchfits.read_header(path, "SPARSE")
    except ValueError as exc:
        raise ValueError(
            f"No FITS extension with EXTNAME 'SPARSE' in {path!r}"
        ) from exc
    return "SPARSE"


def _validate_sparse_operands(
    maps: list[SparseHealpixMap] | tuple[SparseHealpixMap, ...],
) -> tuple[int, bool]:
    if not maps:
        raise ValueError("at least one sparse map is required")
    nside = int(maps[0].nside)
    nest = bool(maps[0].nest)
    for smap in maps:
        if int(smap.nside) != nside or bool(smap.nest) != nest:
            raise ValueError("all sparse maps must have matching nside and ordering")
    return nside, nest


def _promote_sparse_value_dtypes(
    maps: list[SparseHealpixMap] | tuple[SparseHealpixMap, ...],
) -> torch.dtype:
    out_dtype = maps[0].values.dtype
    for smap in maps[1:]:
        out_dtype = torch.promote_types(out_dtype, smap.values.dtype)
    return out_dtype


def sum_union(
    maps: list[SparseHealpixMap] | tuple[SparseHealpixMap, ...],
) -> SparseHealpixMap:
    """Sum sparse maps over the union of valid pixels.

    Missing pixels contribute zero, matching the common healsparse ``sum_union``
    workflow while preserving sparse coverage on the output.
    """
    nside, nest = _validate_sparse_operands(maps)
    device = maps[0].pixels.device
    all_pixels = torch.unique(torch.cat([m.pixels.to(device) for m in maps]))
    out_dtype = _promote_sparse_value_dtypes(maps)
    values = torch.zeros((all_pixels.numel(),), dtype=out_dtype, device=device)
    for smap in maps:
        vals = _to(smap.get_values(all_pixels), dtype=out_dtype, device=device)
        valid = vals != _as_tensor(
            smap.fill_value, dtype=vals.dtype, device=vals.device
        )
        values = values + torch.where(valid, vals, torch.zeros_like(vals))
    return SparseHealpixMap.from_pixels(
        all_pixels,
        values,
        nside_sparse=nside,
        nest=nest,
        nside_coverage=maps[0].nside_coverage,
        sentinel=maps[0].fill_value,
    )


def sum_intersection(
    maps: list[SparseHealpixMap] | tuple[SparseHealpixMap, ...],
) -> SparseHealpixMap:
    """Sum sparse maps only where every input has a valid pixel."""
    nside, nest = _validate_sparse_operands(maps)
    device = maps[0].pixels.device
    common = maps[0].pixels.to(device)
    for smap in maps[1:]:
        common = common[torch.isin(common, smap.pixels.to(device), assume_unique=False)]
    common = torch.unique(common)
    out_dtype = _promote_sparse_value_dtypes(maps)
    values = torch.zeros((common.numel(),), dtype=out_dtype, device=device)
    for smap in maps:
        values = values + _to(smap.get_values(common), dtype=out_dtype, device=device)
    return SparseHealpixMap.from_pixels(
        common,
        values,
        nside_sparse=nside,
        nest=nest,
        nside_coverage=maps[0].nside_coverage,
        sentinel=maps[0].fill_value,
    )


class WideBitMask:
    """Multiword bit mask for flag sets wider than one integer dtype."""

    def __init__(self, n_bits: int, device: torch.device | None = None):
        self.n_bits = n_bits
        self.n_words = (n_bits + 64 - 1) // 64
        self.device = device

    def create_null(self, n_pixels: int) -> Tensor:
        return torch.zeros(
            (n_pixels, self.n_words), dtype=torch.int64, device=self.device
        )


class SkyMaskPipe:
    """Composable HEALPix mask pipeline at a fixed resolution and ordering."""

    def __init__(self, nside: int, nest: bool = True):
        self.nside = nside
        self.nest = nest
        self.stages: dict[str, SparseHealpixMap] = {}

    def add_stage(self, name: str, mask: SparseHealpixMap):
        if mask.nside != self.nside or mask.nest != self.nest:
            raise ValueError("mask is incompatible")
        self.stages[name] = mask

    def combine(self, operation: Literal["and", "or"] = "and") -> SparseHealpixMap:
        if not self.stages:
            raise ValueError("no stages")
        if operation not in {"and", "or"}:
            raise ValueError("operation must be one of {'and', 'or'}")
        stage_names = list(self.stages.keys())
        result_pixels = _positive_mask_pixels(self.stages[stage_names[0]])
        for name in stage_names[1:]:
            other_pixels = _positive_mask_pixels(self.stages[name])
            if operation == "and":
                result_pixels = result_pixels[
                    torch.isin(result_pixels, other_pixels, assume_unique=True)
                ]
            else:
                result_pixels = torch.unique(torch.cat([result_pixels, other_pixels]))

        result_pixels = torch.unique(result_pixels).to(dtype=torch.int64)
        values = torch.ones(
            (result_pixels.numel(),), dtype=torch.float32, device=result_pixels.device
        )
        first = self.stages[stage_names[0]]
        return SparseHealpixMap.from_pixels(
            result_pixels,
            values,
            nside_sparse=self.nside,
            nest=self.nest,
            nside_coverage=first.nside_coverage,
            sentinel=0.0,
        )


def _positive_mask_pixels(mask: SparseHealpixMap) -> Tensor:
    values = mask.values
    valid = values > 0
    if valid.ndim > 1:
        valid = torch.any(valid, dim=tuple(range(valid.ndim - 1)))
    return mask.pixels[valid.reshape(-1)]


__all__ = [
    "HealSparseHostProtocol",
    "SparseHealpixMap",
    "SkyMaskPipe",
    "WideBitMask",
    "sum_intersection",
    "sum_union",
]
