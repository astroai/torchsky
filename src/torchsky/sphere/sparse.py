from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Any, TYPE_CHECKING

import torch
from torch import Tensor

from . import healpix as _healpix

if TYPE_CHECKING:
    from .geom import PixelizedRegion
    from .moc import MOC


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


def _validate_nside_coverage(nside: int, nside_coverage: int) -> None:
    _healpix.nside2npix(nside_coverage)
    if nside_coverage > nside or nside % nside_coverage != 0:
        raise ValueError("nside_coverage must be a power-of-two divisor of nside")


def _copy_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    return dict(metadata) if metadata is not None else None


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

    # Hierarchical extensions (optional)
    nside_coverage: int | None = None
    coverage_map: Tensor | None = None

    def __post_init__(self) -> None:
        _healpix.nside2npix(self.nside)
        if self.pixels.dtype != torch.int64:
            raise ValueError("pixels must be int64")
        if self.pixels.ndim != 1:
            raise ValueError("pixels must be 1D")
        if self.values.shape[-1] != self.pixels.shape[0]:
            raise ValueError("Size of last dimension of values must match pixels")
        if self.pixels.device != self.values.device:
            self.pixels = self.pixels.to(device=self.values.device)
        self.pixels, self.values = _canonicalize_pixels_values(self.pixels, self.values)
        if self.nside_coverage is not None:
            self.nside_coverage = int(self.nside_coverage)
            _validate_nside_coverage(int(self.nside), self.nside_coverage)
            self._build_coverage_map()

    @property
    def nside_sparse(self) -> int:
        return self.nside

    @property
    def sentinel(self) -> float:
        return self.fill_value

    @property
    def pixels_per_cov(self) -> int:
        if self.nside_coverage is None:
            return 1
        ratio = self.nside // self.nside_coverage
        return ratio * ratio

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
    ) -> "SparseHealpixMap":
        vals = torch.as_tensor(dense)
        if nside is None:
            npix = vals.shape[-1]
            ns = _healpix.npix2nside(npix)
        else:
            ns = int(nside)
            npix = _healpix.nside2npix(ns)

        if valid_mask is not None:
            valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=vals.device)
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

        if valid.shape[-1] != npix:
            if valid.numel() == 1 and bool(valid.item()):
                valid = torch.ones((npix,), dtype=torch.bool, device=vals.device)
            else:
                raise ValueError("valid_mask must have shape (npix,)")

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
    ) -> "SparseHealpixMap":
        """Create an empty HealSparseMap with specified configuration."""
        fv = float(sentinel) if sentinel is not None else float(_healpix.UNSEEN)
        dtype = _coerce_torch_dtype(dtype)

        return cls(
            nside=nside_map,
            nest=True,  # HealSparse always uses nesting internally
            pixels=torch.empty((0,), dtype=torch.int64),
            values=torch.empty((0,), dtype=dtype),
            fill_value=fv,
            nside_coverage=nside_coverage,
            metadata=metadata,
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
            return
        npix_cov = _healpix.nside2npix(self.nside_coverage)
        self.coverage_map = torch.full(
            (npix_cov,), -1, dtype=torch.int64, device=self.pixels.device
        )
        if self.pixels.numel() == 0:
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

    def get_covered_pixels(self) -> Tensor:
        return self.pixels

    def to_moc(self) -> "MOC":
        """Create a MOC from this sparse map's coverage."""
        from .moc import MOC

        return MOC.from_sparse_map(self)

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
        """Returns the list of valid pixel indices."""
        return self.pixels

    def get_values(self, pixels: Tensor) -> Tensor:
        pix_flat = torch.as_tensor(
            pixels, dtype=torch.int64, device=self.pixels.device
        ).reshape(-1)
        idx, ok = self._global_to_local(pix_flat)
        out_shape = tuple(self.values.shape[:-1]) + (pix_flat.numel(),)
        fill = torch.full(
            out_shape,
            self.fill_value,
            dtype=self.values.dtype,
            device=self.values.device,
        )
        if self.pixels.numel() == 0:
            return fill
        safe_idx = torch.where(ok, idx, torch.zeros_like(idx))
        gathered = self.values.index_select(-1, safe_idx)
        return torch.where(ok, gathered, fill)

    def get_values_pix(self, pixels: Tensor) -> Tensor:
        """Alias for get_values per healsparse parity."""
        return self.get_values(pixels)

    def update_values_pix(self, pixels: Tensor, values: Tensor) -> None:
        """Update existing or add new pixels to the sparse map."""
        pix_new = torch.as_tensor(pixels, dtype=torch.int64, device=self.pixels.device)
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
            value_t = torch.as_tensor(value, dtype=dtype_t, device=pix.device)
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
                value_t = torch.as_tensor(value, dtype=dtype_t, device=pix.device)
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

    def generate_healpix_map(self) -> Tensor:
        """Alias for to_dense per healsparse parity."""
        return self.to_dense()

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
        pix_t = torch.as_tensor(pix, dtype=torch.int64, device=self.pixels.device)
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
        lon_t = torch.as_tensor(lon_deg, dtype=torch.float64)
        lat_t = torch.as_tensor(lat_deg, dtype=torch.float64)
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
        w4 = w4.to(
            self.values.device,
            dtype=self.values.real.dtype
            if self.values.is_complex()
            else self.values.dtype,
        ).reshape(4, -1)

        out_shape = tuple(self.values.shape[:-1]) + (lon_f.numel(),)
        accum = torch.zeros(
            out_shape, dtype=self.values.dtype, device=self.values.device
        )
        wsum = torch.zeros((lon_f.numel(),), dtype=w4.dtype, device=self.values.device)

        if self.pixels.numel() > 0:
            for k in range(4):
                idx, ok = self._global_to_local(pix4[k])
                safe_idx = torch.where(ok, idx, torch.zeros_like(idx))
                vals = self.values.index_select(-1, safe_idx)
                wk = w4[k]
                wshape = [1] * (vals.ndim - 1) + [wk.shape[-1]]
                contrib = vals * wk.reshape(wshape)
                accum = accum + torch.where(ok, contrib, torch.zeros_like(contrib))
                wsum = wsum + torch.where(ok, wk, torch.zeros_like(wk))

        out = torch.full(
            out_shape,
            float(self.fill_value),
            dtype=self.values.dtype,
            device=self.values.device,
        )
        nz = wsum > 0
        safe_wsum = torch.where(nz, wsum, torch.ones_like(wsum))
        wshape = [1] * (accum.ndim - 1) + [-1]
        filled = accum / safe_wsum.reshape(wshape)
        out = torch.where(nz, filled, out)
        return out.reshape(*self.values.shape[:-1], *shape)

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
                if val_sorted.is_floating_point() or val_sorted.is_complex():
                    base = val_sorted.real if val_sorted.is_complex() else val_sorted
                    goods = torch.isfinite(val_sorted) & (
                        ~torch.isclose(
                            base,
                            torch.as_tensor(fv, dtype=base.dtype, device=base.device),
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

    def write(self, filename: str, clobber: bool = False) -> None:
        """Alias for write_fits per healsparse parity."""
        self.write_fits(filename, overwrite=clobber)

    def write_fits(self, filename: str, overwrite: bool = False) -> None:
        """Write sparse map as a FITS BINTABLE via :func:`torchsky.io.write` (CFITSIO, no Python fitsio)."""
        from ..io import write

        header = {
            "NSIDE": self.nside,
            "ORDERING": "NEST" if self.nest else "RING",
            "SENTINEL": self.fill_value,
            "EXTNAME": "SPARSE",
        }
        if self.nside_coverage:
            # NSIDECOV is the eight-character HEALPix convention; longer FITS
            # keywords require HIERARCH parsing that not every reader preserves.
            header["NSIDECOV"] = self.nside_coverage

        reserved_keys = {
            "NSIDE",
            "ORDERING",
            "SENTINEL",
            "EXTNAME",
            "NSIDE_COV",
            "NSIDECOV",
            "SIMPLE",
            "BITPIX",
            "NAXIS",
            "NAXIS1",
            "NAXIS2",
            "PCOUNT",
            "GCOUNT",
            "TFIELDS",
            "XTENSION",
            "TFORM1",
            "TFORM2",
            "TTYPE1",
            "TTYPE2",
            "TUNIT1",
            "TUNIT2",
        }
        if self.metadata:
            for k, v in self.metadata.items():
                k_upper = str(k).upper()
                if k_upper not in reserved_keys:
                    header[k_upper] = v

        pix = self.pixels.detach().cpu().reshape(-1).to(torch.int64)
        val = self.values.detach().cpu().reshape(-1)
        write(
            filename,
            {"PIXEL": pix, "VALUE": val},
            header=header,
            overwrite=overwrite,
        )

    @classmethod
    def read_fits(cls, filename: str) -> "SparseHealpixMap":
        """Read FITS written by :meth:`write_fits` using :func:`torchsky.io.read_table`."""
        from ..io import get_header, read_table

        hdu = _resolve_sparse_table_hdu_index(filename)
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


def _resolve_sparse_table_hdu_index(path: str) -> int:
    """Return 0-based HDU index of the HealSparse ``SPARSE`` binary table extension."""
    import torchfits.cpp as cpp

    if hasattr(cpp, "resolve_hdu_name_cached"):
        try:
            return int(cpp.resolve_hdu_name_cached(path, "SPARSE"))
        except Exception:
            pass
    i = 0
    while i < 512:
        try:
            h_list = cpp.read_header_dict(path, i)
        except Exception:
            break
        if not h_list:
            break
        ext = h_list.get("EXTNAME")
        if ext == "SPARSE" or (isinstance(ext, str) and ext.strip() == "SPARSE"):
            return i
        i += 1
    raise ValueError(f"No FITS extension with EXTNAME 'SPARSE' in {path!r}")


HealSparseMap = SparseHealpixMap
SparseMap = SparseHealpixMap


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
    out_dtype = torch.result_type(*[m.values for m in maps])
    values = torch.zeros((all_pixels.numel(),), dtype=out_dtype, device=device)
    for smap in maps:
        vals = smap.get_values(all_pixels).to(dtype=out_dtype, device=device)
        valid = vals != torch.as_tensor(
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
    out_dtype = torch.result_type(*[m.values for m in maps])
    values = torch.zeros((common.numel(),), dtype=out_dtype, device=device)
    for smap in maps:
        values = values + smap.get_values(common).to(dtype=out_dtype, device=device)
    return SparseHealpixMap.from_pixels(
        common,
        values,
        nside_sparse=nside,
        nest=nest,
        nside_coverage=maps[0].nside_coverage,
        sentinel=maps[0].fill_value,
    )


class WideBitMask:
    def __init__(self, n_bits: int, device: torch.device | None = None):
        self.n_bits = n_bits
        self.n_words = (n_bits + 64 - 1) // 64
        self.device = device

    def create_null(self, n_pixels: int) -> Tensor:
        return torch.zeros(
            (n_pixels, self.n_words), dtype=torch.int64, device=self.device
        )


class SkyMaskPipe:
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
    "SparseHealpixMap",
    "HealSparseMap",
    "SparseMap",
    "SkyMaskPipe",
    "WideBitMask",
    "sum_intersection",
    "sum_union",
]
