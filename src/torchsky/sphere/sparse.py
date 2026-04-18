from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Any, TYPE_CHECKING

import torch
from torch import Tensor

from ..wcs import healpix as _healpix

if TYPE_CHECKING:
    from .moc import MOC


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
    ) -> "SparseHealpixMap":
        vals = torch.as_tensor(dense)
        if nside is None:
            npix = vals.shape[-1]
            ns = _healpix.npix2nside(npix)
        else:
            ns = int(nside)
            npix = _healpix.nside2npix(ns)

        if valid_mask is not None:
            valid = torch.as_tensor(valid_mask, dtype=torch.bool)
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
        )
        if nside_coverage is not None:
            obj._build_coverage_map()
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
        )

        if nside_coverage is not None:
            obj._build_coverage_map()
        return obj

    @classmethod
    def convert_healpix_map(
        cls,
        dense: Tensor,
        nside_coverage: int,
        nest: bool = True,
        sentinel: float | None = None,
    ) -> "SparseHealpixMap":
        fv = float(sentinel) if sentinel is not None else float(_healpix.UNSEEN)
        # Create from dense normally
        smap = cls.from_dense(dense, nest=nest, fill_value=fv)
        smap.nside_coverage = nside_coverage
        smap._build_coverage_map()
        return smap

    def _build_coverage_map(self):
        if self.nside_coverage is None:
            return
        npix_cov = _healpix.nside2npix(self.nside_coverage)
        self.coverage_map = torch.full(
            (npix_cov,), -1, dtype=torch.int64, device=self.pixels.device
        )
        if self.pixels.numel() == 0:
            return

        ratio = self.nside // self.nside_coverage
        ratio2 = ratio * ratio
        cov_pixels = torch.div(self.pixels, ratio2, rounding_mode="floor")

        # Vectorized assignment of the *first* index to coverage_map
        # We flip the indices so the scatter overwrites duplicates, leaving the lowest (first) index intact.
        arange = torch.arange(cov_pixels.numel(), dtype=torch.int64, device=self.pixels.device)
        self.coverage_map.scatter_(0, cov_pixels.flip([0]), arange.flip([0]))

    def get_covered_pixels(self) -> Tensor:
        return self.pixels

    def get_values(self, pixels: Tensor) -> Tensor:
        idx, ok = self._global_to_local(pixels)
        out_shape = tuple(self.values.shape[:-1]) + (pixels.numel(),)
        out = torch.full(
            out_shape,
            self.fill_value,
            dtype=self.values.dtype,
            device=self.values.device,
        )
        if bool(ok.any()):
            out[..., ok] = self.values.index_select(-1, idx[ok])
        return out

    @classmethod
    def from_moc(
        cls, moc: "MOC", nside: int, nside_coverage: int | None = None
    ) -> "SparseHealpixMap":
        # Project MOC to nside
        mask = moc.flatten(nside)
        return cls.from_dense(
            mask.to(torch.float32), nside=nside, nside_coverage=nside_coverage
        )

    def to_dense(self, fill_value: float | None = None) -> Tensor:
        fv = self.fill_value if fill_value is None else float(fill_value)
        npix = _healpix.nside2npix(self.nside)
        out_shape = tuple(self.values.shape[:-1]) + (npix,)
        out = torch.full(
            out_shape, fv, dtype=self.values.dtype, device=self.values.device
        )
        if self.pixels.numel() > 0:
            out.index_copy_(-1, self.pixels, self.values)
        return out

    @property
    def coverage_mask(self) -> Tensor:
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
        ok = torch.zeros_like(in_bounds, dtype=torch.bool)
        if bool(in_bounds.any()):
            val_at_idx = self.pixels.index_select(
                0, torch.clamp(idx[in_bounds], 0, self.pixels.numel() - 1)
            )
            ok[in_bounds] = val_at_idx == pix_t[in_bounds]
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

        for k in range(4):
            idx, ok = self._global_to_local(pix4[k])
            if not bool(ok.any()):
                continue
            sel = idx[ok]
            wk = w4[k, ok]
            vals = self.values.index_select(-1, sel)
            wshape = [1] * (vals.ndim - 1) + [wk.numel()]
            accum[..., ok] = accum[..., ok] + vals * wk.reshape(wshape)
            wsum[ok] = wsum[ok] + wk

        out = torch.full(
            out_shape,
            float(self.fill_value),
            dtype=self.values.dtype,
            device=self.values.device,
        )
        nz = wsum > 0
        if bool(nz.any()):
            wshape = [1] * (accum.ndim - 1) + [-1]
            out[..., nz] = accum[..., nz] / wsum[nz].reshape(wshape)
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

        if nside_out == int(self.nside):
            if fill_value is None:
                return self
            return SparseHealpixMap(
                nside=self.nside,
                nest=self.nest,
                pixels=self.pixels,
                values=self.values,
                fill_value=fv,
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
                    dtype=val_sorted.dtype,
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
                sum_vals.index_add_(
                    -1, group_ids, val_sorted * goods.to(dtype=val_sorted.dtype)
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
                out_val = torch.zeros_like(sum_vals)
                nz = nhit_f != 0
                out_val[..., nz] = sum_vals[..., nz] / nhit_f[..., nz].to(
                    dtype=sum_vals.dtype
                )
                badout = (nhit != float(parent_mult)) if pess else (nhit == 0.0)
                out_val = out_val.clone()
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
            dense_out, nside=nside_out, nest=self.nest, valid_mask=valid, fill_value=fv
        )

    def write_fits(self, filename: str, overwrite: bool = False) -> None:
        """Write sparse map as a FITS BINTABLE via :func:`torchsky.write` (CFITSIO, no Python fitsio)."""
        import torchsky

        header = {
            "NSIDE": self.nside,
            "ORDERING": "NEST" if self.nest else "RING",
            "SENTINEL": self.fill_value,
            "EXTNAME": "SPARSE",
        }
        if self.nside_coverage:
            header["NSIDE_COV"] = self.nside_coverage
        pix = self.pixels.detach().cpu().reshape(-1).to(torch.int64)
        val = self.values.detach().cpu().reshape(-1).to(torch.float32)
        torchsky.write(
            filename,
            {"PIXEL": pix, "VALUE": val},
            header=header,
            overwrite=overwrite,
        )

    @classmethod
    def read_fits(cls, filename: str) -> "SparseHealpixMap":
        """Read FITS written by :meth:`write_fits` using :func:`torchsky.read_table`."""
        import torchsky

        hdu = _resolve_sparse_table_hdu_index(filename)
        hdr = torchsky.get_header(filename, hdu=hdu)
        table = torchsky.read_table(filename, hdu=hdu)
        nside = int(hdr.get("NSIDE", 0))
        nest = str(hdr.get("ORDERING", "NEST")).strip() == "NEST"
        fv_raw = hdr.get("SENTINEL", float(_healpix.UNSEEN))
        fv = float(fv_raw) if fv_raw is not None else float(_healpix.UNSEEN)
        ns_cov = hdr.get("NSIDE_COV")
        ns_cov_i = int(ns_cov) if ns_cov is not None else None

        pix_key = next((k for k in ("PIXEL", "pixel") if k in table), None)
        val_key = next((k for k in ("VALUE", "value") if k in table), None)
        if pix_key is None or val_key is None:
            keys = list(table.keys()) if hasattr(table, "keys") else []
            raise ValueError(f"SPARSE table missing PIXEL/VALUE columns; have {keys!r}")
        pixels = torch.as_tensor(table[pix_key], dtype=torch.int64).reshape(-1)
        values = torch.as_tensor(table[val_key], dtype=torch.float32).reshape(-1)
        return cls.from_pixels(
            pixels, values, nside, nest=nest, nside_coverage=ns_cov_i, sentinel=fv
        )


def _resolve_sparse_table_hdu_index(path: str) -> int:
    """Return 0-based HDU index of the HealSparse ``SPARSE`` binary table extension."""
    import torchsky.cpp as cpp

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
        stage_names = list(self.stages.keys())
        result = self.stages[stage_names[0]]
        for name in stage_names[1:]:
            other = self.stages[name]
            d1 = result.to_dense(fill_value=0)
            d2 = other.to_dense(fill_value=0)
            res_dense = (
                torch.logical_and(d1 > 0, d2 > 0).to(torch.float32)
                if operation == "and"
                else torch.logical_or(d1 > 0, d2 > 0).to(torch.float32)
            )
            result = SparseHealpixMap.from_dense(
                res_dense, nside=self.nside, nest=self.nest, fill_value=0
            )
        return result


__all__ = [
    "SparseHealpixMap",
    "HealSparseMap",
    "SparseMap",
    "SkyMaskPipe",
    "WideBitMask",
]
