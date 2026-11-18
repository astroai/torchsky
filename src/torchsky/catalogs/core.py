"""Tensor-native catalog helpers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator

import torch

from .._dtype import work_dtype as _work_dtype
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from ._zones import (
    DEFAULT_MAX_PAIRS,
    Combine,
    EngineStats,
    ZoneIndex,
    combine_radii,
    default_zone_height_deg,
    zone_query,
)
from ..device import _as_tensor, _to

if TYPE_CHECKING:
    from .sky_index import SkyIndex


def _identity_partition(partition: Any) -> Any:
    return partition


class CatalogStreamDataset(IterableDataset):
    """Wrap a partition iterator into a PyTorch IterableDataset."""

    def __init__(
        self,
        partitions: Iterable[object] | Callable[[], Iterable[object]],
        loader: Callable[[object], object] = _identity_partition,
    ):
        self._partitions = partitions
        self._loader = loader

    def __iter__(self) -> Iterator[object]:
        worker = get_worker_info()
        partitions = (
            self._partitions() if callable(self._partitions) else self._partitions
        )
        if worker is not None:
            partitions = islice(iter(partitions), worker.id, None, worker.num_workers)
        for partition in partitions:
            yield self._loader(partition)

    @classmethod
    def from_arrow(
        cls,
        source: Any,
        *,
        batch_size: int = 65536,
        columns: tuple[str, str] = ("ra", "dec"),
    ) -> CatalogStreamDataset:
        """Build a CatalogStreamDataset that yields (ra, dec) Tensor pairs from Arrow.

        Parameters
        ----------
        source
            A ``pyarrow.Table``, ``pyarrow.ipc.RecordBatchReader``, or a path
            string / ``pathlib.Path`` to an Arrow IPC Feather file.
        batch_size
            Row count per yielded batch.
        columns
            Column names for RA (degrees) and Dec (degrees).
        """
        import pyarrow as pa

        if isinstance(source, (str, Path)):
            source = pa.ipc.open_file(str(source)).read_all()
        if isinstance(source, pa.Table):
            reader = source.to_batches(max_chunksize=batch_size)
        elif isinstance(source, pa.ipc.RecordBatchReader):
            reader = source
        else:
            batches = list(source) if hasattr(source, "__iter__") else [source]
            reader = iter(batches)

        def _loader(batch: pa.RecordBatch) -> tuple[Tensor, Tensor]:
            col_ra, col_dec = columns
            ra_np = batch.column(col_ra).to_numpy()
            dec_np = batch.column(col_dec).to_numpy()
            return (
                torch.from_numpy(ra_np.copy().astype("f8", copy=False)),
                torch.from_numpy(dec_np.copy().astype("f8", copy=False)),
            )

        return cls(partitions=lambda: iter(reader), loader=_loader)


@dataclass
class CrossmatchResult:
    left_index: Tensor
    right_index: Tensor
    separation_arcsec: Tensor


def _empty_crossmatch(device: torch.device) -> CrossmatchResult:
    empty_i64 = torch.empty(0, dtype=torch.int64, device=device)
    empty_flt = torch.empty(0, dtype=_work_dtype(device), device=device)
    return CrossmatchResult(empty_i64, empty_i64, empty_flt)


def _normalize_radius_arcsec(
    radius_arcsec: float | Tensor,
    n_left: int,
    device: torch.device,
) -> tuple[Tensor, float]:
    """Return per-left radii and the prune radius (max)."""
    dtype = _work_dtype(device)
    radius = _as_tensor(radius_arcsec, dtype=dtype, device=device)
    if radius.ndim == 0:
        value = float(radius.item())
        if value <= 0.0:
            raise ValueError("radius_arcsec must be positive")
        return torch.full((n_left,), value, dtype=dtype, device=device), value
    flat = radius.reshape(-1)
    if int(flat.numel()) != n_left:
        raise ValueError("radius_arcsec must be a scalar or match the left catalog")
    if bool((flat <= 0.0).any()):
        raise ValueError("radius_arcsec must be positive")
    return flat, float(flat.max().item())


def _error_radii(
    value: float | Tensor | None, n: int, device: torch.device, name: str
) -> Tensor | None:
    """Per-row error radii (arcsec): finite and ``>= 0``; ``None`` stays ``None``."""
    if value is None:
        return None
    err = _as_tensor(value, dtype=_work_dtype(device), device=device)
    err = err.reshape(1).expand(n).clone() if err.ndim == 0 else err.reshape(-1)
    if int(err.numel()) != n:
        raise ValueError(f"{name} must be a scalar or have one entry per row")
    if n and not bool(torch.isfinite(err).all() & (err >= 0.0).all()):
        raise ValueError(f"{name} must be finite and non-negative")
    return err


def _distance_filter(
    left_distance: Tensor | None,
    right_distance: Tensor | None,
    distance_tol: float | None,
    n_left: int,
    n_right: int,
    device: torch.device,
) -> Callable[[Tensor, Tensor], Tensor] | None:
    if left_distance is None and right_distance is None and distance_tol is None:
        return None
    if left_distance is None or right_distance is None or distance_tol is None:
        raise ValueError(
            "left_distance, right_distance and distance_tol are given together"
        )
    if not distance_tol >= 0.0:
        raise ValueError("distance_tol must be non-negative")
    dtype = _work_dtype(device)
    dl = _as_tensor(left_distance, dtype=dtype, device=device).reshape(-1)
    dr = _as_tensor(right_distance, dtype=dtype, device=device).reshape(-1)
    if int(dl.numel()) != n_left or int(dr.numel()) != n_right:
        raise ValueError("distance arrays must have one entry per catalogue row")
    tol = float(distance_tol)
    # NaN distance compares False, so rows without a distance never match.
    return lambda lrows, rrows: (dl[lrows] - dr[rrows]).abs() <= tol


def sky_pairs(
    left_ra: Tensor,
    left_dec: Tensor,
    right_ra: Tensor,
    right_dec: Tensor,
    *,
    radius_left: Tensor,
    radius_right: Tensor | None = None,
    find: str = "best1",
    accept: Callable[[Tensor, Tensor], Tensor] | None = None,
    left_chunk_size: int = 65536,
    right_chunk_size: int = DEFAULT_MAX_PAIRS,
    combine: Combine = "sum",
    stats: EngineStats | None = None,
    zones: ZoneIndex | None = None,
    self_upper: bool = False,
) -> CrossmatchResult:
    """Zone-index pairs with ``sep <= combine(radius_left, radius_right)`` for any ``find``.

    ``best2``, ``mutual`` and ``symmetric`` are derived without an ``N * M``
    step: ``best2`` swaps the roles so the per-left best path still applies,
    ``mutual`` intersects the ``best1`` and ``best2`` pair sets, and
    ``symmetric`` resolves the (``K`` pair) candidate list greedily. ``zones``
    is a prebuilt index of the right catalogue.
    """
    from ._pairs import select_pairs

    device = left_ra.device
    if left_ra.numel() == 0 or right_ra.numel() == 0:
        return _empty_crossmatch(device)
    if find == "mutual":
        b1, b2 = (
            sky_pairs(
                left_ra,
                left_dec,
                right_ra,
                right_dec,
                radius_left=radius_left,
                radius_right=radius_right,
                find=mode,
                accept=accept,
                left_chunk_size=left_chunk_size,
                right_chunk_size=right_chunk_size,
                combine=combine,
                zones=zones if mode == "best1" else None,
            )
            for mode in ("best1", "best2")
        )
        n_right = int(right_ra.numel())
        k1 = b1.left_index * n_right + b1.right_index
        k2 = b2.left_index * n_right + b2.right_index
        both = torch.isin(k1, k2)
        return CrossmatchResult(
            b1.left_index[both], b1.right_index[both], b1.separation_arcsec[both]
        )
    if find == "best2":
        zeros = torch.zeros_like(right_ra)
        swapped_accept = None if accept is None else (lambda r, l_: accept(l_, r))
        res = sky_pairs(
            right_ra,
            right_dec,
            left_ra,
            left_dec,
            radius_left=zeros if radius_right is None else radius_right,
            radius_right=radius_left,
            find="best1",
            accept=swapped_accept,
            left_chunk_size=left_chunk_size,
            right_chunk_size=right_chunk_size,
            combine=combine,
        )
        lidx, ridx, sep = select_pairs(
            res.right_index, res.left_index, res.separation_arcsec, "all"
        )
        return CrossmatchResult(lidx, ridx, sep)
    if zones is None:
        window = combine_radii(
            radius_left.detach(),
            0.0 if radius_right is None else float(radius_right.max()),
            combine,
        )
        zones = ZoneIndex(
            right_ra,
            right_dec,
            zone_height_deg=default_zone_height_deg(
                int(right_ra.numel()), float(window.median())
            ),
        )
    li, rj, sep = zone_query(
        zones,
        left_ra,
        left_dec,
        radius_left,
        radius_right=radius_right,
        combine=combine,
        find="best" if find == "best1" else "all",
        accept=accept,
        self_upper=self_upper,
        left_chunk_size=left_chunk_size,
        max_pairs=right_chunk_size,
        stats=stats,
    )
    res = CrossmatchResult(li, rj, sep)
    if find == "symmetric":
        res = CrossmatchResult(
            *select_pairs(
                res.left_index, res.right_index, res.separation_arcsec, "symmetric"
            )
        )
    return res


def crossmatch_sky(
    left_ra_deg: Tensor,
    left_dec_deg: Tensor,
    right_ra_deg: Tensor,
    right_dec_deg: Tensor,
    *,
    radius_arcsec: float | Tensor | None = None,
    left_chunk_size: int = 65536,
    right_chunk_size: int = DEFAULT_MAX_PAIRS,
    find: str = "best",
    index: SkyIndex | None = None,
    device: torch.device | str | None = None,
    left_err_arcsec: float | Tensor | None = None,
    right_err_arcsec: float | Tensor | None = None,
    left_distance: Tensor | None = None,
    right_distance: Tensor | None = None,
    distance_tol: float | None = None,
) -> CrossmatchResult:
    """Chunked exact angular crossmatch on one in-memory tile.

    Matchers (choose one radius rule):

    - **sky**: ``radius_arcsec`` (scalar or per-left-row), pair kept when
      ``separation <= radius``.
    - **skyerr**: ``left_err_arcsec`` / ``right_err_arcsec`` (scalar or per
      row, either side optional, missing side = 0), pair kept when
      ``separation <= err_left[i] + err_right[j]``. Candidate windows use
      ``max(err_left) + max(err_right)``; the exact test uses each pair's sum.
    - **sky + distance** (STILTS ``sky`` with a 1-d cut): also pass
      ``left_distance``, ``right_distance`` and ``distance_tol``; a pair must
      additionally satisfy ``|d_left - d_right| <= distance_tol`` (NaN never
      matches).

    ``find`` selects which pairs are returned:

    - ``"best"`` / ``"best1"``: nearest right row per left row (right rows may
      repeat); ties go to the lowest right index.
    - ``"best2"``: nearest left row per right row (left rows may repeat); ties
      go to the lowest left index.
    - ``"mutual"``: STILTS ``find=best``. Exactly the mutual-best pairs
      (``best1`` intersected with ``best2``, same tie rules); a row whose best
      partner prefers another row is left unmatched.
    - ``"symmetric"``: greedy one-to-one (not STILTS). Ascending separation,
      ties by lowest left then lowest right index; every row is used at most
      once and the assignment is maximal.
    - ``"all"``: every pair inside the threshold.

    ``"all"`` is ordered by left index, separation, then right index; the other
    modes by left index, then right index. Use
    :func:`~torchsky.catalogs.join_rows` for join semantics with unmatched rows.

    ``radius_arcsec`` may be a positive scalar or a per-left-row tensor. Pass
    ``index`` to reuse a :class:`~torchsky.catalogs.SkyIndex` built on the
    right catalogue (any matcher and ``find``). Candidates come from an exact
    zone index; results never depend on chunk sizes, threads or device.
    ``device`` moves inputs onto that device when given. On MPS the work dtype
    is float32, so thresholds below the float32 position resolution are not
    meaningful there.

    Args:
        left_ra_deg: Left-catalog right ascensions in degrees.
        left_dec_deg: Left-catalog declinations in degrees.
        right_ra_deg: Right-catalog right ascensions in degrees.
        right_dec_deg: Right-catalog declinations in degrees.
        radius_arcsec: Scalar or per-left-row match radius in arcseconds.
        left_chunk_size: Maximum left rows processed per outer chunk.
        right_chunk_size: Maximum candidate pairs held per batch.
        find: ``"best"``, ``"best1"``, ``"best2"``, ``"mutual"``, ``"symmetric"`` or ``"all"``.
        index: Reusable :class:`~torchsky.catalogs.SkyIndex` of the right catalog.
        device: Optional compute device.
        left_err_arcsec: Per-left-row error radius (skyerr matcher).
        right_err_arcsec: Per-right-row error radius (skyerr matcher).
        left_distance: Per-left-row distance for the distance cut.
        right_distance: Per-right-row distance for the distance cut.
        distance_tol: Largest allowed distance difference.

    Returns:
        Matching row indices and great-circle separations in arcseconds.

    Examples:
        >>> import torch
        >>> from torchsky.catalogs import crossmatch_sky
        >>> result = crossmatch_sky(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), radius_arcsec=1.0)
        >>> result.right_index.tolist()
        [0]
    """
    from ._pairs import normalize_find

    raw_left_ra = torch.as_tensor(left_ra_deg).reshape(-1)
    work_device = torch.device(device) if device is not None else raw_left_ra.device
    work_dtype = _work_dtype(work_device)
    left_ra = _to(raw_left_ra, device=work_device, dtype=work_dtype)
    left_dec = _as_tensor(left_dec_deg, dtype=work_dtype, device=work_device).reshape(
        -1
    )
    right_ra = _as_tensor(right_ra_deg, dtype=work_dtype, device=work_device).reshape(
        -1
    )
    right_dec = _as_tensor(right_dec_deg, dtype=work_dtype, device=work_device).reshape(
        -1
    )
    n_left, n_right = int(left_ra.numel()), int(right_ra.numel())
    find = normalize_find(find)
    if left_chunk_size <= 0 or right_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if left_ra.shape != left_dec.shape:
        raise ValueError("left_ra_deg and left_dec_deg must have the same shape")
    if right_ra.shape != right_dec.shape:
        raise ValueError("right_ra_deg and right_dec_deg must have the same shape")
    has_err = left_err_arcsec is not None or right_err_arcsec is not None
    if radius_arcsec is None and not has_err:
        raise ValueError("pass radius_arcsec or left_err_arcsec/right_err_arcsec")
    if radius_arcsec is not None and has_err:
        raise ValueError(
            "radius_arcsec and left_err_arcsec/right_err_arcsec are alternatives"
        )
    radius_right: Tensor | None = None
    if has_err:
        radius_left = _error_radii(left_err_arcsec, n_left, work_device, "left_err")
        if radius_left is None:
            radius_left = torch.zeros(n_left, dtype=work_dtype, device=work_device)
        radius_right = _error_radii(right_err_arcsec, n_right, work_device, "right_err")
    else:
        radius_left, _ = _normalize_radius_arcsec(radius_arcsec, n_left, work_device)  # type: ignore[arg-type]
    accept = _distance_filter(
        left_distance, right_distance, distance_tol, n_left, n_right, work_device
    )
    if right_ra.numel() == 0 or left_ra.numel() == 0:
        return _empty_crossmatch(work_device)

    zones = None
    if index is not None:
        index._check_same(right_ra, right_dec)
        zones = index._zones
    return sky_pairs(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_left=radius_left,
        radius_right=radius_right,
        find=find,
        accept=accept,
        left_chunk_size=left_chunk_size,
        right_chunk_size=right_chunk_size,
        zones=zones,
    )
