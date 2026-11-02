"""Memory-aware automatic spatial tiling, host accumulators, footprint skipping and
optional variance propagation of ``coadd_image_plane``."""

from __future__ import annotations

import math

import pytest
import torch

import torchsky.map.image_plane_coadd as coadd_mod
import torchsky.map.image_plane_projection as projection_mod
import torchsky.map.operators.execution as execution_mod
from torchsky.map.image_plane_coadd import coadd_image_plane
from torchsky.map.observation import ImageObservation
from torchsky.map.quality import DQPolicy


def _header(
    ny: int,
    nx: int,
    *,
    ra: float = 12.0,
    dec: float = -3.0,
    scale: float = 0.01,
    angle_deg: float = 0.0,
) -> dict:
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    return {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": (nx + 1) / 2,
        "CRPIX2": (ny + 1) / 2,
        "CRVAL1": ra,
        "CRVAL2": dec,
        "CD1_1": -scale * c,
        "CD1_2": scale * s,
        "CD2_1": scale * s,
        "CD2_2": scale * c,
        "RADESYS": "ICRS",
        "EQUINOX": 2000.0,
    }


def _observations(n: int = 3, size: int = 48, *, with_noise: bool = True, dq=False):
    gen = torch.Generator().manual_seed(7)
    out = []
    # Exposures dithered and rotated across a target larger than any one of them.
    for k in range(n):
        header = _header(
            size,
            size,
            ra=12.0 + 0.12 * (k - (n - 1) / 2) * 1.0,
            dec=-3.0 + 0.10 * ((k % 2) - 0.5),
            scale=0.0105,
            angle_deg=17.0 * k,
        )
        data = torch.randn((size, size), generator=gen) + 3.0
        kwargs = {}
        if with_noise:
            kwargs["weight_map"] = 0.5 + torch.rand((size, size), generator=gen)
        if dq:
            flags = torch.zeros((size, size), dtype=torch.int32)
            flags[5:9, 7:11] = 1
            flags[20, 20] = 2
            kwargs["dq"] = flags
            kwargs["dq_policy"] = DQPolicy(
                reject_bits=1, propagate_bits=1 | 2, unknown_bit_policy="ignore"
            )
        out.append(
            ImageObservation(data=data, header=header, weight=1.0 + 0.5 * k, **kwargs)
        )
    return out


TARGET = _header(150, 170, scale=0.01)
# A budget at which the 150 x 170 bilinear target needs tiles of roughly 45 pixels.
SMALL_BUDGET = 1_400_000


def _assert_same(a, b, *, exact: bool = True) -> None:
    for name in ("data", "weight", "overlap_count", "context"):
        x, y = getattr(a, name), getattr(b, name)
        assert x.shape == y.shape, name
        if exact:
            assert torch.equal(x.cpu(), y.cpu()), name
        else:
            torch.testing.assert_close(x.cpu(), y.cpu(), rtol=1e-6, atol=0.0)
    for name in ("inverse_variance", "correlation_factor"):
        x, y = getattr(a, name), getattr(b, name)
        assert (x is None) == (y is None), name
        if x is not None:
            # Variance terms accumulate squared coefficients whose float32 rounding
            # depends on the vector position of a sample, so tiles may move the last bit.
            torch.testing.assert_close(x.cpu(), y.cpu(), rtol=1e-6, atol=0.0)


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------


def test_plan_does_not_tile_a_target_that_fits() -> None:
    plan = execution_mod.plan_spatial_tiles(
        (512, 512),
        resampling_kernel="lanczos3",
        value_semantics="surface_brightness",
        n_observations=3,
        source_pixels=2048 * 2048,
        device="cpu",
        budget_bytes=4 * 1024**3,
    )
    assert plan.tile_hw is None
    assert not plan.host_accumulators


def test_plan_tile_respects_budget_and_is_balanced() -> None:
    kwargs = dict(
        resampling_kernel="lanczos3",
        value_semantics="surface_brightness",
        n_observations=3,
        source_pixels=2048 * 2048,
        device="cpu",
        budget_bytes=int(1.9 * 1024**3),
    )
    plan = execution_mod.plan_spatial_tiles((2500, 2500), **kwargs)
    assert plan.tile_hw is not None
    th, tw = plan.tile_hw
    assert th <= 2500 and tw <= 2500 and th * tw < 2500 * 2500
    assert plan.tile_bytes <= plan.budget_bytes
    # Equal-sized tiles: no sliver remainder.
    assert th == math.ceil(2500 / math.ceil(2500 / th))
    assert tw == math.ceil(2500 / math.ceil(2500 / tw))
    # Smaller budget -> smaller tiles; richer kernel -> smaller tiles.
    smaller = execution_mod.plan_spatial_tiles(
        (2500, 2500), **{**kwargs, "budget_bytes": int(1.3 * 1024**3)}
    )
    assert smaller.tile_hw[0] * smaller.tile_hw[1] < th * tw
    cheap = execution_mod.plan_spatial_tiles(
        (2500, 2500), **{**kwargs, "resampling_kernel": "bilinear"}
    )
    assert cheap.tile_hw is None or cheap.tile_hw[0] * cheap.tile_hw[1] > th * tw
    flux = execution_mod.plan_spatial_tiles(
        (2500, 2500), **{**kwargs, "value_semantics": "flux"}
    )
    assert flux.tile_hw[0] * flux.tile_hw[1] < th * tw


def test_plan_accelerator_keeps_accumulators_off_the_device() -> None:
    plan = execution_mod.plan_spatial_tiles(
        (20000, 20000),
        resampling_kernel="lanczos3",
        value_semantics="surface_brightness",
        n_observations=100,
        source_pixels=2048 * 2048,
        device="cuda",
        budget_bytes=3 * 1024**3,
    )
    assert plan.tile_hw is not None and plan.host_accumulators
    th, tw = plan.tile_hw
    # The device budget bounds the tile working set only, not the 20k x 20k canvas.
    assert th * tw * plan.bytes_per_pixel <= plan.budget_bytes


def test_budget_precedence(monkeypatch) -> None:
    monkeypatch.delenv("TORCHSKY_TILE_MEMORY_BUDGET_BYTES", raising=False)
    monkeypatch.setattr(execution_mod, "TILE_MEMORY_BUDGET_BYTES", None)
    free = execution_mod.available_device_memory_bytes("cpu")
    assert free is None or free > 0
    base = execution_mod.resolve_tile_memory_budget_bytes("cpu")
    assert base > 0
    if free is not None:
        assert base <= free
    monkeypatch.setattr(execution_mod, "TILE_MEMORY_BUDGET_BYTES", 12345)
    assert execution_mod.resolve_tile_memory_budget_bytes("cpu") == 12345
    monkeypatch.setenv("TORCHSKY_TILE_MEMORY_BUDGET_BYTES", "2e6")
    assert execution_mod.resolve_tile_memory_budget_bytes("cpu") == 2_000_000
    assert (
        execution_mod.resolve_tile_memory_budget_bytes("cpu", budget_bytes=777) == 777
    )
    with pytest.raises(ValueError):
        execution_mod.resolve_tile_memory_budget_bytes("cpu", budget_bytes=0)


def test_cuda_free_memory_uses_mem_get_info(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (3_000, 9_000))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: 700)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 200)
    assert execution_mod.available_device_memory_bytes("cuda") == 3_500


def test_balanced_tile_hw_is_never_a_sliver() -> None:
    assert execution_mod.balanced_tile_hw((100, 100), 10**6) == (100, 100)
    th, tw = execution_mod.balanced_tile_hw((2500, 1000), 1024 * 1024)
    assert th * tw <= 1024 * 1024
    assert th == math.ceil(2500 / math.ceil(2500 / th))
    th, tw = execution_mod.balanced_tile_hw((10, 5000), 1000)
    assert th == 10 and th * tw <= 1000


# ---------------------------------------------------------------------------
# automatic tiling reproduces the single-pass coadd
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", ["bilinear", "lanczos3"])
@pytest.mark.needs_float64
def test_auto_tiling_with_forced_tiny_budget_equals_untiled(kernel) -> None:
    observations = _observations()
    untiled = coadd_image_plane(
        observations,
        target_header=TARGET,
        resampling_kernel=kernel,
        auto_tile=False,
    )
    tiled = coadd_image_plane(
        observations,
        target_header=TARGET,
        resampling_kernel=kernel,
        tile_memory_budget_bytes=SMALL_BUDGET,
    )
    assert tiled.provenance is not None
    th, tw = tiled.provenance["spatial_tiles"]["tile_hw"]
    assert th < 150 or tw < 170, "the tiny budget must actually tile the target"
    assert untiled.inverse_variance is not None
    # Separable kernels go through the chunked sampler, whose vectorised multiply-adds can round the last
    # bit differently depending on where a sample falls in a chunk; tiles change that position.
    _assert_same(untiled, tiled, exact=kernel == "bilinear")
    assert "spatial_tiles" not in (untiled.provenance or {})


def test_auto_tiling_environment_override(monkeypatch) -> None:
    observations = _observations()
    monkeypatch.setenv("TORCHSKY_TILE_MEMORY_BUDGET_BYTES", str(SMALL_BUDGET))
    tiled = coadd_image_plane(
        observations, target_header=TARGET, resampling_kernel="bilinear"
    )
    assert tiled.provenance is not None and "spatial_tiles" in tiled.provenance


def test_target_that_fits_is_not_tiled() -> None:
    observations = _observations()
    result = coadd_image_plane(
        observations,
        target_header=TARGET,
        resampling_kernel="bilinear",
        tile_memory_budget_bytes=8 * 1024**3,
    )
    assert "spatial_tiles" not in (result.provenance or {})


def test_explicit_tile_wins_over_auto_planner() -> None:
    observations = _observations()
    result = coadd_image_plane(
        observations,
        target_header=TARGET,
        resampling_kernel="bilinear",
        spatial_tile_hw=(64, 64),
        tile_memory_budget_bytes=SMALL_BUDGET,
    )
    assert result.provenance["spatial_tiles"]["tile_hw"] == (64, 64)


def test_footprint_order_disables_auto_tiling() -> None:
    observations = _observations(n=2)
    result = coadd_image_plane(
        observations,
        target_header=TARGET,
        resampling_kernel="bilinear",
        footprint_order=6,
        tile_memory_budget_bytes=SMALL_BUDGET,
    )
    assert result.coverage_moc is not None


@pytest.mark.parametrize("stacking", ["median", "sigma_clipped"])
@pytest.mark.needs_float64
def test_auto_tiling_median_and_sigma_clipped_skip_unrelated_exposures(
    stacking, monkeypatch
) -> None:
    observations = _observations(n=4, with_noise=False)
    untiled = coadd_image_plane(
        observations,
        target_header=TARGET,
        resampling_kernel="bilinear",
        stacking_method=stacking,
        auto_tile=False,
    )
    sizes: list[int] = []
    original = coadd_mod.coadd_image_plane

    def counting(obs, **kwargs):
        sizes.append(len(obs))
        return original(obs, **kwargs)

    monkeypatch.setattr(coadd_mod, "coadd_image_plane", counting)
    tiled = coadd_image_plane(
        observations,
        target_header=TARGET,
        resampling_kernel="bilinear",
        stacking_method=stacking,
        tile_memory_budget_bytes=SMALL_BUDGET,
    )
    monkeypatch.undo()
    assert sizes, "tile sub-coadds were expected"
    assert min(sizes) < 4, "tiles must only stack the exposures that reach them"
    assert torch.equal(tiled.data.cpu(), untiled.data.cpu())
    assert torch.equal(tiled.weight.cpu(), untiled.weight.cpu())
    assert torch.equal(tiled.overlap_count.cpu(), untiled.overlap_count.cpu())
    assert torch.equal(tiled.context.cpu(), untiled.context.cpu())


def test_tiled_dq_matches_untiled_dq() -> None:
    observations = _observations(n=3, dq=True)
    untiled = coadd_image_plane(
        observations,
        target_header=TARGET,
        resampling_kernel="bilinear",
        auto_tile=False,
    )
    tiled = coadd_image_plane(
        observations,
        target_header=TARGET,
        resampling_kernel="bilinear",
        spatial_tile_hw=(57, 61),
    )
    assert untiled.dq is not None and tiled.dq is not None
    assert int(untiled.dq.sum()) > 0
    assert torch.equal(tiled.dq, untiled.dq)


# ---------------------------------------------------------------------------
# host-resident accumulators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("semantics", ["surface_brightness", "flux"])
def test_host_accumulator_path_matches_resident_path(semantics, monkeypatch) -> None:
    observations = _observations(n=3)
    kwargs = dict(
        target_header=TARGET,
        resampling_kernel="bilinear",
        value_semantics=semantics,
        spatial_tile_hw=(64, 70),
    )
    resident = coadd_image_plane(observations, **kwargs)

    staged_windows: list[tuple[int, ...]] = []
    original_window = execution_mod.TileAccumulators.window

    def spy(self, y0, y1, x0, x1, **kw):
        assert self.staged
        staged_windows.append((y0, y1, x0, x1))
        return original_window(self, y0, y1, x0, x1, **kw)

    monkeypatch.setattr(coadd_mod, "_FORCE_HOST_ACCUMULATORS", True)
    monkeypatch.setattr(execution_mod.TileAccumulators, "window", spy)
    host = coadd_image_plane(observations, **kwargs)
    assert staged_windows, "windows must be staged through the compute device"
    assert host.provenance["spatial_tiles"]["host_accumulators"] is True
    assert resident.provenance["spatial_tiles"]["host_accumulators"] is False
    _assert_same(resident, host, exact=True)


def test_tile_accumulators_stage_only_the_requested_window() -> None:
    acc = execution_mod.TileAccumulators(
        (6, 8), compute_device=torch.device("cpu"), staged=True
    )
    with acc.window(1, 3, 2, 5) as (n, _d, o, v):
        assert n.shape == (2, 3) and v is not None
        n.add_(1.0)
        o.add_(2)
        v.add_(0.5)
        # The staged copy is detached from storage until the window closes.
        assert float(acc.numer.sum()) == 0.0
    assert float(acc.numer.sum()) == 6.0
    assert int(acc.overlap.sum()) == 12
    assert float(acc.variance.sum()) == 3.0
    with acc.window(0, 6, 0, 8, writeback=False) as (n, *_):
        n.zero_()
    assert float(acc.numer.sum()) == 6.0


# ---------------------------------------------------------------------------
# footprint skipping
# ---------------------------------------------------------------------------


def test_footprint_hull_test_is_conservative_and_prunes_rotated_corners() -> None:
    from torchsky.wcs import WCS

    target = WCS(_header(400, 400, scale=0.01))
    source_header = _header(120, 120, scale=0.01, angle_deg=45.0)
    source = WCS(source_header)
    footprint = coadd_mod._obs_target_footprint(source, (120, 120), target)
    assert footprint is not None
    # Dense reference: target pixels that receive source pixel centres.
    yy, xx = torch.meshgrid(
        torch.arange(120.0, dtype=torch.float64),
        torch.arange(120.0, dtype=torch.float64),
        indexing="ij",
    )
    ra, dec = source.pixel_to_world(xx.reshape(-1), yy.reshape(-1))
    tx, ty = target.world_to_pixel(ra, dec)
    hit = {(round(float(a)), round(float(b))) for a, b in zip(tx, ty)}
    size = 20
    pruned = 0
    for y0 in range(0, 400, size):
        for x0 in range(0, 400, size):
            truly = any(
                (x, y) in hit
                for y in range(y0, y0 + size)
                for x in range(x0, x0 + size)
            )
            decided = footprint.intersects(y0, y0 + size, x0, x0 + size)
            if truly:
                assert decided, (y0, x0)
            in_bbox = not (
                y0 + size <= footprint.bbox[0]
                or y0 >= footprint.bbox[1]
                or x0 + size <= footprint.bbox[2]
                or x0 >= footprint.bbox[3]
            )
            if in_bbox and not decided:
                pruned += 1
    assert pruned > 0, "a 45 degree exposure must have prunable bounding-box corners"


def test_tiles_outside_the_footprint_are_not_projected(monkeypatch) -> None:
    observations = _observations(n=3)
    calls: list[tuple[int, int]] = []
    original = projection_mod._project_observation_to_target_chunk

    def counting(obs, **kwargs):
        calls.append(tuple(kwargs["target_shape"]))
        return original(obs, **kwargs)

    monkeypatch.setattr(coadd_mod, "_project_observation_to_target_chunk", counting)
    coadd_image_plane(
        observations,
        target_header=TARGET,
        resampling_kernel="bilinear",
        spatial_tile_hw=(30, 30),
    )
    n_tiles = math.ceil(150 / 30) * math.ceil(170 / 30)
    assert 0 < len(calls) < len(observations) * n_tiles


# ---------------------------------------------------------------------------
# optional variance propagation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", ["bilinear", "lanczos3", "drizzle_square"])
@pytest.mark.parametrize("tile", [None, (64, 70)])
@pytest.mark.needs_float64("drizzle_square")
def test_propagate_variance_false_only_drops_variance(kernel, tile) -> None:
    observations = _observations(n=2)
    kwargs = dict(
        target_header=TARGET,
        resampling_kernel=kernel,
        spatial_tile_hw=tile,
        auto_tile=False,
    )
    full = coadd_image_plane(observations, **kwargs)
    lean = coadd_image_plane(observations, propagate_variance=False, **kwargs)
    assert full.inverse_variance is not None and full.correlation_factor is not None
    assert lean.inverse_variance is None and lean.correlation_factor is None
    for name in ("data", "weight", "overlap_count", "context"):
        x, y = getattr(full, name), getattr(lean, name)
        if kernel == "lanczos3":
            # The fused sampler adds a variance column, which can change the
            # vectorized multiply-add rounding of the data columns.
            scale = float(x.abs().max())
            torch.testing.assert_close(x, y, rtol=1e-6, atol=1e-6 * scale)
        else:
            assert torch.equal(x, y), name


def test_propagate_variance_false_skips_the_variance_pass(monkeypatch) -> None:
    observations = _observations(n=2)
    seen: list[bool] = []
    original = projection_mod._resample_weighted_to_target

    def spy(*args, **kwargs):
        seen.append(bool(kwargs.get("return_variance_weight", False)))
        return original(*args, **kwargs)

    monkeypatch.setattr(projection_mod, "_resample_weighted_to_target", spy)
    coadd_image_plane(
        observations,
        target_header=TARGET,
        resampling_kernel="lanczos3",
        propagate_variance=False,
        auto_tile=False,
    )
    assert seen and not any(seen)
    seen.clear()
    coadd_image_plane(
        observations,
        target_header=TARGET,
        resampling_kernel="lanczos3",
        auto_tile=False,
    )
    assert seen and all(seen)


@pytest.mark.parametrize("stacking", ["median", "sigma_clipped"])
def test_propagate_variance_false_other_stacks(stacking) -> None:
    observations = _observations(n=3)
    full = coadd_image_plane(
        observations,
        target_header=TARGET,
        resampling_kernel="bilinear",
        stacking_method=stacking,
        auto_tile=False,
    )
    lean = coadd_image_plane(
        observations,
        target_header=TARGET,
        resampling_kernel="bilinear",
        stacking_method=stacking,
        propagate_variance=False,
        auto_tile=False,
    )
    assert lean.inverse_variance is None
    assert torch.equal(full.data, lean.data)
    assert torch.equal(full.weight, lean.weight)


# ---------------------------------------------------------------------------
# Result placement: a tiled coadd returns maps on the requested device.
# ---------------------------------------------------------------------------


def test_restore_to_compute_device_moves_and_warns(monkeypatch) -> None:
    # The meta device stands in for an accelerator: host tensors can be moved to it
    # without hardware, and the copy path / warning path are device agnostic.
    target = torch.device("meta")
    host = [torch.ones(8, 8), None, torch.ones(8, 8, dtype=torch.int64)]
    moved = coadd_mod._restore_to_compute_device(target, host, label="maps")
    assert moved[0].device == target and moved[1] is None and moved[2].device == target

    monkeypatch.setattr(coadd_mod, "available_device_memory_bytes", lambda device: 100)
    with pytest.warns(RuntimeWarning, match="stay on the host"):
        kept = coadd_mod._restore_to_compute_device(target, host, label="maps")
    assert kept[0].device.type == "cpu" and kept[1] is None and kept[2] is host[2]

    # Already on the device (or nothing to move): untouched, no warning.
    same = coadd_mod._restore_to_compute_device(torch.device("cpu"), host, label="maps")
    assert same[0] is host[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("stacking", ["weighted_mean", "median"])
def test_tiled_coadd_on_cuda_returns_cuda_maps(stacking, monkeypatch) -> None:
    observations = _observations(n=3)
    kwargs = dict(
        target_header=TARGET,
        resampling_kernel="bilinear",
        stacking_method=stacking,
        device="cuda",
        auto_tile=False,
    )
    untiled = coadd_image_plane(observations, **kwargs)
    monkeypatch.setattr(coadd_mod, "_FORCE_HOST_ACCUMULATORS", True)
    tiled = coadd_image_plane(observations, spatial_tile_hw=(64, 70), **kwargs)
    for name in ("data", "weight", "overlap_count", "context"):
        t, u = getattr(tiled, name), getattr(untiled, name)
        assert (
            t.device == u.device == torch.device("cuda", torch.cuda.current_device())
        ), name
    assert torch.allclose(tiled.data, untiled.data, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_tiled_coadd_on_cuda_keeps_host_maps_with_warning_when_they_do_not_fit(
    monkeypatch,
) -> None:
    monkeypatch.setattr(coadd_mod, "_FORCE_HOST_ACCUMULATORS", True)
    monkeypatch.setattr(coadd_mod, "available_device_memory_bytes", lambda device: 1)
    with pytest.warns(RuntimeWarning, match="stay on the host"):
        result = coadd_image_plane(
            _observations(n=2),
            target_header=TARGET,
            resampling_kernel="bilinear",
            device="cuda",
            spatial_tile_hw=(64, 70),
            auto_tile=False,
        )
    assert result.data.device.type == "cpu"
