#!/usr/bin/env python3
"""Diagnose stage-level torchsky timing on large-union TAN weighted cases."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BENCHMARKS = ROOT / "benchmarks"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import bench_map_ground_truth as truth_bench  # noqa: E402
import torchsky.map.image_plane as image_plane_mod  # noqa: E402
import torchsky.map.operators.resampling as resampling_mod  # noqa: E402
from torchsky.map import ImageObservation  # noqa: E402
from torchsky.map.operators import (  # noqa: E402
    headers_equivalent_for_direct_copy,
    normalized_header_for_shape,
    target_shape_from_header,
    validate_image_plane_2d,
)


@contextmanager
def _timed_call(target: object, attr: str, sink: list[float]):
    original = getattr(target, attr)

    def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        out = original(*args, **kwargs)
        sink.append((time.perf_counter() - t0) * 1000.0)
        return out

    setattr(target, attr, wrapped)
    try:
        yield
    finally:
        setattr(target, attr, original)


def _is_tpv(header: dict[str, object]) -> bool:
    return "TPV" in str(header.get("CTYPE1", "")) or "TPV" in str(
        header.get("CTYPE2", "")
    )


def _strip_observation(
    obs: ImageObservation,
    *,
    keep_weight: bool,
    keep_mask: bool,
) -> ImageObservation:
    return ImageObservation(
        path=obs.path,
        hdu=obs.hdu,
        weight=obs.weight,
        photometric_scale=obs.photometric_scale,
        background_offset=obs.background_offset,
        wavelength=obs.wavelength,
        spectral_min=obs.spectral_min,
        spectral_max=obs.spectral_max,
        transmission_center=obs.transmission_center,
        transmission_sigma=obs.transmission_sigma,
        band_name=obs.band_name,
        weight_path=obs.weight_path if keep_weight else None,
        weight_hdu=obs.weight_hdu,
        mask_path=obs.mask_path if keep_mask else None,
        mask_hdu=obs.mask_hdu,
    )


def _source_bbox_on_target(
    source_wcs: image_plane_mod.WCS,
    source_shape: tuple[int, int],
    target_wcs: image_plane_mod.WCS,
    target_shape: tuple[int, int],
    *,
    margin: int,
    edge_samples: int,
) -> tuple[int, int, int, int] | None:
    h_src, w_src = int(source_shape[0]), int(source_shape[1])
    edge_samples = max(int(edge_samples), 2)
    xs = image_plane_mod.torch.linspace(
        0.0, float(w_src - 1), steps=edge_samples, dtype=image_plane_mod.torch.float64
    )
    ys = image_plane_mod.torch.linspace(
        0.0, float(h_src - 1), steps=edge_samples, dtype=image_plane_mod.torch.float64
    )
    top_x = xs
    top_y = image_plane_mod.torch.zeros_like(xs)
    bottom_x = xs
    bottom_y = image_plane_mod.torch.full_like(xs, float(h_src - 1))
    left_x = image_plane_mod.torch.zeros_like(ys)
    left_y = ys
    right_x = image_plane_mod.torch.full_like(ys, float(w_src - 1))
    right_y = ys
    x_src = image_plane_mod.torch.cat([top_x, bottom_x, left_x, right_x], dim=0)
    y_src = image_plane_mod.torch.cat([top_y, bottom_y, left_y, right_y], dim=0)
    ra, dec = source_wcs.pixel_to_world(x_src, y_src)
    x_tgt, y_tgt = target_wcs.world_to_pixel(ra, dec)
    finite = image_plane_mod.torch.isfinite(x_tgt) & image_plane_mod.torch.isfinite(
        y_tgt
    )
    if not bool(finite.any()):
        return None
    x_good = x_tgt[finite]
    y_good = y_tgt[finite]
    width = int(target_shape[1])
    height = int(target_shape[0])
    x0 = max(int(image_plane_mod.torch.floor(x_good.min()).item()) - margin, 0)
    x1 = min(int(image_plane_mod.torch.ceil(x_good.max()).item()) + margin + 1, width)
    y0 = max(int(image_plane_mod.torch.floor(y_good.min()).item()) - margin, 0)
    y1 = min(int(image_plane_mod.torch.ceil(y_good.max()).item()) + margin + 1, height)
    if x0 >= x1 or y0 >= y1:
        return None
    return y0, y1, x0, x1


def _slice_prepared_target(
    target: image_plane_mod._PreparedImagePlaneTarget,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> image_plane_mod._PreparedImagePlaneTarget:
    quadrature = None
    if target.quadrature_world_grids is not None:
        quadrature = tuple(
            (ra[y0:y1, x0:x1], dec[y0:y1, x0:x1])
            for ra, dec in target.quadrature_world_grids
        )
    pixel_area = (
        None if target.pixel_area_sr is None else target.pixel_area_sr[y0:y1, x0:x1]
    )
    return image_plane_mod._PreparedImagePlaneTarget(
        header=target.header,
        wcs=target.wcs,
        shape=(y1 - y0, x1 - x0),
        ra_deg=target.ra_deg[y0:y1, x0:x1],
        dec_deg=target.dec_deg[y0:y1, x0:x1],
        center_ra_deg=target.center_ra_deg,
        center_dec_deg=target.center_dec_deg,
        cap_radius_deg=target.cap_radius_deg,
        pixel_area_sr=pixel_area,
        quadrature_world_grids=quadrature,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", default="tan_random10_large_union_rect_weighted_masked"
    )
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--scenario",
        choices=("full", "no_mask", "plain"),
        default="no_mask",
    )
    parser.add_argument(
        "--window-policy",
        choices=("full", "source_bbox"),
        default="full",
    )
    parser.add_argument("--bbox-margin", type=int, default=2)
    parser.add_argument("--bbox-edge-samples", type=int, default=9)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="torchsky_diag_timing_") as tmp:
        tmpdir = Path(tmp)
        observations, target_header, _truth = truth_bench.make_case(
            tmpdir,
            case=args.case,
            size=args.size,
            seed=args.seed,
            width=args.width,
            height=args.height,
        )
        tan_items = [item for item in observations if not _is_tpv(item.header)]
        if args.scenario == "full":
            obs_list = [
                _strip_observation(item.observation, keep_weight=True, keep_mask=True)
                for item in tan_items
            ]
        elif args.scenario == "no_mask":
            obs_list = [
                _strip_observation(item.observation, keep_weight=True, keep_mask=False)
                for item in tan_items
            ]
        else:
            obs_list = [
                _strip_observation(item.observation, keep_weight=False, keep_mask=False)
                for item in tan_items
            ]

        stage_totals_ms = {
            "first_image_read": 0.0,
            "target_prepare": 0.0,
            "image_read": 0.0,
            "wcs_init": 0.0,
            "weight_load": 0.0,
            "calibration": 0.0,
            "window_select": 0.0,
            "resample_total": 0.0,
            "world_to_pixel": 0.0,
            "sampling": 0.0,
            "accumulation": 0.0,
        }
        rows: list[dict[str, object]] = []

        t_total0 = time.perf_counter()
        t0 = time.perf_counter()
        first = obs_list[0]
        first_image, first_header = image_plane_mod.read(
            first.path, hdu=first.hdu, return_header=True
        )
        stage_totals_ms["first_image_read"] = (time.perf_counter() - t0) * 1000.0
        first_image = validate_image_plane_2d(first_image)
        header_dict = normalized_header_for_shape(
            dict(first_header),
            tuple(int(v) for v in first_image.shape[-2:]),
            overrides=target_header,
        )
        target_shape = target_shape_from_header(
            header_dict,
            fallback=tuple(int(v) for v in first_image.shape[-2:]),
        )
        device = first_image.device
        prepared_target: image_plane_mod._PreparedImagePlaneTarget | None = None

        def ensure_prepared_target() -> image_plane_mod._PreparedImagePlaneTarget:
            nonlocal prepared_target
            if prepared_target is None:
                prep0 = time.perf_counter()
                prepared_target = image_plane_mod._prepare_target(
                    header=header_dict,
                    target_wcs=image_plane_mod.WCS(header_dict),
                    target_shape=target_shape,
                    device=device,
                    include_pixel_area=False,
                    include_quadrature=False,
                )
                stage_totals_ms["target_prepare"] += (
                    time.perf_counter() - prep0
                ) * 1000.0
            return prepared_target

        numer = image_plane_mod.torch.zeros(
            target_shape, dtype=image_plane_mod.torch.float32, device=device
        )
        denom = image_plane_mod.torch.zeros(
            target_shape, dtype=image_plane_mod.torch.float32, device=device
        )
        overlap_count = image_plane_mod.torch.zeros(
            target_shape, dtype=image_plane_mod.torch.int32, device=device
        )

        for obs_idx, obs in enumerate(obs_list):
            if obs_idx == 0:
                image = first_image
                header_dict_obs = dict(first_header)
                image_read_ms = 0.0
            else:
                read0 = time.perf_counter()
                image, header = image_plane_mod.read(
                    obs.path, hdu=obs.hdu, return_header=True
                )
                image_read_ms = (time.perf_counter() - read0) * 1000.0
                stage_totals_ms["image_read"] += image_read_ms
                image = validate_image_plane_2d(image)
                header_dict_obs = dict(header)
            image = image.to(device=device)

            direct_copy = (
                headers_equivalent_for_direct_copy(
                    header_dict_obs, header_dict
                )
                and tuple(int(v) for v in image.shape) == target_shape
            )
            need_prepared_target = (
                (
                    True
                    if obs.weight_path is not None or obs.mask_path is not None
                    else False
                )
                or header_dict_obs != header_dict
                or tuple(int(v) for v in image.shape) != target_shape
            )

            source_wcs = None
            wcs_init_ms = 0.0
            if need_prepared_target and not direct_copy:
                wcs0 = time.perf_counter()
                source_wcs = image_plane_mod.WCS(header_dict_obs)
                wcs_init_ms = (time.perf_counter() - wcs0) * 1000.0
                stage_totals_ms["wcs_init"] += wcs_init_ms

            load0 = time.perf_counter()
            source_weight = image_plane_mod._load_observation_weight_map(
                obs, image, device=device
            )
            load_ms = (time.perf_counter() - load0) * 1000.0
            stage_totals_ms["weight_load"] += load_ms

            cal0 = time.perf_counter()
            image_cal, source_weight = image_plane_mod._apply_observation_calibration(
                obs, image, source_weight
            )
            cal_ms = (time.perf_counter() - cal0) * 1000.0
            stage_totals_ms["calibration"] += cal_ms

            resample_ms = 0.0
            world_to_pixel_ms = 0.0
            sampling_ms = 0.0
            window_select_ms = 0.0
            window_fraction = 1.0
            window_shape = list(target_shape)
            if direct_copy:
                sampled_weight = source_weight.to(dtype=image_plane_mod.torch.float32)
                sampled_numer = (
                    image_cal.to(dtype=image_plane_mod.torch.float32) * sampled_weight
                )
            else:
                target = ensure_prepared_target()
                target_view = target
                y0 = 0
                y1 = int(target_shape[0])
                x0 = 0
                x1 = int(target_shape[1])
                if args.window_policy == "source_bbox":
                    bbox0 = time.perf_counter()
                    bbox = _source_bbox_on_target(
                        source_wcs,
                        tuple(int(v) for v in image.shape),
                        target.wcs,
                        target_shape,
                        margin=args.bbox_margin,
                        edge_samples=args.bbox_edge_samples,
                    )
                    window_select_ms = (time.perf_counter() - bbox0) * 1000.0
                    stage_totals_ms["window_select"] += window_select_ms
                    if bbox is not None:
                        y0, y1, x0, x1 = bbox
                        target_view = _slice_prepared_target(
                            target,
                            y0=y0,
                            y1=y1,
                            x0=x0,
                            x1=x1,
                        )
                        window_fraction = float(
                            ((y1 - y0) * (x1 - x0))
                            / float(target_shape[0] * target_shape[1])
                        )
                        window_shape = [int(y1 - y0), int(x1 - x0)]
                world_to_pixel_calls: list[float] = []
                sampling_calls: list[float] = []
                resample0 = time.perf_counter()
                with _timed_call(source_wcs, "world_to_pixel", world_to_pixel_calls):
                    with _timed_call(
                        resampling_mod,
                        "sample_image_plane_batched_channels",
                        sampling_calls,
                    ):
                        sampled_numer, sampled_weight = (
                            image_plane_mod._resample_weighted_to_target(
                                image_cal,
                                source_weight,
                                source_wcs,
                                target_view,
                                resampling_kernel="bilinear",
                                value_semantics="surface_brightness",
                            )
                        )
                resample_ms = (time.perf_counter() - resample0) * 1000.0
                world_to_pixel_ms = float(sum(world_to_pixel_calls))
                sampling_ms = float(sum(sampling_calls))
                stage_totals_ms["resample_total"] += resample_ms
                stage_totals_ms["world_to_pixel"] += world_to_pixel_ms
                stage_totals_ms["sampling"] += sampling_ms
                full_sampled_numer = image_plane_mod.torch.zeros(
                    target_shape, dtype=sampled_numer.dtype, device=device
                )
                full_sampled_weight = image_plane_mod.torch.zeros(
                    target_shape, dtype=sampled_weight.dtype, device=device
                )
                full_sampled_numer[y0:y1, x0:x1] = sampled_numer
                full_sampled_weight[y0:y1, x0:x1] = sampled_weight
                sampled_numer = full_sampled_numer
                sampled_weight = full_sampled_weight

            overlap = (sampled_weight > 0.0).to(image_plane_mod.torch.int32)
            acc0 = time.perf_counter()
            numer = numer + sampled_numer
            denom = denom + sampled_weight
            overlap_count = overlap_count + overlap
            accumulation_ms = (time.perf_counter() - acc0) * 1000.0
            stage_totals_ms["accumulation"] += accumulation_ms

            coverage_fraction = float(
                (sampled_weight > 0.0)
                .to(dtype=image_plane_mod.torch.float32)
                .mean()
                .item()
            )
            rows.append(
                {
                    "path": obs.path,
                    "has_weight_map": obs.weight_path is not None,
                    "has_mask_map": obs.mask_path is not None,
                    "width": int(header_dict_obs["NAXIS1"]),
                    "height": int(header_dict_obs["NAXIS2"]),
                    "projection": str(header_dict_obs.get("CTYPE1", "")),
                    "image_read_ms": image_read_ms,
                    "wcs_init_ms": wcs_init_ms,
                    "weight_load_ms": load_ms,
                    "calibration_ms": cal_ms,
                    "window_select_ms": window_select_ms,
                    "resample_total_ms": resample_ms,
                    "world_to_pixel_ms": world_to_pixel_ms,
                    "sampling_ms": sampling_ms,
                    "resample_other_ms": max(
                        resample_ms - world_to_pixel_ms - sampling_ms, 0.0
                    ),
                    "accumulation_ms": accumulation_ms,
                    "used_reprojection": source_wcs is not None,
                    "same_target_shape": tuple(int(v) for v in image.shape)
                    == tuple(int(v) for v in target_shape),
                    "coverage_fraction": coverage_fraction,
                    "window_fraction": window_fraction,
                    "window_shape": window_shape,
                }
            )

        coadd = image_plane_mod.torch.where(
            denom > 0.0,
            numer / denom,
            image_plane_mod.torch.zeros_like(numer),
        )
        total_ms = (time.perf_counter() - t_total0) * 1000.0

    rows.sort(
        key=lambda row: float(row["resample_total_ms"] + row["image_read_ms"]),
        reverse=True,
    )
    summary = {
        "case": args.case,
        "scenario": args.scenario,
        "window_policy": args.window_policy,
        "size": args.size,
        "width": args.width,
        "height": args.height,
        "n_observations": len(obs_list),
        "total_time_ms": total_ms,
        "stage_totals_ms": stage_totals_ms,
        "output_mean_weight": float(denom.mean().item()),
        "output_overlap_fraction": float(
            (denom > 0.0).to(dtype=image_plane_mod.torch.float32).mean().item()
        ),
        "output_mean_data": float(coadd.mean().item()),
        "per_observation": rows,
    }
    text = json.dumps(summary, indent=2)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
