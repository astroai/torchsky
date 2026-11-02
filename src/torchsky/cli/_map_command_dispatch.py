"""Argument validation and dispatch for ``torchsky map``."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from ..device import default_compute_device
from ..map.plan import load_map_plan


from ._map_command_filters import (
    _run_coadd_per_filter,
    _run_field,
)
from ._map_command_inputs import (
    _field_spectra_from_plan,
    _map_read_device_str,
    _resolve_map_workflow,
)


def _dispatch_map_from_plan(args: argparse.Namespace) -> int:
    if args.inputs:
        raise SystemExit("--from-plan cannot be combined with input paths")
    if getattr(args, "plan_json", None) is not None:
        raise SystemExit("--from-plan cannot be combined with --plan-json")
    if (
        getattr(args, "artifact", None) is not None
        or getattr(args, "resume", None) is not None
    ):
        raise SystemExit(
            "--from-plan uses the recorded artifact; only output and scratch may be overridden"
        )
    try:
        plan = load_map_plan(args.from_plan, verify_inputs=True)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"torchsky map: cannot replay plan: {exc}") from exc
    payload = plan.payload
    execution = payload["execution"]
    target = payload["target"]
    runtime = payload["runtime"]
    resolved_inputs = payload["inputs"]
    args._resolved_plan_inputs = resolved_inputs
    args._plan_target_header = target["header"]
    args._plan_tile_hw = execution.get("tile_hw")
    args.inputs = [str(item["identity"]["path"]) for item in resolved_inputs]
    args.weights = [float(item.get("weight", 1.0)) for item in resolved_inputs]
    args.photometric_scales = [
        float(item.get("photometric_scale", 1.0)) for item in resolved_inputs
    ]
    args.background_offsets = [
        float(item.get("background_offset", 0.0)) for item in resolved_inputs
    ]
    args.weight_maps = [item.get("weight_map") for item in resolved_inputs]
    args.mask_maps = [item.get("mask_map") for item in resolved_inputs]
    args.primary_beam_maps = [item.get("primary_beam_map") for item in resolved_inputs]
    args.dq_maps = [item.get("dq_map") for item in resolved_inputs]
    args.good_bits = None
    args._coadd_band = next(
        (
            item.get("band")
            for item in resolved_inputs
            if isinstance(item.get("band"), str)
        ),
        None,
    )
    args.weight_map_hdu = int(
        next(
            (
                item["weight_identity"].get("hdu", 0)
                for item in resolved_inputs
                if item.get("weight_identity") is not None
            ),
            0,
        )
    )
    args.mask_map_hdu = int(
        next(
            (
                item["mask_identity"].get("hdu", 0)
                for item in resolved_inputs
                if item.get("mask_identity") is not None
            ),
            0,
        )
    )
    args.dq_hdu = next(
        (
            item["dq_identity"].get("hdu", "DQ")
            for item in resolved_inputs
            if item.get("dq_identity") is not None
        ),
        "DQ",
    )
    args.primary_beam_hdu = next(
        (
            item["primary_beam_identity"].get("hdu")
            for item in resolved_inputs
            if item.get("primary_beam_identity") is not None
        ),
        None,
    )
    args.device = str(runtime["device"])
    args.scratch_dir = (
        args.scratch_dir if args.scratch_dir is not None else runtime.get("scratch")
    )
    args.output_planes = list(payload["requested_products"])
    args.reference = None
    args.reference_hdu = int(target.get("reference_hdu", 0))
    args.plan_reference = 0
    args.union_footprint = False
    args.footprint_padding = float(target.get("footprint_padding", 1.0))
    args.footprint_tolerance_px = float(target.get("footprint_tolerance_px", 0.25))
    args.footprint_max_depth = int(target.get("footprint_max_depth", 20))
    args.max_footprint_pixels = int(target["shape"][0]) * int(target["shape"][1])
    args.hdu = 0
    args.plan_json = None
    args.from_plan = None

    if payload["method"] == "field":
        artifact = execution.get("artifact")
        artifact_identities = execution.get("artifact_identities")
        if not isinstance(artifact, str) or not artifact:
            raise SystemExit(
                "torchsky map: field plan replay requires a recorded artifact"
            )
        if not isinstance(artifact_identities, list) or not artifact_identities:
            raise SystemExit(
                "torchsky map: field plan artifact has no content identity"
            )
        try:
            bands, spectral_kws = _field_spectra_from_plan(
                payload.get("spectral"), input_count=len(resolved_inputs)
            )
        except ValueError as exc:
            raise SystemExit(f"torchsky map: invalid field plan: {exc}") from exc
        args.method = "field"
        args._plan_observation_bands = bands
        args._plan_spectral_kws = spectral_kws
        args._plan_field_model_kind = str(execution["field_model"])
        args._plan_field_model_kwargs = dict(execution["field_model_kwargs"])
        args._plan_field_preset = execution["field_preset"]
        args._artifact_render_only = True
        args.artifact = None
        args.resume = Path(artifact)
        args.field_model = None
        args.field_steps = int(execution["field_steps"])
        args.field_lr = float(execution["field_learning_rate"])
        args.field_patch_size = int(execution["field_patch_size"])
        args.field_patches_per_obs = int(execution["field_patches_per_observation"])
        args.field_obs_per_step = int(execution["field_observations_per_step"])
        args.field_spectral_samples = int(execution["field_spectral_samples"])
        args.field_spatial_sampling = str(execution["field_spatial_sampling"])
        args.field_amp = str(execution["field_amp_dtype"])
        render_tile = execution["field_render_tile"]
        args.field_render_tile = (
            int(render_tile)
            if render_tile is not None
            else max(int(target["shape"][0]), int(target["shape"][1]))
        )
        args.field_max_pixels = args.max_footprint_pixels
        args.per_filter = False
        return _core_mod._run_field_single(args)

    args.method = "coadd"
    args.workflow = str(execution["workflow"])
    args.kernel = str(execution["kernel"])
    args.stack = str(execution["stack"])
    args.background_mode = str(execution["background_mode"])
    args.sky = False
    args.drizzle_pixfrac = float(execution["drizzle_pixfrac"])
    args.clip_sigma = float(execution.get("clip_sigma", 4.0))
    args.clip_iterations = int(execution.get("clip_iterations", 3))
    args.value_semantics = str(execution["value_semantics"])
    args.auto_tile = False
    args.tile_pixels = None
    args.tile_above_megapixels = 32.0
    return _core_mod._run_coadd(args)


def _validate_map_option_matrix(args: argparse.Namespace) -> None:
    """Fail closed on options that belong to the other ``--method``."""
    if str(getattr(args, "method", "coadd")) == "field":
        if any(p is not None for p in (getattr(args, "dq_maps", None) or ())):
            raise SystemExit("--dq-maps require --method=coadd")
        return
    field_only = (
        ("--field-band", getattr(args, "field_band", None)),
        ("--field-model", getattr(args, "field_model", None)),
        ("--artifact", getattr(args, "artifact", None)),
        ("--resume", getattr(args, "resume", None)),
        ("--field-transmission", getattr(args, "field_transmission", None)),
        ("--field-transmission-dir", getattr(args, "field_transmission_dir", None)),
        (
            "--field-gaussian-lambda-eff",
            getattr(args, "field_gaussian_lambda_eff", None),
        ),
        (
            "--field-gaussian-lambda-min",
            getattr(args, "field_gaussian_lambda_min", None),
        ),
        (
            "--field-gaussian-lambda-max",
            getattr(args, "field_gaussian_lambda_max", None),
        ),
        ("--field-gaussian-sigma", getattr(args, "field_gaussian_sigma", None)),
        ("--svo-filter-id", getattr(args, "svo_filter_id", None)),
    )
    used = [option for option, value in field_only if value is not None]
    if used:
        raise SystemExit(f"{', '.join(used)} require --method=field")


def _dispatch_map(args: argparse.Namespace) -> int:
    if getattr(args, "from_plan", None) is not None:
        return _dispatch_map_from_plan(args)
    _resolve_map_workflow(args)
    if "SCI" not in getattr(args, "output_planes", ("SCI",)):
        raise SystemExit("--output-planes must include SCI")
    expanded = _core_mod._expand_inputs(args.inputs)
    n_inputs = len(expanded)
    if not hasattr(args, "weight_maps"):
        args.weight_maps = None
    if not hasattr(args, "mask_maps"):
        args.mask_maps = None
    if not hasattr(args, "primary_beam_maps"):
        args.primary_beam_maps = None
    if not hasattr(args, "photometric_scales"):
        args.photometric_scales = None
    if not hasattr(args, "background_offsets"):
        args.background_offsets = None
    if not hasattr(args, "dq_maps") or args.dq_maps is None:
        args.dq_maps = [None] * n_inputs
    args._weights_are_inverse_variance = args.weights is not None
    if args.weights is None:
        args.weights = [1.0] * n_inputs
    if len(args.weights) != n_inputs:
        raise SystemExit("--weights count must match expanded input file count")
    args.weights = list(args.weights)
    if any(not math.isfinite(float(w)) or float(w) <= 0.0 for w in args.weights):
        raise SystemExit("--weights must be finite and positive")
    if args.photometric_scales is None:
        args.photometric_scales = [1.0] * n_inputs
    if len(args.photometric_scales) != n_inputs:
        raise SystemExit(
            "--photometric-scales count must match expanded input file count"
        )
    args.photometric_scales = list(args.photometric_scales)
    if any(
        not math.isfinite(float(s)) or float(s) <= 0.0 for s in args.photometric_scales
    ):
        raise SystemExit("--photometric-scales must be finite and positive")
    if args.background_offsets is None:
        args.background_offsets = [0.0] * n_inputs
    if len(args.background_offsets) != n_inputs:
        raise SystemExit(
            "--background-offsets count must match expanded input file count"
        )
    args.background_offsets = list(args.background_offsets)
    if any(not math.isfinite(float(b)) for b in args.background_offsets):
        raise SystemExit("--background-offsets must be finite")
    if args.weight_maps is None:
        args.weight_maps = [None] * n_inputs
    if len(args.weight_maps) != n_inputs:
        raise SystemExit("--weight-maps count must match expanded input file count")
    args.weight_maps = list(args.weight_maps)
    if args.mask_maps is None:
        args.mask_maps = [None] * n_inputs
    if len(args.mask_maps) != n_inputs:
        raise SystemExit("--mask-maps count must match expanded input file count")
    args.mask_maps = list(args.mask_maps)
    if args.primary_beam_maps is None:
        args.primary_beam_maps = [None] * n_inputs
    if len(args.primary_beam_maps) != n_inputs:
        raise SystemExit(
            "--primary-beam-maps count must match expanded input file count"
        )
    args.primary_beam_maps = list(args.primary_beam_maps)
    if len(args.dq_maps) != n_inputs:
        raise SystemExit("--dq-maps count must match expanded input file count")
    args.dq_maps = list(args.dq_maps)
    field_method = args.method == "field"
    _validate_map_option_matrix(args)
    field_amp = str(getattr(args, "field_amp", "none"))
    if not field_method and (
        field_amp != "none"
        or getattr(args, "artifact", None) is not None
        or getattr(args, "resume", None) is not None
    ):
        raise SystemExit("--field-amp, --artifact, and --resume require --method=field")
    if field_method and field_amp != "none":
        amp_device = (
            torch.device(args.device)
            if getattr(args, "device", None) is not None
            else default_compute_device()
        )
        if amp_device.type != "cuda" or not torch.cuda.is_available():
            raise SystemExit("--field-amp requires CUDA")
        if field_amp == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise SystemExit("this CUDA device does not support bfloat16 AMP")
    if field_method and any(
        not math.isclose(float(s), 1.0, rel_tol=0.0, abs_tol=0.0)
        for s in args.photometric_scales
    ):
        raise SystemExit("--photometric-scales are currently supported for coadd only")
    if field_method and any(p is not None for p in args.primary_beam_maps):
        raise SystemExit("--primary-beam-maps are currently supported for coadd only")
    hdu = int(getattr(args, "hdu", 0))
    reference_hdu = int(getattr(args, "reference_hdu", 0))
    weight_map_hdu = int(getattr(args, "weight_map_hdu", 0))
    mask_map_hdu = int(getattr(args, "mask_map_hdu", 0))
    primary_beam_hdu = getattr(args, "primary_beam_hdu", None)
    if hdu < 0 or reference_hdu < 0:
        raise SystemExit("--hdu and --reference-hdu must be non-negative")
    if weight_map_hdu < 0 or mask_map_hdu < 0:
        raise SystemExit("--weight-map-hdu and --mask-map-hdu must be non-negative")
    if primary_beam_hdu is not None and int(primary_beam_hdu) < 0:
        raise SystemExit("--primary-beam-hdu must be non-negative")
    drizzle_pixfrac = float(getattr(args, "drizzle_pixfrac", 1.0))
    if drizzle_pixfrac <= 0.0 or not math.isfinite(drizzle_pixfrac):
        raise SystemExit("--drizzle-pixfrac must be finite and positive")
    clip_sigma = float(getattr(args, "clip_sigma", 4.0))
    if clip_sigma <= 0.0 or not math.isfinite(clip_sigma):
        raise SystemExit("--clip-sigma must be finite and positive")
    if int(getattr(args, "clip_iterations", 3)) <= 0:
        raise SystemExit("--clip-iterations must be positive")
    if args.stack == "sigma_clipped" and args.background_mode == "overlap":
        raise SystemExit(
            "--stack=sigma_clipped currently requires --background-mode=none "
            "or per-exposure"
        )
    tile_pixels = getattr(args, "tile_pixels", None)
    if tile_pixels is not None and int(tile_pixels) <= 0:
        raise SystemExit("--tile-pixels must be positive")
    tile_above_megapixels = float(getattr(args, "tile_above_megapixels", 32.0))
    if not math.isfinite(tile_above_megapixels) or tile_above_megapixels <= 0.0:
        raise SystemExit("--tile-above-megapixels must be finite and positive")
    footprint_padding = float(getattr(args, "footprint_padding", 1.0))
    if not math.isfinite(footprint_padding) or footprint_padding < 1.0:
        raise SystemExit("--footprint-padding must be finite and at least 1.0")
    footprint_tolerance_px = float(getattr(args, "footprint_tolerance_px", 0.25))
    if not math.isfinite(footprint_tolerance_px) or footprint_tolerance_px <= 0.0:
        raise SystemExit("--footprint-tolerance-px must be finite and positive")
    if int(getattr(args, "footprint_max_depth", 20)) <= 0:
        raise SystemExit("--footprint-max-depth must be positive")
    field_steps = int(getattr(args, "field_steps", 0))
    if field_steps < 0:
        raise SystemExit("--field-steps must be non-negative")
    if (
        field_method
        and hasattr(args, "field_steps")
        and field_steps == 0
        and getattr(args, "resume", None) is None
        and not getattr(args, "dry_run", False)
    ):
        raise SystemExit(
            "--field-steps must be positive; pass --resume to render a saved artifact "
            "without further fitting"
        )
    field_lr = float(getattr(args, "field_lr", 1.0))
    if not math.isfinite(field_lr) or field_lr <= 0.0:
        raise SystemExit("--field-lr must be finite and positive")
    if int(getattr(args, "field_patch_size", 0)) < 0:
        raise SystemExit("--field-patch-size must be non-negative")
    if int(getattr(args, "field_patches_per_obs", 1)) <= 0:
        raise SystemExit("--field-patches-per-obs must be positive")
    if int(getattr(args, "field_obs_per_step", 1)) <= 0:
        raise SystemExit("--field-obs-per-step must be positive")
    if int(getattr(args, "field_spectral_samples", 1)) <= 0:
        raise SystemExit("--field-spectral-samples must be positive")
    if int(getattr(args, "field_render_tile", 1)) <= 0:
        raise SystemExit("--field-render-tile must be positive")
    if int(getattr(args, "field_max_pixels", 1)) <= 0:
        raise SystemExit("--field-max-pixels must be positive")
    args.inputs = expanded
    if args.method == "coadd":
        if getattr(args, "per_filter", False) and (
            getattr(args, "_resolved_plan_inputs", None) is None
        ):
            return _run_coadd_per_filter(
                args, expanded, read_device=_map_read_device_str(args)
            )
        return _core_mod._run_coadd(args)
    return _run_field(args)


# Late-bound so monkeypatching ``map_command`` attributes still reaches this code.
from . import map_command as _core_mod  # noqa: E402
