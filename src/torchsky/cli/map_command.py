"""``torchsky map`` — high-SNR image-plane coadd or continuous field render."""

from __future__ import annotations

import argparse
import glob
from importlib import metadata as importlib_metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, cast

import torch

from ..device import default_compute_device
from torchfits import read
from torchfits import read_header
from torchfits.hdu import HDUList, TensorHDU
from ..map.field.artifacts import load_field_artifact, save_field_artifact
from ..map.field.core import estimate_field_domain, make_field_model
from ..map.field.presets import get_field_model_preset
from ..map.field.training import fit_field_model
from ..map.field.types import FieldObservation, FieldSpatialSampling
from ..map.observation import (
    ImageObservation,
)
from ..map.io_utils import write_map_product
from ..map.product import (
    BandpassSpec,
    plan_map_product,
    render_field_product,
    stack_product,
)
from ..map.plan import MapPlanV2, build_map_input_identity, load_map_plan
from ..map.operators.execution import (
    exact_median_scratch_bytes,
    overlap_background_scratch_bytes,
    sigma_clipped_scratch_bytes,
)
from ..map.sky import CrossExposureSkyCorrector, MeshSkyEstimator, PolynomialSkyModel
from ..spectrophotometry import Passband

from ..filter_profiles import field_observation_spectral_kwargs, resolve_field_bandpass
from ..filter_profiles.field_bandpass import FieldSpectralSource
from ..instrument import NIRCAM_UPDEC_BANDS, nircam_imaging_band

# Offline fallback when FITS headers lack spectral metadata (Angstrom).
_FIELD_BAND_TABLE: dict[str, tuple[float, float, float, float]] = {}
for _band_name in NIRCAM_UPDEC_BANDS:
    _band_metadata = nircam_imaging_band(_band_name)
    _fwhm = (
        _band_metadata.half_power_max_angstrom - _band_metadata.half_power_min_angstrom
    )
    _FIELD_BAND_TABLE[_band_name] = (
        _band_metadata.half_power_min_angstrom,
        _band_metadata.half_power_max_angstrom,
        _band_metadata.pivot_wavelength_angstrom,
        _fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0))),
    )


_WORKFLOW_DEFAULTS: dict[str, tuple[str, str, str]] = {
    "native": ("bilinear", "weighted_mean", "none"),
    "drizzle": ("drizzle_square", "weighted_mean", "none"),
    "swarp": ("lanczos3", "weighted_mean", "per-exposure"),
    "montage": ("conservative_polygon", "weighted_mean", "overlap"),
}


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def _source_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[3],
        check=False,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def _software_manifest() -> dict[str, str | None]:
    return {
        "torchsky": _installed_version("torchsky"),
        "torchfits": _installed_version("torchfits"),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "commit": _source_commit(),
    }


def _runtime_manifest(args: argparse.Namespace, resolved_device: str) -> dict:
    return {
        "dtype": "float32",
        "amp_dtype": str(getattr(args, "field_amp", "none")),
        "requested_device": getattr(args, "device", None),
        "device": resolved_device,
        "seeds": {"torch": int(torch.initial_seed())},
        "scratch": _resolved_path(getattr(args, "scratch_dir", None)),
    }


def _optional_input_identity(
    path: str | None,
    hdu: int | str | None,
) -> dict | None:
    if path is None:
        return None
    return build_map_input_identity(path, hdu=hdu)


def _resolve_map_workflow(args: argparse.Namespace) -> None:
    """Expand one CLI workflow preset into the existing explicit controls."""
    workflow = str(getattr(args, "workflow", "native"))
    try:
        default_kernel, default_stack, default_background = _WORKFLOW_DEFAULTS[workflow]
    except KeyError as exc:
        raise SystemExit(f"unknown map workflow {workflow!r}") from exc
    if getattr(args, "kernel", None) is None:
        args.kernel = default_kernel
    if getattr(args, "stack", None) is None:
        args.stack = default_stack
    explicit_background = getattr(args, "background_mode", None)
    args.background_mode = explicit_background or default_background
    args.workflow = workflow


def _sky_estimator(args: argparse.Namespace):
    if str(getattr(args, "sky_method", "mesh")) == "polynomial":
        return PolynomialSkyModel(
            order=int(getattr(args, "sky_order", 3)),
            statistic=str(getattr(args, "sky_statistic", "huber")),
        )
    mesh_filter = getattr(args, "sky_mesh_filter", "median")
    return MeshSkyEstimator(
        block_size=int(getattr(args, "sky_block_size", 128)),
        statistic=str(getattr(args, "sky_statistic", "huber")),
        mesh_filter=None if mesh_filter == "none" else str(mesh_filter),
        mesh_filter_size=int(getattr(args, "sky_mesh_filter_size", 3)),
        interpolate=str(getattr(args, "sky_interpolate", "bicubic")),
    )


def _background_strategy(args: argparse.Namespace):
    mode = str(getattr(args, "background_mode", "none"))
    if mode == "none":
        return None, None
    estimator = _sky_estimator(args)
    if mode == "per-exposure":
        return estimator, None
    if mode == "overlap":
        return None, CrossExposureSkyCorrector(estimator=estimator)
    raise SystemExit(f"unknown background mode {mode!r}")


def _write_map_plan(path: Path, plan: dict) -> None:
    """Atomically write one deterministic map execution plan."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            json.dump(plan, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
        raise


def _resolved_path(path: str | Path | None) -> str | None:
    return None if path is None else str(Path(path).expanduser().resolve())


def _field_artifact_identities(path: str | Path | None) -> list[dict] | None:
    if path is None:
        return None
    root = Path(path).expanduser().resolve()
    manifest = root / "manifest.json"
    weights = root / "weights.pt"
    if not manifest.exists() and not weights.exists():
        return None
    if not manifest.is_file() or not weights.is_file():
        raise ValueError(f"incomplete field artifact: {root}")
    files = [manifest, weights]
    training_state = root / "training_state.pt"
    if training_state.is_file():
        files.append(training_state)
    return [build_map_input_identity(file, hdu=None) for file in files]


def _expand_inputs(patterns: Sequence[str]) -> list[str]:
    out: list[str] = []
    for p in patterns:
        if any(ch in p for ch in "*?[]"):
            out.extend(sorted(glob.glob(p)))
        else:
            out.append(p)
    seen: set[str] = set()
    uniq: list[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _header_filter_keyword(header: dict) -> str | None:
    for key in ("FILTER", "FILTNAM", "FILTER1"):
        if key in header and header[key] not in (None, "", " "):
            v = header[key]
            if isinstance(v, bytes):
                return v.decode("ascii", errors="replace").strip()
            return str(v).strip()
    return None


def _field_spectral_kwargs(args: argparse.Namespace, band: str) -> dict:
    """Spectral keyword args for :class:`FieldObservation`."""
    try:
        raw = resolve_field_bandpass(
            source=cast(FieldSpectralSource, str(args.field_spectral_source)),
            band_name=str(band).strip().upper(),
            builtin_table=_FIELD_BAND_TABLE,
            svo_facility=str(args.svo_facility),
            svo_instrument=str(args.svo_instrument),
            svo_filter_id=args.svo_filter_id,
            svo_prefer=cast(Literal["classic", "2025", "any"], str(args.svo_prefer)),
            svo_cache_dir=getattr(args, "svo_cache_dir", None),
            svo_use_cache=not getattr(args, "no_svo_cache", False),
            transmission_path=args.field_transmission,
            transmission_dir=args.field_transmission_dir,
            transmission_unit=cast(
                Literal["angstrom", "nm"], str(args.field_transmission_unit)
            ),
            gaussian_lambda_eff=args.field_gaussian_lambda_eff,
            gaussian_lambda_min=args.field_gaussian_lambda_min,
            gaussian_lambda_max=args.field_gaussian_lambda_max,
            gaussian_sigma=args.field_gaussian_sigma,
        )
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc
    except ValueError as exc:
        raise SystemExit(f"torchsky map field: {exc}") from exc
    fid = raw.get("svo_filter_id")
    if (
        str(args.field_spectral_source) == "svo"
        and fid
        and not getattr(args, "field_quiet_svo", False)
    ):
        print(
            f"torchsky map field: SVO bandpass {str(band).upper()} -> {fid}",
            file=sys.stderr,
        )
    return field_observation_spectral_kwargs(raw)


def _map_read_device_str(args: argparse.Namespace) -> str:
    dev = getattr(args, "device", None)
    if dev is not None:
        return str(torch.device(dev))
    return str(default_compute_device())


def _probe_image_hdus(path: str) -> tuple[int, ...]:
    """Return tuple of HDU indices containing image data (TensorHDU with NAXIS>0).

    Returns empty tuple for unreadable files or files with no image HDUs.
    """
    try:
        with HDUList.fromfile(path) as hdul:
            indices: list[int] = []
            for i in range(len(hdul)):
                h = hdul[i]
                if not isinstance(h, TensorHDU):
                    continue
                naxis_raw = h.header.get("NAXIS", "0")
                try:
                    naxis = int(str(naxis_raw).strip())
                except (ValueError, AttributeError):
                    naxis = 0
                if naxis > 0:
                    indices.append(i)
            return tuple(indices)
    except Exception:
        return ()


def _paths_to_field_observations(
    paths: Sequence[str],
    *,
    hdu: int | Sequence[int] = 0,
    spectral_kw: dict | Sequence[dict],
    read_device: str,
    weights: Sequence[float] | None = None,
    background_offsets: Sequence[float] | None = None,
    weight_maps: Sequence[str | None] | None = None,
    mask_maps: Sequence[str | None] | None = None,
    weight_hdu: int | str | None = 0,
    mask_hdu: int | str | None = 0,
) -> list[FieldObservation]:
    out: list[FieldObservation] = []
    scalar_weights = list(weights) if weights is not None else [1.0] * len(paths)
    backgrounds = (
        list(background_offsets)
        if background_offsets is not None
        else [0.0] * len(paths)
    )
    wpaths = list(weight_maps) if weight_maps is not None else [None] * len(paths)
    mpaths = list(mask_maps) if mask_maps is not None else [None] * len(paths)
    if len(scalar_weights) != len(paths):
        raise ValueError("weights count must match input path count")
    if len(backgrounds) != len(paths):
        raise ValueError("background_offsets count must match input path count")
    if len(wpaths) != len(paths):
        raise ValueError("weight_maps count must match input path count")
    if len(mpaths) != len(paths):
        raise ValueError("mask_maps count must match input path count")
    spectral_kws = (
        [spectral_kw] * len(paths)
        if isinstance(spectral_kw, dict)
        else list(spectral_kw)
    )
    if len(spectral_kws) != len(paths):
        raise ValueError("spectral_kw count must match input path count")
    hdus: Sequence[int]
    if isinstance(hdu, int):
        hdus = [hdu] * len(paths)
    else:
        hdus = list(hdu)
        if len(hdus) != len(paths):
            raise ValueError(
                "hdus (len={}) count must match paths count ({})".format(
                    len(hdus), len(paths)
                )
            )
    for p, hdu_i, scalar_weight, background_offset, weight_path, mask_path, skw in zip(
        paths, hdus, scalar_weights, backgrounds, wpaths, mpaths, spectral_kws
    ):
        img, hdr = read(p, hdu=hdu_i, return_header=True, device=read_device)
        if img.ndim != 2:
            raise ValueError(f"{p}: expected 2D image")
        weight = torch.ones_like(img, dtype=torch.float32) * float(scalar_weight)
        if weight_path is not None:
            wmap = read(str(weight_path), hdu=weight_hdu, device=read_device)
            if tuple(wmap.shape) != tuple(img.shape):
                raise ValueError(
                    f"{weight_path}: weight map shape {tuple(wmap.shape)} "
                    f"does not match image shape {tuple(img.shape)}"
                )
            weight = weight * wmap.to(device=img.device, dtype=torch.float32)
        mask = None
        if mask_path is not None:
            mmap = read(str(mask_path), hdu=mask_hdu, device=read_device)
            if tuple(mmap.shape) != tuple(img.shape):
                raise ValueError(
                    f"{mask_path}: mask map shape {tuple(mmap.shape)} "
                    f"does not match image shape {tuple(img.shape)}"
                )
            mask = mmap.to(device=img.device, dtype=torch.float32)
        out.append(
            FieldObservation(
                image=img,
                header=dict(hdr),
                weight=weight,
                mask=mask,
                background_offset=float(background_offset),
                **skw,
            )
        )
    return out


def _resolve_field_observation_spectra(
    args: argparse.Namespace,
    paths: Sequence[str],
    hdus: Sequence[int],
    *,
    read_device: str,
) -> tuple[list[str], list[dict]]:
    """Resolve one passband per exposure, caching repeated filter profiles."""
    explicit_band = getattr(args, "field_band", None)
    if explicit_band is not None:
        bands = [str(explicit_band).strip().upper()] * len(paths)
    else:
        bands = []
        for path, hdu in zip(paths, hdus):
            _, header = read(path, hdu=hdu, return_header=True, device=read_device)
            band = _header_filter_keyword(dict(header))
            if band is None:
                raise SystemExit(
                    f"{path}[{hdu}]: field fitting requires FILTER/FILTNAM "
                    "or an explicit --field-band"
                )
            bands.append(str(band).strip().upper())

    unique_bands = tuple(dict.fromkeys(bands))
    source = str(getattr(args, "field_spectral_source", "svo"))
    if len(unique_bands) > 1:
        if source == "gaussian":
            raise SystemExit(
                "joint multiband fitting cannot use one Gaussian profile for "
                "multiple header filters; use builtin, SVO, or a transmission directory"
            )
        if source == "file" and getattr(args, "field_transmission", None) is not None:
            raise SystemExit(
                "joint multiband fitting requires --field-transmission-dir, not one "
                "--field-transmission file"
            )
        if source == "svo" and getattr(args, "svo_filter_id", None) is not None:
            raise SystemExit(
                "joint multiband fitting cannot apply one --svo-filter-id to "
                "multiple header filters"
            )

    profiles = {band: _field_spectral_kwargs(args, band) for band in unique_bands}
    return bands, [profiles[band] for band in bands]


def _map_bandpass_spec(label: str, spectral_kw: dict) -> BandpassSpec:
    wavelength = spectral_kw.get("transmission_wavelength")
    throughput = spectral_kw.get("transmission_throughput")
    if wavelength is not None or throughput is not None:
        if wavelength is None or throughput is None:
            raise ValueError("tabulated passband requires wavelength and throughput")
        return BandpassSpec(
            label=label,
            spectral_min=spectral_kw.get("spectral_min"),
            spectral_max=spectral_kw.get("spectral_max"),
            passband=Passband.from_samples(
                wavelength,
                throughput,
                name=label,
            ),
        )
    return BandpassSpec(
        label=label,
        transmission_center=spectral_kw.get("transmission_center"),
        transmission_sigma=spectral_kw.get("transmission_sigma"),
        spectral_min=spectral_kw.get("spectral_min"),
        spectral_max=spectral_kw.get("spectral_max"),
    )


def _spectral_plan_entry(label: str, spectral_kw: dict) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "label": label,
        "wavelength": spectral_kw.get("wavelength"),
        "spectral_min": spectral_kw.get("spectral_min"),
        "spectral_max": spectral_kw.get("spectral_max"),
        "transmission_center": spectral_kw.get("transmission_center"),
        "transmission_sigma": spectral_kw.get("transmission_sigma"),
    }
    wavelength = spectral_kw.get("transmission_wavelength")
    throughput = spectral_kw.get("transmission_throughput")
    if wavelength is not None and throughput is not None:
        entry["transmission_wavelength"] = torch.as_tensor(wavelength).tolist()
        entry["transmission_throughput"] = torch.as_tensor(throughput).tolist()
    return entry


def _field_spectra_from_plan(
    spectral: Any,
    *,
    input_count: int,
) -> tuple[list[str], list[dict]]:
    if not isinstance(spectral, dict):
        raise ValueError("field plan requires spectral metadata")
    entries = spectral.get("bands")
    observation_bands = spectral.get("observation_bands")
    if not isinstance(entries, list) or not entries:
        raise ValueError("field plan requires resolved band profiles")
    if not isinstance(observation_bands, list) or len(observation_bands) != input_count:
        raise ValueError("field plan observation_bands must match inputs")
    profiles: dict[str, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("label"), str):
            raise ValueError("field plan band profiles require labels")
        label = str(entry["label"])
        profile = {
            key: value
            for key, value in entry.items()
            if key
            not in {
                "label",
                "transmission_wavelength",
                "transmission_throughput",
            }
            and value is not None
        }
        profile["band_name"] = label
        if profile.get("wavelength") is None:
            profile["wavelength"] = profile.get("transmission_center")
        for key in ("transmission_wavelength", "transmission_throughput"):
            value = entry.get(key)
            if value is not None:
                if not isinstance(value, list):
                    raise ValueError(f"field plan {key} must be a list")
                profile[key] = torch.as_tensor(value, dtype=torch.float32)
        profiles[label] = profile
    bands = [str(value) for value in observation_bands]
    try:
        spectral_kws = [profiles[band] for band in bands]
    except KeyError as exc:
        raise ValueError(
            f"field plan observation references unknown band {exc.args[0]!r}"
        ) from exc
    return bands, spectral_kws


def _compute_spatial_tile_hw(
    shape: tuple[int, int],
    *,
    auto_tile: bool,
    tile_pixels: int | None,
    tile_above_megapixels: float,
) -> tuple[int, int] | None:
    h, w = int(shape[0]), int(shape[1])
    pixels = h * w
    threshold = max(1, int(float(tile_above_megapixels) * 1_000_000))
    if tile_pixels is not None:
        tp = max(32, int(tile_pixels))
        return (tp, tp)
    if not auto_tile:
        return None
    if pixels <= threshold:
        return None
    side = int(math.sqrt(threshold))
    side = max(512, min(4096, side))
    return (side, side)


def _run_coadd(args: argparse.Namespace) -> int:
    _resolve_map_workflow(args)
    clip_sigma = float(getattr(args, "clip_sigma", 4.0))
    clip_iterations = int(getattr(args, "clip_iterations", 3))
    inputs = args.inputs
    if not inputs:
        print("torchsky map: no input files matched", file=sys.stderr)
        return 2
    read_dev = _map_read_device_str(args)

    # Expand MEFs: probe each file for image HDUs and create per-HDU entries.
    resolved_inputs = getattr(args, "_resolved_plan_inputs", None)
    if resolved_inputs is not None:
        expanded = [
            (
                str(item["identity"]["path"]),
                int(item["identity"].get("hdu", 0)),
                float(item.get("weight", 1.0)),
                float(item.get("photometric_scale", 1.0)),
                float(item.get("background_offset", 0.0)),
                item.get("weight_map"),
                bool(
                    item.get(
                        "weight_is_inverse_variance", item.get("weight_map") is not None
                    )
                ),
                item.get("mask_map"),
                item.get("primary_beam_map"),
                item.get("dq_map"),
            )
            for item in resolved_inputs
        ]
    else:
        explicit_scalar_weight = bool(
            getattr(args, "_weights_are_inverse_variance", False)
        )
        dq_maps = getattr(args, "dq_maps", None) or [None] * len(inputs)
        expanded = [
            (
                path,
                hdu,
                float(args.weights[i]),
                float(args.photometric_scales[i]),
                float(args.background_offsets[i]),
                args.weight_maps[i],
                explicit_scalar_weight or args.weight_maps[i] is not None,
                args.mask_maps[i],
                args.primary_beam_maps[i],
                dq_maps[i],
            )
            for i, path, hdu in _image_hdu_expansion(
                inputs, int(args.hdu), announce=True
            )
        ]

    observations = [
        ImageObservation(
            path=path,
            hdu=hdu,
            weight=weight,
            weight_is_inverse_variance=weight_is_inverse_variance,
            photometric_scale=scale,
            background_offset=background,
            weight_path=weight_map,
            weight_hdu=args.weight_map_hdu,
            mask_path=mask_map,
            mask_hdu=args.mask_map_hdu,
            primary_beam_path=primary_beam_map,
            primary_beam_hdu=args.primary_beam_hdu,
            dq_path=dq_map,
            dq_hdu=_parse_hdu_spec(getattr(args, "dq_hdu", "DQ")),
            good_bits=getattr(args, "good_bits", None),
            band_name=getattr(args, "_coadd_band", None),
        )
        for (
            path,
            hdu,
            weight,
            scale,
            background,
            weight_map,
            weight_is_inverse_variance,
            mask_map,
            primary_beam_map,
            dq_map,
        ) in expanded
    ]

    ref_header = None
    replay_header = getattr(args, "_plan_target_header", None)
    if replay_header is not None:
        ref_header = dict(replay_header)
    elif args.reference is not None:
        _, rh = read(
            str(args.reference),
            hdu=args.reference_hdu,
            return_header=True,
            device=read_dev,
        )
        ref_header = dict(rh)

    union_footprint = (
        bool(getattr(args, "union_footprint", False)) and replay_header is None
    )
    reference_index = int(args.plan_reference)
    if not union_footprint and not 0 <= reference_index < len(observations):
        raise SystemExit(
            f"--plan-reference {reference_index} out of range for "
            f"{len(observations)} observations"
        )
    if resolved_inputs is None and len(inputs) > 1:
        band = _reject_mixed_filters(args, inputs, read_device=read_dev)
        args._coadd_band = None if band in (None, "UNKNOWN") else band.upper()

    product = plan_map_product(
        observations,
        reference_index=reference_index,
        header_overrides=ref_header,
        union_footprint=union_footprint,
        footprint_padding=float(getattr(args, "footprint_padding", 1.0)),
        footprint_tolerance_px=float(getattr(args, "footprint_tolerance_px", 0.25)),
        footprint_max_depth=int(getattr(args, "footprint_max_depth", 20)),
        max_footprint_pixels=getattr(args, "max_footprint_pixels", None),
        value_semantics=args.value_semantics,
        device=args.device,
    )

    replay_tile_hw = getattr(args, "_plan_tile_hw", None)
    tile_hw = (
        None
        if replay_tile_hw is None and hasattr(args, "_plan_tile_hw")
        else (
            (int(replay_tile_hw[0]), int(replay_tile_hw[1]))
            if replay_tile_hw is not None
            else _compute_spatial_tile_hw(
                product.shape,
                auto_tile=bool(args.auto_tile),
                tile_pixels=args.tile_pixels,
                tile_above_megapixels=float(args.tile_above_megapixels),
            )
        )
    )
    if tile_hw is not None:
        print(
            f"torchsky map: spatial tiling {tile_hw[0]}×{tile_hw[1]} "
            f"for output {product.shape} pixels",
            file=sys.stderr,
        )
    scratch_required_bytes = 0
    if args.background_mode == "overlap" and len(observations) > 1:
        scratch_required_bytes = overlap_background_scratch_bytes(
            len(observations),
            product.shape,
            stacking_method=str(args.stack),
        )
    elif args.stack in ("median", "sigma_clipped"):
        reducer_shape = (
            product.shape
            if tile_hw is None
            else (
                min(int(tile_hw[0]), int(product.shape[0])),
                min(int(tile_hw[1]), int(product.shape[1])),
            )
        )
        reducer_scratch = (
            exact_median_scratch_bytes
            if args.stack == "median"
            else sigma_clipped_scratch_bytes
        )
        scratch_required_bytes = reducer_scratch(len(observations), reducer_shape)

    plan_path = getattr(args, "plan_json", None)
    if plan_path is not None:
        plan_inputs = []
        for (
            path,
            hdu,
            weight,
            scale,
            background,
            weight_map,
            weight_is_inverse_variance,
            mask_map,
            primary_beam_map,
            dq_map,
        ) in expanded:
            plan_inputs.append(
                {
                    "path": _resolved_path(path),
                    "hdu": int(hdu),
                    "identity": build_map_input_identity(path, hdu=hdu),
                    "weight": float(weight),
                    "weight_is_inverse_variance": bool(weight_is_inverse_variance),
                    "photometric_scale": float(scale),
                    "background_offset": float(background),
                    "weight_map": _resolved_path(weight_map),
                    "weight_identity": _optional_input_identity(
                        weight_map, args.weight_map_hdu
                    ),
                    "mask_map": _resolved_path(mask_map),
                    "mask_identity": _optional_input_identity(
                        mask_map, args.mask_map_hdu
                    ),
                    "primary_beam_map": _resolved_path(primary_beam_map),
                    "primary_beam_identity": _optional_input_identity(
                        primary_beam_map, args.primary_beam_hdu
                    ),
                    "dq_map": _resolved_path(dq_map),
                    "dq_identity": _optional_input_identity(
                        dq_map, _parse_hdu_spec(getattr(args, "dq_hdu", "DQ"))
                    ),
                    "band": getattr(args, "_coadd_band", None),
                }
            )
        payload = {
            "schema_version": 2,
            "method": "coadd",
            "output": _resolved_path(args.output),
            "inputs": plan_inputs,
            "target": {
                "shape": list(product.shape),
                "header": dict(product.target_header),
                "reference": _resolved_path(args.reference),
                "reference_hdu": int(args.reference_hdu),
                "union_footprint": union_footprint,
                "footprint_padding": float(args.footprint_padding),
                "footprint_tolerance_px": float(
                    getattr(args, "footprint_tolerance_px", 0.25)
                ),
                "footprint_max_depth": int(getattr(args, "footprint_max_depth", 20)),
            },
            "execution": {
                "workflow": str(args.workflow),
                "kernel": str(args.kernel),
                "stack": str(args.stack),
                "background_mode": str(args.background_mode),
                "drizzle_pixfrac": float(args.drizzle_pixfrac),
                "clip_sigma": clip_sigma,
                "clip_iterations": clip_iterations,
                "value_semantics": str(args.value_semantics),
                "tile_hw": None if tile_hw is None else list(tile_hw),
                "requested_device": args.device,
                "resolved_device": read_dev,
                "scratch_required_bytes": scratch_required_bytes,
            },
            "requested_products": list(getattr(args, "output_planes", ("SCI",))),
            "software": _software_manifest(),
            "runtime": _runtime_manifest(args, read_dev),
        }
        _write_map_plan(
            Path(plan_path),
            MapPlanV2(payload).to_dict(),
        )

    if getattr(args, "dry_run", False):
        print(
            f"torchsky map: dry-run output_shape={product.shape} "
            f"method=coadd kernel={args.kernel} stack={args.stack} "
            f"workflow={args.workflow} background={args.background_mode} "
            f"value_semantics={args.value_semantics} tiled={tile_hw is not None} "
            f"n_bands={product.n_bands} scratch_bytes={scratch_required_bytes}",
            file=sys.stderr,
        )
        return 0

    background_estimator, overlap_corrector = _background_strategy(args)
    result = stack_product(
        observations,
        product,
        resampling_kernel=args.kernel,
        stacking_method=args.stack,
        drizzle_pixfrac=float(args.drizzle_pixfrac),
        clip_sigma=clip_sigma,
        clip_iterations=clip_iterations,
        spatial_tile_hw=tile_hw,
        device=args.device,
        scratch_dir=getattr(args, "scratch_dir", None),
        background_estimator=background_estimator,
        cross_exposure_sky_corrector=overlap_corrector,
    )
    if not bool(getattr(args, "overwrite", True)) and Path(args.output).exists():
        raise SystemExit(
            f"torchsky map: output exists and --no-overwrite was given: {args.output}"
        )
    written_planes = write_map_product(
        args.output,
        result,
        requested_planes=getattr(args, "output_planes", ("SCI",)),
        overwrite=bool(getattr(args, "overwrite", True)),
    )
    assert result.weight is not None  # coadd products always carry weights
    print(
        f"wrote {args.output} shape={tuple(result.data.shape)} "
        f"mean_weight={float(result.weight.mean().item()):.4f} "
        f"method=coadd kernel={args.kernel} stack={args.stack} "
        f"workflow={args.workflow} background={args.background_mode} "
        f"value_semantics={args.value_semantics} tiled={tile_hw is not None} "
        f"n_bands={result.n_bands} planes={','.join(written_planes)}",
        file=sys.stderr,
    )
    return 0


def _run_field(args: argparse.Namespace) -> int:
    inputs = _expand_inputs(args.inputs)
    if not inputs:
        print("torchsky map: no input files matched", file=sys.stderr)
        return 2
    read_dev = _map_read_device_str(args)
    if args.per_filter:
        groups = _group_inputs_by_filter(args, inputs, read_device=read_dev)
        rc = 0
        for fk, indices in sorted(groups.items()):
            gpaths = [inputs[i] for i in indices]
            if fk == "UNKNOWN" and args.field_band is None:
                print(
                    f"torchsky map field: skip UNKNOWN filter group ({len(gpaths)} files); "
                    "set --field-band",
                    file=sys.stderr,
                )
                continue
            band = str(args.field_band).upper() if fk == "UNKNOWN" else str(fk).upper()
            sub_out = _filter_output_path(args.output, fk)
            sub_out.parent.mkdir(parents=True, exist_ok=True)
            ns = argparse.Namespace(**vars(args))
            ns.output = str(sub_out)
            ns.inputs = gpaths
            ns.weights = [args.weights[i] for i in indices]
            ns.background_offsets = [args.background_offsets[i] for i in indices]
            ns.weight_maps = [args.weight_maps[i] for i in indices]
            ns.mask_maps = [args.mask_maps[i] for i in indices]
            ns.primary_beam_maps = [args.primary_beam_maps[i] for i in indices]
            ns.per_filter = False
            ns.field_band = band
            if getattr(args, "plan_json", None) is not None:
                ns.plan_json = _per_filter_plan_path(Path(args.plan_json), fk)
            # Every band group runs even after an earlier failure; the first
            # non-zero status is reported.
            rc = _run_field_single(ns) or rc
        return rc
    return _run_field_single(args)


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s) or "band"


def _parse_hdu_spec(value: Any) -> Any:
    """Accept an integer HDU index spelled as a decimal string."""
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
    return value


def _per_filter_plan_path(path: Path, band: str) -> Path:
    suffix = path.suffix or ".json"
    return path.with_name(f"{path.stem}_{_safe_name(band)}{suffix}")


def _filter_output_path(output: str, band: str) -> Path:
    """Resolve the per-filter output path for one band."""
    out_path = Path(output)
    if out_path.suffix == "" or out_path.is_dir():
        return out_path / f"{_safe_name(band)}.fits"
    return out_path.parent / f"{out_path.stem}_{_safe_name(band)}{out_path.suffix}"


def _group_inputs_by_filter(
    args: argparse.Namespace,
    inputs: Sequence[str],
    *,
    read_device: str,
) -> dict[str, list[int]]:
    """Group expanded input rows by FITS filter keyword.

    Returns a mapping from filter name to the indices of the input files that
    contribute to that band. A multi-extension file whose HDUs disagree on the
    filter cannot be grouped and fails closed.
    """
    hdu = int(getattr(args, "hdu", 0))
    groups: dict[str, list[int]] = defaultdict(list)
    bands_by_input: dict[int, set[str]] = defaultdict(set)
    bunits: dict[int, str] = {}
    for input_index, path, image_hdu in _image_hdu_expansion(inputs, hdu):
        header = dict(read_header(str(path), image_hdu))
        band = _header_filter_keyword(header) or "UNKNOWN"
        groups[band].append(input_index)
        bands_by_input[input_index].add(band)
        raw_unit = header.get("BUNIT")
        if isinstance(raw_unit, str) and raw_unit.strip():
            bunits[input_index] = raw_unit.strip()
    mixed = sorted(
        str(inputs[index]) for index, bands in bands_by_input.items() if len(bands) > 1
    )
    if mixed:
        raise SystemExit(
            "--per-filter cannot group a multi-extension file whose HDUs carry "
            f"different filters: {mixed}; split it first"
        )
    distinct_units = sorted(set(bunits.values()))
    if len(distinct_units) > 1:
        detail = ", ".join(
            f"{Path(inputs[index]).name}: {unit}" for index, unit in sorted(bunits.items())
        )
        raise SystemExit(
            "torchsky map: inputs declare different BUNIT units "
            f"({detail}); convert them to one unit before combining"
        )
    return {band: sorted(set(indices)) for band, indices in groups.items()}


def _reject_mixed_filters(
    args: argparse.Namespace, inputs: Sequence[str], *, read_device: str
) -> str | None:
    """Refuse to stack different filters into one science plane."""
    groups = _group_inputs_by_filter(args, inputs, read_device=read_device)
    if len(groups) > 1:
        bands = ", ".join(sorted(groups))
        raise SystemExit(
            f"torchsky map: inputs span multiple filters ({bands}); pass "
            "--per-filter to write one product per filter"
        )
    return next(iter(groups), None)


def _run_coadd_per_filter(
    args: argparse.Namespace, inputs: Sequence[str], *, read_device: str
) -> int:
    """Coadd each FITS filter into its own product."""
    groups = _group_inputs_by_filter(args, inputs, read_device=read_device)
    if not groups:
        print("torchsky map: no input files matched", file=sys.stderr)
        return 2
    rc = 0
    for band, indices in sorted(groups.items()):
        sub = argparse.Namespace(**vars(args))
        sub.inputs = [inputs[i] for i in indices]
        sub.weights = [args.weights[i] for i in indices]
        sub.photometric_scales = [args.photometric_scales[i] for i in indices]
        sub.background_offsets = [args.background_offsets[i] for i in indices]
        sub.weight_maps = [args.weight_maps[i] for i in indices]
        sub.mask_maps = [args.mask_maps[i] for i in indices]
        sub.primary_beam_maps = [args.primary_beam_maps[i] for i in indices]
        sub.dq_maps = [args.dq_maps[i] for i in indices]
        sub.per_filter = False
        sub._resolved_plan_inputs = None
        sub._coadd_band = None if band == "UNKNOWN" else band.upper()
        sub.output = str(_filter_output_path(args.output, band))
        Path(sub.output).parent.mkdir(parents=True, exist_ok=True)
        if getattr(args, "plan_json", None) is not None:
            sub.plan_json = _per_filter_plan_path(Path(args.plan_json), band)
        print(
            f"torchsky map: filter {band} -> {sub.output} "
            f"({len(indices)} input file(s))",
            file=sys.stderr,
        )
        rc = _run_coadd(sub) or rc
    return rc


def _image_hdu_expansion(
    paths: Sequence[str], hdu: int, *, announce: bool = False
) -> list[tuple[int, str, int]]:
    """Expand multi-extension files into ``(input_index, path, hdu)`` rows.

    One policy for every map path: an unreadable or image-less file is passed
    through so the reader reports the real error, a single-image file uses that
    image, an explicit ``--hdu`` that exists wins, and an MEF otherwise
    contributes every image HDU.
    """
    rows: list[tuple[int, str, int]] = []
    for i, path in enumerate(paths):
        image_hdus = _probe_image_hdus(path)
        if len(image_hdus) == 0:
            rows.append((i, path, hdu))
        elif len(image_hdus) == 1:
            rows.append((i, path, hdu if hdu in image_hdus else image_hdus[0]))
        elif hdu != 0 and hdu in image_hdus:
            rows.append((i, path, hdu))
        else:
            rows.extend((i, path, h) for h in image_hdus)
            if announce:
                print(
                    f"  {path}: auto-expanded {len(image_hdus)} image HDUs "
                    f"{list(image_hdus)}",
                    file=sys.stderr,
                )
    return rows


def _expand_field_inputs(
    paths: list[str],
    hdu: int,
    weights: list[float],
    background_offsets: list[float],
    weight_maps: list[str | None],
    mask_maps: list[str | None],
) -> tuple[
    list[str], list[int], list[float], list[float], list[str | None], list[str | None]
]:
    """Expand MEF files into per-HDU entries for field processing."""
    rows = _image_hdu_expansion(paths, hdu)
    return (
        [path for _, path, _ in rows],
        [hdu_i for _, _, hdu_i in rows],
        [float(weights[i]) for i, _, _ in rows],
        [float(background_offsets[i]) for i, _, _ in rows],
        [weight_maps[i] for i, _, _ in rows],
        [mask_maps[i] for i, _, _ in rows],
    )


def _run_field_single(args: argparse.Namespace) -> int:
    read_dev = _map_read_device_str(args)
    resolved_plan_inputs = getattr(args, "_resolved_plan_inputs", None)
    if resolved_plan_inputs is None:
        inputs = _expand_inputs(args.inputs)
        (
            field_paths,
            field_hdus,
            field_weights,
            field_backgrounds,
            field_weight_maps,
            field_mask_maps,
        ) = _expand_field_inputs(
            inputs,
            hdu=args.hdu,
            weights=args.weights,
            background_offsets=args.background_offsets,
            weight_maps=args.weight_maps,
            mask_maps=args.mask_maps,
        )
    else:
        field_paths = [str(item["identity"]["path"]) for item in resolved_plan_inputs]
        field_hdus = [int(item["hdu"]) for item in resolved_plan_inputs]
        field_weights = [float(item.get("weight", 1.0)) for item in resolved_plan_inputs]
        field_backgrounds = [
            float(item.get("background_offset", 0.0)) for item in resolved_plan_inputs
        ]
        field_weight_maps = [item.get("weight_map") for item in resolved_plan_inputs]
        field_mask_maps = [item.get("mask_map") for item in resolved_plan_inputs]

    planned_bands = getattr(args, "_plan_observation_bands", None)
    planned_spectral_kws = getattr(args, "_plan_spectral_kws", None)
    if planned_bands is None or planned_spectral_kws is None:
        observation_bands, spectral_kws = _resolve_field_observation_spectra(
            args,
            field_paths,
            field_hdus,
            read_device=read_dev,
        )
    else:
        observation_bands = list(planned_bands)
        spectral_kws = list(planned_spectral_kws)
        if len(observation_bands) != len(field_paths) or len(spectral_kws) != len(
            field_paths
        ):
            raise SystemExit("field plan spectral metadata does not match inputs")
    profiles_by_band: dict[str, dict] = {}
    for band, spectral_kw in zip(observation_bands, spectral_kws, strict=True):
        profiles_by_band.setdefault(band, spectral_kw)
    unique_bands = list(profiles_by_band)
    band_specs = [
        _map_bandpass_spec(band, profiles_by_band[band]) for band in unique_bands
    ]
    band_summary = ",".join(unique_bands)
    planned_model_kind = getattr(args, "_plan_field_model_kind", None)
    requested_model = getattr(args, "field_model", None)
    if planned_model_kind is not None:
        preset_name = getattr(args, "_plan_field_preset", None)
        model_kind = str(planned_model_kind)
        model_kwargs = dict(getattr(args, "_plan_field_model_kwargs", {}))
    elif requested_model is None:
        preset_name = str(getattr(args, "field_preset", "balanced"))
        preset = get_field_model_preset(preset_name)
        model_kind = str(preset.kind)
        model_kwargs = dict(preset.kwargs)
    else:
        preset_name = None
        model_kind = str(requested_model)
        model_kwargs = {}

    # Plan the product — use product pipeline for target planning.
    image_obs = [
        ImageObservation(path=path, hdu=hdu)
        for path, hdu in zip(field_paths, field_hdus)
    ]
    ref_header = getattr(args, "_plan_target_header", None)
    if ref_header is not None:
        ref_header = dict(ref_header)
    elif args.reference is not None:
        _, rh = read(
            str(args.reference),
            hdu=args.reference_hdu,
            return_header=True,
            device=read_dev,
        )
        ref_header = dict(rh)

    product = plan_map_product(
        image_obs,
        bands=band_specs,
        header_overrides=ref_header,
        union_footprint=bool(getattr(args, "union_footprint", False)),
        footprint_padding=float(getattr(args, "footprint_padding", 1.0)),
        footprint_tolerance_px=float(getattr(args, "footprint_tolerance_px", 0.25)),
        footprint_max_depth=int(getattr(args, "footprint_max_depth", 20)),
        max_footprint_pixels=getattr(args, "max_footprint_pixels", None),
        value_semantics="surface_brightness",
        device=args.device,
    )
    h, w = product.shape
    max_px = int(args.field_max_pixels)
    if h * w > max_px:
        raise SystemExit(
            f"field target {h}×{w} exceeds --field-max-pixels={max_px}; "
            "use a smaller subset, coadd first, or increase the limit (memory)."
        )
    th = int(args.field_render_tile)
    render_tiled = h > th or w > th

    resume_path = getattr(args, "resume", None)
    artifact_path = getattr(args, "artifact", None) or resume_path
    plan_path = getattr(args, "plan_json", None)
    plan_payload: dict[str, Any] | None = None
    if plan_path is not None:
        plan_inputs = []
        for (
            path,
            hdu,
            weight,
            background,
            weight_map,
            mask_map,
            observation_band,
        ) in zip(
            field_paths,
            field_hdus,
            field_weights,
            field_backgrounds,
            field_weight_maps,
            field_mask_maps,
            observation_bands,
        ):
            plan_inputs.append(
                {
                    "path": _resolved_path(path),
                    "hdu": int(hdu),
                    "identity": build_map_input_identity(path, hdu=hdu),
                    "weight": float(weight),
                    "background_offset": float(background),
                    "weight_map": _resolved_path(weight_map),
                    "weight_identity": _optional_input_identity(
                        weight_map, args.weight_map_hdu
                    ),
                    "mask_map": _resolved_path(mask_map),
                    "mask_identity": _optional_input_identity(
                        mask_map, args.mask_map_hdu
                    ),
                    "band": observation_band,
                }
            )
        plan_payload = {
            "schema_version": 2,
            "method": "field",
            "output": _resolved_path(args.output),
            "inputs": plan_inputs,
            "target": {
                "shape": [h, w],
                "band_count": len(unique_bands),
                "header": dict(product.target_header),
                "reference": _resolved_path(args.reference),
                "reference_hdu": int(args.reference_hdu),
                "union_footprint": bool(getattr(args, "union_footprint", False)),
                "footprint_padding": float(getattr(args, "footprint_padding", 1.0)),
                "footprint_tolerance_px": float(
                    getattr(args, "footprint_tolerance_px", 0.25)
                ),
                "footprint_max_depth": int(getattr(args, "footprint_max_depth", 20)),
            },
            "spectral": {
                "band": unique_bands[0] if len(unique_bands) == 1 else None,
                "bands": [
                    _spectral_plan_entry(band, profiles_by_band[band])
                    for band in unique_bands
                ],
                "observation_bands": observation_bands,
                "source": str(args.field_spectral_source),
                "spectral_min": (
                    profiles_by_band[unique_bands[0]].get("spectral_min")
                    if len(unique_bands) == 1
                    else None
                ),
                "spectral_max": (
                    profiles_by_band[unique_bands[0]].get("spectral_max")
                    if len(unique_bands) == 1
                    else None
                ),
                "transmission_center": (
                    profiles_by_band[unique_bands[0]].get("transmission_center")
                    if len(unique_bands) == 1
                    else None
                ),
                "transmission_sigma": (
                    profiles_by_band[unique_bands[0]].get("transmission_sigma")
                    if len(unique_bands) == 1
                    else None
                ),
                "transmission_path": _resolved_path(args.field_transmission),
                "transmission_identity": _optional_input_identity(
                    str(args.field_transmission)
                    if args.field_transmission is not None
                    else None,
                    0,
                ),
                "transmission_dir": _resolved_path(args.field_transmission_dir),
                "svo_filter_id": args.svo_filter_id,
            },
            "execution": {
                "field_preset": preset_name,
                "field_model": model_kind,
                "field_model_kwargs": model_kwargs,
                "field_steps": int(args.field_steps),
                "field_learning_rate": float(args.field_lr),
                "field_patch_size": int(args.field_patch_size),
                "field_patches_per_observation": int(args.field_patches_per_obs),
                "field_observations_per_step": int(args.field_obs_per_step),
                "field_spectral_samples": int(args.field_spectral_samples),
                "field_spatial_sampling": str(
                    getattr(args, "field_spatial_sampling", "center")
                ),
                "field_amp_dtype": str(getattr(args, "field_amp", "none")),
                "field_render_tile": th if render_tiled else None,
                "requested_device": args.device,
                "resolved_device": read_dev,
                "artifact": _resolved_path(artifact_path),
                "artifact_identities": _field_artifact_identities(artifact_path),
            },
            "requested_products": list(getattr(args, "output_planes", ("SCI",))),
            "software": _software_manifest(),
            "runtime": _runtime_manifest(args, read_dev),
        }
        _write_map_plan(
            Path(plan_path),
            MapPlanV2(plan_payload).to_dict(),
        )

    if getattr(args, "dry_run", False):
        n_obs = len(field_paths)
        print(
            f"torchsky map: dry-run output_shape={h}×{w} method=field "
            f"model={model_kind} bands={band_summary} render_tiled={render_tiled} "
            f"spatial_sampling={getattr(args, 'field_spatial_sampling', 'center')} "
            f"observations={n_obs} steps={getattr(args, 'field_steps', 0)} "
            f"device={torch.device(args.device) if args.device else default_compute_device()}",
            file=sys.stderr,
        )
        return 0

    fields = _paths_to_field_observations(
        field_paths,
        hdu=field_hdus,
        spectral_kw=spectral_kws,
        read_device=read_dev,
        weights=field_weights,
        background_offsets=field_backgrounds,
        weight_maps=field_weight_maps,
        mask_maps=field_mask_maps,
        weight_hdu=args.weight_map_hdu,
        mask_hdu=args.mask_map_hdu,
    )
    print(
        f"torchsky map field: {len(fields)} observation(s), "
        f"bands={band_summary}, model={model_kind}, "
        f"device={torch.device(args.device) if args.device else default_compute_device()}",
        file=sys.stderr,
    )
    domain = estimate_field_domain(fields, coord_mode="sphere_xyz")
    loaded_artifact = None
    input_identities: list[Mapping[str, Any]] | None = (
        [
            build_map_input_identity(path, hdu=hdu)
            for path, hdu in zip(field_paths, field_hdus)
        ]
        if artifact_path is not None
        else []
    )
    if resume_path is not None:
        loaded_artifact = load_field_artifact(resume_path, device=args.device)
        if loaded_artifact.manifest.payload["input_identities"] != input_identities:
            raise SystemExit("field artifact inputs do not match the requested inputs")
        if (
            requested_model is not None
            and loaded_artifact.manifest.model_kind != model_kind
        ):
            raise SystemExit("--field-model does not match the resumed artifact")
        model = loaded_artifact.model
        domain = loaded_artifact.manifest.domain
        model_kind = str(loaded_artifact.manifest.model_kind)
        model_kwargs = loaded_artifact.manifest.model_kwargs
        preset_name = loaded_artifact.manifest.payload["fit_config"].get("field_preset")
    else:
        model = make_field_model(cast(Any, model_kind), domain, **model_kwargs)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"torchsky map field: domain centre=({domain.center_ra_deg:.2f}°, {domain.center_dec_deg:.2f}°) "
        f"spatial_scale={domain.spatial_scale_deg:.4f}° "
        f"wavelength=[{domain.wavelength_center - domain.wavelength_scale:.0f}, "
        f"{domain.wavelength_center + domain.wavelength_scale:.0f}] Å "
        f"params={total_params:,}",
        file=sys.stderr,
    )
    patch = int(args.field_patch_size)
    steps = int(args.field_steps)
    fit_options = {
        "steps": steps,
        "lr": float(args.field_lr),
        "training_patch_size": None if patch <= 0 else patch,
        "training_patches_per_observation": int(args.field_patches_per_obs),
        "observations_per_step": min(len(fields), int(args.field_obs_per_step)),
        "device": args.device,
        "amp_dtype": str(getattr(args, "field_amp", "none")),
        "amp_scaler_state": (
            loaded_artifact.amp_scaler_state if loaded_artifact is not None else None
        ),
        "optimizer_state": (
            loaded_artifact.optimizer_state if loaded_artifact is not None else None
        ),
        "sampling_rng_state": (
            loaded_artifact.sampling_rng_state if loaded_artifact is not None else None
        ),
    }
    artifact_render_only = bool(getattr(args, "_artifact_render_only", False)) or (
        resume_path is not None and int(args.field_steps) == 0
    )
    fit_result = None
    if not artifact_render_only:
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None
            print(
                "torchsky map field: tqdm not installed — install 'tqdm' for progress bars",
                file=sys.stderr,
            )
        if tqdm is None:
            fit_result = fit_field_model(fields, model, **fit_options)
        else:
            bar = tqdm(total=steps, desc="field fit", unit="step", file=sys.stderr)

            def _progress_callback(step_idx: int, loss_val: float) -> None:
                bar.update(1)
                bar.set_postfix(loss=f"{loss_val:.4f}")

            try:
                fit_result = fit_field_model(
                    fields,
                    model,
                    **fit_options,
                    _progress_callback=_progress_callback,
                )
            finally:
                bar.close()
    if fit_result is not None:
        model = fit_result.model
        if loaded_artifact is not None:
            fit_result.completed_steps += loaded_artifact.completed_steps
    if artifact_path is not None and not artifact_render_only:
        if fit_result is None:
            raise RuntimeError("field fitting did not return resumable state")
        operator_names = (
            list(fit_result.operator_bundle.enabled_operator_names())
            if fit_result.operator_bundle is not None
            else []
        )
        save_field_artifact(
            artifact_path,
            model=model,
            model_kind=cast(Any, model_kind),
            model_kwargs=model_kwargs,
            domain=domain,
            operator_manifest={"enabled_operators": operator_names},
            calibration_manifest={
                "background_offsets": [float(value) for value in field_backgrounds],
            },
            fit_config={
                "field_preset": preset_name,
                "steps": int(fit_result.completed_steps),
                "learning_rate": float(args.field_lr),
                "patch_size": patch,
                "patches_per_observation": int(args.field_patches_per_obs),
                "observations_per_step": min(len(fields), int(args.field_obs_per_step)),
                "spectral_samples": int(args.field_spectral_samples),
                "amp_dtype": str(getattr(args, "field_amp", "none")),
                "observation_bands": observation_bands,
                "validation_split": [],
            },
            input_identities=input_identities,
            fit_result=fit_result,
            overwrite=resume_path is not None,
        )
        if plan_payload is not None and plan_path is not None:
            plan_payload["execution"]["artifact_identities"] = (
                _field_artifact_identities(artifact_path)
            )
            _write_map_plan(
                Path(plan_path),
                MapPlanV2(plan_payload).to_dict(),
            )
    model.eval()
    result = render_field_product(
        model,
        product,
        num_spectral_samples=int(args.field_spectral_samples),
        spatial_sampling=cast(
            FieldSpatialSampling,
            str(getattr(args, "field_spatial_sampling", "center")),
        ),
        spatial_tile_hw=(th, th) if render_tiled else None,
    )
    if not bool(getattr(args, "overwrite", True)) and Path(args.output).exists():
        raise SystemExit(
            f"torchsky map: output exists and --no-overwrite was given: {args.output}"
        )
    written_planes = write_map_product(
        args.output,
        result,
        requested_planes=getattr(args, "output_planes", ("SCI",)),
        overwrite=bool(getattr(args, "overwrite", True)),
    )
    print(
        f"wrote {args.output} shape={tuple(result.data.shape)} method=field "
        f"model={model_kind} bands={band_summary} "
        f"spatial_sampling={getattr(args, 'field_spatial_sampling', 'center')} "
        f"planes={','.join(written_planes)}",
        file=sys.stderr,
    )
    return 0


def register_map_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "map",
        help="High-SNR sky map: coadd (SWarp/drizzle-like) or continuous field render",
        description=(
            "Build a pixelized coadd or fit and render a continuous sky field from "
            "FITS observations. Select the workflow with --method coadd or --method field."
        ),
        epilog=(
            "Examples:\n"
            "  torchsky map coadd.fits exposure1.fits exposure2.fits\n"
            "  torchsky map field.fits exposure*.fits --method field --field-model sine "
            "--field-steps 200 --device cuda"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("output", help="Output FITS path (or directory with --per-filter)")
    p.add_argument(
        "inputs",
        nargs="*",
        help="Input FITS images (paths or globs); omit with --from-plan",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print planned output shape without writing a map",
    )
    p.add_argument(
        "--plan-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Atomically write the resolved map plan before compute",
    )
    p.add_argument(
        "--from-plan",
        type=Path,
        default=None,
        metavar="PATH",
        help="Replay a validated schema-v2 plan; only output/scratch may change",
    )
    p.add_argument(
        "--scratch-dir",
        type=Path,
        default=None,
        help="Scratch location for spillable reducers and large products",
    )
    p.add_argument(
        "--output-planes",
        nargs="+",
        choices=("SCI", "WHT", "IVAR", "CTX", "MASK", "NEXP", "NCORR", "DQ"),
        default=("SCI", "WHT", "IVAR", "MASK", "NEXP"),
        help=(
            "Requested output planes, recorded in plan-v2 "
            "(default: SCI WHT IVAR MASK NEXP; CTX and DQ are opt-in)"
        ),
    )
    p.add_argument(
        "--method",
        choices=("coadd", "field"),
        default="coadd",
        help="coadd: resample+stack; field: fit continuous field then render to grid",
    )
    p.add_argument(
        "--workflow",
        choices=tuple(_WORKFLOW_DEFAULTS),
        default="native",
        help="Calibrated map workflow preset; explicit controls override it",
    )
    p.add_argument("--hdu", default=0, type=int, help="Input image HDU index")
    p.add_argument(
        "--weights",
        default=None,
        nargs="*",
        type=float,
        help=(
            "Per-input scalar inverse-variance weights; when omitted, unit relative "
            "coadd weights do not produce IVAR"
        ),
    )
    p.add_argument(
        "--photometric-scales",
        default=None,
        nargs="*",
        type=float,
        help="Per-input coadd calibration scale applied as scale * (image - background)",
    )
    p.add_argument(
        "--background-offsets",
        default=None,
        nargs="*",
        type=float,
        help="Per-input background offsets; coadd subtracts them, fields model them in rendering",
    )
    p.add_argument(
        "--background-mode",
        choices=("none", "per-exposure", "overlap"),
        default=None,
        help="Background policy; default is selected by --workflow",
    )
    p.add_argument(
        "--sky-method",
        default="mesh",
        choices=("mesh", "polynomial"),
        help="Sky model type (default: mesh)",
    )
    p.add_argument(
        "--sky-statistic",
        default="huber",
        choices=("median", "huber", "tukey"),
        help="Sky per-cell statistic (default: huber)",
    )
    p.add_argument(
        "--sky-block-size",
        type=int,
        default=128,
        help="Mesh block size in pixels (mesh method, default: 128)",
    )
    p.add_argument(
        "--sky-mesh-filter",
        default="median",
        choices=("median", "gaussian", "none"),
        help="Mesh post-filter (mesh method, default: median)",
    )
    p.add_argument(
        "--sky-mesh-filter-size",
        type=int,
        default=3,
        help="Mesh post-filter kernel size (mesh method, default: 3)",
    )
    p.add_argument(
        "--sky-interpolate",
        default="bicubic",
        choices=("bilinear", "bicubic"),
        help="Mesh-to-image interpolation (mesh method, default: bicubic)",
    )
    p.add_argument(
        "--sky-order",
        type=int,
        default=3,
        help="Polynomial degree (polynomial method, default: 3)",
    )
    p.add_argument(
        "--weight-maps",
        default=None,
        nargs="*",
        help="Per-input FITS weight maps; multiplied by --weights when set",
    )
    p.add_argument(
        "--weight-map-hdu",
        default=0,
        type=int,
        help="HDU index for all --weight-maps",
    )
    p.add_argument(
        "--mask-maps",
        default=None,
        nargs="*",
        help="Per-input FITS mask maps; nonzero pixels contribute",
    )
    p.add_argument(
        "--mask-map-hdu",
        default=0,
        type=int,
        help="HDU index for all --mask-maps",
    )
    p.add_argument(
        "--dq-maps",
        default=None,
        nargs="*",
        help="Per-input DQ bitmask maps; rejected bits follow --dq-policy",
    )
    p.add_argument(
        "--dq-hdu",
        default="DQ",
        metavar="HDU",
        help="HDU name or index for all --dq-maps (default: DQ)",
    )
    p.add_argument(
        "--good-bits",
        type=int,
        default=None,
        metavar="MASK",
        help=(
            "Bits that must be zero for a pixel to contribute when a mask map is "
            "read as a bitmask (instead of a boolean/weight mask)"
        ),
    )
    p.add_argument(
        "--primary-beam-maps",
        default=None,
        nargs="*",
        help="Per-input primary-beam maps for coadd weighting",
    )
    p.add_argument(
        "--primary-beam-hdu",
        default=None,
        type=int,
        help="HDU index for all --primary-beam-maps; default follows torchsky image read",
    )
    p.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="FITS whose WCS+shape define the output grid",
    )
    p.add_argument(
        "--reference-hdu",
        default=0,
        type=int,
        help="Zero-based image HDU index in --reference (default: 0)",
    )
    p.add_argument(
        "--plan-reference",
        type=int,
        default=0,
        help="Which input (0-based) defines grid when --reference omitted (coadd/field)",
    )
    p.add_argument(
        "--stack",
        dest="stack",
        default=None,
        choices=("weighted_mean", "mean", "median", "sigma_clipped"),
        help="Stacking after resampling (coadd)",
    )
    p.add_argument(
        "--clip-sigma",
        type=float,
        default=4.0,
        help="Robust sigma threshold for --stack=sigma_clipped (default: 4)",
    )
    p.add_argument(
        "--clip-iterations",
        type=int,
        default=3,
        help="Maximum iterations for --stack=sigma_clipped (default: 3)",
    )
    p.add_argument(
        "--kernel",
        default=None,
        choices=(
            "nearest",
            "bilinear",
            "bicubic",
            "lanczos3",
            "lanczos5",
            "mitchell_netravali",
            "catmull_rom",
            "fft_shift",
            "fft_diagonal_affine",
            "periodic_linear_affine",
            "fft_linear_affine",
            "drizzle_point",
            "drizzle_square",
            "conservative_polygon",
        ),
        help="Resampling kernel (coadd); drizzle_* uses native forward-drizzle when available",
    )
    p.add_argument(
        "--drizzle-pixfrac",
        type=float,
        default=1.0,
        help="Drizzle pixfrac when using drizzle_* kernels (coadd)",
    )
    p.add_argument(
        "--value-semantics",
        default="surface_brightness",
        choices=("surface_brightness", "flux"),
        help="Input pixel value meaning used for area scaling (default: surface_brightness)",
    )
    p.add_argument(
        "--auto-tile",
        dest="auto_tile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tile large outputs into spatial chunks (coadd); use --no-auto-tile to disable",
    )
    p.add_argument(
        "--tile-pixels",
        type=int,
        default=None,
        help="Force tile height=width in pixels (coadd); implies tiling",
    )
    p.add_argument(
        "--tile-above-megapixels",
        type=float,
        default=32.0,
        help="With --auto-tile, tile when H×W exceeds this many megapixels (coadd)",
    )
    p.add_argument(
        "--field-preset",
        choices=("fast", "balanced", "spectral", "operator_constrained"),
        default="balanced",
        help="Supported field recipe (default: balanced)",
    )
    p.add_argument(
        "--field-amp",
        choices=("none", "float16", "bfloat16"),
        default="none",
        help="Opt-in CUDA autocast precision with local gradient scaling",
    )
    p.add_argument(
        "--field-model",
        dest="field_model",
        default=None,
        metavar="MODEL",
        help="Experimental raw field model kind; overrides --field-preset",
    )
    p.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help="Save a safe resumable field artifact directory",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume model, optimizer, and RNG state from a field artifact",
    )
    p.add_argument(
        "--field-band",
        dest="field_band",
        default=None,
        metavar="BAND",
        help="Filter name for the field bandpass, e.g. F277W",
    )
    p.add_argument(
        "--field-spectral-source",
        dest="field_spectral_source",
        choices=("svo", "builtin", "file", "gaussian"),
        default="svo",
        help=(
            "How to set FieldObservation spectral metadata: "
            "svo=astroquery SVO FPS (default; needs pip install astroquery), "
            "builtin=small JWST table in torchsky, "
            "file=two-column wavelength transmission table, "
            "gaussian=effective wavelength + min/max (Gaussian sigma from width or --field-gaussian-sigma)"
        ),
    )
    p.add_argument(
        "--svo-facility",
        default="JWST",
        help="SVO get_filter_list facility (--field-spectral-source=svo)",
    )
    p.add_argument(
        "--svo-instrument",
        default="NIRCAM",
        help="SVO get_filter_list instrument (--field-spectral-source=svo)",
    )
    p.add_argument(
        "--svo-filter-id",
        default=None,
        metavar="ID",
        help=(
            "Exact SVO filterID (e.g. JWST/NIRCam.F277W); if set, transmission is "
            "downloaded for this ID (--field-band still labels band_name)"
        ),
    )
    p.add_argument(
        "--svo-prefer",
        choices=("classic", "2025", "any"),
        default="classic",
        help="When multiple SVO rows match a band token, prefer classic vs NIRCam2025",
    )
    p.add_argument(
        "--svo-cache-dir",
        type=Path,
        default=None,
        help="Directory to cache downloaded SVO filter curves (defaults to $TORCHSKY_SVO_CACHE or ~/.cache/torchsky/svo)",
    )
    p.add_argument(
        "--no-svo-cache",
        action="store_true",
        help="Disable reading or writing persistent local disk cache for SVO filter profiles",
    )
    p.add_argument(
        "--field-transmission",
        dest="field_transmission",
        type=Path,
        default=None,
        metavar="PATH",
        help="Two-column wavelength, transmission file (--field-spectral-source=file)",
    )
    p.add_argument(
        "--field-transmission-dir",
        dest="field_transmission_dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory with {BAND}.txt|dat|csv per filter (--per-filter + file source)"
        ),
    )
    p.add_argument(
        "--field-transmission-unit",
        dest="field_transmission_unit",
        choices=("angstrom", "nm"),
        default="angstrom",
        help="Wavelength unit in --field-transmission file columns",
    )
    p.add_argument(
        "--field-gaussian-lambda-eff",
        dest="field_gaussian_lambda_eff",
        type=float,
        default=None,
        metavar="ANGSTROM",
        help="Effective wavelength (Å) for gaussian spectral source",
    )
    p.add_argument(
        "--field-gaussian-lambda-min",
        dest="field_gaussian_lambda_min",
        type=float,
        default=None,
        metavar="ANGSTROM",
        help="Bandpass lower edge (Å) for gaussian spectral source",
    )
    p.add_argument(
        "--field-gaussian-lambda-max",
        dest="field_gaussian_lambda_max",
        type=float,
        default=None,
        metavar="ANGSTROM",
        help="Bandpass upper edge (Å) for gaussian spectral source",
    )
    p.add_argument(
        "--field-gaussian-sigma",
        dest="field_gaussian_sigma",
        type=float,
        default=None,
        metavar="ANGSTROM",
        help="Gaussian transmission sigma (Å); default from lambda_max-lambda_min FWHM",
    )
    p.add_argument(
        "--field-quiet-svo",
        dest="field_quiet_svo",
        action="store_true",
        help="Do not log resolved SVO filterID to stderr",
    )
    p.add_argument(
        "--field-steps",
        dest="field_steps",
        type=int,
        default=80,
        metavar="N",
        help="Number of field-optimization steps (default: 80)",
    )
    p.add_argument(
        "--field-lr",
        dest="field_lr",
        type=float,
        default=5.0e-3,
        metavar="RATE",
        help="Field optimizer learning rate (default: 0.005)",
    )
    p.add_argument(
        "--field-patch-size",
        dest="field_patch_size",
        type=int,
        default=64,
        metavar="PIXELS",
        help="Training crop size in pixels; 0 = full frame each step",
    )
    p.add_argument(
        "--field-patches-per-obs",
        dest="field_patches_per_obs",
        type=int,
        default=2,
        metavar="N",
        help="Training patches sampled from each observation per step (default: 2)",
    )
    p.add_argument(
        "--field-obs-per-step",
        dest="field_obs_per_step",
        type=int,
        default=2,
        metavar="N",
        help="Observations sampled for each field-optimization step (default: 2)",
    )
    p.add_argument(
        "--field-spectral-samples",
        dest="field_spectral_samples",
        type=int,
        default=9,
        metavar="N",
        help="Quadrature samples across each observation passband (default: 9)",
    )
    p.add_argument(
        "--field-spatial-sampling",
        dest="field_spatial_sampling",
        choices=("center", "gauss_legendre_2x2"),
        default="center",
        help=(
            "Detector-pixel sampling for field rendering: center (fast) or "
            "gauss_legendre_2x2 (anti-aliased pixel average)"
        ),
    )
    p.add_argument(
        "--field-render-tile",
        dest="field_render_tile",
        type=int,
        default=512,
        metavar="PIXELS",
        help="spatial_tile_hw for render when the field map is larger than this",
    )
    p.add_argument(
        "--field-max-pixels",
        dest="field_max_pixels",
        type=int,
        default=2048 * 2048,
        metavar="N",
        help="Refuse field rendering if output grid exceeds this many pixels",
    )
    p.add_argument(
        "--per-filter",
        action="store_true",
        help=(
            "Group inputs by FITS FILTER and write one product per filter "
            "(coadd and field). Without it, inputs from different filters are rejected."
        ),
    )
    p.add_argument(
        "--union-footprint",
        dest="union_footprint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Auto-compute output WCS from union bounding box of all input footprints "
            "(like SWarp). Use --no-union-footprint to use the first input as reference."
        ),
    )
    p.add_argument(
        "--footprint-padding",
        type=float,
        default=1.0,
        help="Whole-pixel padding factor for the union footprint (default: 1.0 = exact)",
    )
    p.add_argument(
        "--footprint-tolerance-px",
        type=float,
        default=0.25,
        help="Maximum curved-WCS boundary chord error in target pixels",
    )
    p.add_argument(
        "--footprint-max-depth",
        type=int,
        default=20,
        help="Maximum adaptive boundary subdivision depth; failure is closed",
    )
    p.add_argument(
        "--max-footprint-pixels",
        type=int,
        default=None,
        help="Refuse targets exceeding this many pixels (memory guard, coadd/field)",
    )
    p.add_argument(
        "--overwrite",
        dest="overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite an existing output file (use --no-overwrite to fail instead)",
    )
    p.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help=(
            "Tensor device for FITS I/O and map math (e.g. cuda, cuda:1, cpu). "
            "Default: CUDA when available else CPU; also settable via TORCHSKY_DEVICE."
        ),
    )
    p.set_defaults(handler=_dispatch_map)


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
        return _run_field_single(args)

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
    return _run_coadd(args)


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
    expanded = _expand_inputs(args.inputs)
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
        return _run_coadd(args)
    return _run_field(args)
