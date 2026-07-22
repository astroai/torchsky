"""``torchsky map`` — high-SNR image-plane coadd or INR field render."""

from __future__ import annotations

import argparse
import glob
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import torch

from ..device import default_compute_device
from ..io import HDUList, TensorHDU, read, write
from ..map.field import (
    FieldObservation,
    estimate_field_domain,
    fit_field_model,
    make_field_model,
)
from ..map.image_plane import (
    ImageObservation,
)
from ..map.product import (
    MapBandpassSpec,
    plan_map_product,
    render_field_product,
    stack_map_product,
)

from ..filter_profiles import field_observation_spectral_kwargs, resolve_inr_bandpass

# Minimal bandpass hints for INR when FITS headers lack spectral metadata (Angstrom).
_INR_BAND_TABLE: dict[str, tuple[float, float, float, float]] = {
    # (spectral_min, spectral_max, transmission_center, transmission_sigma)
    "F090W": (8000.0, 10000.0, 9000.0, 500.0),
    "F150W": (13000.0, 17000.0, 15000.0, 700.0),
    "F200W": (17000.0, 23000.0, 20000.0, 900.0),
    "F277W": (24000.0, 32000.0, 27700.0, 1200.0),
    "F356W": (31000.0, 40000.0, 35600.0, 1400.0),
    "F444W": (38000.0, 50000.0, 44400.0, 1800.0),
}


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


def _inr_spectral_field_kwargs(args: argparse.Namespace, band: str) -> dict:
    """Spectral keyword args for :class:`FieldObservation` (INR only)."""
    try:
        raw = resolve_inr_bandpass(
            source=str(args.inr_spectral_source),
            band_name=str(band).strip().upper(),
            builtin_table=_INR_BAND_TABLE,
            svo_facility=str(args.svo_facility),
            svo_instrument=str(args.svo_instrument),
            svo_filter_id=args.svo_filter_id,
            svo_prefer=str(args.svo_prefer),
            transmission_path=args.inr_transmission,
            transmission_dir=args.inr_transmission_dir,
            transmission_unit=str(args.inr_transmission_unit),
            gaussian_lambda_eff=args.inr_gaussian_lambda_eff,
            gaussian_lambda_min=args.inr_gaussian_lambda_min,
            gaussian_lambda_max=args.inr_gaussian_lambda_max,
            gaussian_sigma=args.inr_gaussian_sigma,
        )
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc
    except ValueError as exc:
        raise SystemExit(f"torchsky map inr: {exc}") from exc
    fid = raw.get("svo_filter_id")
    if (
        str(args.inr_spectral_source) == "svo"
        and fid
        and not getattr(args, "inr_quiet_svo", False)
    ):
        print(
            f"torchsky map inr: SVO bandpass {str(band).upper()} -> {fid}",
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
            for i, h in enumerate(hdul):
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
    spectral_kw: dict,
    read_device: str,
    weights: Sequence[float] | None = None,
    background_offsets: Sequence[float] | None = None,
    weight_maps: Sequence[str | None] | None = None,
    mask_maps: Sequence[str | None] | None = None,
    weight_hdu: int | str | None = 0,
    mask_hdu: int | str | None = 0,
) -> list[FieldObservation]:
    skw = spectral_kw
    out: list[FieldObservation] = []
    scalar_weights = list(weights) if weights is not None else [1.0] * len(paths)
    backgrounds = (
        list(background_offsets) if background_offsets is not None else [0.0] * len(paths)
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
    hdus: Sequence[int]
    if isinstance(hdu, int):
        hdus = [hdu] * len(paths)
    else:
        hdus = list(hdu)
        if len(hdus) != len(paths):
            raise ValueError("hdus (len={}) count must match paths count ({})".format(len(hdus), len(paths)))
    for p, hdu_i, scalar_weight, background_offset, weight_path, mask_path in zip(
        paths, hdus, scalar_weights, backgrounds, wpaths, mpaths
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
    inputs = args.inputs
    if not inputs:
        print("torchsky map: no input files matched", file=sys.stderr)
        return 2
    read_dev = _map_read_device_str(args)

    # Expand MEFs: probe each file for image HDUs and create per-HDU entries.
    expanded_paths: list[str] = []
    expanded_hdus: list[int] = []
    expanded_weights: list[float] = []
    expanded_photometric_scales: list[float] = []
    expanded_background_offsets: list[float] = []
    expanded_weight_maps: list[str | None] = []
    expanded_mask_maps: list[str | None] = []
    expanded_primary_beam_maps: list[str | None] = []

    user_hdu = int(args.hdu)
    for i, p in enumerate(inputs):
        image_hdus = _probe_image_hdus(p)
        if len(image_hdus) == 0:
            expanded_paths.append(p)
            expanded_hdus.append(user_hdu)
            expanded_weights.append(float(args.weights[i]))
            expanded_photometric_scales.append(float(args.photometric_scales[i]))
            expanded_background_offsets.append(float(args.background_offsets[i]))
            expanded_weight_maps.append(args.weight_maps[i])
            expanded_mask_maps.append(args.mask_maps[i])
            expanded_primary_beam_maps.append(args.primary_beam_maps[i])
        elif len(image_hdus) == 1:
            hdu = user_hdu if user_hdu in image_hdus else image_hdus[0]
            expanded_paths.append(p)
            expanded_hdus.append(hdu)
            expanded_weights.append(float(args.weights[i]))
            expanded_photometric_scales.append(float(args.photometric_scales[i]))
            expanded_background_offsets.append(float(args.background_offsets[i]))
            expanded_weight_maps.append(args.weight_maps[i])
            expanded_mask_maps.append(args.mask_maps[i])
            expanded_primary_beam_maps.append(args.primary_beam_maps[i])
        else:
            if user_hdu != 0 and user_hdu in image_hdus:
                expanded_paths.append(p)
                expanded_hdus.append(user_hdu)
                expanded_weights.append(float(args.weights[i]))
                expanded_photometric_scales.append(float(args.photometric_scales[i]))
                expanded_background_offsets.append(float(args.background_offsets[i]))
                expanded_weight_maps.append(args.weight_maps[i])
                expanded_mask_maps.append(args.mask_maps[i])
                expanded_primary_beam_maps.append(args.primary_beam_maps[i])
            else:
                for hdu in image_hdus:
                    expanded_paths.append(p)
                    expanded_hdus.append(hdu)
                    expanded_weights.append(float(args.weights[i]))
                    expanded_photometric_scales.append(float(args.photometric_scales[i]))
                    expanded_background_offsets.append(float(args.background_offsets[i]))
                    expanded_weight_maps.append(args.weight_maps[i])
                    expanded_mask_maps.append(args.mask_maps[i])
                    expanded_primary_beam_maps.append(args.primary_beam_maps[i])
                n_hdus = len(image_hdus)
                print(
                    f"  {p}: auto-expanded {n_hdus} image HDUs {list(image_hdus)}",
                    file=sys.stderr,
                )

    observations = [
        ImageObservation(
            path=p,
            hdu=hdu,
            weight=float(w),
            photometric_scale=float(ps),
            background_offset=float(bg),
            weight_path=wp,
            weight_hdu=args.weight_map_hdu,
            mask_path=mp,
            mask_hdu=args.mask_map_hdu,
            primary_beam_path=pbp,
            primary_beam_hdu=args.primary_beam_hdu,
        )
        for p, hdu, w, ps, bg, wp, mp, pbp in zip(
            expanded_paths, expanded_hdus, expanded_weights,
            expanded_photometric_scales, expanded_background_offsets,
            expanded_weight_maps, expanded_mask_maps, expanded_primary_beam_maps,
        )
    ]

    if getattr(args, "sky", False):
        print(
            "torchsky map: --sky is no longer built into the coadd pipeline. "
            "Pre-compute per-exposure backgrounds (e.g. via the MegaCam benchmark "
            "script's estimate_background_offsets) and pass them with --background-offsets.",
            file=sys.stderr,
        )
    ref_header = None
    if args.reference is not None:
        _, rh = read(str(args.reference), hdu=args.reference_hdu, return_header=True, device=read_dev)
        ref_header = dict(rh)

    union_footprint = bool(getattr(args, "union_footprint", False))
    reference_index = int(args.plan_reference)
    if not union_footprint and not 0 <= reference_index < len(observations):
        raise SystemExit(
            f"--plan-reference {reference_index} out of range for "
            f"{len(observations)} observations"
        )

    product = plan_map_product(
        observations,
        reference_index=reference_index,
        header_overrides=ref_header,
        union_footprint=union_footprint,
        footprint_padding=float(getattr(args, "footprint_padding", 1.0)),
        max_footprint_pixels=getattr(args, "max_footprint_pixels", None),
        value_semantics=args.value_semantics,
        device=args.device,
    )

    tile_hw = _compute_spatial_tile_hw(
        product.shape,
        auto_tile=bool(args.auto_tile),
        tile_pixels=args.tile_pixels,
        tile_above_megapixels=float(args.tile_above_megapixels),
    )
    if tile_hw is not None:
        print(
            f"torchsky map: spatial tiling {tile_hw[0]}×{tile_hw[1]} "
            f"for output {product.shape} pixels",
            file=sys.stderr,
        )

    if getattr(args, "dry_run", False):
        print(
            f"torchsky map: dry-run output_shape={product.shape} "
            f"method=coadd kernel={args.kernel} stack={args.stack} "
            f"value_semantics={args.value_semantics} tiled={tile_hw is not None} "
            f"n_bands={product.n_bands}",
            file=sys.stderr,
        )
        return 0

    result = stack_map_product(
        observations,
        product,
        resampling_kernel=args.kernel,
        stacking_method=args.stack,
        drizzle_pixfrac=float(args.drizzle_pixfrac),
        spatial_tile_hw=tile_hw,
        device=args.device,
    )
    write(str(args.output), result.data.squeeze(0), header=result.header, overwrite=True)
    print(
        f"wrote {args.output} shape={tuple(result.data.shape)} "
        f"mean_weight={float(result.weight.mean().item()):.4f} "
        f"method=coadd kernel={args.kernel} stack={args.stack} "
        f"value_semantics={args.value_semantics} tiled={tile_hw is not None} "
        f"n_bands={result.n_bands}",
        file=sys.stderr,
    )
    return 0


def _run_inr(args: argparse.Namespace) -> int:
    inputs = _expand_inputs(args.inputs)
    if not inputs:
        print("torchsky map: no input files matched", file=sys.stderr)
        return 2
    read_dev = _map_read_device_str(args)
    if args.per_filter:
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, p in enumerate(inputs):
            _, hdr = read(p, hdu=args.hdu, return_header=True, device=read_dev)
            fk = _header_filter_keyword(dict(hdr)) or "UNKNOWN"
            groups[fk].append(idx)
        rc = 0
        out_path = Path(args.output)
        for fk, indices in sorted(groups.items()):
            gpaths = [inputs[i] for i in indices]
            if fk == "UNKNOWN" and args.inr_band is None:
                print(
                    f"torchsky map inr: skip UNKNOWN filter group ({len(gpaths)} files); "
                    "set --inr-band",
                    file=sys.stderr,
                )
                continue
            band = str(args.inr_band).upper() if fk == "UNKNOWN" else str(fk).upper()
            sub_out = (
                out_path / f"{_safe_name(fk)}.fits"
                if out_path.suffix == "" or out_path.is_dir()
                else out_path.parent
                / f"{out_path.stem}_{_safe_name(fk)}{out_path.suffix}"
            )
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
            ns.inr_band = band
            rc = rc or _run_inr_single(ns)
        return rc
    return _run_inr_single(args)


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s) or "band"


def _expand_paths_for_inr(
    paths: list[str],
    hdu: int,
    weights: list[float],
    background_offsets: list[float],
    weight_maps: list[str | None],
    mask_maps: list[str | None],
) -> tuple[list[str], list[int], list[float], list[float], list[str | None], list[str | None]]:
    """Expand MEF files into per-HDU entries for INR processing."""
    expanded_paths: list[str] = []
    expanded_hdus: list[int] = []
    expanded_weights: list[float] = []
    expanded_background_offsets: list[float] = []
    expanded_weight_maps: list[str | None] = []
    expanded_mask_maps: list[str | None] = []

    for i, p in enumerate(paths):
        image_hdus = _probe_image_hdus(p)
        if len(image_hdus) == 0:
            expanded_paths.append(p)
            expanded_hdus.append(hdu)
            expanded_weights.append(float(weights[i]))
            expanded_background_offsets.append(float(background_offsets[i]))
            expanded_weight_maps.append(weight_maps[i])
            expanded_mask_maps.append(mask_maps[i])
        elif len(image_hdus) == 1:
            h = hdu if hdu in image_hdus else image_hdus[0]
            expanded_paths.append(p)
            expanded_hdus.append(h)
            expanded_weights.append(float(weights[i]))
            expanded_background_offsets.append(float(background_offsets[i]))
            expanded_weight_maps.append(weight_maps[i])
            expanded_mask_maps.append(mask_maps[i])
        else:
            if hdu != 0 and hdu in image_hdus:
                expanded_paths.append(p)
                expanded_hdus.append(hdu)
                expanded_weights.append(float(weights[i]))
                expanded_background_offsets.append(float(background_offsets[i]))
                expanded_weight_maps.append(weight_maps[i])
                expanded_mask_maps.append(mask_maps[i])
            else:
                for h in image_hdus:
                    expanded_paths.append(p)
                    expanded_hdus.append(h)
                    expanded_weights.append(float(weights[i]))
                    expanded_background_offsets.append(float(background_offsets[i]))
                    expanded_weight_maps.append(weight_maps[i])
                    expanded_mask_maps.append(mask_maps[i])
    return (
        expanded_paths, expanded_hdus,
        expanded_weights, expanded_background_offsets,
        expanded_weight_maps, expanded_mask_maps,
    )


def _run_inr_single(args: argparse.Namespace) -> int:
    inputs = _expand_inputs(args.inputs)
    read_dev = _map_read_device_str(args)
    inr_paths, inr_hdus, inr_weights, inr_bg, inr_wmaps, inr_mmaps = _expand_paths_for_inr(
        inputs,
        hdu=args.hdu,
        weights=args.weights,
        background_offsets=args.background_offsets,
        weight_maps=args.weight_maps,
        mask_maps=args.mask_maps,
    )

    if args.inr_band is None:
        _, h0 = read(inr_paths[0], hdu=inr_hdus[0], return_header=True, device=read_dev)
        fk = _header_filter_keyword(dict(h0))
        if fk is None:
            raise SystemExit(
                "INR requires --inr-band (e.g. F277W) or FITS FILTER/FILTNAM in header"
            )
        band = str(fk).upper()
        if str(args.inr_spectral_source) == "builtin" and band not in _INR_BAND_TABLE:
            raise SystemExit(
                f"Band {band!r} not in builtin INR table; use --inr-spectral-source=svo, "
                "file, or gaussian, or pass --inr-band with a known name"
            )
    else:
        band = str(args.inr_band).upper()

    skw = _inr_spectral_field_kwargs(args, band)

    # Plan the product — use product pipeline for target planning.
    image_obs = [ImageObservation(path=p, hdu=h) for p, h in zip(inr_paths, inr_hdus)]
    ref_header = None
    if args.reference is not None:
        _, rh = read(str(args.reference), hdu=args.reference_hdu, return_header=True, device=read_dev)
        ref_header = dict(rh)

    band_spec = MapBandpassSpec(
        label=band,
        transmission_center=skw.get("transmission_center"),
        transmission_sigma=skw.get("transmission_sigma"),
        spectral_min=skw.get("spectral_min"),
        spectral_max=skw.get("spectral_max"),
    )
    product = plan_map_product(
        image_obs,
        bands=[band_spec],
        header_overrides=ref_header,
        union_footprint=bool(getattr(args, "union_footprint", False)),
        footprint_padding=float(getattr(args, "footprint_padding", 1.0)),
        max_footprint_pixels=getattr(args, "max_footprint_pixels", None),
        value_semantics="surface_brightness",
        device=args.device,
    )
    h, w = product.shape
    max_px = int(args.inr_max_pixels)
    if h * w > max_px:
        raise SystemExit(
            f"INR target {h}×{w} exceeds --inr-max-pixels={max_px}; "
            "use a smaller subset, coadd first, or increase the limit (memory)."
        )
    th = int(args.inr_render_tile)
    render_tiled = h > th or w > th

    if getattr(args, "dry_run", False):
        n_obs = len(inr_paths)
        print(
            f"torchsky map: dry-run output_shape={h}×{w} method=inr "
            f"model={args.inr_model} band={band} render_tiled={render_tiled} "
            f"observations={n_obs} steps={getattr(args, 'inr_steps', 0)} "
            f"device={torch.device(args.device) if args.device else default_compute_device()}",
            file=sys.stderr,
        )
        return 0

    fields = _paths_to_field_observations(
        inr_paths,
        hdu=inr_hdus,
        spectral_kw=skw,
        read_device=read_dev,
        weights=inr_weights,
        background_offsets=inr_bg,
        weight_maps=inr_wmaps,
        mask_maps=inr_mmaps,
        weight_hdu=args.weight_map_hdu,
        mask_hdu=args.mask_map_hdu,
    )
    print(
        f"torchsky map inr: {len(fields)} observation(s), "
        f"band={band}, model={args.inr_model}, "
        f"device={torch.device(args.device) if args.device else default_compute_device()}",
        file=sys.stderr,
    )
    domain = estimate_field_domain(fields, coord_mode="sphere_xyz")
    model = make_field_model(args.inr_model, domain)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"torchsky map inr: domain centre=({domain.center_ra_deg:.2f}°, {domain.center_dec_deg:.2f}°) "
        f"spatial_scale={domain.spatial_scale_deg:.4f}° "
        f"wavelength=[{domain.wavelength_center - domain.wavelength_scale:.0f}, "
        f"{domain.wavelength_center + domain.wavelength_scale:.0f}] Å "
        f"params={total_params:,}",
        file=sys.stderr,
    )
    patch = int(args.inr_patch_size)
    steps = int(args.inr_steps)
    try:
        from tqdm import tqdm as _tqdm

        def _progress_callback(step_idx: int, loss_val: float) -> None:
            _progress_callback._bar.update(1)
            _progress_callback._bar.set_postfix(loss=f"{loss_val:.4f}")

        _progress_callback._bar = _tqdm(
            total=steps, desc="INR fit", unit="step", file=sys.stderr
        )
        if not _progress_callback._bar.disable:
            fit_field_model(
                fields,
                model,
                steps=steps,
                lr=float(args.inr_lr),
                training_patch_size=None if patch <= 0 else patch,
                training_patches_per_observation=int(args.inr_patches_per_obs),
                observations_per_step=min(len(fields), int(args.inr_obs_per_step)),
                device=args.device,
                _progress_callback=_progress_callback,
            )
        else:
            fit_field_model(
                fields,
                model,
                steps=steps,
                lr=float(args.inr_lr),
                training_patch_size=None if patch <= 0 else patch,
                training_patches_per_observation=int(args.inr_patches_per_obs),
                observations_per_step=min(len(fields), int(args.inr_obs_per_step)),
                device=args.device,
            )
        _progress_callback._bar.close()
    except ImportError:
        print(
            "torchsky map inr: tqdm not installed — install 'tqdm' for progress bars",
            file=sys.stderr,
        )
        fit_field_model(
            fields,
            model,
            steps=steps,
            lr=float(args.inr_lr),
            training_patch_size=None if patch <= 0 else patch,
            training_patches_per_observation=int(args.inr_patches_per_obs),
            observations_per_step=min(len(fields), int(args.inr_obs_per_step)),
            device=args.device,
        )
    model.eval()
    result = render_field_product(
        model,
        product,
        num_spectral_samples=int(args.inr_spectral_samples),
        spatial_tile_hw=(th, th) if render_tiled else None,
    )
    write(str(args.output), result.data.squeeze(0), header=result.header, overwrite=True)
    print(
        f"wrote {args.output} shape={tuple(result.data.shape)} method=inr "
        f"model={args.inr_model} band={band}",
        file=sys.stderr,
    )
    return 0


def register_map_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "map",
        help="High-SNR sky map: coadd (SWarp/drizzle-like) or INR field render",
    )
    p.add_argument("output", help="Output FITS path (or directory with --per-filter)")
    p.add_argument(
        "inputs",
        nargs="+",
        help="Input FITS images (paths or globs)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print planned output shape without writing a map",
    )
    p.add_argument(
        "--method",
        choices=("coadd", "inr"),
        default="coadd",
        help="coadd: resample+stack; inr: fit continuous field then render to grid",
    )
    p.add_argument("--hdu", default=0, type=int, help="Input image HDU index")
    p.add_argument(
        "--weights",
        default=None,
        nargs="*",
        type=float,
        help="Per-input scalar inverse-variance weights; default 1.0 each",
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
        help="Per-input background offsets; coadd subtracts them, INR models them in rendering",
    )
    p.add_argument(
        "--sky",
        action="store_true",
        help="Auto-estimate and subtract 2D sky background per input (replaces --background-offsets)",
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
        default="median_3",
        choices=("median_3", "gaussian_3", "none"),
        help="Mesh post-filter (mesh method, default: median_3)",
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
    p.add_argument("--reference-hdu", default=0, type=int)
    p.add_argument(
        "--plan-reference",
        type=int,
        default=0,
        help="Which input (0-based) defines grid when --reference omitted (coadd/inr)",
    )
    p.add_argument(
        "--stack",
        dest="stack",
        default="weighted_mean",
        choices=("weighted_mean", "mean", "median"),
        help="Stacking after resampling (coadd)",
    )
    p.add_argument(
        "--kernel",
        default="bilinear",
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
        "--inr-model",
        default="sine_implicit",
        help="Field model kind for make_field_model (inr)",
    )
    p.add_argument(
        "--inr-band", default=None, help="Filter name for bandpass, e.g. F277W (inr)"
    )
    p.add_argument(
        "--inr-spectral-source",
        choices=("svo", "builtin", "file", "gaussian"),
        default="svo",
        help=(
            "How to set FieldObservation spectral metadata for INR: "
            "svo=astroquery SVO FPS (default; needs pip install astroquery), "
            "builtin=small JWST table in torchsky, "
            "file=two-column wavelength transmission table, "
            "gaussian=effective wavelength + min/max (Gaussian sigma from width or --inr-gaussian-sigma)"
        ),
    )
    p.add_argument(
        "--svo-facility",
        default="JWST",
        help="SVO get_filter_list facility (inr --inr-spectral-source=svo)",
    )
    p.add_argument(
        "--svo-instrument",
        default="NIRCAM",
        help="SVO get_filter_list instrument (inr --inr-spectral-source=svo)",
    )
    p.add_argument(
        "--svo-filter-id",
        default=None,
        metavar="ID",
        help=(
            "Exact SVO filterID (e.g. JWST/NIRCam.F277W); if set, transmission is "
            "downloaded for this ID (--inr-band still labels band_name)"
        ),
    )
    p.add_argument(
        "--svo-prefer",
        choices=("classic", "2025", "any"),
        default="classic",
        help="When multiple SVO rows match a band token, prefer classic vs NIRCam2025 (inr)",
    )
    p.add_argument(
        "--inr-transmission",
        type=Path,
        default=None,
        help="Two-column wavelength, transmission file (inr --inr-spectral-source=file)",
    )
    p.add_argument(
        "--inr-transmission-dir",
        type=Path,
        default=None,
        help=(
            "Directory with {BAND}.txt|dat|csv per filter (inr --per-filter + file source)"
        ),
    )
    p.add_argument(
        "--inr-transmission-unit",
        choices=("angstrom", "nm"),
        default="angstrom",
        help="Wavelength unit in --inr-transmission file columns (inr)",
    )
    p.add_argument(
        "--inr-gaussian-lambda-eff",
        type=float,
        default=None,
        metavar="ANGSTROM",
        help="Effective wavelength (Å) for gaussian spectral source (inr)",
    )
    p.add_argument(
        "--inr-gaussian-lambda-min",
        type=float,
        default=None,
        metavar="ANGSTROM",
        help="Bandpass lower edge (Å) for gaussian spectral source (inr)",
    )
    p.add_argument(
        "--inr-gaussian-lambda-max",
        type=float,
        default=None,
        metavar="ANGSTROM",
        help="Bandpass upper edge (Å) for gaussian spectral source (inr)",
    )
    p.add_argument(
        "--inr-gaussian-sigma",
        type=float,
        default=None,
        metavar="ANGSTROM",
        help="Gaussian transmission sigma (Å); default from lambda_max-lambda_min FWHM (inr)",
    )
    p.add_argument(
        "--inr-quiet-svo",
        action="store_true",
        help="Do not log resolved SVO filterID to stderr (inr)",
    )
    p.add_argument("--inr-steps", type=int, default=80)
    p.add_argument("--inr-lr", type=float, default=5.0e-3)
    p.add_argument(
        "--inr-patch-size",
        type=int,
        default=64,
        help="Training crop size in pixels (inr); 0 = full frame each step",
    )
    p.add_argument("--inr-patches-per-obs", type=int, default=2)
    p.add_argument("--inr-obs-per-step", type=int, default=2)
    p.add_argument("--inr-spectral-samples", type=int, default=9)
    p.add_argument(
        "--inr-render-tile",
        type=int,
        default=512,
        help="spatial_tile_hw for render when map is larger than this (inr)",
    )
    p.add_argument(
        "--inr-max-pixels",
        type=int,
        default=2048 * 2048,
        help="Refuse INR if output grid exceeds this many pixels (memory guard)",
    )
    p.add_argument(
        "--per-filter",
        action="store_true",
        help="Group inputs by FITS FILTER/FILTNAM; one output per band (inr)",
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
        default=1.05,
        help="Padding factor applied to the union footprint bounding box (default: 1.05 = 5%%)",
    )
    p.add_argument(
        "--max-footprint-pixels",
        type=int,
        default=None,
        help="Refuse targets exceeding this many pixels (memory guard, coadd/inr)",
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


def _dispatch_map(args: argparse.Namespace) -> int:
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
    if args.weights is None:
        args.weights = [1.0] * n_inputs
    if len(args.weights) != n_inputs:
        raise SystemExit("--weights count must match expanded input file count")
    args.weights = list(args.weights)
    if any(not math.isfinite(float(w)) or float(w) < 0.0 for w in args.weights):
        raise SystemExit("--weights must be finite and non-negative")
    if args.photometric_scales is None:
        args.photometric_scales = [1.0] * n_inputs
    if len(args.photometric_scales) != n_inputs:
        raise SystemExit(
            "--photometric-scales count must match expanded input file count"
        )
    args.photometric_scales = list(args.photometric_scales)
    if any(
        not math.isfinite(float(s)) or float(s) <= 0.0
        for s in args.photometric_scales
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
    if args.method == "inr" and any(
        not math.isclose(float(s), 1.0, rel_tol=0.0, abs_tol=0.0)
        for s in args.photometric_scales
    ):
        raise SystemExit("--photometric-scales are currently supported for coadd only")
    if args.method == "inr" and any(p is not None for p in args.primary_beam_maps):
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
    tile_pixels = getattr(args, "tile_pixels", None)
    if tile_pixels is not None and int(tile_pixels) <= 0:
        raise SystemExit("--tile-pixels must be positive")
    tile_above_megapixels = float(getattr(args, "tile_above_megapixels", 32.0))
    if not math.isfinite(tile_above_megapixels) or tile_above_megapixels <= 0.0:
        raise SystemExit("--tile-above-megapixels must be finite and positive")
    footprint_padding = float(getattr(args, "footprint_padding", 1.05))
    if not math.isfinite(footprint_padding) or footprint_padding < 1.0:
        raise SystemExit("--footprint-padding must be finite and at least 1.0")
    if int(getattr(args, "inr_steps", 0)) < 0:
        raise SystemExit("--inr-steps must be non-negative")
    inr_lr = float(getattr(args, "inr_lr", 1.0))
    if not math.isfinite(inr_lr) or inr_lr <= 0.0:
        raise SystemExit("--inr-lr must be finite and positive")
    if int(getattr(args, "inr_patch_size", 0)) < 0:
        raise SystemExit("--inr-patch-size must be non-negative")
    if int(getattr(args, "inr_patches_per_obs", 1)) <= 0:
        raise SystemExit("--inr-patches-per-obs must be positive")
    if int(getattr(args, "inr_obs_per_step", 1)) <= 0:
        raise SystemExit("--inr-obs-per-step must be positive")
    if int(getattr(args, "inr_spectral_samples", 1)) <= 0:
        raise SystemExit("--inr-spectral-samples must be positive")
    if int(getattr(args, "inr_render_tile", 1)) <= 0:
        raise SystemExit("--inr-render-tile must be positive")
    if int(getattr(args, "inr_max_pixels", 1)) <= 0:
        raise SystemExit("--inr-max-pixels must be positive")
    args.inputs = expanded
    if args.method == "coadd":
        return _run_coadd(args)
    return _run_inr(args)
