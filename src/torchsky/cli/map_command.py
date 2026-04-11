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

from .. import read, write
from ..device import default_compute_device, module_compute_device
from ..map import (
    FieldObservation,
    ImageObservation,
    ImagePlaneMapMaker,
    estimate_field_domain,
    fit_field_model,
    make_field_model,
    plan_image_plane_target,
    render_field_observation,
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
    if str(args.inr_spectral_source) == "svo" and fid and not getattr(
        args, "inr_quiet_svo", False
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


def _paths_to_field_observations(
    paths: Sequence[str],
    *,
    hdu: int,
    spectral_kw: dict,
    read_device: str,
) -> list[FieldObservation]:
    skw = spectral_kw
    out: list[FieldObservation] = []
    for p in paths:
        img, hdr = read(p, hdu=hdu, return_header=True, device=read_device)
        if img.ndim != 2:
            raise ValueError(f"{p}: expected 2D image")
        out.append(
            FieldObservation(
                image=img,
                header=dict(hdr),
                weight=torch.ones_like(img, dtype=torch.float32),
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
    observations = [
        ImageObservation(path=p, hdu=args.hdu, weight=float(w))
        for p, w in zip(inputs, args.weights)
    ]
    pr = int(args.plan_reference)
    if pr < 0 or pr >= len(observations):
        raise SystemExit(
            f"--plan-reference {pr} out of range for {len(observations)} input(s) (0-based)"
        )
    target_header = None
    if args.reference is not None:
        _, rh = read(
            str(args.reference),
            hdu=args.reference_hdu,
            return_header=True,
            device=read_dev,
        )
        target_header = dict(rh)

    planned_shape: tuple[int, int] | None = None
    if target_header is None:
        plan = plan_image_plane_target(
            observations,
            reference_index=pr,
            device=args.device,
        )
        planned_shape = plan.shape
    else:
        planned_shape = (
            int(target_header["NAXIS2"]),
            int(target_header["NAXIS1"]),
        )

    tile_hw = _compute_spatial_tile_hw(
        planned_shape,
        auto_tile=bool(args.auto_tile),
        tile_pixels=args.tile_pixels,
        tile_above_megapixels=float(args.tile_above_megapixels),
    )
    if tile_hw is not None:
        print(
            f"torchsky map: spatial tiling {tile_hw[0]}×{tile_hw[1]} "
            f"for output {planned_shape[0]}×{planned_shape[1]} pixels",
            file=sys.stderr,
        )

    maker = ImagePlaneMapMaker(
        target_header=target_header,
        resampling_kernel=args.kernel,
        value_semantics=args.value_semantics,
        stacking_method=args.stack,
        drizzle_pixfrac=float(args.drizzle_pixfrac),
        spatial_tile_hw=tile_hw,
        device=args.device,
    )
    result = maker.run(observations, output_path=str(args.output))
    print(
        f"wrote {args.output} shape={tuple(result.data.shape)} "
        f"mean_weight={float(result.weight.mean().item()):.4f} "
        f"method=coadd kernel={args.kernel} stack={args.stack} "
        f"value_semantics={args.value_semantics} tiled={tile_hw is not None}",
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
        groups: dict[str, list[str]] = defaultdict(list)
        for p in inputs:
            _, hdr = read(p, hdu=args.hdu, return_header=True, device=read_dev)
            fk = _header_filter_keyword(dict(hdr)) or "UNKNOWN"
            groups[fk].append(p)
        rc = 0
        out_path = Path(args.output)
        for fk, gpaths in sorted(groups.items()):
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
                else out_path.parent / f"{out_path.stem}_{_safe_name(fk)}{out_path.suffix}"
            )
            sub_out.parent.mkdir(parents=True, exist_ok=True)
            ns = argparse.Namespace(**vars(args))
            ns.output = str(sub_out)
            ns.inputs = gpaths
            ns.per_filter = False
            ns.inr_band = band
            rc = rc or _run_inr_single(ns)
        return rc
    return _run_inr_single(args)


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s) or "band"


def _run_inr_single(args: argparse.Namespace) -> int:
    inputs = _expand_inputs(args.inputs)
    read_dev = _map_read_device_str(args)
    if args.inr_band is None:
        _, h0 = read(inputs[0], hdu=args.hdu, return_header=True, device=read_dev)
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
    fields = _paths_to_field_observations(
        inputs, hdu=args.hdu, spectral_kw=skw, read_device=read_dev
    )
    image_obs = [ImageObservation(path=p, hdu=args.hdu) for p in inputs]
    pr = int(args.plan_reference)
    if pr < 0 or pr >= len(image_obs):
        raise SystemExit(
            f"--plan-reference {pr} out of range for {len(image_obs)} input(s) (0-based)"
        )
    plan = plan_image_plane_target(
        image_obs,
        reference_index=pr,
        device=args.device,
    )
    h, w = int(plan.shape[0]), int(plan.shape[1])
    max_px = int(args.inr_max_pixels)
    if h * w > max_px:
        raise SystemExit(
            f"INR target {h}×{w} exceeds --inr-max-pixels={max_px}; "
            "use a smaller subset, coadd first, or increase the limit (memory)."
        )

    domain = estimate_field_domain(fields, coord_mode="sphere_xyz")
    model = make_field_model(args.inr_model, domain)
    patch = int(args.inr_patch_size)
    fit_field_model(
        fields,
        model,
        steps=int(args.inr_steps),
        lr=float(args.inr_lr),
        training_patch_size=None if patch <= 0 else patch,
        training_patches_per_observation=int(args.inr_patches_per_obs),
        observations_per_step=min(len(fields), int(args.inr_obs_per_step)),
        device=args.device,
    )
    dev = module_compute_device(model)
    target_obs = FieldObservation(
        image=torch.zeros((h, w), dtype=torch.float32, device=dev),
        header=dict(plan.header),
        weight=torch.ones((h, w), dtype=torch.float32, device=dev),
        **skw,
    )
    model.eval()
    render_kw: dict = {"num_spectral_samples": int(args.inr_spectral_samples)}
    th, tw = int(args.inr_render_tile), int(args.inr_render_tile)
    if h > th or w > tw:
        render_kw["spatial_tile_hw"] = (th, tw)
    with torch.no_grad():
        pred = render_field_observation(model, target_obs, **render_kw)
    write(str(args.output), pred, header=dict(plan.header), overwrite=True)
    print(
        f"wrote {args.output} shape={tuple(pred.shape)} method=inr "
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
        help="Per-input scalar weights (coadd only); default 1.0 each",
    )
    p.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="FITS whose WCS+shape define the output grid (coadd)",
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
    p.add_argument("--inr-band", default=None, help="Filter name for bandpass, e.g. F277W (inr)")
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
    if args.weights is None:
        args.weights = [1.0] * len(expanded)
    if len(args.weights) != len(expanded):
        raise SystemExit("--weights count must match expanded input file count")
    args.weights = list(args.weights)
    args.inputs = expanded
    if args.method == "coadd":
        return _run_coadd(args)
    return _run_inr(args)
