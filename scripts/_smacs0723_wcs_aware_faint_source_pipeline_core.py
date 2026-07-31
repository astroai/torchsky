#!/usr/bin/env python3
"""Two-stage WCS-aware INR pipeline optimized for faint-source recovery.

The training and post-training operators are deliberately separated:

Stage 1 (training)
    No-PSF pretraining against native CRF/CAL pixels. This quickly learns the
    broad multi-band scene and exposure backgrounds.

Stage 2 (training)
    PSF-aware fine-tuning with an unbiased exact-core/stochastic-wing estimator.
    For every exposure and wavelength, the smallest centered square enclosing
    ``--psf-exact-core-mass`` is evaluated exactly. Remaining PSF mass is
    estimated by importance sampling from the normalized wing distribution.

Optional final PSF convolution (post-training only)
    Render the intrinsic bandpass-integrated field on each band's UPDC grid,
    then convolve that final field once with a same-band effective PSF. This is
    controlled by ``--final-psf-convolution`` and is not a training stage.

The script owns data loading, model construction, both training stages, final
rendering, PSF post-processing, product bookkeeping, and detector invocation.
It imports reusable TorchSky model/calibration primitives but does not launch or
monkey-patch either of the older SMACS0723 experiment scripts.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Sequence

import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS as AstropyWCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.ndimage import zoom
from scipy.signal import oaconvolve
from torch import Tensor
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for search_path in (ROOT, SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from torchsky.map.wcs_aware_calibration import build_psf_calibration  # noqa: E402
from torchsky.map.wcs_aware_epoch_sampling import install_epoch_sampler  # noqa: E402
from torchsky.map.wcs_aware_inr import (  # noqa: E402
    PSFKernelBank,
    UnionCanvas,
    weighted_robust_loss,
)
from torchsky.map.wcs_aware_literature_inr import (  # noqa: E402
    WCSAwareLiteratureINR,
)
from torchsky.map.wcs_aware_target_canvas import (  # noqa: E402
    filter_sampled_pools_to_canvas,
    load_target_canvas_from_fits,
    sampled_pool_canvas_coverage,
)
import torchsky.map.wcs_aware_training as training_mod  # noqa: E402


BANDS = ("F090W", "F150W", "F200W", "F277W", "F356W", "F444W")
ARCHITECTURES = ("A1_nerf", "A2_siren")
PSF_MODES = ("none", "gaussian", "npy", "stpsf")
SAMPLING_MODES = ("with_replacement", "epoch_without_replacement")
DEFAULT_UPDC_DIR = Path("/arc/projects/ots/jwst_smacs0723/data")
DEFAULT_UPDC_PATTERN = "{band}b_UPDC-v2_iter10_i2d.fits.fz"
DEFAULT_STPSF_DATA_DIR = Path("/arc/projects/ots/jwst_smacs0723/stpsf-data")
NIRCAM_SHORT_MIN_AA = 6000.0
NIRCAM_CHANNEL_SPLIT_AA = 23500.0
NIRCAM_LONG_MAX_AA = 53000.0
NIRCAM_BOUNDARY_EPSILON_AA = 1.0


@dataclass(frozen=True)
class CoreWingPlan:
    """CPU representation of the exact-core/stochastic-wing PSF estimator."""

    offsets_xy: Tensor
    core_indices_by_wavelength: tuple[Tensor, ...]
    core_weights_by_wavelength: tuple[Tensor, ...]
    wing_probabilities: Tensor
    wing_mass: Tensor
    core_count: Tensor
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DeviceCoreWingPlan:
    """Small device-resident tensors reused by every fine-tuning step."""

    offsets_xy: Tensor
    core_indices_by_wavelength: tuple[Tensor, ...]
    core_weights_by_wavelength: tuple[Tensor, ...]
    wing_mass: Tensor


@dataclass
class TrainingStageResult:
    stage: str
    steps: int
    first_global_step: int
    last_global_step: int
    seconds: float
    records: list[dict[str, Any]]
    checkpoint: str


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")


def parse_band_float_map(text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"expected BAND=value, got {token!r}")
        band, value = token.split("=", 1)
        result[band.strip().upper()] = float(value)
    return result


def parse_band_path_map(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for token in values:
        if "=" not in token:
            raise ValueError(f"expected BAND=FITS, got {token!r}")
        band, raw = token.split("=", 1)
        band = band.strip().upper()
        if band not in BANDS:
            raise ValueError(f"unsupported band {band!r}")
        result[band] = Path(raw).expanduser().resolve()
    return result


def normalize_batch(values: Sequence[int]) -> int | list[int]:
    parsed = [int(value) for value in values]
    if len(parsed) not in (1, 2) or min(parsed) <= 0:
        raise ValueError("--batch-size accepts one total or two positive SW/LW values")
    return parsed[0] if len(parsed) == 1 else parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="A1_nerf")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--pretrain-steps", type=int, default=8000)
    parser.add_argument("--finetune-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, nargs="+", default=[400, 400])
    parser.add_argument("--max-pixels-per-exposure", type=int, default=0)
    parser.add_argument(
        "--sampling-mode",
        choices=SAMPLING_MODES,
        default="epoch_without_replacement",
    )
    parser.add_argument("--pretrain-learning-rate", type=float, default=None)
    parser.add_argument("--finetune-learning-rate", type=float, default=None)
    parser.add_argument("--finetune-lr-scale", type=float, default=0.2)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use mixed precision. Omitted means inherit the baseline setting.",
    )

    parser.add_argument("--use-baseline-canvas", action="store_true")
    parser.add_argument("--target-canvas-fits", type=Path, default=None)
    parser.add_argument("--target-canvas-hdu", default=None)
    parser.add_argument("--keep-outside-target-samples", action="store_true")
    parser.add_argument("--render-tile-pixels", type=int, default=65536)

    parser.add_argument("--nerf-num-frequencies", type=int, default=10)
    parser.add_argument("--nerf-hidden-dim", type=int, default=256)
    parser.add_argument("--nerf-depth", type=int, default=8)
    parser.add_argument("--nerf-skip-after-layer", type=int, default=4)
    parser.add_argument("--siren-hidden-dim", type=int, default=256)
    parser.add_argument("--siren-hidden-layers", type=int, default=3)
    parser.add_argument("--siren-first-omega0", type=float, default=30.0)
    parser.add_argument("--siren-hidden-omega0", type=float, default=30.0)

    parser.add_argument("--psf-mode", choices=PSF_MODES, default="stpsf")
    parser.add_argument("--psf-size", type=int, default=15)
    parser.add_argument("--psf-fwhm-pix", default="")
    parser.add_argument("--default-psf-fwhm-pix", type=float, default=None)
    parser.add_argument("--psf-kernel-dir", type=Path, default=None)
    parser.add_argument("--psf-cache-dir", type=Path, default=None)
    parser.add_argument("--stpsf-extension", default="DET_DIST")
    parser.add_argument("--stpsf-data-dir", type=Path, default=DEFAULT_STPSF_DATA_DIR)
    parser.add_argument("--allow-static-stpsf-fallback", action="store_true")
    parser.add_argument("--psf-exact-core-mass", type=float, default=0.90)
    parser.add_argument("--psf-wing-samples", type=int, default=8)
    parser.add_argument("--finetune-offset-chunk", type=int, default=4)
    parser.add_argument(
        "--checkpoint-finetune-chunks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--updc-dir", type=Path, default=DEFAULT_UPDC_DIR)
    parser.add_argument("--updc-pattern", default=DEFAULT_UPDC_PATTERN)
    parser.add_argument(
        "--updc-reference", action="append", default=[], metavar="BAND=FITS"
    )
    parser.add_argument(
        "--updc-zero-is-invalid",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--final-psf-convolution",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After training, convolve each final intrinsic bandpass field once "
            "with a same-band effective PSF. This never changes training."
        ),
    )
    parser.add_argument("--final-psf-minimum-support", type=float, default=0.999)

    parser.add_argument(
        "--run-detector",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--detector-script",
        type=Path,
        default=ROOT / "scripts" / "compare_wcs_aware_source_catalogs.py",
    )
    parser.add_argument("--detector-threshold-sigma", type=float, default=3.0)
    parser.add_argument("--detector-npixels", type=int, default=5)
    parser.add_argument("--detector-connectivity", type=int, choices=(4, 8), default=8)
    parser.add_argument("--detector-background-box-size", type=int, default=60)
    parser.add_argument("--detector-background-filter-size", type=int, default=5)
    parser.add_argument("--detector-sigma-clip", type=float, default=3.0)
    parser.add_argument("--detector-sigma-clip-maxiters", type=int, default=10)
    parser.add_argument(
        "--detector-deblend",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--detector-deblend-nlevels", type=int, default=32)
    parser.add_argument("--detector-deblend-contrast", type=float, default=0.001)
    parser.add_argument("--detector-match-radius-arcsec", type=float, default=0.05)
    parser.add_argument(
        "--detector-match-radius-grid-arcsec",
        default="0.05,0.075,0.10,0.15",
    )
    parser.add_argument(
        "--detector-snr-bins",
        default="3,4,5,6,8,10,20,inf",
    )
    return parser


def load_manifest(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    for key in ("exposures", "intensity_scale", "union_canvas", "bandpass", "args"):
        if key not in payload:
            raise ValueError(f"baseline manifest is missing {key!r}")
    return resolved, payload


def validate_baseline(manifest: dict[str, Any]) -> None:
    baseline_args = manifest.get("args", {})
    alignment_mode = str(baseline_args.get("alignment_mode", ""))
    if alignment_mode not in {"original_wcs", "tweakreg_frozen"}:
        raise ValueError(
            "the faint-source pipeline requires frozen geometry; "
            f"got alignment_mode={alignment_mode!r}"
        )
    baseline_psf_mode = str(baseline_args.get("psf_mode", "none"))
    if baseline_psf_mode != "none" or manifest.get("psf") not in (None, []):
        raise ValueError("the initialization baseline must be the completed no-PSF run")
    loss_name = str(
        baseline_args.get("robust_loss", baseline_args.get("loss", ""))
    )
    if loss_name != "whitened_l2":
        raise ValueError("the pipeline expects the whitened_l2 run2 baseline")


def selected_from_manifest(
    manifest: dict[str, Any],
) -> list[tuple[Path, str, str]]:
    rows = sorted(manifest["exposures"], key=lambda item: int(item["index"]))
    selected: list[tuple[Path, str, str]] = []
    for expected, row in enumerate(rows):
        index = int(row["index"])
        if index != expected:
            raise ValueError(
                "baseline exposure indices must be contiguous from zero; "
                f"got {index} at position {expected}"
            )
        path = Path(row["path"]).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        selected.append((path, str(row["band"]), str(row["detector"])))
    if not selected:
        raise ValueError("baseline manifest contains no exposures")
    return selected


def bandpass_from_manifest(
    manifest: dict[str, Any],
    *,
    num_exposures: int,
) -> tuple[Tensor, Tensor]:
    metadata = manifest["bandpass"]
    if not isinstance(metadata, list) or len(metadata) != num_exposures:
        raise ValueError("baseline bandpass must contain one row per exposure")
    wave_rows: list[list[float]] = []
    weight_rows: list[list[float]] = []
    for index, row in enumerate(metadata):
        try:
            wave = [float(value) for value in row["quadrature_wavelength"]]
            weight = [float(value) for value in row["quadrature_weight"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid bandpass row {index}") from exc
        if not wave or len(wave) != len(weight):
            raise ValueError(f"invalid bandpass lengths in row {index}")
        wave_rows.append(wave)
        weight_rows.append(weight)
    wavelength = torch.tensor(wave_rows, dtype=torch.float32)
    weight = torch.tensor(weight_rows, dtype=torch.float32)
    if wavelength.ndim != 2 or weight.shape != wavelength.shape:
        raise ValueError("bandpass samples must form rectangular [E,L] tensors")
    if not bool(torch.isfinite(wavelength).all()) or bool((wavelength <= 0).any()):
        raise ValueError("bandpass wavelengths must be finite and positive")
    if not bool(torch.isfinite(weight).all()) or bool((weight < 0).any()):
        raise ValueError("bandpass weights must be finite and non-negative")
    weight /= weight.sum(dim=-1, keepdim=True).clamp_min(1.0e-20)
    return wavelength, weight


def canvas_from_manifest(manifest: dict[str, Any]) -> UnionCanvas:
    info = manifest["union_canvas"]
    return UnionCanvas(
        header=dict(info["header"]),
        shape=tuple(int(value) for value in info["shape"]),
        pixel_scale_arcsec=float(info["pixel_scale_arcsec"]),
    )


def sanitize_nircam_channel_quadrature(
    manifest: dict[str, Any],
    wavelength: Tensor,
    weight: Tensor,
) -> tuple[Tensor, Tensor, list[dict[str, Any]]]:
    """Zero tiny SVO tails outside the selected NIRCam detector channel."""

    exposures = sorted(manifest["exposures"], key=lambda row: int(row["index"]))
    if len(exposures) != wavelength.shape[0]:
        raise ValueError("exposure metadata and bandpass rows have different lengths")
    safe_wave = wavelength.detach().clone()
    safe_weight = weight.detach().clone()
    records: list[dict[str, Any]] = []

    for index, exposure in enumerate(exposures):
        detector = str(exposure.get("detector", "")).strip().upper().replace(" ", "")
        is_long = detector.endswith("5") or detector.endswith("LONG")
        if is_long:
            lower = NIRCAM_CHANNEL_SPLIT_AA + NIRCAM_BOUNDARY_EPSILON_AA
            upper = NIRCAM_LONG_MAX_AA - NIRCAM_BOUNDARY_EPSILON_AA
            channel = "long"
        else:
            lower = NIRCAM_SHORT_MIN_AA + NIRCAM_BOUNDARY_EPSILON_AA
            upper = NIRCAM_CHANNEL_SPLIT_AA - NIRCAM_BOUNDARY_EPSILON_AA
            channel = "short"
        row_wave = safe_wave[index]
        row_weight = safe_weight[index]
        invalid = (row_wave < lower) | (row_wave > upper)
        if not bool(invalid.any()):
            continue
        removed_weight = float(row_weight[invalid].sum())
        row_weight[invalid] = 0.0
        row_wave.clamp_(min=lower, max=upper)
        remaining = row_weight.sum()
        if not bool(torch.isfinite(remaining)) or float(remaining) <= 0.0:
            raise ValueError(
                f"all bandpass weight lies outside the NIRCam {channel} channel "
                f"for exposure {index}"
            )
        row_weight /= remaining
        records.append(
            {
                "exposure_index": index,
                "band": str(exposure.get("band", "")),
                "detector": detector,
                "channel": channel,
                "invalid_sample_count": int(invalid.sum()),
                "removed_pre_normalization_weight": removed_weight,
                "requested_wavelength_angstrom": wavelength[index].tolist(),
                "sanitized_wavelength_angstrom": row_wave.tolist(),
                "sanitized_weight": row_weight.tolist(),
            }
        )
    return safe_wave, safe_weight, records


def resolve_references(args: argparse.Namespace) -> dict[str, Path]:
    overrides = parse_band_path_map(args.updc_reference)
    root = args.updc_dir.expanduser().resolve()
    references = {
        band: overrides.get(band, root / args.updc_pattern.format(band=band)).resolve()
        for band in BANDS
    }
    missing = [str(path) for path in references.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing UPDC references: " + ", ".join(missing))
    return references


def prepare_output_directory(output_dir: Path, *, overwrite: bool) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    managed = {
        "checkpoints": output_dir / "checkpoints",
        "logs": output_dir / "logs",
        "products": output_dir / "products",
        "detector": output_dir / "detector",
    }
    sentinel_files = [
        output_dir / "pipeline_manifest.json",
        output_dir / "pipeline_summary.json",
        output_dir / "pipeline_failure.json",
    ]
    collisions = [path for path in [*managed.values(), *sentinel_files] if path.exists()]
    if collisions and not overwrite:
        shown = ", ".join(path.name for path in collisions)
        raise FileExistsError(
            f"refusing to overwrite existing pipeline artifacts: {shown}. "
            "Use a fresh output directory or pass --overwrite."
        )
    if overwrite:
        for path in managed.values():
            if path.exists():
                shutil.rmtree(path)
        for path in sentinel_files:
            path.unlink(missing_ok=True)
    for path in managed.values():
        path.mkdir(parents=True, exist_ok=True)
    return managed


def build_core_wing_plan(psf: PSFKernelBank, *, exact_core_mass: float) -> CoreWingPlan:
    """Construct an exact centered-core/stochastic-wing decomposition.

    For every exposure/wavelength kernel, the core is the smallest centered odd
    square whose enclosed PSF mass reaches ``exact_core_mass``. The remaining
    mass is sampled from its normalized wing distribution:

        K * S = sum_{r in C} K_r S_r + alpha E_{r~q}[S_r],

    where ``alpha`` is the wing mass and ``q_r = K_r / alpha`` outside the core.
    This makes the *observation-operator estimate* unbiased. Its derivative
    before applying the nonlinear loss is also unbiased; the stochastic gradient
    of a nonlinear loss is not claimed to be exactly unbiased.
    """

    if not 0.0 <= exact_core_mass <= 1.0:
        raise ValueError("exact_core_mass must lie in [0, 1]")
    normalized = psf.normalized()
    kernel = normalized.kernel.detach().cpu().float()
    exposures, wavelengths, height, width = kernel.shape
    flat = kernel.reshape(exposures, wavelengths, height * width)
    radius_y = height // 2
    radius_x = width // 2
    y, x = torch.meshgrid(
        torch.arange(height),
        torch.arange(width),
        indexing="ij",
    )
    offsets = torch.stack((x - radius_x, y - radius_y), dim=-1).reshape(-1, 2).float()
    chebyshev_radius = offsets.abs().amax(dim=1).long()
    maximum_radius = int(chebyshev_radius.max())

    wing_probabilities = torch.zeros_like(flat)
    wing_mass = torch.zeros((exposures, wavelengths), dtype=torch.float32)
    core_count = torch.zeros((exposures, wavelengths), dtype=torch.long)
    core_radius = torch.full((exposures, wavelengths), -1, dtype=torch.long)
    core_index_rows: list[Tensor] = []
    core_weight_rows: list[Tensor] = []
    core_mass_actual = torch.zeros((exposures, wavelengths), dtype=torch.float32)

    for sample_index in range(wavelengths):
        selected_indices: list[Tensor] = []
        selected_weights: list[Tensor] = []
        for exposure_index in range(exposures):
            row = flat[exposure_index, sample_index]
            if not bool((row > 0).any()):
                raise ValueError("PSF kernel contains no positive entries")

            if exact_core_mass <= 0.0:
                chosen_radius = -1
                indices = torch.empty(0, dtype=torch.long)
            else:
                chosen_radius = maximum_radius
                for radius in range(maximum_radius + 1):
                    candidate = torch.nonzero(
                        chebyshev_radius <= radius,
                        as_tuple=False,
                    ).flatten()
                    enclosed = float(row[candidate].sum())
                    if enclosed >= exact_core_mass or radius == maximum_radius:
                        chosen_radius = radius
                        break
                indices = torch.nonzero(
                    chebyshev_radius <= chosen_radius,
                    as_tuple=False,
                ).flatten()

            weights = row[indices]
            selected_indices.append(indices)
            selected_weights.append(weights)
            core_count[exposure_index, sample_index] = int(indices.numel())
            core_radius[exposure_index, sample_index] = chosen_radius
            core_mass_actual[exposure_index, sample_index] = weights.sum()

            wing = row.clone()
            if indices.numel():
                wing[indices] = 0.0
            alpha = wing.sum()
            wing_mass[exposure_index, sample_index] = alpha
            if float(alpha) > 0.0:
                wing_probabilities[exposure_index, sample_index] = wing / alpha
            else:
                wing_probabilities[exposure_index, sample_index, 0] = 1.0

        max_count = max(int(indices.numel()) for indices in selected_indices)
        padded_indices = torch.zeros((exposures, max_count), dtype=torch.long)
        padded_weights = torch.zeros((exposures, max_count), dtype=torch.float32)
        for exposure_index, (indices, weights) in enumerate(
            zip(selected_indices, selected_weights, strict=True)
        ):
            count = int(indices.numel())
            if count:
                padded_indices[exposure_index, :count] = indices
                padded_weights[exposure_index, :count] = weights
        core_index_rows.append(padded_indices)
        core_weight_rows.append(padded_weights)

    reconstructed = torch.zeros_like(flat)
    for sample_index, (indices, weights) in enumerate(
        zip(core_index_rows, core_weight_rows, strict=True)
    ):
        reconstructed[:, sample_index].scatter_add_(1, indices, weights)
    reconstructed += wing_mass[..., None] * wing_probabilities
    torch.testing.assert_close(reconstructed, flat, rtol=1.0e-5, atol=1.0e-7)

    counts_float = core_count.float()
    nonnegative_radius = core_radius.clamp_min(0)
    padded_counts = [int(indices.shape[1]) for indices in core_index_rows]
    metadata = {
        "estimator": "exact_centered_square_core_plus_importance_sampled_wings",
        "exact_core_mass_requested": exact_core_mass,
        "exact_core_mass_actual_min": float(core_mass_actual.min()),
        "exact_core_mass_actual_median": float(core_mass_actual.median()),
        "exact_core_mass_actual_max": float(core_mass_actual.max()),
        "core_offset_count_min": int(core_count.min()),
        "core_offset_count_mean": float(counts_float.mean()),
        "core_offset_count_median": float(counts_float.median()),
        "core_offset_count_max": int(core_count.max()),
        "core_radius_pixels_min": int(nonnegative_radius.min()),
        "core_radius_pixels_median": float(nonnegative_radius.float().median()),
        "core_radius_pixels_max": int(nonnegative_radius.max()),
        "padded_core_offsets_by_wavelength": padded_counts,
        "wing_mass_min": float(wing_mass.min()),
        "wing_mass_median": float(wing_mass.median()),
        "wing_mass_max": float(wing_mass.max()),
        "kernel_size": int(width),
        "total_offsets": int(width * height),
        "sampling_scope": (
            "one independent wing sample set per exposure and wavelength per "
            "optimizer step; shared by native pixels from that exposure"
        ),
        "observation_estimator_unbiased_identity_verified": True,
        "nonlinear_loss_gradient_unbiased_claimed": False,
    }
    return CoreWingPlan(
        offsets_xy=offsets,
        core_indices_by_wavelength=tuple(core_index_rows),
        core_weights_by_wavelength=tuple(core_weight_rows),
        wing_probabilities=wing_probabilities,
        wing_mass=wing_mass,
        core_count=core_count,
        metadata=metadata,
    )


def device_core_wing_plan(plan: CoreWingPlan, device: torch.device) -> DeviceCoreWingPlan:
    return DeviceCoreWingPlan(
        offsets_xy=plan.offsets_xy.to(device=device),
        core_indices_by_wavelength=tuple(
            tensor.to(device=device) for tensor in plan.core_indices_by_wavelength
        ),
        core_weights_by_wavelength=tuple(
            tensor.to(device=device) for tensor in plan.core_weights_by_wavelength
        ),
        wing_mass=plan.wing_mass.to(device=device),
    )


def sample_wing_offsets(
    plan: CoreWingPlan,
    *,
    samples: int,
    generator: torch.Generator,
) -> Tensor:
    exposures, wavelengths, offsets = plan.wing_probabilities.shape
    if samples == 0:
        return torch.empty((exposures, wavelengths, 0), dtype=torch.long)
    probabilities = plan.wing_probabilities.reshape(exposures * wavelengths, offsets)
    sampled = torch.multinomial(
        probabilities,
        samples,
        replacement=True,
        generator=generator,
    )
    return sampled.reshape(exposures, wavelengths, samples)


def predict_core_wing_observation(
    model: WCSAwareLiteratureINR,
    uv: Tensor,
    exposure_id: Tensor,
    bandpass_wavelength: Tensor,
    bandpass_weight: Tensor,
    *,
    plan: CoreWingPlan,
    device_plan: DeviceCoreWingPlan,
    wing_samples: int,
    wing_generator: torch.Generator,
    offset_chunk: int,
    checkpoint_chunks: bool,
) -> tuple[Tensor, Tensor]:
    """Render a native batch with the unbiased core/wing PSF estimator."""

    wavelengths, weights = model._validate_bandpass_tensors(
        bandpass_wavelength,
        bandpass_weight,
        num_exposures=model.num_exposures,
    )
    wavelengths = wavelengths.to(device=uv.device)[exposure_id]
    weights = weights.to(device=uv.device)[exposure_id]
    sampled_wings = sample_wing_offsets(
        plan,
        samples=wing_samples,
        generator=wing_generator,
    ).to(device=uv.device)
    _, _, center_valid = model.geometry.native_to_union(uv, exposure_id)
    prediction = torch.zeros(uv.shape[0], device=uv.device, dtype=uv.dtype)
    batch_count = int(uv.shape[0])

    for sample_index in range(wavelengths.shape[1]):
        per_exposure_indices = device_plan.core_indices_by_wavelength[sample_index]
        per_exposure_weights = device_plan.core_weights_by_wavelength[sample_index]
        if wing_samples:
            wing_indices = sampled_wings[:, sample_index]
            wing_weights = (
                device_plan.wing_mass[:, sample_index : sample_index + 1]
                / float(wing_samples)
            ).expand(-1, wing_samples)
            per_exposure_indices = torch.cat(
                (per_exposure_indices, wing_indices), dim=1
            )
            per_exposure_weights = torch.cat(
                (per_exposure_weights, wing_weights), dim=1
            )

        selected_indices = per_exposure_indices[exposure_id]
        selected_weights = per_exposure_weights[exposure_id]
        wavelength_value = wavelengths[:, sample_index]
        convolved = torch.zeros(batch_count, device=uv.device, dtype=uv.dtype)
        total_offsets = int(selected_indices.shape[1])

        for start in range(0, total_offsets, offset_chunk):
            stop = min(start + offset_chunk, total_offsets)
            index_chunk = selected_indices[:, start:stop]
            weight_chunk = selected_weights[:, start:stop]
            chunk_count = int(index_chunk.shape[1])
            offsets = device_plan.offsets_xy[index_chunk]
            query_uv = uv[:, None, :] + offsets.to(dtype=uv.dtype)
            query_uv_flat = query_uv.reshape(-1, 2)
            query_exposure = (
                exposure_id[:, None].expand(-1, chunk_count).reshape(-1)
            )
            query_wave = (
                wavelength_value[:, None].expand(-1, chunk_count).reshape(-1)
            )

            def evaluate_chunk(
                query: Tensor,
                query_exposure_id: Tensor,
                wave: Tensor,
                local_weight: Tensor,
                fixed_batch_count: int = batch_count,
                fixed_chunk_count: int = chunk_count,
            ) -> Tensor:
                latent, valid = model.spatial_latent(query, query_exposure_id)
                scene = model.decode(latent, wave, valid).reshape(
                    fixed_batch_count, fixed_chunk_count
                )
                return (scene * local_weight).sum(dim=-1)

            if checkpoint_chunks and model.training and torch.is_grad_enabled():
                chunk_value = checkpoint(
                    evaluate_chunk,
                    query_uv_flat,
                    query_exposure,
                    query_wave,
                    weight_chunk,
                    use_reentrant=False,
                )
            else:
                chunk_value = evaluate_chunk(
                    query_uv_flat,
                    query_exposure,
                    query_wave,
                    weight_chunk,
                )
            convolved += chunk_value
        prediction += weights[:, sample_index] * convolved

    return prediction + model.background[exposure_id], center_valid


def parameter_groups(
    model: WCSAwareLiteratureINR,
    *,
    learning_rate: float,
    alignment_learning_rate: float,
    background_learning_rate: float,
) -> list[dict[str, Any]]:
    alignment = list(model.feature_alignment.parameters())
    background = [model.background] if model.background.requires_grad else []
    excluded = {id(parameter) for parameter in [*alignment, *background]}
    main = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in excluded
    ]
    groups: list[dict[str, Any]] = [
        {"params": main, "lr": learning_rate, "name": "scene"}
    ]
    if alignment:
        groups.append(
            {
                "params": alignment,
                "lr": alignment_learning_rate,
                "name": "alignment",
            }
        )
    if background:
        groups.append(
            {
                "params": background,
                "lr": background_learning_rate,
                "name": "background",
            }
        )
    return groups


def save_pipeline_checkpoint(
    path: Path,
    *,
    model: WCSAwareLiteratureINR,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    stage: str,
    stage_step: int,
    global_step: int,
    intensity_scale: float,
    canvas: UnionCanvas,
    pools: Sequence[Any],
    wavelength: Tensor,
    weight: Tensor,
    configuration: dict[str, Any],
    records: Sequence[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "torchsky-smacs0723-faint-source-pipeline-v1",
            "saved_utc": now_utc(),
            "stage": stage,
            "stage_step": stage_step,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "intensity_scale": intensity_scale,
            "union_header": canvas.header,
            "union_shape": canvas.shape,
            "exposure_paths": [str(pool.path) for pool in pools],
            "exposure_bands": [pool.band for pool in pools],
            "bandpass_wavelength": wavelength,
            "bandpass_weight": weight,
            "configuration": configuration,
            "training_records": list(records),
        },
        path,
    )


def run_training_stage(
    *,
    stage: str,
    model: WCSAwareLiteratureINR,
    pools: Sequence[Any],
    wavelength: Tensor,
    weight: Tensor,
    device: torch.device,
    steps: int,
    global_step_start: int,
    batch_size: int | Sequence[int],
    data_generator: torch.Generator,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    amp_enabled: bool,
    gradient_clip: float,
    robust_loss: str,
    robust_delta: float,
    max_relative_weight: float,
    alignment_regularization: float,
    background_regularization: float,
    log_every: int,
    checkpoint_every: int,
    checkpoint_path: Path,
    intensity_scale: float,
    canvas: UnionCanvas,
    checkpoint_configuration: dict[str, Any],
    loss_handle: Any,
    core_wing_plan: CoreWingPlan | None = None,
    device_core_wing: DeviceCoreWingPlan | None = None,
    wing_samples: int = 0,
    wing_generator: torch.Generator | None = None,
    offset_chunk: int = 1,
    checkpoint_chunks: bool = True,
) -> TrainingStageResult:
    if steps == 0:
        return TrainingStageResult(
            stage=stage,
            steps=0,
            first_global_step=global_step_start,
            last_global_step=global_step_start,
            seconds=0.0,
            records=[],
            checkpoint=str(checkpoint_path),
        )
    if stage == "finetune":
        if core_wing_plan is None or device_core_wing is None or wing_generator is None:
            raise ValueError("fine-tuning requires a core/wing plan and RNG")

    model.train()
    model.to(device)
    wavelengths_device = wavelength.to(device=device)
    weights_device = weight.to(device=device)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()

    for stage_step in range(1, steps + 1):
        global_step = global_step_start + stage_step
        uv, exposure_id, target, err, mask = training_mod.sample_balanced_batch(
            pools,
            batch_size=batch_size,
            generator=data_generator,
        )
        uv = uv.to(device)
        exposure_id = exposure_id.to(device)
        target = target.to(device)
        err = err.to(device)
        mask = mask.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            if stage == "pretrain":
                prediction, geometry_valid = model.predict_observation(
                    uv,
                    exposure_id,
                    wavelengths_device,
                    weights_device,
                    psf=None,
                )
            elif stage == "finetune":
                prediction, geometry_valid = predict_core_wing_observation(
                    model,
                    uv,
                    exposure_id,
                    wavelengths_device,
                    weights_device,
                    plan=core_wing_plan,
                    device_plan=device_core_wing,
                    wing_samples=wing_samples,
                    wing_generator=wing_generator,
                    offset_chunk=offset_chunk,
                    checkpoint_chunks=checkpoint_chunks,
                )
            else:
                raise ValueError(f"unsupported training stage {stage!r}")

            data_loss, diagnostics = weighted_robust_loss(
                prediction,
                target,
                err,
                mask & geometry_valid,
                kind=robust_loss,
                delta=robust_delta,
                max_relative_weight=max_relative_weight,
            )
            alignment_reg = model.alignment_regularization()
            background_reg = model.background.square().mean()
            loss = (
                data_loss
                + alignment_regularization * alignment_reg
                + background_regularization * background_reg
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        step_loss = float(loss.detach())
        loss_row = {
            "stage": stage,
            "stage_step": stage_step,
            "global_step": global_step,
            "loss": step_loss,
        }
        loss_handle.write(json.dumps(loss_row) + "\n")
        if stage_step % min(log_every, 100) == 0 or stage_step == steps:
            loss_handle.flush()

        if stage_step == 1 or stage_step % log_every == 0 or stage_step == steps:
            record = {
                **loss_row,
                "data_loss": float(data_loss.detach()),
                "rmse": float(diagnostics["rmse"]),
                "mae": float(diagnostics["mae"]),
                "whitened_rmse": float(
                    diagnostics.get("whitened_rmse", torch.tensor(float("nan")))
                ),
                "whitened_mae": float(
                    diagnostics.get("whitened_mae", torch.tensor(float("nan")))
                ),
                "valid_fraction": float(diagnostics["valid_fraction"]),
                "alignment_regularization": float(alignment_reg.detach()),
                "background_regularization": float(background_reg.detach()),
                "seconds": time.perf_counter() - started,
                "learning_rates": [
                    float(group["lr"]) for group in optimizer.param_groups
                ],
            }
            records.append(record)
            print(json.dumps(record), flush=True)

        if checkpoint_every > 0 and stage_step % checkpoint_every == 0:
            save_pipeline_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                stage=stage,
                stage_step=stage_step,
                global_step=global_step,
                intensity_scale=intensity_scale,
                canvas=canvas,
                pools=pools,
                wavelength=wavelength,
                weight=weight,
                configuration=checkpoint_configuration,
                records=records,
            )

    final_global_step = global_step_start + steps
    save_pipeline_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        stage=stage,
        stage_step=steps,
        global_step=final_global_step,
        intensity_scale=intensity_scale,
        canvas=canvas,
        pools=pools,
        wavelength=wavelength,
        weight=weight,
        configuration=checkpoint_configuration,
        records=records,
    )
    return TrainingStageResult(
        stage=stage,
        steps=steps,
        first_global_step=global_step_start + 1,
        last_global_step=final_global_step,
        seconds=time.perf_counter() - started,
        records=records,
        checkpoint=str(checkpoint_path),
    )


def read_reference(
    path: Path,
    *,
    zero_is_invalid: bool,
) -> tuple[np.ndarray, fits.Header, np.ndarray, dict[str, Any]]:
    with fits.open(path, memmap=False) as hdul:
        image_hdu = None
        for hdu in hdul:
            if hdu.data is None or np.ndim(hdu.data) != 2:
                continue
            if str(hdu.header.get("EXTNAME", "")).upper() == "SCI":
                image_hdu = hdu
                break
            if image_hdu is None:
                image_hdu = hdu
        if image_hdu is None:
            raise ValueError(f"no 2D image HDU in {path}")
        data = np.asarray(image_hdu.data, dtype=np.float32)
        header = image_hdu.header.copy()
        valid = np.isfinite(data)
        coverage_name = None
        for name in ("WHT", "WEIGHT", "COVERAGE", "NEXP"):
            if name in hdul and hdul[name].data is not None:
                coverage = np.asarray(hdul[name].data)
                if coverage.shape == data.shape:
                    valid &= np.isfinite(coverage) & (coverage > 0)
                    coverage_name = name
                    break
        if "UPDCMASK" in hdul and hdul["UPDCMASK"].data is not None:
            extra = np.asarray(hdul["UPDCMASK"].data)
            if extra.shape == data.shape:
                valid &= np.isfinite(extra) & (extra == 0)
        if zero_is_invalid:
            valid &= data != 0.0
    return data, header, valid, {
        "path": str(path),
        "shape": list(data.shape),
        "coverage_extension": coverage_name,
        "zero_is_invalid": zero_is_invalid,
        "valid_fraction": float(valid.mean()),
    }


def bandpass_for_band(
    pools: Sequence[Any],
    wavelength: Tensor,
    weight: Tensor,
    band: str,
) -> tuple[int, Tensor, Tensor, dict[str, Any]]:
    indices = [index for index, pool in enumerate(pools) if pool.band == band]
    if not indices:
        raise ValueError(f"no exposure for {band}")
    representative = indices[0]
    wave = wavelength[representative].detach().cpu().float()
    band_weight = weight[representative].detach().cpu().float()
    band_weight /= band_weight.sum().clamp_min(1.0e-20)
    max_wave_difference = max(
        float((wavelength[index].cpu() - wave).abs().max()) for index in indices
    )
    max_weight_difference = max(
        float((weight[index].cpu() - band_weight).abs().max()) for index in indices
    )
    return representative, wave, band_weight, {
        "representative_exposure": representative,
        "exposure_indices": indices,
        "quadrature_wavelength": wave.tolist(),
        "quadrature_weight": band_weight.tolist(),
        "max_same_band_wavelength_difference": max_wave_difference,
        "max_same_band_weight_difference": max_weight_difference,
    }


def render_intrinsic_bandpass_field(
    *,
    model: WCSAwareLiteratureINR,
    canvas: UnionCanvas,
    template_header: fits.Header,
    representative: int,
    wavelength: Tensor,
    weight: Tensor,
    intensity_scale: float,
    device: torch.device,
    tile_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    height = int(template_header["NAXIS2"])
    width = int(template_header["NAXIS1"])
    template_wcs = AstropyWCS(template_header).celestial
    union_wcs = AstropyWCS(canvas.header).celestial
    field = np.full((height, width), np.nan, dtype=np.float32)
    valid_output = np.zeros((height, width), dtype=bool)
    field_flat = field.reshape(-1)
    valid_flat = valid_output.reshape(-1)
    wave_values = wavelength.numpy()
    weight_values = weight.numpy()

    model.to(device).eval()
    with torch.inference_mode():
        for start in range(0, height * width, tile_pixels):
            stop = min(start + tile_pixels, height * width)
            linear = np.arange(start, stop, dtype=np.int64)
            y = (linear // width).astype(np.float64)
            x = (linear % width).astype(np.float64)
            ra, dec = template_wcs.pixel_to_world_values(x, y)
            union_x, union_y = union_wcs.world_to_pixel_values(ra, dec)
            union_x = np.asarray(union_x)
            union_y = np.asarray(union_y)
            valid = (
                np.isfinite(union_x)
                & np.isfinite(union_y)
                & (union_x >= 0)
                & (union_x <= canvas.shape[1] - 1)
                & (union_y >= 0)
                & (union_y <= canvas.shape[0] - 1)
            )
            tile = np.full(stop - start, np.nan, dtype=np.float32)
            if valid.any():
                xy = torch.from_numpy(
                    np.stack((union_x[valid], union_y[valid]), axis=-1).astype(np.float32)
                ).to(device)
                exposure_id = torch.full(
                    (xy.shape[0],),
                    representative,
                    dtype=torch.long,
                    device=device,
                )
                raw = model.position_encoder(xy)
                latent = model.trunk(model.feature_alignment(raw, exposure_id))
                inside = torch.ones(xy.shape[0], dtype=torch.bool, device=device)
                prediction = torch.zeros(xy.shape[0], dtype=xy.dtype, device=device)
                for wave_value, weight_value in zip(
                    wave_values, weight_values, strict=True
                ):
                    if float(weight_value) == 0.0:
                        continue
                    wave_tensor = torch.full_like(prediction, float(wave_value))
                    prediction += float(weight_value) * model.decode(
                        latent, wave_tensor, inside
                    )
                tile[valid] = (
                    prediction.detach().float().cpu().numpy() * intensity_scale
                )
            field_flat[start:stop] = tile
            valid_flat[start:stop] = valid
    return field, valid_output


def pixel_scale_arcsec(header: fits.Header | dict[str, Any]) -> float:
    scales = proj_plane_pixel_scales(AstropyWCS(header).celestial)[:2]
    value = float(np.sqrt(np.prod(scales)) * 3600.0)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("could not infer a positive pixel scale")
    return value


def center_fit_square(array: np.ndarray, size: int) -> np.ndarray:
    if size <= 0 or size % 2 != 1:
        raise ValueError("target kernel size must be a positive odd integer")
    source = np.asarray(array, dtype=np.float32)
    output = np.zeros((size, size), dtype=np.float32)
    copy_y = min(source.shape[0], size)
    copy_x = min(source.shape[1], size)
    source_y = max((source.shape[0] - copy_y) // 2, 0)
    source_x = max((source.shape[1] - copy_x) // 2, 0)
    target_y = max((size - copy_y) // 2, 0)
    target_x = max((size - copy_x) // 2, 0)
    output[
        target_y : target_y + copy_y,
        target_x : target_x + copy_x,
    ] = source[
        source_y : source_y + copy_y,
        source_x : source_x + copy_x,
    ]
    return output


def resample_psf_kernel(
    kernel: np.ndarray,
    *,
    native_pixel_scale_arcsec: float,
    target_pixel_scale_arcsec: float,
) -> np.ndarray:
    source = np.asarray(kernel, dtype=np.float32)
    ratio = native_pixel_scale_arcsec / target_pixel_scale_arcsec
    half_size = max(1, int(round(((source.shape[0] - 1) / 2.0) * ratio)))
    target_size = 2 * half_size + 1
    resized = zoom(
        source,
        zoom=target_size / source.shape[0],
        order=1,
        prefilter=False,
    )
    resized = center_fit_square(np.asarray(resized, dtype=np.float32), target_size)
    resized = np.clip(resized, 0.0, None)
    total = float(resized.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("resampled PSF has non-positive mass")
    return resized / total


def effective_post_bandpass_psf(
    *,
    pools: Sequence[Any],
    psf_bank: PSFKernelBank,
    wavelength_weight: Tensor,
    band: str,
    target_header: fits.Header,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build one final-grid PSF for convolution after bandpass integration."""

    exposure_indices = [
        index for index, pool in enumerate(pools) if str(pool.band).upper() == band
    ]
    if not exposure_indices:
        raise ValueError(f"no exposure for {band}")
    target_scale = pixel_scale_arcsec(target_header)
    bank = psf_bank.normalized().kernel.detach().cpu().float()
    resampled: list[np.ndarray] = []
    native_scales: list[float] = []

    for exposure_index in exposure_indices:
        weights = wavelength_weight[exposure_index].detach().cpu().float()
        weights /= weights.sum().clamp_min(1.0e-20)
        effective_native = (
            bank[exposure_index] * weights[:, None, None]
        ).sum(dim=0).numpy()
        effective_native = np.clip(effective_native, 0.0, None)
        effective_native /= max(
            float(effective_native.sum()), np.finfo(np.float32).tiny
        )
        native_scale = pixel_scale_arcsec(pools[exposure_index].header)
        native_scales.append(native_scale)
        resampled.append(
            resample_psf_kernel(
                effective_native,
                native_pixel_scale_arcsec=native_scale,
                target_pixel_scale_arcsec=target_scale,
            )
        )

    common_size = max(kernel.shape[0] for kernel in resampled)
    if common_size % 2 == 0:
        common_size += 1
    aligned = [center_fit_square(kernel, common_size) for kernel in resampled]
    effective = np.mean(np.stack(aligned), axis=0).astype(np.float32)
    effective = np.clip(effective, 0.0, None)
    effective /= max(float(effective.sum()), np.finfo(np.float32).tiny)
    return effective, {
        "definition": (
            "For each same-band exposure, integrate its wavelength-dependent "
            "detector PSF using that exposure's bandpass weights, resample to "
            "the final UPDC pixel scale, then average exposures."
        ),
        "operator_order": "effective_PSF * [sum_j w_j S_theta(lambda_j)]",
        "band": band,
        "exposure_indices": exposure_indices,
        "exposure_count": len(exposure_indices),
        "native_pixel_scale_arcsec": native_scales,
        "target_pixel_scale_arcsec": target_scale,
        "kernel_size": int(effective.shape[0]),
        "kernel_mass": float(effective.sum()),
    }


def convolve_field_with_psf(
    field: np.ndarray,
    kernel: np.ndarray,
    *,
    minimum_support: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(field)
    numerator = oaconvolve(
        np.where(valid, field, 0.0).astype(np.float32),
        kernel.astype(np.float32),
        mode="same",
    ).astype(np.float32)
    support = oaconvolve(
        valid.astype(np.float32),
        kernel.astype(np.float32),
        mode="same",
    ).astype(np.float32)
    output_valid = support >= minimum_support
    output = np.full(field.shape, np.nan, dtype=np.float32)
    output[output_valid] = numerator[output_valid] / np.maximum(
        support[output_valid], np.finfo(np.float32).tiny
    )
    return output, output_valid


def write_image(
    path: Path,
    data: np.ndarray,
    header: fits.Header,
    *,
    band: str,
    stage: str,
    overwrite: bool,
) -> None:
    output_header = header.copy()
    output_header["BAND"] = band
    output_header["STAGE"] = stage
    fits.PrimaryHDU(np.asarray(data), header=output_header).writeto(
        path, overwrite=overwrite
    )


def run_detector(
    *,
    args: argparse.Namespace,
    band: str,
    model_image: Path,
    reference_image: Path,
    common_mask: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if not args.run_detector:
        return {"enabled": False}
    script = args.detector_script.expanduser().resolve()
    if not script.exists():
        raise FileNotFoundError(script)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_label = args.architecture
    command = [
        sys.executable,
        str(script),
        "--image",
        f"{model_label}={model_image}",
        "--image",
        f"UPDC={reference_image}",
        "--coverage",
        f"{model_label}={common_mask}",
        "--coverage",
        f"UPDC={common_mask}",
        "--reference-label",
        "UPDC",
        "--band",
        band,
        "--output-dir",
        str(output_dir),
        "--threshold-sigma",
        str(args.detector_threshold_sigma),
        "--npixels",
        str(args.detector_npixels),
        "--connectivity",
        str(args.detector_connectivity),
        "--background-box-size",
        str(args.detector_background_box_size),
        "--background-filter-size",
        str(args.detector_background_filter_size),
        "--sigma-clip",
        str(args.detector_sigma_clip),
        "--sigma-clip-maxiters",
        str(args.detector_sigma_clip_maxiters),
        "--match-radius-arcsec",
        str(args.detector_match_radius_arcsec),
        "--match-radius-grid-arcsec",
        str(args.detector_match_radius_grid_arcsec),
        "--snr-bins",
        str(args.detector_snr_bins),
    ]
    if args.detector_deblend:
        command.extend(
            [
                "--deblend",
                "--deblend-nlevels",
                str(args.detector_deblend_nlevels),
                "--deblend-contrast",
                str(args.detector_deblend_contrast),
            ]
        )
    if args.overwrite:
        command.append("--overwrite")
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    stdout_path = output_dir / "detector_stdout.log"
    stderr_path = output_dir / "detector_stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    record = {
        "enabled": True,
        "status": "complete" if completed.returncode == 0 else "failed",
        "returncode": int(completed.returncode),
        "command": command,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "summary": str(output_dir / "catalog_comparison_summary.json"),
        "threshold_sigma": args.detector_threshold_sigma,
        "input_image": str(model_image),
        "reference_image": str(reference_image),
        "common_mask": str(common_mask),
    }
    write_json(output_dir / "detector_invocation.json", record)
    return record


def validate_arguments(args: argparse.Namespace) -> None:
    if args.pretrain_steps < 0 or args.finetune_steps < 0:
        raise ValueError("training step counts must be non-negative")
    if args.pretrain_steps + args.finetune_steps <= 0:
        raise ValueError("at least one training stage must have positive steps")
    if args.max_pixels_per_exposure < 0:
        raise ValueError("max-pixels-per-exposure must be non-negative")
    if args.psf_size <= 0 or args.psf_size % 2 != 1:
        raise ValueError("psf-size must be a positive odd integer")
    if not 0.0 <= args.psf_exact_core_mass <= 1.0:
        raise ValueError("psf-exact-core-mass must lie in [0, 1]")
    if args.psf_wing_samples < 0:
        raise ValueError("psf-wing-samples must be non-negative")
    if (
        args.finetune_steps > 0
        and args.psf_exact_core_mass < 1.0
        and args.psf_wing_samples == 0
    ):
        raise ValueError(
            "positive wing samples are required when fine-tuning leaves wing mass"
        )
    if args.finetune_offset_chunk <= 0:
        raise ValueError("finetune-offset-chunk must be positive")
    if args.render_tile_pixels <= 0:
        raise ValueError("render-tile-pixels must be positive")
    if not 0.0 < args.final_psf_minimum_support <= 1.0:
        raise ValueError("final-psf-minimum-support must lie in (0, 1]")
    if args.finetune_lr_scale <= 0:
        raise ValueError("finetune-lr-scale must be positive")
    if args.log_every <= 0:
        raise ValueError("log-every must be positive")
    if args.checkpoint_every < 0:
        raise ValueError("checkpoint-every must be non-negative")
    if args.finetune_steps > 0 and args.psf_mode == "none":
        raise ValueError("PSF-aware fine-tuning requires a non-none psf-mode")
    if args.final_psf_convolution and args.psf_mode == "none":
        raise ValueError("final PSF convolution requires a non-none psf-mode")
    if args.detector_threshold_sigma <= 0:
        raise ValueError("detector-threshold-sigma must be positive")


def main() -> int:
    args = build_parser().parse_args()
    validate_arguments(args)
    output_dir = args.output_dir.expanduser().resolve()
    paths = prepare_output_directory(output_dir, overwrite=args.overwrite)
    manifest_path = output_dir / "pipeline_manifest.json"
    summary_path = output_dir / "pipeline_summary.json"
    failure_path = output_dir / "pipeline_failure.json"

    try:
        baseline_path, baseline = load_manifest(args.baseline_manifest)
        validate_baseline(baseline)
        references = resolve_references(args)
        baseline_args = baseline["args"]
        seed = int(args.seed if args.seed is not None else baseline_args.get("seed", 17))
        training_mod.seed_everything(seed)
        batch_spec = normalize_batch(args.batch_size)

        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if args.sampling_mode == "epoch_without_replacement":
            install_epoch_sampler()

        if args.psf_mode == "stpsf":
            stpsf_dir = args.stpsf_data_dir.expanduser().resolve()
            if not stpsf_dir.exists():
                raise FileNotFoundError(stpsf_dir)
            os.environ["STPSF_PATH"] = str(stpsf_dir)
            print(json.dumps({"STPSF_PATH": str(stpsf_dir)}, indent=2), flush=True)
        else:
            stpsf_dir = None

        selected = selected_from_manifest(baseline)
        rng = np.random.default_rng(seed)
        dq_mode = str(baseline_args.get("dq_mode", "strict"))
        pools = [
            training_mod.load_exposure_pool(
                path,
                band,
                detector,
                max_pixels=args.max_pixels_per_exposure,
                dq_mode=dq_mode,
                rng=rng,
            )
            for path, band, detector in selected
        ]
        intensity_scale = float(baseline["intensity_scale"])
        if not np.isfinite(intensity_scale) or intensity_scale <= 0:
            raise ValueError("baseline intensity_scale must be finite and positive")
        pools = training_mod.normalize_pools(pools, intensity_scale)

        if args.use_baseline_canvas:
            if args.target_canvas_fits is not None:
                raise ValueError(
                    "--use-baseline-canvas and --target-canvas-fits are mutually exclusive"
                )
            canvas = canvas_from_manifest(baseline)
            canvas_metadata = {
                "mode": "baseline_manifest",
                "source_manifest": str(baseline_path),
                "shape": list(canvas.shape),
                "pixel_scale_arcsec": canvas.pixel_scale_arcsec,
            }
            canvas_coverage = sampled_pool_canvas_coverage(pools, canvas)
            canvas_coverage["filter_applied"] = False
        else:
            target_fits = (
                args.target_canvas_fits.expanduser().resolve()
                if args.target_canvas_fits is not None
                else references["F090W"]
            )
            canvas, canvas_metadata = load_target_canvas_from_fits(
                target_fits,
                hdu_selector=args.target_canvas_hdu,
            )
            if args.keep_outside_target_samples:
                canvas_coverage = sampled_pool_canvas_coverage(pools, canvas)
                canvas_coverage["filter_applied"] = False
            else:
                pools, canvas_coverage = filter_sampled_pools_to_canvas(pools, canvas)

        wavelength, weight = bandpass_from_manifest(
            baseline,
            num_exposures=len(pools),
        )
        wavelength, weight, trim_records = sanitize_nircam_channel_quadrature(
            baseline,
            wavelength,
            weight,
        )
        active_wavelength = wavelength[weight > 0]
        wavelength_min = float(active_wavelength.min())
        wavelength_max = float(active_wavelength.max())

        anchor_exposure = int(baseline_args.get("anchor_exposure", 0))
        alignment_mode = str(baseline_args["alignment_mode"])
        learn_background = not bool(baseline_args.get("no_background", False))
        output_activation = str(baseline_args.get("output_activation", "softplus"))
        model = WCSAwareLiteratureINR(
            [pool.header for pool in pools],
            canvas,
            architecture=args.architecture,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            anchor_exposure=anchor_exposure,
            alignment_mode=alignment_mode,
            learn_background=learn_background,
            output_activation=output_activation,
            nerf_num_frequencies=args.nerf_num_frequencies,
            nerf_hidden_dim=args.nerf_hidden_dim,
            nerf_depth=args.nerf_depth,
            nerf_skip_after_layer=args.nerf_skip_after_layer,
            siren_hidden_dim=args.siren_hidden_dim,
            siren_hidden_layers=args.siren_hidden_layers,
            siren_first_omega_0=args.siren_first_omega0,
            siren_hidden_omega_0=args.siren_hidden_omega0,
        )
        model.to(device)

        pretrain_lr = float(
            args.pretrain_learning_rate
            if args.pretrain_learning_rate is not None
            else baseline_args.get("learning_rate", 2.0e-4)
        )
        alignment_lr = float(baseline_args.get("alignment_learning_rate", 2.0e-5))
        background_lr = float(baseline_args.get("background_learning_rate", 5.0e-5))
        optimizer = torch.optim.AdamW(
            parameter_groups(
                model,
                learning_rate=pretrain_lr,
                alignment_learning_rate=alignment_lr,
                background_learning_rate=background_lr,
            ),
            weight_decay=float(baseline_args.get("weight_decay", 1.0e-6)),
        )
        amp_requested = (
            args.amp
            if args.amp is not None
            else not bool(baseline_args.get("no_amp", False))
        )
        amp_enabled = device.type == "cuda" and bool(amp_requested)
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except TypeError:
            scaler = torch.amp.GradScaler(enabled=amp_enabled)

        effective_batch = training_mod.effective_batch_size(pools, batch_spec)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        configuration = {
            "created_utc": now_utc(),
            "baseline_manifest": str(baseline_path),
            "architecture": args.architecture,
            "architecture_metadata": model.architecture_metadata,
            "parameter_count": parameter_count,
            "seed": seed,
            "device": str(device),
            "sampling": {
                "mode": args.sampling_mode,
                "max_pixels_per_exposure": args.max_pixels_per_exposure,
                "all_valid_pixels": args.max_pixels_per_exposure == 0,
                "batch_spec": batch_spec,
                "effective_total_batch_size": effective_batch,
            },
            "stages": {
                "stage1_pretrain": {
                    "steps": args.pretrain_steps,
                    "operator": "bandpass only; no PSF in training forward",
                    "learning_rate": pretrain_lr,
                },
                "stage2_finetune": {
                    "steps": args.finetune_steps,
                    "operator": (
                        "wavelength-dependent per-exposure PSF with exact "
                        "centered core and sampled wings"
                    ),
                    "exact_core_mass": args.psf_exact_core_mass,
                    "wing_samples": args.psf_wing_samples,
                    "offset_chunk": args.finetune_offset_chunk,
                    "checkpoint_chunks": args.checkpoint_finetune_chunks,
                },
                "post_training_final_psf_convolution": {
                    "enabled": args.final_psf_convolution,
                    "operator": (
                        "effective_PSF * final_intrinsic_bandpass_field"
                    ),
                    "is_training_stage": False,
                },
            },
            "target_canvas": canvas_metadata,
            "target_canvas_sample_coverage": canvas_coverage,
            "bandpass_channel_trim": trim_records,
            "psf": {
                "mode": args.psf_mode,
                "size": args.psf_size,
                "stpsf_data_dir": None if stpsf_dir is None else str(stpsf_dir),
            },
            "detector": {
                "enabled": args.run_detector,
                "threshold_sigma": args.detector_threshold_sigma,
                "npixels": args.detector_npixels,
                "deblend": args.detector_deblend,
            },
            "arguments": vars(args),
            "exposures": [
                {
                    "index": index,
                    "path": str(pool.path),
                    "band": pool.band,
                    "detector": pool.detector,
                    "shape": list(pool.shape),
                    "loaded_pixels": pool.size,
                }
                for index, pool in enumerate(pools)
            ],
            "union_canvas": {
                "shape": list(canvas.shape),
                "pixel_scale_arcsec": canvas.pixel_scale_arcsec,
                "header": canvas.header,
            },
            "intensity_scale": intensity_scale,
            "bandpass_wavelength": wavelength.tolist(),
            "bandpass_weight": weight.tolist(),
        }
        write_json(manifest_path, configuration)

        data_generator = torch.Generator(device="cpu").manual_seed(seed + 101)
        wing_generator = torch.Generator(device="cpu").manual_seed(seed + 707)
        loss_path = paths["logs"] / "training_loss.jsonl"
        stage_results: list[TrainingStageResult] = []
        psf_calibration = None
        core_wing = None
        device_plan = None

        with loss_path.open("w", encoding="utf-8") as loss_handle:
            pretrain_result = run_training_stage(
                stage="pretrain",
                model=model,
                pools=pools,
                wavelength=wavelength,
                weight=weight,
                device=device,
                steps=args.pretrain_steps,
                global_step_start=0,
                batch_size=batch_spec,
                data_generator=data_generator,
                optimizer=optimizer,
                scaler=scaler,
                amp_enabled=amp_enabled,
                gradient_clip=float(baseline_args.get("gradient_clip", 1.0)),
                robust_loss=str(baseline_args.get("robust_loss", "whitened_l2")),
                robust_delta=float(baseline_args.get("robust_delta", 0.1)),
                max_relative_weight=float(
                    baseline_args.get("max_relative_weight", 100.0)
                ),
                alignment_regularization=float(
                    baseline_args.get("alignment_regularization", 1.0e-4)
                ),
                background_regularization=float(
                    baseline_args.get("background_regularization", 1.0e-5)
                ),
                log_every=args.log_every,
                checkpoint_every=args.checkpoint_every,
                checkpoint_path=paths["checkpoints"] / "stage1_pretrain_checkpoint.pt",
                intensity_scale=intensity_scale,
                canvas=canvas,
                checkpoint_configuration=configuration,
                loss_handle=loss_handle,
            )
            stage_results.append(pretrain_result)

            needs_psf = args.finetune_steps > 0 or args.final_psf_convolution
            if needs_psf:
                cache_dir = (
                    args.psf_cache_dir.expanduser().resolve()
                    if args.psf_cache_dir is not None
                    else output_dir / "calibration_cache" / "psf"
                )
                kernel_dir = (
                    args.psf_kernel_dir.expanduser().resolve()
                    if args.psf_kernel_dir is not None
                    else None
                )
                psf_calibration = build_psf_calibration(
                    [pool.calibration_input() for pool in pools],
                    wavelength,
                    mode=args.psf_mode,
                    kernel_size=args.psf_size,
                    gaussian_fwhm_by_band=parse_band_float_map(args.psf_fwhm_pix),
                    default_gaussian_fwhm=args.default_psf_fwhm_pix,
                    kernel_dir=kernel_dir,
                    cache_dir=cache_dir,
                    stpsf_extension=args.stpsf_extension,
                    allow_static_stpsf_fallback=args.allow_static_stpsf_fallback,
                )
                if psf_calibration.bank is None:
                    raise RuntimeError("PSF calibration did not produce a kernel bank")
                configuration["psf"].update(
                    {
                        "cache_dir": str(cache_dir),
                        "kernel_dir": None if kernel_dir is None else str(kernel_dir),
                        "metadata": psf_calibration.metadata,
                    }
                )

            if args.finetune_steps > 0:
                assert psf_calibration is not None
                core_wing = build_core_wing_plan(
                    psf_calibration.bank,
                    exact_core_mass=args.psf_exact_core_mass,
                )
                device_plan = device_core_wing_plan(core_wing, device)
                configuration["stages"]["stage2_finetune"]["plan"] = core_wing.metadata
                padded_core_counts = core_wing.metadata[
                    "padded_core_offsets_by_wavelength"
                ]
                vectorized_offsets = [
                    int(count) + args.psf_wing_samples
                    for count in padded_core_counts
                ]
                configuration["stages"]["stage2_finetune"].update(
                    {
                        "vectorized_offsets_by_wavelength": vectorized_offsets,
                        "estimated_query_reduction_vs_full_psf": float(
                            (args.psf_size * args.psf_size)
                            / float(np.mean(vectorized_offsets))
                        ),
                    }
                )
                write_json(manifest_path, configuration)

                current_main_lr = float(optimizer.param_groups[0]["lr"])
                target_finetune_lr = float(
                    args.finetune_learning_rate
                    if args.finetune_learning_rate is not None
                    else current_main_lr * args.finetune_lr_scale
                )
                learning_rate_ratio = target_finetune_lr / current_main_lr
                for group in optimizer.param_groups:
                    group["lr"] = float(group["lr"]) * learning_rate_ratio
                configuration["stages"]["stage2_finetune"].update(
                    {
                        "learning_rate": target_finetune_lr,
                        "optimizer_state": (
                            "continued from stage 1; Adam moments are preserved"
                        ),
                    }
                )

                finetune_result = run_training_stage(
                    stage="finetune",
                    model=model,
                    pools=pools,
                    wavelength=wavelength,
                    weight=weight,
                    device=device,
                    steps=args.finetune_steps,
                    global_step_start=args.pretrain_steps,
                    batch_size=batch_spec,
                    data_generator=data_generator,
                    optimizer=optimizer,
                    scaler=scaler,
                    amp_enabled=amp_enabled,
                    gradient_clip=float(baseline_args.get("gradient_clip", 1.0)),
                    robust_loss=str(
                        baseline_args.get("robust_loss", "whitened_l2")
                    ),
                    robust_delta=float(baseline_args.get("robust_delta", 0.1)),
                    max_relative_weight=float(
                        baseline_args.get("max_relative_weight", 100.0)
                    ),
                    alignment_regularization=float(
                        baseline_args.get("alignment_regularization", 1.0e-4)
                    ),
                    background_regularization=float(
                        baseline_args.get("background_regularization", 1.0e-5)
                    ),
                    log_every=args.log_every,
                    checkpoint_every=args.checkpoint_every,
                    checkpoint_path=(
                        paths["checkpoints"] / "stage2_finetune_checkpoint.pt"
                    ),
                    intensity_scale=intensity_scale,
                    canvas=canvas,
                    checkpoint_configuration=configuration,
                    loss_handle=loss_handle,
                    core_wing_plan=core_wing,
                    device_core_wing=device_plan,
                    wing_samples=args.psf_wing_samples,
                    wing_generator=wing_generator,
                    offset_chunk=args.finetune_offset_chunk,
                    checkpoint_chunks=args.checkpoint_finetune_chunks,
                )
                stage_results.append(finetune_result)

        write_json(manifest_path, configuration)

        products: dict[str, Any] = {}
        detector_records: dict[str, Any] = {}
        for band in BANDS:
            reference, header, reference_valid, reference_info = read_reference(
                references[band],
                zero_is_invalid=args.updc_zero_is_invalid,
            )
            representative, wave, band_weight, bandpass_info = bandpass_for_band(
                pools,
                wavelength,
                weight,
                band,
            )
            band_dir = paths["products"] / band
            band_dir.mkdir(parents=True, exist_ok=True)
            intrinsic, intrinsic_valid = render_intrinsic_bandpass_field(
                model=model,
                canvas=canvas,
                template_header=header,
                representative=representative,
                wavelength=wave,
                weight=band_weight,
                intensity_scale=intensity_scale,
                device=device,
                tile_pixels=args.render_tile_pixels,
            )
            intrinsic_path = band_dir / f"{band}_intrinsic_bandpass_field.fits"
            write_image(
                intrinsic_path,
                intrinsic,
                header,
                band=band,
                stage="INTRINSIC_BANDPASS_FIELD",
                overwrite=args.overwrite,
            )

            effective_psf_path = None
            if args.final_psf_convolution:
                assert psf_calibration is not None
                effective_psf, effective_psf_info = effective_post_bandpass_psf(
                    pools=pools,
                    psf_bank=psf_calibration.bank,
                    wavelength_weight=weight,
                    band=band,
                    target_header=header,
                )
                final_field, final_valid = convolve_field_with_psf(
                    intrinsic,
                    effective_psf,
                    minimum_support=args.final_psf_minimum_support,
                )
                final_stage = "POST_BANDPASS_EFFECTIVE_PSF_FIELD"
                effective_psf_path = band_dir / f"{band}_effective_post_bandpass_psf.fits"
                fits.PrimaryHDU(effective_psf).writeto(
                    effective_psf_path, overwrite=args.overwrite
                )
            else:
                final_field = intrinsic.copy()
                final_valid = intrinsic_valid.copy()
                final_stage = "INTRINSIC_BANDPASS_FIELD"
                effective_psf_info = None

            final_path = band_dir / f"{band}_final_detector_input.fits"
            write_image(
                final_path,
                final_field,
                header,
                band=band,
                stage=final_stage,
                overwrite=args.overwrite,
            )
            common_valid = (
                final_valid
                & reference_valid
                & np.isfinite(final_field)
                & np.isfinite(reference)
            )
            common_mask_path = band_dir / f"{band}_model_UPDC_common_valid_mask.fits"
            write_image(
                common_mask_path,
                common_valid.astype(np.uint8),
                header,
                band=band,
                stage="MODEL_UPDC_COMMON_VALID_MASK",
                overwrite=args.overwrite,
            )
            detector_record = run_detector(
                args=args,
                band=band,
                model_image=final_path,
                reference_image=references[band],
                common_mask=common_mask_path,
                output_dir=paths["detector"] / band,
            )
            detector_records[band] = detector_record
            products[band] = {
                "reference": reference_info,
                "bandpass": bandpass_info,
                "intrinsic_bandpass_field": str(intrinsic_path),
                "final_detector_input": str(final_path),
                "final_psf_convolution_enabled": args.final_psf_convolution,
                "effective_post_bandpass_psf": (
                    None if effective_psf_path is None else str(effective_psf_path)
                ),
                "effective_post_bandpass_psf_metadata": effective_psf_info,
                "common_valid_mask": str(common_mask_path),
                "common_valid_pixels": int(common_valid.sum()),
                "detector": detector_record,
            }
            write_json(band_dir / f"{band}_product_manifest.json", products[band])

        summary = {
            "status": "complete",
            "completed_utc": now_utc(),
            "baseline_manifest": str(baseline_path),
            "manifest": str(manifest_path),
            "architecture": args.architecture,
            "training_loss": str(loss_path),
            "training_stages": [asdict(result) for result in stage_results],
            "final_psf_convolution": {
                "enabled": args.final_psf_convolution,
                "is_training_stage": False,
                "operator_order": (
                    "effective_PSF * final_intrinsic_bandpass_field"
                    if args.final_psf_convolution
                    else "no post-training PSF convolution"
                ),
            },
            "products": products,
            "detector": {
                "threshold_sigma": args.detector_threshold_sigma,
                "records": detector_records,
            },
            "alignment": model.alignment_summary(),
            "background_physical": (
                model.background.detach().cpu() * intensity_scale
            ).tolist(),
        }
        write_json(summary_path, summary)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "manifest": str(manifest_path),
                    "summary": str(summary_path),
                    "training_loss": str(loss_path),
                    "products": str(paths["products"]),
                    "detector": str(paths["detector"]),
                },
                indent=2,
            ),
            flush=True,
        )
        return 0
    except BaseException as exc:
        write_json(
            failure_path,
            {
                "status": "failed",
                "failed_utc": now_utc(),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "arguments": vars(args),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
