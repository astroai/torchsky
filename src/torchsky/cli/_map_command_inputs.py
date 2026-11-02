"""Plan, manifest, input-expansion, and spectral-metadata helpers for ``torchsky map``."""

from __future__ import annotations

import argparse
from importlib import metadata as importlib_metadata
import json
import math
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

import torch

from ..device import default_compute_device
from torchfits.hdu import HDUList, TensorHDU
from ..map.product import (
    BandpassSpec,
)
from ..map.plan import build_map_input_identity
from ..map.sky import CrossExposureSkyCorrector, MeshSkyEstimator, PolynomialSkyModel
from ..spectrophotometry import Passband


# Offline fallback when FITS headers lack spectral metadata (Angstrom).
_FIELD_BAND_TABLE: dict[str, tuple[float, float, float, float]] = {}


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


def _header_filter_keyword(header: dict) -> str | None:
    for key in ("FILTER", "FILTNAM", "FILTER1"):
        if key in header and header[key] not in (None, "", " "):
            v = header[key]
            if isinstance(v, bytes):
                return v.decode("ascii", errors="replace").strip()
            return str(v).strip()
    return None


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
            _, header = _core_mod.read(
                path, hdu=hdu, return_header=True, device=read_device
            )
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

    profiles = {
        band: _core_mod._field_spectral_kwargs(args, band) for band in unique_bands
    }
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


# Late-bound so monkeypatching ``map_command`` attributes still reaches this code.
from . import map_command as _core_mod  # noqa: E402
