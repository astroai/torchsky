#!/usr/bin/env python3
"""Run A1 NeRF with all-valid epoch sampling, 400/400 batches, and optional PSF.

This launcher reuses the completed baseline manifest for exposure selection,
frozen WCS, bandpass quadrature, normalization, likelihood, and optimizer
settings, while replacing only the scene representation with A1 NeRF.

Defaults specific to this launcher are:

* ``--architecture A1_nerf``;
* ``--max-pixels-per-exposure 0`` (load every valid native pixel);
* ``--sampling-mode epoch_without_replacement``;
* ``--batch-size 400 400`` (per-exposure short-/long-wave counts);
* ``--psf-mode stpsf --psf-size 15``.

All regular options accepted by ``smacs0723_wcs_aware_literature_ablation.py``
remain available. Launcher-only PSF and batch options are removed before the
underlying parser runs.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.smacs0723_wcs_aware_literature_ablation as a1  # noqa: E402
from torchsky.map.wcs_aware_calibration import build_psf_calibration  # noqa: E402


def _parse_band_float_map(text: str) -> dict[str, float]:
    output: dict[str, float] = {}
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"expected BAND=value, got {token!r}")
        key, value = token.split("=", 1)
        output[key.strip().upper()] = float(value)
    return output


def _launcher_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--batch-size", type=int, nargs="+", default=[400, 400])
    parser.add_argument(
        "--psf-mode",
        choices=("none", "gaussian", "npy", "stpsf"),
        default="stpsf",
    )
    parser.add_argument("--psf-size", type=int, default=15)
    parser.add_argument("--psf-fwhm-pix", default="")
    parser.add_argument("--default-psf-fwhm-pix", type=float, default=None)
    parser.add_argument("--psf-kernel-dir", type=Path, default=None)
    parser.add_argument("--psf-cache-dir", type=Path, default=None)
    parser.add_argument("--stpsf-extension", default="DET_DIST")
    parser.add_argument("--allow-static-stpsf-fallback", action="store_true")
    parser.add_argument("--psf-offset-chunk", type=int, default=32)
    parser.add_argument("--no-checkpoint-psf", action="store_true")
    return parser


def _has_cli_option(arguments: Sequence[str], name: str) -> bool:
    return any(argument == name or argument.startswith(name + "=") for argument in arguments)


def _cli_option_value(arguments: Sequence[str], name: str) -> str | None:
    for index, argument in enumerate(arguments):
        if argument.startswith(name + "="):
            return argument.split("=", 1)[1]
        if argument == name and index + 1 < len(arguments):
            return arguments[index + 1]
    return None


def _normalized_batch_spec(values: Sequence[int]) -> int | list[int]:
    parsed = [int(value) for value in values]
    if len(parsed) not in {1, 2} or min(parsed) <= 0:
        raise ValueError("--batch-size accepts one total or two positive SW/LW integers")
    return parsed[0] if len(parsed) == 1 else parsed


def _batch_label(batch_spec: int | Sequence[int]) -> str:
    if isinstance(batch_spec, int):
        return f"total{batch_spec}"
    return "_".join(str(int(value)) for value in batch_spec)


def _psf_label(mode: str, size: int) -> str:
    return "no_psf" if mode == "none" else f"{mode}_k{int(size)}"


def main() -> int:
    launcher_args, remaining = _launcher_parser().parse_known_args(sys.argv[1:])
    batch_spec = _normalized_batch_spec(launcher_args.batch_size)

    architecture = _cli_option_value(remaining, "--architecture")
    if architecture is not None and architecture != "A1_nerf":
        raise SystemExit(
            "this launcher is dedicated to A1 NeRF; remove --architecture or use "
            "--architecture A1_nerf"
        )
    if architecture is None:
        remaining.extend(["--architecture", "A1_nerf"])
    if not _has_cli_option(remaining, "--max-pixels-per-exposure"):
        remaining.extend(["--max-pixels-per-exposure", "0"])
    if not _has_cli_option(remaining, "--sampling-mode"):
        remaining.extend(["--sampling-mode", "epoch_without_replacement"])
    sys.argv = [sys.argv[0], *remaining]

    original_load_manifest = a1._load_manifest
    original_artifact_prefix = a1._artifact_prefix
    original_train_model = a1.train_model

    def load_manifest_with_batch_override(path: Path) -> tuple[Path, dict[str, Any]]:
        resolved, payload = original_load_manifest(path)
        payload = copy.deepcopy(payload)
        payload.setdefault("args", {})["batch_size"] = batch_spec
        # The effective size is recomputed after exposure loading. Clearing this
        # avoids comparing the launcher override against the old baseline value.
        payload["effective_total_batch_size"] = None
        return resolved, payload

    def a1_psf_artifact_prefix(
        architecture: str,
        *,
        target_canvas: bool,
        sampling_mode: str,
        max_pixels_override: int | None,
        resolved_max_pixels: int,
    ) -> str:
        base = original_artifact_prefix(
            architecture,
            target_canvas=target_canvas,
            sampling_mode=sampling_mode,
            max_pixels_override=max_pixels_override,
            resolved_max_pixels=resolved_max_pixels,
        )
        return (
            f"{base}_batch{_batch_label(batch_spec)}_"
            f"{_psf_label(launcher_args.psf_mode, launcher_args.psf_size)}"
        )

    def train_model_with_psf(
        model,
        pools,
        wavelength,
        weight,
        *,
        psf,
        **kwargs,
    ):
        del psf
        checkpoint_path = Path(kwargs["checkpoint_path"])
        cache_dir = (
            launcher_args.psf_cache_dir.expanduser().resolve()
            if launcher_args.psf_cache_dir is not None
            else checkpoint_path.parent / "calibration_cache" / "psf"
        )
        kernel_dir = (
            launcher_args.psf_kernel_dir.expanduser().resolve()
            if launcher_args.psf_kernel_dir is not None
            else None
        )
        psf_calibration = build_psf_calibration(
            [pool.calibration_input() for pool in pools],
            wavelength,
            mode=launcher_args.psf_mode,
            kernel_size=int(launcher_args.psf_size),
            gaussian_fwhm_by_band=_parse_band_float_map(
                launcher_args.psf_fwhm_pix
            ),
            default_gaussian_fwhm=launcher_args.default_psf_fwhm_pix,
            kernel_dir=kernel_dir,
            cache_dir=cache_dir,
            stpsf_extension=launcher_args.stpsf_extension,
            allow_static_stpsf_fallback=(
                launcher_args.allow_static_stpsf_fallback
            ),
        )
        psf_record = {
            "mode": launcher_args.psf_mode,
            "kernel_size": int(launcher_args.psf_size),
            "offset_chunk": int(launcher_args.psf_offset_chunk),
            "checkpoint_chunks": not launcher_args.no_checkpoint_psf,
            "cache_dir": str(cache_dir),
            "kernel_dir": None if kernel_dir is None else str(kernel_dir),
            "stpsf_extension": launcher_args.stpsf_extension,
            "allow_static_stpsf_fallback": (
                launcher_args.allow_static_stpsf_fallback
            ),
            "metadata": psf_calibration.metadata,
        }
        psf_metadata_path = checkpoint_path.with_name(
            checkpoint_path.stem.replace("_checkpoint", "")
            + "_psf_calibration.json"
        )
        psf_metadata_path.write_text(
            json.dumps(psf_record, indent=2),
            encoding="utf-8",
        )
        checkpoint_args = dict(kwargs["checkpoint_args"])
        checkpoint_args["launcher"] = {
            "kind": "A1_all_valid_epoch_psf",
            "batch_size": batch_spec,
            "psf": psf_record,
        }
        kwargs["checkpoint_args"] = checkpoint_args
        kwargs["psf_offset_chunk"] = int(launcher_args.psf_offset_chunk)
        kwargs["checkpoint_psf_chunks"] = not launcher_args.no_checkpoint_psf
        print(
            json.dumps(
                {
                    "launcher": "A1_all_valid_epoch_psf",
                    "batch_size": batch_spec,
                    "psf": {
                        key: value
                        for key, value in psf_record.items()
                        if key != "metadata"
                    },
                    "psf_metadata": str(psf_metadata_path),
                },
                indent=2,
            ),
            flush=True,
        )
        return original_train_model(
            model,
            pools,
            wavelength,
            weight,
            psf=psf_calibration.bank,
            **kwargs,
        )

    a1._load_manifest = load_manifest_with_batch_override
    a1._artifact_prefix = a1_psf_artifact_prefix
    a1.train_model = train_model_with_psf
    return a1.main()


if __name__ == "__main__":
    raise SystemExit(main())
