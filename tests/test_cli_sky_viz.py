from __future__ import annotations

from pathlib import Path
import types

import pytest
import torch
import torchfits

from torchsky.cli import build_parser


def _write_image(path: Path, offset: float = 0.0) -> None:
    y, x = torch.meshgrid(
        torch.arange(16, dtype=torch.float32),
        torch.arange(16, dtype=torch.float32),
        indexing="ij",
    )
    torchfits.write(
        str(path),
        offset + 0.01 * x + 0.02 * y,
        header={"NAXIS": 2, "NAXIS1": 16, "NAXIS2": 16},
        overwrite=True,
    )


@pytest.mark.parametrize("method", ["mesh", "polynomial"])
def test_sky_estimate_executes_supported_methods(tmp_path: Path, method: str) -> None:
    source = tmp_path / "input.fits"
    mask = tmp_path / "mask.fits"
    output = tmp_path / f"sky-{method}.fits"
    _write_image(source)
    torchfits.write(
        str(mask),
        torch.zeros((16, 16), dtype=torch.uint8),
        overwrite=True,
    )
    args = build_parser().parse_args(
        [
            "sky",
            "estimate",
            str(source),
            str(output),
            "--mask",
            str(mask),
            "--method",
            method,
            "--statistic",
            "median" if method == "mesh" else "huber",
            "--block-size",
            "4",
            "--mesh-filter",
            "none",
            "--interpolate",
            "bilinear",
            "--order",
            "1",
            "--device",
            "cpu",
        ]
    )
    assert args.handler(args) == 0
    result = torchfits.read(str(output))
    assert result.shape == (16, 16)
    assert torch.isfinite(result).all()


def test_sky_match_executes_with_masks_and_outputs(tmp_path: Path) -> None:
    sources = [tmp_path / f"input-{index}.fits" for index in range(2)]
    masks = [tmp_path / f"mask-{index}.fits" for index in range(2)]
    for index, path in enumerate(sources):
        _write_image(path, offset=float(index))
    for path in masks:
        torchfits.write(
            str(path),
            torch.zeros((16, 16), dtype=torch.uint8),
            overwrite=True,
        )
    pattern = tmp_path / "corrected-{i}.fits"
    args = build_parser().parse_args(
        [
            "sky",
            "match",
            *(str(path) for path in sources),
            "--masks",
            *(str(path) for path in masks),
            "--output",
            str(pattern),
            "--method",
            "mesh",
            "--statistic",
            "huber",
            "--block-size",
            "4",
            "--mesh-filter",
            "gaussian",
            "--mesh-filter-size",
            "3",
            "--interpolate",
            "bicubic",
            "--device",
            "cpu",
        ]
    )
    assert args.handler(args) == 0
    for index in range(2):
        corrected = torchfits.read(str(tmp_path / f"corrected-{index}.fits"))
        assert corrected.shape == (16, 16)
        assert torch.isfinite(corrected).all()


def test_sky_match_rejects_mismatched_masks(tmp_path: Path) -> None:
    sources = [tmp_path / f"input-{index}.fits" for index in range(2)]
    for path in sources:
        _write_image(path)
    mask = tmp_path / "mask.fits"
    torchfits.write(str(mask), torch.zeros((16, 16), dtype=torch.uint8), overwrite=True)
    args = build_parser().parse_args(
        ["sky", "match", *(str(path) for path in sources), "--masks", str(mask)]
    )
    with pytest.raises(SystemExit, match="--masks count"):
        args.handler(args)


def test_polynomial_sky_rejects_mesh_only_median(tmp_path: Path) -> None:
    source = tmp_path / "input.fits"
    output = tmp_path / "sky.fits"
    _write_image(source)
    args = build_parser().parse_args(
        [
            "sky",
            "estimate",
            str(source),
            str(output),
            "--method",
            "polynomial",
            "--statistic",
            "median",
        ]
    )
    with pytest.raises(SystemExit, match="median is available only"):
        args.handler(args)


def test_viz_forwards_every_public_option(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setitem(__import__("sys").modules, "uvicorn", types.ModuleType("uvicorn"))
    monkeypatch.setattr(
        "torchsky.viz.serve",
        lambda **kwargs: captured.update(kwargs),
    )
    args = build_parser().parse_args(
        [
            "viz",
            "--root",
            str(tmp_path),
            "--port",
            "5000",
            "--host",
            "0.0.0.0",
            "--no-browser",
            "--cache-mb",
            "64",
            "--log-level",
            "info",
            "--max-registered-surveys",
            "7",
            "--max-validation-pairs",
            "5",
            "--max-image-pixels",
            "4096",
            "--expose-health-root",
            "--cors-origin",
            "https://example.invalid",
            "--cors-credentials",
        ]
    )
    assert args.handler(args) == 0
    assert captured == {
        "root": str(tmp_path),
        "port": 5000,
        "host": "0.0.0.0",
        "open_browser": False,
        "cache_mb": 64,
        "log_level": "info",
        "max_registered_surveys": 7,
        "max_validation_pairs": 5,
        "max_image_pixels": 4096,
        "expose_root_in_health": True,
        "cors_allow_origins": ["https://example.invalid"],
        "cors_allow_credentials": True,
    }
