"""SkyFitConfig: validation, pinned defaults (literal values of the prototype's command-line parser), the private mapping, and source hygiene."""

import ast
import dataclasses
from pathlib import Path

import pytest

from torchsky.map.exposure_fit import SkyFitConfig
from torchsky.map.exposure_fit._config import (
    _INTERNAL_SETTINGS,
    _PUBLIC_TO_ATTR,
    _namespace,
)

PUBLIC_DEFAULTS = {
    "sky_pitch": 0.04,
    "item_count": 60000,
    "item_width_floor": 0.0,
    "item_width_max": 2.56,
    "spectral_k": 12,
    "kernel_extent": 10.0,
    "kernel_norm_radius": 2.0,
    "background_order": 2,
    "noise_scale_max": 2.0,
    "noise_floor_max": 0.2,
    "reference_instrument": None,
    "steps": 1000000,
    "budget_minutes": 30.0,
    "patches_per_step": 5,
    "patch_size": 2.5,
    "seed": 0,
    "device": "auto",
}

INTERNAL_DEFAULTS = {
    "affine_prior": 0.01,
    "amplitude": "structured",
    "blur_floor": 1.0,
    "ckpt_kernel": 0,
    "clip": 10.0,
    "coarse_reg": "off",
    "coarse_reg_block": 0.6,
    "coarse_reg_instruments": "",
    "coarse_reg_min_peak": 0.05,
    "coarse_reg_min_ratio": 1.5,
    "coarse_reg_radius": 1.0,
    "coarse_reg_smooth": 0.1,
    "dh": 16,
    "dilation_stop": 0.0,
    "em_damp": 0.5,
    "enc_mmax": 12,
    "enc_nrad": 8,
    "enc_r0": 0.0,
    "filter_input": "curve",
    "floor0": 0.01,
    "gpu_resident_gb": 4.0,
    "held_ref_cuts": 24.0,
    "held_reg_steps": 100,
    "held_rounds": 4,
    "held_step_rows": 1,
    "held_steps": 150,
    "held_xfit": 1,
    "huber_delta": 3.0,
    "inner_sky": 2,
    "is_frac": 0.5,
    "is_refresh": 200,
    "is_snr": 20.0,
    "item_chunk": 0,
    "item_levels": 3,
    "item_lr_angle": 0.05,
    "item_lr_coef": 0.05,
    "item_lr_pos": 0.05,
    "item_lr_width": 0.03,
    "k_exp": "learned",
    "k_hidden": 64,
    "k_layers": 3,
    "kernel_centroid_gauge": 0,
    "level_factor": 2,
    "level_nh": 24,
    "level_overlap": 0.5,
    "log_every": 25,
    "loss": "huber",
    "lr_affine": 0.1,
    "lr_bkg": 0.1,
    "lr_codes": 0.02,
    "lr_op": 0.001,
    "n_lambda": 5,
    "nuisance_steps": "exposure",
    "pedestal_gauge": 1,
    "pix_default": 0.031,
    "pred_batch": 8,
    "pred_margin": 0.3,
    "prior_pixels": 0.0,
    "reg_prepass_iters": 2,
    "reg_prepass_steps": 150,
    "region_px": 256,
    "render_frame": "detector",
    "render_jitter": 1,
    "render_os_pix": 6.0,
    "render_pitch": 0.0,
    "sky_swap": 0,
    "spec_k_rule": "fixed",
    "spectral_domain": "union",
    "stripe_l2": 0.001,
    "stripes": 1,
    "tau0_code": 0.3,
    "tau0_g": 0.02,
    "tau_accum": 4,
    "tau_mode": "learned",
    "tau_start": 0.35,
    "tile_weight_power": 0.5,
    "val_every": 4,
    "visit_steps": 10,
    "warm_kernel_steps": 400,
    "sky_margin": 2.0,
    "data_margin": 1.0,
}


def test_public_defaults_pinned():
    assert dataclasses.asdict(SkyFitConfig()) == PUBLIC_DEFAULTS


def test_internal_defaults_pinned():
    assert dict(_INTERNAL_SETTINGS) == INTERNAL_DEFAULTS
    assert len(INTERNAL_DEFAULTS) == 81
    with pytest.raises(TypeError):
        _INTERNAL_SETTINGS["loss"] = "gauss"  # a read-only mapping


def test_namespace_has_every_attribute_once_and_is_fresh():
    ns = vars(_namespace(SkyFitConfig()))
    assert set(ns) == set(_PUBLIC_TO_ATTR.values()) | set(INTERNAL_DEFAULTS)
    assert len(ns) == len(PUBLIC_DEFAULTS) - 1 + len(
        INTERNAL_DEFAULTS
    )  # background_order has no attribute: the moved code fits a quadratic
    assert (
        ns["item_count"] == 60000
        and ns["budget_min"] == 30.0
        and ns["batch_patches"] == 5
    )
    a, b = _namespace(SkyFitConfig()), _namespace(SkyFitConfig())
    a.spec_k = 99
    assert b.spec_k == 12  # a fresh object every call (the moved code writes to it)
    assert _namespace(SkyFitConfig(), {"loss": "gauss"}).loss == "gauss"
    with pytest.raises(KeyError):
        _namespace(SkyFitConfig(), {"no_such_setting": 1})


def test_round_trip_and_frozen():
    cfg = SkyFitConfig(steps=7, device="cpu", reference_instrument=-1)
    assert SkyFitConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.steps = 1
    with pytest.raises(TypeError):
        SkyFitConfig.from_dict({"nope": 1})


@pytest.mark.parametrize(
    "bad",
    [
        {"sky_pitch": 0.0},
        {"sky_pitch": float("nan")},
        {"item_count": 0},
        {"item_count": 2.5},
        {"item_width_floor": -1.0},
        {"item_width_floor": 3.0, "item_width_max": 2.0},
        {"spectral_k": 1},
        {"kernel_extent": 0.0},
        {"kernel_norm_radius": -1.0},
        {"background_order": 3},
        {"noise_scale_max": 0.5},
        {"noise_floor_max": 1.5},
        {"reference_instrument": -2},
        {"steps": 0},
        {"budget_minutes": 0.0},
        {"patches_per_step": 0},
        {"patch_size": -1.0},
        {"seed": -1},
        {"device": "banana"},
        {"device": 3},
        {"steps": True},
    ],
)
def test_validation_errors(bad):
    with pytest.raises(ValueError):
        SkyFitConfig(**bad)


def test_allowed_edge_values():
    SkyFitConfig(
        item_width_floor=0.0,
        item_width_max=0.0,
        noise_scale_max=0.0,
        noise_floor_max=0.0,
        kernel_norm_radius=0.0,
        reference_instrument=-1,
    )


def test_source_hygiene():
    root = Path(__import__("torchsky.map.exposure_fit", fromlist=["x"]).__file__).parent
    bad = {"fieldop", "fitsio", "astropy", "argparse", "harness"}
    for p in list(root.glob("*.py")) + [
        Path(__file__).parent / "_exposure_fit_synth.py"
    ]:
        for n in ast.walk(ast.parse(p.read_text())):
            if isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and not n.level:
                names = [n.module or ""]
            else:
                names = []
            for m in names:
                assert m.split(".")[0] not in bad, (p.name, m)
                assert not m.startswith("torchsky_harness"), (p.name, m)
