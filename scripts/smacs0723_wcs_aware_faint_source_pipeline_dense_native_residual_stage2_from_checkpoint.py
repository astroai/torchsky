#!/usr/bin/env python3
"""Run N/F/P/J continuations from one Stage-1 dense-pipeline checkpoint.

The N and F branches run through
``smacs0723_wcs_aware_faint_source_pipeline_dense_native_residual.py``.
The P and J branches run through the learnable-PSF launcher, which itself is
built on that same dense launcher. All branches initialize the scene and
background from the same checkpoint and create a fresh optimizer.

N: no-PSF continuation; scene/background trainable.
F: fixed STPSF forward; scene/background trainable.
P: learnable PSF forward; only PSF parameters trainable.
J: learnable PSF forward; scene/background/PSF jointly trainable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for search_path in (ROOT, SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import scripts.smacs0723_wcs_aware_faint_source_pipeline_dense_native_residual as dense  # noqa: E402
import scripts.smacs0723_wcs_aware_faint_source_pipeline_learnable_psf_dense_native_residual as learnable  # noqa: E402


BRANCHES = ("N", "F", "P", "J")


def parse_wrapper_options(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--initialize-from-checkpoint",
        type=Path,
        required=True,
        help="Stage-1 checkpoint whose model state initializes this branch.",
    )
    parser.add_argument(
        "--stage2-branch",
        choices=BRANCHES,
        required=True,
        help="N=no PSF, F=fixed STPSF, P=PSF-only, J=joint learnable PSF.",
    )
    namespace, remaining = parser.parse_known_args(list(argv))
    checkpoint = namespace.initialize_from_checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    namespace.initialize_from_checkpoint = checkpoint
    return namespace, remaining


def remove_option(argv: list[str], name: str, *, takes_value: bool) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == name:
            index += 2 if takes_value else 1
        elif takes_value and token.startswith(name + "="):
            index += 1
        else:
            result.append(token)
            index += 1
    return result


def set_value(argv: list[str], name: str, value: str) -> list[str]:
    return [*remove_option(argv, name, takes_value=True), name, value]


def set_flag(argv: list[str], positive: str, negative: str, enabled: bool) -> list[str]:
    result = remove_option(argv, positive, takes_value=False)
    result = remove_option(result, negative, takes_value=False)
    return [*result, positive if enabled else negative]


def checkpoint_state(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    state = payload.get("model")
    if state is None:
        state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint has neither 'model' nor 'model_state_dict'")
    return payload, state


def validate_checkpoint_architecture(
    payload: dict[str, Any], remaining: Sequence[str]
) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--architecture", default="A1_nerf")
    requested, _ = parser.parse_known_args(list(remaining))
    configuration = payload.get("configuration", {})
    stored = None
    if isinstance(configuration, dict):
        stored = configuration.get("architecture")
    if stored is not None and str(stored) != str(requested.architecture):
        raise ValueError(
            f"checkpoint architecture {stored!r} != requested {requested.architecture!r}"
        )


def install_dense_initializer(path: Path) -> None:
    payload, state = checkpoint_state(path)
    validate_checkpoint_architecture(payload, sys.argv[1:])
    original_model = dense.base.WCSAwareLiteratureINR

    class InitializedDenseModel(original_model):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.load_state_dict(state, strict=True)
            self.initialization_checkpoint = str(path)
            print(
                json.dumps(
                    {
                        "initialized_from_checkpoint": str(path),
                        "load_mode": "strict",
                    }
                ),
                flush=True,
            )

    dense.base.WCSAwareLiteratureINR = InitializedDenseModel


def install_learnable_initializer(path: Path, *, psf_only: bool) -> None:
    payload, state = checkpoint_state(path)
    validate_checkpoint_architecture(payload, sys.argv[1:])
    original_model = learnable.LearnablePSFModel

    class InitializedLearnableModel(original_model):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            incompatible = self.load_state_dict(state, strict=False)
            missing = list(incompatible.missing_keys)
            unexpected = list(incompatible.unexpected_keys)
            invalid_missing = [
                key for key in missing if not key.startswith("learnable_psf.")
            ]
            if invalid_missing or unexpected:
                raise RuntimeError(
                    "checkpoint/model mismatch: "
                    f"invalid missing={invalid_missing}, unexpected={unexpected}"
                )
            if psf_only:
                for name, parameter in self.named_parameters():
                    parameter.requires_grad_(name.startswith("learnable_psf."))
            self.initialization_checkpoint = str(path)
            print(
                json.dumps(
                    {
                        "initialized_from_checkpoint": str(path),
                        "load_mode": "scene strict; learnable_psf newly initialized",
                        "missing_learnable_psf_keys": len(missing),
                        "optimization": "psf_only" if psf_only else "joint",
                    }
                ),
                flush=True,
            )

    learnable.LearnablePSFModel = InitializedLearnableModel


def configure_branch(branch: str, argv: Sequence[str]) -> list[str]:
    result = list(argv)
    result = set_value(result, "--pretrain-steps", "0")
    result = set_value(result, "--finetune-steps", "1000")
    result = set_value(result, "--batch-size", "100")
    # The parser accepts one total batch or two channel-specific values. Append
    # the second value explicitly after removing any existing batch option.
    result = remove_option(list(argv), "--batch-size", takes_value=True)
    # Remove a possible second value left by nargs='+' from the normal commands.
    while result and result[-1].isdigit() and result[-1] in {"50", "100", "200", "400"}:
        break
    result = set_value(result, "--pretrain-steps", "0")
    result = set_value(result, "--finetune-steps", "1000")
    result.extend(["--batch-size", "100", "100"])
    result = set_value(result, "--sampling-mode", "epoch_without_replacement")
    result = set_value(result, "--checkpoint-every", "1000")
    result = set_value(result, "--save-residual-every-steps", "1000")
    result = set_flag(
        result,
        "--final-psf-convolution",
        "--no-final-psf-convolution",
        False,
    )

    if branch == "N":
        result = set_value(result, "--pretrain-steps", "1000")
        result = set_value(result, "--finetune-steps", "0")
        result = set_value(result, "--psf-mode", "none")
    elif branch == "F":
        result = set_value(result, "--psf-mode", "stpsf")
    else:
        # The learnable launcher replaces these internally, but explicit values
        # make the effective intent visible in the command/manifest.
        result = set_value(result, "--psf-mode", "none")
    return result


def main() -> int:
    wrapper, remaining = parse_wrapper_options(sys.argv[1:])
    branch = str(wrapper.stage2_branch)
    configured = configure_branch(branch, remaining)
    sys.argv = [sys.argv[0], *configured]

    if branch in {"N", "F"}:
        install_dense_initializer(wrapper.initialize_from_checkpoint)
        return int(dense.main())

    install_learnable_initializer(
        wrapper.initialize_from_checkpoint,
        psf_only=(branch == "P"),
    )
    return int(learnable.main())


if __name__ == "__main__":
    raise SystemExit(main())
