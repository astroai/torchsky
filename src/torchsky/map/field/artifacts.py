"""Safe, resumable continuous-field artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from ... import __version__ as torchsky_version
from ..plan import verify_map_input_identity
from .core import (
    DiffuseFoamHybridField,
    FoamInitializedINRField,
    SpectralPowerFoamField,
    make_field_model,
)
from .registry import FIELD_MODEL_KINDS
from .types import FieldDomain, FieldFitResult, FieldModelKind


_MANIFEST_NAME = "manifest.json"
_WEIGHTS_NAME = "weights.pt"
_TRAINING_STATE_NAME = "training_state.pt"
_FOAM_CONSTRUCTOR = "spectral_power_foam_v1"
_FOAM_INR_CONSTRUCTOR = "foam_initialized_inr_v1"
_DIFFUSE_FOAM_HYBRID_CONSTRUCTOR = "diffuse_foam_hybrid_v1"


def _source_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[4],
        check=False,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("field artifact metadata must be finite JSON data") from exc


def _cpu_safe_state(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_safe_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_safe_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_safe_state(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported training-state value {type(value).__name__}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = stream.name
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        raise


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(_cpu_safe_state(dict(payload)), temporary)
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _foam_constructor_metadata(model: SpectralPowerFoamField) -> dict[str, Any]:
    gradient = model.foam.cell_gradient_coefficients
    return {
        "temperature_start": float(model.temperature_start),
        "temperature_end": float(model.temperature_end),
        "current_temperature": float(model._temperature),
        "compact_fixed_geometry": bool(model.compact_fixed_geometry),
        "log_intensity_spectra": bool(model.foam.log_intensity_spectra),
        "sites_requires_grad": bool(model.foam.sites.requires_grad),
        "radii_requires_grad": bool(model.foam.raw_radii.requires_grad),
        "coefficients_requires_grad": bool(
            model.foam.cell_coefficients.requires_grad
        ),
        "gradient_coefficients_requires_grad": (
            None if gradient is None else bool(gradient.requires_grad)
        ),
    }


def _artifact_model_kwargs(
    model: nn.Module,
    model_kind: FieldModelKind,
    model_kwargs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if model_kind == "spectral_power_foam":
        if not isinstance(model, SpectralPowerFoamField):
            raise ValueError("model_kind spectral_power_foam requires a Foam model")
        return {
            "artifact_constructor": _FOAM_CONSTRUCTOR,
            **_foam_constructor_metadata(model),
        }
    if model_kind == "foam_initialized_inr":
        if not isinstance(model, FoamInitializedINRField):
            raise ValueError(
                "model_kind foam_initialized_inr requires a Foam-initialized model"
            )
        hidden_layers = sum(
            type(module) is nn.Linear for module in model.residual_trunk
        )
        return {
            "artifact_constructor": _FOAM_INR_CONSTRUCTOR,
            "hidden_dim": int(model.residual_output.in_features),
            "hidden_layers": int(hidden_layers),
            "intensity_floor": float(model.intensity_floor),
            "max_log_correction": float(model.max_log_correction),
            "reference_foam": _foam_constructor_metadata(model.reference_foam),
        }
    if model_kind == "diffuse_foam_hybrid":
        if not isinstance(model, DiffuseFoamHybridField):
            raise ValueError(
                "model_kind diffuse_foam_hybrid requires a diffuse/Foam model"
            )
        return {
            "artifact_constructor": _DIFFUSE_FOAM_HYBRID_CONSTRUCTOR,
            "diffuse_hidden_dim": int(model.diffuse_field.output.in_features),
            "diffuse_hidden_layers": len(model.diffuse_field.hidden),
            "diffuse_omega0": float(model.diffuse_field.omega0),
            "training_stage": model.training_stage,
            "localized_foam": _foam_constructor_metadata(model.localized_foam),
        }
    return _json_copy(dict(model_kwargs or {}))


def _validate_positive_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"field artifact {name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"field artifact {name} must be finite and positive")
    return result


def _validate_foam_constructor_metadata(payload: Mapping[str, Any]) -> None:
    for name in ("temperature_start", "temperature_end", "current_temperature"):
        _validate_positive_number(payload.get(name), name=name)
    for name in (
        "compact_fixed_geometry",
        "log_intensity_spectra",
        "sites_requires_grad",
        "radii_requires_grad",
        "coefficients_requires_grad",
    ):
        if not isinstance(payload.get(name), bool):
            raise ValueError(f"field artifact {name} must be boolean")
    gradient_flag = payload.get("gradient_coefficients_requires_grad")
    if gradient_flag is not None and not isinstance(gradient_flag, bool):
        raise ValueError(
            "field artifact gradient_coefficients_requires_grad must be boolean or null"
        )


def _validate_artifact_model_kwargs(
    model_kind: str,
    kwargs: Mapping[str, Any],
) -> None:
    constructor = kwargs.get("artifact_constructor")
    if model_kind == "spectral_power_foam":
        if constructor != _FOAM_CONSTRUCTOR:
            raise ValueError("Foam artifact requires its tensor-safe constructor")
        _validate_foam_constructor_metadata(kwargs)
    elif model_kind == "foam_initialized_inr":
        if constructor != _FOAM_INR_CONSTRUCTOR:
            raise ValueError(
                "Foam-initialized artifact requires its tensor-safe constructor"
            )
        hidden_dim = kwargs.get("hidden_dim")
        hidden_layers = kwargs.get("hidden_layers")
        if (
            isinstance(hidden_dim, bool)
            or not isinstance(hidden_dim, int)
            or hidden_dim < 1
            or isinstance(hidden_layers, bool)
            or not isinstance(hidden_layers, int)
            or hidden_layers < 1
        ):
            raise ValueError("Foam-initialized artifact hidden sizes must be positive")
        _validate_positive_number(
            kwargs.get("intensity_floor"),
            name="intensity_floor",
        )
        _validate_positive_number(
            kwargs.get("max_log_correction"),
            name="max_log_correction",
        )
        reference = kwargs.get("reference_foam")
        if not isinstance(reference, dict):
            raise ValueError("Foam-initialized artifact requires reference_foam")
        _validate_foam_constructor_metadata(reference)
    elif model_kind == "diffuse_foam_hybrid":
        if constructor != _DIFFUSE_FOAM_HYBRID_CONSTRUCTOR:
            raise ValueError("diffuse/Foam artifact requires its tensor-safe constructor")
        hidden_dim = kwargs.get("diffuse_hidden_dim")
        hidden_layers = kwargs.get("diffuse_hidden_layers")
        if (
            isinstance(hidden_dim, bool)
            or not isinstance(hidden_dim, int)
            or hidden_dim < 1
            or isinstance(hidden_layers, bool)
            or not isinstance(hidden_layers, int)
            or hidden_layers < 1
        ):
            raise ValueError("diffuse/Foam artifact hidden sizes must be positive")
        _validate_positive_number(
            kwargs.get("diffuse_omega0"),
            name="diffuse_omega0",
        )
        if kwargs.get("training_stage") not in DiffuseFoamHybridField._STAGES:
            raise ValueError("diffuse/Foam artifact training_stage is invalid")
        localized = kwargs.get("localized_foam")
        if not isinstance(localized, dict):
            raise ValueError("diffuse/Foam artifact requires localized_foam")
        _validate_foam_constructor_metadata(localized)
    elif constructor is not None:
        raise ValueError("artifact_constructor is reserved for structured models")


def _state_tensor(state: Mapping[str, Any], name: str) -> Tensor:
    value = state.get(name)
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError(f"field artifact state requires floating tensor {name!r}")
    return value


def _foam_from_state(
    domain: FieldDomain,
    state: Mapping[str, Any],
    *,
    prefix: str,
    metadata: Mapping[str, Any],
) -> SpectralPowerFoamField:
    sites = _state_tensor(state, f"{prefix}foam.sites")
    raw_radii = _state_tensor(state, f"{prefix}foam.raw_radii")
    coefficients = _state_tensor(state, f"{prefix}foam.cell_coefficients")
    minimum_radius = _state_tensor(state, f"{prefix}foam._minimum_radius")
    if sites.ndim != 2 or sites.shape[1] != 2 or sites.shape[0] < 1:
        raise ValueError("field artifact Foam sites must have shape (cells, 2)")
    if raw_radii.shape != (sites.shape[0],):
        raise ValueError("field artifact Foam radii shape does not match sites")
    if coefficients.ndim != 2 or coefficients.shape[0] != sites.shape[0]:
        raise ValueError("field artifact Foam coefficient shape does not match sites")
    if minimum_radius.numel() != 1:
        raise ValueError("field artifact Foam minimum radius must be scalar")
    gradient_name = f"{prefix}foam.cell_gradient_coefficients"
    gradient = state.get(gradient_name)
    gradient_flag = metadata["gradient_coefficients_requires_grad"]
    if (gradient is None) != (gradient_flag is None):
        raise ValueError(
            "field artifact Foam gradient metadata does not match tensor state"
        )
    if gradient is not None and (
        not isinstance(gradient, Tensor)
        or not gradient.is_floating_point()
        or gradient.shape != coefficients.shape + (2,)
    ):
        raise ValueError("field artifact Foam gradient coefficient shape is invalid")
    radius_floor = float(minimum_radius.item())
    placeholder_radii = torch.full_like(raw_radii, radius_floor + 1.0)
    model = SpectralPowerFoamField(
        domain,
        sites=torch.zeros_like(sites),
        radii=placeholder_radii,
        cell_coefficients=torch.zeros_like(coefficients),
        cell_gradient_coefficients=(
            None if gradient is None else torch.zeros_like(gradient)
        ),
        minimum_radius=radius_floor,
        temperature_start=float(metadata["temperature_start"]),
        temperature_end=float(metadata["temperature_end"]),
        compact_fixed_geometry=bool(metadata["compact_fixed_geometry"]),
        log_intensity_spectra=bool(metadata["log_intensity_spectra"]),
    )
    model._temperature = float(metadata["current_temperature"])
    model.foam.sites.requires_grad_(bool(metadata["sites_requires_grad"]))
    model.foam.raw_radii.requires_grad_(bool(metadata["radii_requires_grad"]))
    model.foam.cell_coefficients.requires_grad_(
        bool(metadata["coefficients_requires_grad"])
    )
    if model.foam.cell_gradient_coefficients is not None:
        model.foam.cell_gradient_coefficients.requires_grad_(
            bool(metadata["gradient_coefficients_requires_grad"])
        )
    return model


def _model_from_artifact_state(
    manifest: FieldArtifactManifest,
    state: Mapping[str, Any],
    *,
    device: torch.device,
) -> nn.Module:
    kwargs = manifest.model_kwargs
    constructor = kwargs.get("artifact_constructor")
    if constructor == _FOAM_CONSTRUCTOR:
        return _foam_from_state(
            manifest.domain,
            state,
            prefix="",
            metadata=kwargs,
        ).to(device=device)
    if constructor == _FOAM_INR_CONSTRUCTOR:
        reference_metadata = kwargs["reference_foam"]
        assert isinstance(reference_metadata, dict)
        reference = _foam_from_state(
            manifest.domain,
            state,
            prefix="reference_foam.",
            metadata=reference_metadata,
        )
        return FoamInitializedINRField(
            manifest.domain,
            foam=reference,
            hidden_dim=int(kwargs["hidden_dim"]),
            hidden_layers=int(kwargs["hidden_layers"]),
            intensity_floor=float(kwargs["intensity_floor"]),
            max_log_correction=float(kwargs["max_log_correction"]),
            copy_reference=False,
        ).to(device=device)
    if constructor == _DIFFUSE_FOAM_HYBRID_CONSTRUCTOR:
        localized_metadata = kwargs["localized_foam"]
        assert isinstance(localized_metadata, dict)
        localized = _foam_from_state(
            manifest.domain,
            state,
            prefix="localized_foam.",
            metadata=localized_metadata,
        )
        return DiffuseFoamHybridField(
            manifest.domain,
            foam=localized,
            diffuse_hidden_dim=int(kwargs["diffuse_hidden_dim"]),
            diffuse_hidden_layers=int(kwargs["diffuse_hidden_layers"]),
            diffuse_omega0=float(kwargs["diffuse_omega0"]),
            training_stage=str(kwargs["training_stage"]),
        ).to(device=device)
    return make_field_model(
        manifest.model_kind,
        manifest.domain,
        **kwargs,
    ).to(device=device)


def capture_field_rng_state(*, sampling_rng_state: Tensor | None = None) -> dict:
    state: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    get_mps_state = getattr(torch.mps, "get_rng_state", None)
    if callable(get_mps_state) and torch.backends.mps.is_available():
        state["mps"] = get_mps_state()
    if sampling_rng_state is not None:
        state["sampling"] = sampling_rng_state
    return _cpu_safe_state(state)


def restore_field_rng_state(state: Mapping[str, Any]) -> Tensor | None:
    cpu_state = state.get("cpu")
    if isinstance(cpu_state, Tensor):
        torch.set_rng_state(cpu_state.to(device="cpu", dtype=torch.uint8))
    cuda_state = state.get("cuda")
    if torch.cuda.is_available() and isinstance(cuda_state, (list, tuple)):
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(item, dtype=torch.uint8, device="cpu") for item in cuda_state]
        )
    mps_state = state.get("mps")
    set_mps_state = getattr(torch.mps, "set_rng_state", None)
    if isinstance(mps_state, Tensor) and callable(set_mps_state):
        set_mps_state(mps_state.to(device="cpu", dtype=torch.uint8))
    sampling = state.get("sampling")
    return (
        sampling.to(device="cpu", dtype=torch.uint8)
        if isinstance(sampling, Tensor)
        else None
    )


@dataclass(frozen=True)
class FieldArtifactManifest:
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        payload = _json_copy(self.payload)
        if payload.get("schema_version") != 1:
            raise ValueError("field artifact schema_version must be 1")
        model = payload.get("model")
        if not isinstance(model, dict) or model.get("kind") not in FIELD_MODEL_KINDS:
            raise ValueError("field artifact requires a registered model kind")
        if not isinstance(model.get("kwargs"), dict):
            raise ValueError("field artifact model kwargs must be a mapping")
        _validate_artifact_model_kwargs(model["kind"], model["kwargs"])
        domain_payload = model.get("domain")
        if not isinstance(domain_payload, dict):
            raise ValueError("field artifact requires a model domain")
        domain = FieldDomain(**domain_payload)
        for value in (
            domain.center_ra_deg,
            domain.center_dec_deg,
            domain.spatial_scale_deg,
            domain.wavelength_center,
            domain.wavelength_scale,
        ):
            if not math.isfinite(float(value)):
                raise ValueError("field artifact domain must be finite")
        if domain.spatial_scale_deg <= 0.0 or domain.wavelength_scale <= 0.0:
            raise ValueError("field artifact domain scales must be positive")
        for key in ("operator_manifest", "calibration_manifest", "fit_config"):
            if not isinstance(payload.get(key), dict):
                raise ValueError(f"field artifact {key} must be a mapping")
        inputs = payload.get("input_identities")
        if not isinstance(inputs, list):
            raise ValueError("field artifact input_identities must be a list")
        selected = payload.get("selected_checkpoint")
        if selected != _WEIGHTS_NAME:
            raise ValueError(f"selected_checkpoint must be {_WEIGHTS_NAME!r}")
        training_state = payload.get("training_state")
        if training_state not in (None, _TRAINING_STATE_NAME):
            raise ValueError("unsupported field artifact training_state path")
        object.__setattr__(self, "payload", payload)

    @property
    def model_kind(self) -> FieldModelKind:
        return self.payload["model"]["kind"]

    @property
    def model_kwargs(self) -> dict[str, Any]:
        return dict(self.payload["model"]["kwargs"])

    @property
    def domain(self) -> FieldDomain:
        return FieldDomain(**self.payload["model"]["domain"])

    def to_dict(self) -> dict[str, Any]:
        return _json_copy(self.payload)


@dataclass(frozen=True)
class LoadedFieldArtifact:
    manifest: FieldArtifactManifest
    model: nn.Module
    optimizer_state: dict[str, Any] | None
    amp_scaler_state: dict[str, Any] | None
    rng_state: dict[str, Any] | None
    sampling_rng_state: Tensor | None
    completed_steps: int


def save_field_artifact(
    path: str | Path,
    *,
    model: nn.Module,
    model_kind: FieldModelKind,
    model_kwargs: Mapping[str, Any] | None = None,
    domain: FieldDomain | None = None,
    operator_manifest: Mapping[str, Any] | None = None,
    calibration_manifest: Mapping[str, Any] | None = None,
    fit_config: Mapping[str, Any] | None = None,
    input_identities: list[Mapping[str, Any]] | None = None,
    fit_result: FieldFitResult | None = None,
    overwrite: bool = False,
) -> FieldArtifactManifest:
    """Save a field model, manifest, and optional training state."""
    root = Path(path).expanduser().resolve()
    manifest_path = root / _MANIFEST_NAME
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"field artifact already exists: {root}")
    resolved_domain = domain or getattr(model, "domain", None)
    if not isinstance(resolved_domain, FieldDomain):
        raise ValueError("field artifact requires an explicit FieldDomain")
    if model_kind not in FIELD_MODEL_KINDS:
        raise ValueError(f"unknown field model kind {model_kind!r}")
    has_training_state = fit_result is not None
    payload = {
        "schema_version": 1,
        "model": {
            "kind": model_kind,
            "kwargs": _artifact_model_kwargs(model, model_kind, model_kwargs),
            "domain": asdict(resolved_domain),
        },
        "operator_manifest": dict(operator_manifest or {}),
        "calibration_manifest": dict(calibration_manifest or {}),
        "fit_config": dict(fit_config or {}),
        "selected_checkpoint": _WEIGHTS_NAME,
        "training_state": _TRAINING_STATE_NAME if has_training_state else None,
        "software": {"torchsky": torchsky_version, "commit": _source_commit()},
        "input_identities": [dict(identity) for identity in input_identities or []],
    }
    manifest = FieldArtifactManifest(payload)
    root.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(
        root / _WEIGHTS_NAME,
        {"model": model.state_dict()},
    )
    if fit_result is not None:
        rng_state = capture_field_rng_state(
            sampling_rng_state=fit_result.sampling_rng_state
        )
        _atomic_torch_save(
            root / _TRAINING_STATE_NAME,
            {
                "optimizer": fit_result.optimizer_state,
                "amp_scaler": fit_result.amp_scaler_state,
                "rng": rng_state,
                "completed_steps": int(fit_result.completed_steps),
            },
        )
    _atomic_json(manifest_path, manifest.to_dict())
    return manifest


def load_field_artifact(
    path: str | Path,
    *,
    device: torch.device | str | None = None,
    verify_inputs: bool = True,
) -> LoadedFieldArtifact:
    """Load and validate a saved field model artifact."""
    root = Path(path).expanduser().resolve()
    with (root / _MANIFEST_NAME).open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError("field artifact manifest root must be a JSON object")
    manifest = FieldArtifactManifest(raw)
    if verify_inputs:
        for identity in manifest.payload["input_identities"]:
            verify_map_input_identity(identity)
    map_location = torch.device(device or "cpu")
    weights = torch.load(
        root / manifest.payload["selected_checkpoint"],
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(weights, dict) or not isinstance(weights.get("model"), dict):
        raise ValueError("field artifact weights must contain a model state dict")
    model_state = weights["model"]
    if not model_state or any(
        not isinstance(name, str) or not isinstance(value, Tensor)
        for name, value in model_state.items()
    ):
        raise ValueError("field artifact model state must contain named tensors")
    if any(
        value.is_floating_point() and not bool(torch.isfinite(value).all())
        for value in model_state.values()
    ):
        raise ValueError("field artifact model state must be finite")
    model = _model_from_artifact_state(
        manifest,
        model_state,
        device=map_location,
    )
    model.load_state_dict(model_state, strict=True)

    optimizer_state = None
    amp_scaler_state = None
    rng_state = None
    sampling_rng_state = None
    completed_steps = 0
    training_state_name = manifest.payload.get("training_state")
    if training_state_name is not None:
        training_state = torch.load(
            root / training_state_name,
            map_location=map_location,
            weights_only=True,
        )
        if not isinstance(training_state, dict):
            raise ValueError("field artifact training state must be a mapping")
        optimizer = training_state.get("optimizer")
        optimizer_state = optimizer if isinstance(optimizer, dict) else None
        amp_scaler = training_state.get("amp_scaler")
        amp_scaler_state = (
            amp_scaler if isinstance(amp_scaler, dict) else None
        )
        rng = training_state.get("rng")
        rng_state = rng if isinstance(rng, dict) else None
        if rng_state is not None:
            sampling_rng_state = restore_field_rng_state(rng_state)
        completed_steps = int(training_state.get("completed_steps", 0))
    return LoadedFieldArtifact(
        manifest=manifest,
        model=model,
        optimizer_state=optimizer_state,
        amp_scaler_state=amp_scaler_state,
        rng_state=rng_state,
        sampling_rng_state=sampling_rng_state,
        completed_steps=completed_steps,
    )


__all__ = [
    "FieldArtifactManifest",
    "LoadedFieldArtifact",
    "capture_field_rng_state",
    "load_field_artifact",
    "restore_field_rng_state",
    "save_field_artifact",
]
