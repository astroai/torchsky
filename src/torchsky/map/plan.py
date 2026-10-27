"""Validated, content-addressed map execution plans."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from torchfits import read_header as get_header, verify_checksums
from .operators.config import (
    MAP_RESAMPLING_KERNELS,
    MAP_STACKING_METHODS,
    MAP_VALUE_SEMANTICS,
)


_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_PLAN_WORKFLOWS = ("native", "drizzle", "swarp", "montage")
_PLAN_BACKGROUND_MODES = ("none", "per-exposure", "overlap")
_PLAN_PRODUCTS = ("SCI", "WHT", "IVAR", "DQ", "CTX", "MASK", "NEXP", "NCORR")
_PLAN_DTYPES = ("float16", "bfloat16", "float32", "float64")
_PLAN_FIELD_PRESETS = ("fast", "balanced", "spectral", "operator_constrained")
_PLAN_FIELD_SAMPLING = ("center", "gauss_legendre_2x2")
_PLAN_AMP_DTYPES = ("none", "float16", "bfloat16")


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"map plan {label} must be an object")
    return dict(value)


def _require_string(value: Any, label: str, *, allow_none: bool = False) -> None:
    if allow_none and value is None:
        return
    if not isinstance(value, str) or not value:
        raise ValueError(f"map plan {label} must be a non-empty string")


def _require_choice(value: Any, label: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"map plan {label} must be one of {choices}")


def _require_bool(value: Any, label: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"map plan {label} must be boolean")


def _require_number(
    value: Any,
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"map plan {label} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"map plan {label} must be finite")
    if positive and value <= 0:
        raise ValueError(f"map plan {label} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"map plan {label} must be non-negative")


def _require_integer(
    value: Any,
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"map plan {label} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"map plan {label} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"map plan {label} must be non-negative")


def _validate_hdu(value: Any, label: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"map plan {label} must be a non-negative HDU selector")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"map plan {label} must be non-negative")
        return
    if isinstance(value, str) and value and "\x00" not in value:
        return
    raise ValueError(f"map plan {label} must be a non-negative HDU selector")


def _validate_input_identity(identity: Mapping[str, Any], label: str) -> None:
    value = _require_mapping(identity, label)
    _require_string(value.get("path"), f"{label}.path")
    _require_integer(value.get("size"), f"{label}.size", nonnegative=True)
    _validate_hdu(value.get("hdu", 0), f"{label}.hdu")
    cards = value.get("fits_checksum_cards")
    sha256 = value.get("sha256")
    if cards is not None:
        cards_mapping = _require_mapping(cards, f"{label}.fits_checksum_cards")
        if not cards_mapping:
            raise ValueError(f"map plan {label}.fits_checksum_cards must not be empty")
        if any(key not in ("CHECKSUM", "DATASUM") for key in cards_mapping):
            raise ValueError(
                f"map plan {label}.fits_checksum_cards has an unknown card"
            )
        if any(
            not isinstance(card_value, str) or not card_value
            for card_value in cards_mapping.values()
        ):
            raise ValueError(f"map plan {label}.fits_checksum_cards values must be strings")
        if sha256 is not None:
            raise ValueError(f"map plan {label} cannot contain both checksums and SHA-256")
    else:
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ValueError(f"map plan {label} requires checksum cards or SHA-256")
        if any(char not in "0123456789abcdefABCDEF" for char in sha256):
            raise ValueError(f"map plan {label}.sha256 must be hexadecimal")


def _validate_optional_identity_pair(
    item: Mapping[str, Any],
    path_key: str,
    identity_key: str,
) -> None:
    path = item.get(path_key)
    identity = item.get(identity_key)
    if path is not None:
        _require_string(path, f"inputs[].{path_key}")
        if not isinstance(identity, Mapping):
            raise ValueError(
                f"map plan inputs[].{identity_key} is required for {path_key}"
            )
    elif identity is not None:
        raise ValueError(f"map plan inputs[].{path_key} is required for {identity_key}")
    if identity is not None:
        _validate_input_identity(identity, f"inputs[].{identity_key}")


def _validate_input(item: Mapping[str, Any], *, method: str) -> None:
    value = _require_mapping(item, "inputs[]")
    identity = _require_mapping(value.get("identity"), "inputs[].identity")
    _validate_input_identity(identity, "inputs[].identity")
    if "path" in value:
        _require_string(value["path"], "inputs[].path")
        if value["path"] != identity["path"]:
            raise ValueError("map plan inputs[].path must match identity.path")
    if "hdu" in value:
        _validate_hdu(value["hdu"], "inputs[].hdu")
        if value["hdu"] != identity.get("hdu", 0):
            raise ValueError("map plan inputs[].hdu must match identity.hdu")
    if "weight" in value:
        _require_number(value["weight"], "inputs[].weight", positive=True)
    if "weight_is_inverse_variance" in value:
        _require_bool(
            value["weight_is_inverse_variance"],
            "inputs[].weight_is_inverse_variance",
        )
    if "photometric_scale" in value:
        _require_number(
            value["photometric_scale"],
            "inputs[].photometric_scale",
            positive=True,
        )
    if "background_offset" in value:
        _require_number(value["background_offset"], "inputs[].background_offset")
    for path_key, identity_key in (
        ("weight_map", "weight_identity"),
        ("mask_map", "mask_identity"),
        ("primary_beam_map", "primary_beam_identity"),
    ):
        _validate_optional_identity_pair(value, path_key, identity_key)
    if method == "field":
        _require_string(value.get("band"), "inputs[].band")


def _validate_target(target: Mapping[str, Any]) -> None:
    value = _require_mapping(target, "target")
    shape = value.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in shape)
    ):
        raise ValueError("map plan target shape must contain two positive integers")
    header = _require_mapping(value.get("header"), "target.header")
    for key, expected in (("NAXIS1", shape[1]), ("NAXIS2", shape[0])):
        if key in header:
            _require_integer(header[key], f"target.header.{key}", positive=True)
            if header[key] != expected:
                raise ValueError(
                    f"map plan target.header.{key} must match target.shape"
                )
    if "band_count" in value:
        _require_integer(value["band_count"], "target.band_count", positive=True)
    if "reference" in value:
        _require_string(value["reference"], "target.reference", allow_none=True)
    if "reference_hdu" in value:
        _validate_hdu(value["reference_hdu"], "target.reference_hdu")
    if "union_footprint" in value:
        _require_bool(value["union_footprint"], "target.union_footprint")
    if "footprint_padding" in value:
        _require_number(value["footprint_padding"], "target.footprint_padding", nonnegative=True)
        if value["footprint_padding"] < 1:
            raise ValueError("map plan target.footprint_padding must be at least 1")
    if "footprint_tolerance_px" in value:
        _require_number(
            value["footprint_tolerance_px"],
            "target.footprint_tolerance_px",
            positive=True,
        )
    if "footprint_max_depth" in value:
        _require_integer(
            value["footprint_max_depth"],
            "target.footprint_max_depth",
            positive=True,
        )


def _validate_execution(execution: Mapping[str, Any], method: str) -> None:
    value = _require_mapping(execution, "execution")
    if "workflow" in value:
        _require_choice(value["workflow"], "execution.workflow", _PLAN_WORKFLOWS)
    if "kernel" in value:
        _require_choice(value["kernel"], "execution.kernel", MAP_RESAMPLING_KERNELS)
    if "stack" in value:
        _require_choice(value["stack"], "execution.stack", MAP_STACKING_METHODS)
    if "background_mode" in value:
        _require_choice(
            value["background_mode"],
            "execution.background_mode",
            _PLAN_BACKGROUND_MODES,
        )
    if "value_semantics" in value:
        _require_choice(
            value["value_semantics"],
            "execution.value_semantics",
            MAP_VALUE_SEMANTICS,
        )
    if "drizzle_pixfrac" in value:
        _require_number(value["drizzle_pixfrac"], "execution.drizzle_pixfrac", positive=True)
    if "clip_sigma" in value:
        _require_number(value["clip_sigma"], "execution.clip_sigma", positive=True)
    if "clip_iterations" in value:
        _require_integer(value["clip_iterations"], "execution.clip_iterations", positive=True)
    if "tile_hw" in value and value["tile_hw"] is not None:
        tile_hw = value["tile_hw"]
        if (
            not isinstance(tile_hw, list)
            or len(tile_hw) != 2
            or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
                for dim in tile_hw
            )
        ):
            raise ValueError("map plan execution.tile_hw must contain two positive integers")
    for key in ("requested_device", "resolved_device"):
        if key in value:
            _require_string(value[key], f"execution.{key}", allow_none=True)
    if "scratch_required_bytes" in value:
        _require_integer(
            value["scratch_required_bytes"],
            "execution.scratch_required_bytes",
            nonnegative=True,
        )

    if method == "field":
        _require_string(value.get("field_model"), "execution.field_model")
        _require_mapping(
            value.get("field_model_kwargs"),
            "execution.field_model_kwargs",
        )
        if "field_preset" not in value:
            raise ValueError("map plan execution.field_preset is required")
        if value.get("field_preset") is not None:
            _require_choice(
                value["field_preset"],
                "execution.field_preset",
                _PLAN_FIELD_PRESETS,
            )
        _require_integer(
            value.get("field_steps"),
            "execution.field_steps",
            nonnegative=True,
        )
        _require_number(
            value.get("field_learning_rate"),
            "execution.field_learning_rate",
            positive=True,
        )
        _require_integer(
            value.get("field_patch_size"),
            "execution.field_patch_size",
            nonnegative=True,
        )
        for key in (
            "field_patches_per_observation",
            "field_observations_per_step",
            "field_spectral_samples",
        ):
            _require_integer(value.get(key), f"execution.{key}", positive=True)
        if "field_render_tile" not in value:
            raise ValueError("map plan execution.field_render_tile is required")
        if value["field_render_tile"] is not None:
            _require_integer(
                value["field_render_tile"],
                "execution.field_render_tile",
                positive=True,
            )
        _require_choice(
            value.get("field_spatial_sampling"),
            "execution.field_spatial_sampling",
            _PLAN_FIELD_SAMPLING,
        )
        _require_choice(
            value.get("field_amp_dtype"),
            "execution.field_amp_dtype",
            _PLAN_AMP_DTYPES,
        )
    elif method == "coadd":
        required = (
            "workflow",
            "kernel",
            "stack",
            "background_mode",
            "drizzle_pixfrac",
            "value_semantics",
        )
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(
                f"map plan coadd execution is missing {', '.join(missing)}"
            )
        if value.get("stack") == "sigma_clipped" and value.get(
            "background_mode"
        ) == "overlap":
            raise ValueError(
                "map plan sigma_clipped execution cannot use overlap background"
            )


def _validate_runtime(runtime: Mapping[str, Any]) -> None:
    value = _require_mapping(runtime, "runtime")
    _require_choice(value.get("dtype"), "runtime.dtype", _PLAN_DTYPES)
    device = value.get("device")
    if not isinstance(device, str) or not device:
        raise ValueError("map plan runtime device must be a non-empty string")
    device_parts = device.split(":")
    if device_parts[0] not in ("cpu", "mps", "cuda") or len(device_parts) > 2:
        raise ValueError("map plan runtime device must be cpu, mps, or cuda")
    if len(device_parts) == 2 and (
        device_parts[0] != "cuda"
        or not device_parts[1].isdigit()
    ):
        raise ValueError("map plan runtime device index is invalid")
    if "amp_dtype" in value:
        _require_choice(value["amp_dtype"], "runtime.amp_dtype", _PLAN_AMP_DTYPES)
    seeds = value.get("seeds")
    seeds_mapping = _require_mapping(seeds, "runtime.seeds")
    for key, seed in seeds_mapping.items():
        _require_string(key, "runtime.seeds key")
        _require_integer(seed, f"runtime.seeds[{key!r}]", nonnegative=True)
    if "requested_device" in value:
        _require_string(value["requested_device"], "runtime.requested_device", allow_none=True)
    if "scratch" in value:
        _require_string(value["scratch"], "runtime.scratch", allow_none=True)


def _validate_requested_products(products: Any) -> None:
    if not isinstance(products, list) or not products:
        raise ValueError("map plan requires requested_products")
    if any(not isinstance(product, str) or product not in _PLAN_PRODUCTS for product in products):
        raise ValueError(
            f"map plan requested_products must use only {_PLAN_PRODUCTS}"
        )
    if len(set(products)) != len(products):
        raise ValueError("map plan requested_products must not contain duplicates")
    if "SCI" not in products:
        raise ValueError("map plan requested_products must include SCI")


def _validate_spectral(spectral: Any, *, inputs: list[dict[str, Any]]) -> None:
    value = _require_mapping(spectral, "spectral")
    bands = value.get("bands")
    observation_bands = value.get("observation_bands")
    if not isinstance(bands, list) or not bands:
        raise ValueError("map plan spectral.bands must be a non-empty list")
    if not isinstance(observation_bands, list) or len(observation_bands) != len(inputs):
        raise ValueError("map plan spectral.observation_bands must match inputs")
    labels: set[str] = set()
    for entry in bands:
        profile = _require_mapping(entry, "spectral.bands[]")
        _require_string(profile.get("label"), "spectral.bands[].label")
        label = profile["label"]
        if label in labels:
            raise ValueError("map plan spectral band labels must be unique")
        labels.add(label)
        for key in (
            "wavelength",
            "spectral_min",
            "spectral_max",
            "transmission_center",
            "transmission_sigma",
        ):
            if profile.get(key) is not None:
                _require_number(profile[key], f"spectral.bands[].{key}")
        sample_wavelength = profile.get("transmission_wavelength")
        sample_throughput = profile.get("transmission_throughput")
        if (sample_wavelength is None) != (sample_throughput is None):
            raise ValueError(
                "map plan tabulated passbands require wavelength and throughput"
            )
        if sample_wavelength is not None:
            if (
                not isinstance(sample_wavelength, list)
                or not isinstance(sample_throughput, list)
                or len(sample_wavelength) != len(sample_throughput)
                or len(sample_wavelength) < 2
            ):
                raise ValueError(
                    "map plan tabulated passband samples must have matching lengths"
                )
            for index, sample in enumerate(sample_wavelength):
                _require_number(sample, f"spectral.bands[].transmission_wavelength[{index}]")
            for index, sample in enumerate(sample_throughput):
                _require_number(
                    sample,
                    f"spectral.bands[].transmission_throughput[{index}]",
                    nonnegative=True,
                )
            if any(
                right <= left
                for left, right in zip(sample_wavelength, sample_wavelength[1:])
            ):
                raise ValueError(
                    "map plan tabulated passband wavelengths must be strictly increasing"
                )
            if not any(float(sample) > 0.0 for sample in sample_throughput):
                raise ValueError("map plan tabulated passband integral must be positive")
    if any(not isinstance(label, str) or label not in labels for label in observation_bands):
        raise ValueError("map plan observation_bands references an unknown band")
    for item, band in zip(inputs, observation_bands):
        if item.get("band") != band:
            raise ValueError("map plan input band must match spectral observation_bands")
    _validate_optional_identity_pair(
        value,
        "transmission_path",
        "transmission_identity",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checksum_cards(path: Path, hdu: int | str | None) -> dict[str, str]:
    if hdu is None:
        return {}
    try:
        if isinstance(hdu, str) and not hdu.isdigit():
            return {}
        header = dict(get_header(str(path), 0 if hdu is None else hdu))
    except Exception:
        return {}
    return {
        key: str(header[key]).strip()
        for key in ("CHECKSUM", "DATASUM")
        if header.get(key) not in (None, "")
    }


def build_map_input_identity(
    path: str | Path,
    *,
    hdu: int | str | None = 0,
) -> dict[str, Any]:
    """Build a replay identity without parsing FITS inside torchsky."""
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    cards = _checksum_cards(resolved, hdu)
    identity: dict[str, Any] = {
        "path": str(resolved),
        "size": int(stat.st_size),
        "hdu": hdu,
    }
    if cards:
        identity["fits_checksum_cards"] = cards
    else:
        identity["sha256"] = _sha256(resolved)
    return identity


def verify_map_input_identity(identity: Mapping[str, Any]) -> None:
    """Reject a missing, replaced, or modified input recorded by a plan."""
    path_value = identity.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("map input identity requires a path")
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"map plan input is missing: {path}")
    expected_size = identity.get("size")
    if not isinstance(expected_size, int) or path.stat().st_size != expected_size:
        raise ValueError(f"map plan input size changed: {path}")
    hdu = identity.get("hdu", 0)
    expected_cards = identity.get("fits_checksum_cards")
    if expected_cards is not None:
        if not isinstance(expected_cards, Mapping) or not expected_cards:
            raise ValueError("fits_checksum_cards must be a non-empty mapping")
        current_cards = _checksum_cards(path, hdu)
        normalized = {str(key): str(value) for key, value in expected_cards.items()}
        if current_cards != normalized:
            raise ValueError(f"map plan FITS checksum cards changed: {path}")
        status = verify_checksums(str(path), int(hdu or 0))
        if not bool(status.get("ok", False)):
            raise ValueError(f"map plan FITS checksum validation failed: {path}")
        return
    expected_sha256 = identity.get("sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("map input identity requires checksum cards or SHA-256")
    if _sha256(path) != expected_sha256:
        raise ValueError(f"map plan input content changed: {path}")


@dataclass(frozen=True)
class MapPlanV2:
    """A minimally validated immutable snapshot of a map plan-v2 payload."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        try:
            copied = json.loads(json.dumps(self.payload, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("map plan must be finite JSON object data") from exc
        if not isinstance(copied, dict):
            raise ValueError("map plan must be a finite JSON object")
        if copied.get("schema_version") != 2:
            raise ValueError("map plan schema_version must be 2")
        if copied.get("method") not in ("coadd", "field"):
            raise ValueError("map plan method must be 'coadd' or 'field'")
        inputs = copied.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise ValueError("map plan requires at least one resolved input")
        for item in inputs:
            if not isinstance(item, dict) or not isinstance(item.get("identity"), dict):
                raise ValueError("every map plan input requires an identity")
            _validate_input(item, method=copied["method"])
        _validate_target(copied.get("target"))
        _validate_execution(copied.get("execution"), copied["method"])
        _validate_requested_products(copied.get("requested_products"))
        software = _require_mapping(copied.get("software"), "software")
        for key, version in software.items():
            _require_string(key, "software key")
            _require_string(version, f"software[{key!r}]", allow_none=True)
        _validate_runtime(copied.get("runtime"))
        if "output" in copied:
            _require_string(copied["output"], "output", allow_none=True)
        if copied["method"] == "field":
            _validate_spectral(copied.get("spectral"), inputs=inputs)
        object.__setattr__(self, "payload", copied)

    @classmethod
    def load(cls, path: str | Path) -> MapPlanV2:
        with Path(path).open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("map plan root must be a JSON object")
        return cls(payload)

    def verify_inputs(self) -> None:
        for item in self.payload["inputs"]:
            verify_map_input_identity(item["identity"])
            for key in ("weight_identity", "mask_identity", "primary_beam_identity"):
                identity = item.get(key)
                if identity is not None:
                    verify_map_input_identity(identity)
        spectral = self.payload.get("spectral")
        if isinstance(spectral, dict) and spectral.get("transmission_identity"):
            verify_map_input_identity(spectral["transmission_identity"])
        execution = self.payload.get("execution")
        if isinstance(execution, dict):
            artifact_identities = execution.get("artifact_identities")
            if artifact_identities is not None:
                if not isinstance(artifact_identities, list):
                    raise ValueError("artifact_identities must be a list")
                for identity in artifact_identities:
                    if not isinstance(identity, dict):
                        raise ValueError("every artifact identity must be a mapping")
                    verify_map_input_identity(identity)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.payload))


def load_map_plan(path: str | Path, *, verify_inputs: bool = True) -> MapPlanV2:
    plan = MapPlanV2.load(path)
    if verify_inputs:
        plan.verify_inputs()
    return plan


__all__ = [
    "MapPlanV2",
    "build_map_input_identity",
    "load_map_plan",
    "verify_map_input_identity",
]
