"""Pinned astrometry.net bootstrap for caller-supplied centroid lists."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import Tensor
from torchfits import read_header
from torchfits.table import read_torch, write as write_table

from .astrometry import (
    AstrometricMatchGraph,
    AstrometricObservation,
    AstrometryConfig,
    AstrometrySolution,
    solve_astrometry,
)
from .core import WCS


class AstrometryNetError(RuntimeError):
    pass


class AstrometryNetUnavailableError(AstrometryNetError):
    pass


class AstrometryNetIdentityError(AstrometryNetError):
    pass


class AstrometryNetExecutionError(AstrometryNetError):
    pass


class AstrometryNetHypothesisRejected(AstrometryNetError):
    pass


@dataclass(frozen=True)
class AstrometryNetConfig:
    executable: str | Path
    executable_sha256: str
    executable_version: str
    index_manifest: str | Path
    index_manifest_sha256: str
    timeout_seconds: float = 60.0
    version_args: tuple[str, ...] = ("--version",)

    def __post_init__(self) -> None:
        if not _is_sha256(self.executable_sha256):
            raise ValueError("executable_sha256 must be a lowercase SHA-256 digest")
        if not _is_sha256(self.index_manifest_sha256):
            raise ValueError("index_manifest_sha256 must be a lowercase SHA-256 digest")
        if not self.executable_version.strip():
            raise ValueError("executable_version must be non-empty")
        if not self.version_args:
            raise ValueError("version_args must be non-empty")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be finite and positive")


@dataclass(frozen=True)
class AstrometryNetVerificationConfig:
    min_matches: int
    max_rms_pixels: float
    max_residual_pixels: float
    reference_sigma_arcsec: float = 0.01

    def __post_init__(self) -> None:
        if self.min_matches < 1:
            raise ValueError("min_matches must be positive")
        for name in (
            "max_rms_pixels",
            "max_residual_pixels",
            "reference_sigma_arcsec",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class AstrometryNetIndexIdentity:
    index_id: str
    version: str
    manifest_path: str
    manifest_sha256: str
    backend_config_path: str
    backend_config_sha256: str
    astrometry_index_ids: tuple[int, ...]
    files: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class AstrometryNetHypothesis:
    header: Mapping[str, Any]
    center_ra_deg: float
    center_dec_deg: float
    pixel_scale_arcsec: tuple[float, float]
    orientation_deg: float
    determinant_deg2_per_pixel2: float
    handedness: Literal["right", "left"]
    parity: Literal["normal", "flipped"]
    index_id: int
    index_healpix: int
    index_hpnside: int
    log_odds: float
    matched_sources: int

    @property
    def wcs(self) -> WCS:
        return WCS(dict(self.header))


@dataclass(frozen=True)
class AstrometryNetVerification:
    supplied_count: int
    matched_count: int
    residuals_pixels: Tensor
    rms_pixels: float
    max_pixels: float


@dataclass(frozen=True)
class AstrometryNetProvenance:
    executable_path: str
    executable_sha256: str
    executable_version: str
    index: AstrometryNetIndexIdentity
    bootstrap_arguments: tuple[str, ...]
    verification_arguments: tuple[str, ...]
    bootstrap_input_count: int
    verification_input_count: int
    runtime_downloads: bool = False


@dataclass(frozen=True)
class AstrometryNetBootstrapResult:
    hypothesis: AstrometryNetHypothesis
    verification: AstrometryNetVerification
    bundle_solution: AstrometrySolution
    provenance: AstrometryNetProvenance


@dataclass(frozen=True)
class _Correspondences:
    original_rows: Tensor
    detector_xy: Tensor
    reference_sky: Tensor
    residuals_pixels: Tensor


class _OriginWCS:
    def __init__(self, wcs: WCS, origin: int) -> None:
        self.wcs = wcs
        self.origin = origin

    def pixel_to_world(self, x: Any, y: Any) -> tuple[Tensor, Tensor]:
        result = self.wcs.pixel_to_world(x, y, origin=self.origin)
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("two-dimensional WCS must return longitude and latitude")
        return result


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise AstrometryNetUnavailableError(f"{description} is absent: {resolved}")
    return resolved


def _resolve_executable(config: AstrometryNetConfig) -> Path:
    raw = Path(config.executable).expanduser()
    candidate = (
        str(raw)
        if raw.is_absolute() or raw.parent != Path(".")
        else shutil.which(str(raw))
    )
    if candidate is None:
        raise AstrometryNetUnavailableError(
            f"astrometry.net executable is absent: {config.executable}"
        )
    executable = _require_file(Path(candidate), "astrometry.net executable")
    if _sha256(executable) != config.executable_sha256:
        raise AstrometryNetIdentityError(
            "astrometry.net executable SHA-256 does not match"
        )
    return executable


def _relative_manifest_path(manifest: Path, value: Any, description: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise AstrometryNetIdentityError(f"index manifest {description} path is absent")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest.parent / path
    return _require_file(path, f"index manifest {description}")


def _load_index_identity(config: AstrometryNetConfig) -> AstrometryNetIndexIdentity:
    manifest = _require_file(
        Path(config.index_manifest), "astrometry.net index manifest"
    )
    digest = _sha256(manifest)
    if digest != config.index_manifest_sha256:
        raise AstrometryNetIdentityError(
            "astrometry.net index manifest SHA-256 does not match"
        )
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AstrometryNetIdentityError(
            "astrometry.net index manifest is not valid JSON"
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise AstrometryNetIdentityError(
            "astrometry.net index manifest schema_version must be 1"
        )
    index_id = payload.get("id")
    version = payload.get("version")
    if not isinstance(index_id, str) or not index_id.strip():
        raise AstrometryNetIdentityError("astrometry.net index manifest id is absent")
    if not isinstance(version, str) or not version.strip():
        raise AstrometryNetIdentityError(
            "astrometry.net index manifest version is absent"
        )
    backend = payload.get("backend_config")
    if not isinstance(backend, dict) or not _is_sha256(str(backend.get("sha256", ""))):
        raise AstrometryNetIdentityError(
            "index manifest backend_config identity is incomplete"
        )
    backend_path = _relative_manifest_path(
        manifest, backend.get("path"), "backend config"
    )
    backend_sha = str(backend["sha256"])
    if _sha256(backend_path) != backend_sha:
        raise AstrometryNetIdentityError(
            "astrometry.net backend config SHA-256 does not match"
        )
    raw_ids = payload.get("astrometry_index_ids")
    if (
        not isinstance(raw_ids, list)
        or not raw_ids
        or any(
            not isinstance(value, int) or isinstance(value, bool) for value in raw_ids
        )
    ):
        raise AstrometryNetIdentityError(
            "index manifest astrometry_index_ids must be a non-empty integer list"
        )
    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise AstrometryNetIdentityError("index manifest files must be non-empty")
    files: list[tuple[str, str]] = []
    for row in raw_files:
        if not isinstance(row, dict) or not _is_sha256(str(row.get("sha256", ""))):
            raise AstrometryNetIdentityError(
                "index manifest file identity is incomplete"
            )
        path = _relative_manifest_path(manifest, row.get("path"), "index file")
        file_digest = str(row["sha256"])
        if _sha256(path) != file_digest:
            raise AstrometryNetIdentityError(
                f"astrometry.net index SHA-256 does not match: {path}"
            )
        files.append((str(path), file_digest))
    return AstrometryNetIndexIdentity(
        index_id=index_id,
        version=version,
        manifest_path=str(manifest),
        manifest_sha256=digest,
        backend_config_path=str(backend_path),
        backend_config_sha256=backend_sha,
        astrometry_index_ids=tuple(raw_ids),
        files=tuple(files),
    )


def _run(
    command: Sequence[str], *, timeout: float, purpose: str
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise AstrometryNetExecutionError(
            f"astrometry.net {purpose} timed out after {timeout:g} seconds"
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-4000:]
        raise AstrometryNetExecutionError(
            f"astrometry.net {purpose} failed with exit code {completed.returncode}: {detail}"
        )
    return completed


def _read_columns(path: Path) -> dict[str, Tensor]:
    table = read_torch(str(path), hdu=1)
    if not isinstance(table, Mapping):
        raise AstrometryNetExecutionError(
            f"astrometry.net output is not a table: {path.name}"
        )
    return {str(name).lower(): torch.as_tensor(value) for name, value in table.items()}


def _column(columns: Mapping[str, Tensor], name: str, path: Path) -> Tensor:
    try:
        return columns[name.lower()]
    except KeyError as exc:
        raise AstrometryNetExecutionError(
            f"astrometry.net output {path.name} lacks column {name}"
        ) from exc


def _scalar(columns: Mapping[str, Tensor], name: str, path: Path) -> float:
    value = _column(columns, name, path).reshape(-1)
    if value.numel() == 0:
        raise AstrometryNetExecutionError(
            f"astrometry.net output {path.name} has no rows"
        )
    return float(value[0].item())


def _parse_hypothesis(wcs_path: Path, match_path: Path) -> AstrometryNetHypothesis:
    header = dict(read_header(str(wcs_path), hdu=0))
    required = (
        "CRVAL1",
        "CRVAL2",
        "CRPIX1",
        "CRPIX2",
        "CD1_1",
        "CD1_2",
        "CD2_1",
        "CD2_2",
        "CTYPE1",
        "CTYPE2",
    )
    if any(key not in header for key in required):
        raise AstrometryNetExecutionError("astrometry.net WCS hypothesis is incomplete")
    cd = torch.tensor(
        [
            [float(header["CD1_1"]), float(header["CD1_2"])],
            [float(header["CD2_1"]), float(header["CD2_2"])],
        ],
        dtype=torch.float64,
    )
    determinant = float(torch.linalg.det(cd).item())
    if not math.isfinite(determinant) or determinant == 0.0:
        raise AstrometryNetExecutionError("astrometry.net WCS hypothesis is singular")
    scales = torch.linalg.vector_norm(cd, dim=0) * 3600.0
    columns = _read_columns(match_path)
    parity_value = bool(round(_scalar(columns, "PARITY", match_path)))
    return AstrometryNetHypothesis(
        header=header,
        center_ra_deg=float(header["CRVAL1"]),
        center_dec_deg=float(header["CRVAL2"]),
        pixel_scale_arcsec=(float(scales[0]), float(scales[1])),
        orientation_deg=math.degrees(math.atan2(float(cd[1, 0]), float(cd[0, 0]))),
        determinant_deg2_per_pixel2=determinant,
        handedness="right" if determinant > 0.0 else "left",
        parity="flipped" if parity_value else "normal",
        index_id=int(round(_scalar(columns, "INDEXID", match_path))),
        index_healpix=int(round(_scalar(columns, "HEALPIX", match_path))),
        index_hpnside=int(round(_scalar(columns, "HPNSIDE", match_path))),
        log_odds=_scalar(columns, "LOGODDS", match_path),
        matched_sources=int(round(_scalar(columns, "NMATCH", match_path))),
    )


def _same_hypothesis(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    text_keys = ("CTYPE1", "CTYPE2")
    numeric_keys = (
        "CRVAL1",
        "CRVAL2",
        "CRPIX1",
        "CRPIX2",
        "CD1_1",
        "CD1_2",
        "CD2_1",
        "CD2_2",
    )
    return (
        all(str(left.get(key)) == str(right.get(key)) for key in text_keys)
        and all(
            math.isclose(
                float(left[key]), float(right[key]), rel_tol=0.0, abs_tol=1.0e-12
            )
            for key in numeric_keys
            if key in left and key in right
        )
        and all(key in left and key in right for key in numeric_keys)
    )


def _coerce_inputs(
    x: Any,
    y: Any,
    flux: Any | None,
    covariance: Any | None,
) -> tuple[Tensor, Tensor | None, Tensor]:
    x_tensor = torch.as_tensor(x)
    if not x_tensor.is_floating_point() or x_tensor.is_complex() or x_tensor.ndim != 1:
        raise ValueError("x must be a real floating-point tensor with shape (N,)")
    y_tensor = torch.as_tensor(y, dtype=x_tensor.dtype, device=x_tensor.device)
    if y_tensor.shape != x_tensor.shape or y_tensor.is_complex():
        raise ValueError("y must have the same shape and real dtype as x")
    if x_tensor.numel() < 5 or not bool(
        (torch.isfinite(x_tensor) & torch.isfinite(y_tensor))
        .all()
        .detach()
        .cpu()
        .item()
    ):
        raise ValueError("x and y must contain at least five finite centroids")
    points = torch.stack((x_tensor, y_tensor), dim=-1).detach().cpu().to(torch.float64)
    flux_tensor = None
    if flux is not None:
        flux_tensor = torch.as_tensor(
            flux, dtype=x_tensor.dtype, device=x_tensor.device
        )
        if flux_tensor.shape != x_tensor.shape or not bool(
            torch.isfinite(flux_tensor).all()
        ):
            raise ValueError("flux must contain finite values with shape (N,)")
        flux_tensor = flux_tensor.detach().cpu().to(torch.float64)
    if covariance is None:
        covariance_tensor = (
            torch.eye(2, dtype=torch.float64).expand(points.shape[0], 2, 2).clone()
        )
    else:
        covariance_tensor = torch.as_tensor(covariance, dtype=torch.float64)
        if covariance_tensor.shape == (2, 2):
            covariance_tensor = covariance_tensor.expand(points.shape[0], 2, 2).clone()
        if covariance_tensor.shape != (points.shape[0], 2, 2):
            raise ValueError("covariance must have shape (2, 2) or (N, 2, 2)")
        covariance_tensor = covariance_tensor.detach().cpu()
    covariance_tensor = 0.5 * (covariance_tensor + covariance_tensor.mT)
    _, info = torch.linalg.cholesky_ex(covariance_tensor)
    if not bool(torch.isfinite(covariance_tensor).all()) or bool((info != 0).any()):
        raise ValueError("covariance must be finite and positive definite")
    return points, flux_tensor, covariance_tensor


def _ordered_rows(rows: Tensor, flux: Tensor | None) -> Tensor:
    if flux is None:
        return rows
    order = torch.argsort(flux.index_select(0, rows), descending=True, stable=True)
    return rows.index_select(0, order)


def _write_xyls(
    path: Path,
    points: Tensor,
    flux: Tensor | None,
    covariance: Tensor,
    rows: Tensor,
    pixel_origin: int,
) -> Tensor:
    ordered = _ordered_rows(rows, flux)
    selected = points.index_select(0, ordered) + float(1 - pixel_origin)
    selected_covariance = covariance.index_select(0, ordered)
    columns: dict[str, Any] = {
        "X": selected[:, 0],
        "Y": selected[:, 1],
        "ROW_ID": ordered,
        "COV_XX": selected_covariance[:, 0, 0],
        "COV_XY": selected_covariance[:, 0, 1],
        "COV_YY": selected_covariance[:, 1, 1],
    }
    if flux is not None:
        columns["FLUX"] = flux.index_select(0, ordered)
    write_table(str(path), columns, overwrite=True, extname="XYLS")
    return ordered


def _correspondences(
    path: Path,
    ordered_rows: Tensor,
    serialized_points: Tensor,
) -> _Correspondences:
    columns = _read_columns(path)
    field_id = _column(columns, "field_id", path).to(torch.int64).reshape(-1)
    if field_id.numel() == 0:
        return _Correspondences(
            original_rows=torch.empty(0, dtype=torch.int64),
            detector_xy=torch.empty((0, 2), dtype=torch.float64),
            reference_sky=torch.empty((0, 2), dtype=torch.float64),
            residuals_pixels=torch.empty(0, dtype=torch.float64),
        )
    if bool(((field_id < 0) | (field_id >= ordered_rows.numel())).any()):
        raise AstrometryNetExecutionError(
            "astrometry.net correspondence field_id is invalid"
        )
    field_xy = torch.stack(
        (
            _column(columns, "field_x", path).to(torch.float64).reshape(-1),
            _column(columns, "field_y", path).to(torch.float64).reshape(-1),
        ),
        dim=-1,
    )
    index_xy = torch.stack(
        (
            _column(columns, "index_x", path).to(torch.float64).reshape(-1),
            _column(columns, "index_y", path).to(torch.float64).reshape(-1),
        ),
        dim=-1,
    )
    reference_sky = torch.stack(
        (
            _column(columns, "index_ra", path).to(torch.float64).reshape(-1),
            _column(columns, "index_dec", path).to(torch.float64).reshape(-1),
        ),
        dim=-1,
    )
    expected = serialized_points.index_select(0, field_id)
    if not torch.allclose(field_xy, expected, atol=1.0e-7, rtol=0.0):
        raise AstrometryNetExecutionError(
            "astrometry.net correspondence rows do not match supplied centroids"
        )
    return _Correspondences(
        original_rows=ordered_rows.index_select(0, field_id),
        detector_xy=field_xy,
        reference_sky=reference_sky,
        residuals_pixels=torch.linalg.vector_norm(index_xy - field_xy, dim=-1),
    )


def _base_command(
    executable: Path,
    index: AstrometryNetIndexIdentity,
    work: Path,
    width: int,
    height: int,
    scale: tuple[float, float],
) -> list[str]:
    return [
        str(executable),
        "--config",
        index.backend_config_path,
        "--dir",
        str(work),
        "--temp-dir",
        str(work),
        "--no-plots",
        "--overwrite",
        "--no-tweak",
        "--no-remove-lines",
        "--uniformize",
        "0",
        "--x-column",
        "X",
        "--y-column",
        "Y",
        "--width",
        str(width),
        "--height",
        str(height),
        "--scale-units",
        "arcsecperpix",
        "--scale-low",
        f"{scale[0]:.17g}",
        "--scale-high",
        f"{scale[1]:.17g}",
    ]


def _solved(path: Path, purpose: str) -> None:
    if not path.is_file() or not any(path.read_bytes()):
        raise AstrometryNetExecutionError(
            f"astrometry.net {purpose} produced no accepted solution"
        )


def _sanitized(command: Sequence[str], work: Path) -> tuple[str, ...]:
    root = str(work)
    return tuple(value.replace(root, "<temporary>") for value in command[1:])


def solve_with_astrometry_net(
    x: Any,
    y: Any,
    *,
    image_width: int,
    image_height: int,
    scale_arcsec_per_pixel: tuple[float, float],
    verification_mask: Any,
    external: AstrometryNetConfig,
    verification: AstrometryNetVerificationConfig,
    flux: Any | None = None,
    covariance: Any | None = None,
    pixel_origin: Literal[0, 1] = 0,
    bundle_config: AstrometryConfig | None = None,
) -> AstrometryNetBootstrapResult:
    """Bootstrap a supplied centroid list and refine accepted correspondences."""
    points, flux_tensor, covariance_tensor = _coerce_inputs(x, y, flux, covariance)
    if image_width < 1 or image_height < 1:
        raise ValueError("image dimensions must be positive")
    if pixel_origin not in {0, 1}:
        raise ValueError("pixel_origin must be 0 or 1")
    scale = tuple(float(value) for value in scale_arcsec_per_pixel)
    if (
        len(scale) != 2
        or not all(math.isfinite(value) and value > 0.0 for value in scale)
        or scale[0] > scale[1]
    ):
        raise ValueError("scale_arcsec_per_pixel must be finite positive (low, high)")
    mask = torch.as_tensor(verification_mask, dtype=torch.bool).detach().cpu()
    if mask.shape != (points.shape[0],):
        raise ValueError("verification_mask must have shape (N,)")
    bootstrap_rows = torch.nonzero(~mask, as_tuple=False).flatten()
    verification_rows = torch.nonzero(mask, as_tuple=False).flatten()
    if bootstrap_rows.numel() < 4:
        raise ValueError("at least four centroids must remain for bootstrap")
    if verification_rows.numel() < verification.min_matches:
        raise ValueError("verification_mask supplies fewer rows than min_matches")

    executable = _resolve_executable(external)
    index = _load_index_identity(external)
    version_result = _run(
        [str(executable), *external.version_args],
        timeout=min(external.timeout_seconds, 15.0),
        purpose="version probe",
    )
    version_text = "\n".join(
        value.strip()
        for value in (version_result.stdout, version_result.stderr)
        if value.strip()
    )
    version_line = version_text.splitlines()[0] if version_text else ""
    if external.executable_version not in version_line:
        raise AstrometryNetIdentityError(
            f"astrometry.net version {version_line!r} does not match {external.executable_version!r}"
        )

    with tempfile.TemporaryDirectory(prefix="torchsky-astrometry-net-") as temporary:
        work = Path(temporary)
        bootstrap_xyls = work / "bootstrap.xyls"
        verify_xyls = work / "verification.xyls"
        bootstrap_order = _write_xyls(
            bootstrap_xyls,
            points,
            flux_tensor,
            covariance_tensor,
            bootstrap_rows,
            pixel_origin,
        )
        verify_order = _write_xyls(
            verify_xyls,
            points,
            flux_tensor,
            covariance_tensor,
            verification_rows,
            pixel_origin,
        )
        hypothesis_wcs = work / "hypothesis.wcs"
        hypothesis_match = work / "hypothesis.match"
        bootstrap_corr = work / "bootstrap.corr"
        bootstrap_solved = work / "bootstrap.solved"
        bootstrap_command = _base_command(
            executable, index, work, image_width, image_height, scale
        ) + [
            "--wcs",
            str(hypothesis_wcs),
            "--match",
            str(hypothesis_match),
            "--corr",
            str(bootstrap_corr),
            "--solved",
            str(bootstrap_solved),
            str(bootstrap_xyls),
        ]
        _run(
            bootstrap_command,
            timeout=external.timeout_seconds,
            purpose="bootstrap",
        )
        _solved(bootstrap_solved, "bootstrap")
        hypothesis = _parse_hypothesis(hypothesis_wcs, hypothesis_match)
        if hypothesis.index_id not in index.astrometry_index_ids:
            raise AstrometryNetIdentityError(
                f"solution index id {hypothesis.index_id} is absent from the pinned manifest"
            )

        verify_wcs = work / "verification.wcs"
        verify_match = work / "verification.match"
        verify_corr = work / "verification.corr"
        verify_solved = work / "verification.solved"
        verification_command = _base_command(
            executable, index, work, image_width, image_height, scale
        ) + [
            "--verify",
            str(hypothesis_wcs),
            "--wcs",
            str(verify_wcs),
            "--match",
            str(verify_match),
            "--corr",
            str(verify_corr),
            "--solved",
            str(verify_solved),
            str(verify_xyls),
        ]
        _run(
            verification_command,
            timeout=external.timeout_seconds,
            purpose="held-out verification",
        )
        _solved(verify_solved, "held-out verification")
        verification_header = dict(read_header(str(verify_wcs), hdu=0))
        if not _same_hypothesis(hypothesis.header, verification_header):
            raise AstrometryNetHypothesisRejected(
                "held-out verification replaced the bootstrap hypothesis"
            )
        serialized_verify = points.index_select(0, verify_order) + float(
            1 - pixel_origin
        )
        heldout = _correspondences(verify_corr, verify_order, serialized_verify)
        matched_count = int(heldout.residuals_pixels.numel())
        if matched_count < verification.min_matches:
            raise AstrometryNetHypothesisRejected(
                f"held-out hypothesis matched {matched_count} centroids; "
                f"required {verification.min_matches}"
            )
        rms = float(torch.sqrt(torch.mean(heldout.residuals_pixels.square())).item())
        maximum = float(heldout.residuals_pixels.max().item())
        if (
            rms > verification.max_rms_pixels
            or maximum > verification.max_residual_pixels
        ):
            raise AstrometryNetHypothesisRejected(
                f"held-out residuals exceed limits: rms={rms:.8g} pixels, "
                f"max={maximum:.8g} pixels"
            )
        verification_result = AstrometryNetVerification(
            supplied_count=int(verification_rows.numel()),
            matched_count=matched_count,
            residuals_pixels=heldout.residuals_pixels,
            rms_pixels=rms,
            max_pixels=maximum,
        )

        serialized_bootstrap = points.index_select(0, bootstrap_order) + float(
            1 - pixel_origin
        )
        bootstrap = _correspondences(
            bootstrap_corr, bootstrap_order, serialized_bootstrap
        )
        original_rows = torch.cat((bootstrap.original_rows, heldout.original_rows))
        reference_sky = torch.cat((bootstrap.reference_sky, heldout.reference_sky))
        if original_rows.numel() < verification.min_matches:
            raise AstrometryNetExecutionError(
                "astrometry.net returned too few correspondences for bundle refinement"
            )
        source_ids = torch.arange(original_rows.numel(), dtype=torch.int64)
        native_wcs = WCS(dict(hypothesis.header))
        observation = AstrometricObservation(
            "astrometry-net-bootstrap",
            points.index_select(0, original_rows),
            source_ids,
            covariance_tensor.index_select(0, original_rows),
            approximate_wcs=_OriginWCS(native_wcs, pixel_origin),
            covariance_space="detector",
        )
        reference_covariance = (
            torch.eye(2, dtype=torch.float64) * verification.reference_sigma_arcsec**2
        )
        graph = AstrometricMatchGraph(
            [observation],
            reference_source_ids=source_ids,
            reference_positions=reference_sky,
            reference_covariance=reference_covariance,
        )
        bundle_solution = solve_astrometry(
            graph,
            AstrometryConfig(pose_model="rscale")
            if bundle_config is None
            else bundle_config,
        )
        provenance = AstrometryNetProvenance(
            executable_path=str(executable),
            executable_sha256=external.executable_sha256,
            executable_version=version_line,
            index=index,
            bootstrap_arguments=_sanitized(bootstrap_command, work),
            verification_arguments=_sanitized(verification_command, work),
            bootstrap_input_count=int(bootstrap_rows.numel()),
            verification_input_count=int(verification_rows.numel()),
        )
        return AstrometryNetBootstrapResult(
            hypothesis=hypothesis,
            verification=verification_result,
            bundle_solution=bundle_solution,
            provenance=provenance,
        )


__all__ = [
    "AstrometryNetBootstrapResult",
    "AstrometryNetConfig",
    "AstrometryNetError",
    "AstrometryNetExecutionError",
    "AstrometryNetHypothesis",
    "AstrometryNetHypothesisRejected",
    "AstrometryNetIdentityError",
    "AstrometryNetIndexIdentity",
    "AstrometryNetProvenance",
    "AstrometryNetUnavailableError",
    "AstrometryNetVerification",
    "AstrometryNetVerificationConfig",
    "solve_with_astrometry_net",
]
