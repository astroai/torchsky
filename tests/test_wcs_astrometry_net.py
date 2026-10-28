from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import pytest
import torch
import torchfits
from torchfits.table import read_torch, write as write_table

from torchsky.wcs import (
    AstrometryNetConfig,
    AstrometryNetExecutionError,
    AstrometryNetHypothesisRejected,
    AstrometryNetIdentityError,
    AstrometryNetUnavailableError,
    AstrometryNetVerificationConfig,
    WCS,
    solve_with_astrometry_net,
)


FAKE_EXECUTABLE = r"""#!/usr/bin/env python3
import json
import os
from pathlib import Path
import shutil
import sys
import time

args = sys.argv[1:]
if "--version" in args:
    print("solve-field 0.97-fake")
    raise SystemExit(0)

if os.environ.get("FAKE_SLEEP"):
    time.sleep(float(os.environ["FAKE_SLEEP"]))
if os.environ.get("FAKE_EXIT"):
    print("fake solver error", file=sys.stderr)
    raise SystemExit(int(os.environ["FAKE_EXIT"]))

def value(name):
    return args[args.index(name) + 1]

input_path = Path(args[-1])
is_verify = "--verify" in args
prefix = "FAKE_VERIFY" if is_verify else "FAKE_BOOTSTRAP"
shutil.copyfile(os.environ[prefix + "_WCS"], value("--wcs"))
shutil.copyfile(os.environ[prefix + "_CORR"], value("--corr"))
shutil.copyfile(os.environ[prefix + "_MATCH"], value("--match"))
Path(value("--solved")).write_bytes(b"\0" if os.environ.get("FAKE_UNSOLVED") else b"\1")
capture = Path(os.environ["FAKE_CAPTURE_DIR"]) / ("verification.xyls" if is_verify else "bootstrap.xyls")
shutil.copyfile(input_path, capture)
log_path = os.environ.get("FAKE_LOG")
if log_path:
    record = {
        "args": args,
        "verify": is_verify,
    }
    with Path(log_path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _header(cd: tuple[tuple[float, float], tuple[float, float]]) -> dict[str, object]:
    return {
        "WCSAXES": 2,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CRVAL1": 123.4,
        "CRVAL2": -22.5,
        "CRPIX1": 50.5,
        "CRPIX2": 40.5,
        "CD1_1": cd[0][0],
        "CD1_2": cd[0][1],
        "CD2_1": cd[1][0],
        "CD2_2": cd[1][1],
    }


def _external(tmp_path: Path, *, timeout: float = 30.0, version: str = "0.97") -> AstrometryNetConfig:
    executable = tmp_path / "solve-field"
    executable.write_text(FAKE_EXECUTABLE, encoding="utf-8")
    executable.chmod(0o755)
    index = tmp_path / "index-42.fits"
    index.write_bytes(b"pinned-index")
    backend = tmp_path / "astrometry.cfg"
    backend.write_text(f"index {index}\n", encoding="utf-8")
    manifest = tmp_path / "index-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "fake-gaia-index",
                "version": "2026.10",
                "astrometry_index_ids": [42],
                "backend_config": {
                    "path": backend.name,
                    "sha256": _sha256(backend),
                },
                "files": [{"path": index.name, "sha256": _sha256(index)}],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return AstrometryNetConfig(
        executable=executable,
        executable_sha256=_sha256(executable),
        executable_version=version,
        index_manifest=manifest,
        index_manifest_sha256=_sha256(manifest),
        timeout_seconds=timeout,
    )


def _inputs(count: int = 16) -> tuple[torch.Tensor, ...]:
    x = torch.linspace(8.0, 91.0, count, dtype=torch.float64)
    y = torch.remainder(torch.arange(count, dtype=torch.float64) * 17.0, 70.0) + 5.0
    flux = torch.linspace(100.0, 10.0, count, dtype=torch.float64)
    covariance = torch.eye(2, dtype=torch.float64).expand(count, 2, 2).clone() * 0.04
    mask = torch.zeros(count, dtype=torch.bool)
    mask[::4] = True
    return x, y, flux, covariance, mask


def _prepare_fake_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    header: dict[str, object],
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    capture = tmp_path / "captured"
    capture.mkdir(exist_ok=True)
    monkeypatch.setenv("FAKE_CAPTURE_DIR", str(capture))
    wcs = WCS(header)
    for label, rows in (
        ("BOOTSTRAP", torch.nonzero(~mask, as_tuple=False).flatten()),
        ("VERIFY", torch.nonzero(mask, as_tuple=False).flatten()),
    ):
        selected_x = x.index_select(0, rows) + 1.0
        selected_y = y.index_select(0, rows) + 1.0
        count = selected_x.numel()
        if label == "VERIFY" and os.environ.get("FAKE_VERIFY_MATCHES"):
            count = min(count, int(os.environ["FAKE_VERIFY_MATCHES"]))
        selected_x = selected_x[:count]
        selected_y = selected_y[:count]
        ra, dec = wcs.pixel_to_world(selected_x, selected_y, origin=1)
        residual = (
            float(os.environ.get("FAKE_VERIFY_RESIDUAL", "0"))
            if label == "VERIFY"
            else 0.0
        )
        output_header = dict(header)
        if label == "VERIFY" and os.environ.get("FAKE_REPLACE_HYPOTHESIS"):
            output_header["CRVAL1"] = float(output_header["CRVAL1"]) + 1.0
        wcs_path = tmp_path / f"{label.lower()}.wcs"
        corr_path = tmp_path / f"{label.lower()}.corr"
        match_path = tmp_path / f"{label.lower()}.match"
        torchfits.write(
            str(wcs_path),
            torch.zeros((2, 2), dtype=torch.float32),
            header=output_header,
            overwrite=True,
        )
        write_table(
            str(corr_path),
            {
                "field_x": selected_x,
                "field_y": selected_y,
                "field_ra": ra,
                "field_dec": dec,
                "index_x": selected_x + residual,
                "index_y": selected_y,
                "index_ra": ra,
                "index_dec": dec,
                "index_id": torch.arange(count, dtype=torch.int32),
                "field_id": torch.arange(count, dtype=torch.int32),
                "match_weight": torch.ones(count, dtype=torch.float64),
            },
            overwrite=True,
        )
        write_table(
            str(match_path),
            {
                "INDEXID": torch.tensor([42], dtype=torch.int32),
                "HEALPIX": torch.tensor([7], dtype=torch.int32),
                "HPNSIDE": torch.tensor([8], dtype=torch.int32),
                "LOGODDS": torch.tensor([30.0], dtype=torch.float64),
                "NMATCH": torch.tensor([count], dtype=torch.int32),
                "PARITY": torch.tensor(
                    [int(os.environ.get("FAKE_PARITY", "0"))], dtype=torch.int32
                ),
            },
            overwrite=True,
        )
        monkeypatch.setenv(f"FAKE_{label}_WCS", str(wcs_path))
        monkeypatch.setenv(f"FAKE_{label}_CORR", str(corr_path))
        monkeypatch.setenv(f"FAKE_{label}_MATCH", str(match_path))


def _solve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    cd: tuple[tuple[float, float], tuple[float, float]] = ((-0.0001, 0.0), (0.0, 0.0001)),
    external: AstrometryNetConfig | None = None,
):
    header = _header(cd)
    x, y, flux, covariance, mask = _inputs()
    _prepare_fake_outputs(tmp_path, monkeypatch, header, x, y, mask)
    return solve_with_astrometry_net(
        x,
        y,
        image_width=100,
        image_height=80,
        scale_arcsec_per_pixel=(0.3, 0.5),
        verification_mask=mask,
        external=_external(tmp_path) if external is None else external,
        verification=AstrometryNetVerificationConfig(
            min_matches=4,
            max_rms_pixels=0.5,
            max_residual_pixels=1.0,
        ),
        flux=flux,
        covariance=covariance,
    )


def test_fake_executable_contract_bundle_refinement_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "commands.jsonl"
    monkeypatch.setenv("FAKE_LOG", str(log))

    result = _solve(tmp_path, monkeypatch)

    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    bootstrap_table = read_torch(str(tmp_path / "captured" / "bootstrap.xyls"), hdu=1)
    assert len(records) == 2
    assert records[0]["verify"] is False
    assert records[1]["verify"] is True
    assert set(bootstrap_table) >= {
        "X",
        "Y",
        "FLUX",
        "ROW_ID",
        "COV_XX",
        "COV_XY",
        "COV_YY",
    }
    assert torch.as_tensor(bootstrap_table["X"]).tolist() == sorted(
        torch.as_tensor(bootstrap_table["X"]).tolist()
    )
    for record in records:
        assert "--x-column" in record["args"]
        assert "--y-column" in record["args"]
        assert "--width" in record["args"]
        assert "--height" in record["args"]
        assert "--scale-units" in record["args"]
        assert "--no-tweak" in record["args"]
        assert "--use-source-extractor" not in record["args"]
    assert result.verification.matched_count == 4
    assert result.verification.max_pixels == pytest.approx(0.0)
    assert result.bundle_solution.converged
    assert torch.linalg.vector_norm(result.bundle_solution.pose_parameters) < 1.0e-5
    assert result.provenance.executable_version == "solve-field 0.97-fake"
    assert result.provenance.index.index_id == "fake-gaia-index"
    assert result.provenance.index.astrometry_index_ids == (42,)
    assert result.provenance.runtime_downloads is False
    assert all("<temporary>" in value or "torchsky-astrometry-net" not in value for value in result.provenance.bootstrap_arguments)


def test_flux_and_covariance_are_optional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    header = _header(((-0.0001, 0.0), (0.0, 0.0001)))
    x, y, _, _, mask = _inputs()
    _prepare_fake_outputs(tmp_path, monkeypatch, header, x, y, mask)

    result = solve_with_astrometry_net(
        x,
        y,
        image_width=100,
        image_height=80,
        scale_arcsec_per_pixel=(0.3, 0.5),
        verification_mask=mask,
        external=_external(tmp_path),
        verification=AstrometryNetVerificationConfig(4, 0.5, 1.0),
    )

    table = read_torch(str(tmp_path / "captured" / "bootstrap.xyls"), hdu=1)
    assert "FLUX" not in table
    torch.testing.assert_close(
        torch.as_tensor(table["COV_XX"]),
        torch.ones(12, dtype=torch.float64),
    )
    assert result.verification.matched_count == 4


@pytest.mark.parametrize(
    ("cd", "parity", "handedness"),
    [
        (((-1.0e-4, 0.0), (0.0, 2.0e-4)), "normal", "left"),
        (((0.0, -2.0e-4), (1.0e-4, 0.0)), "flipped", "right"),
    ],
)
def test_hypothesis_scale_orientation_handedness_and_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cd: tuple[tuple[float, float], tuple[float, float]],
    parity: str,
    handedness: str,
) -> None:
    monkeypatch.setenv("FAKE_PARITY", "1" if parity == "flipped" else "0")

    result = _solve(tmp_path, monkeypatch, cd=cd)

    matrix = torch.tensor(cd, dtype=torch.float64)
    scales = torch.linalg.vector_norm(matrix, dim=0) * 3600.0
    hypothesis = result.hypothesis
    assert hypothesis.pixel_scale_arcsec == pytest.approx(tuple(scales.tolist()))
    assert hypothesis.orientation_deg == pytest.approx(
        math.degrees(math.atan2(cd[1][0], cd[0][0]))
    )
    assert hypothesis.determinant_deg2_per_pixel2 == pytest.approx(
        torch.linalg.det(matrix).item()
    )
    assert hypothesis.handedness == handedness
    assert hypothesis.parity == parity
    assert hypothesis.index_id == 42


def test_heldout_residual_rejects_false_solution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_VERIFY_RESIDUAL", "3.0")

    with pytest.raises(AstrometryNetHypothesisRejected, match="residuals exceed"):
        _solve(tmp_path, monkeypatch)


def test_heldout_verification_rejects_replacement_solution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_REPLACE_HYPOTHESIS", "1")

    with pytest.raises(AstrometryNetHypothesisRejected, match="replaced"):
        _solve(tmp_path, monkeypatch)


@pytest.mark.parametrize(("environment", "message"), [("FAKE_EXIT", "exit code 7"), ("FAKE_UNSOLVED", "no accepted solution")])
def test_external_error_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment: str,
    message: str,
) -> None:
    monkeypatch.setenv(environment, "7" if environment == "FAKE_EXIT" else "1")

    with pytest.raises(AstrometryNetExecutionError, match=message):
        _solve(tmp_path, monkeypatch)


def test_external_timeout_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    external = _external(tmp_path, timeout=0.05)
    monkeypatch.setenv("FAKE_SLEEP", "0.2")

    with pytest.raises(AstrometryNetExecutionError, match="timed out"):
        _solve(tmp_path, monkeypatch, external=external)


def test_binary_and_index_identity_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    external = _external(tmp_path)
    missing = AstrometryNetConfig(
        executable=tmp_path / "missing-solve-field",
        executable_sha256="0" * 64,
        executable_version="0.97",
        index_manifest=external.index_manifest,
        index_manifest_sha256=external.index_manifest_sha256,
    )
    x, y, flux, covariance, mask = _inputs()
    kwargs = {
        "image_width": 100,
        "image_height": 80,
        "scale_arcsec_per_pixel": (0.3, 0.5),
        "verification_mask": mask,
        "verification": AstrometryNetVerificationConfig(4, 0.5, 1.0),
        "flux": flux,
        "covariance": covariance,
    }
    with pytest.raises(AstrometryNetUnavailableError, match="executable is absent"):
        solve_with_astrometry_net(x, y, external=missing, **kwargs)

    manifest = Path(external.index_manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    del payload["astrometry_index_ids"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    incomplete = AstrometryNetConfig(
        executable=external.executable,
        executable_sha256=external.executable_sha256,
        executable_version=external.executable_version,
        index_manifest=manifest,
        index_manifest_sha256=_sha256(manifest),
    )
    with pytest.raises(AstrometryNetIdentityError, match="astrometry_index_ids"):
        solve_with_astrometry_net(x, y, external=incomplete, **kwargs)


def test_executable_version_is_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    external = _external(tmp_path, version="0.98")

    with pytest.raises(AstrometryNetIdentityError, match="does not match"):
        _solve(tmp_path, monkeypatch, external=external)
