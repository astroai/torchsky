import json
from pathlib import Path
import sys

import pytest
import torch
import torchsky

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from score_smacs0723_foam import (  # noqa: E402
    BANDS,
    METHOD,
    _audit_foam_batch,
    _build_effective_psfs,
    _foam_product,
    _session_commit,
)


def test_foam_product_layout_and_full_commit(tmp_path: Path) -> None:
    assert _foam_product(tmp_path, "blank", "F090W", "prediction") == (
        tmp_path
        / "blank"
        / "products"
        / "f090w"
        / METHOD
        / "prediction.fits"
    )
    full_sha = "1" * 40
    (tmp_path / "session.log").write_text(f"requested_commit={full_sha}\n")
    assert _session_commit(tmp_path) == full_sha


def test_foam_batch_audit_fails_before_product_reads(tmp_path: Path) -> None:
    batch_root = tmp_path / "blank"
    batch_root.mkdir()
    (batch_root / "shared-foam.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"band": BANDS[0], "model": METHOD, "status": "OK"},
                    {"band": BANDS[1], "model": METHOD, "status": "FAIL"},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Foam bands are not complete"):
        _audit_foam_batch(tmp_path, "blank")


def test_foam_effective_psf_uses_product_dtype(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    blank = torch.zeros((48, 48), dtype=torch.float32)
    injected = blank.clone()
    injected[15, 15] = 1.0
    # The fainter-bin response is deliberately non-positive: calibration-bin
    # selection must come from the manifest, not the model response.
    injected[32, 32] = -2.0
    for band in BANDS:
        for batch, image in (("blank", blank), ("validation-000", injected)):
            path = input_root / batch / band / "products" / f"{METHOD}_sci.fits"
            path.parent.mkdir(parents=True, exist_ok=True)
            torchsky.write(str(path), image, overwrite=True)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "target_xy": [[15.0, 15.0], [32.0, 32.0]],
                "reference_ab_magnitude": [27.0, 28.0],
            }
        )
    )

    rows = _build_effective_psfs(input_root, manifest, tmp_path / "psfs")

    assert set(rows) == set(BANDS)
    for band in BANDS:
        psf = torchsky.read(str(rows[band]["path"]))
        torch.testing.assert_close(psf.sum(), torch.tensor(1.0))
        assert rows[band]["calibration_ab_magnitude"] == 27.0
        assert rows[band]["calibration_source_count"] == 1
