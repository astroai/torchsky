"""Exercise manifest pairing and publication through the standalone CLI."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_updc_manifest_flexible.py"


def test_manifest_pairing_rejects_ambiguous_inputs_and_preserves_output():
    with TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        exposures, psfs = root / "exposures", root / "psfs"
        exposures.mkdir()
        psfs.mkdir()
        target, output = root / "reference.fits", root / "manifest.json"
        target.touch()
        exposure = exposures / "jwst_cal_1overf_completed_dqfix.fits"
        psf = psfs / "jwst_hybpsf_1501.fits"
        exposure.touch()
        psf.touch()
        command = [
            sys.executable,
            str(SCRIPT),
            "--exposure-dir",
            str(exposures),
            "--psf-dir",
            str(psfs),
            "--target-reference",
            str(target),
            "--output",
            str(output),
            "--band",
            "F277W",
        ]

        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        original = output.read_bytes()
        assert json.loads(original) == {
            "band": "F277W",
            "target_reference": str(target),
            "observations": [{"exposure": str(exposure), "psf": str(psf)}],
        }
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode != 0 and "exists" in result.stderr
        assert output.read_bytes() == original

        def reject(message, *options):
            result = subprocess.run(
                [*command, "--overwrite", *options], capture_output=True, text=True
            )
            assert result.returncode != 0, result.stdout
            assert message in result.stderr, result.stderr
            assert output.read_bytes() == original
            assert sorted(root.glob(".manifest.json.*")) == []

        reject("positive", "--expected-exposures", "0")
        reject("positive", "--expected-exposures", "-1")
        reject("expected 2 exposures", "--expected-exposures", "2")
        target.unlink()
        reject("target reference")
        target.touch()
        psf.unlink()
        reject("missing PSF")
        psf.touch()
        duplicate_psf = psfs / "jwst_hybpsf.fits"
        duplicate_psf.touch()
        reject("duplicate PSF", "--psf-glob", "*.fits")
        duplicate_psf.unlink()
        duplicate_exposure = exposures / "jwst_cal.fits"
        duplicate_exposure.touch()
        reject("duplicate exposure")
        duplicate_exposure.unlink()
        exposure.unlink()
        reject("no exposures")
        exposure.touch()
        result = subprocess.run(
            [*command, "--overwrite", "--expected-exposures", "1"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert output.read_bytes() == original
