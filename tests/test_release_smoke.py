from __future__ import annotations

from importlib import metadata
from pathlib import Path
import numpy as np
import torch
from astropy.io import fits

import torchsky


def _declared_version() -> str:
    try:
        return metadata.version("torchsky")
    except metadata.PackageNotFoundError:
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        import tomllib

        with pyproject.open("rb") as fh:
            data = tomllib.load(fh)
        return str(data["project"]["version"])


def test_runtime_version_matches_declared_version() -> None:
    assert torchsky.__version__ == _declared_version()


def test_release_smoke_wcs_header_build(tmp_path: Path) -> None:
    data = np.zeros((8, 8), dtype=np.float32)
    hdu = fits.PrimaryHDU(data)
    hdu.header["CTYPE1"] = "RA---TAN"
    hdu.header["CTYPE2"] = "DEC--TAN"
    hdu.header["CRPIX1"] = 4.0
    hdu.header["CRPIX2"] = 4.0
    hdu.header["CRVAL1"] = 180.0
    hdu.header["CRVAL2"] = 0.0
    hdu.header["CD1_1"] = -0.01
    hdu.header["CD1_2"] = 0.0
    hdu.header["CD2_1"] = 0.0
    hdu.header["CD2_2"] = 0.01

    path = tmp_path / "wcs.fits"
    hdu.writeto(path, overwrite=True)
    wcs = torchsky.get_wcs(path.as_posix(), hdu=0)
    world = wcs.pixel_to_world(torch.tensor([3.0]), torch.tensor([3.0]))
    assert len(world) == 2
    assert torch.isfinite(world[0]).all()
    assert torch.isfinite(world[1]).all()
