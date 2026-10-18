"""Read the image and table extensions of a small torchfits MEF file."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torchfits



def create_test_file(filename: str | Path) -> None:
    science_header = {
        "EXTNAME": "SCI",
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 5.5,
        "CRPIX2": 5.5,
        "CRVAL1": 150.0,
        "CRVAL2": 45.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
    }
    hdus = torchfits.HDUList(
        [
            torchfits.TensorHDU(data=torch.empty(0)),
            torchfits.TensorHDU(
                data=torch.arange(100, dtype=torch.float32).reshape(10, 10),
                header=science_header,
            ),
            torchfits.TensorHDU(
                data=torch.full((20, 20), 0.5, dtype=torch.float32),
                header={"EXTNAME": "ERR"},
            ),
            torchfits.TableHDU(
                {
                    "ra": torch.tensor([150.0, 151.0, 152.0], dtype=torch.float64),
                    "dec": torch.tensor([45.0, 46.0, 47.0], dtype=torch.float64),
                    "flux": torch.tensor([10.0, 12.0, 15.0], dtype=torch.float32),
                },
                header={"EXTNAME": "CATALOG"},
            ),
        ]
    )
    torchfits.write(filename, hdus, overwrite=True)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="torchsky-mef-example-") as directory:
        test_file = Path(directory) / "mef_example.fits"
        create_test_file(test_file)

        with torchfits.open(str(test_file)) as hdul:
            print(f"Number of HDUs: {len(hdul)}")
            for index, hdu in enumerate(hdul):
                name = hdu.header.get("EXTNAME", "PRIMARY")
                if isinstance(hdu, torchfits.TensorHDU):
                    data = hdu.to_tensor()
                    if data.numel() == 0:
                        kind = "EMPTY"
                    else:
                        kind = f"IMAGE {tuple(data.shape)}"
                else:
                    kind = f"TABLE columns={hdu.columns}"
                print(f"  HDU {index} {name}: {kind}")

            try:
                science_hdu = hdul["SCI"]
                catalog_hdu = hdul["CATALOG"]
            except KeyError as exc:
                raise RuntimeError("MEF example is missing a named extension") from exc
            print(f"SCI CTYPE1: {science_hdu.header['CTYPE1']}")
            print(f"CATALOG columns: {catalog_hdu.columns}")

        science, science_header = torchfits.read(
            str(test_file), hdu="SCI", return_header=True
        )
        catalog, _ = torchfits.read(
            str(test_file), hdu="CATALOG", return_header=True
        )
        print(f"SCI image shape: {tuple(science.shape)}")
        print(f"SCI WCS axes: {science_header['CTYPE1']}, {science_header['CTYPE2']}")
        print(f"CATALOG columns: {list(catalog.keys())}")


if __name__ == "__main__":
    main()
