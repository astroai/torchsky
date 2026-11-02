import os
import tempfile

import numpy as np
import torch
import torchfits

from torchfits.hdu import HDUList, Header, TableHDU, TensorHDU
from torchsky.wcs import WCS
from torchsky.wcs.temporal import TemporalWCS


def create_test_files(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)

    torchfits.write(
        os.path.join(data_dir, "test_image_2d.fits"),
        torch.from_numpy(np.random.rand(100, 100).astype(np.float32)),
        header={
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRVAL1": 200.0,
            "CRVAL2": 45.0,
            "CRPIX1": 50.0,
            "CRPIX2": 50.0,
            "CDELT1": -0.01,
            "CDELT2": 0.01,
        },
        overwrite=True,
    )
    torchfits.write(
        os.path.join(data_dir, "test_spectrum_1d.fits"),
        torch.from_numpy(np.random.rand(1000).astype(np.float32)),
        header={
            "CTYPE1": "WAVE",
            "CRVAL1": 4000.0,
            "CRPIX1": 1.0,
            "CDELT1": 3.0,
        },
        overwrite=True,
    )
    torchfits.write(
        os.path.join(data_dir, "test_cube_3d.fits"),
        torch.from_numpy(np.random.rand(10, 20, 30).astype(np.float32)),
        header={
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CTYPE3": "WAVE",
            "CRVAL1": 200.0,
            "CRVAL2": 45.0,
            "CRVAL3": 5000.0,
            "CRPIX1": 5.0,
            "CRPIX2": 10.0,
            "CRPIX3": 1.0,
            "CDELT1": -0.01,
            "CDELT2": 0.01,
            "CDELT3": 5.0,
        },
        overwrite=True,
    )

    image_header = Header()
    image_header["CTYPE1"] = "WAVE-TAB"
    image_header["CUNIT1"] = "nm"
    image_header["CRVAL1"] = 0.0
    image_header["CRPIX1"] = 1.0
    image_header["CDELT1"] = 1.0
    image_header["PS1_0"] = "WCS-TAB"
    image_header["PS1_1"] = "WAVE"
    image_header["PS1_2"] = "PIX"
    table_header = Header()
    table_header["XTENSION"] = "BINTABLE"
    table_header["EXTNAME"] = "WCS-TAB"
    HDUList(
        [
            TensorHDU(data=torch.zeros(10), header=image_header),
            TableHDU(
                tensor_dict={
                    "PIX": torch.tensor([0.0, 2.0, 5.0], dtype=torch.float64),
                    "WAVE": torch.tensor([400.0, 420.0, 510.0], dtype=torch.float64),
                },
                header=table_header,
            ),
        ]
    ).write(os.path.join(data_dir, "test_wave_tab_1d.fits"), overwrite=True)


def _show_roundtrip(path: str) -> None:
    with torchfits.open(path) as hdul:
        wcs = WCS(dict(torchfits.read_header(path, 0)), tables=hdul)
    ctypes = [wcs.wcs_params.get(f"CTYPE{i}", "") for i in range(1, wcs.naxis + 1)]
    print(f"{os.path.basename(path)}: naxis={wcs.naxis}, ctype={ctypes}")

    if wcs.naxis == 1 and getattr(wcs, "tabular_axes", {}):
        pixels = torch.tensor([[0.0], [2.0], [5.0]], dtype=torch.float64)
    elif wcs.naxis == 1:
        pixels = torch.tensor([[0.0], [10.0], [100.0]], dtype=torch.float64)
    elif wcs.naxis == 2:
        pixels = torch.tensor(
            [[0.0, 0.0], [49.0, 49.0], [99.0, 99.0]], dtype=torch.float64
        )
    else:
        pixels = torch.tensor([[4.0, 9.0, 0.0], [10.0, 10.0, 5.0]], dtype=torch.float64)

    world = wcs.pixel_to_world(pixels)
    pixels_back = wcs.world_to_pixel(world)
    max_err = (pixels_back - pixels).abs().max().item()

    print(f"  pixel sample: {pixels[0].tolist()}")
    print(f"  world sample: {world[0].tolist()}")
    print(f"  roundtrip max pixel error: {max_err:.3e}")


def _show_spectral_temporal_samples() -> None:
    grism_header = {
        "NAXIS": 1,
        "CTYPE1": "WAVE-GRA",
        "CUNIT1": "m",
        "CRPIX1": 1.0,
        "CRVAL1": 5.0e-7,
        "CDELT1": 1.0e-9,
        "PV1_0": 300000.0,
        "PV1_1": 1.0,
        "PV1_2": 15.0,
        "PV1_3": 1.5,
        "PV1_4": -1.0e5,
        "PV1_5": 0.0,
        "PV1_6": 10.0,
    }
    grism_wcs = WCS(grism_header)
    world = grism_wcs.pixel_to_world(
        torch.tensor([[0.0], [10.0]], dtype=torch.float64),
        origin=0,
    )
    print("WAVE-GRA sample wavelengths:", world[:, 0].tolist())

    twcs = TemporalWCS(
        {
            "MJDREF": 60000.0,
            "TREFPOS": "BARYCENTER",
            "OBSGEO-X": 6378137.0,
            "OBSGEO-Y": 0.0,
            "OBSGEO-Z": 0.0,
        }
    )
    mjd = torch.tensor([60000.0], dtype=torch.float64)
    corrected = twcs.apply_corrections(
        mjd,
        torch.tensor([120.0], dtype=torch.float64),
        torch.tensor([30.0], dtype=torch.float64),
    )
    print(
        "Barycentric light-time correction [s]:", ((corrected - mjd) * 86400.0).item()
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="torchsky-wcs-example-") as data_dir:
        _run(data_dir)


def _run(data_dir: str) -> None:
    create_test_files(data_dir)

    paths = [
        os.path.join(data_dir, "test_image_2d.fits"),
        os.path.join(data_dir, "test_spectrum_1d.fits"),
        os.path.join(data_dir, "test_cube_3d.fits"),
        os.path.join(data_dir, "test_wave_tab_1d.fits"),
    ]

    for path in paths:
        _show_roundtrip(path)
    _show_spectral_temporal_samples()

    if torch.cuda.is_available():
        with torchfits.open(paths[0]) as hdul:
            wcs = WCS(dict(torchfits.read_header(paths[0], 0)), tables=hdul).to("cuda")
        pixels = torch.tensor([[49.0, 49.0]], dtype=torch.float64, device="cuda")
        world = wcs.pixel_to_world(pixels)
        print("CUDA sample world coord:", world[0].tolist())


if __name__ == "__main__":
    main()
