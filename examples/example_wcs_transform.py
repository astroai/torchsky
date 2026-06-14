import os

import numpy as np
import torch
from astropy.io import fits

import torchsky
from torchsky.wcs import WCS
from torchsky.wcs.temporal import TemporalWCS


def create_test_files(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)

    # 2D image (RA/DEC TAN)
    image_file = os.path.join(data_dir, "test_image_2d.fits")
    image_data = np.random.rand(100, 100).astype(np.float32)
    hdu = fits.PrimaryHDU(image_data)
    hdu.header["CTYPE1"] = "RA---TAN"
    hdu.header["CTYPE2"] = "DEC--TAN"
    hdu.header["CRVAL1"] = 200.0
    hdu.header["CRVAL2"] = 45.0
    hdu.header["CRPIX1"] = 50.0
    hdu.header["CRPIX2"] = 50.0
    hdu.header["CDELT1"] = -0.01
    hdu.header["CDELT2"] = 0.01
    hdu.writeto(image_file, overwrite=True)

    # 1D spectrum (wavelength)
    spectrum_file = os.path.join(data_dir, "test_spectrum_1d.fits")
    flux = np.random.rand(1000).astype(np.float32)
    hdu = fits.PrimaryHDU(flux)
    hdu.header["CTYPE1"] = "WAVE"
    hdu.header["CRVAL1"] = 4000.0
    hdu.header["CRPIX1"] = 1.0
    hdu.header["CDELT1"] = 3.0
    hdu.writeto(spectrum_file, overwrite=True)

    # 3D cube (RA/DEC/WAVE)
    cube_file = os.path.join(data_dir, "test_cube_3d.fits")
    cube_data = np.random.rand(10, 20, 30).astype(np.float32)
    hdu = fits.PrimaryHDU(cube_data)
    hdu.header["CTYPE1"] = "RA---TAN"
    hdu.header["CTYPE2"] = "DEC--TAN"
    hdu.header["CTYPE3"] = "WAVE"
    hdu.header["CRVAL1"] = 200.0
    hdu.header["CRVAL2"] = 45.0
    hdu.header["CRVAL3"] = 5000.0
    hdu.header["CRPIX1"] = 5.0
    hdu.header["CRPIX2"] = 10.0
    hdu.header["CRPIX3"] = 1.0
    hdu.header["CDELT1"] = -0.01
    hdu.header["CDELT2"] = 0.01
    hdu.header["CDELT3"] = 5.0
    hdu.writeto(cube_file, overwrite=True)

    # 1D non-uniform spectral lookup axis (WAVE-TAB + table extension)
    tab_file = os.path.join(data_dir, "test_wave_tab_1d.fits")
    hdu = fits.PrimaryHDU(np.zeros(10, dtype=np.float32))
    hdu.header["CTYPE1"] = "WAVE-TAB"
    hdu.header["CUNIT1"] = "nm"
    hdu.header["CRVAL1"] = 0.0
    hdu.header["CRPIX1"] = 1.0
    hdu.header["CDELT1"] = 1.0
    hdu.header["PS1_0"] = "WCS-TAB"
    hdu.header["PS1_1"] = "WAVE"
    hdu.header["PS1_2"] = "PIX"
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="PIX", format="D", array=np.array([0.0, 2.0, 5.0])),
            fits.Column(name="WAVE", format="D", array=np.array([400.0, 420.0, 510.0])),
        ],
        name="WCS-TAB",
    )
    fits.HDUList([hdu, table]).writeto(tab_file, overwrite=True)


def _show_roundtrip(path: str) -> None:
    wcs = torchsky.get_wcs(path, hdu="auto")
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
    data_dir = "data_wcs_examples"
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
        wcs = torchsky.get_wcs(paths[0], device="cuda")
        pixels = torch.tensor([[49.0, 49.0]], dtype=torch.float64, device="cuda")
        world = wcs.pixel_to_world(pixels)
        print("CUDA sample world coord:", world[0].tolist())


if __name__ == "__main__":
    main()
