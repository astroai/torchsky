from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from torchsky.io import fits as ts_fits

astropy = pytest.importorskip("astropy")
fitsio = pytest.importorskip("fitsio")

from astropy.io import fits as astropy_fits  # noqa: E402


ASTROPY_ROOT = Path(astropy.__path__[0])
VLA_FIXTURE = (
    ASTROPY_ROOT / "io" / "fits" / "tests" / "data" / "variable_length_table.fits"
)


def _column_rows(value) -> list[object]:
    return value.to_pylist() if hasattr(value, "to_pylist") else list(value)


def test_generated_mef_image_matrix_matches_astropy(tmp_path) -> None:
    path = tmp_path / "matrix_mef.fits"
    primary = np.arange(12, dtype=np.float32).reshape(3, 4)
    science = (np.arange(20, dtype=np.int16).reshape(4, 5) - 7).astype(np.int16)
    mask = (np.arange(20, dtype=np.uint8).reshape(4, 5) % 3).astype(np.uint8)

    astropy_fits.HDUList(
        [
            astropy_fits.PrimaryHDU(primary),
            astropy_fits.ImageHDU(science, name="SCI"),
            astropy_fits.ImageHDU(mask, name="MASK"),
        ]
    ).writeto(path, overwrite=True)

    data, headers = ts_fits.read_hdus(
        path.as_posix(),
        [0, "SCI", "MASK"],
        mmap=False,
        return_header=True,
    )

    np.testing.assert_allclose(data[0].cpu().numpy(), primary, atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(data[1].cpu().numpy(), science)
    np.testing.assert_array_equal(data[2].cpu().numpy(), mask)
    assert headers[1]["EXTNAME"] == "SCI"
    assert headers[2]["EXTNAME"] == "MASK"


def test_generated_table_matrix_matches_astropy_and_fitsio(tmp_path) -> None:
    path = tmp_path / "matrix_table.fits"
    ids = np.array([1, 2, 3, 4], dtype=np.int32)
    flux = np.array([1.5, -2.0, 3.25, 4.5], dtype=np.float64)
    flags = np.array([True, False, True, False], dtype=np.bool_)
    names = np.array(["alpha", "beta", "gamma", "delta"], dtype="S8")

    cols = [
        astropy_fits.Column(name="ID", format="J", array=ids),
        astropy_fits.Column(name="FLUX", format="D", array=flux, unit="adu"),
        astropy_fits.Column(name="FLAG", format="L", array=flags),
        astropy_fits.Column(name="NAME", format="8A", array=names),
    ]
    astropy_fits.HDUList(
        [
            astropy_fits.PrimaryHDU(),
            astropy_fits.BinTableHDU.from_columns(cols, name="CATALOG"),
        ]
    ).writeto(path, overwrite=True)

    table = ts_fits.read_table(path.as_posix(), hdu=1, columns=["ID", "FLUX", "FLAG"])
    with ts_fits.open(path.as_posix()) as hdul:
        table_hdu = hdul["CATALOG"]
        got_names = table_hdu.get_string_column("NAME")
        assert table_hdu.header["TUNIT2"] == "adu"

    fitsio_data = fitsio.read(path.as_posix(), ext=1)
    np.testing.assert_array_equal(table["ID"].squeeze(-1).cpu().numpy(), ids)
    np.testing.assert_allclose(table["FLUX"].squeeze(-1).cpu().numpy(), flux)
    np.testing.assert_array_equal(table["FLAG"].squeeze(-1).cpu().numpy(), flags)
    assert got_names == [name.decode("ascii").rstrip() for name in names]
    np.testing.assert_array_equal(fitsio_data["ID"], ids)
    np.testing.assert_allclose(fitsio_data["FLUX"], flux)


def test_generated_compressed_image_matrix_matches_fitsio(tmp_path) -> None:
    path = tmp_path / "matrix_compressed.fits"
    image = (torch.arange(64, dtype=torch.int16).reshape(8, 8) - 32).contiguous()

    ts_fits.write(path.as_posix(), image, overwrite=True, compress=True)

    torchsky_data = ts_fits.read(path.as_posix(), hdu=1, mmap=False)
    fitsio_data = fitsio.read(path.as_posix(), ext=1)
    np.testing.assert_array_equal(torchsky_data.cpu().numpy(), image.cpu().numpy())
    np.testing.assert_array_equal(fitsio_data, image.cpu().numpy())


def test_astropy_variable_length_fixture_matrix_matches_astropy() -> None:
    table = ts_fits.read_table(VLA_FIXTURE.as_posix(), hdu=1)
    got_var = _column_rows(table["var"])
    got_xyz = _column_rows(table["xyz"])

    with astropy_fits.open(VLA_FIXTURE) as hdul:
        expected = hdul[1].data
        expected_var = [np.asarray(row) for row in expected["var"].tolist()]
        expected_xyz = [np.asarray(row) for row in expected["xyz"].tolist()]

    assert [list(row) for row in got_var] == [row.tolist() for row in expected_var]
    assert [list(row) for row in got_xyz] == [row.tolist() for row in expected_xyz]


def test_generated_radio_visibility_table_fixture_roundtrips_and_mutates(tmp_path) -> None:
    path = tmp_path / "radio_visibility_table.fits"
    nrow = 6
    time = np.linspace(60_000.0, 60_000.5, nrow, dtype=np.float64)
    antenna1 = np.array([0, 0, 1, 1, 2, 2], dtype=np.int16)
    antenna2 = np.array([1, 2, 2, 3, 3, 4], dtype=np.int16)
    uvw = np.stack(
        [
            np.linspace(-120.0, 120.0, nrow),
            np.linspace(10.0, 40.0, nrow),
            np.linspace(-4.0, 4.0, nrow),
        ],
        axis=1,
    ).astype(np.float64)
    vis = (
        np.linspace(1.0, 2.0, nrow) + 1j * np.linspace(-0.5, 0.5, nrow)
    ).astype(np.complex64)
    weight = np.linspace(0.5, 1.5, nrow, dtype=np.float32)
    flag = np.array([False, False, True, False, True, False], dtype=np.bool_)

    ts_fits.write(
        path.as_posix(),
        {
            "TIME": time,
            "ANTENNA1": antenna1,
            "ANTENNA2": antenna2,
            "UVW": uvw,
            "VIS": vis,
            "WEIGHT": weight,
            "FLAG": flag,
        },
        header={"EXTNAME": "VISDATA", "TELESCOP": "VLA", "RESTFRQ": 1.420405752e9},
        overwrite=True,
    )

    with ts_fits.open(path.as_posix()) as hdul:
        table = hdul["VISDATA"]
        np.testing.assert_allclose(table["TIME"].squeeze(-1).cpu().numpy(), time)
        np.testing.assert_array_equal(
            table["ANTENNA1"].squeeze(-1).cpu().numpy(),
            antenna1,
        )
        np.testing.assert_array_equal(
            table["ANTENNA2"].squeeze(-1).cpu().numpy(),
            antenna2,
        )
        np.testing.assert_allclose(table["UVW"].cpu().numpy(), uvw)
        np.testing.assert_allclose(table["VIS"].squeeze(-1).cpu().numpy(), vis)
        np.testing.assert_array_equal(table["FLAG"].squeeze(-1).cpu().numpy(), flag)
        assert table.header["TELESCOP"] == "VLA"
        assert float(table.header["RESTFRQ"]) == pytest.approx(1.420405752e9)

    with astropy_fits.open(path) as hdul:
        assert hdul["VISDATA"].header["TFORM4"].strip().upper() == "3D"
        assert hdul["VISDATA"].header["TFORM5"].strip().upper() in {"C", "1C"}

    import torchsky.table as ts_table

    ts_table.update_rows(
        path.as_posix(),
        {"WEIGHT": np.array([9.0, 8.0], dtype=np.float32)},
        row_slice=slice(1, 3),
        hdu=1,
    )
    updated = ts_fits.read_table(path.as_posix(), hdu=1, columns=["WEIGHT"])
    expected_weight = weight.copy()
    expected_weight[1:3] = [9.0, 8.0]
    np.testing.assert_allclose(
        updated["WEIGHT"].squeeze(-1).cpu().numpy(),
        expected_weight,
    )
