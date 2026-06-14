import os
import tempfile

import numpy as np
import pytest
import torch

import torchsky


def _make_basic_table_file() -> str:
    handle = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    handle.close()
    torchsky.write(
        handle.name,
        {
            "ID": np.array([1, 2, 3], dtype=np.int32),  # type: ignore
            "VAL": np.array([0.1, 0.2, 0.3], dtype=np.float32),  # type: ignore
            "FLAG": np.array([True, False, True], dtype=np.bool_),  # type: ignore
        },
        overwrite=True,
    )
    return handle.name


def test_table_append_update_rename_drop():
    path = _make_basic_table_file()
    try:
        torchsky.table.append_rows(
            path,
            {
                "ID": np.array([4, 5], dtype=np.int32),  # type: ignore
                "VAL": np.array([0.4, 0.5], dtype=np.float32),  # type: ignore
                "FLAG": np.array([False, True], dtype=np.bool_),  # type: ignore
            },
            hdu=1,
        )
        with torchsky.open(path) as hdul:  # type: ignore
            table_hdu = hdul[1]  # type: ignore
            ids = table_hdu["ID"]  # type: ignore
            vals = table_hdu["VAL"]  # type: ignore
            flags = table_hdu["FLAG"]  # type: ignore
            assert isinstance(ids, torch.Tensor)
            assert isinstance(vals, torch.Tensor)
            assert isinstance(flags, torch.Tensor)
            assert ids.squeeze(-1).tolist() == [1, 2, 3, 4, 5]  # type: ignore
            assert np.allclose(
                vals.squeeze(-1).numpy(), [0.1, 0.2, 0.3, 0.4, 0.5], atol=1e-6
            )
            assert flags.squeeze(-1).tolist() == [True, False, True, False, True]  # type: ignore

        with pytest.raises(ValueError):
            torchsky.table.append_rows(
                path,
                {
                    "ID": [6],
                    "VAL": [0.6],
                    "FLAG": [False],
                    "EXTRA": [1],
                },
                hdu=1,
            )

        torchsky.table.update_rows(
            path,
            {"VAL": np.array([9.9, 8.8], dtype=np.float32)},  # type: ignore
            row_slice=slice(1, 3),
            hdu=1,
        )
        with torchsky.open(path) as hdul:  # type: ignore
            table_hdu = hdul[1]  # type: ignore
            vals = table_hdu["VAL"].squeeze(-1).numpy()  # type: ignore
            assert np.allclose(vals, [0.1, 9.9, 8.8, 0.4, 0.5], atol=1e-6)

        torchsky.table.rename_columns(path, {"VAL": "FLUX"}, hdu=1)  # type: ignore
        with torchsky.open(path) as hdul:  # type: ignore
            table_hdu = hdul[1]  # type: ignore
            assert "VAL" not in table_hdu.columns  # type: ignore
            assert "FLUX" in table_hdu.columns  # type: ignore

        torchsky.table.drop_columns(path, ["FLAG"], hdu=1)  # type: ignore
        with torchsky.open(path) as hdul:  # type: ignore
            table_hdu = hdul[1]  # type: ignore
            assert "FLAG" not in table_hdu.columns  # type: ignore
    finally:
        os.unlink(path)


def test_tablehdu_to_fits_rich_types_pending_cfitsio_impl():
    dst = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    dst.close()
    table_hdu = torchsky.TableHDU(  # type: ignore
        {
            "ID": np.array([1, 2, 3], dtype=np.int32),  # type: ignore
            "NAME": ["alpha", "beta", "gamma"],
            "Z": np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64),  # type: ignore
            "VLA": [
                np.array([1, 2], dtype=np.int32),  # type: ignore
                np.array([3], dtype=np.int32),  # type: ignore
                np.array([4, 5, 6], dtype=np.int32),  # type: ignore
            ],
        }
    )

    try:
        table_hdu.to_fits(dst.name, overwrite=True)  # type: ignore
        with torchsky.open(dst.name) as hdul:  # type: ignore
            table = hdul[1]  # type: ignore
            assert table.get_string_column("NAME") == ["alpha", "beta", "gamma"]  # type: ignore
            vals = table["Z"].squeeze(-1)  # type: ignore
            assert np.allclose(
                vals.numpy(),
                np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64),  # type: ignore
            )
            vla = table.get_vla_column("VLA")  # type: ignore
            assert [v.tolist() for v in vla] == [[1, 2], [3], [4, 5, 6]]  # type: ignore
    finally:
        if os.path.exists(dst.name):
            os.unlink(dst.name)


def test_table_vla_roundtrip_and_updates():
    path = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    path.close()
    try:
        table = {
            "ID": np.array([1, 2, 3], dtype=np.int32),  # type: ignore
            "VLA": [
                np.array([1, 2], dtype=np.int16),  # type: ignore
                np.array([3], dtype=np.int16),  # type: ignore
                np.array([], dtype=np.int16),  # type: ignore
            ],
        }
        torchsky.write(path.name, table, overwrite=True)

        with torchsky.open(path.name) as hdul:  # type: ignore
            vla = hdul[1].get_vla_column("VLA")  # type: ignore
            assert [v.tolist() for v in vla] == [[1, 2], [3], []]  # type: ignore

        torchsky.table.append_rows(
            path.name,
            {
                "ID": np.array([4], dtype=np.int32),  # type: ignore
                "VLA": [np.array([9, 10], dtype=np.int16)],  # type: ignore
            },
            hdu=1,
        )
        with torchsky.open(path.name) as hdul:  # type: ignore
            vla = hdul[1].get_vla_column("VLA")  # type: ignore
            assert vla[-1].tolist() == [9, 10]  # type: ignore

        torchsky.table.update_rows(
            path.name,
            {"VLA": [np.array([7], dtype=np.int16)]},  # type: ignore
            row_slice=slice(1, 2),
            hdu=1,
        )
        with torchsky.open(path.name) as hdul:  # type: ignore
            vla = hdul[1].get_vla_column("VLA")  # type: ignore
            assert vla[1].tolist() == [7]  # type: ignore
    finally:
        os.unlink(path.name)


def test_table_write_schema_pending_cfitsio_impl():
    path = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    path.close()
    try:
        torchsky.table.write(
            path.name,
            data={"ID": np.array([1, 2, 3], dtype=np.int32)},  # type: ignore
            schema={"ID": {"format": "J"}},
            overwrite=True,
        )
        with torchsky.open(path.name) as hdul:  # type: ignore
            table = hdul[1]  # type: ignore
            assert table["ID"].squeeze(-1).tolist() == [1, 2, 3]  # type: ignore
            assert str(table.header.get("TFORM1", "")).upper().startswith("J")
    finally:
        os.unlink(path.name)


def test_table_write_ascii_pending_cfitsio_impl():
    path = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    path.close()
    try:
        torchsky.table.write(
            path.name,
            data={"A": np.array([1, 2, 3], dtype=np.int32)},  # type: ignore
            table_type="ascii",
            overwrite=True,
        )
        with torchsky.open(path.name) as hdul:  # type: ignore
            table = hdul[1]  # type: ignore
            assert table["A"].squeeze(-1).tolist() == [1, 2, 3]  # type: ignore
            assert str(table.header.get("XTENSION", "")).upper() == "TABLE"
    finally:
        os.unlink(path.name)


def test_table_reader_cache_invalidated_after_append():
    path = _make_basic_table_file()
    try:
        # Prime Arrow/cpp reader cache for this path.
        initial = torchsky.table.read(path, hdu=1)
        assert initial.num_rows == 3

        torchsky.table.append_rows(
            path,
            {
                "ID": np.array([4], dtype=np.int32),  # type: ignore
                "VAL": np.array([0.4], dtype=np.float32),  # type: ignore
                "FLAG": np.array([False], dtype=np.bool_),  # type: ignore
            },
            hdu=1,
        )

        updated = torchsky.table.read(path, hdu=1)
        assert updated.num_rows == 4
        assert updated.column("ID").to_pylist() == [1, 2, 3, 4]
    finally:
        os.unlink(path)


def test_table_insert_rows_mid_table_preserves_order():
    path = _make_basic_table_file()
    try:
        torchsky.table.insert_rows(
            path,
            {
                "ID": np.array([99, 100], dtype=np.int32),  # type: ignore
                "VAL": np.array([9.9, 10.0], dtype=np.float32),  # type: ignore
                "FLAG": np.array([False, True], dtype=np.bool_),  # type: ignore
            },
            row=1,
            hdu=1,
        )

        with torchsky.open(path) as hdul:  # type: ignore
            table_hdu = hdul[1]  # type: ignore
            assert table_hdu["ID"].squeeze(-1).tolist() == [1, 99, 100, 2, 3]  # type: ignore
            assert np.allclose(
                table_hdu["VAL"].squeeze(-1).numpy(),  # type: ignore
                [0.1, 9.9, 10.0, 0.2, 0.3],
                atol=1e-6,
            )
            assert table_hdu["FLAG"].squeeze(-1).tolist() == [  # type: ignore
                True,
                False,
                True,
                False,
                True,
            ]
    finally:
        os.unlink(path)


def test_table_delete_rows_slice_and_single():
    path = _make_basic_table_file()
    try:
        torchsky.table.append_rows(
            path,
            {
                "ID": np.array([4, 5], dtype=np.int32),  # type: ignore
                "VAL": np.array([0.4, 0.5], dtype=np.float32),  # type: ignore
                "FLAG": np.array([False, True], dtype=np.bool_),  # type: ignore
            },
            hdu=1,
        )
        torchsky.table.delete_rows(path, 1, hdu=1)
        torchsky.table.delete_rows(path, slice(2, 4), hdu=1)

        with torchsky.open(path) as hdul:  # type: ignore
            table_hdu = hdul[1]  # type: ignore
            assert table_hdu["ID"].squeeze(-1).tolist() == [1, 3]  # type: ignore
            assert np.allclose(
                table_hdu["VAL"].squeeze(-1).numpy(),
                [0.1, 0.3],
                atol=1e-6,  # type: ignore
            )
            assert table_hdu["FLAG"].squeeze(-1).tolist() == [True, True]  # type: ignore
    finally:
        os.unlink(path)


def test_append_rows_partial_payload_string_vla_defaults():
    path = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    path.close()
    try:
        torchsky.write(
            path.name,
            {
                "ID": np.array([1, 2], dtype=np.int32),  # type: ignore
                "NAME": ["alpha", "beta"],
                "VLA": [
                    np.array([10, 11], dtype=np.int16),  # type: ignore
                    np.array([20], dtype=np.int16),  # type: ignore
                ],
            },
            overwrite=True,
        )

        torchsky.table.append_rows(
            path.name,
            {"ID": np.array([3], dtype=np.int32)},  # type: ignore
            hdu=1,
        )

        with torchsky.open(path.name) as hdul:  # type: ignore
            table_hdu = hdul[1]  # type: ignore
            assert table_hdu["ID"].squeeze(-1).tolist() == [1, 2, 3]  # type: ignore
            assert table_hdu.get_string_column("NAME") == ["alpha", "beta", ""]  # type: ignore
            vla = table_hdu.get_vla_column("VLA")  # type: ignore
            assert [v.tolist() for v in vla] == [[10, 11], [20], []]  # type: ignore
    finally:
        os.unlink(path.name)


def test_append_rows_partial_payload_respects_tnull():
    path = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    path.close()
    try:
        torchsky.table.write(
            path.name,
            data={
                "ID": np.array([1, 2], dtype=np.int32),  # type: ignore
                "QUAL": np.array([7, 8], dtype=np.int16),  # type: ignore
            },
            schema={
                "ID": {"format": "J"},
                "QUAL": {"format": "I", "tnull": -999},
            },
            overwrite=True,
        )

        torchsky.table.append_rows(
            path.name,
            {"ID": np.array([3], dtype=np.int32)},  # type: ignore
            hdu=1,
        )

        with torchsky.open(path.name) as hdul:  # type: ignore
            table_hdu = hdul[1]  # type: ignore
            assert table_hdu["ID"].squeeze(-1).tolist() == [1, 2, 3]  # type: ignore
            assert table_hdu["QUAL"].squeeze(-1).tolist() == [7, 8, -999]  # type: ignore
    finally:
        os.unlink(path.name)


def test_table_insert_delete_with_vla_and_string_columns():
    path = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    path.close()
    try:
        torchsky.write(
            path.name,
            {
                "ID": np.array([1, 2, 3], dtype=np.int32),  # type: ignore
                "NAME": ["a", "b", "c"],
                "VLA": [
                    np.array([1], dtype=np.int16),  # type: ignore
                    np.array([2, 3], dtype=np.int16),  # type: ignore
                    np.array([4], dtype=np.int16),  # type: ignore
                ],
            },
            overwrite=True,
        )

        torchsky.table.insert_rows(
            path.name,
            {"ID": np.array([99], dtype=np.int32)},  # type: ignore
            row=1,
            hdu=1,
        )
        torchsky.table.delete_rows(path.name, slice(2, 3), hdu=1)

        with torchsky.open(path.name) as hdul:  # type: ignore
            table_hdu = hdul[1]  # type: ignore
            assert table_hdu["ID"].squeeze(-1).tolist() == [1, 99, 3]  # type: ignore
            assert table_hdu.get_string_column("NAME") == ["a", "", "c"]  # type: ignore
            vla = table_hdu.get_vla_column("VLA")  # type: ignore
            assert [v.tolist() for v in vla] == [[1], [], [4]]  # type: ignore
    finally:
        os.unlink(path.name)


def test_insert_column_with_explicit_format_metadata():
    path = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    path.close()
    try:
        torchsky.table.write(
            path.name,
            data={
                "ID": np.array([1, 2], dtype=np.int32),  # type: ignore
                "QUAL": np.array([10, 20], dtype=np.int16),  # type: ignore
            },
            schema={
                "ID": {"format": "J"},
                "QUAL": {"format": "I", "unit": "adu", "tnull": -999},
            },
            overwrite=True,
        )

        torchsky.table.insert_column(
            path.name,
            "FLAGS",
            np.array([7, 8], dtype=np.int16),  # type: ignore
            hdu=1,
            index=1,
            format="I",
            unit="flag",
            tnull=-1,
        )

        with torchsky.open(path.name) as hdul:  # type: ignore
            table = hdul[1]  # type: ignore
            assert table.columns == ["ID", "FLAGS", "QUAL"]  # type: ignore
            assert table["FLAGS"].squeeze(-1).tolist() == [7, 8]  # type: ignore
            assert table.header.get("TTYPE2") == "FLAGS"
            assert str(table.header.get("TFORM2", "")).upper().startswith("I")
            assert table.header.get("TUNIT2") == "flag"
            assert int(table.header.get("TNULL2")) == -1  # type: ignore
            # Existing metadata moves with the original column.
            assert table.header.get("TTYPE3") == "QUAL"
            assert table.header.get("TUNIT3") == "adu"
            assert int(table.header.get("TNULL3")) == -999  # type: ignore
    finally:
        os.unlink(path.name)


def test_replace_column_preserves_metadata_contract():
    path = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    path.close()
    try:
        torchsky.table.write(
            path.name,
            data={
                "ID": np.array([1, 2, 3], dtype=np.int32),  # type: ignore
                "QUAL": np.array([11, 12, 13], dtype=np.int16),  # type: ignore
            },
            schema={
                "ID": {"format": "J"},
                "QUAL": {"format": "I", "unit": "adu", "tnull": -999},
            },
            overwrite=True,
        )

        torchsky.table.replace_column(
            path.name,
            "QUAL",
            np.array([101, 102, 103], dtype=np.int16),  # type: ignore
            hdu=1,
        )

        with torchsky.open(path.name) as hdul:  # type: ignore
            table = hdul[1]  # type: ignore
            assert table["QUAL"].squeeze(-1).tolist() == [101, 102, 103]  # type: ignore
            assert str(table.header.get("TFORM2", "")).upper().startswith("I")
            assert table.header.get("TUNIT2") == "adu"
            assert int(table.header.get("TNULL2")) == -999  # type: ignore
    finally:
        os.unlink(path.name)


def test_table_update_forced_mmap_rejects_vla_and_strings():
    path = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    path.close()
    try:
        torchsky.write(
            path.name,
            {
                "ID": np.array([1, 2], dtype=np.int32),  # type: ignore
                "NAME": ["a", "b"],
                "VLA": [
                    np.array([1], dtype=np.int16),  # type: ignore
                    np.array([2, 3], dtype=np.int16),  # type: ignore
                ],
            },
            overwrite=True,
        )

        with pytest.raises(ValueError, match="mmap table updates"):
            torchsky.table.update_rows(
                path.name,
                {"VLA": [np.array([7], dtype=np.int16)]},  # type: ignore
                row_slice=slice(0, 1),
                hdu=1,
                mmap=True,
            )

        torchsky.table.update_rows(
            path.name,
            {"VLA": [np.array([7], dtype=np.int16)]},  # type: ignore
            row_slice=slice(0, 1),
            hdu=1,
            mmap="auto",
        )
        with torchsky.open(path.name) as hdul:  # type: ignore
            assert hdul[1].get_vla_column("VLA")[0].tolist() == [7]  # type: ignore
    finally:
        os.unlink(path.name)


def test_table_rename_and_drop_validate_column_names_before_mutating():
    path = _make_basic_table_file()
    try:
        with pytest.raises(ValueError, match="collide"):
            torchsky.table.rename_columns(path, {"ID": "VAL"}, hdu=1)

        with pytest.raises(KeyError, match="MISSING"):
            torchsky.table.rename_columns(path, {"MISSING": "X"}, hdu=1)

        with pytest.raises(ValueError, match="duplicate"):
            torchsky.table.drop_columns(path, ["FLAG", "FLAG"], hdu=1)

        with pytest.raises(KeyError, match="MISSING"):
            torchsky.table.drop_columns(path, ["MISSING"], hdu=1)

        with torchsky.open(path) as hdul:  # type: ignore
            table = hdul[1]  # type: ignore
            assert table.columns == ["ID", "VAL", "FLAG"]  # type: ignore
    finally:
        os.unlink(path)


def test_compress_true_writes_table_hdu_without_tensorframe_placeholder():
    path = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    path.close()
    try:
        torchsky.write(
            path.name,
            {
                "ID": np.array([1, 2, 3], dtype=np.int32),  # type: ignore
                "NAME": ["a", "b", "c"],
            },
            header={"EXTNAME": "CAT"},
            overwrite=True,
            compress=True,
        )

        header = torchsky.get_header(path.name, 1)
        assert str(header.get("XTENSION", "")).upper() == "BINTABLE"
        assert header.get("EXTNAME") == "CAT"
        table = torchsky.table.read(path.name, hdu=1)
        assert table.column("ID").to_pylist() == [1, 2, 3]
        assert table.column("NAME").to_pylist() == ["a", "b", "c"]
    finally:
        os.unlink(path.name)


def test_compressed_mixed_image_and_table_roundtrip():
    path = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    path.close()
    try:
        image = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        torchsky.write(
            path.name,
            [
                {"data": image, "header": {"EXTNAME": "SCI"}},
                {
                    "ID": np.array([10, 20], dtype=np.int32),  # type: ignore
                    "FLUX": np.array([1.5, 2.5], dtype=np.float32),  # type: ignore
                },
            ],
            overwrite=True,
            compress=True,
        )

        np.testing.assert_allclose(
            torchsky.read(path.name, hdu=1).numpy(), image.numpy()
        )
        table = torchsky.table.read(path.name, hdu=2)
        assert table.column("ID").to_pylist() == [10, 20]
        np.testing.assert_allclose(table.column("FLUX").to_pylist(), [1.5, 2.5])
    finally:
        os.unlink(path.name)
