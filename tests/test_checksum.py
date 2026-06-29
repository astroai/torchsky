from __future__ import annotations

import struct

import torch
import pytest

import torchsky


def test_file_checksum_roundtrip_and_corruption(tmp_path):
    path = tmp_path / "chk.fits"
    data = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    torchsky.write(str(path), data, header={"FOO": 1}, overwrite=True)

    # No keywords yet.
    out = torchsky.verify_checksums(str(path), hdu=0)
    assert out["datastatus"] == 0
    assert out["hdustatus"] == 0
    assert out["ok"] is False

    torchsky.write_checksums(str(path), hdu=0)
    out = torchsky.verify_checksums(str(path), hdu=0)
    assert out["ok"] is True

    # Corrupt an unrelated header keyword: DATASUM should remain valid, CHECKSUM should fail.
    with open(path, "r+b") as f:
        raw = f.read()
        needle = b"FOO     ="
        idx = raw.find(needle)
        assert idx != -1
        card = bytearray(raw[idx : idx + 80])
        one = card.find(b"1")
        assert one != -1
        card[one : one + 1] = b"2"
        f.seek(idx)
        f.write(card)

    out = torchsky.verify_checksums(str(path), hdu=0)
    assert out["datastatus"] == 1
    assert out["hdustatus"] == -1
    assert out["ok"] is False


def test_checksum_verifier_in_memory_methods() -> None:
    import numpy as np

    assert torchsky.ChecksumVerifier.verify_datasum(np.arange(3), "0") is True
    assert torchsky.ChecksumVerifier.verify_checksum({"CHECKSUM": "unused"}) is True


def test_checksum_multi_hdu_isolation(tmp_path):
    path = tmp_path / "multi_chk.fits"

    torchsky.write(
        str(path),
        [
            {"data": torch.arange(6, dtype=torch.float32).reshape(2, 3)},
            {"data": torch.arange(4, dtype=torch.float32).reshape(2, 2)},
        ],
        overwrite=True,
    )

    torchsky.write_checksums(str(path), hdu=1)

    primary = torchsky.verify_checksums(str(path), hdu=0)
    extension = torchsky.verify_checksums(str(path), hdu=1)
    assert primary == {"datastatus": 0, "hdustatus": 0, "ok": False}
    assert extension["ok"] is True

    torchsky.write_checksums(str(path), hdu=0)
    assert torchsky.verify_checksums(str(path), hdu=0)["ok"] is True
    assert torchsky.verify_checksums(str(path), hdu=1)["ok"] is True


def test_checksum_detects_data_payload_corruption(tmp_path):
    path = tmp_path / "data_corrupt.fits"
    data = torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32)

    torchsky.write(str(path), data, overwrite=True)
    torchsky.write_checksums(str(path), hdu=0)
    assert torchsky.verify_checksums(str(path), hdu=0)["ok"] is True

    needle = struct.pack(">f", 20.0)
    with open(path, "r+b") as f:
        raw = f.read()
        idx = raw.find(needle)
        assert idx != -1
        f.seek(idx)
        f.write(struct.pack(">f", 21.0))

    out = torchsky.verify_checksums(str(path), hdu=0)
    assert out["datastatus"] == -1
    assert out["hdustatus"] == -1
    assert out["ok"] is False


def test_checksum_rejects_invalid_hdu_argument(tmp_path):
    path = tmp_path / "invalid_hdu.fits"
    torchsky.write(str(path), torch.arange(3, dtype=torch.float32), overwrite=True)

    with pytest.raises(RuntimeError):
        torchsky.verify_checksums(str(path), hdu=-1)

    with pytest.raises((TypeError, RuntimeError)):
        torchsky.write_checksums(str(path), hdu=True)


def test_checksum_roundtrip_on_compressed_image_hdu(tmp_path):
    path = tmp_path / "compressed_chk.fits"
    data = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    torchsky.write(str(path), data, overwrite=True, compress=True)
    torchsky.write_checksums(str(path), hdu=1)

    out = torchsky.verify_checksums(str(path), hdu=1)
    assert out["ok"] is True


def test_table_mutation_invalidates_then_refreshes_checksum(tmp_path):
    import numpy as np

    path = tmp_path / "table_mutation_chk.fits"
    torchsky.write(
        str(path),
        {
            "ID": np.array([1, 2, 3], dtype=np.int32),
            "FLUX": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        },
        overwrite=True,
    )
    torchsky.write_checksums(str(path), hdu=1)
    assert torchsky.verify_checksums(str(path), hdu=1)["ok"] is True

    torchsky.table.update_rows(
        str(path),
        {"FLUX": np.array([9.0], dtype=np.float32)},
        row_slice=slice(1, 2),
        hdu=1,
    )
    stale = torchsky.verify_checksums(str(path), hdu=1)
    assert stale["ok"] is False

    torchsky.write_checksums(str(path), hdu=1)
    assert torchsky.verify_checksums(str(path), hdu=1)["ok"] is True
