from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
import torchfits

_SCRIPT = Path(__file__).parents[1] / "scripts" / "apply_wcs_to_mef.py"
_SPEC = importlib.util.spec_from_file_location("apply_wcs_to_mef", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_HELPER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPER)
_read_external_header = _HELPER._read_external_header
_read_wcs_file = _HELPER._read_wcs_file
cmd_propagate = _HELPER.cmd_propagate


def _wcs_header(extname: str, *, extver: int = 1) -> torchfits.Header:
    header = torchfits.Header(
        {
            "EXTNAME": extname,
            "EXTVER": extver,
            "OBSERVER": "torchsky test",
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRPIX1": 2.0,
            "CRPIX2": 2.0,
            "CRVAL1": 150.0,
            "CRVAL2": 45.0,
            "CD1_1": -0.001,
            "CD1_2": 0.0,
            "CD2_1": 0.0,
            "CD2_2": 0.001,
        }
    )
    return header


def test_external_wcs_and_scamp_header_card_layouts(tmp_path):
    values = {
        "CRVAL1": "150.25",
        "CRVAL2": "44.5",
        "CD1_1": "-0.001",
        "CD2_2": "0.001",
    }
    cards = [f"{key:<8}= {value}" for key, value in values.items()]
    scamp_path = tmp_path / "scamp.head"
    scamp_path.write_text("\n".join(cards + ["END"]) + "\n", encoding="ascii")
    packed_wcs_path = tmp_path / "solution.wcs"
    packed_wcs_path.write_text(
        "".join(card[:80].ljust(80) for card in cards + ["END"]),
        encoding="ascii",
    )

    for parsed in (
        _read_external_header(str(scamp_path)),
        _read_wcs_file(str(packed_wcs_path)),
    ):
        assert parsed["CRVAL1"] == 150.25
        assert parsed["CRVAL2"] == 44.5
        assert parsed["CD1_1"] == -0.001
        assert parsed["CD2_2"] == 0.001


def test_propagate_preserves_mef_payloads_headers_and_unsigned_data(tmp_path):
    source = tmp_path / "input.fits"
    output = tmp_path / "propagated.fits"
    science = (torch.arange(9, dtype=torch.int16).reshape(3, 3) + 300).to(
        torch.uint16
    )
    error = torch.arange(9, dtype=torch.float32).reshape(3, 3) / 4
    catalog = {
        "ID": torch.tensor([11, 12], dtype=torch.int32),
        "FLUX": torch.tensor([2.5, 4.5], dtype=torch.float32),
    }
    torchfits.write(
        source,
        torchfits.HDUList(
            [
                torchfits.TensorHDU(data=torch.empty(0)),
                torchfits.TensorHDU(data=science, header=_wcs_header("SCI", extver=2)),
                torchfits.TensorHDU(data=error, header=_wcs_header("ERR", extver=2)),
                torchfits.TableHDU(catalog, header={"EXTNAME": "CATALOG"}),
            ]
        ),
        overwrite=True,
    )
    source_science = torchfits.read(str(source), hdu="SCI")
    assert source_science.dtype == torch.uint16
    assert torch.equal(source_science, science)

    result = cmd_propagate(
        SimpleNamespace(
            mef=source,
            output=output,
            solved_wcs=None,
            ref_hdu=1,
            distortion="tpv",
            max_crpix_offset=float("inf"),
        )
    )
    assert result is None

    with torchfits.open(str(output)) as hdul:
        assert [hdu.header.get("EXTNAME", "PRIMARY") for hdu in hdul] == [
            "PRIMARY",
            "SCI",
            "ERR",
            "CATALOG",
        ]
        assert int(hdul["SCI"].header["EXTVER"]) == 2
        assert hdul["SCI"].header["OBSERVER"] == "torchsky test"
        assert hdul["CATALOG"].data["ID"].tolist() == [11, 12]
        assert hdul["CATALOG"].data["FLUX"].tolist() == [2.5, 4.5]
    propagated_science = torchfits.read(str(output), hdu="SCI")
    propagated_error = torchfits.read(str(output), hdu="ERR")
    assert propagated_science.dtype == torch.uint16
    assert torch.equal(propagated_science, source_science)
    assert torch.equal(propagated_error, error)
