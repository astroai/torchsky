from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("requests")


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from download_jwst_smacs0723_mast import (  # noqa: E402
    MastProduct,
    _detector_module_tokens,
    download_product,
    select_products,
)


class _FakeResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int = 0):
        del chunk_size
        yield from self._chunks


def test_download_product_renames_complete_part_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    product = MastProduct(
        parent_obsid=1,
        product_filename="test_cal.fits",
        data_uri="mast:test",
        product_type="SCIENCE",
        subgroup="CAL",
        size=6,
        filters="F277W",
    )
    monkeypatch.setattr(
        "download_jwst_smacs0723_mast.requests.get",
        lambda *args, **kwargs: _FakeResponse([b"abc", b"def"]),
    )

    dest = download_product(product, tmp_path, overwrite=False)

    assert dest.exists()
    assert dest.read_bytes() == b"abcdef"
    assert not dest.with_suffix(".fits.part").exists()


def test_download_product_rejects_truncated_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    product = MastProduct(
        parent_obsid=1,
        product_filename="test_cal.fits",
        data_uri="mast:test",
        product_type="SCIENCE",
        subgroup="CAL",
        size=6,
        filters="F277W",
    )
    monkeypatch.setattr(
        "download_jwst_smacs0723_mast.requests.get",
        lambda *args, **kwargs: _FakeResponse([b"abc"]),
    )

    with pytest.raises(RuntimeError, match="downloaded size mismatch"):
        download_product(product, tmp_path, overwrite=False)

    assert not (tmp_path / "test_cal.fits").exists()


def test_select_products_can_filter_by_filename(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "download_jwst_smacs0723_mast.query_nircam_observations",
        lambda **kwargs: [{"obsid": 1, "filters": "F277W"}],
    )
    monkeypatch.setattr(
        "download_jwst_smacs0723_mast.query_products",
        lambda obsid: [
            MastProduct(1, "a_nrcalong_cal.fits", "mast:a", "SCIENCE", "CAL", 1, "F277W"),
            MastProduct(1, "a_nrcblong_cal.fits", "mast:b", "SCIENCE", "CAL", 1, "F277W"),
        ],
    )

    chosen = select_products(
        band="F277W",
        subgroup="CAL",
        max_exposures=10,
        filename_contains="nrcblong",
    )

    assert [product.product_filename for product in chosen] == ["a_nrcblong_cal.fits"]


def test_select_products_matches_prefixed_subgroup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "download_jwst_smacs0723_mast.query_nircam_observations",
        lambda **kwargs: [{"obsid": 1, "filters": "F277W"}],
    )
    monkeypatch.setattr(
        "download_jwst_smacs0723_mast.query_products",
        lambda obsid: [
            MastProduct(1, "a_nrcblong_o001_crf.fits", "mast:a", "SCIENCE", "ANNNN_CRF", 1, "F277W"),
            MastProduct(1, "a_nrcblong_cal.fits", "mast:b", "SCIENCE", "CAL", 1, "F277W"),
        ],
    )

    chosen = select_products(
        band="F277W",
        subgroup="CRF",
        max_exposures=10,
        filename_contains="nrcblong",
    )

    assert [product.product_filename for product in chosen] == ["a_nrcblong_o001_crf.fits"]


def test_detector_module_tokens_support_short_and_long_bands() -> None:
    assert _detector_module_tokens("F277W", "b") == ("nrcblong",)
    assert _detector_module_tokens("F200W", "b") == ("nrcb1", "nrcb2", "nrcb3", "nrcb4")
