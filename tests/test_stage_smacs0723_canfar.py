from __future__ import annotations

from pathlib import Path
import sys

import pytest

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from stage_smacs0723_canfar import (  # noqa: E402
    expected_cal_filenames,
    is_complete_fits,
    link_existing_aligned,
    link_existing_crfs,
)


def test_expected_cal_grid_is_complete_and_unique() -> None:
    filenames = expected_cal_filenames()
    assert len(filenames) == len(set(filenames)) == 135
    assert sum("_nrcblong_" in filename for filename in filenames) == 27
    for pattern in range(1, 10):
        assert (
            sum(f"_{pattern:05d}_" in filename for filename in filenames)
            == 15
        )


def test_fits_gate_and_crf_links_are_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "existing"
    source.mkdir()
    crf = source / "example_crf.fits"
    crf.write_bytes(b"SIMPLE  " + b" " * 2872)
    invalid = source / "truncated_cal.fits"
    invalid.write_bytes(b"SIMPLE  ")

    destination = tmp_path / "canonical" / "crf"
    first = link_existing_crfs(source, destination)
    second = link_existing_crfs(source, destination)

    assert is_complete_fits(crf)
    assert not is_complete_fits(invalid)
    assert first == second
    assert len(first) == 1
    assert (destination / crf.name).resolve() == crf.resolve()


def test_crf_link_refuses_to_replace_regular_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "same_crf.fits").write_bytes(b"SIMPLE  " + b" " * 2872)
    (destination / "same_crf.fits").write_text("do not replace")

    with pytest.raises(RuntimeError, match="refusing"):
        link_existing_crfs(source, destination)


def test_aligned_products_use_separate_canonical_links(tmp_path: Path) -> None:
    source = tmp_path / "alignment"
    source.mkdir()
    aligned = source / "001_example_rshift_tweakreg.fits"
    aligned.write_bytes(b"SIMPLE  " + b" " * 2872)

    records = link_existing_aligned(source, tmp_path / "canonical" / "aligned")

    assert len(records) == 1
    assert Path(str(records[0]["link"])).resolve() == aligned.resolve()
