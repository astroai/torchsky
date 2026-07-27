import importlib.util
from pathlib import Path
import sys

from astropy.table import Table


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_wcs_aware_source_catalogs.py"
spec = importlib.util.spec_from_file_location("catalog_compare", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _catalog(rows):
    table = Table()
    table["image_label"] = [row[0] for row in rows]
    table["source_id"] = [row[1] for row in rows]
    table["ra_deg"] = [row[2] for row in rows]
    table["dec_deg"] = [row[3] for row in rows]
    table["segment_flux"] = [row[4] for row in rows]
    table["peak_sigma"] = [row[5] for row in rows]
    table["area_pix2"] = [row[6] for row in rows]
    return table


def test_one_to_one_crossmatch_selects_closest_unique_pairs() -> None:
    source = _catalog(
        [
            ("A1", 1, 10.000000, 20.0, 10.0, 8.0, 5.0),
            ("A1", 2, 10.000010, 20.0, 20.0, 12.0, 6.0),
        ]
    )
    reference = _catalog(
        [
            ("UPDC", 101, 10.000001, 20.0, 11.0, 8.5, 5.5),
            ("UPDC", 102, 10.000011, 20.0, 19.0, 11.0, 6.5),
        ]
    )

    matches = module._one_to_one_sky_matches(
        source,
        reference,
        radius_arcsec=0.1,
    )

    assert len(matches) == 2
    assert {(si, ri) for si, ri, _ in matches} == {(0, 0), (1, 1)}
    assert all(separation < 0.1 for _, _, separation in matches)


def test_match_summary_counts_unmatched_without_calling_them_false_positives() -> None:
    source = _catalog(
        [
            ("A1", 1, 10.000000, 20.0, 10.0, 7.0, 5.0),
            ("A1", 2, 10.001000, 20.0, 20.0, 15.0, 6.0),
        ]
    )
    reference = _catalog(
        [
            ("UPDC", 101, 10.000001, 20.0, 11.0, 7.5, 5.5),
        ]
    )
    matches = module._one_to_one_sky_matches(source, reference, radius_arcsec=0.1)
    summary = module._match_summary(
        source,
        reference,
        matches,
        radius_arcsec=0.1,
        snr_edges=[6.0, 8.0, 20.0, float("inf")],
    )

    assert summary["matched_pairs"] == 1
    assert summary["source_only"] == 1
    assert summary["reference_only"] == 0
    assert summary["reference_recovered_fraction"] == 1.0
