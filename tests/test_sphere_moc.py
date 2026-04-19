import json

import pytest
import torch
from torchsky.sphere.moc import MOC, nest2uniq, _max_order_from_uniq, _uniq_to_ranges, _ranges_to_uniq, _normalize_moc
from torchsky.sphere import HealSparseMap

def test_uniq_to_ranges_empty():
    ranges = _uniq_to_ranges(torch.tensor([], dtype=torch.int64), 10)
    assert ranges == []

def test_ranges_to_uniq_empty():
    uniqs = _ranges_to_uniq([], 10)
    assert uniqs.numel() == 0

def test_max_order_from_uniq_empty():
    assert _max_order_from_uniq(torch.tensor([], dtype=torch.int64)) == 0
    assert _max_order_from_uniq(torch.tensor([])) == 0
    assert _max_order_from_uniq(torch.empty((0,))) == 0

def test_normalize_moc_empty():
    uniq = torch.tensor([], dtype=torch.int64)
    assert _normalize_moc(uniq).numel() == 0

def test_moc_max_order_empty():
    moc = MOC(torch.tensor([], dtype=torch.int64))
    assert moc.max_order == 0

def test_moc_contains_empty():
    moc = MOC(torch.tensor([], dtype=torch.int64))
    assert not moc.contains(0.0, 0.0).item()

def test_moc_area_empty():
    moc = MOC(torch.tensor([], dtype=torch.int64))
    assert moc.area == 0.0
    assert moc.area_sq_deg == 0.0

def test_moc_difference_complex():
    # Test complex difference scenario
    moc1 = MOC(nest2uniq(4, torch.tensor([0, 1, 2])))
    moc2 = MOC(nest2uniq(4, torch.tensor([1, 3])))
    diff = moc1.difference(moc2)
    assert diff.uniq.numel() == 2

def test_moc_from_sparse_map_empty():
    smap = HealSparseMap.convert_healpix_map(torch.zeros(12), nside_coverage=1, sentinel=0)
    moc = MOC.from_sparse_map(smap)
    assert moc.uniq.numel() == 0

def test_moc_from_threshold():
    pixels = torch.tensor([0, 1, 2])
    values = torch.tensor([10.0, 5.0, 20.0])
    smap = HealSparseMap.from_pixels(pixels, nside_sparse=4, nside_coverage=1, values=values)

    # Test op >
    moc_gt = MOC.from_threshold(smap, 10.0, op=">")
    assert moc_gt.uniq.numel() > 0

    # Test op >=
    moc_ge = MOC.from_threshold(smap, 10.0, op=">=")
    assert moc_ge.uniq.numel() > 0

    # Test op <
    moc_lt = MOC.from_threshold(smap, 10.0, op="<")
    assert moc_lt.uniq.numel() > 0

    # Test op <=
    moc_le = MOC.from_threshold(smap, 10.0, op="<=")
    assert moc_le.uniq.numel() > 0

    # Test invalid op
    with pytest.raises(ValueError):
        MOC.from_threshold(smap, 10.0, op="==")

def test_moc_from_threshold_empty():
    pixels = torch.tensor([0, 1, 2])
    values = torch.tensor([10.0, 5.0, 20.0])
    smap = HealSparseMap.from_pixels(pixels, nside_sparse=4, nside_coverage=1, values=values)

    moc = MOC.from_threshold(smap, 100.0, op=">")
    assert moc.uniq.numel() == 0

def test_moc_read_fits_fallback(tmp_path):
    """Non-standard UNIQ column name: first int64 column is used."""
    import torch
    import torchsky

    path = str(tmp_path / "test.fits")
    torchsky.write(
        path,
        {"OLD_UNIQ": torch.tensor([4, 16, 17], dtype=torch.int64)},
        header={"EXTNAME": "MOC"},
        overwrite=True,
    )
    m = MOC.read_fits(path)
    # Values normalize/merge into a valid MOC (exact UNIQ count may shrink).
    assert m.uniq.numel() >= 1
    assert m.uniq.dtype == torch.int64

def test_moc_boundary_empty():
    moc = MOC(torch.tensor([], dtype=torch.int64))
    assert moc.boundary() == []

def test_moc_boundary_complex():
    moc = MOC(nest2uniq(2, torch.tensor([0, 1, 2])))
    boundary = moc.boundary()
    assert len(boundary) > 0

def test_moc_to_ascii_empty():
    moc = MOC(torch.tensor([], dtype=torch.int64))
    assert moc.to_ascii() == ""

def test_moc_from_ascii_empty():
    assert MOC.from_ascii("").uniq.numel() == 0
    assert MOC.from_ascii("   ").uniq.numel() == 0
    assert MOC.from_ascii("0/").uniq.numel() == 0

def test_moc_to_json_empty():
    moc = MOC(torch.tensor([], dtype=torch.int64))
    assert moc.to_json() == "{}"

def test_moc_from_json_empty():
    assert MOC.from_json("").uniq.numel() == 0
    assert MOC.from_json("{}").uniq.numel() == 0

def test_moc_from_json_invalid():
    with pytest.raises(ValueError):
        MOC.from_json('"a string"')

def test_moc_from_json_empty_pixels():
    js = json.dumps({"0": []})
    assert MOC.from_json(js).uniq.numel() == 0
