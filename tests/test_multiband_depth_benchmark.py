from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from bench_multiband_depth import (  # noqa: E402
    _gate0_artifact,
    calibrate_purity_threshold,
    detection_statistic,
    find_detections,
    generate_truth_catalog,
    make_seed_manifest,
    maximum_cardinality_match,
    render_truth,
    score_detections,
    split_identifiers,
)


def test_seed_catalog_render_and_artifact_are_deterministic() -> None:
    first = _gate0_artifact(seed=101, size=32, bands=6, sources=10)
    second = _gate0_artifact(seed=101, size=32, bands=6, sources=10)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert len(set(first["seeds"].values())) == 4


def test_truth_catalog_has_required_strata_and_conserves_flux() -> None:
    catalog = generate_truth_catalog(
        height=48, width=48, bands=6, source_count=10, seed=81
    )
    assert {
        "continuum_point",
        "compact_galaxy",
        "emission_line",
        "blue_dropout",
        "red_dropout",
    }.issubset(catalog.kind)
    cube = render_truth(catalog, height=48, width=48, spatial_oversample=4)
    torch.testing.assert_close(
        cube.sum(dim=(-2, -1)),
        catalog.band_flux.sum(dim=0),
        rtol=2e-12,
        atol=2e-12,
    )
    assert 2.0 < torch.linalg.vector_norm(catalog.xy[-1] - catalog.xy[-2]) < 4.0


def test_blank_truth_scene() -> None:
    catalog = generate_truth_catalog(
        height=32, width=32, bands=3, source_count=0, seed=4
    )
    cube = render_truth(catalog, height=32, width=32)
    assert cube.shape == (3, 32, 32)
    assert torch.count_nonzero(cube) == 0


def test_split_is_disjoint_complete_and_deterministic() -> None:
    identifiers = [f"scene-{index}" for index in range(20)]
    split = split_identifiers(identifiers, seed=19)
    assert split == split_identifiers(identifiers, seed=19)
    assert set(split.train).isdisjoint(split.validation)
    assert set(split.train).isdisjoint(split.test)
    assert set(split.validation).isdisjoint(split.test)
    assert set(split.train + split.validation + split.test) == set(identifiers)


def test_detector_recovers_noiseless_bright_sources_and_null_is_empty() -> None:
    catalog = generate_truth_catalog(
        height=48,
        width=48,
        bands=6,
        source_count=1,
        seed=5,
        include_close_pair=False,
    )
    cube = render_truth(catalog, height=48, width=48)
    statistic = detection_statistic(cube, 0.01)
    detections, scores = find_detections(
        statistic, threshold=float(statistic.max()) * 0.8, nms_radius_px=2
    )
    match = maximum_cardinality_match(catalog.xy, detections, radius_px=1.0)
    assert len(match.pairs) == 1
    assert match.distances_px[0] < 0.75
    assert scores.shape == (1,)

    null_statistic = detection_statistic(torch.zeros_like(cube), 0.01)
    null_detections, _ = find_detections(null_statistic, threshold=1.0)
    assert null_detections.shape == (0, 2)


def test_matching_maximizes_cardinality_for_close_pair_and_counts_duplicate() -> None:
    truth = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    detections = torch.tensor(
        [[0.45, 0.0], [0.1, 0.0], [0.0, 0.2]], dtype=torch.float64
    )
    match = maximum_cardinality_match(truth, detections, radius_px=0.65)
    assert match.pairs == ((0, 1), (1, 0))
    assert match.unmatched_truth == ()
    assert match.unmatched_detections == (2,)
    score = score_detections(
        truth, detections, radius_px=0.65, area_square_arcmin=2.0
    )
    assert score["completeness"] == 1.0
    assert score["purity"] == pytest.approx(2 / 3)
    assert score["false_detections_per_square_arcmin"] == 0.5


def test_matching_is_order_invariant_in_cardinality() -> None:
    truth = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    detections = torch.tensor([[0.6, 0.0], [1.5, 0.0], [2.2, 0.0]])
    first = maximum_cardinality_match(truth, detections, radius_px=0.65)
    second = maximum_cardinality_match(
        truth.flip(0), detections.flip(0), radius_px=0.65
    )
    assert len(first.pairs) == len(second.pairs) == 3


def test_validation_purity_threshold_uses_lowest_accepted_score() -> None:
    scores = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0])
    labels = torch.tensor([True, True, False, True, False])
    assert calibrate_purity_threshold(scores, labels, target_purity=0.75) == 6.0
    assert calibrate_purity_threshold(scores, labels, target_purity=0.9) == 8.0
    assert calibrate_purity_threshold(
        scores, torch.zeros_like(labels), target_purity=0.95
    ) == float("inf")
    tied_scores = torch.tensor([9.0, 8.0, 8.0])
    tied_labels = torch.tensor([True, True, False])
    assert (
        calibrate_purity_threshold(tied_scores, tied_labels, target_purity=0.75)
        == 9.0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_detector_and_matching_cpu_cuda_parity() -> None:
    generator = torch.Generator().manual_seed(92)
    cube = torch.randn((4, 32, 32), generator=generator, dtype=torch.float64)
    cpu_statistic = detection_statistic(cube, torch.tensor([1.0, 1.2, 0.8, 1.5]))
    cuda_statistic = detection_statistic(
        cube.cuda(), torch.tensor([1.0, 1.2, 0.8, 1.5], device="cuda")
    )
    torch.testing.assert_close(cpu_statistic, cuda_statistic.cpu(), rtol=1e-12, atol=1e-12)
    cpu_xy, cpu_scores = find_detections(cpu_statistic, threshold=5.0)
    cuda_xy, cuda_scores = find_detections(cuda_statistic, threshold=5.0)
    torch.testing.assert_close(cpu_xy, cuda_xy.cpu())
    torch.testing.assert_close(cpu_scores, cuda_scores.cpu(), rtol=1e-12, atol=1e-12)
    assert maximum_cardinality_match(cpu_xy, cpu_xy, radius_px=0.0).pairs == tuple(
        (index, index) for index in range(cpu_xy.shape[0])
    )


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="root_seed"):
        make_seed_manifest(-1)
    with pytest.raises(ValueError, match="unique"):
        split_identifiers(["x", "x"], seed=1)
    with pytest.raises(ValueError, match="below one"):
        split_identifiers(["a", "b", "c"], seed=1, validation_fraction=0.5, test_fraction=0.5)
    with pytest.raises(ValueError, match="at least three"):
        generate_truth_catalog(height=32, width=32, bands=2, source_count=1, seed=1)
