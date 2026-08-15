from __future__ import annotations

import numpy as np
import pytest
import torch

from torchsky.wcs import sample_gwcs_detector_to_tangent


def _smooth_reference(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        1.0 + 0.02 * x + 2.0e-7 * x**3 + 0.001 * y,
        -2.0 + 0.015 * y + 1.0e-7 * y**3 + 0.0005 * x,
    )


def test_gwcs_sampling_refines_and_records_edge_chip_provenance() -> None:
    result = sample_gwcs_detector_to_tangent(
        _smooth_reference,
        detector_bounds=((0.0, 64.0), (0.0, 64.0)),
        initial_grid_shape=(5, 5),
        maximum_grid_shape=(17, 17),
        interpolation="bicubic",
        coordinates="plane_arcsec",
        random_validation_points=128,
        chip_boundaries_x=(32.0,),
        p99_limit_mas=1.0e-5,
        max_limit_mas=1.0e-4,
        calibration_provenance={"crds_context": "jwst_test.pmap", "gwcs": "test"},
    )

    assert result.passed
    assert set(result.validation) == {"random", "edges", "chip_boundaries"}
    assert result.provenance_dict()["calibration"] == {
        "crds_context": "jwst_test.pmap",
        "gwcs": "test",
    }
    points = torch.tensor([[3.5, 8.25], [32.0, 60.0]], dtype=torch.float64)
    expected = torch.as_tensor(
        np.stack(_smooth_reference(points[:, 0].numpy(), points[:, 1].numpy()), axis=-1)
    )
    torch.testing.assert_close(result.transform(points), expected, atol=1e-10, rtol=1e-10)


def test_gwcs_sampling_fails_closed_when_maximum_grid_cannot_meet_gate() -> None:
    def oscillatory(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return x + 0.01 * np.sin(x), y + 0.01 * np.sin(y)

    with pytest.raises(ValueError, match="maximum grid"):
        sample_gwcs_detector_to_tangent(
            oscillatory,
            detector_bounds=((0.0, 8.0), (0.0, 8.0)),
            initial_grid_shape=(4, 4),
            maximum_grid_shape=(4, 4),
            random_validation_points=32,
            p99_limit_mas=1.0e-6,
            max_limit_mas=1.0e-6,
        )
