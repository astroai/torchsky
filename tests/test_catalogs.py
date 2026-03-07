import torch

from torchsky.catalogs import crossmatch_sky


def test_crossmatch_sky_returns_nearest_matches_within_radius() -> None:
    left_ra = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
    left_dec = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    right_ra = torch.tensor([10.0001, 25.0, 29.9998], dtype=torch.float64)
    right_dec = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)

    result = crossmatch_sky(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=2.0,
    )

    assert result.left_index.tolist() == [0, 2]
    assert result.right_index.tolist() == [0, 2]
    assert torch.all(result.separation_arcsec < 2.0)
