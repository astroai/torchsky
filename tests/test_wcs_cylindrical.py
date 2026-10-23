import pytest
import torch
from astropy.wcs import WCS as AstropyWCS

from torchsky.wcs import WCS
from torchsky.wcs.cylindrical import deproject_cylindrical, project_cylindrical


def test_cylindrical_not_implemented() -> None:
    phi = torch.tensor([0.0])
    theta = torch.tensor([0.0])

    with pytest.raises(
        NotImplementedError, match="Cylindrical code UNK not implemented"
    ):
        project_cylindrical(phi, theta, "UNK")

    with pytest.raises(
        NotImplementedError, match="Cylindrical forward projection UNK not implemented"
    ):
        deproject_cylindrical(phi, theta, "UNK")


def test_mer_invalid_latitudes_are_nan_and_match_astropy() -> None:
    header = {
        "NAXIS": 2,
        "CTYPE1": "RA---MER",
        "CTYPE2": "DEC--MER",
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CRVAL1": 0.0,
        "CRVAL2": 0.0,
        "CD1_1": 1.0,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 1.0,
    }
    world = torch.tensor(
        [[10.0, 20.0], [10.0, 90.0], [10.0, -90.0], [10.0, 91.0], [10.0, -91.0]]
    )
    actual = WCS(header).world_to_pixel(world, origin=1)
    x, y = WCS(header).world_to_pixel(world[:, 0], world[:, 1], origin=1)
    separate = torch.stack((x, y), dim=-1)
    expected = torch.from_numpy(AstropyWCS(header).all_world2pix(world.numpy(), 1))
    torch.testing.assert_close(actual[:1], expected[:1], atol=1e-10, rtol=0.0)
    torch.testing.assert_close(separate[:1], expected[:1], atol=1e-10, rtol=0.0)
    assert torch.isnan(actual[1:]).all()
    assert torch.isnan(separate[1:]).all()
    assert torch.isnan(expected[1:]).all()

    xi, eta = deproject_cylindrical(world[:, 0], world[:, 1], "MER")
    assert torch.isfinite(xi[0]) and torch.isfinite(eta[0])
    assert torch.isnan(xi[1:]).all() and torch.isnan(eta[1:]).all()


@pytest.mark.parametrize(
    ("projection", "parameters"),
    [("CEA", {}), ("CYP", {"PV2_1": 1.0, "PV2_2": 1.0})],
)
def test_center_cylindrical_matches_astropy_for_stacked_and_separate_inputs(
    projection: str, parameters: dict[str, float]
) -> None:
    header = {
        "NAXIS": 2,
        "CTYPE1": f"RA---{projection}",
        "CTYPE2": f"DEC--{projection}",
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CRVAL1": 0.0,
        "CRVAL2": 0.0,
        "CD1_1": 1.0,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 1.0,
        **parameters,
    }
    world = torch.tensor([[10.0, 20.0], [-30.0, -45.0]], dtype=torch.float64)
    wcs = WCS(header)
    stacked = wcs.world_to_pixel(world, origin=1)
    x, y = wcs.world_to_pixel(world[:, 0], world[:, 1], origin=1)
    separate = torch.stack((x, y), dim=-1)
    expected = torch.from_numpy(AstropyWCS(header).all_world2pix(world.numpy(), 1))
    torch.testing.assert_close(stacked, expected, atol=1e-10, rtol=0.0)
    torch.testing.assert_close(separate, expected, atol=1e-10, rtol=0.0)
