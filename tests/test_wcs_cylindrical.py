import pytest
import torch

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
