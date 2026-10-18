import torch

def test_hpx_cpp_project_marks_polar_out_of_facet_points_invalid() -> None:
    import torchsky.cpp as cpp

    xi = torch.tensor([45.0, 56.0, 0.0], dtype=torch.float64)
    eta = torch.tensor([80.0, 80.0, 80.0], dtype=torch.float64)

    phi, theta = cpp.wcs_hpx_project(xi, eta, 4.0, 3.0)

    assert torch.isfinite(phi[0]) and torch.isfinite(theta[0])
    assert torch.isnan(phi[1]) and torch.isnan(theta[1])
    assert torch.isnan(phi[2]) and torch.isnan(theta[2])
