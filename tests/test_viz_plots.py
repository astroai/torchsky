"""Tests for quick visualizers in torchsky.viz."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
import torch

from torchsky.spectral import SpectralAxis
import torchsky.viz as viz


def test_viz_imports() -> None:
    # Ensure they are exposed in __all__
    assert "plot_pseudo_rgb" in viz.__all__
    assert "plot_pixel_spectrum" in viz.__all__
    # Ensure lazy loading works
    assert viz.plot_pseudo_rgb is not None
    assert viz.plot_pixel_spectrum is not None


def test_plot_pseudo_rgb_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    L, H, W = 6, 8, 8
    cube = torch.randn(L, H, W)
    wave = torch.linspace(4000, 7000, L)
    axis = SpectralAxis(values=wave, unit="Angstrom", type="WAVE")

    mock_ax = MagicMock()
    mock_plt = MagicMock()

    # monkeypatch (not patch.dict) so only these two keys are touched: restoring
    # the whole of sys.modules drops lazily imported torch submodules and makes
    # the next import re-register their TORCH_LIBRARY namespaces.
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", mock_plt)
    monkeypatch.setitem(sys.modules, "matplotlib", MagicMock())
    res_ax = viz.plot_pseudo_rgb(cube, axis, ax=mock_ax)
    assert res_ax is mock_ax
    mock_ax.imshow.assert_called_once()
    # Verify the rgb array shape passed to imshow is (8, 8, 3)
    args, kwargs = mock_ax.imshow.call_args
    rgb_np = args[0]
    assert rgb_np.shape == (H, W, 3)


def test_plot_pixel_spectrum_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    L, H, W = 6, 8, 8
    cube = torch.randn(L, H, W)
    wave = torch.linspace(4000, 7000, L)
    axis = SpectralAxis(values=wave, unit="Angstrom", type="WAVE")

    mock_ax = MagicMock()
    mock_plt = MagicMock()

    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", mock_plt)
    monkeypatch.setitem(sys.modules, "matplotlib", MagicMock())
    res_ax = viz.plot_pixel_spectrum(cube, axis, y=2, x=3, ax=mock_ax)
    assert res_ax is mock_ax
    mock_ax.plot.assert_called_once()
    mock_ax.set_xlabel.assert_called_once()
    mock_ax.set_ylabel.assert_called_once()


def test_plot_pixel_spectrum_middle_axis_uses_yx_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    H, L, W = 2, 3, 4
    cube = torch.arange(H * L * W, dtype=torch.float32).reshape(H, L, W)
    wave = torch.linspace(4000, 7000, L)
    axis = SpectralAxis(values=wave, unit="Angstrom", type="WAVE")

    mock_ax = MagicMock()
    mock_plt = MagicMock()

    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", mock_plt)
    monkeypatch.setitem(sys.modules, "matplotlib", MagicMock())
    viz.plot_pixel_spectrum(
        cube,
        axis,
        y=1,
        x=2,
        spectral_axis_index=1,
        ax=mock_ax,
    )
    args, _kwargs = mock_ax.plot.call_args
    torch.testing.assert_close(
        torch.as_tensor(args[1]),
        cube[1, :, 2],
    )


def test_visualizers_raise_importerror_when_matplotlib_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    L, H, W = 6, 8, 8
    cube = torch.randn(L, H, W)
    wave = torch.linspace(4000, 7000, L)
    axis = SpectralAxis(values=wave, unit="Angstrom", type="WAVE")

    # Mock matplotlib as missing
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(ImportError, match="Matplotlib is required"):
        viz.plot_pseudo_rgb(cube, axis)

    with pytest.raises(ImportError, match="Matplotlib is required"):
        viz.plot_pixel_spectrum(cube, axis, 2, 3)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)
def test_display_limits_accept_mps_tensors() -> None:
    from torchsky.viz.plots import _limits

    image = torch.linspace(0.0, 1.0, 64 * 64).reshape(64, 64)
    assert _limits([image.to("mps")], (1.0, 99.0)) == pytest.approx(
        _limits([image], (1.0, 99.0))
    )
