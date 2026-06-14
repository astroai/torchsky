from __future__ import annotations

from typing import Any, Literal
import torch

from ..spectral import SpectralAxis


def plot_pseudo_rgb(
    cube: torch.Tensor,
    spectral_axis: SpectralAxis,
    *,
    spectral_axis_index: int = 0,
    stretch: Literal["linear", "asinh", "log", "sqrt"] = "asinh",
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Plot a pseudo-RGB composite image from a 3D spectral cube.

    Splits the spectral axis into three equal wavelength bands (Blue, Green, Red)
    and collapses each band to produce a composite RGB image.

    Args:
        cube: 3D data cube tensor.
        spectral_axis: SpectralAxis associated with the spectral dimension.
        spectral_axis_index: Dimension index of the spectral axis in the cube (0, 1, or 2).
        stretch: Non-linear stretch to apply: 'linear', 'asinh', 'log', or 'sqrt'.
        ax: Optional matplotlib Axes object to plot on.
        kwargs: Additional arguments passed to matplotlib's imshow.

    Returns:
        The matplotlib Axes object.
    """
    if cube.ndim != 3:
        raise ValueError(f"cube must be 3D, got shape {tuple(cube.shape)}")
    if spectral_axis_index not in (0, 1, 2):
        raise ValueError("spectral_axis_index must be 0, 1, or 2")

    L = cube.shape[spectral_axis_index]
    if len(spectral_axis.values) != L:
        raise ValueError(
            f"spectral_axis values length {len(spectral_axis.values)} "
            f"does not match cube dimension length {L}"
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "Matplotlib is required for plotting visualizers. "
            "Install it via: pip install matplotlib"
        ) from e

    # Convert values to wavelength (in Angstroms) for band selection
    try:
        wave = spectral_axis.to_wavelength().detach().cpu()
    except Exception:
        wave = spectral_axis.values.detach().cpu()

    sorted_wave, sorted_idx = torch.sort(wave)
    min_w = sorted_wave[0].item()
    max_w = sorted_wave[-1].item()

    if L >= 3:
        w_range = max_w - min_w
        if w_range > 0:
            b_mask = (wave >= min_w) & (wave < min_w + w_range / 3)
            g_mask = (wave >= min_w + w_range / 3) & (wave < min_w + 2 * w_range / 3)
            r_mask = (wave >= min_w + 2 * w_range / 3) & (wave <= max_w)

            if not b_mask.any() or not g_mask.any() or not r_mask.any():
                b_idx = sorted_idx[: L // 3]
                g_idx = sorted_idx[L // 3 : 2 * (L // 3)]
                r_idx = sorted_idx[2 * (L // 3) :]
            else:
                b_idx = torch.where(b_mask)[0]
                g_idx = torch.where(g_mask)[0]
                r_idx = torch.where(r_mask)[0]
        else:
            b_idx = sorted_idx[: L // 3]
            g_idx = sorted_idx[L // 3 : 2 * (L // 3)]
            r_idx = sorted_idx[2 * (L // 3) :]
    else:
        if L == 1:
            b_idx = g_idx = r_idx = torch.tensor([0])
        else:  # L == 2
            b_idx = torch.tensor([0])
            g_idx = torch.tensor([0, 1])
            r_idx = torch.tensor([1])

    def get_band_image(indices: torch.Tensor) -> torch.Tensor:
        if len(indices) == 0:
            shape = [d for i, d in enumerate(cube.shape) if i != spectral_axis_index]
            return torch.zeros(shape[0], shape[1], device=cube.device)
        selected = torch.index_select(
            cube, spectral_axis_index, indices.to(cube.device)
        )
        return torch.mean(selected, dim=spectral_axis_index)

    r_img = get_band_image(r_idx)
    g_img = get_band_image(g_idx)
    b_img = get_band_image(b_idx)

    def apply_stretch(img: torch.Tensor, mode: str) -> torch.Tensor:
        img = img.detach().clone().float()
        # Avoid negative values before non-linear scaling
        vmin = img.min()
        img = img - vmin
        if mode == "asinh":
            img = torch.asinh(img)
        elif mode == "log":
            img = torch.log1p(img)
        elif mode == "sqrt":
            img = torch.sqrt(img)
        elif mode != "linear":
            raise ValueError(f"Unknown stretch mode: {mode}")

        vmax = img.max()
        if vmax > 0:
            img = img / vmax
        return img

    r_stretched = apply_stretch(r_img, stretch)
    g_stretched = apply_stretch(g_img, stretch)
    b_stretched = apply_stretch(b_img, stretch)

    rgb = torch.stack([r_stretched, g_stretched, b_stretched], dim=-1)
    rgb_np = rgb.cpu().numpy()

    if ax is None:
        _, ax = plt.subplots()

    imshow_kwargs = {"origin": "lower"}
    imshow_kwargs.update(kwargs)
    ax.imshow(rgb_np, **imshow_kwargs)
    return ax


def plot_pixel_spectrum(
    cube: torch.Tensor,
    spectral_axis: SpectralAxis,
    y: int,
    x: int,
    *,
    spectral_axis_index: int = 0,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Plot the 1D spectrum at a specific pixel (y, x) of a data cube.

    Args:
        cube: 3D data cube tensor.
        spectral_axis: SpectralAxis associated with the spectral dimension.
        y: Pixel y-coordinate.
        x: Pixel x-coordinate.
        spectral_axis_index: Dimension index of the spectral axis in the cube (0, 1, or 2).
        ax: Optional matplotlib Axes object to plot on.
        kwargs: Additional arguments passed to matplotlib's plot function.

    Returns:
        The matplotlib Axes object.
    """
    if cube.ndim != 3:
        raise ValueError(f"cube must be 3D, got shape {tuple(cube.shape)}")
    if spectral_axis_index not in (0, 1, 2):
        raise ValueError("spectral_axis_index must be 0, 1, or 2")

    L = cube.shape[spectral_axis_index]
    if len(spectral_axis.values) != L:
        raise ValueError(
            f"spectral_axis values length {len(spectral_axis.values)} "
            f"does not match cube dimension length {L}"
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "Matplotlib is required for plotting visualizers. "
            "Install it via: pip install matplotlib"
        ) from e

    if spectral_axis_index == 0:
        flux = cube[:, y, x]
    elif spectral_axis_index == 1:
        flux = cube[y, :, x]
    elif spectral_axis_index == 2:
        flux = cube[y, x, :]

    flux_np = flux.detach().cpu().numpy()
    wave_np = spectral_axis.values.detach().cpu().numpy()

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(wave_np, flux_np, **kwargs)
    ax.set_xlabel(f"{spectral_axis.type} ({spectral_axis.unit})")
    ax.set_ylabel("Flux")
    return ax
