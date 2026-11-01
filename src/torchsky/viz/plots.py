"""Publication-quality quick-look plots for spectral cubes.

Both helpers theme their artists explicitly (fonts, tick colours, graticule)
rather than through global ``rcParams``, so a figure keeps its look whenever it
is saved later. When you pass your own ``ax`` only labels and annotations are
added; the axes styling is left to the caller.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

from ..spectral import SpectralAxis
from . import theme as _theme

StretchName = Literal["linear", "asinh", "log", "sqrt"]

# Asinh softening width as a fraction of the display range (Lupton 2004).
_ASINH_WIDTH = 0.05


def _stretch_unit(x: Tensor, stretch: str) -> Tensor:
    """Apply a display stretch to ``x`` already normalized to ``[0, 1]``."""
    if stretch == "linear":
        return x
    if stretch == "asinh":
        return torch.asinh(x / _ASINH_WIDTH) / math.asinh(1.0 / _ASINH_WIDTH)
    if stretch == "log":
        return torch.log1p(x / _ASINH_WIDTH) / math.log1p(1.0 / _ASINH_WIDTH)
    if stretch == "sqrt":
        return torch.sqrt(x)
    raise ValueError(
        f"Unknown stretch mode: {stretch!r}. Choose linear, asinh, log or sqrt."
    )


def _limits(images: list[Tensor], percentile: tuple[float, float]) -> tuple[float, float]:
    finite = torch.cat(
        [img[torch.isfinite(img)].flatten().detach().cpu() for img in images]
    )
    if finite.numel() == 0:
        return 0.0, 1.0
    stride = max(1, finite.numel() // (1 << 20))
    q = torch.quantile(
        finite[::stride].double(),
        torch.tensor([percentile[0] / 100.0, percentile[1] / 100.0]).double(),
    )
    lo, hi = float(q[0]), float(q[1])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _band_labels(
    labels: list[tuple[int, int]], wave: Tensor, unit: str
) -> list[str]:
    names = ("B", "G", "R")
    out = []
    for name, idx in zip(names, labels, strict=False):
        start = int(idx[0])
        stop = int(idx[-1])
        lo = float(wave[start])
        hi = float(wave[stop])
        out.append(f"{name}  {min(lo, hi):.0f}\u2013{max(lo, hi):.0f} {unit}")
    return out


def plot_pseudo_rgb(
    cube: torch.Tensor,
    spectral_axis: SpectralAxis,
    *,
    spectral_axis_index: int = 0,
    stretch: str = "asinh",
    ax: Any = None,
    theme: str = "dark",
    title: str | None = None,
    subtitle: str | None = None,
    percentile: tuple[float, float] = (0.5, 99.5),
    show_band_key: bool = True,
    save: str | Path | None = None,
    dpi: int = 200,
    **kwargs: Any,
) -> Any:
    """Render a pseudo-RGB composite from a spectral cube.

    The spectral axis is split into three equal-count bands (blue, green, red)
    so every channel carries the same amount of data. By default all three
    channels share one display range, which keeps colour ratios meaningful;
    ``percentile`` clips the bright tail.

    Args:
        cube: 3D data cube tensor.
        spectral_axis: SpectralAxis for the spectral dimension.
        spectral_axis_index: Dimension index of the spectral axis (0, 1 or 2).
        stretch: ``asinh`` (default), ``linear``, ``log`` or ``sqrt``.
        ax: Optional matplotlib axes; when omitted a themed figure is created.
        theme: ``dark`` (default) or ``paper``.
        title: Figure headline.
        subtitle: Quieter second line under the headline.
        percentile: Display limits in percentiles of the finite data.
        show_band_key: Print the wavelength range of each channel.
        save: Path to write the figure; returns the axes either way.
        dpi: Output resolution used when creating/saving the figure.
        kwargs: Forwarded to ``imshow``.

    Returns:
        The matplotlib axes.
    """
    if cube.ndim != 3:
        raise ValueError(f"cube must be 3D, got shape {tuple(cube.shape)}")
    if spectral_axis_index not in (0, 1, 2):
        raise ValueError("spectral_axis_index must be 0, 1, or 2")

    n_spectral = cube.shape[spectral_axis_index]
    if len(spectral_axis.values) != n_spectral:
        raise ValueError(
            f"spectral_axis values length {len(spectral_axis.values)} "
            f"does not match cube dimension length {n_spectral}"
        )
    if not 0.0 <= percentile[0] < percentile[1] <= 100.0:
        raise ValueError("percentile must satisfy 0 <= lo < hi <= 100")

    plt = _theme.pyplot()

    wave = spectral_axis.values.detach().to("cpu", torch.float64).flatten()
    try:
        wave_display = spectral_axis.to_wavelength().detach().to("cpu").flatten()
        unit = spectral_axis.unit
    except Exception:
        wave_display = wave
        unit = spectral_axis.unit
    order = torch.argsort(wave_display)
    groups = [g for g in torch.tensor_split(order, 3) if g.numel() > 0]

    def band_image(indices: Tensor) -> Tensor:
        selected = torch.index_select(
            cube, spectral_axis_index, indices.to(cube.device)
        )
        return selected.mean(dim=spectral_axis_index)

    channels = {name: band_image(idx) for name, idx in zip(("r", "g", "b"), groups[::-1])}
    images = list(channels.values())
    lo, hi = _limits(images, percentile)

    def render(img: Tensor) -> Tensor:
        unit_img = ((img.float() - lo) / (hi - lo)).clamp(0.0, 1.0)
        return _stretch_unit(unit_img, stretch)

    rgb = torch.stack([render(channels[c]) for c in ("r", "g", "b")], dim=-1)
    rgb_np = rgb.detach().cpu().numpy()

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(7.6, 7.0), dpi=dpi)
        _theme.apply_figure_theme(fig, theme)
        _theme.title_block(fig, title, subtitle, theme)
        folio = _theme.theme_tokens(theme)
        _theme.apply_axes_theme(ax, folio, grid=False)
        fig.subplots_adjust(top=0.80 if (title or subtitle) else 0.94)

    tokens = _theme.theme_tokens(theme)
    imshow_kwargs = {"origin": "lower", "interpolation": "nearest"}
    imshow_kwargs.update(kwargs)
    ax.imshow(rgb_np, **imshow_kwargs)

    ax.set_xlabel(
        "pixel x",
        color=tokens.ink_muted,
        fontsize=_theme.TYPE.label,
        fontfamily=_theme.font_family(),
    )
    ax.set_ylabel(
        "pixel y",
        color=tokens.ink_muted,
        fontsize=_theme.TYPE.label,
        fontfamily=_theme.font_family(),
    )
    if not created and title:
        ax.set_title(title, color=tokens.ink, fontsize=_theme.TYPE.subtitle)

    if show_band_key and groups:
        text = "\n".join(_band_labels(groups, wave, unit))
        ax.text(
            0.02,
            0.98,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=tokens.ink,
            fontsize=_theme.TYPE.footnote + 0.5,
            fontfamily=_theme.mono_family(),
            linespacing=1.45,
            bbox={
                "facecolor": tokens.figure_bg,
                "edgecolor": "none",
                "alpha": 0.55,
                "pad": 3.0,
            },
        )

    if save is not None:
        _theme.save_figure(ax.figure, save, dpi=dpi, theme=theme)
    return ax


def plot_pixel_spectrum(
    cube: torch.Tensor,
    spectral_axis: SpectralAxis,
    y: int,
    x: int,
    *,
    spectral_axis_index: int = 0,
    ax: Any = None,
    theme: str = "dark",
    title: str | None = None,
    subtitle: str | None = None,
    label: str | None = None,
    legend: bool = True,
    grid: bool = True,
    save: str | Path | None = None,
    dpi: int = 200,
    **kwargs: Any,
) -> Any:
    """Plot the 1D spectrum at one pixel of a data cube.

    Args:
        cube: 3D data cube tensor.
        spectral_axis: SpectralAxis for the spectral dimension.
        y: Row coordinate of the spectrum in pixel space.
        x: Column coordinate of the spectrum in pixel space.
        spectral_axis_index: Dimension index of the spectral axis (0, 1 or 2).
        ax: Optional matplotlib axes; a themed figure is created when omitted.
        theme: ``dark`` (default) or ``paper``.
        title: Figure headline.
        subtitle: Quieter second line under the headline.
        label: Legend label for the spectrum; defaults to the pixel position.
        legend: Toggle the legend.
        grid: Toggle the graticule.
        save: Path to write the figure.
        dpi: Output resolution used when creating/saving the figure.
        kwargs: Forwarded to ``plot``.

    Returns:
        The matplotlib axes.
    """
    if cube.ndim != 3:
        raise ValueError(f"cube must be 3D, got shape {tuple(cube.shape)}")
    if spectral_axis_index not in (0, 1, 2):
        raise ValueError("spectral_axis_index must be 0, 1, or 2")

    n_spectral = cube.shape[spectral_axis_index]
    if len(spectral_axis.values) != n_spectral:
        raise ValueError(
            f"spectral_axis values length {len(spectral_axis.values)} "
            f"does not match cube dimension length {n_spectral}"
        )

    plt = _theme.pyplot()
    if spectral_axis_index == 0:
        flux = cube[:, y, x]
    elif spectral_axis_index == 1:
        flux = cube[y, :, x]
    else:
        flux = cube[y, x, :]

    wave_np = spectral_axis.values.detach().cpu().flatten().numpy()
    flux_np = flux.detach().cpu().flatten().numpy()

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=dpi)
        _theme.apply_figure_theme(fig, theme)
        _theme.title_block(fig, title, subtitle, theme)
        tokens = _theme.apply_axes_theme(ax, theme, grid=grid)
        fig.subplots_adjust(top=0.80 if (title or subtitle) else 0.92, bottom=0.16)
    else:
        tokens = _theme.theme_tokens(theme)
        if title:
            ax.set_title(title, color=tokens.ink, fontsize=_theme.TYPE.subtitle)

    plot_kwargs = {"linewidth": 1.4, "color": tokens.accent}
    plot_kwargs.update(kwargs)
    ax.plot(
        wave_np,
        flux_np,
        label=label if label is not None else f"({y}, {x})",
        **plot_kwargs,
    )
    ax.set_xlabel(
        f"{spectral_axis.type} ({spectral_axis.unit})",
        color=tokens.ink_muted,
        fontsize=_theme.TYPE.label,
        fontfamily=_theme.font_family(),
    )
    ax.set_ylabel(
        "Flux",
        color=tokens.ink_muted,
        fontsize=_theme.TYPE.label,
        fontfamily=_theme.font_family(),
    )
    if legend:
        ax.legend(
            loc="upper right",
            frameon=False,
            fontsize=_theme.TYPE.legend,
            labelcolor=tokens.ink,
        )
    if save is not None:
        _theme.save_figure(ax.figure, save, dpi=dpi, theme=theme)
    return ax


__all__ = ["plot_pixel_spectrum", "plot_pseudo_rgb"]
