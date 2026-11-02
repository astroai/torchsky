"""Typography and colour tokens for torchsky figures.

Everything here is lazy with respect to matplotlib so importing
``torchsky.viz`` stays free of an optional plotting dependency. The style is
applied to concrete artists (tick labels, labels, spines, graticule) rather than
through global ``rcParams``, so a figure keeps its look whenever it is saved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ThemeName = Literal["dark", "paper"]

_DISPLAY_STACK = (
    "Inter",
    "SF Pro Display",
    "Helvetica Neue",
    "Helvetica",
    "Arial",
    "DejaVu Sans",
)
_MONO_STACK = (
    "IBM Plex Mono",
    "SF Mono",
    "Menlo",
    "Consolas",
    "DejaVu Sans Mono",
)


@dataclass(frozen=True)
class Theme:
    """Colour tokens for one figure theme."""

    name: str
    figure_bg: str
    axes_bg: str
    ink: str
    ink_muted: str
    grid: str
    accent: str
    accent_alt: str
    spine: str
    grid_alpha: float = 0.5


@dataclass(frozen=True)
class Typography:
    """Point sizes for one figure."""

    title: float = 17.0
    subtitle: float = 11.0
    label: float = 10.5
    tick: float = 9.0
    legend: float = 9.5
    footnote: float = 8.0


TYPE = Typography()

_THEMES: dict[str, Theme] = {
    "dark": Theme(
        name="dark",
        figure_bg="#05060d",
        axes_bg="#0b0b16",
        ink="#f2f4fb",
        ink_muted="#97a0bb",
        grid="#3a4263",
        accent="#7fd4ff",
        accent_alt="#ffb347",
        spine="#2b3350",
        # A bright graticule on a near-black map competes with the data.
        grid_alpha=0.34,
    ),
    "paper": Theme(
        name="paper",
        figure_bg="#ffffff",
        axes_bg="#ffffff",
        ink="#111726",
        ink_muted="#5b6274",
        grid="#d7dbe6",
        accent="#1a5fb4",
        accent_alt="#c64600",
        spine="#c9cdd9",
        grid_alpha=0.7,
    ),
}


def theme_tokens(name: str = "dark") -> Theme:
    """Return the tokens for ``name`` (``"dark"`` or ``"paper"``)."""
    key = str(name).strip().lower()
    if key not in _THEMES:
        raise ValueError(f"Unknown theme {name!r}. Choose from {list(_THEMES)}")
    return _THEMES[key]


_FAMILY_CACHE: dict[str, str] = {}


def _installed_families() -> frozenset[str] | None:
    """Installed family names, or ``None`` when matplotlib cannot be imported."""
    try:
        from matplotlib import font_manager
    except ImportError:
        return None
    return frozenset(font.name for font in font_manager.fontManager.ttflist)


def _resolve(stack: tuple[str, ...], fallback: str, cache_key: str) -> str:
    cached = _FAMILY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    families = _installed_families()
    if families is None:
        # Do not cache the fallback: matplotlib may appear later in a session.
        return fallback
    chosen = next((name for name in stack if name in families), fallback)
    _FAMILY_CACHE[cache_key] = chosen
    return chosen


def font_family() -> str:
    """Best installed display family, falling back to matplotlib's default."""
    return _resolve(_DISPLAY_STACK, "DejaVu Sans", "display")


def mono_family() -> str:
    """Best installed monospace family, falling back to DejaVu Sans Mono."""
    return _resolve(_MONO_STACK, "DejaVu Sans Mono", "mono")


def pyplot() -> Any:
    """Import ``matplotlib.pyplot``, raising a dependency hint when missing."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Matplotlib is required for plotting visualizers. "
            "Install it via: pip install matplotlib"
        ) from exc
    return plt


def halo_effect(background: str, linewidth: float = 1.8) -> list[Any]:
    """A background-coloured stroke that keeps small text legible on maps."""
    from matplotlib import patheffects as pe

    return [pe.withStroke(linewidth=linewidth, foreground=background)]


def _style_tick_labels(ax: Any, theme: Theme, size: float) -> None:
    family = mono_family()
    halo = halo_effect(theme.axes_bg)
    for axis in (ax.xaxis, ax.yaxis):
        for label in axis.get_ticklabels():
            label.set_fontfamily(family)
            label.set_fontsize(size)
            label.set_color(theme.ink_muted)
            label.set_path_effects(halo)


def apply_axes_theme(ax: Any, theme: str = "dark", *, grid: bool = True) -> Theme:
    """Apply theme colours, graticule and typography to one axes."""
    tokens = theme_tokens(theme)
    ax.set_facecolor(tokens.axes_bg)
    ax.tick_params(colors=tokens.ink_muted, labelsize=TYPE.tick, length=3, width=0.8)
    for spine in ax.spines.values():
        spine.set_color(tokens.spine)
    if grid:
        ax.grid(
            True,
            color=tokens.grid,
            linewidth=0.5,
            alpha=tokens.grid_alpha,
            zorder=0,
        )
    _style_tick_labels(ax, tokens, TYPE.tick)
    return tokens


def apply_figure_theme(fig: Any, theme: str = "dark") -> Theme:
    """Set the figure background for one theme."""
    tokens = theme_tokens(theme)
    fig.set_facecolor(tokens.figure_bg)
    return tokens


def title_block(
    fig: Any,
    title: str | None,
    subtitle: str | None,
    theme: str = "dark",
    *,
    y: float = 0.975,
    gap: float = 0.045,
) -> None:
    """Draw a title with a quieter subtitle above the axes.

    ``gap`` is in figure fraction; pass ``points / figure height``-derived
    values so the two lines stay together on both tall and wide figures.
    """
    tokens = theme_tokens(theme)
    if title:
        fig.text(
            0.5,
            y,
            title,
            ha="center",
            va="top",
            color=tokens.ink,
            fontsize=TYPE.title,
            fontfamily=font_family(),
            fontweight="bold",
        )
    if subtitle:
        offset = gap if title else 0.0
        fig.text(
            0.5,
            y - offset,
            subtitle,
            ha="center",
            va="top",
            color=tokens.ink_muted,
            fontsize=TYPE.subtitle,
            fontfamily=font_family(),
        )


def style_colorbar(cbar: Any, theme: str = "dark", *, label: str | None = None) -> None:
    """Match a colorbar to the figure theme."""
    tokens = theme_tokens(theme)
    if cbar.outline is not None:
        cbar.outline.set_edgecolor(tokens.spine)
        cbar.outline.set_linewidth(0.7)
    cbar.ax.tick_params(colors=tokens.ink_muted, labelsize=TYPE.tick, length=2)
    _style_tick_labels(cbar.ax, tokens, TYPE.tick)
    if label:
        cbar.set_label(
            label,
            color=tokens.ink_muted,
            fontsize=TYPE.label,
            fontfamily=font_family(),
        )


def save_figure(
    fig: Any, path: str | Path, *, dpi: int = 200, theme: str = "dark"
) -> Path:
    """Save ``fig`` with a themed background; returns the written path."""
    tokens = theme_tokens(theme)
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        target,
        dpi=dpi,
        facecolor=tokens.figure_bg,
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.35,
    )
    return target


__all__ = [
    "TYPE",
    "Theme",
    "ThemeName",
    "Typography",
    "apply_axes_theme",
    "apply_figure_theme",
    "font_family",
    "mono_family",
    "pyplot",
    "save_figure",
    "style_colorbar",
    "theme_tokens",
    "title_block",
]
