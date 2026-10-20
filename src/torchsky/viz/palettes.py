"""Colour palettes shared by the PNG tile server and matplotlib figures.

The PNG/HiPS path needs a fixed 256-entry lookup table, so the six bundled
palettes are defined once here as RGB keypoints and sampled into an LUT.
Matplotlib figures resolve the requested name against matplotlib's own
colormap registry first and only fall back to the keypoints, so figures get the
canonical version of ``viridis``/``magma``/... while tiles stay byte-stable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

CMAP_KEYPOINTS: dict[str, tuple[tuple[float, tuple[int, int, int]], ...]] = {
    "gray": (
        (0.0, (0, 0, 0)),
        (1.0, (255, 255, 255)),
    ),
    "viridis": (
        (0.000, (68, 1, 84)),
        (0.125, (72, 40, 120)),
        (0.250, (62, 83, 140)),
        (0.375, (49, 120, 150)),
        (0.500, (38, 153, 137)),
        (0.625, (53, 183, 121)),
        (0.750, (110, 206, 88)),
        (0.875, (181, 222, 43)),
        (1.000, (253, 231, 37)),
    ),
    "hot": (
        (0.000, (0, 0, 0)),
        (0.330, (255, 0, 0)),
        (0.660, (255, 255, 0)),
        (1.000, (255, 255, 255)),
    ),
    "thermal": (
        (0.000, (0, 0, 128)),
        (0.250, (0, 128, 255)),
        (0.500, (0, 255, 0)),
        (0.750, (255, 165, 0)),
        (1.000, (255, 0, 0)),
    ),
    "rdbu_r": (
        (0.000, (103, 0, 31)),
        (0.200, (214, 96, 77)),
        (0.350, (244, 165, 130)),
        (0.500, (247, 247, 247)),
        (0.650, (146, 197, 222)),
        (0.800, (67, 147, 195)),
        (1.000, (5, 48, 97)),
    ),
    "seismic": (
        (0.000, (0, 0, 76)),
        (0.250, (0, 102, 255)),
        (0.450, (138, 190, 255)),
        (0.500, (255, 255, 255)),
        (0.550, (255, 138, 138)),
        (0.750, (255, 26, 26)),
        (1.000, (76, 0, 0)),
    ),
}

COLORMAP_NAMES = list(CMAP_KEYPOINTS)


def normalize_name(name: str) -> str:
    """Normalize a colormap name the way the tile server exposes it."""
    return str(name).strip().lower().replace("-", "_")


def _make_lut(
    *keypoints: tuple[float, tuple[int, int, int]],
) -> np.ndarray:
    """Linearly interpolate between (position, RGB) keypoints → 256×3 uint8."""
    lut = np.zeros((256, 3), dtype=np.float32)
    positions = [p for p, _ in keypoints]
    colors = [np.array(c, dtype=np.float32) for _, c in keypoints]
    for i in range(256):
        t = i / 255.0
        idx = 0
        for j in range(len(positions) - 1):
            if t >= positions[j]:
                idx = j
        if idx >= len(positions) - 1:
            lut[i] = colors[-1]
        else:
            t0, t1 = positions[idx], positions[idx + 1]
            alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            lut[i] = (1 - alpha) * colors[idx] + alpha * colors[idx + 1]
    return np.clip(lut, 0, 255).astype(np.uint8)


_LUTS: dict[str, np.ndarray] = {
    name: _make_lut(*keypoints) for name, keypoints in CMAP_KEYPOINTS.items()
}


def get_lut(name: str) -> np.ndarray:
    """Return the 256×3 uint8 colormap LUT by name."""
    key = normalize_name(name)
    if key not in _LUTS:
        raise ValueError(
            f"Unknown colormap {name!r}. Choose from {list(CMAP_KEYPOINTS)}"
        )
    return _LUTS[key]


def matplotlib_cmap(name: str) -> Any:
    """Return a matplotlib colormap for ``name``.

    Matplotlib's registry wins so figures use the canonical map; names that only
    exist here (``thermal``) are rebuilt from the shared keypoints. Requires
    matplotlib.
    """
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Matplotlib is required for plotting visualizers. "
            "Install it via: pip install matplotlib"
        ) from exc

    key = normalize_name(name)
    try:
        return matplotlib.colormaps[key]
    except KeyError:
        pass
    if key in CMAP_KEYPOINTS:
        from matplotlib.colors import LinearSegmentedColormap

        stops = [
            (pos, tuple(channel / 255.0 for channel in rgb))
            for pos, rgb in CMAP_KEYPOINTS[key]
        ]
        return LinearSegmentedColormap.from_list(f"torchsky_{key}", stops, N=256)
    raise ValueError(
        f"Unknown colormap {name!r}. Choose a matplotlib colormap name or one of "
        f"{list(CMAP_KEYPOINTS)}"
    )


__all__ = [
    "CMAP_KEYPOINTS",
    "COLORMAP_NAMES",
    "get_lut",
    "matplotlib_cmap",
    "normalize_name",
]
