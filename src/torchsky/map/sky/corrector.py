from collections.abc import Callable

import torch
from torch import Tensor

from .estimator import MeshSkyEstimator


def _offset_equations_from_levels(
    levels: Tensor,
) -> Tensor:
    n = len(levels)
    if n <= 1:
        return torch.zeros(n, device=levels.device)
    return levels - levels.mean()


class CrossExposureSkyCorrector:
    """Compute additive sky corrections across overlapping exposures.

    Given a list of images of the same field (same pixel grid), this
    estimates per-image additive sky offsets by comparing mean image
    levels and producing a spatially-varying sky model per image.

    Parameters
    ----------
    estimator
        A callable ``est(image) -> Tensor`` that produces a full-resolution
        sky model for a single image (e.g. ``MeshSkyEstimator`` or
        ``PolynomialSkyModel``).  If ``None``, defaults to a
        ``MeshSkyEstimator(block_size=128, statistic='median')``.
    """

    def __init__(
        self,
        estimator: Callable[..., Tensor] | None = None,
    ):
        self._estimator = estimator

    def _get_estimator(self) -> Callable[..., Tensor]:
        if self._estimator is not None:
            return self._estimator
        return MeshSkyEstimator(block_size=128, statistic="median")

    def _fit_offsets_and_skies(
        self,
        images: list[Tensor],
        masks: list[Tensor | None] | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        n = len(images)
        if n == 0:
            return torch.empty(0), []

        est = self._get_estimator()

        # 1. Compute per-image mean level from the whole image
        per_image_levels = torch.stack([img.mean() for img in images])

        # 2. Demean images so sky model captures only spatial variation
        skies = []
        for i, img in enumerate(images):
            m = masks[i] if masks is not None else None
            sky = est(img - per_image_levels[i]) if m is None else est(img - per_image_levels[i], m)
            skies.append(sky)

        # 3. Relative offsets (zero-sum constraint)
        offsets = _offset_equations_from_levels(per_image_levels)
        return offsets, skies

    def fit_offsets(
        self,
        images: list[Tensor],
        masks: list[Tensor | None] | None = None,
    ) -> Tensor:
        """Compute per-image additive sky corrections.

        Parameters
        ----------
        images
            List of ``[H, W]`` tensors, all on the same pixel grid.
        masks
            Optional per-image masks (``True`` = exclude from stat).

        Returns
        -------
        offsets
            ``[n]`` tensor of additive corrections per image.
        """
        offsets, _ = self._fit_offsets_and_skies(images, masks)
        return offsets

    def correct(
        self,
        images: list[Tensor],
        masks: list[Tensor | None] | None = None,
    ) -> list[Tensor]:
        """Estimate sky for each image and return sky-subtracted images.

        Parameters
        ----------
        images
            List of ``[H, W]`` tensors.
        masks
            Optional per-image masks.

        Returns
        -------
        corrected
            List of sky-subtracted images (``image - sky_model - offset``).
        """
        offsets, skies = self._fit_offsets_and_skies(images, masks)
        return [img - skies[i] - offsets[i] for i, img in enumerate(images)]
