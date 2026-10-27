"""Display stretches, normalisation, cropping, and augmentation transforms.

Every transform is a callable object that maps one tensor to one tensor and
keeps the last two axes spatial for the geometric transforms. Stretches are
monotone display mappings on non-negative data; normalisations are affine and
dtype-preserving.
"""

from __future__ import annotations

import math
from typing import Any, Sequence, Union

import torch
from torch import Tensor

from ._dtype import work_dtype

_Size = Union[int, tuple[int, int]]
_EPS = 1e-30


class AsinhStretch:
    """Inverse-hyperbolic-sine stretch ``asinh(Q x / a) / asinh(Q / a)``."""

    def __init__(self, a: float = 0.1, Q: float = 8.0) -> None:
        if a <= 0:
            raise ValueError("a must be positive")
        if Q <= 0:
            raise ValueError("Q must be positive")
        self.a = a
        self.Q = Q

    def __call__(self, tensor: Tensor) -> Tensor:
        scale = self.Q / self.a
        return torch.asinh(scale * tensor) / math.asinh(scale)


class LogStretch:
    """Logarithmic stretch ``log(a x + 1) / log(a + 1)`` for ``x >= 0``."""

    def __init__(self, a: float = 1000.0) -> None:
        if a <= 0:
            raise ValueError("a must be positive")
        self.a = a
        self._log_factor = math.log(self.a + 1.0)

    def __call__(self, tensor: Tensor) -> Tensor:
        return torch.log(self.a * tensor + 1) / self._log_factor


class PowerStretch:
    """Power-law stretch ``x ** gamma`` for ``x >= 0``."""

    def __init__(self, gamma: float = 0.5) -> None:
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and non-negative")
        self.gamma = gamma

    def __call__(self, tensor: Tensor) -> Tensor:
        return torch.pow(tensor, self.gamma)


def _fit_sorted_line(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """Least-squares ``(slope, intercept)`` of ``y`` against ``x``."""
    x_mean = x.mean()
    y_mean = y.mean()
    centred = x - x_mean
    slope = (centred * (y - y_mean)).sum() / centred.square().sum().clamp_min(_EPS)
    return slope, y_mean - slope * x_mean


def _zscale_bounds_tensor(
    tensor: Tensor,
    *,
    contrast: float = 0.25,
    n_samples: int = 600,
    max_reject: float = 0.5,
    min_npixels: int = 5,
    krej: float = 2.5,
    max_iterations: int = 5,
) -> tuple[Tensor, Tensor] | None:
    """Sample ``(z1, z2)`` display limits from a robust linear fit of the CDF.

    The fit ranks the finite samples, so a heavily skewed or partially masked
    pixel distribution still yields a stable range.
    """
    flat = tensor.flatten()
    valid = flat[torch.isfinite(flat)]
    n_valid = int(valid.numel())
    if n_valid == 0:
        return None

    if n_valid > n_samples:
        step = -(-n_valid // n_samples)
        sample = valid[::step]
    else:
        sample = valid

    sample = torch.sort(sample).values
    n = int(sample.numel())
    if n < min_npixels:
        return sample[0], sample[-1]

    dtype = work_dtype(tensor.device)
    x = torch.arange(n, dtype=dtype, device=sample.device)
    y = sample.to(dtype, copy=True)

    good = torch.ones(n, dtype=torch.bool, device=sample.device)
    for _ in range(max_iterations):
        if int(good.sum()) < min_npixels:
            break
        slope, intercept = _fit_sorted_line(x[good], y[good])
        residuals = y - (slope * x + intercept)
        sigma = residuals[good].std()
        if not bool(sigma > _EPS):
            break
        new_good = residuals.abs() < krej * sigma
        if int(new_good.sum()) < min_npixels:
            break
        if float(new_good.sum()) / n < 1.0 - max_reject:
            break
        good = new_good

    if int(good.sum()) < 2:
        return sample[n // 4], sample[3 * n // 4]

    slope, intercept = _fit_sorted_line(x[good], y[good])
    centre = n / 2.0
    z1 = torch.maximum(
        sample[0],
        (intercept + slope * (0.0 - centre) / contrast).to(sample.dtype),
    )
    z2 = torch.minimum(
        sample[-1],
        (intercept + slope * (n - centre) / contrast).to(sample.dtype),
    )
    if bool(z2 > z1):
        return z1, z2
    median = torch.median(sample)
    spread = sample.std().to(sample.dtype)
    return median - 3.0 * spread, median + 3.0 * spread


class ZScale:
    """Robust display limits mapped to ``[0, 1]`` (astropy-style zscale)."""

    def __init__(self, contrast: float = 0.25, max_reject: float = 0.5) -> None:
        if contrast <= 0:
            raise ValueError("contrast must be positive")
        if max_reject < 0 or max_reject >= 1:
            raise ValueError("max_reject must be in [0, 1)")
        self.contrast = contrast
        self.max_reject = max_reject

    def __call__(self, tensor: Tensor) -> Tensor:
        bounds = _zscale_bounds_tensor(
            tensor, contrast=self.contrast, max_reject=self.max_reject
        )
        if bounds is None:
            return torch.zeros_like(tensor)
        z1, z2 = bounds
        span = z2 - z1
        if not bool(span > 0):
            return torch.zeros_like(tensor)
        return ((tensor - z1) / span).clamp(0, 1)


class Normalize:
    """Subtract the mean and divide by the standard deviation."""

    def __init__(self, mean: float | None = None, std: float | None = None) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, tensor: Tensor) -> Tensor:
        mean = (
            torch.as_tensor(self.mean, dtype=tensor.dtype, device=tensor.device)
            if self.mean is not None
            else tensor.mean()
        )
        centred = tensor - mean
        std = (
            torch.as_tensor(self.std, dtype=tensor.dtype, device=tensor.device)
            if self.std is not None
            else tensor.std()
        )
        if not bool(std > 0):
            return centred
        return centred / std


class MinMaxScale:
    """Affinely map the data range onto ``[min_val, max_val]``."""

    def __init__(self, min_val: float = 0.0, max_val: float = 1.0) -> None:
        if not math.isfinite(min_val) or not math.isfinite(max_val):
            raise ValueError("min_val and max_val must be finite")
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, tensor: Tensor) -> Tensor:
        t_min = tensor.min()
        t_max = tensor.max()
        if not bool(t_max > t_min):
            return torch.full_like(tensor, self.min_val)
        scaled = (tensor - t_min) / (t_max - t_min)
        return scaled * (self.max_val - self.min_val) + self.min_val


class RandomCrop:
    """Crop a random ``size`` window from the trailing two axes."""

    def __init__(self, size: _Size) -> None:
        self.size = (size, size) if isinstance(size, int) else tuple(size)

    def __call__(self, tensor: Tensor) -> Tensor:
        h, w = tensor.shape[-2:]
        th, tw = self.size
        if h < th or w < tw:
            raise ValueError(f"Input size {(h, w)} smaller than crop size {self.size}")
        i = int(torch.randint(0, h - th + 1, ()))
        j = int(torch.randint(0, w - tw + 1, ()))
        return tensor[..., i : i + th, j : j + tw]


class CenterCrop:
    """Crop a centred ``size`` window from the trailing two axes."""

    def __init__(self, size: _Size) -> None:
        self.size = (size, size) if isinstance(size, int) else tuple(size)

    def __call__(self, tensor: Tensor) -> Tensor:
        h, w = tensor.shape[-2:]
        th, tw = self.size
        if h < th or w < tw:
            raise ValueError(f"Input size {(h, w)} smaller than crop size {self.size}")
        i = (h - th) // 2
        j = (w - tw) // 2
        return tensor[..., i : i + th, j : j + tw]


class RandomFlip:
    """Flip the trailing two axes independently with probability ``p``."""

    def __init__(
        self, horizontal: bool = True, vertical: bool = True, p: float = 0.5
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        self.horizontal = horizontal
        self.vertical = vertical
        self.p = p

    def __call__(self, tensor: Tensor) -> Tensor:
        if self.horizontal and float(torch.rand(())) < self.p:
            tensor = tensor.flip(-1)
        if self.vertical and float(torch.rand(())) < self.p:
            tensor = tensor.flip(-2)
        return tensor


class GaussianNoise:
    """Additive Gaussian noise, either absolute or scaled to the signal.

    ``snr_based=True`` sets the noise standard deviation to
    ``signal_std / std``, so ``std`` is then a signal-to-noise ratio.
    """

    def __init__(self, std: float = 0.01, snr_based: bool = False) -> None:
        if not math.isfinite(std) or std < 0 or (snr_based and std == 0):
            raise ValueError("std must be finite and non-negative")
        self.std = std
        self.snr_based = snr_based

    def __call__(self, tensor: Tensor) -> Tensor:
        noise_std = tensor.std() / self.std if self.snr_based else self.std
        return tensor + torch.randn_like(tensor) * noise_std


class PoissonNoise:
    """Poisson sample of a non-negative signal, rescaled to the input scale."""

    def __init__(self, lam: float = 1.0) -> None:
        if not math.isfinite(lam) or lam <= 0:
            raise ValueError("lam must be finite and positive")
        self.lam = lam

    def __call__(self, tensor: Tensor) -> Tensor:
        counts = tensor.relu() * self.lam
        return torch.poisson(counts) / self.lam


class RandomRotation:
    """Rotate the trailing two axes by a uniformly random angle."""

    def __init__(self, interpolation: str = "bilinear") -> None:
        self.interpolation = interpolation

    def __call__(self, tensor: Tensor) -> Tensor:
        theta = math.radians(float(torch.rand(())) * 360.0)
        c, s = math.cos(theta), math.sin(theta)
        dtype = (
            tensor.dtype if tensor.is_floating_point() else torch.get_default_dtype()
        )
        rotation = torch.tensor(
            [[c, -s, 0.0], [s, c, 0.0]], dtype=dtype, device=tensor.device
        ).unsqueeze(0)
        batched = tensor if tensor.dtype == dtype else tensor.to(dtype)
        if batched.dim() == 3:
            batched = batched.unsqueeze(0)
        elif batched.dim() == 2:
            batched = batched.unsqueeze(0).unsqueeze(0)
        grid = torch.nn.functional.affine_grid(
            rotation.expand(batched.size(0), -1, -1),
            batched.size(),
            align_corners=False,
        )
        rotated = torch.nn.functional.grid_sample(
            batched, grid, mode=self.interpolation, align_corners=False
        )
        if tensor.dim() == 3:
            return rotated.squeeze(0)
        if tensor.dim() == 2:
            return rotated.squeeze(0).squeeze(0)
        return rotated


class ToDevice:
    """Move a tensor to ``device``."""

    def __init__(self, device: str | torch.device) -> None:
        self.device = device

    def __call__(self, tensor: Tensor) -> Tensor:
        return tensor.to(self.device)


class Compose:
    """Apply transforms left to right.

    Args:
        transforms: Ordered callables, each accepting the previous tensor.

    Examples:
        >>> import torch
        >>> from torchsky.transforms import Compose
        >>> Compose([lambda value: value + 1])(torch.tensor(1)).item()
        2
    """

    def __init__(self, transforms: Sequence[Any]) -> None:
        self.transforms = list(transforms)

    def __call__(self, tensor: Tensor) -> Tensor:
        for transform in self.transforms:
            tensor = transform(tensor)
        return tensor

    def __repr__(self) -> str:
        body = "".join(f"\n    {transform}" for transform in self.transforms)
        return f"{self.__class__.__name__}({body}\n)"


class PerturbByError:
    """Add ``scale``-weighted Gaussian noise drawn from a companion error map."""

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def __call__(self, inputs: tuple[Tensor, Tensor]) -> Tensor:
        data, error = inputs
        return data + torch.randn_like(data) * error * self.scale


class RobustScale:
    """Centre on the median and optionally divide by the interquartile range."""

    def __init__(self, center: bool = True, scale: bool = True) -> None:
        self.center = center
        self.scale = scale

    def __call__(self, tensor: Tensor) -> Tensor:
        if self.center:
            tensor = tensor - torch.median(tensor)
        if self.scale:
            dtype = (
                tensor.dtype if tensor.is_floating_point() else torch.get_default_dtype()
            )
            levels = torch.tensor([0.25, 0.75], dtype=dtype, device=tensor.device)
            q25, q75 = torch.quantile(tensor, levels).unbind()
            iqr = q75 - q25
            if bool(iqr > 0):
                tensor = tensor / iqr
        return tensor


def create_display_transform(stretch: str = "asinh", **kwargs: Any) -> Compose:
    """Build a one-step display ``Compose`` for a named stretch."""
    builders = {
        "asinh": AsinhStretch,
        "log": LogStretch,
        "power": PowerStretch,
        "zscale": ZScale,
    }
    try:
        return Compose([builders[stretch](**kwargs)])
    except KeyError:
        raise ValueError(f"Unknown stretch type: {stretch}") from None


def create_training_transform(
    crop_size: int = 224, normalize: bool = True, augment: bool = True
) -> Compose:
    """Random crop, optional flips/noise, optional zscale."""
    transforms: list[Any] = [RandomCrop(crop_size)]
    if augment:
        transforms.extend([RandomFlip(p=0.5), GaussianNoise(std=0.01)])
    if normalize:
        transforms.append(ZScale())
    return Compose(transforms)


def create_validation_transform(crop_size: int = 224, normalize: bool = True) -> Compose:
    """Centre crop and optional zscale."""
    transforms: list[Any] = [CenterCrop(crop_size)]
    if normalize:
        transforms.append(ZScale())
    return Compose(transforms)


def create_inference_transform(normalize: bool = True) -> Compose:
    """Optional zscale only, keeping the input shape."""
    return Compose([ZScale()] if normalize else [])


__all__ = [
    "AsinhStretch",
    "CenterCrop",
    "Compose",
    "GaussianNoise",
    "LogStretch",
    "MinMaxScale",
    "Normalize",
    "PerturbByError",
    "PoissonNoise",
    "PowerStretch",
    "RandomCrop",
    "RandomFlip",
    "RandomRotation",
    "RobustScale",
    "ToDevice",
    "ZScale",
    "create_display_transform",
    "create_inference_transform",
    "create_training_transform",
    "create_validation_transform",
]
