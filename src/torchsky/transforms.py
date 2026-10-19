
import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

class AsinhStretch:

    def __init__(self, a: float = 0.1, Q: float = 8.0):
        if a <= 0:
            raise ValueError("a must be positive")
        if Q <= 0:
            raise ValueError("Q must be positive")
        self.a = a
        self.Q = Q

    def __call__(self, tensor: Tensor) -> Tensor:
        scale = self.Q / self.a
        return torch.asinh(scale * tensor) / math.asinh(scale)

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
    flat = tensor.flatten()
    valid = flat[torch.isfinite(flat)]
    if valid.numel() == 0:
        return None

    if valid.numel() > n_samples:
        step = max(1, valid.numel() // n_samples)
        sample = valid[::step]
    else:
        sample = valid

    sample, _ = torch.sort(sample)
    n = sample.numel()
    if n < min_npixels:
        return sample[0], sample[-1]

    work_dtype = torch.float32 if sample.device.type == "mps" else torch.float64
    good = torch.ones(n, dtype=torch.bool, device=sample.device)
    x = torch.arange(n, dtype=work_dtype, device=sample.device)
    y = sample.to(work_dtype)
    for _ in range(max_iterations):
        xg = x[good]
        yg = y[good]
        if xg.numel() < min_npixels:
            break
        xm = xg.mean()
        ym = yg.mean()
        slope = ((xg - xm) * (yg - ym)).sum() / (
            (xg - xm) ** 2
        ).sum().clamp_min(1e-30)
        intercept = ym - slope * xm
        residuals = y - (slope * x + intercept)
        sigma = residuals[good].std()
        if float(sigma.item()) < 1e-10:
            break
        new_good = torch.abs(residuals) < krej * sigma
        if int(new_good.sum().item()) < min_npixels:
            break
        if float(new_good.sum().item()) / n < 1.0 - max_reject:
            break
        good = new_good

    xg = x[good]
    yg = y[good]
    if xg.numel() < 2:
        return sample[n // 4], sample[3 * n // 4]

    xm = xg.mean()
    ym = yg.mean()
    slope = ((xg - xm) * (yg - ym)).sum() / (
        (xg - xm) ** 2
    ).sum().clamp_min(1e-30)
    intercept = ym - slope * xm
    center = n / 2.0
    z1 = torch.maximum(
        sample[0],
        (intercept + slope * (0.0 - center) / contrast).to(sample.dtype),
    )
    z2 = torch.minimum(
        sample[-1],
        (intercept + slope * (n - center) / contrast).to(sample.dtype),
    )
    if bool(z2 <= z1):
        median = torch.median(sample)
        std = sample.float().std().to(dtype=sample.dtype)
        z1, z2 = median - 3.0 * std, median + 3.0 * std
    return z1, z2


class ZScale:

    def __init__(self, contrast: float = 0.25, max_reject: float = 0.5):
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

        # Prevent division by zero
        range_val = z2 - z1
        if torch.abs(range_val) < 1e-8:
            return torch.zeros_like(tensor)

        # Normalize to [0, 1]
        normalized = (tensor - z1) / range_val
        return torch.clamp(normalized, 0, 1)

class LogStretch:

    def __init__(self, a: float = 1000.0):
        if a <= 0:
            raise ValueError("a must be positive")
        self.a = a
        self._log_factor = math.log(self.a + 1.0)

    def __call__(self, tensor: Tensor) -> Tensor:
        return torch.log(self.a * tensor + 1) / self._log_factor

class PowerStretch:

    def __init__(self, gamma: float = 0.5):
        self.gamma = gamma

    def __call__(self, tensor: Tensor) -> Tensor:
        return torch.pow(tensor, self.gamma)

class Normalize:

    def __init__(self, mean: Optional[float] = None, std: Optional[float] = None):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: Tensor) -> Tensor:
        mean = (
            torch.as_tensor(self.mean, dtype=tensor.dtype, device=tensor.device)
            if self.mean is not None
            else tensor.mean()
        )
        std = (
            torch.as_tensor(self.std, dtype=tensor.dtype, device=tensor.device)
            if self.std is not None
            else tensor.std()
        )

        # Prevent division by zero
        if torch.abs(std) < 1e-8:
            return tensor - mean

        return (tensor - mean) / std

class MinMaxScale:

    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, tensor: Tensor) -> Tensor:
        t_min = tensor.min()
        t_max = tensor.max()

        # Prevent division by zero
        if torch.abs(t_max - t_min) < 1e-8:
            return torch.zeros_like(tensor)

        normalized = (tensor - t_min) / (t_max - t_min)
        return normalized * (self.max_val - self.min_val) + self.min_val

class RandomCrop:

    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, tensor: Tensor) -> Tensor:
        h, w = tensor.shape[-2:]
        th, tw = self.size

        if h < th or w < tw:
            raise ValueError(f"Input size {(h, w)} smaller than crop size {self.size}")

        # Random crop position - use scalar tensor generation for efficiency
        i = torch.randint(0, h - th + 1, ()).item()
        j = torch.randint(0, w - tw + 1, ()).item()

        return tensor[..., i : i + th, j : j + tw]

class CenterCrop:

    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, tensor: Tensor) -> Tensor:
        h, w = tensor.shape[-2:]
        th, tw = self.size

        if h < th or w < tw:
            raise ValueError(f"Input size {(h, w)} smaller than crop size {self.size}")

        # Center crop position
        i = (h - th) // 2
        j = (w - tw) // 2

        return tensor[..., i : i + th, j : j + tw]

class RandomFlip:

    def __init__(self, horizontal: bool = True, vertical: bool = True, p: float = 0.5):
        self.horizontal = horizontal
        self.vertical = vertical
        self.p = p

    def __call__(self, tensor: Tensor) -> Tensor:
        if self.horizontal and torch.rand(1) < self.p:
            tensor = torch.flip(tensor, [-1])
        if self.vertical and torch.rand(1) < self.p:
            tensor = torch.flip(tensor, [-2])
        return tensor

class GaussianNoise:

    def __init__(self, std: float = 0.01, snr_based: bool = False):
        self.std = std
        self.snr_based = snr_based

    def __call__(self, tensor: Tensor) -> Tensor:
        if self.snr_based:
            signal_std = tensor.std()
            noise_std = signal_std / self.std
        else:
            noise_std = self.std

        noise = torch.randn_like(tensor) * noise_std
        return tensor + noise

class ToDevice:

    def __init__(self, device: Union[str, torch.device]):
        self.device = device

    def __call__(self, tensor: Tensor) -> Tensor:
        return tensor.to(self.device)

class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, tensor: Tensor) -> Tensor:
        for transform in self.transforms:
            tensor = transform(tensor)
        return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n    {0}".format(t)
        format_string += "\n)"
        return format_string

# Convenience functions
def create_display_transform(stretch: str = "asinh", **kwargs):
    transforms = []

    # Add stretch transformation
    if stretch == "asinh":
        transforms.append(AsinhStretch(**kwargs))
    elif stretch == "log":
        transforms.append(LogStretch(**kwargs))
    elif stretch == "power":
        transforms.append(PowerStretch(**kwargs))
    elif stretch == "zscale":
        transforms.append(ZScale(**kwargs))
    else:
        raise ValueError(f"Unknown stretch type: {stretch}")

    return Compose(transforms)

def create_training_transform(
    crop_size: int = 224, normalize: bool = True, augment: bool = True
):
    transforms = [RandomCrop(crop_size)]

    if augment:
        transforms.extend([RandomFlip(p=0.5), GaussianNoise(std=0.01)])

    if normalize:
        transforms.append(ZScale())

    return Compose(transforms)

def create_validation_transform(crop_size: int = 224, normalize: bool = True):
    transforms = [CenterCrop(crop_size)]

    if normalize:
        transforms.append(ZScale())

    return Compose(transforms)

def create_inference_transform(normalize: bool = True):
    transforms = []
    if normalize:
        transforms.append(ZScale())
    return Compose(transforms)

class PoissonNoise:

    def __init__(self, lam: float = 1.0):
        self.lam = lam

    def __call__(self, tensor: Tensor) -> Tensor:
        # Convert to counts (rate parameter)
        counts = torch.relu(tensor) * self.lam
        # Sample Poisson
        noisy_counts = torch.poisson(counts)
        # Convert back
        return noisy_counts / self.lam

class RandomRotation:

    def __init__(self, interpolation: str = "bilinear"):
        self.interpolation = interpolation

    def __call__(self, tensor: Tensor) -> Tensor:
        angle = torch.rand(1).item() * 360.0
        theta = math.radians(angle)

        # Create rotation matrix
        c, s = math.cos(theta), math.sin(theta)
        # PyTorch affine grid expects 2x3 matrix for 2D
        # [[cos, -sin, 0], [sin, cos, 0]]
        rot_mat = torch.tensor(
            [[c, -s, 0], [s, c, 0]], device=tensor.device, dtype=tensor.dtype
        )
        rot_mat = rot_mat.unsqueeze(0)  # Batch dim

        # Grid sample expects (N, C, H, W)
        if tensor.dim() == 3:
            x = tensor.unsqueeze(0)
        elif tensor.dim() == 2:
            x = tensor.unsqueeze(0).unsqueeze(0)
        else:
            x = tensor

        grid = torch.nn.functional.affine_grid(
            rot_mat.expand(x.size(0), -1, -1), x.size(), align_corners=False
        )
        x_rot = torch.nn.functional.grid_sample(
            x, grid, mode=self.interpolation, align_corners=False
        )

        if tensor.dim() == 3:
            return x_rot.squeeze(0)
        elif tensor.dim() == 2:
            return x_rot.squeeze(0).squeeze(0)
        return x_rot

class PerturbByError:

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def __call__(self, inputs: Tuple[Tensor, Tensor]) -> Tensor:
        data, error = inputs
        noise = torch.randn_like(data) * error * self.scale
        return data + noise

class RobustScale:

    def __init__(self, center: bool = True, scale: bool = True):
        self.center = center
        self.scale = scale

    def __call__(self, tensor: Tensor) -> Tensor:
        if self.center:
            median = torch.median(tensor)
            tensor = tensor - median

        if self.scale:
            qs = torch.quantile(
                tensor,
                torch.tensor([0.25, 0.75], dtype=tensor.dtype, device=tensor.device),
            )
            q25, q75 = qs[0], qs[1]
            iqr = q75 - q25
            if iqr > 1e-8:
                tensor = tensor / iqr

        return tensor

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
