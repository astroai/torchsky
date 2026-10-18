"""Continuous-position pixel-integrated responses and Gaussian stamp profiling.

This bounded lane consumes independent native detector stamps with known,
fixed, diagonal Gaussian variances. It does not register/resample observations,
infer a source catalogue, or require a global field. Response callables return
the already pixel-integrated response per unit incident stellar flux: applying
another pixel integral or normalizing a cropped stamp changes that convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


class PixelIntegratedResponse(Protocol):
    """Evaluate native unit-square pixel fractions at continuous offsets.

    ``offset_xy[..., 2]`` is pixel center minus source center, in detector pixel
    units and (x, y) order. Output has shape ``offset_xy.shape[:-1]``. The
    response includes pixel integration exactly once. Its full-domain integral
    and any throughput loss are the callable's declared calibration convention;
    a finite stamp's sum is never forced to one. Response parameters may belong
    to a callable ``nn.Module`` or retain an external tensor computation graph.
    """

    def __call__(self, offset_xy: Tensor) -> Tensor: ...


def _floating_tensor(value: Tensor, name: str) -> None:
    if not value.is_floating_point() or value.dtype not in (
        torch.float32,
        torch.float64,
    ):
        raise ValueError(f"{name} must have float32 or float64 dtype")


def _same_tensor_type(value: Tensor, reference: Tensor, name: str) -> None:
    if value.dtype != reference.dtype or value.device != reference.device:
        raise ValueError(f"{name} must match the data dtype and device")


@dataclass
class GaussianPixelResponse:
    """Exact axis-aligned Gaussian integrated over unit-square detector pixels.

    ``sigma_xy`` is a positive scalar or (sigma_x, sigma_y) tensor, in pixels.
    The underlying continuous density has unit integral on the infinite plane
    and zero centroid. The ePSF has infinite support; every finite footprint
    loses the analytically determined tail rather than redistributing it.
    This reference family assumes uniform intra-pixel sensitivity, no rotation,
    and a response independent of source flux/color/detector history.

    The supplied tensor retains its graph. For positive nonlinear fitting,
    construct a response with ``log_sigma.exp()`` within the optimization step.
    """

    sigma_xy: Tensor

    def __call__(self, offset_xy: Tensor) -> Tensor:
        _floating_tensor(offset_xy, "offset_xy")
        _floating_tensor(self.sigma_xy, "sigma_xy")
        _same_tensor_type(self.sigma_xy, offset_xy, "sigma_xy")
        if offset_xy.ndim < 1 or offset_xy.shape[-1] != 2:
            raise ValueError("offset_xy must have trailing shape (2,)")
        if self.sigma_xy.ndim != 0 and self.sigma_xy.shape != (2,):
            raise ValueError("sigma_xy must be scalar or have shape (2,)")
        if not bool(torch.isfinite(self.sigma_xy).all()) or bool(
            (self.sigma_xy <= 0).any()
        ):
            raise ValueError("sigma_xy must be finite and positive")
        if not bool(torch.isfinite(offset_xy).all()):
            raise ValueError("offset_xy must be finite")
        scale = math.sqrt(2.0) * self.sigma_xy
        # erfc differences avoid catastrophic subtraction of two CDFs near one.
        # Reflect positive intervals to the negative half-plane; erfc handles
        # both intervals crossing zero and tails without changing the integral.
        reflected = torch.where(offset_xy > 0, -offset_xy, offset_xy)
        lower = (reflected - 0.5) / scale
        upper = (reflected + 0.5) / scale
        fractions = 0.5 * (torch.erfc(-upper) - torch.erfc(-lower))
        return fractions.prod(dim=-1)


@dataclass
class NativePixelStamp:
    """One unresampled detector footprint with fixed independent variances.

    ``data`` and ``variance`` are [H,W]; ``pixel_xy`` is [H,W,2], containing
    actual detector pixel centers, not registered offsets. ``mask=True`` keeps
    a measurement. Masked data/variance may be nonfinite; kept data must be
    finite and kept variances strictly positive. Pixel coordinates remain
    finite even at masked measurements. No transformations are inferred.
    """

    data: Tensor
    variance: Tensor
    pixel_xy: Tensor
    mask: Tensor | None = None

    def __post_init__(self) -> None:
        _floating_tensor(self.data, "data")
        if self.data.ndim != 2 or self.data.numel() == 0:
            raise ValueError("data must be a nonempty 2D native stamp")
        _same_tensor_type(self.variance, self.data, "variance")
        _same_tensor_type(self.pixel_xy, self.data, "pixel_xy")
        if self.variance.shape != self.data.shape:
            raise ValueError("variance must have the data shape")
        if self.pixel_xy.shape != (*self.data.shape, 2):
            raise ValueError("pixel_xy must have shape [H,W,2]")
        if not bool(torch.isfinite(self.pixel_xy).all()):
            raise ValueError("pixel_xy must be finite")
        if self.mask is not None:
            if (
                self.mask.dtype != torch.bool
                or self.mask.device != self.data.device
                or self.mask.shape != self.data.shape
            ):
                raise ValueError("mask must be bool with the data shape and device")
        valid = self.valid_mask
        if not bool(valid.any()):
            raise ValueError("stamp must contain at least one kept pixel")
        if not bool(torch.isfinite(self.data[valid]).all()):
            raise ValueError("kept data must be finite")
        if not bool(torch.isfinite(self.variance[valid]).all()) or bool(
            (self.variance[valid] <= 0).any()
        ):
            raise ValueError("kept variance must be finite and positive")

    @property
    def valid_mask(self) -> Tensor:
        return (
            torch.ones_like(self.data, dtype=torch.bool)
            if self.mask is None
            else self.mask
        )


def native_pixel_stamp(
    image: Tensor,
    variance: Tensor,
    *,
    x_slice: slice,
    y_slice: slice,
    mask: Tensor | None = None,
) -> NativePixelStamp:
    """Slice native exposure pixels without interpolation or normalization.

    Exposure pixel centers use zero-based integer (x,y) coordinates. Slices
    must have unit step and explicit in-bounds endpoints; detector edges are
    represented by choosing a smaller footprint, never padded measurements.
    Data/variance/mask are tensor views of the supplied exposure.
    """
    if image.ndim != 2 or variance.shape != image.shape:
        raise ValueError("image and variance must be matching 2D exposures")
    if mask is not None and mask.shape != image.shape:
        raise ValueError("exposure mask must have the image shape")
    for selection, length in zip((y_slice, x_slice), image.shape, strict=True):
        if (
            selection.start is None
            or selection.stop is None
            or not isinstance(selection.start, int)
            or not isinstance(selection.stop, int)
            or selection.step not in (None, 1)
            or not 0 <= selection.start < selection.stop <= length
        ):
            raise ValueError("stamp slices must be explicit, in bounds, and unit-step")
    yy, xx = torch.meshgrid(
        torch.arange(
            y_slice.start, y_slice.stop, device=image.device, dtype=image.dtype
        ),
        torch.arange(
            x_slice.start, x_slice.stop, device=image.device, dtype=image.dtype
        ),
        indexing="ij",
    )
    return NativePixelStamp(
        data=image[y_slice, x_slice],
        variance=variance[y_slice, x_slice],
        pixel_xy=torch.stack((xx, yy), dim=-1),
        mask=None if mask is None else mask[y_slice, x_slice],
    )


@dataclass
class NativeStarFit:
    """Gaussian variable-projection result at fixed centroid and response.

    ``conditional_covariance`` concerns linear coefficients (flux first),
    conditional on the supplied response, centroid, basis and known variances.
    It excludes their uncertainty and is neither joint nuisance uncertainty
    nor a residual-scaled/regularized covariance. Flux/background coefficients
    are signed unbiased Gaussian estimators even for a nonnegative response.
    ``retained_response`` sums response over the whole footprint;
    ``valid_response`` sums over kept pixels only. These are fractions for a
    unit-throughput response, and retain throughput for a lossy response.
    """

    flux: Tensor
    background_coefficients: Tensor
    prediction: Tensor
    residual: Tensor
    chi2: Tensor
    degrees_of_freedom: int
    conditional_covariance: Tensor
    retained_response: Tensor
    valid_response: Tensor


def profile_native_star(
    stamp: NativePixelStamp,
    response: PixelIntegratedResponse,
    centroid_xy: Tensor,
    *,
    background_basis: Tensor | None = None,
    rcond: float | None = None,
) -> NativeStarFit:
    """Profile stellar flux and local additive coefficients on native pixels.

    The mean is ``flux * response(pixel_xy-centroid_xy) + basis @ background``.
    By default the background basis is a constant column. Supply [H,W,K]
    basis values explicitly (K=0 for no background). Response and basis must
    be finite on the footprint. The weighted QR least-squares solve preserves
    gradients with respect to the continuous centroid and response parameters.

    Rank is checked on the whitened design via singular values. ``rcond`` is
    only a relative numerical rank tolerance (default eps*max(rows,columns)),
    never a truncation, ridge penalty or pseudoinverse: deficiency fails closed.
    The nonlinear centroid/response optimizer is chosen by the caller. Only
    fixed diagonal Gaussian noise is supported here; this is not a Poisson,
    correlated-noise, crowded-field or shared-exposure uncertainty solver.
    """
    stamp.__post_init__()
    _same_tensor_type(centroid_xy, stamp.data, "centroid_xy")
    if centroid_xy.shape != (2,) or not bool(torch.isfinite(centroid_xy).all()):
        raise ValueError("centroid_xy must be finite with shape (2,)")
    profile = response(stamp.pixel_xy - centroid_xy)
    _same_tensor_type(profile, stamp.data, "response output")
    if profile.shape != stamp.data.shape or not bool(torch.isfinite(profile).all()):
        raise ValueError("response must return a finite tensor with the data shape")
    if bool((profile < 0).any()):
        raise ValueError("physical pixel-integrated response must be nonnegative")
    if background_basis is None:
        background_basis = torch.ones_like(stamp.data).unsqueeze(-1)
    _same_tensor_type(background_basis, stamp.data, "background_basis")
    if (
        background_basis.ndim != 3
        or background_basis.shape[:2] != stamp.data.shape
        or not bool(torch.isfinite(background_basis).all())
    ):
        raise ValueError("background_basis must be finite with shape [H,W,K]")
    design = torch.cat((profile.unsqueeze(-1), background_basis), dim=-1)
    valid = stamp.valid_mask
    n_rows = int(valid.sum())
    n_columns = design.shape[-1]
    if n_rows < n_columns:
        raise ValueError(
            "rank-deficient stamp: fewer kept pixels than linear coefficients"
        )
    if rcond is None:
        rcond = torch.finfo(stamp.data.dtype).eps * max(n_rows, n_columns)
    if not math.isfinite(rcond) or not 0 <= rcond < 1:
        raise ValueError("rcond must be finite in [0,1)")
    inverse_std = stamp.variance[valid].rsqrt()
    whitened = design[valid] * inverse_std[:, None]
    target = stamp.data[valid] * inverse_std
    if not bool(torch.isfinite(whitened).all()) or not bool(
        torch.isfinite(target).all()
    ):
        raise ValueError("whitened data/design overflow; check variance and data scale")
    singular_values = torch.linalg.svdvals(whitened.detach())
    if bool(singular_values[-1] <= rcond * singular_values[0]):
        raise ValueError(
            "rank-deficient stamp: flux/background coefficients are not identifiable"
        )
    q, r = torch.linalg.qr(whitened, mode="reduced")
    coefficients = torch.linalg.solve_triangular(
        r, (q.mT @ target).unsqueeze(-1), upper=True
    ).squeeze(-1)
    prediction = design @ coefficients
    # Masked nonfinite pixels are removed before arithmetic, including residuals.
    safe_data = torch.where(valid, stamp.data, torch.zeros_like(stamp.data))
    residual = torch.where(valid, safe_data - prediction, torch.zeros_like(prediction))
    chi2 = (residual[valid] * inverse_std).square().sum()
    inverse_r = torch.linalg.solve_triangular(
        r, torch.eye(n_columns, dtype=r.dtype, device=r.device), upper=True
    )
    return NativeStarFit(
        flux=coefficients[0],
        background_coefficients=coefficients[1:],
        prediction=prediction,
        residual=residual,
        chi2=chi2,
        degrees_of_freedom=n_rows - n_columns,
        conditional_covariance=inverse_r @ inverse_r.mT,
        retained_response=profile.sum(),
        valid_response=profile[valid].sum(),
    )
