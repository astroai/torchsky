"""Tensor-native structured-noise primitives for detector observations.

The functions in this module deliberately keep the noise model bounded and
interpretable. Pipeline variance planes are combined with positive scalar or
leading-axis calibration factors, row-correlated noise is represented by an
explicit one-over-frequency PSD, and residual contamination uses a two-
component Gaussian/Student-t mixture. There is no free-form variance image.

The FFT whitening filter is *relative* whitening: it multiplies Fourier modes
by ``sqrt(P_white / P(f))``. Thus its output has the same units as its input
and a constant PSD is exactly the identity filter. ``P(f)`` is a one-sided
power spectral density in input-units-squared per Hz; the ratio is independent
of the particular orthonormal/non-orthonormal FFT convention.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Final, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .base import NoiseOp
from .image_grid import image_shape_hw


VAR_POISSON: Final[str] = "VAR_POISSON"
VAR_RNOISE: Final[str] = "VAR_RNOISE"
VAR_FLAT: Final[str] = "VAR_FLAT"
_VARIANCE_KEYS = (VAR_POISSON, VAR_RNOISE, VAR_FLAT)


@dataclass
class ExplicitNoiseOp(NoiseOp):
    def __init__(self, *, name: str = "explicit_noise") -> None:
        super().__init__(name=name, learned=False)


@dataclass
class LearnedNoiseOp(NoiseOp):
    def __init__(self, *, name: str = "learned_noise") -> None:
        super().__init__(name=name, learned=True)


@dataclass
class StructuredNoiseOp(NoiseOp):
    model: nn.Module | None = None

    def __init__(
        self,
        model: nn.Module | None = None,
        *,
        name: str = "structured_noise",
        frozen: bool = True,
    ) -> None:
        super().__init__(name=name, learned=True, frozen=bool(frozen))
        self.model = model


def sanitize_inverse_variance(weight: Tensor) -> Tensor:
    return torch.where(
        torch.isfinite(weight) & (weight > 0.0),
        weight,
        torch.zeros((), dtype=weight.dtype, device=weight.device),
    )


def _reference_tensor(*values: object) -> Tensor | None:
    """Return a floating tensor from ``values`` for dtype/device coercion."""
    for value in values:
        if isinstance(value, Tensor) and value.is_floating_point():
            return value
    for value in values:
        if isinstance(value, Tensor):
            return value.to(dtype=torch.get_default_dtype())
    return None


def _coerce(value: Tensor | float, reference: Tensor) -> Tensor:
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def _validate_finite(value: Tensor, *, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_nonnegative(value: Tensor, *, name: str) -> None:
    _validate_finite(value, name=name)
    if bool((value < 0.0).any()):
        raise ValueError(f"{name} must be non-negative")


def _validate_positive(value: Tensor, *, name: str) -> None:
    _validate_finite(value, name=name)
    if bool((value <= 0.0).any()):
        raise ValueError(f"{name} must be positive")


def _broadcast_parameter(value: Tensor, shape: tuple[int, ...], *, name: str) -> Tensor:
    """Broadcast a scalar/map or a leading exposure parameter to ``shape``.

    Standard PyTorch broadcasting handles explicit detector maps (``H x W``
    against ``E x H x W``). A one-dimensional calibration vector always
    indexes the leading exposure/amp axis, even when its length also equals a
    trailing dimension. Trailing-axis calibration maps must therefore include
    explicit singleton dimensions, such as ``(1, W)``.
    """
    if value.ndim > len(shape):
        raise ValueError(f"{name} has too many dimensions for target shape {shape}")
    if value.ndim == 1 and len(shape) > 1:
        leading = value.reshape(value.shape + (1,) * (len(shape) - 1))
        try:
            torch.broadcast_shapes(tuple(leading.shape), shape)
        except RuntimeError as exc:
            raise ValueError(
                f"{name} leading-axis shape {tuple(value.shape)} is not "
                f"broadcastable to {shape}; use an explicit shape for a trailing map"
            ) from exc
        return leading
    try:
        torch.broadcast_shapes(tuple(value.shape), shape)
    except RuntimeError:
        leading = value.reshape(value.shape + (1,) * (len(shape) - value.ndim))
        try:
            torch.broadcast_shapes(tuple(leading.shape), shape)
        except RuntimeError as exc:
            raise ValueError(
                f"{name} shape {tuple(value.shape)} is not broadcastable to {shape}"
            ) from exc
        return leading
    return value


def _inverse_softplus(value: Tensor) -> Tensor:
    """Stable inverse of softplus for strictly positive initialization values."""
    tiny = torch.finfo(value.dtype).tiny
    shifted = value.clamp_min(tiny)
    return shifted + torch.log(-torch.expm1(-shifted))


def _positive_parameter(raw: Tensor, minimum: float) -> Tensor:
    return F.softplus(raw) + raw.new_tensor(float(minimum))


def pipeline_variance(
    components: Mapping[str, Tensor] | None = None,
    *,
    var_poisson: Tensor | None = None,
    var_rnoise: Tensor | None = None,
    var_flat: Tensor | None = None,
    err: Tensor | None = None,
    poisson_scale: Tensor | float = 1.0,
    rnoise_scale: Tensor | float = 1.0,
    flat_scale: Tensor | float = 1.0,
    err_scale: Tensor | float = 1.0,
    variance_floor: Tensor | float = 0.0,
    floor: Tensor | float | None = None,
) -> Tensor:
    """Combine pipeline variance components into a positive variance map.

    The preferred input is a mapping with keys :data:`VAR_POISSON`,
    :data:`VAR_RNOISE`, and :data:`VAR_FLAT`. Explicit ``var_*`` keywords are
    accepted as a lightweight adapter for observation objects. For available
    component planes the result is

    ``s_P**2 * VAR_POISSON + s_R**2 * VAR_RNOISE + s_F**2 * VAR_FLAT + floor``.

    If no component plane is available, ``err`` is interpreted as a standard
    deviation and the result is ``err_scale**2 * err**2 + floor``. Missing
    component planes contribute zero. ``floor`` is an additive *variance*
    floor, not a standard-deviation floor; zero is allowed by this functional
    form, while :class:`PipelineVarianceCalibration` enforces a positive floor
    by default.
    """
    if floor is not None:
        if variance_floor != 0.0:
            raise ValueError("pass only one of variance_floor and floor")
        variance_floor = floor

    planes: dict[str, Tensor] = {}
    if components is not None:
        unknown = set(components).difference(_VARIANCE_KEYS)
        if unknown:
            raise ValueError(f"unknown variance component(s): {sorted(unknown)}")
        planes.update(components)
    for key, value in (
        (VAR_POISSON, var_poisson),
        (VAR_RNOISE, var_rnoise),
        (VAR_FLAT, var_flat),
    ):
        if value is not None:
            if key in planes:
                raise ValueError(f"variance component {key} was provided twice")
            planes[key] = value
    if not planes and err is None:
        raise ValueError("provide at least one variance component or ERR")

    reference = _reference_tensor(
        *planes.values(),
        err,
        poisson_scale,
        rnoise_scale,
        flat_scale,
        err_scale,
        variance_floor,
    )
    if reference is None:
        reference = torch.tensor(0.0, dtype=torch.get_default_dtype())
    if not reference.is_floating_point():
        reference = reference.to(dtype=torch.get_default_dtype())

    values = {key: _coerce(value, reference) for key, value in planes.items()}
    for key, value in values.items():
        _validate_nonnegative(value, name=key)
    floor_tensor = _coerce(variance_floor, reference)
    _validate_nonnegative(floor_tensor, name="variance_floor")

    if values:
        try:
            shape = torch.broadcast_shapes(
                *(tuple(value.shape) for value in values.values())
            )
        except RuntimeError as exc:
            raise ValueError("variance component planes are not broadcastable") from exc
        result: Tensor | None = None
        scales = {
            VAR_POISSON: _coerce(poisson_scale, reference),
            VAR_RNOISE: _coerce(rnoise_scale, reference),
            VAR_FLAT: _coerce(flat_scale, reference),
        }
        for key, value in values.items():
            scale = scales[key]
            _validate_positive(scale, name=f"{key} scale")
            term = (
                _broadcast_parameter(scale, shape, name=f"{key} scale").square() * value
            )
            result = term if result is None else result + term
        floor_term = _broadcast_parameter(floor_tensor, shape, name="variance_floor")
        assert result is not None
        return result + floor_term

    assert err is not None
    err_tensor = _coerce(err, reference)
    _validate_nonnegative(err_tensor, name="ERR")
    scale = _coerce(err_scale, reference)
    _validate_positive(scale, name="ERR scale")
    scale = _broadcast_parameter(scale, tuple(err_tensor.shape), name="ERR scale")
    floor_tensor = _broadcast_parameter(
        floor_tensor, tuple(err_tensor.shape), name="variance_floor"
    )
    try:
        shape = torch.broadcast_shapes(
            tuple(err_tensor.shape), tuple(scale.shape), tuple(floor_tensor.shape)
        )
    except RuntimeError as exc:
        raise ValueError(
            "ERR, ERR scale, and variance_floor are not broadcastable"
        ) from exc
    return scale.square() * err_tensor.square() + floor_tensor


calibrated_pipeline_variance = pipeline_variance


class PipelineVarianceCalibration(nn.Module):
    """Positive, differentiable calibration factors for pipeline variances.

    Parameters are stored in unconstrained softplus coordinates. Their
    leading dimensions are preserved, allowing scalar factors, per-exposure
    factors, or per-exposure/per-amplifier factors without introducing a
    detector-pixel variance image.
    """

    def __init__(
        self,
        *,
        poisson_scale: Tensor | float = 1.0,
        rnoise_scale: Tensor | float = 1.0,
        flat_scale: Tensor | float = 1.0,
        err_scale: Tensor | float = 1.0,
        variance_floor: Tensor | float = 1.0e-8,
        min_scale: float = 1.0e-6,
        min_floor: float = 1.0e-12,
    ) -> None:
        super().__init__()
        if not math.isfinite(float(min_scale)) or min_scale <= 0.0:
            raise ValueError("min_scale must be finite and positive")
        if not math.isfinite(float(min_floor)) or min_floor < 0.0:
            raise ValueError("min_floor must be finite and non-negative")
        self.min_scale = float(min_scale)
        self.min_floor = float(min_floor)
        reference = _reference_tensor(
            poisson_scale,
            rnoise_scale,
            flat_scale,
            err_scale,
            variance_floor,
        )
        if reference is None:
            reference = torch.tensor(0.0, dtype=torch.get_default_dtype())
        if not reference.is_floating_point():
            reference = reference.to(dtype=torch.get_default_dtype())

        def parameter(
            value: Tensor | float,
            *,
            name: str,
            minimum: float,
        ) -> nn.Parameter:
            tensor = _coerce(value, reference)
            _validate_finite(tensor, name=name)
            if bool((tensor <= minimum).any()):
                raise ValueError(f"{name} must be greater than its minimum")
            return nn.Parameter(_inverse_softplus(tensor - tensor.new_tensor(minimum)))

        self.raw_poisson_scale = parameter(
            poisson_scale, name="poisson_scale", minimum=self.min_scale
        )
        self.raw_rnoise_scale = parameter(
            rnoise_scale, name="rnoise_scale", minimum=self.min_scale
        )
        self.raw_flat_scale = parameter(
            flat_scale, name="flat_scale", minimum=self.min_scale
        )
        self.raw_err_scale = parameter(
            err_scale, name="err_scale", minimum=self.min_scale
        )
        self.raw_variance_floor = parameter(
            variance_floor, name="variance_floor", minimum=self.min_floor
        )

    @property
    def poisson_scale(self) -> Tensor:
        return _positive_parameter(self.raw_poisson_scale, self.min_scale)

    @property
    def rnoise_scale(self) -> Tensor:
        return _positive_parameter(self.raw_rnoise_scale, self.min_scale)

    @property
    def flat_scale(self) -> Tensor:
        return _positive_parameter(self.raw_flat_scale, self.min_scale)

    @property
    def err_scale(self) -> Tensor:
        return _positive_parameter(self.raw_err_scale, self.min_scale)

    @property
    def variance_floor(self) -> Tensor:
        return _positive_parameter(self.raw_variance_floor, self.min_floor)

    def forward(
        self,
        components: Mapping[str, Tensor] | None = None,
        *,
        var_poisson: Tensor | None = None,
        var_rnoise: Tensor | None = None,
        var_flat: Tensor | None = None,
        err: Tensor | None = None,
    ) -> Tensor:
        return pipeline_variance(
            components,
            var_poisson=var_poisson,
            var_rnoise=var_rnoise,
            var_flat=var_flat,
            err=err,
            poisson_scale=self.poisson_scale,
            rnoise_scale=self.rnoise_scale,
            flat_scale=self.flat_scale,
            err_scale=self.err_scale,
            variance_floor=self.variance_floor,
        )


def one_over_f_row_psd(
    num_rows: int,
    *,
    white_psd: Tensor | float = 1.0,
    knee_frequency: Tensor | float = 0.0,
    exponent: Tensor | float = 1.0,
    sample_spacing: float = 1.0,
    minimum_frequency: float | None = None,
) -> Tensor:
    """Return an amplifier-broadcastable one-sided ``1/f`` row PSD.

    ``num_rows`` is the number of samples along a detector row-process axis.
    ``sample_spacing`` is the interval between adjacent rows in seconds, so
    frequencies are in Hz. ``white_psd`` is the high-frequency PSD baseline
    in input-units-squared/Hz; ``knee_frequency`` is in Hz and ``exponent`` is
    dimensionless. Leading parameter dimensions are preserved, e.g. four
    amplifier values produce ``(4, num_rows // 2 + 1)``.

    The DC bin uses the lowest resolvable frequency ``1/(N*dt)`` rather than
    an infinite value. This is the finite-window convention used by the FFT
    whitening filter and keeps gradients finite at DC.
    """
    if not isinstance(num_rows, int) or num_rows < 1:
        raise ValueError("num_rows must be a positive integer")
    if not math.isfinite(float(sample_spacing)) or sample_spacing <= 0.0:
        raise ValueError("sample_spacing must be finite and positive")
    reference = _reference_tensor(white_psd, knee_frequency, exponent)
    if reference is None:
        reference = torch.tensor(0.0, dtype=torch.get_default_dtype())
    if not reference.is_floating_point():
        reference = reference.to(dtype=torch.get_default_dtype())
    white = _coerce(white_psd, reference)
    knee = _coerce(knee_frequency, reference)
    power = _coerce(exponent, reference)
    _validate_positive(white, name="white_psd")
    _validate_nonnegative(knee, name="knee_frequency")
    _validate_positive(power, name="exponent")
    try:
        white, knee, power = torch.broadcast_tensors(white, knee, power)
    except RuntimeError as exc:
        raise ValueError("PSD parameters are not broadcastable") from exc

    fft_dtype = reference.dtype
    if fft_dtype in (torch.float16, torch.bfloat16):
        fft_dtype = torch.float32
        white = white.to(dtype=fft_dtype)
        knee = knee.to(dtype=fft_dtype)
        power = power.to(dtype=fft_dtype)
    frequencies = torch.fft.rfftfreq(
        num_rows,
        d=float(sample_spacing),
        device=reference.device,
        dtype=fft_dtype,
    )
    if minimum_frequency is None:
        minimum = 1.0 / (float(num_rows) * float(sample_spacing))
    else:
        minimum = float(minimum_frequency)
        if not math.isfinite(minimum) or minimum <= 0.0:
            raise ValueError("minimum_frequency must be finite and positive")
    minimum_tensor = frequencies.new_tensor(minimum)
    safe_frequency = torch.where(frequencies > 0.0, frequencies, minimum_tensor)
    ratio = knee[..., None] / safe_frequency
    # ``0 ** p`` has an infinite (p < 1) or zero (p >= 1) knee gradient and a NaN
    # exponent gradient at knee == 0; the knee == 0 branch is exactly white, so
    # evaluate the power only where knee > 0 to keep every gradient finite.
    has_knee = ratio > 0.0
    knee_term = torch.where(
        has_knee,
        torch.where(has_knee, ratio, torch.ones_like(ratio)).pow(power[..., None]),
        torch.zeros_like(ratio),
    )
    return white[..., None] * (1.0 + knee_term)


one_over_f_psd = one_over_f_row_psd


def fft_whiten_rows(
    rows: Tensor,
    psd: Tensor | None = None,
    *,
    white_psd: Tensor | float | None = None,
    knee_frequency: Tensor | float = 0.0,
    exponent: Tensor | float = 1.0,
    sample_spacing: float = 1.0,
    dim: int = -1,
) -> Tensor:
    """Apply relative FFT whitening along a row-process dimension.

    ``rows`` keeps its input units. If ``psd`` is omitted it is generated by
    :func:`one_over_f_row_psd`; otherwise its final axis must have length
    ``N//2 + 1``. ``white_psd`` sets the reference PSD for the dimensionless
    gain ``sqrt(white_psd / psd)``. When an explicit PSD is supplied and no
    reference is given, its last (Nyquist-side) bin is used per leading
    parameter slice. A constant PSD therefore gives an exact identity limit.
    """
    if not isinstance(rows, Tensor) or not rows.is_floating_point():
        raise ValueError("rows must be a floating-point tensor")
    if rows.ndim < 1:
        raise ValueError("rows must have at least one dimension")
    if dim < -rows.ndim or dim >= rows.ndim:
        raise ValueError("dim is out of range for rows")
    dim = dim % rows.ndim
    moved = rows.movedim(dim, -1)
    num_rows = int(moved.shape[-1])
    if num_rows == 0:
        return rows.clone()
    if psd is None:
        reference_white = 1.0 if white_psd is None else white_psd
        psd_tensor = one_over_f_row_psd(
            num_rows,
            white_psd=reference_white,
            knee_frequency=knee_frequency,
            exponent=exponent,
            sample_spacing=sample_spacing,
        ).to(device=rows.device, dtype=rows.dtype)
        reference = _coerce(reference_white, rows)
        white = _broadcast_parameter(
            reference,
            tuple(psd_tensor.shape[:-1]),
            name="white_psd",
        )[..., None]
    else:
        psd_tensor = torch.as_tensor(psd, device=rows.device, dtype=rows.dtype)
        if psd_tensor.ndim < 1 or int(psd_tensor.shape[-1]) != num_rows // 2 + 1:
            raise ValueError("psd final axis must have length num_rows // 2 + 1")
        _validate_positive(psd_tensor, name="psd")
        if white_psd is None:
            white = psd_tensor[..., -1:]
        else:
            reference = _coerce(white_psd, rows)
            white = _broadcast_parameter(
                reference,
                tuple(psd_tensor.shape[:-1]),
                name="white_psd",
            )[..., None]
        _validate_positive(white, name="white_psd")

    try:
        gain = torch.sqrt(white / psd_tensor)
        transformed = torch.fft.rfft(moved, dim=-1)
        gain, transformed = torch.broadcast_tensors(gain, transformed)
    except RuntimeError as exc:
        raise ValueError(
            "psd leading dimensions are not broadcastable to rows"
        ) from exc
    whitened = torch.fft.irfft(transformed * gain, n=num_rows, dim=-1)
    return whitened.movedim(-1, dim)


whiten_rows_fft = fft_whiten_rows


class OneOverFRowNoise(nn.Module):
    """Positive learnable one-over-frequency PSDs for amp/exposure groups.

    The knee is learned in log space (``knee = min_knee_frequency + exp(raw)``)
    so its gradient scales with the knee instead of vanishing at small values.
    ``knee_frequency=None`` starts at ``0.01 / sample_spacing`` (one hundredth of
    the sampling rate, 50x below Nyquist): a nearly white initial PSD that still
    has a usable knee gradient. An explicit ``0`` is accepted but sits at the
    ``min_knee_frequency`` floor and is effectively unlearnable.
    """

    def __init__(
        self,
        parameter_shape: int | tuple[int, ...] = (4,),
        *,
        white_psd: Tensor | float = 1.0,
        knee_frequency: Tensor | float | None = None,
        exponent: Tensor | float = 1.0,
        sample_spacing: float = 1.0,
        min_white_psd: float = 1.0e-8,
        min_knee_frequency: float = 1.0e-12,
        min_exponent: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if isinstance(parameter_shape, int):
            parameter_shape = (parameter_shape,)
        parameter_shape = tuple(int(size) for size in parameter_shape)
        if not parameter_shape or any(size < 1 for size in parameter_shape):
            raise ValueError("parameter_shape must contain positive dimensions")
        if not math.isfinite(float(sample_spacing)) or sample_spacing <= 0.0:
            raise ValueError("sample_spacing must be finite and positive")
        for name, value in (
            ("min_white_psd", min_white_psd),
            ("min_knee_frequency", min_knee_frequency),
            ("min_exponent", min_exponent),
        ):
            if not math.isfinite(float(value)) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        self.parameter_shape = parameter_shape
        self.sample_spacing = float(sample_spacing)
        self.min_white_psd = float(min_white_psd)
        self.min_knee_frequency = float(min_knee_frequency)
        self.min_exponent = float(min_exponent)
        if knee_frequency is None:
            knee_frequency = 0.01 / float(sample_spacing)
        reference = _reference_tensor(white_psd, knee_frequency, exponent)
        if reference is None:
            reference = torch.tensor(0.0, dtype=torch.get_default_dtype())
        if not reference.is_floating_point():
            reference = reference.to(dtype=torch.get_default_dtype())

        def init(
            value: Tensor | float,
            *,
            name: str,
            minimum: float,
            allow_zero: bool = False,
            log_space: bool = False,
        ) -> nn.Parameter:
            tensor = _coerce(value, reference)
            if tensor.numel() == 1:
                tensor = tensor.expand(parameter_shape).clone()
            elif tuple(tensor.shape) != parameter_shape:
                raise ValueError(
                    f"{name} must be scalar or have shape {parameter_shape}"
                )
            _validate_finite(tensor, name=name)
            if allow_zero:
                if bool((tensor < 0.0).any()):
                    raise ValueError(f"{name} must be non-negative")
            elif bool((tensor < minimum).any()):
                raise ValueError(f"{name} must be at least {minimum}")
            shifted = (tensor - tensor.new_tensor(minimum)).clamp_min(
                torch.finfo(tensor.dtype).tiny
            )
            if log_space:
                return nn.Parameter(torch.log(shifted))
            return nn.Parameter(_inverse_softplus(shifted))

        self.raw_white_psd = init(
            white_psd, name="white_psd", minimum=self.min_white_psd
        )
        self.raw_knee_frequency = init(
            knee_frequency,
            name="knee_frequency",
            minimum=self.min_knee_frequency,
            allow_zero=True,
            log_space=True,
        )
        self.raw_exponent = init(exponent, name="exponent", minimum=self.min_exponent)

    @property
    def white_psd(self) -> Tensor:
        return _positive_parameter(self.raw_white_psd, self.min_white_psd)

    @property
    def knee_frequency(self) -> Tensor:
        return self.raw_knee_frequency.exp() + self.raw_knee_frequency.new_tensor(
            self.min_knee_frequency
        )

    @property
    def exponent(self) -> Tensor:
        return _positive_parameter(self.raw_exponent, self.min_exponent)

    def psd(self, num_rows: int) -> Tensor:
        return one_over_f_row_psd(
            num_rows,
            white_psd=self.white_psd,
            knee_frequency=self.knee_frequency,
            exponent=self.exponent,
            sample_spacing=self.sample_spacing,
        )

    def forward(self, rows: Tensor, *, dim: int = -1) -> Tensor:
        if dim < -rows.ndim or dim >= rows.ndim:
            raise ValueError("dim is out of range for rows")
        if rows.shape[dim] == 0:
            return rows.clone()
        num_rows = int(rows.shape[dim])
        return fft_whiten_rows(
            rows,
            self.psd(num_rows),
            white_psd=self.white_psd,
            dim=dim,
        )


def _mixture_terms(
    residual: Tensor,
    variance: Tensor,
    *,
    outlier_scale: Tensor | float,
    degrees_of_freedom: Tensor | float,
    outlier_probability: Tensor | float,
    prior: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(residual, Tensor) or not isinstance(variance, Tensor):
        raise ValueError("residual and variance must be tensors")
    reference = _reference_tensor(
        residual,
        variance,
        outlier_scale,
        degrees_of_freedom,
        outlier_probability,
        prior,
    )
    if reference is None:
        reference = torch.tensor(0.0, dtype=torch.get_default_dtype())
    if not reference.is_floating_point():
        reference = reference.to(dtype=torch.get_default_dtype())
    residual = _coerce(residual, reference)
    variance = _coerce(variance, reference)
    try:
        residual, variance = torch.broadcast_tensors(residual, variance)
    except RuntimeError as exc:
        raise ValueError("residual and variance are not broadcastable") from exc
    _validate_finite(residual, name="residual")
    _validate_positive(variance, name="variance")
    shape = tuple(residual.shape)
    scale = _broadcast_parameter(
        _coerce(outlier_scale, reference), shape, name="outlier_scale"
    )
    dof = _broadcast_parameter(
        _coerce(degrees_of_freedom, reference), shape, name="degrees_of_freedom"
    )
    probability_value = outlier_probability if prior is None else prior
    probability = _broadcast_parameter(
        _coerce(probability_value, reference), shape, name="outlier prior"
    )
    _validate_positive(scale, name="outlier_scale")
    _validate_positive(dof, name="degrees_of_freedom")
    _validate_finite(probability, name="outlier prior")
    if bool((probability < 0.0).any()) or bool((probability > 1.0).any()):
        raise ValueError("outlier prior must be in [0, 1]")

    gaussian_log_prob = -0.5 * (
        residual.new_tensor(math.log(2.0 * math.pi))
        + variance.log()
        + residual.square() / variance
    )
    student_scale = scale * torch.sqrt(variance)
    student_log_prob = (
        torch.lgamma((dof + 1.0) / 2.0)
        - torch.lgamma(dof / 2.0)
        - 0.5
        * (torch.log(dof * residual.new_tensor(math.pi)) + 2.0 * student_scale.log())
        - ((dof + 1.0) / 2.0)
        * torch.log1p(residual.square() / (dof * student_scale.square()))
    )
    eps = torch.finfo(residual.dtype).eps
    probability = probability.clamp(min=eps, max=1.0 - eps)
    log_inlier = gaussian_log_prob + torch.log1p(-probability)
    log_outlier = student_log_prob + probability.log()
    return log_inlier, log_outlier, torch.logaddexp(log_inlier, log_outlier)


def gaussian_student_t_mixture_nll(
    residual: Tensor,
    variance: Tensor,
    *,
    outlier_scale: Tensor | float = 10.0,
    degrees_of_freedom: Tensor | float = 3.0,
    outlier_probability: Tensor | float = 0.01,
    prior: Tensor | None = None,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> Tensor:
    """NLL for Gaussian inliers plus Student-t outliers.

    ``variance`` is the inlier Gaussian variance. ``outlier_scale`` is a
    positive, dimensionless multiplier on its standard deviation for the
    Student-t component. ``prior`` is an optional per-pixel outlier
    probability map; when supplied it overrides ``outlier_probability``.
    """
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    _, _, log_mixture = _mixture_terms(
        residual,
        variance,
        outlier_scale=outlier_scale,
        degrees_of_freedom=degrees_of_freedom,
        outlier_probability=outlier_probability,
        prior=prior,
    )
    nll = -log_mixture
    if reduction == "none":
        return nll
    if nll.numel() == 0:
        raise ValueError("mean/sum reduction is undefined for an empty residual tensor")
    return nll.mean() if reduction == "mean" else nll.sum()


def _reduce_nll(nll: Tensor, reduction: Literal["none", "mean", "sum"]) -> Tensor:
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if reduction == "none":
        return nll
    if nll.numel() == 0:
        raise ValueError("mean/sum reduction is undefined for an empty tensor")
    return nll.mean() if reduction == "mean" else nll.sum()


def reduce_detector_nll(
    nll: Tensor,
    *,
    mask: Tensor | None = None,
    sample_weight: Tensor | None = None,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> Tensor:
    """Reduce per-pixel detector NLL with ``True = usable`` mask semantics.

    ``sum`` with no ``sample_weight`` is the normalized likelihood sum over the
    selected measurements. ``mean`` is a valid-pixel reporting/training
    objective. Supplying non-binary ``sample_weight`` defines a weighted
    composite objective; it must not be reported as a normalized likelihood.
    ``none`` returns the original per-pixel values unchanged.
    """
    values = torch.as_tensor(nll)
    if not values.is_floating_point():
        raise TypeError("nll must be floating point")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if reduction == "none":
        return values

    if mask is None:
        valid = torch.ones(values.shape, device=values.device, dtype=torch.bool)
    else:
        mask_t = torch.as_tensor(mask, device=values.device)
        if mask_t.dtype != torch.bool:
            raise TypeError("mask must be boolean (True = usable)")
        try:
            valid = torch.broadcast_to(mask_t, values.shape)
        except RuntimeError as exc:
            raise ValueError("mask must broadcast to nll shape") from exc
    selected = values[valid]
    if selected.numel() == 0:
        raise ValueError("detector NLL reduction has no usable values")
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("usable detector NLL values must be finite")

    if sample_weight is None:
        return selected.sum() if reduction == "sum" else selected.mean()
    weights = torch.as_tensor(
        sample_weight,
        device=values.device,
        dtype=values.dtype,
    )
    try:
        weights = torch.broadcast_to(weights, values.shape)[valid]
    except RuntimeError as exc:
        raise ValueError("sample_weight must broadcast to nll shape") from exc
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0.0).any()):
        raise ValueError("sample_weight must be finite and non-negative")
    total_weight = weights.sum()
    if not bool(total_weight > 0.0):
        raise ValueError("sample_weight must have positive usable mass")
    numerator = torch.sum(weights * selected)
    return numerator if reduction == "sum" else numerator / total_weight


def gaussian_nll(
    observed: Tensor,
    expected: Tensor,
    variance: Tensor,
    *,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> Tensor:
    """Normalized Gaussian detector negative log likelihood.

    Per pixel ``0.5 * (log(2 pi variance) + (observed - expected)^2 / variance)``,
    including the normalization, so values are comparable across noise models
    and differentiable with respect to ``expected`` and ``variance``.

    Args:
        observed: Measured values.
        expected: Model prediction, broadcastable against ``observed``.
        variance: Strictly positive per-pixel variance, broadcastable.
        reduction: ``"none"`` (per pixel), ``"mean"``, or ``"sum"``.

    Returns:
        Negative log-likelihood in nats, reduced as requested.

    Raises:
        ValueError: If inputs are not finite or a variance is not positive.

    Examples:
        >>> import math, torch
        >>> from torchsky.map.operators import gaussian_nll
        >>> value = gaussian_nll(torch.zeros(3), torch.zeros(3), torch.ones(3))
        >>> abs(float(value) - 0.5 * math.log(2.0 * math.pi)) < 1e-6
        True
    """
    observed = torch.as_tensor(observed)
    expected = torch.as_tensor(expected, device=observed.device, dtype=observed.dtype)
    variance = torch.as_tensor(variance, device=observed.device, dtype=observed.dtype)
    observed, expected, variance = torch.broadcast_tensors(observed, expected, variance)
    _validate_finite(observed, name="observed")
    _validate_finite(expected, name="expected")
    _validate_positive(variance, name="variance")
    nll = 0.5 * (
        variance.log()
        + observed.new_tensor(math.log(2.0 * math.pi))
        + (observed - expected).square() / variance
    )
    return _reduce_nll(nll, reduction)


def student_t_nll(
    observed: Tensor,
    expected: Tensor,
    scale: Tensor,
    degrees_of_freedom: Tensor | float,
    *,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> Tensor:
    """Normalized location-scale Student-t detector negative log likelihood."""
    observed = torch.as_tensor(observed)
    expected = torch.as_tensor(expected, device=observed.device, dtype=observed.dtype)
    scale = torch.as_tensor(scale, device=observed.device, dtype=observed.dtype)
    observed, expected, scale = torch.broadcast_tensors(observed, expected, scale)
    dof = torch.broadcast_to(
        torch.as_tensor(
            degrees_of_freedom, device=observed.device, dtype=observed.dtype
        ),
        observed.shape,
    )
    _validate_finite(observed, name="observed")
    _validate_finite(expected, name="expected")
    _validate_positive(scale, name="scale")
    _validate_positive(dof, name="degrees_of_freedom")
    residual = (observed - expected) / scale
    log_prob = (
        torch.lgamma((dof + 1.0) / 2.0)
        - torch.lgamma(dof / 2.0)
        - 0.5 * torch.log(dof * observed.new_tensor(math.pi))
        - scale.log()
        - 0.5 * (dof + 1.0) * torch.log1p(residual.square() / dof)
    )
    return _reduce_nll(-log_prob, reduction)


def _integer_probability_powers(base: Tensor, degree: Tensor) -> Tensor:
    power = degree.abs()
    nonlinear_base = torch.where(power >= 2, base, 1.0)
    return torch.where(
        power == 0,
        1.0,
        torch.where(power == 1, base, nonlinear_base.pow(power.clamp_min(2.0))),
    )


def _poisson_mode_probability_terms(
    rate: Tensor, counts: Tensor, included: Tensor
) -> Tensor:
    """Mode-scaled finite probabilities without log-rate differentiation."""
    mode = rate.detach().floor().unsqueeze(-1)
    degree = counts.to(rate.dtype) - mode
    below_mode = degree < 0
    # Integer reference modes bound both coefficients and power bases by one.
    # The zero-rate value is frozen here; its right derivative is supplied by
    # the same analytic boundary jet used by the logarithmic density path.
    positive_rate = torch.where(rate == 0, 1.0, rate).unsqueeze(-1)
    left_scale = mode.clamp_min(1.0)
    right_scale = mode + 1.0
    # These recurrences depend only on the detached mode. Keeping rate
    # dependence in explicit powers avoids cumprod's division-based Hessian;
    # ratios also avoid subtracting large, nearly equal log-factorials.
    right_coefficient = torch.cumprod(
        torch.where(degree > 0, right_scale / counts.clamp_min(1), 1.0), dim=-1
    )
    left_coefficient = torch.cumprod(
        torch.where(below_mode, (counts + 1) / left_scale, 1.0).flip(-1), dim=-1
    ).flip(-1)
    coefficient = torch.where(below_mode, left_coefficient, right_coefficient)
    left_rate = torch.where(mode > 0, positive_rate, 1.0)
    base = torch.where(below_mode, left_scale / left_rate, positive_rate / right_scale)
    # Isolate degrees zero/one: the generic tensor-power Hessian otherwise
    # evaluates zero times an overflowing inverse power at tiny positive rates.
    powers = _integer_probability_powers(base, degree)
    weights = (coefficient * powers).masked_fill(~included, 0.0)
    weights = torch.where(
        (rate == 0).unsqueeze(-1), (counts == 0).to(rate.dtype), weights
    )
    return weights


def _normalized_poisson_probability_weights(
    rate: Tensor, counts: Tensor, included: Tensor
) -> Tensor:
    weights = _poisson_mode_probability_terms(rate, counts, included)
    return weights / weights.sum(dim=-1, keepdim=True)


def _prepare_poisson_gaussian(
    value: Tensor,
    expected_electrons: Tensor,
    read_noise_variance: Tensor,
    gain: Tensor | float,
    background: Tensor | float,
    tail_sigma: float,
    max_terms: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Validate and broadcast detector inputs to a common shape."""
    if not math.isfinite(float(tail_sigma)) or tail_sigma <= 0.0:
        raise ValueError("tail_sigma must be finite and positive")
    if int(max_terms) < 2:
        raise ValueError("max_terms must be at least 2")

    value = torch.as_tensor(value)
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError("value must have real float32 or float64 dtype")
    if float(tail_sigma) > math.sqrt(torch.finfo(value.dtype).max):
        raise ValueError("Poisson--Gaussian support radius exceeds the dtype range")
    rate = torch.as_tensor(expected_electrons, device=value.device, dtype=value.dtype)
    read_var = torch.as_tensor(
        read_noise_variance, device=value.device, dtype=value.dtype
    )
    gain_t = torch.as_tensor(gain, device=value.device, dtype=value.dtype)
    background_t = torch.as_tensor(background, device=value.device, dtype=value.dtype)
    value, rate, read_var, gain_t, background_t = torch.broadcast_tensors(
        value, rate, read_var, gain_t, background_t
    )
    _validate_finite(value, name="value")
    _validate_nonnegative(rate, name="expected_electrons")
    _validate_positive(read_var, name="read_noise_variance")
    _validate_positive(gain_t, name="gain")
    _validate_finite(background_t, name="background")
    return value, rate, read_var, gain_t, background_t


# Working-set budget per chunk (bytes); peak memory is independent of frame size.
_POISSON_GAUSSIAN_CHUNK_BYTES: Final[int] = 256 << 20
# Autograd-saved bytes allowed before chunks are checkpointed (recomputed in
# backward). Below it the graph is kept: faster, and memory is still bounded.
_POISSON_GAUSSIAN_SAVE_BYTES: Final[int] = 8 << 30
# Live (pixel x count) tensors per term, in units of the element size.
_VALUE_TEMPORARIES: Final[int] = 12
_DERIVATIVE_TEMPORARIES: Final[int] = 32
# Streaming moment pass (values plus analytic gradients), same units.
_MOMENT_TEMPORARIES: Final[int] = 18
# Bisection brackets the posterior mode to within this many counts; the support
# is taken from the bracket ends, so the window bound stays rigorous.
_POISSON_MODE_TOLERANCE: Final[float] = 0.25


def _poisson_gaussian_window(
    value: Tensor,
    rate: Tensor,
    read_var: Tensor,
    gain: Tensor,
    background: Tensor,
    *,
    tail_sigma: float,
    cdf: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-pixel inclusive count window ``(lower, upper)`` and a ``far`` flag.

    The prior window is ``rate +- (tail_sigma sqrt(rate + 1) + tail_sigma^2)``.
    The observation-conditioned weight ``Pois(n; rate) N(value; gain n +
    background, read_var)`` is log-concave in ``n``, so its maximiser lies
    between the Poisson mode and the Gaussian centre ``(value - background) /
    gain``; it is located by bisection on the exactly monotone score. The
    posterior standard deviation is at most ``sqrt(1/(1/(m+1) + gain^2/var))``
    (``psi'(x) >= 1/x``), so the same ``tail_sigma`` Gaussian-equivalent radius
    around the mode retains the posterior tail beyond ``tail_sigma`` sigmas.
    Overlapping windows are merged (a superset of the prior window, so
    in-distribution pixels are unchanged up to the prior tail mass). A
    posterior window disjoint from the prior window is used on its own and
    flagged ``far`` (+1 above, -1 below): its terms are evaluated with the
    exact, unrenormalised Poisson log pmf. For the CDF a far-above pixel needs
    only the prior window (the CDF is one there), so it is not flagged.
    """
    dtype = value.dtype
    eps = torch.finfo(dtype).eps
    tail = float(tail_sigma)
    rate = rate.detach()
    radius = tail * torch.sqrt(rate + 1.0) + tail**2
    plo = torch.floor((rate - radius).clamp_min(0.0))
    phi = torch.ceil(rate + radius)
    positive = rate > 0
    cap = 2.0 / eps
    centre = ((value - background) / gain).detach()
    lo = (torch.minimum(rate, centre) - 1.0).clamp_min(0.0)
    hi = torch.maximum(rate, centre).clamp(0.0, cap)
    lo = torch.minimum(lo, hi)
    log_rate = torch.where(positive, rate, 1.0).log()
    slope = (gain / read_var).detach()
    resid = (value - background).detach()
    active = positive & ((hi - lo) > _POISSON_MODE_TOLERANCE)
    if bool(active.any()):
        idx = active.nonzero(as_tuple=True)
        a, b = lo[idx], hi[idx]
        lr, sl, rs, g = log_rate[idx], slope[idx], resid[idx], gain.detach()[idx]

        def score_at(x):
            return lr - torch.digamma(x + 1.0) + sl * (rs - g * x)

        # The score is monotone decreasing: a boundary root is located exactly.
        a = torch.where(score_at(b) >= 0, b, a)
        b = torch.where(score_at(a) <= 0, a, b)
        steps = int(
            math.ceil(
                math.log2(
                    max(float((b - a).max()), _POISSON_MODE_TOLERANCE)
                    / _POISSON_MODE_TOLERANCE
                )
            )
        )
        for _ in range(max(steps, 0)):
            mid = 0.5 * (a + b)
            up = score_at(mid) > 0
            a = torch.where(up, mid, a)
            b = torch.where(up, b, mid)
        lo = lo.clone()
        hi = hi.clone()
        lo[idx], hi[idx] = a, b
    # Posterior variance is largest at the upper bracket end (psi'(x) >= 1/x).
    var = 1.0 / (1.0 / (hi + 1.0) + (gain.detach() ** 2) / read_var.detach())
    post_radius = tail * torch.sqrt(var + 1.0) + tail**2
    mlo = torch.floor((lo - post_radius).clamp_min(0.0))
    mhi = torch.ceil(hi + post_radius)
    above = positive & (mlo > phi + 1.0)
    below = positive & (mhi < plo - 1.0)
    far = above | below
    lower = torch.where(positive, torch.where(far, mlo, torch.minimum(plo, mlo)), plo)
    upper = torch.where(positive, torch.where(far, mhi, torch.maximum(phi, mhi)), phi)
    if cdf:
        keep_prior = above
        lower = torch.where(keep_prior, plo, lower)
        upper = torch.where(keep_prior, phi, upper)
        far = below
    # The prior support must itself be representable, even when a far outlier
    # is evaluated on its own (smaller) window.
    if (
        not bool(torch.isfinite(torch.maximum(upper, phi)).all())
        or bool((torch.maximum(upper, phi) >= 2**63).any())
        or bool((torch.maximum(upper, phi) > cap).any())
    ):
        raise ValueError(
            "Poisson--Gaussian support exceeds the dtype's exact count range"
        )
    side = torch.where(far, torch.where(below, -1, 1), 0).to(torch.int8)
    return lower, upper, side


def _poisson_log_weights_mode_anchored(
    rate: Tensor, counts_i: Tensor, included: Tensor
) -> Tensor:
    """Normalised Poisson log weights as cumulative log-ratios from the mode.

    ``log p(n) - log p(m)`` is a sum of ``log1p`` ratios whose errors scale with
    the distance from the mode, not with ``lgamma(n)``; this keeps float32
    accurate at large rates. The window contains the prior mode. Values only
    (derivatives use the modal polynomial form).
    """
    counts = counts_i.to(rate.dtype)
    mode = rate.detach().floor().unsqueeze(-1)
    positive = rate.unsqueeze(-1) > 0
    lam = torch.where(positive, rate.unsqueeze(-1), 1.0)
    above = counts > mode
    below = counts < mode
    # log(x / y) via log1p when the ratio is near one (small cancellation
    # error), else a difference of logs; both are the same exact quantity.
    safe_c = counts.clamp_min(1.0)
    near = (lam - safe_c).abs() < 0.5 * safe_c
    right_term = torch.where(
        near,
        torch.log1p((lam - safe_c) / safe_c),
        lam.log() - safe_c.log(),
    )
    next_c = counts + 1.0
    near_l = (next_c - lam).abs() < 0.5 * lam
    left_term = torch.where(
        near_l,
        torch.log1p((next_c - lam) / lam),
        next_c.log() - lam.log(),
    )
    right = torch.where(above, right_term, 0.0).cumsum(-1)
    left = torch.where(below, left_term, 0.0).flip(-1).cumsum(-1).flip(-1)
    log_w = (right + left).masked_fill(~included, -torch.inf)
    # A zero-rate row is the point mass at count zero.
    point = torch.where(counts_i == 0, 0.0, -torch.inf)
    log_w = torch.where(positive, log_w, point.masked_fill(~included, -torch.inf))
    return log_w - torch.logsumexp(log_w, dim=-1, keepdim=True)


def _poisson_gaussian_components(
    value: Tensor,
    rate: Tensor,
    read_var: Tensor,
    gain_t: Tensor,
    background_t: Tensor,
    lower_float: Tensor,
    upper_float: Tensor,
    *,
    max_terms: int,
    logarithmic_weights: bool = True,
    normalize: bool = True,
    stable_values: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    width = int((upper_float - lower_float + 1).max().item()) if value.numel() else 0
    if width > int(max_terms):
        raise ValueError(
            f"Poisson--Gaussian support requires {width} terms, exceeding "
            f"max_terms={int(max_terms)}; use the calibrated Gaussian limit or "
            "a certified saddle-point implementation"
        )
    if width == 0:
        empty = torch.empty_like(value).unsqueeze(-1)
        return value, read_var, empty, empty, empty.to(torch.int64)

    lower, upper = lower_float.to(torch.int64), upper_float.to(torch.int64)

    offset = torch.arange(width, device=value.device, dtype=torch.int64)
    counts_i = lower.unsqueeze(-1) + offset
    included = counts_i <= upper.unsqueeze(-1)
    counts = counts_i.to(dtype=value.dtype)
    if stable_values:
        weights = _poisson_log_weights_mode_anchored(rate, counts_i, included)
    elif logarithmic_weights:
        safe_rate = torch.where(rate == 0, 1.0, rate).unsqueeze(-1)
        poisson_log = (
            counts * safe_rate.log() - rate.unsqueeze(-1) - torch.lgamma(counts + 1.0)
        )
        weights = torch.where(
            (rate == 0).unsqueeze(-1),
            torch.where(counts_i == 0, torch.zeros_like(poisson_log), -torch.inf),
            poisson_log,
        )
        weights = weights.masked_fill(~included, -torch.inf)
        if normalize:
            weights = weights - torch.logsumexp(weights, dim=-1, keepdim=True)
    else:
        weights = _normalized_poisson_probability_weights(rate, counts_i, included)

    conditional_mean = gain_t.unsqueeze(-1) * counts + background_t.unsqueeze(-1)
    return value, read_var, weights, conditional_mean, counts_i


def _standard_normal_interval_mass(lower: Tensor, upper: Tensor) -> Tensor:
    """Stable Phi(upper)-Phi(lower) for ordered standardized endpoints."""
    root_two = math.sqrt(2.0)
    lower_tail = 0.5 * (torch.erfc(-upper / root_two) - torch.erfc(-lower / root_two))
    upper_tail = 0.5 * (torch.erfc(lower / root_two) - torch.erfc(upper / root_two))
    across_zero = 0.5 * (torch.erf(upper / root_two) - torch.erf(lower / root_two))
    return torch.where(
        upper <= 0, lower_tail, torch.where(lower >= 0, upper_tail, across_zero)
    )


def _poisson_gaussian_zero_rate_jet(
    result: Tensor,
    value: Tensor,
    expected_electrons: Tensor,
    read_var: Tensor,
    *,
    gain: Tensor | float,
    background: Tensor | float,
    tail_sigma: float,
    cdf: bool,
) -> Tensor:
    """Restore exact right-hand first/second derivatives at zero Poisson rate.

    Log weights cannot represent the derivative of a zero probability. Their
    zero-rate branch already gives the exact value and read/location gradients;
    this adds its rate Taylor jet through second order. Higher rate derivatives
    at zero are not supplied. No positive rate uses a Taylor approximation.
    """
    rate = torch.broadcast_to(_coerce(expected_electrons, value), value.shape)
    zero = rate == 0
    forward_tangent = torch.autograd.forward_ad.unpack_dual(rate).tangent
    reverse_grad = torch.is_grad_enabled() and rate.requires_grad
    if not (reverse_grad or forward_tangent is not None) or not bool(zero.any()):
        return result
    gain_t = torch.broadcast_to(_coerce(gain, value), value.shape)
    background_t = torch.broadcast_to(_coerce(background, value), value.shape)
    # Mask before exponentiation: a positive-rate observation can have an
    # enormous f_1/f_0 ratio that is irrelevant to this boundary correction.
    residual = torch.where(zero, value - background_t, 0.0)
    boundary_gain = torch.where(zero, gain_t, 1.0)
    variance = torch.where(zero, read_var, 1.0)
    # At an integer cutoff, immediately positive rates add the next count.
    # In particular a zero radius of one includes count two in the right jet.
    # Use the same tensor operations/dtype as the represented radius. Even a
    # radius underflowing to zero adds count one immediately to the right.
    zero_radius = float(tail_sigma) * value.new_tensor(1.0) + float(tail_sigma) ** 2
    includes_two = bool(zero_radius >= 1)
    if cdf:
        sigma = variance.sqrt()
        z0 = residual / sigma
        z1 = (residual - boundary_gain) / sigma
        interval01 = _standard_normal_interval_mass(z1, z0)
        score = -interval01
        curvature = (
            interval01
            - _standard_normal_interval_mass((residual - 2 * boundary_gain) / sigma, z1)
            if includes_two
            else 2 * interval01
        )
    else:
        log_ratio = boundary_gain * (residual - 0.5 * boundary_gain) / variance
        score = -torch.expm1(log_ratio)
        if includes_two:
            # f_1^2/f_0^2 - f_2/f_0 in log form avoids cancellation and
            # avoids overflowing the individual squared ratio unnecessarily.
            squared_spacing = boundary_gain.square() / variance
            underflow = squared_spacing == 0
            safe_spacing = torch.where(underflow, 1.0, squared_spacing)
            log_factor = torch.where(
                underflow,
                2 * boundary_gain.log() - variance.log(),
                (-torch.expm1(-safe_spacing)).log(),
            )
            curvature = torch.exp(2 * log_ratio + log_factor)
        else:
            curvature = torch.expm1(2 * log_ratio)
    if not bool(torch.isfinite(score[zero]).all()) or not bool(
        torch.isfinite(curvature[zero]).all()
    ):
        raise ValueError(
            "zero-rate first/second derivatives exceed the dtype range; "
            "the zero-rate density/CDF value remains available without rate gradients"
        )
    boundary_rate = torch.where(zero, rate, 0.0)
    return result + boundary_rate * score + 0.5 * boundary_rate.square() * curvature


def _poisson_gaussian_modal_nll(
    observed: Tensor,
    rate: Tensor,
    read_var: Tensor,
    gain: Tensor,
    background: Tensor,
    counts_i: Tensor,
    poisson_log: Tensor,
    gaussian_log: Tensor,
) -> Tensor:
    """Ratio of modal polynomial sums with a frozen Gaussian reference."""
    included = torch.isfinite(poisson_log)
    counts = counts_i.to(observed.dtype)
    mode = rate.detach().floor().unsqueeze(-1)
    prior_terms = _poisson_mode_probability_terms(rate, counts_i, included)
    joint_index = (poisson_log + gaussian_log).detach().argmax(dim=-1, keepdim=True)
    joint_mode = counts.gather(-1, joint_index)
    degree = counts - joint_mode
    below = (degree < 0) & included
    above = (degree > 0) & included
    has_left = below.any(dim=-1, keepdim=True)
    has_right = above.any(dim=-1, keepdim=True)
    # Zero-rate rows retain the exact Gaussian value and the separate right jet.
    positive = (rate != 0).unsqueeze(-1)
    reference_gain = torch.where(positive, gain.detach().unsqueeze(-1), 1.0)
    reference_var = torch.where(positive, read_var.detach().unsqueeze(-1), 1.0)
    residual = torch.where(
        positive, (observed - background).detach().unsqueeze(-1), 0.0
    )
    reference_sigma = reference_var.sqrt()
    standardized_gain = reference_gain / reference_sigma
    standardized_residual = residual / reference_sigma
    spacing = standardized_gain.square()
    left_log_scale = joint_mode.clamp_min(1.0).log() + standardized_gain * (
        standardized_gain * (joint_mode - 0.5) - standardized_residual
    )
    right_log_scale = (joint_mode + 1.0).log() + standardized_gain * (
        standardized_gain * (joint_mode + 0.5) - standardized_residual
    )
    left_scale = torch.where(has_left, left_log_scale, 0.0).exp()
    right_inverse_scale = torch.where(has_right, -right_log_scale, 0.0).exp()
    bad_scaling = (
        (has_left & ((left_scale == 0) | ~torch.isfinite(left_scale)))
        | (
            has_right
            & ((right_inverse_scale == 0) | ~torch.isfinite(right_inverse_scale))
        )
    ).squeeze(-1)
    safe_rate = torch.where(rate == 0, 1.0, rate)
    # Degree multipliers can overflow before cancellation to a finite Hessian.
    left_order = torch.where(below, -degree + 1.0, 0.0).amax(dim=-1)
    right_order = torch.where(above, degree, 0.0).amax(dim=-1)
    bad_curvature_range = ~torch.isfinite((left_order / safe_rate.detach()).square())
    bad_curvature_range |= ~torch.isfinite(
        (right_order * right_inverse_scale.squeeze(-1)).square()
    )
    bad = bad_scaling | bad_curvature_range
    if bool(bad.any()):
        parameters = (observed, rate, read_var, gain, background)
        reverse_grad = torch.is_grad_enabled() and any(
            p.requires_grad for p in parameters
        )
        forward_grad = any(
            torch.autograd.forward_ad.unpack_dual(p).tangent is not None
            for p in parameters
        )
        if reverse_grad or forward_grad:
            raise ValueError(
                "Poisson--Gaussian modal derivative intermediates exceed the dtype "
                "range; evaluate without gradients or use a wider dtype"
            )
    # Mask ill-conditioned reference scalings before arithmetic; inference-only
    # rows fall back to the same retained law's logarithmic value evaluation.
    bad_row = bad.unsqueeze(-1)
    left_scale = torch.where(bad_row, 1.0, left_scale)
    right_inverse_scale = torch.where(bad_row, 1.0, right_inverse_scale)
    left_rate = torch.where(has_left & ~bad_row, safe_rate.unsqueeze(-1), 1.0)
    live_rate = torch.where(bad, 1.0, safe_rate).unsqueeze(-1)
    base = torch.where(below, left_scale / left_rate, live_rate * right_inverse_scale)
    left_coefficient = torch.cumprod(
        torch.where(
            below,
            (counts + 1.0)
            / joint_mode.clamp_min(1.0)
            * torch.exp(-(-degree - 1.0).clamp_min(0.0) * spacing),
            1.0,
        ).flip(-1),
        dim=-1,
    ).flip(-1)
    right_coefficient = torch.cumprod(
        torch.where(
            above,
            (joint_mode + 1.0)
            / counts.clamp_min(1.0)
            * torch.exp(-(degree - 1.0).clamp_min(0.0) * spacing),
            1.0,
        ),
        dim=-1,
    )
    coefficient = torch.where(below, left_coefficient, right_coefficient)
    finite_gaussian = torch.where(torch.isfinite(gaussian_log), gaussian_log, 0.0)
    conditional_change = (finite_gaussian - finite_gaussian.detach()).exp()
    joint_terms = coefficient * _integer_probability_powers(base, degree)
    joint_terms = (joint_terms * conditional_change).masked_fill(~included, 0.0)

    # Sum short adjacent factorial ratios, avoiding cancellation of large lgamma
    # values in the prior-to-joint anchor, including supports starting above zero.
    right_ratio = (mode + 1.0) / counts.clamp_min(1.0)
    left_ratio = (counts + 1.0) / mode.clamp_min(1.0)
    log_coefficient = torch.where(
        (counts > mode) & (counts <= joint_mode), right_ratio.log(), 0.0
    ).sum(dim=-1)
    log_coefficient = log_coefficient + torch.where(
        (counts >= joint_mode) & (counts < mode), left_ratio.log(), 0.0
    ).sum(dim=-1)
    anchor_degree = (joint_mode - mode).squeeze(-1)
    left_anchor_rate = torch.where(mode.squeeze(-1) > 0, safe_rate, 1.0)
    anchor_base = torch.where(
        anchor_degree < 0,
        mode.clamp_min(1.0).squeeze(-1) / left_anchor_rate,
        safe_rate / (mode.squeeze(-1) + 1.0),
    )
    anchor_base = torch.where((anchor_degree == 0) | bad, 1.0, anchor_base)
    log_prior_anchor = log_coefficient + anchor_degree.abs() * anchor_base.log()
    result = (
        prior_terms.sum(dim=-1).log()
        - joint_terms.sum(dim=-1).log()
        - log_prior_anchor
        - gaussian_log.detach().gather(-1, joint_index).squeeze(-1)
    )
    fallback = -torch.logsumexp((poisson_log + gaussian_log).detach(), dim=-1)
    return torch.where(bad, fallback, result)


def _needs_derivatives(*tensors: Tensor) -> bool:
    """True when reverse- or forward-mode derivatives of the inputs are live."""
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        return True
    return any(
        torch.autograd.forward_ad.unpack_dual(t).tangent is not None for t in tensors
    )


def _chunk_plan(
    rate: Tensor, tail_sigma: float, derivatives: bool, live: int | None = None
) -> tuple[int, int]:
    """Pixels per chunk (working-set budget) and total terms of the frame."""
    if rate.numel() == 0:
        return 1, 0
    radius = float(tail_sigma) * math.sqrt(float(rate.detach().max()) + 1.0)
    # Prior window width, with slack for merged observation windows.
    terms = 3.0 * (radius + float(tail_sigma) ** 2) + 2.0
    if live is None:
        live = _DERIVATIVE_TEMPORARIES if derivatives else _VALUE_TEMPORARIES
    per_term = live * rate.element_size() + 16
    chunk = max(1, int(_POISSON_GAUSSIAN_CHUNK_BYTES // (per_term * terms)))
    # Saved-graph estimate uses the prior window width (merged slack excluded).
    width = 2.0 * (radius + float(tail_sigma) ** 2) + 1.0
    return chunk, int(width * rate.numel())


def _run_chunks(
    fn, tensors: tuple[Tensor, ...], chunk: int, total_terms: int
) -> Tensor:
    from torch.utils.checkpoint import checkpoint

    count = tensors[0].numel()
    reverse = torch.is_grad_enabled() and any(t.requires_grad for t in tensors)
    dual = any(
        torch.autograd.forward_ad.unpack_dual(t).tangent is not None for t in tensors
    )
    saved = total_terms * tensors[0].element_size() * _DERIVATIVE_TEMPORARIES
    # Never under no_grad or forward mode; only when the graph would exceed the
    # budget and there is more than one chunk to recompute.
    checkpointed = (
        reverse and not dual and count > chunk and saved > _POISSON_GAUSSIAN_SAVE_BYTES
    )
    pieces = []
    for start in range(0, count, chunk):
        part = tuple(t[start : start + chunk] for t in tensors)
        if checkpointed:
            pieces.append(checkpoint(fn, *part, use_reentrant=False))
        else:
            pieces.append(fn(*part))
    return torch.cat(pieces) if pieces else tensors[0].new_empty(0)


def _scatter_rows(
    size: int, parts: list[tuple[Tensor, Tensor]], like: Tensor
) -> Tensor:
    out = like.new_zeros(size)
    for index, values in parts:
        out = out.index_put((index,), values)
    return out


def _poisson_gaussian_nll_near(
    obs, rate, var, gain, bg, lower, upper, *, max_terms, tail_sigma, derivatives
) -> Tensor:
    _, var, poisson_log, conditional_mean, counts = _poisson_gaussian_components(
        obs,
        rate,
        var,
        gain,
        bg,
        lower,
        upper,
        max_terms=max_terms,
        stable_values=not derivatives,
    )
    if poisson_log.shape[-1] == 0:
        return torch.empty_like(obs)
    gaussian_log = -0.5 * (
        var.unsqueeze(-1).log()
        + obs.new_tensor(math.log(2.0 * math.pi))
        + ((obs.unsqueeze(-1) - conditional_mean) / var.sqrt().unsqueeze(-1)).square()
    )
    if not derivatives:
        # Value only: the same normalised law as a plain log-sum-exp. The modal
        # polynomial form exists to keep derivatives well conditioned.
        return -torch.logsumexp(poisson_log + gaussian_log, dim=-1)
    return _poisson_gaussian_modal_nll(
        obs, rate, var, gain, bg, counts, poisson_log, gaussian_log
    )


def _poisson_gaussian_nll_far(
    obs, rate, var, gain, bg, lower, upper, *, max_terms
) -> Tensor:
    _, var, poisson_log, conditional_mean, _ = _poisson_gaussian_components(
        obs, rate, var, gain, bg, lower, upper, max_terms=max_terms, normalize=False
    )
    gaussian_log = -0.5 * (
        var.unsqueeze(-1).log()
        + obs.new_tensor(math.log(2.0 * math.pi))
        + ((obs.unsqueeze(-1) - conditional_mean) / var.sqrt().unsqueeze(-1)).square()
    )
    return -torch.logsumexp(poisson_log + gaussian_log, dim=-1)


def _streaming_gradients(flat: tuple[Tensor, ...]) -> bool:
    """Reverse-mode only: use the O(pixels)-memory analytic-gradient Function."""
    if not (torch.is_grad_enabled() and any(t.requires_grad for t in flat)):
        return False
    return not any(
        torch.autograd.forward_ad.unpack_dual(t).tangent is not None for t in flat
    )


def _zero_rate_score(
    obs: Tensor, var: Tensor, gain: Tensor, bg: Tensor, *, cdf: bool
) -> Tensor:
    """Exact right derivative with respect to the rate at zero rate (first order)."""
    residual = obs - bg
    if cdf:
        sigma = var.sqrt()
        return -_standard_normal_interval_mass(
            (residual - gain) / sigma, residual / sigma
        )
    return -torch.expm1(gain * (residual - 0.5 * gain) / var)


def _pg_rows_nll(obs, rt, var, g, bg, lower, upper, *, far, max_terms):
    """Per-pixel NLL and analytic gradients ``(y, rate, var, gain, background)``.

    With posterior ``q_n`` over the retained counts, ``r_n = y - g n - b``:
    ``dy = E[r]/s``, ``db = -dy``, ``dg = -E[n r]/s``,
    ``ds = 1/(2 s) - E[r^2]/(2 s^2)`` (``s`` the read variance) and, for the
    renormalised prior window, ``dλ = -(E_q[n] - E_prior[n])/λ`` (the unrenormalised
    far case is ``1 - E_q[n]/λ``). ``E[n r] = -g Var(n) + E[n] E[r]`` and
    ``E[r^2] = E[r]^2 + g^2 Var(n)`` keep everything cancellation free.
    """
    _, s, lw, mean, counts = _poisson_gaussian_components(
        obs,
        rt,
        var,
        g,
        bg,
        lower,
        upper,
        max_terms=max_terms,
        normalize=False,
        stable_values=not far,
    )
    gl = -0.5 * (
        s.unsqueeze(-1).log()
        + obs.new_tensor(math.log(2.0 * math.pi))
        + ((obs.unsqueeze(-1) - mean) / s.sqrt().unsqueeze(-1)).square()
    )
    joint = lw + gl
    lse = torch.logsumexp(joint, dim=-1)
    post = (joint - lse.unsqueeze(-1)).exp()
    c = (counts - lower.to(torch.int64).unsqueeze(-1)).to(obs.dtype)
    cm = (post * c).sum(-1)
    var_n = (post * (c - cm.unsqueeze(-1)).square()).sum(-1)
    nbar = lower + cm
    mr = ((obs - bg) - g * lower) - g * cm
    dy = mr / s
    d_gain = (g * var_n - nbar * mr) / s
    d_var = 0.5 / s - (mr.square() + g.square() * var_n) / (2.0 * s.square())
    positive = rt > 0
    safe = torch.where(positive, rt, 1.0)
    if far:
        d_rate = 1.0 - nbar / safe
    else:
        prior_cm = (lw.exp() * c).sum(-1)
        d_rate = -(cm - prior_cm) / safe
    return -lse, (dy, d_rate, d_var, d_gain, -dy)


def _pg_rows_cdf(obs, rt, var, g, bg, lower, upper, *, far, max_terms):
    """Per-pixel CDF and analytic gradients ``(y, rate, var, gain, background)``.

    Near rows: ``F = Σ w_n Φ(z_n)`` over the renormalised prior, so
    ``dλ = Cov_w(n, Φ)/λ`` (survival form in the upper half), ``dy = Σ w φ(z)/σ``,
    ``dg = -Σ w n φ/σ``, ``ds = -Σ w z φ/(2 s)``. Far-below rows use the same
    sums over unrenormalised Poisson weights in the log domain.
    """
    root_two = math.sqrt(2.0)
    _, s, lw, mean, counts = _poisson_gaussian_components(
        obs,
        rt,
        var,
        g,
        bg,
        lower,
        upper,
        max_terms=max_terms,
        normalize=False,
        stable_values=not far,
    )
    sigma = s.sqrt()
    z = (obs.unsqueeze(-1) - mean) / sigma.unsqueeze(-1)
    c = (counts - lower.to(torch.int64).unsqueeze(-1)).to(obs.dtype)
    positive = rt > 0
    safe = torch.where(positive, rt, 1.0)
    log_pdf = -0.5 * z.square() - 0.5 * math.log(2.0 * math.pi)
    if far:
        joint = lw + torch.special.log_ndtr(z)
        lse = torch.logsumexp(joint, dim=-1)
        cdf = lse.exp().clamp(0.0, 1.0)
        q = (joint - lse.unsqueeze(-1)).exp()
        nbar = lower + (q * c).sum(-1)
        d_rate = cdf * (nbar / safe - 1.0)
        dens_log = lw + log_pdf - sigma.log().unsqueeze(-1)
        dl = torch.logsumexp(dens_log, dim=-1)
        dens = dl.exp()
        q2 = (dens_log - dl.unsqueeze(-1)).exp()
        dy = dens
        d_gain = -dens * (lower + (q2 * c).sum(-1))
        d_var = -dens * sigma * (q2 * z).sum(-1) / (2.0 * s)
    else:
        w = lw.exp()
        big_phi = 0.5 * torch.erfc(-z / root_two)
        surv = 0.5 * torch.erfc(z / root_two)
        cdf_sum = (w * big_phi).sum(-1)
        survival_sum = (w * surv).sum(-1)
        low = cdf_sum <= 0.5
        cdf = torch.where(low, cdf_sum, 1.0 - survival_sum).clamp(0.0, 1.0)
        cdev = c - (w * c).sum(-1, keepdim=True)
        cov = torch.where(
            low,
            (w * cdev * big_phi).sum(-1),
            -(w * cdev * surv).sum(-1),
        )
        d_rate = cov / safe
        pdf = log_pdf.exp() / sigma.unsqueeze(-1)
        dy = (w * pdf).sum(-1)
        d_gain = -(lower * dy + (w * c * pdf).sum(-1))
        d_var = -sigma * (w * pdf * z).sum(-1) / (2.0 * s)
    return cdf, (dy, d_rate, d_var, d_gain, -dy)


def _poisson_gaussian_streaming(flat, *, cdf, tail_sigma, max_terms, need):
    """One chunked pass returning values and analytic gradients (O(pixels))."""
    obs, rate = flat[0], flat[1]
    count = obs.numel()
    out = obs.new_empty(count)
    grads = [obs.new_empty(count) if n else None for n in need]
    chunk, _ = _chunk_plan(rate, tail_sigma, False, live=_MOMENT_TEMPORARIES)
    rows = _pg_rows_cdf if cdf else _pg_rows_nll
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        part = tuple(t[start:stop] for t in flat)
        lower, upper, side = _poisson_gaussian_window(
            *part, tail_sigma=tail_sigma, cdf=cdf
        )
        far = side != 0
        size = stop - start
        c_out = obs.new_empty(size)
        c_grads = [obs.new_empty(size) for _ in range(5)]
        for mask, is_far in ((~far, False), (far, True)):
            idx = mask.nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            args = tuple(t[idx] for t in (*part, lower, upper))
            value, gs = rows(*args, far=is_far, max_terms=max_terms)
            c_out[idx] = value
            for dst, src in zip(c_grads, gs):
                dst[idx] = src
        zero = part[1] == 0
        if need[1] and bool(zero.any()):
            score = _zero_rate_score(part[0], part[2], part[3], part[4], cdf=cdf)
            if not bool(torch.isfinite(score[zero]).all()):
                raise ValueError(
                    "zero-rate first/second derivatives exceed the dtype range; "
                    "the zero-rate density/CDF value remains available without "
                    "rate gradients"
                )
            c_grads[1] = torch.where(zero, score, c_grads[1])
        out[start:stop] = c_out
        for dst, src in zip(grads, c_grads):
            if dst is not None:
                dst[start:stop] = src
    return out, grads


class _PoissonGaussianStreaming(torch.autograd.Function):
    """Poisson--Gaussian NLL/CDF with analytic, streaming first derivatives.

    The forward pass computes the per-pixel value and its gradients from
    posterior moments over the Poisson count in one chunked sweep, saving only
    O(pixels) tensors. The retained support is detached exactly as in the
    differentiable modal path. Double backward (``create_graph=True``) recomputes
    through that modal path, which is differentiable to any order but keeps the
    per-count graph (memory O(pixels x window)).
    """

    @staticmethod
    def forward(ctx, cdf, tail_sigma, max_terms, *flat):
        need = tuple(bool(n) for n in ctx.needs_input_grad[3:])
        out, grads = _poisson_gaussian_streaming(
            flat, cdf=cdf, tail_sigma=tail_sigma, max_terms=max_terms, need=need
        )
        ctx.cdf, ctx.tail_sigma, ctx.max_terms = cdf, tail_sigma, max_terms
        ctx.need = need
        ctx.n_inputs = len(flat)
        ctx.save_for_backward(*flat, *[g for g in grads if g is not None])
        return out

    @staticmethod
    def backward(ctx, grad_out):
        saved = ctx.saved_tensors
        flat, stored = saved[: ctx.n_inputs], saved[ctx.n_inputs :]
        if torch.is_grad_enabled():
            with torch.enable_grad():
                path = (
                    _poisson_gaussian_cdf_modal_path
                    if ctx.cdf
                    else _poisson_gaussian_nll_modal_path
                )
                out = path(tuple(flat), ctx.tail_sigma, ctx.max_terms)
                wrt = [t for t, n in zip(flat, ctx.need) if n]
                got = iter(
                    torch.autograd.grad(
                        out, wrt, grad_out, create_graph=True, allow_unused=True
                    )
                )
            result = [next(got) if n else None for n in ctx.need]
        else:
            it = iter(stored)
            result = [grad_out * next(it) if n else None for n in ctx.need]
        return (None, None, None, *result)


def poisson_gaussian_nll(
    observed: Tensor,
    expected_electrons: Tensor,
    read_noise_variance: Tensor,
    *,
    gain: Tensor | float = 1.0,
    background: Tensor | float = 0.0,
    tail_sigma: float = 12.0,
    max_terms: int = 4096,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> Tensor:
    """Normalized finite-mixture Poisson--Gaussian detector NLL.

    The latent electron count is Poisson and the measured value conditional on
    that count is Gaussian. A per-pixel Poisson support containing at least
    ``tail_sigma`` standard deviations is retained and explicitly renormalized,
    so the returned finite mixture is a normalized density. Support selection is
    detached from autograd. Positive-rate derivatives are local to the retained
    support; changing support can change the finite-mixture law discontinuously.
    At exactly zero rate, first/second rate and mixed derivatives are the exact
    right-hand continuation, including a cutoff that increases immediately to
    the right of zero. Third/higher rate derivatives at
    zero are unsupported. A requested zero-rate derivative outside the dtype
    range rejects, while inference-only zero-rate values remain available.
    Positive-rate autograd uses modal polynomial sums, with separate prior and
    observation-conditioned pivots, avoiding tiny-rate logarithmic curvature
    cancellation and density underflow. Derivatives exceeding the modal
    intermediate range reject; inference-only values use the same retained
    law's log-sum evaluation in that domain. This is a numeric range contract,
    not an intensity cutoff or change of physical model. It can conservatively
    reject a finite final derivative when reference scales underflow or larger
    intermediate curvature terms are not representable.

    The Poisson support is chosen per pixel from the expected rate *and* the
    observed value (see ``_poisson_gaussian_window``): the prior window is
    merged with a ``tail_sigma`` window around the mode of the
    observation-conditioned weight. A far outlier whose posterior window is
    disjoint from the prior one is scored with the exact unrenormalised Poisson
    log pmf over its own window. Pixels are processed in chunks of about
    ``2**22`` count terms (with activation checkpointing under reverse-mode
    autograd), so peak memory is independent of frame size.

    ``gain`` is output units per electron, ``background`` is a signed
    deterministic output-unit offset, and ``read_noise_variance`` is in output
    units squared. Expected electrons are a physical nonnegative integrated
    Poisson mean, including all independent photon/dark contributions. This
    normalized finite mixture is not the full untruncated physical law, nor a
    joint likelihood for correlated charge mixing or read noise.
    Observations require real float32/float64 tensors; integer and low-precision
    observations are unsupported. Other tensor parameters follow observation
    dtype/device by the existing explicit coercion convention.

    Args:
        observed: Measured values in output units (float32 or float64).
        expected_electrons: Non-negative Poisson mean of the electron count.
        read_noise_variance: Positive Gaussian read variance in output units
            squared.
        gain: Output units per electron.
        background: Signed deterministic offset in output units.
        tail_sigma: Retained Poisson and posterior support, in standard
            deviations.
        max_terms: Largest retained count window per pixel.
        reduction: ``"none"`` (per pixel), ``"mean"``, or ``"sum"``.

    Returns:
        Negative log-likelihood in nats, reduced as requested.

    Examples:
        >>> import torch
        >>> from torchsky.map.operators import poisson_gaussian_nll
        >>> observed = torch.tensor([0.4, 2.1, 5.3], dtype=torch.float64)
        >>> rate = torch.full((3,), 2.0, dtype=torch.float64)
        >>> nll = poisson_gaussian_nll(observed, rate, torch.tensor(0.25, dtype=torch.float64))
        >>> bool(torch.isfinite(nll))
        True
    """
    observed, rate, read_var, gain_t, background_t = _prepare_poisson_gaussian(
        observed,
        expected_electrons,
        read_noise_variance,
        gain,
        background,
        tail_sigma,
        max_terms,
    )
    shape = observed.shape
    flat = tuple(
        t.reshape(-1) for t in (observed, rate, read_var, gain_t, background_t)
    )
    if _streaming_gradients(flat):
        nll = _PoissonGaussianStreaming.apply(False, tail_sigma, max_terms, *flat)
        return _reduce_nll(nll.reshape(shape), reduction)
    nll = _poisson_gaussian_nll_modal_path(flat, tail_sigma, max_terms)
    return _reduce_nll(nll.reshape(shape), reduction)


def _poisson_gaussian_nll_modal_path(
    flat: tuple[Tensor, ...], tail_sigma: float, max_terms: int
) -> Tensor:
    """Differentiable (modal polynomial) chunked NLL; supports any-order AD."""
    derivatives = _needs_derivatives(*flat)
    rate = flat[1]

    def chunk_nll(obs, rt, var, g, bg):
        lower, upper, side = _poisson_gaussian_window(
            obs, rt, var, g, bg, tail_sigma=tail_sigma, cdf=False
        )
        far = side != 0
        if not bool(far.any()):
            nll = _poisson_gaussian_nll_near(
                obs,
                rt,
                var,
                g,
                bg,
                lower,
                upper,
                max_terms=max_terms,
                tail_sigma=tail_sigma,
                derivatives=derivatives,
            )
        else:
            parts = []
            for mask in (~far, far):
                idx = mask.nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                args = tuple(t[idx] for t in (obs, rt, var, g, bg, lower, upper))
                if mask is far:
                    values = _poisson_gaussian_nll_far(*args, max_terms=max_terms)
                else:
                    values = _poisson_gaussian_nll_near(
                        *args,
                        max_terms=max_terms,
                        tail_sigma=tail_sigma,
                        derivatives=derivatives,
                    )
                parts.append((idx, values))
            nll = _scatter_rows(obs.numel(), parts, obs)
        return _poisson_gaussian_zero_rate_jet(
            nll,
            obs,
            rt,
            var,
            gain=g,
            background=bg,
            tail_sigma=tail_sigma,
            cdf=False,
        )

    chunk, total = _chunk_plan(rate, tail_sigma, derivatives)
    return _run_chunks(chunk_nll, flat, chunk, total)


def poisson_gaussian_cdf(
    value: Tensor,
    expected_electrons: Tensor,
    read_noise_variance: Tensor,
    *,
    gain: Tensor | float = 1.0,
    background: Tensor | float = 0.0,
    tail_sigma: float = 12.0,
    max_terms: int = 4096,
) -> Tensor:
    """CDF of the normalized finite Poisson--Gaussian detector mixture.

    Units, detached-support locality and zero-rate derivative-order/range
    contracts are the same as :func:`poisson_gaussian_nll`.
    Gaussian CDF/survival sums use erfc; the lower half uses the CDF sum and
    the upper half one minus survival. This is numerical tail conditioning of
    the same retained law, with no intensity cutoff or Gaussian approximation.
    Mode-scaled probability powers avoid logarithmic-weight curvature
    cancellation at tiny positive rates. This does not change the NLL path.
    A pixel far below the prior window uses its own posterior window and a
    log-domain sum, so its tiny CDF is not truncated to the prior support;
    far above the prior window the prior support already gives CDF one.
    """
    value, rate, read_var, gain_t, background_t = _prepare_poisson_gaussian(
        value,
        expected_electrons,
        read_noise_variance,
        gain,
        background,
        tail_sigma,
        max_terms,
    )
    shape = value.shape
    flat = tuple(t.reshape(-1) for t in (value, rate, read_var, gain_t, background_t))
    if _streaming_gradients(flat):
        out = _PoissonGaussianStreaming.apply(True, tail_sigma, max_terms, *flat)
        return out.reshape(shape)
    return _poisson_gaussian_cdf_modal_path(flat, tail_sigma, max_terms).reshape(shape)


def _poisson_gaussian_cdf_modal_path(
    flat: tuple[Tensor, ...], tail_sigma: float, max_terms: int
) -> Tensor:
    """Differentiable chunked CDF; supports any-order AD."""
    root_two = math.sqrt(2.0)
    derivatives = _needs_derivatives(*flat)
    rate = flat[1]

    def near(v, rt, var, g, bg, lower, upper):
        _, var, weights, mean, _c = _poisson_gaussian_components(
            v,
            rt,
            var,
            g,
            bg,
            lower,
            upper,
            max_terms=max_terms,
            logarithmic_weights=not derivatives,
            stable_values=not derivatives,
        )
        if weights.shape[-1] == 0:
            return torch.empty_like(v)
        if not derivatives:
            # Value only: normalised log weights, exponentiated (the same law
            # as the mode-scaled polynomial weights used for derivatives).
            weights = weights.exp()
        standardized = (v.unsqueeze(-1) - mean) / var.sqrt().unsqueeze(-1)
        # erfc retains representable lower-tail mass where ndtr/erf rounds to zero.
        conditional_cdf = 0.5 * torch.erfc(-standardized / root_two)
        conditional_survival = 0.5 * torch.erfc(standardized / root_two)
        cdf_sum = torch.sum(weights * conditional_cdf, dim=-1)
        survival_sum = torch.sum(weights * conditional_survival, dim=-1)
        # Same normalized law; the probability-half crossover only selects the
        # better-conditioned tail calculation, never a physical model.
        result = torch.where(cdf_sum <= 0.5, cdf_sum, 1 - survival_sum)
        # Clamp only out-of-range rounding: clamp's gradient is zero at the
        # bound, and the upper tail rounds to exactly 1 with nonzero derivative.
        in_range = (result >= 0.0) & (result <= 1.0)
        return torch.where(in_range, result, result.clamp(0.0, 1.0))

    def far_low(v, rt, var, g, bg, lower, upper):
        _, var, log_p, mean, _c = _poisson_gaussian_components(
            v, rt, var, g, bg, lower, upper, max_terms=max_terms, normalize=False
        )
        z = (v.unsqueeze(-1) - mean) / var.sqrt().unsqueeze(-1)
        return torch.exp(
            torch.logsumexp(log_p + torch.special.log_ndtr(z), dim=-1)
        ).clamp(0.0, 1.0)

    def chunk_cdf(v, rt, var, g, bg):
        lower, upper, side = _poisson_gaussian_window(
            v, rt, var, g, bg, tail_sigma=tail_sigma, cdf=True
        )
        low = side != 0
        if not bool(low.any()):
            result = near(v, rt, var, g, bg, lower, upper)
        else:
            parts = []
            for mask in (~low, low):
                idx = mask.nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                args = tuple(t[idx] for t in (v, rt, var, g, bg, lower, upper))
                parts.append((idx, (far_low if mask is low else near)(*args)))
            result = _scatter_rows(v.numel(), parts, v)
        return _poisson_gaussian_zero_rate_jet(
            result,
            v,
            rt,
            var,
            gain=g,
            background=bg,
            tail_sigma=tail_sigma,
            cdf=True,
        )

    chunk, total = _chunk_plan(rate, tail_sigma, derivatives)
    return _run_chunks(chunk_cdf, flat, chunk, total)


def gaussian_student_t_mixture_responsibilities(
    residual: Tensor,
    variance: Tensor,
    *,
    outlier_scale: Tensor | float = 10.0,
    degrees_of_freedom: Tensor | float = 3.0,
    outlier_probability: Tensor | float = 0.01,
    prior: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return posterior ``(inlier_probability, outlier_probability)`` maps."""
    log_inlier, log_outlier, log_mixture = _mixture_terms(
        residual,
        variance,
        outlier_scale=outlier_scale,
        degrees_of_freedom=degrees_of_freedom,
        outlier_probability=outlier_probability,
        prior=prior,
    )
    inlier = torch.exp(log_inlier - log_mixture)
    outlier = torch.exp(log_outlier - log_mixture)
    return inlier, outlier


mixture_nll = gaussian_student_t_mixture_nll
mixture_responsibilities = gaussian_student_t_mixture_responsibilities


def profile_amplifier_row_offsets(
    residual: Tensor,
    variance: Tensor,
    amplifier_index: Tensor,
    *,
    valid: Tensor | None = None,
    num_amplifiers: int = 4,
) -> tuple[Tensor, Tensor]:
    """Variable-project signed row offsets for each detector amplifier."""
    if residual.ndim != 2 or variance.shape != residual.shape:
        raise ValueError("residual and variance must be matching 2D tensors")
    if amplifier_index.shape != residual.shape:
        raise ValueError("amplifier_index must match the residual shape")
    support = torch.isfinite(residual) & torch.isfinite(variance) & (variance > 0)
    if valid is not None:
        if valid.shape != residual.shape:
            raise ValueError("valid must match the residual shape")
        support &= valid.to(dtype=torch.bool)
    offsets = torch.zeros(
        (int(num_amplifiers), residual.shape[0]),
        device=residual.device,
        dtype=residual.dtype,
    )
    offset_map = torch.zeros_like(residual)
    for amplifier in range(int(num_amplifiers)):
        selected = support & (amplifier_index == amplifier)
        inverse_variance = torch.where(
            selected, variance.reciprocal(), torch.zeros_like(variance)
        )
        denominator = inverse_variance.sum(dim=-1)
        numerator = (inverse_variance * torch.where(selected, residual, 0.0)).sum(
            dim=-1
        )
        row_offset = torch.where(
            denominator > 0.0,
            numerator / denominator.clamp_min(torch.finfo(residual.dtype).tiny),
            torch.zeros_like(numerator),
        )
        offsets[amplifier] = row_offset
        offset_map = torch.where(
            amplifier_index == amplifier, row_offset[:, None], offset_map
        )
    return offsets, residual - offset_map


class StructuredDetectorNoise(nn.Module):
    """Bounded detector likelihood with pipeline variance, 1/f rows, and outliers."""

    def __init__(
        self,
        *,
        num_amplifiers: int = 4,
        variance_calibration: PipelineVarianceCalibration | None = None,
        row_noise: OneOverFRowNoise | None = None,
        outlier_probability: float = 0.01,
        outlier_scale: float = 10.0,
        degrees_of_freedom: float = 3.0,
    ) -> None:
        super().__init__()
        if int(num_amplifiers) < 1:
            raise ValueError("num_amplifiers must be positive")
        if not 0.0 < float(outlier_probability) < 1.0:
            raise ValueError("outlier_probability must lie in (0, 1)")
        if float(outlier_scale) <= 1.0 or float(degrees_of_freedom) <= 0.0:
            raise ValueError("outlier_scale must exceed one and dof must be positive")
        self.num_amplifiers = int(num_amplifiers)
        self.variance_calibration = (
            PipelineVarianceCalibration()
            if variance_calibration is None
            else variance_calibration
        )
        self.row_noise = (
            OneOverFRowNoise((self.num_amplifiers,)) if row_noise is None else row_noise
        )
        self.outlier_logit = nn.Parameter(
            torch.tensor(
                math.log(
                    float(outlier_probability) / (1.0 - float(outlier_probability))
                )
            )
        )
        self.raw_outlier_scale = nn.Parameter(
            _inverse_softplus(torch.tensor(float(outlier_scale) - 1.0))
        )
        self.degrees_of_freedom = float(degrees_of_freedom)

    @property
    def outlier_probability(self) -> Tensor:
        return torch.sigmoid(self.outlier_logit)

    @property
    def outlier_scale(self) -> Tensor:
        return 1.0 + F.softplus(self.raw_outlier_scale)

    def default_amplifier_index(
        self, shape: tuple[int, int], device: torch.device
    ) -> Tensor:
        height, width = map(int, shape)
        columns = torch.arange(width, device=device)
        amplifier = torch.clamp(
            columns * self.num_amplifiers // max(width, 1),
            max=self.num_amplifiers - 1,
        )
        return amplifier[None, :].expand(height, width)

    def _row_nll(self, row_offsets: Tensor) -> Tensor:
        spectrum = torch.fft.rfft(row_offsets, dim=-1, norm="ortho")
        psd = self.row_noise.psd(int(row_offsets.shape[-1])).to(
            device=row_offsets.device, dtype=row_offsets.dtype
        )
        multiplicity = torch.full_like(psd, 2.0)
        multiplicity[..., 0] = 1.0
        if int(row_offsets.shape[-1]) % 2 == 0:
            multiplicity[..., -1] = 1.0
        power = spectrum.real.square() + spectrum.imag.square()
        return (
            0.5
            * torch.sum(multiplicity * (power / psd + psd.log()))
            / multiplicity.sum().clamp_min(1.0)
        )

    def _variance_and_support(
        self,
        residual: Tensor,
        *,
        components: Mapping[str, Tensor] | None,
        err: Tensor | None,
        valid: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if residual.ndim != 2 or not residual.is_floating_point():
            raise ValueError(
                "structured detector residual must be a floating 2D tensor"
            )
        support = torch.ones_like(residual, dtype=torch.bool)
        if valid is not None:
            valid_t = valid.to(device=residual.device)
            if valid_t.shape != residual.shape:
                raise ValueError("valid must match the residual shape")
            support &= valid_t.to(dtype=torch.bool)

        safe_components: dict[str, Tensor] | None = None
        safe_err: Tensor | None = None
        if components:
            safe_components = {}
            for name, value in components.items():
                component = torch.broadcast_to(
                    value.to(device=residual.device, dtype=residual.dtype),
                    residual.shape,
                )
                finite_nonnegative = torch.isfinite(component) & (component >= 0.0)
                support &= finite_nonnegative
                safe_components[name] = torch.where(
                    finite_nonnegative,
                    component,
                    torch.zeros_like(component),
                )
        elif err is not None:
            err_t = torch.broadcast_to(
                err.to(device=residual.device, dtype=residual.dtype),
                residual.shape,
            )
            finite_nonnegative = torch.isfinite(err_t) & (err_t >= 0.0)
            support &= finite_nonnegative
            safe_err = torch.where(
                finite_nonnegative,
                err_t,
                torch.ones_like(err_t),
            )

        variance = self.variance_calibration(
            safe_components,
            err=safe_err,
        ).to(device=residual.device, dtype=residual.dtype)
        if variance.shape != residual.shape:
            variance = torch.broadcast_to(variance, residual.shape)
        support &= torch.isfinite(residual) & torch.isfinite(variance) & (variance > 0)
        return variance, support

    def negative_log_likelihood(
        self,
        residual: Tensor,
        *,
        components: Mapping[str, Tensor] | None = None,
        err: Tensor | None = None,
        valid: Tensor | None = None,
        amplifier_index: Tensor | None = None,
        contamination_prior: Tensor | None = None,
    ) -> Tensor:
        variance, support = self._variance_and_support(
            residual,
            components=components,
            err=err,
            valid=valid,
        )
        if not bool(support.any()):
            raise ValueError("structured likelihood requires at least one valid pixel")
        safe_residual = torch.where(support, residual, torch.zeros_like(residual))
        safe_variance = torch.where(support, variance, torch.ones_like(variance))
        amp = (
            self.default_amplifier_index(
                image_shape_hw(residual.shape), residual.device
            )
            if amplifier_index is None
            else amplifier_index.to(device=residual.device)
        )
        row_offsets, corrected = profile_amplifier_row_offsets(
            safe_residual,
            safe_variance,
            amp,
            valid=support,
            num_amplifiers=self.num_amplifiers,
        )
        prior = None
        if contamination_prior is not None:
            prior = torch.broadcast_to(
                contamination_prior.to(
                    device=residual.device,
                    dtype=residual.dtype,
                ),
                residual.shape,
            )
            if not bool(torch.isfinite(prior[support]).all()) or bool(
                ((prior[support] < 0.0) | (prior[support] > 1.0)).any()
            ):
                raise ValueError(
                    "usable contamination_prior values must be finite in [0, 1]"
                )
            prior = torch.where(support, prior, torch.zeros_like(prior))
        pixel_nll = gaussian_student_t_mixture_nll(
            corrected,
            safe_variance,
            outlier_scale=self.outlier_scale,
            degrees_of_freedom=self.degrees_of_freedom,
            outlier_probability=self.outlier_probability,
            prior=prior,
            reduction="none",
        )
        return reduce_detector_nll(
            pixel_nll,
            mask=support,
            reduction="mean",
        ) + self._row_nll(row_offsets)

    def contamination_responsibilities(
        self,
        residual: Tensor,
        *,
        components: Mapping[str, Tensor] | None = None,
        err: Tensor | None = None,
        valid: Tensor | None = None,
        amplifier_index: Tensor | None = None,
        contamination_prior: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        variance, support = self._variance_and_support(
            residual,
            components=components,
            err=err,
            valid=valid,
        )
        if not bool(support.any()):
            raise ValueError("contamination responsibilities require a valid pixel")
        safe_residual = torch.where(support, residual, torch.zeros_like(residual))
        safe_variance = torch.where(support, variance, torch.ones_like(variance))
        amp = (
            self.default_amplifier_index(
                image_shape_hw(residual.shape), residual.device
            )
            if amplifier_index is None
            else amplifier_index.to(device=residual.device)
        )
        _, corrected = profile_amplifier_row_offsets(
            safe_residual,
            safe_variance,
            amp,
            valid=support,
            num_amplifiers=self.num_amplifiers,
        )
        prior = None
        if contamination_prior is not None:
            prior = torch.broadcast_to(
                contamination_prior.to(
                    device=residual.device,
                    dtype=residual.dtype,
                ),
                residual.shape,
            )
            if not bool(torch.isfinite(prior[support]).all()) or bool(
                ((prior[support] < 0.0) | (prior[support] > 1.0)).any()
            ):
                raise ValueError(
                    "usable contamination_prior values must be finite in [0, 1]"
                )
            prior = torch.where(support, prior, torch.zeros_like(prior))
        inlier, outlier = gaussian_student_t_mixture_responsibilities(
            corrected,
            safe_variance,
            outlier_scale=self.outlier_scale,
            degrees_of_freedom=self.degrees_of_freedom,
            outlier_probability=self.outlier_probability,
            prior=prior,
        )
        zeros = torch.zeros_like(inlier)
        return torch.where(support, inlier, zeros), torch.where(support, outlier, zeros)


__all__ = [
    "ExplicitNoiseOp",
    "LearnedNoiseOp",
    "OneOverFRowNoise",
    "PipelineVarianceCalibration",
    "StructuredDetectorNoise",
    "StructuredNoiseOp",
    "VAR_FLAT",
    "VAR_POISSON",
    "VAR_RNOISE",
    "calibrated_pipeline_variance",
    "fft_whiten_rows",
    "gaussian_nll",
    "gaussian_student_t_mixture_nll",
    "gaussian_student_t_mixture_responsibilities",
    "mixture_nll",
    "mixture_responsibilities",
    "one_over_f_psd",
    "one_over_f_row_psd",
    "pipeline_variance",
    "poisson_gaussian_cdf",
    "poisson_gaussian_nll",
    "profile_amplifier_row_offsets",
    "reduce_detector_nll",
    "sanitize_inverse_variance",
    "student_t_nll",
    "whiten_rows_fft",
]
