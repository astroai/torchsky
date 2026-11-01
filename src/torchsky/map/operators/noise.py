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
    """Normalized Gaussian detector negative log likelihood."""
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


def _poisson_gaussian_components(
    value: Tensor,
    expected_electrons: Tensor,
    read_noise_variance: Tensor,
    *,
    gain: Tensor | float = 1.0,
    background: Tensor | float = 0.0,
    tail_sigma: float = 12.0,
    max_terms: int = 4096,
    logarithmic_weights: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
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
    value, rate, read_var = torch.broadcast_tensors(value, rate, read_var)
    gain_t = torch.broadcast_to(
        torch.as_tensor(gain, device=value.device, dtype=value.dtype),
        value.shape,
    )
    background_t = torch.broadcast_to(
        torch.as_tensor(background, device=value.device, dtype=value.dtype),
        value.shape,
    )
    _validate_finite(value, name="value")
    _validate_nonnegative(rate, name="expected_electrons")
    _validate_positive(read_var, name="read_noise_variance")
    _validate_positive(gain_t, name="gain")
    _validate_finite(background_t, name="background")

    # ponytail: bounded finite mixture; replace with a saddle-point tail
    # approximation if certified high-count exposures exceed max_terms.
    detached_rate = rate.detach()
    radius = (
        float(tail_sigma) * torch.sqrt(detached_rate + 1.0) + float(tail_sigma) ** 2
    )
    lower_float = torch.floor((detached_rate - radius).clamp_min(0.0))
    upper_float = torch.ceil(detached_rate + radius)
    if (
        not bool(torch.isfinite(upper_float).all())
        or bool((upper_float >= 2**63).any())
        or bool((upper_float > 2.0 / torch.finfo(value.dtype).eps).any())
    ):
        raise ValueError(
            "Poisson--Gaussian support exceeds the dtype's exact count range"
        )
    width = (
        int((upper_float - lower_float + 1).max().item()) if upper_float.numel() else 0
    )
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
    if logarithmic_weights:
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

    ``gain`` is output units per electron, ``background`` is a signed
    deterministic output-unit offset, and ``read_noise_variance`` is in output
    units squared. Expected electrons are a physical nonnegative integrated
    Poisson mean, including all independent photon/dark contributions. This
    normalized finite mixture is not the full untruncated physical law, nor a
    joint likelihood for correlated charge mixing or read noise.
    Observations require real float32/float64 tensors; integer and low-precision
    observations are unsupported. Other tensor parameters follow observation
    dtype/device by the existing explicit coercion convention.
    """
    observed, read_var, poisson_log, conditional_mean, counts = (
        _poisson_gaussian_components(
            observed,
            expected_electrons,
            read_noise_variance,
            gain=gain,
            background=background,
            tail_sigma=tail_sigma,
            max_terms=max_terms,
        )
    )
    if poisson_log.shape[-1] == 0:
        return _reduce_nll(torch.empty_like(observed), reduction)
    gaussian_log = -0.5 * (
        read_var.unsqueeze(-1).log()
        + observed.new_tensor(math.log(2.0 * math.pi))
        + (
            (observed.unsqueeze(-1) - conditional_mean) / read_var.sqrt().unsqueeze(-1)
        ).square()
    )
    rate = torch.broadcast_to(_coerce(expected_electrons, observed), observed.shape)
    gain_t = torch.broadcast_to(_coerce(gain, observed), observed.shape)
    background_t = torch.broadcast_to(_coerce(background, observed), observed.shape)
    nll = _poisson_gaussian_modal_nll(
        observed,
        rate,
        read_var,
        gain_t,
        background_t,
        counts,
        poisson_log,
        gaussian_log,
    )
    nll = _poisson_gaussian_zero_rate_jet(
        nll,
        observed,
        expected_electrons,
        read_var,
        gain=gain,
        background=background,
        tail_sigma=tail_sigma,
        cdf=False,
    )
    return _reduce_nll(nll, reduction)


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
    """
    value, read_var, weights, conditional_mean, _counts = _poisson_gaussian_components(
        value,
        expected_electrons,
        read_noise_variance,
        gain=gain,
        background=background,
        tail_sigma=tail_sigma,
        max_terms=max_terms,
        logarithmic_weights=False,
    )
    if weights.shape[-1] == 0:
        return torch.empty_like(value)
    standardized = (value.unsqueeze(-1) - conditional_mean) / read_var.sqrt().unsqueeze(
        -1
    )
    # erfc retains representable lower-tail mass where ndtr/erf rounds to zero.
    conditional_cdf = 0.5 * torch.erfc(-standardized / math.sqrt(2.0))
    conditional_survival = 0.5 * torch.erfc(standardized / math.sqrt(2.0))
    cdf_sum = torch.sum(weights * conditional_cdf, dim=-1)
    survival_sum = torch.sum(weights * conditional_survival, dim=-1)
    # These are the same normalized law. The probability-half crossover only
    # selects the better-conditioned tail calculation, never a physical model.
    result = torch.where(cdf_sum <= 0.5, cdf_sum, 1 - survival_sum)
    # Clamp only out-of-range rounding: clamp's gradient is zero at the bound,
    # and the upper tail rounds to exactly 1 while its derivative is nonzero.
    in_range = (result >= 0.0) & (result <= 1.0)
    result = torch.where(in_range, result, result.clamp(0.0, 1.0))
    return _poisson_gaussian_zero_rate_jet(
        result,
        value,
        expected_electrons,
        read_var,
        gain=gain,
        background=background,
        tail_sigma=tail_sigma,
        cdf=True,
    )


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
            self.default_amplifier_index(tuple(residual.shape), residual.device)
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
            self.default_amplifier_index(tuple(residual.shape), residual.device)
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
