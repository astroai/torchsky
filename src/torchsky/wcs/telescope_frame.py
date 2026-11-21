"""Detector-to-sky astrometry for exposures that carry no distortion in the header.

Count-rate products from space telescopes (for example JWST ``_rate`` files)
store only a plain tangent-plane WCS plus pointing keywords. The detector
distortion is added later from reference files. This module provides the
standard "SIAF" construction as differentiable tensor operations::

    science pixel (x, y)
      --polynomial "science to ideal"-->  ideal frame (arcsec, tangent plane
                                           at the aperture reference point)
      --rotate by V3IdlYAngle, parity, shift by (V2Ref, V3Ref)-->
                                          telescope frame (V2, V3) (arcsec)
      --attitude (V2_REF, V3_REF, RA_REF, DEC_REF, ROLL_REF)-->
                                          sky (RA, Dec) (degrees)

Conventions
-----------
* Detector pixels are zero-based, pixel centres at integers, as everywhere in
  torchsky. SIAF science pixels are one-based; ``x_sci = x + 1``.
* The ideal-to-telescope step is the planar SIAF convention (a rotation and a
  shift of angular coordinates in arcsec), which is what the JWST pipeline and
  ``pysiaf.Aperture.idl_to_tel`` (default method) implement. The telescope to
  sky step is an exact 3-D rotation of unit vectors in float64.
* All geometry is evaluated in float64 and cast back to the dtype of the
  input coordinates. The modules keep their parameters in float64 when moved
  between devices or cast with ``.to(dtype)``.
* No calibration data ship with torchsky: coefficients are plain tensors.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor, nn

from ..device import _as_tensor, _to
from .detector_astrometry import AffineFreeResidualGrid2D

_ARCSEC_PER_DEG = 3600.0
_HEADER_KEYWORDS = (
    "V2_REF",
    "V3_REF",
    "ROLL_REF",
    "RA_REF",
    "DEC_REF",
    "VPARITY",
    "V3I_YANG",
)
_F64 = torch.float64


def _finite(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all().detach().cpu().item()):
        raise ValueError(f"{name} must contain only finite values")


def _coords(points: Any, name: str) -> Tensor:
    pts = torch.as_tensor(points)
    if not pts.is_floating_point() or pts.is_complex():
        raise ValueError(f"{name} must be a real floating-point tensor")
    if pts.shape[-1:] != (2,):
        raise ValueError(f"{name} last dimension must be 2")
    _finite(pts, name)
    return pts


def _scalar(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


class _DeviceOnlyApply(nn.Module):
    """Mixin: ``.to(dtype)`` and ``.float()`` keep float64 parameters.

    ``FieldObservation.to(dtype=...)`` casts the whole detector transform.
    Astrometry in float32 would be limited to tens of milliarcseconds, so only
    the device part of a move is honoured.
    """

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> Any:
        def device_only(tensor: Tensor) -> Tensor:
            probe = fn(tensor.new_empty(0))
            return tensor.to(device=probe.device)

        return super()._apply(device_only, recurse)


def _rotation(axis: int, angle: Tensor) -> Tensor:
    """Right-handed rotation about ``axis`` (0, 1, 2), angle in radians."""
    c, s = torch.cos(angle), torch.sin(angle)
    one, zero = torch.ones_like(c), torch.zeros_like(c)
    if axis == 0:
        rows = ((one, zero, zero), (zero, c, -s), (zero, s, c))
    elif axis == 1:
        rows = ((c, zero, s), (zero, one, zero), (-s, zero, c))
    else:
        rows = ((c, -s, zero), (s, c, zero), (zero, zero, one))
    return torch.stack([torch.stack(row) for row in rows])


def _unit_vector(lon_rad: Tensor, lat_rad: Tensor) -> Tensor:
    cos_lat = torch.cos(lat_rad)
    return torch.stack(
        (
            cos_lat * torch.cos(lon_rad),
            cos_lat * torch.sin(lon_rad),
            torch.sin(lat_rad),
        ),
        dim=-1,
    )


def _lon_lat(vector: Tensor) -> tuple[Tensor, Tensor]:
    lon = torch.atan2(vector[..., 1], vector[..., 0])
    lat = torch.atan2(vector[..., 2], torch.hypot(vector[..., 0], vector[..., 1]))
    return lon, lat


def _triangular_to_flat(array: Any, name: str) -> tuple[Tensor, int]:
    """Return SIAF-ordered flat coefficients (float64) and the degree."""
    # Python lists and NumPy arrays become float64 (never torch's float32 default).
    values = array if isinstance(array, Tensor) else torch.as_tensor(array, dtype=_F64)
    if not values.is_floating_point() or values.is_complex():
        raise ValueError(f"{name} must be a real floating-point array")
    values = values.detach().to(_F64)
    _finite(values, name)
    if values.ndim == 2:
        rows, cols = values.shape
        if rows != cols or rows < 2:
            raise ValueError(
                f"{name} must be a square triangular array of shape "
                f"(degree + 1, degree + 1) with degree >= 1, got {tuple(values.shape)}"
            )
        degree = rows - 1
        upper = torch.triu(values, diagonal=1)
        if bool((upper != 0).any()):
            raise ValueError(f"{name} must be zero above the diagonal (j > i)")
        flat = torch.cat([values[i, : i + 1] for i in range(rows)])
        return flat, degree
    if values.ndim == 1:
        n = values.numel()
        degree = int((math.isqrt(8 * n + 1) - 3) // 2)
        if (degree + 1) * (degree + 2) // 2 != n or degree < 1:
            raise ValueError(
                f"{name} flat length {n} is not a triangular number "
                "(degree + 1) * (degree + 2) / 2 with degree >= 1"
            )
        return values, degree
    raise ValueError(f"{name} must be a triangular 2-D array or a flat 1-D array")


def _powers(t: Tensor, degree: int) -> Tensor:
    out = [torch.ones_like(t)]
    for _ in range(degree):
        out.append(out[-1] * t)
    return torch.stack(out, dim=-1)


class PolynomialDetectorDistortion(_DeviceOnlyApply):
    """Two-dimensional polynomial from detector pixels to ideal-frame arcsec.

    For ``u = x - x_ref`` and ``v = y - y_ref`` (zero-based detector pixels
    minus the zero-based reference pixel) the ideal coordinates are::

        X(x, y) = sum_{i=0..d} sum_{j=0..i} a[i, j] * u**(i - j) * v**j
        Y(x, y) = sum_{i=0..d} sum_{j=0..i} b[i, j] * u**(i - j) * v**j

    This is the SIAF ``Sci2IdlX{i}{j}`` / ``Sci2IdlY{i}{j}`` layout: the first
    index ``i`` is the total order, the second ``j`` is the power of ``v``
    (the y direction), so ``a[1, 0]`` multiplies ``u`` and ``a[1, 1]``
    multiplies ``v``. Triangular arrays have shape ``(d + 1, d + 1)`` and must
    be zero for ``j > i``. A flat 1-D array in SIAF order ``a00, a10, a11, a20,
    a21, a22, ...`` of length ``(d + 1)(d + 2) / 2`` is also accepted.

    SIAF science pixels are one-based (the first pixel centre is 1), and the
    SIAF reference pixel ``XSciRef``/``YSciRef`` is in that convention. The
    constructor takes the zero-based reference pixel ``ref_pixel = (XSciRef -
    1, YSciRef - 1)``; :meth:`from_siaf` does the subtraction.

    Coefficients are stored as one float64 tensor of shape ``(2, n)`` with
    ``n = (d + 1)(d + 2) / 2``; it is an ``nn.Parameter`` that requires
    gradients only when ``learnable=True``. The Jacobian determinant must be
    strictly positive over a validation lattice covering the detector
    (:meth:`assert_invertible`); reflected polynomials are rejected, the
    reflection belongs in the parity of the ideal-to-telescope step.

    Args:
        coeffs_x: Triangular (or flat SIAF-ordered) coefficients of ``X``,
            arcsec per pixel**(i).
        coeffs_y: Same for ``Y``.
        ref_pixel: Zero-based reference pixel ``(x_ref, y_ref)``.
        detector_shape: ``(height, width)`` of the detector in pixels. Defines
            the validation lattice ``[0, width - 1] x [0, height - 1]``.
        learnable: Make the coefficients trainable.
        validation_nodes: Nodes per axis of the validation lattice (>= 2).

    Raises:
        ValueError: On malformed or non-finite coefficients, a bad shape, or a
            folded or singular polynomial over the detector.

    Examples:
        >>> import torch
        >>> a = torch.tensor([[0.0, 0.0], [0.03, 0.0]], dtype=torch.float64)
        >>> b = torch.tensor([[0.0, 0.0], [0.0, 0.03]], dtype=torch.float64)
        >>> poly = PolynomialDetectorDistortion(
        ...     a, b, ref_pixel=(1.5, 1.5), detector_shape=(4, 4)
        ... )
        >>> poly(torch.tensor([[1.5, 1.5], [2.5, 1.5]], dtype=torch.float64)).tolist()
        [[0.0, 0.0], [0.03, 0.0]]
    """

    coefficients: nn.Parameter
    ref_pixel: Tensor
    exponent_u: Tensor
    exponent_v: Tensor

    def __init__(
        self,
        coeffs_x: Any,
        coeffs_y: Any,
        *,
        ref_pixel: Any,
        detector_shape: tuple[int, int],
        learnable: bool = False,
        validation_nodes: int = 33,
    ) -> None:
        super().__init__()
        flat_x, degree_x = _triangular_to_flat(coeffs_x, "coeffs_x")
        flat_y, degree_y = _triangular_to_flat(coeffs_y, "coeffs_y")
        if degree_x != degree_y:
            raise ValueError(
                f"coeffs_x (degree {degree_x}) and coeffs_y (degree {degree_y}) "
                "must have the same degree"
            )
        ref = torch.as_tensor(ref_pixel)
        if ref.shape != (2,) or ref.is_complex():
            raise ValueError("ref_pixel must be a real array of shape (2,)")
        ref = ref.detach().to(_F64)
        _finite(ref, "ref_pixel")
        height, width = (int(detector_shape[0]), int(detector_shape[1]))
        if len(tuple(detector_shape)) != 2 or height < 2 or width < 2:
            raise ValueError("detector_shape must be (height, width) with both >= 2")
        if int(validation_nodes) < 2:
            raise ValueError("validation_nodes must be at least 2")
        pairs = [(i - j, j) for i in range(degree_x + 1) for j in range(i + 1)]
        self.degree = degree_x
        self.detector_shape = (height, width)
        self.validation_nodes = int(validation_nodes)
        self.register_buffer("ref_pixel", ref.clone())
        self.register_buffer(
            "exponent_u", torch.tensor([p[0] for p in pairs], dtype=torch.long)
        )
        self.register_buffer(
            "exponent_v", torch.tensor([p[1] for p in pairs], dtype=torch.long)
        )
        self.coefficients = nn.Parameter(
            torch.stack((flat_x, flat_y)).clone(), requires_grad=bool(learnable)
        )
        self.assert_invertible()

    @classmethod
    def from_siaf(
        cls,
        coeffs_x: Any,
        coeffs_y: Any,
        *,
        x_sci_ref: float,
        y_sci_ref: float,
        detector_shape: tuple[int, int],
        learnable: bool = False,
        validation_nodes: int = 33,
    ) -> PolynomialDetectorDistortion:
        """Build from SIAF ``Sci2Idl`` coefficients and one-based ``XSciRef``/``YSciRef``.

        Args:
            coeffs_x: ``Sci2IdlX{i}{j}`` as a triangular array ``[i, j]`` or a
                flat array in SIAF order.
            coeffs_y: ``Sci2IdlY{i}{j}``, same layout.
            x_sci_ref: SIAF ``XSciRef`` (one-based pixels).
            y_sci_ref: SIAF ``YSciRef`` (one-based pixels).
            detector_shape: ``(height, width)`` in pixels.
            learnable: Make the coefficients trainable.
            validation_nodes: Nodes per axis of the validation lattice.

        Returns:
            The distortion, with ``ref_pixel = (x_sci_ref - 1, y_sci_ref - 1)``.

        Raises:
            ValueError: As the constructor, or on a non-finite reference pixel.
        """
        ref = (
            _scalar(x_sci_ref, "x_sci_ref") - 1.0,
            _scalar(y_sci_ref, "y_sci_ref") - 1.0,
        )
        return cls(
            coeffs_x,
            coeffs_y,
            ref_pixel=ref,
            detector_shape=detector_shape,
            learnable=learnable,
            validation_nodes=validation_nodes,
        )

    @property
    def triangular_coefficients(self) -> tuple[Tensor, Tensor]:
        """Differentiable ``(a, b)`` triangular arrays of shape ``(d + 1, d + 1)``."""
        d = self.degree
        i_idx = self.exponent_u + self.exponent_v
        out = []
        for row in range(2):
            grid = self.coefficients.new_zeros(d + 1, d + 1)
            out.append(grid.index_put((i_idx, self.exponent_v), self.coefficients[row]))
        return out[0], out[1]

    def evaluate_with_jacobian(self, points: Any) -> tuple[Tensor, Tensor]:
        """Return ``(ideal, jacobian)``: arcsec ``(..., 2)`` and ``d(X, Y)/d(x, y)``.

        Args:
            points: Zero-based detector pixels ``(x, y)``, shape ``(..., 2)``.

        Returns:
            Ideal-frame coordinates ``(..., 2)`` in arcsec and the Jacobian
            ``(..., 2, 2)`` in arcsec per pixel, rows ``(X, Y)``, columns
            ``(x, y)``. Computed in float64, returned in the input dtype.

        Raises:
            ValueError: If ``points`` is not a finite real floating tensor with
                last dimension 2.
        """
        pts = _coords(points, "points")
        value, jac = self._evaluate(pts, jacobian=True)
        assert jac is not None
        return value, jac

    def _evaluate(self, pts: Tensor, *, jacobian: bool) -> tuple[Tensor, Tensor | None]:
        work = _to(pts, device=pts.device, dtype=_F64)
        ref = _to(self.ref_pixel, device=pts.device, dtype=_F64)
        coeff = _to(self.coefficients, device=pts.device, dtype=_F64)
        u = work[..., 0] - ref[0]
        v = work[..., 1] - ref[1]
        pu = _powers(u, self.degree)
        pv = _powers(v, self.degree)
        # Dense (2, d + 1, d + 1) matrix C[k, a, b] of the coefficient of u^a v^b:
        # contracting P_u with C first keeps memory at O(N * d) per output instead
        # of O(N * d^2) for the full term table (matters for full-detector grids).
        dense = coeff.new_zeros(2, self.degree + 1, self.degree + 1)
        dense = dense.index_put(
            (
                torch.arange(2, device=pts.device)[:, None],
                self.exponent_u.to(pts.device)[None, :],
                self.exponent_v.to(pts.device)[None, :],
            ),
            coeff,
        )
        pu_c = torch.einsum("...a,kab->...kb", pu, dense)
        value = (pu_c * pv[..., None, :]).sum(-1)
        if not jacobian:
            return value.to(pts.dtype), None
        zero = torch.zeros_like(u)[..., None]
        k = torch.arange(self.degree + 1, device=pts.device, dtype=_F64)
        dpu = k * torch.cat((zero, pu[..., :-1]), dim=-1)
        dpv = k * torch.cat((zero, pv[..., :-1]), dim=-1)
        d_x = (torch.einsum("...a,kab->...kb", dpu, dense) * pv[..., None, :]).sum(-1)
        d_y = (pu_c * dpv[..., None, :]).sum(-1)
        jac = torch.stack((d_x, d_y), dim=-1)
        return value.to(pts.dtype), jac.to(pts.dtype)

    def forward(self, points: Any) -> Tensor:
        """Map zero-based detector pixels ``(..., 2)`` to ideal arcsec ``(..., 2)``."""
        return self._evaluate(_coords(points, "points"), jacobian=False)[0]

    def jacobian(self, points: Any) -> Tensor:
        """Return ``d(X, Y)/d(x, y)`` with shape ``(..., 2, 2)``, arcsec per pixel."""
        return self.evaluate_with_jacobian(points)[1]

    def validation_points(self) -> Tensor:
        """Return the float64 validation lattice ``(n, n, 2)`` over the detector."""
        height, width = self.detector_shape
        device = self.coefficients.device
        x = torch.linspace(
            0, width - 1, self.validation_nodes, dtype=_F64, device=device
        )
        y = torch.linspace(
            0, height - 1, self.validation_nodes, dtype=_F64, device=device
        )
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=-1)

    def assert_invertible(self) -> Tensor:
        """Return lattice determinants, or raise if the map is folded or singular.

        Raises:
            ValueError: If any value or Jacobian entry on the lattice is
                non-finite, or any determinant is not strictly positive.
        """
        with torch.no_grad():
            value, jac = self.evaluate_with_jacobian(self.validation_points())
            _finite(value, "distortion output")
            _finite(jac, "distortion Jacobian")
            det = jac[..., 0, 0] * jac[..., 1, 1] - jac[..., 0, 1] * jac[..., 1, 0]
            if not bool((det > 0).all().cpu().item()):
                raise ValueError(
                    "distortion polynomial is folded, reflected or singular: the "
                    "Jacobian determinant must be positive over the detector "
                    f"(minimum {float(det.min().cpu().item()):.8g} arcsec^2/pixel^2)"
                )
        return det

    def inverse(
        self,
        values: Any,
        *,
        max_iter: int = 25,
        atol: float = 1.0e-9,
        fail_closed: bool = True,
    ) -> Tensor:
        """Newton inverse: ideal arcsec ``(..., 2)`` to zero-based pixels.

        Starts from the linearization at the reference pixel. A point counts as
        converged when the forward residual is at most ``atol`` arcsec and the
        Jacobian determinant at the solution is positive (so the root is on the
        branch connected to the detector).

        Args:
            values: Ideal-frame coordinates, arcsec, shape ``(..., 2)``.
            max_iter: Newton iterations (>= 1).
            atol: Forward residual tolerance in arcsec (finite, >= 0).
            fail_closed: Raise on non-convergence. With ``False`` the
                non-converged points are returned as NaN.

        Returns:
            Zero-based pixel coordinates ``(..., 2)`` in the input dtype.

        Raises:
            ValueError: On bad input, or (``fail_closed``) when any point does
                not converge to ``atol``.
        """
        target = _coords(values, "values")
        if int(max_iter) < 1:
            raise ValueError("max_iter must be at least 1")
        if not math.isfinite(atol) or atol < 0:
            raise ValueError("atol must be finite and non-negative")
        work = _to(target, device=target.device, dtype=_F64)
        ref = _to(self.ref_pixel, device=target.device, dtype=_F64)
        with torch.no_grad():
            value0, jac0 = self.evaluate_with_jacobian(ref)
            x = ref + torch.linalg.solve(jac0, (work - value0).unsqueeze(-1)).squeeze(
                -1
            )
            for _ in range(int(max_iter)):
                pred, jac = self.evaluate_with_jacobian(x)
                x = x - torch.linalg.solve(jac, (pred - work).unsqueeze(-1)).squeeze(-1)
            pred, jac = self.evaluate_with_jacobian(x)
            det = jac[..., 0, 0] * jac[..., 1, 1] - jac[..., 0, 1] * jac[..., 1, 0]
            residual = (pred - work).abs().amax(dim=-1)
            ok = torch.isfinite(residual) & (residual <= atol) & (det > 0)
        if not bool(ok.all().cpu().item()):
            if fail_closed:
                bad = int((~ok).sum().cpu().item())
                worst = float(
                    torch.nan_to_num(residual, nan=float("inf")).max().cpu().item()
                )
                raise ValueError(
                    f"polynomial inverse did not converge for {bad} point(s) "
                    f"(worst residual {worst:.8g} arcsec, atol={atol:.8g}, "
                    f"max_iter={int(max_iter)})"
                )
            x = torch.where(ok[..., None], x, torch.full_like(x, float("nan")))
        if torch.is_grad_enabled() and (
            target.requires_grad or self.coefficients.requires_grad
        ):
            # One differentiable Newton step from the converged root gives the
            # implicit-function-theorem gradients w.r.t. target and coefficients.
            safe = torch.where(ok[..., None], x, ref.expand_as(x)).detach()
            pred, jac = self.evaluate_with_jacobian(safe)
            step = torch.linalg.solve(jac, (pred - work).unsqueeze(-1)).squeeze(-1)
            x = torch.where(ok[..., None], safe - step, x)
        return x.to(target.dtype)


class TelescopeAttitude(_DeviceOnlyApply):
    """Ideal-frame arcsec to sky ``(RA, Dec)`` degrees through the telescope frame.

    Steps (all float64):

    1. ideal ``(x, y)`` to telescope ``(V2, V3)`` in arcsec, the planar SIAF
       convention with parity ``p`` (``VIdlParity``) and angle ``theta``
       (``V3IdlYAngle``)::

           V2 = V2_ref + p * cos(theta) * x + sin(theta) * y
           V3 = V3_ref - p * sin(theta) * x + cos(theta) * y

    2. ``(V2, V3)`` are angles on the telescope sphere (longitude, latitude).
       The attitude matrix, as in JWST-STScI-001550 section 5.1 and
       ``pysiaf.utils.rotations.attitude``, is
       ``A = Rz(RA) Ry(-Dec) Rx(-PA) Ry(V3_ref) Rz(-V2_ref)``; the sky unit
       vector is ``A @ unit(V2, V3)``. No small-angle approximation is made.

    Learnable corrections ``(d_east, d_north, d_roll)``, in arcsec and zero by
    default, are applied to the attitude inputs: ``(RA_REF, DEC_REF)`` moves
    by the exact tangent-plane offset ``(d_east, d_north)`` (east and north
    of the reference point, gnomonic deprojection), and ``ROLL_REF`` gains
    ``d_roll`` arcsec. They are an ``nn.Parameter`` that requires gradients
    only when ``learnable=True``. Output RA is in ``[0, 360)``.

    Args:
        v2_ref_arcsec: ``V2_REF`` (aperture reference point, arcsec).
        v3_ref_arcsec: ``V3_REF`` (arcsec).
        ra_ref_deg: ``RA_REF``: sky RA of the reference point, degrees.
        dec_ref_deg: ``DEC_REF``, degrees, in ``[-90, 90]``.
        roll_ref_deg: ``ROLL_REF``: position angle of V3, degrees east of north.
        v3_idl_yangle_deg: ``V3IdlYAngle`` (header ``V3I_YANG``), degrees.
        v_idl_parity: ``VIdlParity`` (header ``VPARITY``), ``+1`` or ``-1``.
        learnable: Make the three corrections trainable.

    Raises:
        ValueError: On non-finite values, ``dec_ref_deg`` outside ``[-90, 90]``
            or a parity other than +1 or -1.

    Examples:
        >>> import torch
        >>> att = TelescopeAttitude(
        ...     -122.4, -457.7, 110.87, -73.45, 144.94,
        ...     v3_idl_yangle_deg=0.345, v_idl_parity=-1,
        ... )
        >>> ra, dec = att(torch.zeros(1, 2, dtype=torch.float64)).unbind(-1)
        >>> round(float(dec), 6)
        -73.45
    """

    constants: Tensor
    correction: nn.Parameter

    def __init__(
        self,
        v2_ref_arcsec: float,
        v3_ref_arcsec: float,
        ra_ref_deg: float,
        dec_ref_deg: float,
        roll_ref_deg: float,
        *,
        v3_idl_yangle_deg: float,
        v_idl_parity: int,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        values = (
            _scalar(v2_ref_arcsec, "v2_ref_arcsec"),
            _scalar(v3_ref_arcsec, "v3_ref_arcsec"),
            _scalar(ra_ref_deg, "ra_ref_deg"),
            _scalar(dec_ref_deg, "dec_ref_deg"),
            _scalar(roll_ref_deg, "roll_ref_deg"),
            _scalar(v3_idl_yangle_deg, "v3_idl_yangle_deg"),
        )
        if abs(values[3]) > 90.0:
            raise ValueError("dec_ref_deg must be within [-90, 90]")
        parity = _scalar(v_idl_parity, "v_idl_parity")
        if parity not in (-1.0, 1.0):
            raise ValueError("v_idl_parity must be +1 or -1")
        self.register_buffer("constants", torch.tensor((*values, parity), dtype=_F64))
        self.correction = nn.Parameter(
            torch.zeros(3, dtype=_F64), requires_grad=bool(learnable)
        )

    def _constants(self, device: torch.device) -> list[Tensor]:
        return list(_to(self.constants, device=device, dtype=_F64).unbind())

    def attitude_matrix(self, device: torch.device | None = None) -> Tensor:
        """Return the float64 ``(3, 3)`` attitude matrix, telescope to sky.

        Args:
            device: Device of the result (default: that of the parameters).

        Returns:
            The rotation matrix including the learnable corrections.
        """
        dev = self.constants.device if device is None else device
        v2, v3, ra, dec, roll, _, _ = self._constants(dev)
        corr = _to(self.correction, device=dev, dtype=_F64)
        deg = math.pi / 180.0
        arc = deg / _ARCSEC_PER_DEG
        ra_r, dec_r = ra * deg, dec * deg
        east = torch.stack((-torch.sin(ra_r), torch.cos(ra_r), torch.zeros_like(ra_r)))
        north = torch.stack(
            (
                -torch.sin(dec_r) * torch.cos(ra_r),
                -torch.sin(dec_r) * torch.sin(ra_r),
                torch.cos(dec_r),
            )
        )
        point = _unit_vector(ra_r, dec_r) + corr[0] * arc * east + corr[1] * arc * north
        ra_c, dec_c = _lon_lat(point)
        roll_c = roll * deg + corr[2] * arc
        return (
            _rotation(2, ra_c)
            @ _rotation(1, -dec_c)
            @ _rotation(0, -roll_c)
            @ _rotation(1, v3 * arc)
            @ _rotation(2, -v2 * arc)
        )

    def ideal_to_telescope(self, ideal: Any) -> Tensor:
        """Planar SIAF map from ideal arcsec ``(..., 2)`` to ``(V2, V3)`` arcsec."""
        pts = _coords(ideal, "ideal")
        work = _to(pts, device=pts.device, dtype=_F64)
        v2r, v3r, _, _, _, theta, parity = self._constants(pts.device)
        t = theta * (math.pi / 180.0)
        c, s = torch.cos(t), torch.sin(t)
        x, y = work[..., 0], work[..., 1]
        out = torch.stack(
            (v2r + parity * c * x + s * y, v3r - parity * s * x + c * y), dim=-1
        )
        return out.to(pts.dtype)

    def telescope_to_ideal(self, telescope: Any) -> Tensor:
        """Inverse of :meth:`ideal_to_telescope` (planar), arcsec ``(..., 2)``."""
        pts = _coords(telescope, "telescope")
        work = _to(pts, device=pts.device, dtype=_F64)
        v2r, v3r, _, _, _, theta, parity = self._constants(pts.device)
        t = theta * (math.pi / 180.0)
        c, s = torch.cos(t), torch.sin(t)
        d2, d3 = work[..., 0] - v2r, work[..., 1] - v3r
        out = torch.stack((parity * c * d2 - parity * s * d3, s * d2 + c * d3), dim=-1)
        return out.to(pts.dtype)

    def forward(self, ideal: Any) -> Tensor:
        """Map ideal-frame arcsec ``(..., 2)`` to ``(RA, Dec)`` degrees ``(..., 2)``.

        Args:
            ideal: Ideal-frame ``(x, y)`` in arcsec, shape ``(..., 2)``.

        Returns:
            ``(RA, Dec)`` in degrees, RA in ``[0, 360)``, input dtype.

        Raises:
            ValueError: If ``ideal`` is not a finite real floating tensor with
                last dimension 2.
        """
        pts = _coords(ideal, "ideal")
        tel = self.ideal_to_telescope(_to(pts, device=pts.device, dtype=_F64))
        arc = math.pi / 180.0 / _ARCSEC_PER_DEG
        vec = _unit_vector(tel[..., 0] * arc, tel[..., 1] * arc)
        sky = vec @ self.attitude_matrix(pts.device).T
        ra, dec = _lon_lat(sky)
        deg = 180.0 / math.pi
        world = torch.stack((torch.remainder(ra * deg, 360.0), dec * deg), dim=-1)
        return world.to(pts.dtype)

    def inverse(self, world: Any) -> Tensor:
        """Map ``(RA, Dec)`` degrees ``(..., 2)`` back to ideal-frame arcsec.

        Args:
            world: ``(RA, Dec)`` in degrees, shape ``(..., 2)``.

        Returns:
            Ideal-frame ``(x, y)`` in arcsec, input dtype. Exact inverse of
            :meth:`forward` for every point whose telescope-frame position is
            within 90 degrees of the boresight.

        Raises:
            ValueError: If ``world`` is not a finite real floating tensor with
                last dimension 2, or has ``|Dec| > 90``.
        """
        pts = _coords(world, "world")
        work = _to(pts, device=pts.device, dtype=_F64)
        if bool((work[..., 1].abs() > 90.0).any().cpu().item()):
            raise ValueError("world latitude must be within [-90, 90] degrees")
        deg = math.pi / 180.0
        vec = _unit_vector(work[..., 0] * deg, work[..., 1] * deg)
        tel = vec @ self.attitude_matrix(pts.device)
        lon, lat = _lon_lat(tel)
        arcsec = 180.0 / math.pi * _ARCSEC_PER_DEG
        ideal = self.telescope_to_ideal(
            torch.stack((lon * arcsec, lat * arcsec), dim=-1)
        )
        return ideal.to(pts.dtype)


class TelescopeFrameDetectorToSky(_DeviceOnlyApply):
    """Zero-based detector pixels to sky and back, from a polynomial and an attitude.

    Satisfies the ``FieldObservation.detector_to_sky`` protocol
    (``pixel_to_world``) and adds ``world_to_pixel``, which shared-sky cropping
    needs. The ideal frame is ``distortion(x, y) + residual(x, y)``; the
    optional residual is an :class:`AffineFreeResidualGrid2D` whose control
    values are arcsec in the ideal frame and whose grid must cover the detector
    in zero-based pixels (points outside raise ``ValueError``, as for the
    class itself). Its constant and affine modes are projected out, so it cannot
    absorb pointing, roll or plate-scale offsets. Pose corrections belong to
    :class:`TelescopeAttitude`. They overlap with an overall detector offset
    and rotation, so do not also fit a free affine on the same data.

    Args:
        distortion: Pixel to ideal polynomial.
        attitude: Ideal to sky.
        residual: Optional learnable detector-shared residual, arcsec.

    Raises:
        TypeError: If an argument has the wrong type.

    Examples:
        >>> import torch
        >>> a = torch.zeros(2, 2, dtype=torch.float64); a[1, 0] = 0.03
        >>> b = torch.zeros(2, 2, dtype=torch.float64); b[1, 1] = 0.03
        >>> poly = PolynomialDetectorDistortion(
        ...     a, b, ref_pixel=(1.5, 1.5), detector_shape=(4, 4)
        ... )
        >>> att = TelescopeAttitude(
        ...     -120.0, -450.0, 10.0, 20.0, 30.0, v3_idl_yangle_deg=0.0, v_idl_parity=-1
        ... )
        >>> t = TelescopeFrameDetectorToSky(poly, att)
        >>> ra, dec = t.pixel_to_world(torch.tensor([0.5]), torch.tensor([2.5]))
        >>> x, y = t.world_to_pixel(ra, dec)
    """

    def __init__(
        self,
        distortion: PolynomialDetectorDistortion,
        attitude: TelescopeAttitude,
        residual: AffineFreeResidualGrid2D | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(distortion, PolynomialDetectorDistortion):
            raise TypeError("distortion must be a PolynomialDetectorDistortion")
        if not isinstance(attitude, TelescopeAttitude):
            raise TypeError("attitude must be a TelescopeAttitude")
        if residual is not None and not isinstance(residual, AffineFreeResidualGrid2D):
            raise TypeError("residual must be an AffineFreeResidualGrid2D or None")
        self.distortion = distortion
        self.attitude = attitude
        self.residual = residual
        self.assert_invertible()

    @classmethod
    def from_header(
        cls,
        header: Mapping[str, Any],
        coeffs_x: Any,
        coeffs_y: Any,
        *,
        x_sci_ref: float,
        y_sci_ref: float,
        detector_shape: tuple[int, int] | None = None,
        learnable: bool = False,
        residual: AffineFreeResidualGrid2D | None = None,
    ) -> TelescopeFrameDetectorToSky:
        """Build from a count-rate ``SCI`` header dict plus ``Sci2Idl`` coefficients.

        Uses ``V2_REF``, ``V3_REF``, ``ROLL_REF``, ``RA_REF``, ``DEC_REF``,
        ``VPARITY`` and ``V3I_YANG`` (keys matched case-insensitively). The
        header's own tangent-plane WCS and any distortion keywords are ignored.

        Args:
            header: Plain header dict, for example from
                ``torchfits.read_header(path, "SCI")``.
            coeffs_x: SIAF ``Sci2IdlX{i}{j}`` (triangular or flat).
            coeffs_y: SIAF ``Sci2IdlY{i}{j}``.
            x_sci_ref: SIAF ``XSciRef`` (one-based pixel).
            y_sci_ref: SIAF ``YSciRef`` (one-based pixel).
            detector_shape: ``(height, width)``; default ``NAXIS2``/``NAXIS1``.
            learnable: Make the polynomial coefficients and the attitude
                corrections trainable.
            residual: Optional detector residual.

        Returns:
            The composed transform.

        Raises:
            ValueError: If any required keyword is missing (the message names
                them) or not a finite number, or the detector shape is unknown.
        """
        upper = {str(k).upper(): v for k, v in header.items()}
        missing = [k for k in _HEADER_KEYWORDS if upper.get(k) is None]
        if missing:
            raise ValueError(
                "header lacks the pointing keyword(s) "
                + ", ".join(missing)
                + f"; required: {', '.join(_HEADER_KEYWORDS)}"
            )
        if detector_shape is None:
            if upper.get("NAXIS1") is None or upper.get("NAXIS2") is None:
                raise ValueError(
                    "detector_shape not given and header lacks NAXIS1/NAXIS2"
                )
            detector_shape = (int(upper["NAXIS2"]), int(upper["NAXIS1"]))
        values = {k: _scalar(upper[k], k) for k in _HEADER_KEYWORDS}
        if values["VPARITY"] not in (-1.0, 1.0):
            raise ValueError("VPARITY must be +1 or -1")
        attitude = TelescopeAttitude(
            values["V2_REF"],
            values["V3_REF"],
            values["RA_REF"],
            values["DEC_REF"],
            values["ROLL_REF"],
            v3_idl_yangle_deg=values["V3I_YANG"],
            v_idl_parity=int(values["VPARITY"]),
            learnable=learnable,
        )
        distortion = PolynomialDetectorDistortion.from_siaf(
            coeffs_x,
            coeffs_y,
            x_sci_ref=x_sci_ref,
            y_sci_ref=y_sci_ref,
            detector_shape=detector_shape,
            learnable=learnable,
        )
        return cls(distortion, attitude, residual)

    def _ideal_with_jacobian(self, work: Tensor) -> tuple[Tensor, Tensor]:
        ideal, jac = self.distortion.evaluate_with_jacobian(work)
        if self.residual is not None:
            extra, extra_jac = self.residual.evaluate_with_jacobian(work)
            ideal, jac = ideal + extra, jac + extra_jac
        return ideal, jac

    def _ideal(self, work: Tensor) -> Tensor:
        ideal = self.distortion._evaluate(work, jacobian=False)[0]
        if self.residual is not None:
            ideal = ideal + self.residual(work)
        return ideal

    def pixel_to_ideal(self, points: Any) -> Tensor:
        """Map zero-based pixels ``(..., 2)`` to ideal-frame arcsec ``(..., 2)``."""
        pts = _coords(points, "points")
        return self._ideal(_to(pts, device=pts.device, dtype=_F64)).to(pts.dtype)

    def forward(self, points: Any) -> Tensor:
        """Map zero-based pixels ``(..., 2)`` to ``(RA, Dec)`` degrees ``(..., 2)``."""
        pts = _coords(points, "points")
        work = _to(pts, device=pts.device, dtype=_F64)
        return self.attitude(self._ideal(work)).to(pts.dtype)

    def pixel_to_world(self, x: Any, y: Any) -> tuple[Tensor, Tensor]:
        """Map full-detector zero-based pixels to ``(RA, Dec)`` in degrees.

        Args:
            x: Pixel x, real floating tensor of any shape.
            y: Pixel y, broadcastable with ``x``.

        Returns:
            ``(ra, dec)`` in degrees, RA in ``[0, 360)``, shape of the
            broadcast, dtype of ``x``. Differentiable in ``x``, ``y`` and all
            learnable parameters.

        Raises:
            ValueError: On non-floating or non-finite input, or a pixel outside
                the residual grid.
        """
        x_t = torch.as_tensor(x)
        if not x_t.is_floating_point() or x_t.is_complex():
            raise ValueError("x must be a real floating-point tensor")
        y_t = _as_tensor(y, dtype=x_t.dtype, device=x_t.device)
        x_b, y_b = torch.broadcast_tensors(x_t, y_t)
        world = self(torch.stack((x_b, y_b), dim=-1))
        return world[..., 0], world[..., 1]

    def world_to_pixel(
        self,
        lon: Any,
        lat: Any,
        *,
        max_iter: int = 25,
        atol: float = 1.0e-9,
        fail_closed: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Map ``(RA, Dec)`` degrees to zero-based full-detector pixels.

        The sky to ideal step is exact; the ideal to pixel step is a Newton
        iteration started from the polynomial's reference-pixel linearization.
        With a residual grid, iterates are clamped to the grid domain.

        Args:
            lon: Right ascension, degrees.
            lat: Declination, degrees, broadcastable with ``lon``.
            max_iter: Newton iterations (>= 1).
            atol: Tolerance on the forward residual in ideal arcsec.
            fail_closed: Raise if any point does not converge (including points
                far outside the detector where the polynomial is folded).
                With ``False`` such points are returned as NaN, which the
                shared-sky crop code treats as "not on this detector".

        Returns:
            ``(x, y)`` zero-based pixels, dtype of ``lon``.

        Raises:
            ValueError: On bad input or (``fail_closed``) non-convergence.
        """
        lon_t = torch.as_tensor(lon)
        if not lon_t.is_floating_point() or lon_t.is_complex():
            raise ValueError("lon must be a real floating-point tensor")
        lat_t = _as_tensor(lat, dtype=lon_t.dtype, device=lon_t.device)
        lon_b, lat_b = torch.broadcast_tensors(lon_t, lat_t)
        world = torch.stack((lon_b, lat_b), dim=-1)
        if int(max_iter) < 1:
            raise ValueError("max_iter must be at least 1")
        if not math.isfinite(atol) or atol < 0:
            raise ValueError("atol must be finite and non-negative")
        target = _to(self.attitude.inverse(world), device=world.device, dtype=_F64)
        if self.residual is None:
            pix = self.distortion.inverse(
                target, max_iter=max_iter, atol=atol, fail_closed=fail_closed
            )
            pix = pix.to(_F64)
        else:
            pix = self._inverse_with_residual(target, max_iter, atol, fail_closed)
        pix = pix.to(lon_t.dtype)
        return pix[..., 0], pix[..., 1]

    def _inverse_with_residual(
        self, target: Tensor, max_iter: int, atol: float, fail_closed: bool
    ) -> Tensor:
        res = self.residual
        assert res is not None
        dev = target.device
        origin = _to(res.origin, device=dev, dtype=_F64)
        spacing = _to(res.spacing, device=dev, dtype=_F64)
        height, width = res.control.shape[:2]
        lo = origin
        hi = origin + spacing * torch.tensor(
            [width - 1, height - 1], dtype=_F64, device=dev
        )
        with torch.no_grad():
            x = self.distortion.inverse(
                target, max_iter=max_iter, atol=atol, fail_closed=False
            ).to(_F64)
            x = torch.where(
                torch.isfinite(x),
                x,
                _to(self.distortion.ref_pixel, device=dev, dtype=_F64).expand_as(x),
            )
            for _ in range(int(max_iter)):
                x = torch.minimum(torch.maximum(x, lo), hi)
                pred, jac = self._ideal_with_jacobian(x)
                x = x - torch.linalg.solve(jac, (pred - target).unsqueeze(-1)).squeeze(
                    -1
                )
            x = torch.minimum(torch.maximum(x, lo), hi)
            pred, jac = self._ideal_with_jacobian(x)
            det = jac[..., 0, 0] * jac[..., 1, 1] - jac[..., 0, 1] * jac[..., 1, 0]
            residual = (pred - target).abs().amax(dim=-1)
            ok = torch.isfinite(residual) & (residual <= atol) & (det > 0)
        if not bool(ok.all().cpu().item()):
            if fail_closed:
                bad = int((~ok).sum().cpu().item())
                worst = float(
                    torch.nan_to_num(residual, nan=float("inf")).max().cpu().item()
                )
                raise ValueError(
                    f"inverse did not converge for {bad} point(s) (worst residual "
                    f"{worst:.8g} arcsec, atol={atol:.8g}, max_iter={int(max_iter)})"
                )
            x = torch.where(ok[..., None], x, torch.full_like(x, float("nan")))
        if torch.is_grad_enabled() and (
            target.requires_grad or any(p.requires_grad for p in self.parameters())
        ):
            safe = torch.where(ok[..., None], x, lo.expand_as(x)).detach()
            pred, jac = self._ideal_with_jacobian(safe)
            step = torch.linalg.solve(jac, (pred - target).unsqueeze(-1)).squeeze(-1)
            x = torch.where(ok[..., None], safe - step, x)
        return x

    def assert_invertible(self) -> Tensor:
        """Return composite determinants on the validation lattice or raise.

        Raises:
            ValueError: If the polynomial, or the polynomial plus residual, has
                a non-positive Jacobian determinant on the lattice.
        """
        self.distortion.assert_invertible()
        with torch.no_grad():
            pts = self.distortion.validation_points()
            _, jac = self._ideal_with_jacobian(pts)
            det = jac[..., 0, 0] * jac[..., 1, 1] - jac[..., 0, 1] * jac[..., 1, 0]
            if not bool(torch.isfinite(det).all().cpu().item()) or not bool(
                (det > 0).all().cpu().item()
            ):
                raise ValueError("composite detector transform is folded or singular")
        return det


__all__ = [
    "PolynomialDetectorDistortion",
    "TelescopeAttitude",
    "TelescopeFrameDetectorToSky",
]
