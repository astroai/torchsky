"""Internal geometry-operator family declarations."""

from __future__ import annotations

from dataclasses import dataclass

from .base import GeometryOp


@dataclass
class ExplicitGeometryOp(GeometryOp):
    def __init__(self, *, name: str = "explicit_geometry") -> None:
        super().__init__(name=name, learned=False)


@dataclass
class LearnedGeometryOp(GeometryOp):
    def __init__(self, *, name: str = "learned_geometry") -> None:
        super().__init__(name=name, learned=True)


@dataclass
class InterpolationGeometryOp(ExplicitGeometryOp):
    def __init__(self) -> None:
        super().__init__(name="interpolation_geometry")


@dataclass
class ExactDrizzleGeometryOp(ExplicitGeometryOp):
    def __init__(self) -> None:
        super().__init__(name="exact_drizzle_geometry")


@dataclass
class ForwardSplatGeometryOp(ExplicitGeometryOp):
    def __init__(self) -> None:
        super().__init__(name="forward_splat_geometry")


@dataclass
class ConservativePolygonRemapOp(ExplicitGeometryOp):
    def __init__(self) -> None:
        super().__init__(name="conservative_polygon_geometry")


@dataclass
class RestrictedAffineFFTGeometryOp(ExplicitGeometryOp):
    def __init__(self) -> None:
        super().__init__(name="restricted_affine_fft_geometry")


@dataclass
class SpectralGlobalGeometryOp(ExplicitGeometryOp):
    def __init__(self) -> None:
        super().__init__(name="spectral_global_geometry")


@dataclass
class SphericalSpectralGeometryOp(ExplicitGeometryOp):
    mode: str = "harmonic_direct"

    def __init__(self, *, mode: str = "harmonic_direct") -> None:
        super().__init__(name="spherical_spectral_geometry")
        self.mode = str(mode)
