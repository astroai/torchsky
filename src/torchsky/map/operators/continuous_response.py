"""Source-quadrature transport into native, already integrated detector pixels.

This is a finite linear quadrature operator, not a continuous-field precision
certificate. Spatial/spectral quadrature accuracy, coordinate transforms and
response calibration belong to the caller. No source raster or delta image is
constructed. Responses may vary with source position and wavelength.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from .epsf import PixelIntegratedResponse, _floating_tensor, _same_tensor_type


class NativeSpectralPixelResponse(Protocol):
    """An intensity response integrated over each native detector footprint.

    Inputs broadcast to [detector, source, wavelength]: ``detector_xy`` has
    shape [D,1,1,2], ``source_xy`` [1,S,1,2], and ``wavelength_angstrom``
    [1,1,L]. For an explicit interaction graph both positions instead have
    shape [E,1,1,2], pairing corresponding edges; output is [E,1,L]. Positions
    are in the response's common coordinate chart, in (x,y)
    order. ``detector_footprint`` is None or [D,K] metadata defined by the
    callable (e.g. pixel side lengths), [E,K] for paired edges; None must have
    an explicitly declared pixel convention. Output is finite, nonnegative
    [D,S,L] (or [E,1,L] for paired edges).

    The result is the detector-integrated response per unit incident source
    flux at a single wavelength. It may include declared throughput/QE. It
    must not depend on the transported intensity; the adjoint assumes a fixed,
    deterministic response. Physical optics or a learned module can implement
    this contract. Neither infinite-domain normalization nor aperture
    normalization is inferred here.
    """

    def __call__(
        self,
        detector_xy: Tensor,
        source_xy: Tensor,
        wavelength_angstrom: Tensor,
        *,
        detector_footprint: Tensor | None,
    ) -> Tensor: ...


@dataclass
class AchromaticPixelResponse:
    """Adapt an offset-only unit-square ``PixelIntegratedResponse``.

    Both position tensors must be in detector-pixel coordinates. Wavelength
    does not affect this reference response; no further pixel integral is
    applied. Arbitrary footprints require a different callable.
    """

    response: PixelIntegratedResponse

    def __call__(
        self,
        detector_xy: Tensor,
        source_xy: Tensor,
        wavelength_angstrom: Tensor,
        *,
        detector_footprint: Tensor | None,
    ) -> Tensor:
        if detector_footprint is not None:
            raise ValueError("offset-only response requires unit-square pixels")
        return self.response(detector_xy - source_xy).expand(
            *torch.broadcast_shapes(
                detector_xy.shape[:-1], source_xy.shape[:-1], wavelength_angstrom.shape
            )
        )


@dataclass
class NativeQuadratureResponseOp:
    """Transport sampled source measures into native detector intensities.

    Forward accepts arbitrary prefix axes followed by [N,L] and returns the
    same prefix followed by [P]. Its discrete matrix is
    ``A[p,n,l] = R[p,n,l] * source_measure[n] * wavelength_measure[l]
    * source_spectral_weight[n,l]``. The optional last factor defaults to one
    and accepts [L] or [N,L]. It weights each source spectrum before transport;
    it never mixes rendered output bands. Separate operators represent bands.

    ``source_measure`` is explicit: use ones for point-source spectral flux
    density, or source-node solid-angle quadrature weights (sr) for spectral
    radiance per sr. These are source areas, never detector-pixel areas. Use
    ``wavelength_measure`` in Angstrom for densities per Angstrom, or ones for
    already bin-integrated spectral flux. The response already includes the
    detector footprint integral exactly once. Output units are input units
    times these two measures and the response's calibration units. Electron
    conversion (area, duration, QE) must be declared by the caller; no implicit
    conversion from energy density or photon density is made.

    Source and detector positions, measures and response are fixed when taking
    the Euclidean adjoint. They and response parameters retain their gradients
    when differentiating forward/adjoint. ``mask=True`` keeps detector rows;
    removed rows are zero without renormalizing retained flux or PSF tails.

    ``interaction_pairs[E,2]`` optionally gives detector/source index pairs.
    Response evaluation costs O(E*L), with edge and wavelength chunk limits.
    Every forward/adjoint call also validates tensor values and graph bounds
    and checks graph uniqueness with ``torch.unique``; this validation is not
    restricted to the response chunk limits or a one-time construction cost. The
    required ``interaction_support`` string declares why omitted edges are zero in the
    represented response (e.g. exact compact support, or a calibrated core
    whose wings are transported separately). The declaration is provenance,
    not a support proof. Never truncate an infinite-tail response implicitly.
    Duplicate pairs are rejected. The graph can represent radiance mesh nodes
    and does not require a source catalogue.

    Without a graph all pairs are evaluated: time is O(P*N*L), a bounded
    reference/fallback rather than a survey-scale algorithm. Three chunk
    limits bound each evaluated response block, avoiding a dense global
    matrix. Autograd may retain block graphs; this is not a bounded training
    memory claim. Quadrature refinement and support verification remain the
    caller's responsibility. This operator does not certify wavefront fidelity.
    """

    response: NativeSpectralPixelResponse
    source_xy: Tensor
    detector_xy: Tensor
    wavelength_angstrom: Tensor
    source_measure: Tensor
    wavelength_measure: Tensor
    source_spectral_weight: Tensor | None = None
    detector_footprint: Tensor | None = None
    mask: Tensor | None = None
    interaction_pairs: Tensor | None = None
    interaction_support: str | None = None
    source_chunk_size: int = 128
    detector_chunk_size: int = 128
    wavelength_chunk_size: int = 16
    pair_chunk_size: int = 1024

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _floating_tensor(self.source_xy, "source_xy")
        for value, name in (
            (self.source_xy, "source_xy"),
            (self.detector_xy, "detector_xy"),
            (self.wavelength_angstrom, "wavelength_angstrom"),
            (self.source_measure, "source_measure"),
            (self.wavelength_measure, "wavelength_measure"),
        ):
            _same_tensor_type(value, self.source_xy, name)
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")
        for value, name in (
            (self.source_xy, "source_xy"),
            (self.detector_xy, "detector_xy"),
        ):
            if value.ndim != 2 or value.shape[-1] != 2:
                raise ValueError(f"{name} must have shape [count,2]")
        wave = self.wavelength_angstrom
        if wave.ndim != 1 or wave.numel() == 0:
            raise ValueError("wavelength_angstrom must be nonempty and 1D")
        if bool((wave <= 0).any()) or not bool((wave[1:] > wave[:-1]).all()):
            raise ValueError("wavelength_angstrom must be positive and increasing")
        for value, shape, name in (
            (self.source_measure, (self.source_xy.shape[0],), "source_measure"),
            (self.wavelength_measure, wave.shape, "wavelength_measure"),
        ):
            if value.shape != shape or bool((value < 0).any()):
                raise ValueError(f"{name} must be nonnegative with shape {shape}")
        weight = self.source_spectral_weight
        if weight is not None:
            _same_tensor_type(weight, self.source_xy, "source_spectral_weight")
            if (
                weight.shape
                not in (
                    wave.shape,
                    (self.source_xy.shape[0], wave.numel()),
                )
                or not bool(torch.isfinite(weight).all())
                or bool((weight < 0).any())
            ):
                raise ValueError(
                    "source_spectral_weight must be nonnegative [L] or [N,L]"
                )
        footprint = self.detector_footprint
        if footprint is not None:
            _same_tensor_type(footprint, self.source_xy, "detector_footprint")
            if (
                footprint.ndim != 2
                or footprint.shape[0] != self.detector_xy.shape[0]
                or not bool(torch.isfinite(footprint).all())
            ):
                raise ValueError("detector_footprint must be finite with shape [P,K]")
        if self.mask is not None and (
            self.mask.shape != (self.detector_xy.shape[0],)
            or self.mask.dtype != torch.bool
            or self.mask.device != self.source_xy.device
        ):
            raise ValueError(
                "mask must be bool with shape [P] on the coordinate device"
            )
        pairs = self.interaction_pairs
        if pairs is not None:
            if (
                pairs.ndim != 2
                or pairs.shape[1] != 2
                or pairs.dtype != torch.int64
                or pairs.device != self.source_xy.device
            ):
                raise ValueError(
                    "interaction_pairs must be int64 [E,2] on the coordinate device"
                )
            if (
                bool((pairs < 0).any())
                or bool((pairs[:, 0] >= self.detector_xy.shape[0]).any())
                or bool((pairs[:, 1] >= self.source_xy.shape[0]).any())
            ):
                raise ValueError("interaction_pairs indices are out of bounds")
            if torch.unique(pairs, dim=0).shape[0] != pairs.shape[0]:
                raise ValueError("interaction_pairs must not contain duplicate edges")
            if (
                not isinstance(self.interaction_support, str)
                or not self.interaction_support.strip()
            ):
                raise ValueError(
                    "interaction_pairs require an explicit interaction_support declaration"
                )
        elif self.interaction_support is not None:
            raise ValueError("interaction_support requires interaction_pairs")
        for value in (
            self.source_chunk_size,
            self.detector_chunk_size,
            self.wavelength_chunk_size,
            self.pair_chunk_size,
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("chunk sizes must be positive integers")

    def _pair_block(self, pairs: Tensor, waves: slice) -> Tensor:
        pixels, sources = pairs[:, 0], pairs[:, 1]
        response = self.response(
            self.detector_xy[pixels, None, None, :],
            self.source_xy[sources, None, None, :],
            self.wavelength_angstrom[None, None, waves],
            detector_footprint=None
            if self.detector_footprint is None
            else self.detector_footprint[pixels],
        )
        _same_tensor_type(response, self.source_xy, "response output")
        if response.shape != (
            pairs.shape[0],
            1,
            self.wavelength_angstrom[waves].numel(),
        ) or not bool(torch.isfinite(response).all()):
            raise ValueError("paired response must return finite [E,1,L] blocks")
        if bool((response < 0).any()):
            raise ValueError("intensity response must be nonnegative")
        matrix = (
            response[:, 0, :]
            * self.source_measure[sources, None]
            * self.wavelength_measure[None, waves]
        )
        if self.source_spectral_weight is not None:
            weight = self.source_spectral_weight
            matrix = matrix * (
                weight[None, waves] if weight.ndim == 1 else weight[sources, waves]
            )
        if self.mask is not None:
            matrix = torch.where(self.mask[pixels, None], matrix, 0.0)
        return matrix

    def _block(self, pixels: slice, sources: slice, waves: slice) -> Tensor:
        response = self.response(
            self.detector_xy[pixels, None, None, :],
            self.source_xy[None, sources, None, :],
            self.wavelength_angstrom[None, None, waves],
            detector_footprint=(
                None
                if self.detector_footprint is None
                else self.detector_footprint[pixels]
            ),
        )
        _same_tensor_type(response, self.source_xy, "response output")
        shape = (
            self.detector_xy[pixels].shape[0],
            self.source_xy[sources].shape[0],
            self.wavelength_angstrom[waves].numel(),
        )
        if response.shape != shape or not bool(torch.isfinite(response).all()):
            raise ValueError("response must return finite [D,S,L] blocks")
        if bool((response < 0).any()):
            raise ValueError("intensity response must be nonnegative")
        matrix = (
            response
            * self.source_measure[None, sources, None]
            * self.wavelength_measure[None, None, waves]
        )
        if self.source_spectral_weight is not None:
            weight = self.source_spectral_weight
            matrix = matrix * (
                weight[None, None, waves]
                if weight.ndim == 1
                else weight[None, sources, waves]
            )
        if self.mask is not None:
            matrix = torch.where(self.mask[pixels, None, None], matrix, 0.0)
        return matrix

    def forward(self, intensity: Tensor) -> Tensor:
        """Apply the quadrature matrix to ``intensity[...,N,L]``."""
        self._validate()
        _same_tensor_type(intensity, self.source_xy, "intensity")
        ns, nw = self.source_xy.shape[0], self.wavelength_angstrom.numel()
        if intensity.ndim < 2 or intensity.shape[-2:] != (ns, nw):
            raise ValueError("intensity must have trailing shape [N,L]")
        if not bool(torch.isfinite(intensity).all()):
            raise ValueError(
                "intensity must be finite (signed linear inputs are allowed)"
            )
        npix = self.detector_xy.shape[0]
        if (
            ns == 0
            or npix == 0
            or (
                self.interaction_pairs is not None
                and self.interaction_pairs.numel() == 0
            )
        ):
            # Retain the exact zero Jacobian for empty support without calling
            # a response or reducing the entire input tensor.
            zero = intensity[..., :0, :0].sum(dim=(-2, -1))
            return zero[..., None].expand(*intensity.shape[:-2], npix).clone()
        if self.interaction_pairs is not None:
            out = intensity.new_zeros(
                (*intensity.shape[:-2], self.detector_xy.shape[0])
            )
            for e in range(0, self.interaction_pairs.shape[0], self.pair_chunk_size):
                pairs = self.interaction_pairs[e : e + self.pair_chunk_size]
                for start in range(0, nw, self.wavelength_chunk_size):
                    waves = slice(start, start + self.wavelength_chunk_size)
                    edge_values = (
                        intensity[..., pairs[:, 1], waves]
                        * self._pair_block(pairs, waves)
                    ).sum(dim=-1)
                    index = pairs[:, 0].expand_as(edge_values)
                    out.scatter_add_(-1, index, edge_values)
            return out
        chunks = []
        for p in range(0, self.detector_xy.shape[0], self.detector_chunk_size):
            pixels = slice(p, p + self.detector_chunk_size)
            out = intensity.new_zeros(
                (*intensity.shape[:-2], self.detector_xy[pixels].shape[0])
            )
            for n in range(0, ns, self.source_chunk_size):
                sources = slice(n, n + self.source_chunk_size)
                for start in range(0, nw, self.wavelength_chunk_size):
                    waves = slice(start, start + self.wavelength_chunk_size)
                    out = out + torch.einsum(
                        "...nl,pnl->...p",
                        intensity[..., sources, waves],
                        self._block(pixels, sources, waves),
                    )
            chunks.append(out)
        return (
            torch.cat(chunks, dim=-1)
            if chunks
            else intensity.new_zeros((*intensity.shape[:-2], 0))
        )

    def adjoint(self, detector_values: Tensor) -> Tensor:
        """Apply the exact discrete transpose to ``detector_values[...,P]``.

        Nonfinite masked values are allowed and replaced with zero before the
        multiplication. This is the Euclidean transpose of the stated finite
        quadrature matrix, not an inverse or a continuous-density adjoint.
        """
        self._validate()
        _same_tensor_type(detector_values, self.source_xy, "detector_values")
        npix = self.detector_xy.shape[0]
        if detector_values.ndim < 1 or detector_values.shape[-1] != npix:
            raise ValueError("detector_values must have trailing shape [P]")
        values = (
            detector_values
            if self.mask is None
            else torch.where(self.mask, detector_values, 0.0)
        )
        if not bool(torch.isfinite(values).all()):
            raise ValueError("kept detector_values must be finite")
        ns, nw = self.source_xy.shape[0], self.wavelength_angstrom.numel()
        if (
            ns == 0
            or npix == 0
            or (
                self.interaction_pairs is not None
                and self.interaction_pairs.numel() == 0
            )
        ):
            zero = values[..., :0].sum(dim=-1)
            return zero[..., None, None].expand(*values.shape[:-1], ns, nw).clone()
        if self.interaction_pairs is not None:
            out = values.new_zeros((*values.shape[:-1], ns * nw))
            for e in range(0, self.interaction_pairs.shape[0], self.pair_chunk_size):
                pairs = self.interaction_pairs[e : e + self.pair_chunk_size]
                for start in range(0, nw, self.wavelength_chunk_size):
                    waves = slice(start, start + self.wavelength_chunk_size)
                    matrix = self._pair_block(pairs, waves)
                    edge_values = values[..., pairs[:, 0], None] * matrix
                    index = pairs[:, 1, None] * nw + torch.arange(
                        start, start + matrix.shape[-1], device=values.device
                    )
                    edge_values = edge_values.flatten(-2)
                    out.scatter_add_(
                        -1, index.flatten().expand_as(edge_values), edge_values
                    )
            return out.reshape(*values.shape[:-1], ns, nw)
        source_chunks = []
        for n in range(0, ns, self.source_chunk_size):
            sources = slice(n, n + self.source_chunk_size)
            wave_chunks = []
            for start in range(0, nw, self.wavelength_chunk_size):
                waves = slice(start, start + self.wavelength_chunk_size)
                out = values.new_zeros(
                    (
                        *values.shape[:-1],
                        self.source_xy[sources].shape[0],
                        self.wavelength_angstrom[waves].numel(),
                    )
                )
                for p in range(0, npix, self.detector_chunk_size):
                    pixels = slice(p, p + self.detector_chunk_size)
                    out = out + torch.einsum(
                        "...p,pnl->...nl",
                        values[..., pixels],
                        self._block(pixels, sources, waves),
                    )
                wave_chunks.append(out)
            source_chunks.append(torch.cat(wave_chunks, dim=-1))
        return (
            torch.cat(source_chunks, dim=-2)
            if source_chunks
            else values.new_zeros((*values.shape[:-1], 0, nw))
        )

__all__ = [
    "AchromaticPixelResponse",
    "NativeQuadratureResponseOp",
    "NativeSpectralPixelResponse",
]
