"""Internal scene-representation adapters for mapmaking and INR paths."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .target import TargetRenderSpec


def target_to_field_observation(
    target: TargetRenderSpec,
    *,
    device: torch.device | None = None,
):
    from .field.types import FieldObservation

    if target.transmission_center is not None:
        wavelength = float(target.transmission_center)
    elif target.spectral_min is not None and target.spectral_max is not None:
        wavelength = 0.5 * (float(target.spectral_min) + float(target.spectral_max))
    else:
        wavelength = None
    return FieldObservation(
        image=torch.zeros(target.shape, dtype=torch.float32, device=device),
        header=dict(target.header),
        wavelength=wavelength,
        spectral_min=target.spectral_min,
        spectral_max=target.spectral_max,
        transmission_center=target.transmission_center,
        transmission_sigma=target.transmission_sigma,
        psf_spec=target.psf_spec,
        band_name=target.band_label,
    )


class SceneRepresentation(ABC):
    @abstractmethod
    def render_to_target(self, target: TargetRenderSpec, **kwargs) -> Tensor:
        raise NotImplementedError


def as_scene_representation(
    scene: SceneRepresentation | nn.Module | Tensor,
) -> SceneRepresentation:
    if isinstance(scene, SceneRepresentation):
        return scene
    if isinstance(scene, nn.Module):
        return ContinuousFieldModel(scene)
    if isinstance(scene, Tensor):
        return DiscreteTargetImage(scene)
    raise TypeError(
        "scene must be a SceneRepresentation, torch.nn.Module, or torch.Tensor"
    )


@dataclass
class DiscreteTargetImage(SceneRepresentation):
    data: Tensor
    header: dict[str, Any] | None = None
    value_semantics: str = "surface_brightness"

    def render_to_target(self, target: TargetRenderSpec, **kwargs) -> Tensor:
        if tuple(self.data.shape[-2:]) != tuple(target.shape):
            raise ValueError(
                "DiscreteTargetImage shape must match TargetRenderSpec shape"
            )
        return self.data


@dataclass
class HealpixMapScene(SceneRepresentation):
    data: Tensor
    nside: int | None = None
    nest: bool = False
    band_labels: tuple[str | None, ...] | None = None
    lmax: int | None = None
    target_nside: int | None = None
    fwhm_rad: float | None = None
    beam_window: Tensor | None = None
    iter: int = 0
    mode: str | None = None

    def __post_init__(self) -> None:
        data = torch.as_tensor(self.data)
        if data.ndim not in {1, 2}:
            raise ValueError(
                "HealpixMapScene data must be a 1D map or 2D band-first map stack"
            )
        if self.band_labels is not None:
            if data.ndim != 2:
                raise ValueError(
                    "band_labels require 2D band-first HealpixMapScene data"
                )
            if len(self.band_labels) != int(data.shape[0]):
                raise ValueError(
                    "band_labels length must match HealpixMapScene band count"
                )

    def _resolved_nside(self) -> int:
        from ..sphere import healpix as hp

        if self.nside is not None:
            return int(self.nside)
        return int(hp.npix2nside(int(torch.as_tensor(self.data).shape[-1])))

    def _resolved_mode(self, *, operators: object | None) -> str:
        from .operators.geometry import SphericalSpectralGeometryOp

        if self.mode is not None:
            return str(self.mode)
        geometry = getattr(operators, "geometry", None)
        if isinstance(geometry, SphericalSpectralGeometryOp):
            return str(geometry.mode)
        raise ValueError(
            "HealpixMapScene requires a spherical operator bundle or an explicit scene mode"
        )

    def _project_single_map(
        self,
        healpix_map: Tensor,
        target: TargetRenderSpec,
        *,
        mode: str,
    ) -> Tensor:
        from .healpix_spectral import (
            sample_healpix_map_on_image_header_harmonic,
            sample_healpix_map_on_image_header_spectral,
        )
        from .healpix_image_plane import sample_healpix_map_on_image_header

        nside = self._resolved_nside()
        lmax = self.lmax
        if mode == "pixel_bilinear":
            return sample_healpix_map_on_image_header(
                healpix_map,
                nside=nside,
                nest=bool(self.nest),
                target_header=target.header,
            )
        if mode == "harmonic_direct":
            return sample_healpix_map_on_image_header_harmonic(
                healpix_map,
                nside=nside,
                nest=bool(self.nest),
                target_header=target.header,
                lmax=lmax,
                fwhm_rad=self.fwhm_rad,
                beam_window=self.beam_window,
                iter=int(self.iter),
            )
        if mode == "healpix_bridge":
            return sample_healpix_map_on_image_header_spectral(
                healpix_map,
                nside=nside,
                nest=bool(self.nest),
                target_header=target.header,
                lmax=lmax,
                target_nside=self.target_nside,
                fwhm_rad=self.fwhm_rad,
                beam_window=self.beam_window,
                iter=int(self.iter),
            )
        raise ValueError(
            "spherical operator mode must be 'pixel_bilinear', 'harmonic_direct', or 'healpix_bridge'"
        )

    def render_to_target(self, target: TargetRenderSpec, **kwargs) -> Tensor:
        data = torch.as_tensor(self.data)
        mode = self._resolved_mode(operators=kwargs.get("operators"))
        if data.ndim == 1:
            return self._project_single_map(data, target, mode=mode)
        if self.band_labels is not None and target.band_label in self.band_labels:
            band_index = self.band_labels.index(target.band_label)
            return self._project_single_map(data[band_index], target, mode=mode)
        return torch.stack(
            [
                self._project_single_map(data[idx], target, mode=mode)
                for idx in range(int(data.shape[0]))
            ],
            dim=0,
        )


@dataclass
class HarmonicAlmScene(SceneRepresentation):
    data: Tensor
    lmax: int | None = None
    mmax: int | None = None
    band_labels: tuple[str | None, ...] | None = None

    def __post_init__(self) -> None:
        data = torch.as_tensor(self.data)
        if data.ndim not in {1, 2}:
            raise ValueError(
                "HarmonicAlmScene data must be a 1D alm vector or 2D band-first stack"
            )
        if self.band_labels is not None:
            if data.ndim != 2:
                raise ValueError(
                    "band_labels require 2D band-first HarmonicAlmScene data"
                )
            if len(self.band_labels) != int(data.shape[0]):
                raise ValueError(
                    "band_labels length must match HarmonicAlmScene band count"
                )

    def _project_single_alm(
        self, alm_values: Tensor, target: TargetRenderSpec
    ) -> Tensor:
        from .healpix_spectral import sample_alm_on_image_header_spectral

        return sample_alm_on_image_header_spectral(
            alm_values,
            target_header=target.header,
            lmax=self.lmax,
            mmax=self.mmax,
        )

    def render_to_target(self, target: TargetRenderSpec, **kwargs) -> Tensor:
        data = torch.as_tensor(self.data)
        if data.ndim == 1:
            return self._project_single_alm(data, target)
        if self.band_labels is not None and target.band_label in self.band_labels:
            band_index = self.band_labels.index(target.band_label)
            return self._project_single_alm(data[band_index], target)
        return torch.stack(
            [
                self._project_single_alm(data[idx], target)
                for idx in range(int(data.shape[0]))
            ],
            dim=0,
        )


@dataclass
class SpinHarmonicAlmScene(SceneRepresentation):
    alm_e: Tensor
    alm_b: Tensor
    spin: int
    lmax: int | None = None
    mmax: int | None = None

    def __post_init__(self) -> None:
        alm_e = torch.as_tensor(self.alm_e)
        alm_b = torch.as_tensor(self.alm_b)
        if alm_e.ndim != 1 or alm_b.ndim != 1:
            raise ValueError(
                "SpinHarmonicAlmScene alm_e and alm_b must be 1D coefficient vectors"
            )
        if int(alm_e.numel()) != int(alm_b.numel()):
            raise ValueError(
                "SpinHarmonicAlmScene alm_e and alm_b must have matching size"
            )

    def render_to_target(self, target: TargetRenderSpec, **kwargs) -> Tensor:
        from .healpix_spectral import sample_spin_alm_on_image_header_spectral

        return sample_spin_alm_on_image_header_spectral(
            (self.alm_e, self.alm_b),
            spin=int(self.spin),
            target_header=target.header,
            lmax=self.lmax,
            mmax=self.mmax,
        )


@dataclass
class ContinuousFieldModel(SceneRepresentation):
    model: nn.Module

    def render_to_target(self, target: TargetRenderSpec, **kwargs) -> Tensor:
        observation = kwargs.get("observation")
        if observation is None:
            observation = target_to_field_observation(
                target,
                device=kwargs.get("device"),
            )
        from .field.rendering import render_field_observation

        return render_field_observation(
            self.model,
            observation,
            num_spectral_samples=int(kwargs.get("num_spectral_samples", 9)),
            operator_bundle=kwargs.get("operator_bundle"),
        )

    def render_distribution_to_target(
        self,
        target: TargetRenderSpec,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        observation = kwargs.get("observation")
        if observation is None:
            observation = target_to_field_observation(
                target,
                device=kwargs.get("device"),
            )
        from .field.rendering import render_field_observation_distribution

        if callable(
            getattr(self.model, "render_bandpass_distribution", None)
        ) or callable(getattr(self.model, "forward_distribution", None)):
            return render_field_observation_distribution(
                self.model,
                observation,
                num_spectral_samples=int(kwargs.get("num_spectral_samples", 9)),
                operator_bundle=kwargs.get("operator_bundle"),
            )
        rendered = self.render_to_target(
            target,
            observation=observation,
            num_spectral_samples=int(kwargs.get("num_spectral_samples", 9)),
        )
        return rendered, torch.zeros_like(rendered)
