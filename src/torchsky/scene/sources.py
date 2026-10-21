"""Known-source catalog scenes with fixed morphology and linear photometry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor

from ..map.field.types import FieldObservation
from ..map.operators.psf import apply_psf_spec, psf_support_radius
from ..map.operators.response import bilinear_response
from ..map.scenes import SceneRepresentation, target_to_field_observation
from ..map.target import TargetRenderSpec
from ..sphere.moc import MOC
from ..wcs import WCS


POINT_MORPHOLOGY = 0
GAUSSIAN_MORPHOLOGY = 1
TEMPLATE_MORPHOLOGY = 2
SERSIC_MORPHOLOGY = 3

_MORPHOLOGY_NAMES = {
    "point": POINT_MORPHOLOGY,
    "gaussian": GAUSSIAN_MORPHOLOGY,
    "template": TEMPLATE_MORPHOLOGY,
    "sersic": SERSIC_MORPHOLOGY,
}


@dataclass(frozen=True)
class SourceStamp:
    source_index: int
    y_slice: slice
    x_slice: slice
    data: Tensor


def _real_rows(value: Any, *, name: str, device: torch.device) -> Tensor:
    tensor = torch.as_tensor(value, device=device)
    if not tensor.is_floating_point() or tensor.is_complex():
        raise TypeError(f"{name} must be a real floating tensor")
    tensor = tensor.reshape(-1)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")
    return tensor


def _integer_rows(value: Any, *, name: str, device: torch.device) -> Tensor:
    tensor = torch.as_tensor(value, device=device)
    if tensor.dtype == torch.bool or tensor.is_floating_point() or tensor.is_complex():
        raise TypeError(f"{name} must be an integer tensor")
    return tensor.to(torch.int64).reshape(-1)


def _normalize_morphology(
    morphology: str | Sequence[str] | Tensor,
    count: int,
    *,
    device: torch.device,
) -> Tensor:
    if isinstance(morphology, str):
        try:
            code = _MORPHOLOGY_NAMES[morphology.strip().lower()]
        except KeyError as exc:
            raise ValueError(f"unsupported morphology {morphology!r}") from exc
        return torch.full((count,), code, dtype=torch.int64, device=device)
    if isinstance(morphology, Tensor):
        codes = _integer_rows(morphology, name="morphology", device=device)
    else:
        try:
            codes = torch.tensor(
                [_MORPHOLOGY_NAMES[str(item).strip().lower()] for item in morphology],
                dtype=torch.int64,
                device=device,
            )
        except KeyError as exc:
            raise ValueError(f"unsupported morphology {exc.args[0]!r}") from exc
    if codes.numel() != count:
        raise ValueError("morphology must match the number of components")
    if bool(((codes < POINT_MORPHOLOGY) | (codes > SERSIC_MORPHOLOGY)).any()):
        raise ValueError(
            "morphology codes must be 0 (point), 1 (gaussian), 2 (template), or 3 (sersic)"
        )
    return codes


def _packed_slices(lengths: Tensor) -> Tensor:
    stops = torch.cumsum(lengths.to(torch.int64), dim=0)
    starts = torch.cat((torch.zeros_like(stops[:1]), stops[:-1]))
    return torch.stack((starts, stops), dim=1)


class SourceCatalogScene(SceneRepresentation):
    """A caller-defined fixed-morphology source catalog with linear coefficients."""

    def __init__(
        self,
        lon: Tensor | None = None,
        lat: Tensor | None = None,
        *,
        positions: Tensor | None = None,
        component_ids: Tensor | None = None,
        morphology: str | Sequence[str] | Tensor = "point",
        gaussian_parameters: Tensor | None = None,
        sersic_parameters: Tensor | None = None,
        templates: Tensor | None = None,
        fixed_templates: Tensor | None = None,
        template_data: Tensor | None = None,
        template_slices: Tensor | None = None,
        template_shapes: Tensor | None = None,
        template_pixel_scale_arcsec: Tensor | float | None = None,
        coefficients: Tensor | None = None,
        spectral_coefficients: Tensor | None = None,
        per_band_coefficients: Tensor | None = None,
        band_labels: Sequence[str] | None = None,
        spectral_basis: Tensor | None = None,
        parameter_slices: Tensor | None = None,
        support_footprints: Sequence[MOC] | None = None,
        support_uniq: Tensor | None = None,
        support_slices: Tensor | None = None,
        support_radius_arcsec: Tensor | float | None = None,
        footprint_order: int = 16,
    ) -> None:
        if positions is not None:
            if lon is not None or lat is not None:
                raise ValueError("positions and lon/lat are mutually exclusive")
            position_tensor = torch.as_tensor(positions)
            if position_tensor.ndim != 2 or position_tensor.shape[1] != 2:
                raise ValueError("positions must have shape (components, 2)")
            lon = position_tensor[:, 0]
            lat = position_tensor[:, 1]
        if lon is None or lat is None:
            raise ValueError("supply either positions or both lon and lat")
        lon_t = torch.as_tensor(lon)
        if not lon_t.is_floating_point() or lon_t.is_complex():
            raise TypeError("lon must be a real floating tensor")
        device = lon_t.device
        self.lon = _real_rows(lon_t, name="lon", device=device)
        self.lat = _real_rows(lat, name="lat", device=device).to(dtype=self.lon.dtype)
        if self.lon.numel() != self.lat.numel():
            raise ValueError("lon and lat must have the same number of components")
        if bool((self.lat.abs() > 90.0).any()):
            raise ValueError("lat must be in [-90, 90] degrees")
        self.lon = torch.remainder(self.lon, 360.0)
        count = int(self.lon.numel())
        self.component_ids = (
            torch.arange(count, dtype=torch.int64, device=device)
            if component_ids is None
            else _integer_rows(component_ids, name="component_ids", device=device)
        )
        if self.component_ids.numel() != count:
            raise ValueError("component_ids must match the number of components")
        if self.component_ids.numel() != torch.unique(self.component_ids).numel():
            raise ValueError("component_ids must be unique")
        self.morphology_codes = _normalize_morphology(morphology, count, device=device)
        self.footprint_order = int(footprint_order)
        if self.footprint_order < 0 or self.footprint_order > 29:
            raise ValueError("footprint_order must be between 0 and 29")

        if gaussian_parameters is None:
            gaussian = self.lon.new_zeros((count, 3))
        else:
            gaussian = torch.as_tensor(
                gaussian_parameters, dtype=self.lon.dtype, device=device
            )
            if gaussian.shape != (count, 3):
                raise ValueError("gaussian_parameters must have shape (components, 3)")
            if not bool(torch.isfinite(gaussian).all()):
                raise ValueError("gaussian_parameters must be finite")
        gaussian_rows = self.morphology_codes == GAUSSIAN_MORPHOLOGY
        if bool(gaussian_rows.any()):
            if bool((gaussian[gaussian_rows, :2] <= 0).any()):
                raise ValueError(
                    "Gaussian major/minor sigma must be positive arcseconds"
                )
            if bool((gaussian[gaussian_rows, 1] > gaussian[gaussian_rows, 0]).any()):
                raise ValueError("Gaussian minor sigma cannot exceed major sigma")
        self.gaussian_parameters = gaussian

        if sersic_parameters is None:
            sersic = self.lon.new_zeros((count, 4))
        else:
            sersic = torch.as_tensor(
                sersic_parameters, dtype=self.lon.dtype, device=device
            )
            if sersic.shape != (count, 4):
                raise ValueError("sersic_parameters must have shape (components, 4)")
            if not bool(torch.isfinite(sersic).all()):
                raise ValueError("sersic_parameters must be finite")
        sersic_rows = self.morphology_codes == SERSIC_MORPHOLOGY
        if bool(sersic_rows.any()):
            selected = sersic[sersic_rows]
            if bool((selected[:, 0] <= 0).any()):
                raise ValueError("Sersic half-light radius must be positive arcseconds")
            if bool(((selected[:, 1] <= 0) | (selected[:, 1] > 1)).any()):
                raise ValueError("Sersic axis ratio must be in (0, 1]")
            if bool(((selected[:, 3] < 0.3) | (selected[:, 3] > 8.0)).any()):
                raise ValueError("Sersic index must be in [0.3, 8]")
        self.sersic_parameters = sersic

        if templates is not None:
            if fixed_templates is not None:
                raise ValueError(
                    "templates and fixed_templates are aliases; supply one"
                )
            fixed_templates = templates
        if fixed_templates is not None:
            if (
                template_data is not None
                or template_slices is not None
                or template_shapes is not None
            ):
                raise ValueError(
                    "fixed_templates and packed template inputs are mutually exclusive"
                )
            dense = torch.as_tensor(
                fixed_templates, dtype=self.lon.dtype, device=device
            )
            if dense.ndim != 3 or int(dense.shape[0]) != count:
                raise ValueError(
                    "fixed_templates must have shape (components, height, width)"
                )
            template_shapes = torch.tensor(
                [[int(dense.shape[1]), int(dense.shape[2])]] * count,
                dtype=torch.int64,
                device=device,
            )
            lengths = template_shapes.prod(dim=1)
            template_slices = _packed_slices(lengths)
            template_data = dense.reshape(-1)
        self.template_data = (
            self.lon.new_empty((0,))
            if template_data is None
            else torch.as_tensor(
                template_data, dtype=self.lon.dtype, device=device
            ).reshape(-1)
        )
        self.template_slices = (
            torch.zeros((count, 2), dtype=torch.int64, device=device)
            if template_slices is None
            else torch.as_tensor(template_slices, dtype=torch.int64, device=device)
        )
        self.template_shapes = (
            torch.zeros((count, 2), dtype=torch.int64, device=device)
            if template_shapes is None
            else torch.as_tensor(template_shapes, dtype=torch.int64, device=device)
        )
        if self.template_slices.shape != (count, 2) or self.template_shapes.shape != (
            count,
            2,
        ):
            raise ValueError(
                "template_slices and template_shapes must have shape (components, 2)"
            )
        if template_pixel_scale_arcsec is None:
            scale = self.lon.new_zeros((count, 2))
        else:
            scale = torch.as_tensor(
                template_pixel_scale_arcsec, dtype=self.lon.dtype, device=device
            )
            if scale.ndim == 0:
                scale = scale.expand(count, 2).clone()
            elif scale.ndim == 1 and scale.numel() == count:
                scale = scale[:, None].expand(count, 2).clone()
            elif scale.shape == (2,):
                scale = scale[None, :].expand(count, 2).clone()
            elif scale.shape != (count, 2):
                raise ValueError(
                    "template_pixel_scale_arcsec must be scalar, (2,), (components,), or (components, 2)"
                )
        self.template_pixel_scale_arcsec = scale
        template_rows = self.morphology_codes == TEMPLATE_MORPHOLOGY
        for index in torch.nonzero(template_rows, as_tuple=False).flatten().tolist():
            start, stop = (int(value) for value in self.template_slices[index].tolist())
            shape = tuple(int(value) for value in self.template_shapes[index].tolist())
            if min(shape) <= 0 or stop - start != math.prod(shape):
                raise ValueError(
                    "each template morphology requires a valid packed slice and shape"
                )
            if start < 0 or stop > self.template_data.numel() or start >= stop:
                raise ValueError("template slice is outside template_data")
            values = self.template_data[start:stop]
            if (
                not bool(torch.isfinite(values).all())
                or bool((values < 0).any())
                or not bool(values.sum() > 0)
            ):
                raise ValueError(
                    "fixed templates must be finite, non-negative, and have positive sum"
                )
            if bool((self.template_pixel_scale_arcsec[index] <= 0).any()):
                raise ValueError("template morphologies require positive pixel scale")

        supplied = [
            value
            for value in (coefficients, spectral_coefficients, per_band_coefficients)
            if value is not None
        ]
        if len(supplied) > 1:
            raise ValueError("coefficient inputs are aliases; supply only one")
        supplied_coefficients = supplied[0] if supplied else None
        self.band_labels = tuple(str(value) for value in (band_labels or ()))
        if len(set(self.band_labels)) != len(self.band_labels):
            raise ValueError("band_labels must be unique")
        basis = (
            None
            if spectral_basis is None
            else torch.as_tensor(spectral_basis, dtype=self.lon.dtype, device=device)
        )
        if basis is not None:
            if basis.ndim != 2 or int(basis.shape[0]) != len(self.band_labels):
                raise ValueError(
                    "spectral_basis must have shape (len(band_labels), basis_size)"
                )
            if not bool(torch.isfinite(basis).all()):
                raise ValueError("spectral_basis must be finite")
        self.spectral_basis = basis
        if supplied_coefficients is None:
            width = (
                int(basis.shape[1])
                if basis is not None
                else max(len(self.band_labels), 1)
            )
            coefficient_matrix = self.lon.new_zeros((count, width))
        else:
            coefficient_matrix = torch.as_tensor(
                supplied_coefficients, dtype=self.lon.dtype, device=device
            )
            if coefficient_matrix.ndim == 1:
                if count == 0:
                    coefficient_matrix = coefficient_matrix.reshape(0, 0)
                elif coefficient_matrix.numel() == count:
                    coefficient_matrix = coefficient_matrix[:, None]
                else:
                    coefficient_matrix = coefficient_matrix.reshape(1, -1)
            if (
                coefficient_matrix.ndim != 2
                or int(coefficient_matrix.shape[0]) != count
            ):
                raise ValueError(
                    "coefficients must have shape (components, coefficient_count)"
                )
            if not bool(torch.isfinite(coefficient_matrix).all()):
                raise ValueError("coefficients must be finite")
        width = int(coefficient_matrix.shape[1])
        if basis is not None and int(basis.shape[1]) != width:
            raise ValueError("coefficient width must match spectral_basis width")
        if basis is None and self.band_labels and width != len(self.band_labels):
            raise ValueError("per-band coefficient width must match band_labels")
        default_slices = torch.stack(
            (
                torch.arange(count, device=device, dtype=torch.int64) * width,
                (torch.arange(count, device=device, dtype=torch.int64) + 1) * width,
            ),
            dim=1,
        )
        self.parameter_slices = (
            default_slices
            if parameter_slices is None
            else torch.as_tensor(parameter_slices, dtype=torch.int64, device=device)
        )
        if self.parameter_slices.shape != (count, 2):
            raise ValueError("parameter_slices must have shape (components, 2)")
        if bool((self.parameter_slices[:, 0] < 0).any()) or bool(
            (self.parameter_slices[:, 1] <= self.parameter_slices[:, 0]).any()
        ):
            raise ValueError(
                "parameter_slices must contain ordered non-negative ranges"
            )
        lengths = self.parameter_slices[:, 1] - self.parameter_slices[:, 0]
        if count and bool((lengths != width).any()):
            raise ValueError("each parameter slice length must match coefficient width")
        self.parameter_count = (
            0 if count == 0 else int(self.parameter_slices[:, 1].max().item())
        )
        parameter_values = self.lon.new_zeros((self.parameter_count,))
        assigned = torch.zeros(self.parameter_count, dtype=torch.bool, device=device)
        for index in range(count):
            start, stop = (
                int(value) for value in self.parameter_slices[index].tolist()
            )
            current = coefficient_matrix[index]
            overlap = assigned[start:stop]
            if bool(overlap.any()) and not bool(
                torch.equal(parameter_values[start:stop][overlap], current[overlap])
            ):
                raise ValueError(
                    "shared parameter slices require identical initial coefficients"
                )
            parameter_values[start:stop] = current
            assigned[start:stop] = True
        self.coefficients = parameter_values

        if support_radius_arcsec is None:
            radius = self.lon.new_zeros(count)
            radius[gaussian_rows] = 4.0 * gaussian[gaussian_rows, 0]
            radius[sersic_rows] = 8.0 * sersic[sersic_rows, 0]
            for index in (
                torch.nonzero(template_rows, as_tuple=False).flatten().tolist()
            ):
                h, w = (int(value) for value in self.template_shapes[index].tolist())
                sy, sx = self.template_pixel_scale_arcsec[index]
                radius[index] = 0.5 * torch.sqrt((h * sy).square() + (w * sx).square())
        else:
            radius = torch.as_tensor(
                support_radius_arcsec, dtype=self.lon.dtype, device=device
            )
            if radius.ndim == 0:
                radius = radius.expand(count).clone()
            else:
                radius = radius.reshape(-1)
            if (
                radius.numel() != count
                or not bool(torch.isfinite(radius).all())
                or bool((radius < 0).any())
            ):
                raise ValueError(
                    "support_radius_arcsec must be finite, non-negative, and match components"
                )
        self.support_radius_arcsec = radius
        if support_footprints is not None and (
            support_uniq is not None or support_slices is not None
        ):
            raise ValueError(
                "support_footprints and packed support inputs are mutually exclusive"
            )
        if support_footprints is not None:
            if len(support_footprints) != count or any(
                not isinstance(item, MOC) for item in support_footprints
            ):
                raise ValueError(
                    "support_footprints must contain one MOC per component"
                )
            lengths = torch.tensor(
                [item.uniq.numel() for item in support_footprints],
                dtype=torch.int64,
                device=device,
            )
            support_slices = _packed_slices(lengths)
            support_uniq = (
                torch.cat([item.uniq.to(device=device) for item in support_footprints])
                if count and int(lengths.sum().item())
                else torch.empty(0, dtype=torch.int64, device=device)
            )
        self.support_uniq = (
            None
            if support_uniq is None
            else torch.as_tensor(
                support_uniq, dtype=torch.int64, device=device
            ).reshape(-1)
        )
        self.support_slices = (
            None
            if support_slices is None
            else torch.as_tensor(support_slices, dtype=torch.int64, device=device)
        )
        if (self.support_uniq is None) != (self.support_slices is None):
            raise ValueError(
                "support_uniq and support_slices must be supplied together"
            )
        if self.support_slices is not None:
            if self.support_slices.shape != (count, 2):
                raise ValueError("support_slices must have shape (components, 2)")
            if (
                bool((self.support_slices[:, 0] < 0).any())
                or bool((self.support_slices[:, 1] < self.support_slices[:, 0]).any())
                or bool((self.support_slices[:, 1] > self.support_uniq.numel()).any())
            ):
                raise ValueError(
                    "support_slices must be ordered ranges inside support_uniq"
                )

    @property
    def device(self) -> torch.device:
        return self.lon.device

    @property
    def dtype(self) -> torch.dtype:
        return self.lon.dtype

    @property
    def component_count(self) -> int:
        return int(self.lon.numel())

    def component_parameters(self, index: int, values: Tensor | None = None) -> Tensor:
        start, stop = (int(value) for value in self.parameter_slices[index].tolist())
        state = self.coefficients if values is None else values
        if state.ndim != 1 or state.numel() != self.parameter_count:
            raise ValueError("parameter values must match scene.parameter_count")
        return state[start:stop]

    def spectral_weights(self, band_name: str | None) -> Tensor:
        width = (
            1
            if self.component_count == 0
            else int((self.parameter_slices[0, 1] - self.parameter_slices[0, 0]).item())
        )
        if self.spectral_basis is not None:
            if band_name not in self.band_labels:
                return self.lon.new_zeros(width)
            return self.spectral_basis[self.band_labels.index(str(band_name))]
        if self.band_labels:
            weights = self.lon.new_zeros(width)
            if band_name in self.band_labels:
                weights[self.band_labels.index(str(band_name))] = 1.0
            return weights
        if width != 1:
            raise ValueError(
                "multi-coefficient scenes require band_labels or spectral_basis"
            )
        return self.lon.new_ones(1)

    def component_flux(
        self,
        index: int,
        band_name: str | None,
        values: Tensor | None = None,
    ) -> Tensor:
        return torch.dot(
            self.component_parameters(index, values), self.spectral_weights(band_name)
        )

    def support_footprint(self, index: int) -> MOC:
        if self.support_uniq is not None and self.support_slices is not None:
            start, stop = (int(value) for value in self.support_slices[index].tolist())
            return MOC(self.support_uniq[start:stop])
        radius_deg = float(self.support_radius_arcsec[index].detach().item()) / 3600.0
        if radius_deg == 0.0:
            return MOC.from_lonlat(
                self.lon[index], self.lat[index], self.footprint_order
            )
        return MOC.from_circle(
            float(self.lon[index].detach().item()),
            float(self.lat[index].detach().item()),
            radius_deg,
            max_order=self.footprint_order,
        )

    def support_footprints(self) -> tuple[MOC, ...]:
        return tuple(
            self.support_footprint(index) for index in range(self.component_count)
        )

    def _template(self, index: int) -> Tensor:
        start, stop = (int(value) for value in self.template_slices[index].tolist())
        height, width = (int(value) for value in self.template_shapes[index].tolist())
        return self.template_data[start:stop].reshape(height, width)

    def _pixel_scale_arcsec(self, wcs: WCS, x: Tensor, y: Tensor) -> float:
        lon0, lat0 = wcs.pixel_to_world(x, y)
        lonx, latx = wcs.pixel_to_world(x + 1.0, y)
        lony, laty = wcs.pixel_to_world(x, y + 1.0)
        cos_lat = torch.cos(torch.deg2rad(lat0)).abs().clamp_min(1.0e-8)
        dx = torch.sqrt(((lonx - lon0) * cos_lat).square() + (latx - lat0).square())
        dy = torch.sqrt(((lony - lon0) * cos_lat).square() + (laty - lat0).square())
        return max(float(torch.minimum(dx, dy).detach().item()) * 3600.0, 1.0e-8)

    def _bounds(
        self, index: int, observation: FieldObservation
    ) -> tuple[int, int, int, int] | None:
        height, width = (int(value) for value in observation.image.shape)
        wcs = WCS(observation.header).to(self.device)
        x, y = wcs.world_to_pixel(self.lon[index], self.lat[index])
        if not bool(torch.isfinite(x) & torch.isfinite(y)):
            return None
        scale = self._pixel_scale_arcsec(wcs, x, y)
        halo_y, halo_x = psf_support_radius(observation.psf_spec)
        intrinsic = float(self.support_radius_arcsec[index].detach().item()) / scale
        radius_x = max(1, int(math.ceil(intrinsic)) + halo_x + 2)
        radius_y = max(1, int(math.ceil(intrinsic)) + halo_y + 2)
        x0 = max(0, int(math.floor(float(x.detach().item()))) - radius_x)
        x1 = min(width, int(math.floor(float(x.detach().item()))) + radius_x + 2)
        y0 = max(0, int(math.floor(float(y.detach().item()))) - radius_y)
        y1 = min(height, int(math.floor(float(y.detach().item()))) + radius_y + 2)
        if x0 >= x1 or y0 >= y1:
            return None
        return y0, y1, x0, x1

    def _tangent_world(
        self, index: int, east_arcsec: Tensor, north_arcsec: Tensor
    ) -> tuple[Tensor, Tensor]:
        cos_lat = torch.cos(torch.deg2rad(self.lat[index])).abs().clamp_min(1.0e-8)
        lon = self.lon[index] + east_arcsec / (3600.0 * cos_lat)
        lat = self.lat[index] + north_arcsec / 3600.0
        return torch.remainder(lon, 360.0), lat

    def _point_stamp(
        self,
        index: int,
        wcs: WCS,
        bounds: tuple[int, int, int, int],
    ) -> Tensor:
        y0, y1, x0, x1 = bounds
        x, y = wcs.world_to_pixel(self.lon[index], self.lat[index])
        response = bilinear_response(
            (y1 - y0, x1 - x0),
            (x - x0).reshape(1),
            (y - y0).reshape(1),
        )
        return response.adjoint(torch.ones(1, dtype=x.dtype, device=x.device))

    def _template_stamp(
        self,
        index: int,
        wcs: WCS,
        bounds: tuple[int, int, int, int],
        *,
        chunk_size: int | None,
    ) -> Tensor:
        y0, y1, x0, x1 = bounds
        template = self._template(index)
        height, width = template.shape
        sy, sx = self.template_pixel_scale_arcsec[index]
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=self.dtype, device=self.device),
            torch.arange(width, dtype=self.dtype, device=self.device),
            indexing="ij",
        )
        east = (xx - 0.5 * (width - 1)) * sx
        north = (yy - 0.5 * (height - 1)) * sy
        lon, lat = self._tangent_world(index, east, north)
        x, y = wcs.world_to_pixel(lon, lat)
        response = bilinear_response(
            (y1 - y0, x1 - x0),
            x - x0,
            y - y0,
        )
        normalized = (template / template.sum()).to(dtype=x.dtype)
        return response.adjoint(normalized, chunk_size=chunk_size)

    def _gaussian_stamp(
        self,
        index: int,
        wcs: WCS,
        bounds: tuple[int, int, int, int],
    ) -> Tensor:
        y0, y1, x0, x1 = bounds
        x, y = wcs.world_to_pixel(self.lon[index], self.lat[index])
        one = self.lon.new_tensor(1.0)
        lon_e, lat_e = self._tangent_world(index, one, self.lon.new_zeros(()))
        lon_n, lat_n = self._tangent_world(index, self.lon.new_zeros(()), one)
        x_e, y_e = wcs.world_to_pixel(lon_e, lat_e)
        x_n, y_n = wcs.world_to_pixel(lon_n, lat_n)
        jacobian = torch.stack(
            (torch.stack((x_e - x, x_n - x)), torch.stack((y_e - y, y_n - y)))
        )
        major, minor, angle_deg = self.gaussian_parameters[index]
        angle = torch.deg2rad(angle_deg)
        rotation = torch.stack(
            (
                torch.stack((torch.sin(angle), -torch.cos(angle))),
                torch.stack((torch.cos(angle), torch.sin(angle))),
            )
        )
        covariance_sky = (
            rotation
            @ torch.diag(torch.stack((major.square(), minor.square())))
            @ rotation.T
        )
        covariance = jacobian @ covariance_sky @ jacobian.T
        inverse = torch.linalg.inv(covariance)
        determinant = torch.linalg.det(covariance).clamp_min(
            torch.finfo(self.dtype).tiny
        )
        yy, xx = torch.meshgrid(
            torch.arange(y0, y1, dtype=self.dtype, device=self.device),
            torch.arange(x0, x1, dtype=self.dtype, device=self.device),
            indexing="ij",
        )
        offsets = torch.stack((xx - x, yy - y), dim=-1)
        exponent = torch.einsum("...i,ij,...j->...", offsets, inverse, offsets)
        return torch.exp(-0.5 * exponent) / (2.0 * torch.pi * torch.sqrt(determinant))

    def _sersic_stamp(
        self,
        index: int,
        wcs: WCS,
        bounds: tuple[int, int, int, int],
    ) -> Tensor:
        y0, y1, x0, x1 = bounds
        x, y = wcs.world_to_pixel(self.lon[index], self.lat[index])
        one = self.lon.new_tensor(1.0)
        lon_e, lat_e = self._tangent_world(index, one, self.lon.new_zeros(()))
        lon_n, lat_n = self._tangent_world(index, self.lon.new_zeros(()), one)
        x_e, y_e = wcs.world_to_pixel(lon_e, lat_e)
        x_n, y_n = wcs.world_to_pixel(lon_n, lat_n)
        sky_from_pixel = torch.linalg.inv(
            torch.stack(
                (
                    torch.stack((x_e - x, x_n - x)),
                    torch.stack((y_e - y, y_n - y)),
                )
            )
        )
        yy, xx = torch.meshgrid(
            torch.arange(y0, y1, dtype=self.dtype, device=self.device),
            torch.arange(x0, x1, dtype=self.dtype, device=self.device),
            indexing="ij",
        )
        offsets = torch.stack((xx - x, yy - y), dim=-1)
        tangent = torch.einsum("ij,...j->...i", sky_from_pixel, offsets)
        radius, axis_ratio, angle_deg, index_n = self.sersic_parameters[index]
        angle = torch.deg2rad(angle_deg)
        major = tangent[..., 0] * torch.sin(angle) + tangent[..., 1] * torch.cos(angle)
        minor = -tangent[..., 0] * torch.cos(angle) + tangent[..., 1] * torch.sin(angle)
        elliptical_radius = torch.sqrt(major.square() + (minor / axis_ratio).square())
        bn = 2.0 * index_n - 1.0 / 3.0 + 4.0 / (405.0 * index_n)
        profile = torch.exp(-bn * ((elliptical_radius / radius).clamp_min(0).pow(1.0 / index_n) - 1.0))
        return profile / profile.sum().clamp_min(torch.finfo(profile.dtype).tiny)

    def render_component(
        self,
        index: int,
        observation: FieldObservation,
        *,
        chunk_size: int | None = None,
    ) -> SourceStamp | None:
        if index < 0 or index >= self.component_count:
            raise IndexError("component index is out of range")
        if observation.image.device != self.device:
            raise ValueError("scene and observation must share a device")
        bounds = self._bounds(index, observation)
        if bounds is None:
            return None
        wcs = WCS(observation.header).to(self.device)
        code = int(self.morphology_codes[index].item())
        if code == POINT_MORPHOLOGY:
            data = self._point_stamp(index, wcs, bounds)
        elif code == GAUSSIAN_MORPHOLOGY:
            data = self._gaussian_stamp(index, wcs, bounds)
        elif code == SERSIC_MORPHOLOGY:
            data = self._sersic_stamp(index, wcs, bounds)
        else:
            data = self._template_stamp(index, wcs, bounds, chunk_size=chunk_size)
        data = apply_psf_spec(data, observation.psf_spec)
        y0, y1, x0, x1 = bounds
        return SourceStamp(index, slice(y0, y1), slice(x0, x1), data)

    def render_observation(
        self,
        observation: FieldObservation,
        *,
        parameter_values: Tensor | None = None,
        source_indices: Tensor | Sequence[int] | None = None,
        chunk_size: int | None = None,
    ) -> Tensor:
        output = torch.zeros_like(
            observation.image, dtype=self.dtype, device=self.device
        )
        indices = (
            range(self.component_count)
            if source_indices is None
            else torch.as_tensor(source_indices, dtype=torch.int64, device=self.device)
            .reshape(-1)
            .tolist()
        )
        for index in indices:
            flux = self.component_flux(index, observation.band_name, parameter_values)
            stamp = self.render_component(index, observation, chunk_size=chunk_size)
            if stamp is not None:
                output[stamp.y_slice, stamp.x_slice] = (
                    output[stamp.y_slice, stamp.x_slice] + flux * stamp.data
                )
        return output.to(dtype=observation.image.dtype)

    def render_to_target(self, target: TargetRenderSpec, **kwargs: Any) -> Tensor:
        observation = kwargs.get("observation")
        if observation is None:
            observation = target_to_field_observation(target, device=self.device)
        return self.render_observation(
            observation,
            parameter_values=kwargs.get("parameter_values"),
            source_indices=kwargs.get("source_indices"),
            chunk_size=kwargs.get("chunk_size"),
        )

    def tensor_catalog(
        self, parameter_values: Tensor | None = None
    ) -> Mapping[str, Tensor]:
        values = self.coefficients if parameter_values is None else parameter_values
        rows = [
            self.component_parameters(index, values)
            for index in range(self.component_count)
        ]
        coefficients = torch.stack(rows) if rows else self.lon.new_empty((0, 0))
        return {
            "component_id": self.component_ids,
            "lon": self.lon,
            "lat": self.lat,
            "morphology_code": self.morphology_codes,
            "coefficients": coefficients,
            "parameter_slices": self.parameter_slices,
        }


__all__ = [
    "GAUSSIAN_MORPHOLOGY",
    "POINT_MORPHOLOGY",
    "SERSIC_MORPHOLOGY",
    "SourceCatalogScene",
    "SourceStamp",
    "TEMPLATE_MORPHOLOGY",
]
