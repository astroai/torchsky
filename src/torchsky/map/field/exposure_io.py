"""Read one calibrated exposure file into a :class:`FieldObservation` through torchfits.

The reader opens one FITS file with :mod:`torchfits` (never ``astropy.io.fits``), reads the
science, error and data-quality image extensions of one detector chip, builds the validity
mask and inverse-variance weight, and optionally keeps only the rectangle of the detector
that falls inside a box on the sky. The cut is made with the exposure's own astrometry and
read from disk as a region, so a small box on a large detector reads a small part of the file.

Conventions used throughout:

* pixel coordinates are zero-based, ``(x, y)`` = (column, row), pixel centres at integers;
* ``image`` / ``err`` / ``weight`` / ``mask`` are ``(H, W)`` tensors;
* sky positions are ICRS-like longitude / latitude in degrees;
* geometry (the sky-box test) is always evaluated in float64 on the CPU; only the returned
  planes follow ``device`` and ``dtype``.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import torch
import torchfits
from torch import Tensor

from ...device import _to
from ...wcs import WCS
from ..quality import DQPolicy
from .observation_ops import copy_header_with_crop
from .types import DetectorToSkyTransform, FieldObservation

__all__ = ["field_observation_from_exposure_file"]

# Keywords tried in order for the exposure time [s], PRIMARY header first and then the science
# header. EFFEXPTM (JWST effective exposure time) is the duration that converts a count rate
# into counts; XPOSURE is the JWST exposure duration; EXPTIME is the HST (and common) keyword.
_EXPOSURE_TIME_KEYWORDS = ("EFFEXPTM", "XPOSURE", "EXPTIME")
# PUPIL values that mean "no pupil element" and are therefore not part of the band name.
_EMPTY_PUPIL = {"", "CLEAR", "N/A", "NONE"}
_VARIANCE_HDUS = (
    ("var_poisson", "VAR_POISSON"),
    ("var_rnoise", "VAR_RNOISE"),
    ("var_flat", "VAR_FLAT"),
)
_DROPPED_HEADER_KEYS = {"", "COMMENT", "HISTORY"}
# Spacing [pixel] of the coarse grid that finds the region of the detector near the sky box.
_COARSE_STRIDE_PIX = 32
_ARCSEC_PER_RAD = 180.0 * 3600.0 / math.pi
# A unit string that is a rate per solid angle: "/sr", "sr-1", "sr^-1", "/arcsec2", "arcsec-2".
_PER_SOLID_ANGLE = re.compile(
    r"(/\s*sr\b)|(\bsr\s*(\^|\*\*)?\s*-1\b)"
    r"|(/\s*(arcsec|arcmin|deg)\s*(\^|\*\*)?\s*2\b)"
    r"|(\b(arcsec|arcmin|deg)\s*(\^|\*\*)?\s*-2\b)",
    re.IGNORECASE,
)


def field_observation_from_exposure_file(
    path: str | Path,
    *,
    science_hdu: str = "SCI",
    error_hdu: str = "ERR",
    quality_hdu: str | None = "DQ",
    extver: int | None = None,
    dq_policy: DQPolicy | None = None,
    sky_box: tuple[float, float, float] | None = None,
    mask_outside_box: bool = False,
    min_valid_pixels: int = 1,
    read_variance_planes: bool = False,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    detector_to_sky: DetectorToSkyTransform | None = None,
    band_name: str | None = None,
    value_unit: str | None = None,
    value_semantics: str | None = None,
    exposure_time_s: float | None = None,
) -> FieldObservation | None:
    """Read one exposure file (one detector chip) into a :class:`FieldObservation`.

    Reads the ``science_hdu`` image, the ``error_hdu`` image (one-sigma uncertainty, same
    units as the science image) and, unless ``quality_hdu`` is ``None``, the integer
    data-quality image, all with the same ``extver``. A pixel is usable when its science
    value is finite, its error is finite and positive, its inverse variance ``1 / err**2``
    is finite in the working precision, and ``dq_policy`` accepts its data-quality value.
    The weight is ``1 / err**2`` on usable pixels and ``0`` elsewhere; unusable pixels hold
    ``0`` in ``image`` (so that no ``NaN`` reaches a loss) and keep their finite ``err``
    (non-finite ``err`` becomes ``0``).

    With ``sky_box``, only the smallest detector rectangle that contains every pixel centre
    inside the box is read (a region read, not a whole-image read). The box is a square on
    the tangent plane at ``(lon, lat)``: a pixel is inside when its gnomonic offsets
    (east, north) are both within ``half_size`` of the centre. Pixels in the rectangle but
    outside the square keep their own validity unless ``mask_outside_box`` is true. The
    returned header describes the cut (``NAXIS``, ``CRPIX`` shifted by
    :func:`~torchsky.map.field.observation_ops.copy_header_with_crop`) and
    ``detector_x0_pix`` / ``detector_y0_pix`` hold the cut's offset in the full chip, so
    full-detector pixel coordinates are recovered as ``x_det = x_cut + detector_x0_pix``.

    Data-quality bits are never interpreted implicitly. The default policy
    (``DQPolicy()``) rejects any non-zero flag. Instrument bit tables (JWST ``DO_NOT_USE``
    = 1, ``SATURATED`` = 2, ``JUMP_DET`` = 4, ...) are not mapped in
    :class:`~torchsky.quality.QualityBit`, so the caller states the integer bits that
    reject: ``DQPolicy(reject_bits=1)`` rejects only ``DO_NOT_USE`` pixels and keeps
    pixels flagged only with other bits.

    The pixel-area plane (JWST ``AREA``) is not read: :class:`FieldObservation` has no slot
    for it. Slice it with the returned offsets, or evaluate it from the WCS with
    :func:`torchsky.map.operators.pixel_area.pixel_area_sr_map`.

    Args:
        path: FITS file path.
        science_hdu: ``EXTNAME`` of the science image.
        error_hdu: ``EXTNAME`` of the one-sigma error image.
        quality_hdu: ``EXTNAME`` of the integer data-quality image, or ``None`` for none
            (then ``dq_policy`` must be ``None``).
        extver: ``EXTVER`` of the chip. ``None`` accepts a file in which every requested
            name occurs once, and raises when a name occurs for several ``EXTVER`` values
            (multi-chip HST ``_flt`` / ``_flc`` files).
        dq_policy: Which data-quality bits reject a pixel. ``None`` is ``DQPolicy()``.
        sky_box: ``(lon_deg, lat_deg, half_size_arcsec)`` or ``None`` for the whole chip.
        mask_outside_box: With ``sky_box``, also mark pixels of the rectangle that are
            outside the square as unusable.
        min_valid_pixels: The read returns ``None`` when fewer usable pixels remain.
        read_variance_planes: Also read ``VAR_POISSON`` / ``VAR_RNOISE`` / ``VAR_FLAT``
            when the file has them (an absent plane stays ``None``).
        device: Device of the returned planes. ``None`` is CPU.
        dtype: Floating dtype of the returned planes (geometry stays float64).
        detector_to_sky: Native detector-to-sky transform on full-detector zero-based pixel
            coordinates (CPU float64 tensors in, degrees out). It replaces the header WCS
            for the sky-box test and is attached to the result.
        band_name: Band label. ``None`` reads ``FILTER`` (and a non-trivial ``PUPIL`` as
            ``"FILTER-PUPIL"``) from the PRIMARY header, then the science header.
        value_unit: Unit string of the pixel values. ``None`` reads ``BUNIT``.
        value_semantics: ``FieldObservation.value_semantics``. ``None`` gives
            ``"surface_brightness"`` when the unit is per solid angle (``MJy/sr``,
            ``MJy sr-1``, ``/arcsec2``) and stays ``None`` otherwise (count rates per pixel).
        exposure_time_s: Exposure time [s]. ``None`` reads ``EFFEXPTM``, ``XPOSURE`` or
            ``EXPTIME`` (first present and positive) from the PRIMARY header, then the
            science header.

    Returns:
        The observation, or ``None`` when the cut is empty: no pixel centre inside
        ``sky_box``, or fewer than ``min_valid_pixels`` usable pixels in the cut. An empty
        cut is the normal outcome of scanning many files for the ones that overlap a
        region, so it is not an error; every malformed input raises.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: A requested extension is missing or ambiguous, the planes disagree in
            shape, an argument is invalid, or the science image is not 2D.

    Examples:
        >>> import torch, torchfits, tempfile, os
        >>> from torchsky.map.field.exposure_io import field_observation_from_exposure_file
        >>> header = {"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRPIX1": 3.0, "CRPIX2": 3.0,
        ...           "CRVAL1": 150.0, "CRVAL2": 2.0, "CD1_1": -1e-5, "CD2_2": 1e-5, "BUNIT": "MJy/sr"}
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     p = os.path.join(tmp, "x_cal.fits")
        ...     hdus = [torchfits.TensorHDU(data=torch.zeros(1), header={}),
        ...             torchfits.TensorHDU(data=torch.ones(4, 5), header={**header, "EXTNAME": "SCI"}),
        ...             torchfits.TensorHDU(data=torch.full((4, 5), 0.5), header={"EXTNAME": "ERR"}),
        ...             torchfits.TensorHDU(data=torch.zeros(4, 5, dtype=torch.int32), header={"EXTNAME": "DQ"})]
        ...     torchfits.HDUList(hdus).write(p)
        ...     obs = field_observation_from_exposure_file(p)
        >>> tuple(obs.image.shape), float(obs.weight[0, 0]), obs.value_semantics
        ((4, 5), 4.0, 'surface_brightness')
    """
    policy = _checked_arguments(
        quality_hdu=quality_hdu,
        dq_policy=dq_policy,
        sky_box=sky_box,
        min_valid_pixels=min_valid_pixels,
        extver=extver,
        dtype=dtype,
        exposure_time_s=exposure_time_s,
    )
    path_str = str(path)
    if not Path(path_str).is_file():
        raise FileNotFoundError(f"exposure file not found: {path_str}")
    out_device = torch.device("cpu" if device is None else device)

    hdu_science = _resolve_hdu(path_str, science_hdu, extver)
    hdu_error = _resolve_hdu(path_str, error_hdu, extver)
    hdu_quality = (
        None if quality_hdu is None else _resolve_hdu(path_str, quality_hdu, extver)
    )
    hdu_variance: dict[str, int] = {}
    if read_variance_planes:
        for field_name, extname in _VARIANCE_HDUS:
            found = _find_hdu(path_str, extname, extver, required=False)
            if found is not None:
                hdu_variance[field_name] = found

    shape = _plane_shape(path_str, hdu_science, science_hdu)
    planes = {
        error_hdu: hdu_error,
        **({quality_hdu: hdu_quality} if quality_hdu else {}),
    }
    planes.update(
        {name: hdu_variance[key] for key, name in _VARIANCE_HDUS if key in hdu_variance}
    )
    for name, index in planes.items():
        other = _plane_shape(path_str, index, name)
        if other != shape:
            raise ValueError(
                f"{name} plane has shape {other}, but {science_hdu} has shape {shape}"
            )

    header = _clean_header(torchfits.read_header(path_str, hdu_science))
    height, width = shape

    x0, y0, x1, y1 = 0, 0, width, height
    inside: Tensor | None = None
    if sky_box is not None:
        world = _world_function(header, detector_to_sky)
        window = _window_for_sky_box(world, height, width, sky_box)
        if window is None:
            return None
        x0, y0, x1, y1, inside = window

    def region(index: int) -> Tensor:
        if (x0, y0, x1, y1) == (0, 0, width, height):
            return torchfits.read(path_str, hdu=index)
        return torchfits.read_subset(path_str, index, x0, y0, x1, y1)

    image = region(hdu_science)
    err = region(hdu_error)
    for name, plane in (("science", image), ("error", err)):
        if plane.is_complex() or not (
            plane.is_floating_point() or plane.dtype in _INTEGER_DTYPES
        ):
            raise ValueError(f"{name} plane has unsupported dtype {plane.dtype}")
    dq = None if hdu_quality is None else region(hdu_quality)
    variances = {name: region(index) for name, index in hdu_variance.items()}

    compute = torch.float32 if out_device.type == "mps" else torch.float64
    image = _to(image, out_device, compute)
    err = _to(err, out_device, compute)
    usable = torch.isfinite(image) & torch.isfinite(err) & (err > 0)
    safe_err = torch.where(usable, err, torch.ones_like(err))
    weight = torch.where(usable, safe_err.reciprocal().square(), torch.zeros_like(err))
    weight = weight.to(dtype)
    # An error so small that 1 / err**2 overflows the output precision is not usable either.
    usable = usable & torch.isfinite(weight) & (weight > 0)
    if dq is not None:
        assert policy is not None
        usable = usable & policy.usability_mask(dq.to(out_device))
    if mask_outside_box and inside is not None:
        usable = usable & inside.to(out_device)
    weight = torch.where(usable, weight, torch.zeros_like(weight))
    image_out = torch.where(usable, image, torch.zeros_like(image)).to(dtype)
    err_out = torch.where(torch.isfinite(err), err, torch.zeros_like(err)).to(dtype)

    if int(usable.sum()) < int(min_valid_pixels):
        return None

    science_header = header
    primary = _clean_header(torchfits.read_header(path_str, 0))
    unit = value_unit if value_unit is not None else _text(science_header.get("BUNIT"))
    semantics = value_semantics
    if semantics is None and unit is not None and _PER_SOLID_ANGLE.search(unit):
        semantics = "surface_brightness"

    return FieldObservation(
        image=image_out,
        header=copy_header_with_crop(
            science_header, x0=x0, y0=y0, width=x1 - x0, height=y1 - y0
        ),
        weight=weight,
        mask=usable,
        err=err_out,
        var_poisson=_variance(variances.get("var_poisson"), out_device, dtype),
        var_rnoise=_variance(variances.get("var_rnoise"), out_device, dtype),
        var_flat=_variance(variances.get("var_flat"), out_device, dtype),
        band_name=band_name
        if band_name is not None
        else _band_name(primary, science_header),
        detector_x0_pix=float(x0),
        detector_y0_pix=float(y0),
        detector_to_sky=detector_to_sky,
        value_unit=unit,
        value_semantics=semantics,
        exposure_time_s=(
            float(exposure_time_s)
            if exposure_time_s is not None
            else _exposure_time(primary, science_header)
        ),
    )


_INTEGER_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint16,
    torch.uint32,
)


def _checked_arguments(
    *,
    quality_hdu: str | None,
    dq_policy: DQPolicy | None,
    sky_box: tuple[float, float, float] | None,
    min_valid_pixels: int,
    extver: int | None,
    dtype: torch.dtype,
    exposure_time_s: float | None,
) -> DQPolicy | None:
    if not dtype.is_floating_point:
        raise TypeError("dtype must be a floating-point torch dtype")
    if dq_policy is not None and quality_hdu is None:
        raise ValueError("dq_policy needs a quality_hdu to apply to")
    if dq_policy is not None and not isinstance(dq_policy, DQPolicy):
        raise TypeError("dq_policy must be a DQPolicy or None")
    if isinstance(min_valid_pixels, bool) or int(min_valid_pixels) < 1:
        raise ValueError("min_valid_pixels must be an integer >= 1")
    if extver is not None and (isinstance(extver, bool) or int(extver) < 1):
        raise ValueError("extver must be a positive integer or None")
    if exposure_time_s is not None and not (
        math.isfinite(float(exposure_time_s)) and float(exposure_time_s) > 0.0
    ):
        raise ValueError("exposure_time_s must be finite and positive")
    if sky_box is not None:
        if len(sky_box) != 3 or not all(math.isfinite(float(v)) for v in sky_box):
            raise ValueError(
                "sky_box must be three finite numbers (lon, lat, half_size)"
            )
        if not -90.0 <= float(sky_box[1]) <= 90.0:
            raise ValueError("sky_box latitude must lie in [-90, 90] degrees")
        if float(sky_box[2]) <= 0.0:
            raise ValueError("sky_box half_size must be positive arcsec")
    if quality_hdu is None:
        return None
    return dq_policy if dq_policy is not None else DQPolicy()


def _scan_hdus(path: str) -> list[tuple[int, str]]:
    """Extension HDUs as ``(index, EXTNAME upper-case)``; unnamed HDUs are skipped."""
    found: list[tuple[int, str]] = []
    for index in range(1, int(torchfits.read_num_hdus(path))):
        name = torchfits.read_extname(path, index)
        if name:
            found.append((index, str(name).strip().upper()))
    return found


def _find_hdu(
    path: str, name: str, extver: int | None, *, required: bool
) -> int | None:
    key = str(name).strip().upper()
    matches = [
        (index, int(torchfits.read_header(path, index).get("EXTVER", 1)))
        for index, name_found in _scan_hdus(path)
        if name_found == key
    ]
    if extver is not None:
        matches = [(i, v) for i, v in matches if v == int(extver)]
    if not matches:
        if not required:
            return None
        where = "" if extver is None else f" with EXTVER={int(extver)}"
        raise ValueError(f"extension {name!r}{where} not found in {path}")
    if len(matches) > 1:
        listed = ", ".join(str(v) for _, v in matches)
        raise ValueError(
            f"extension {name!r} occurs for several EXTVER values ({listed}) in {path}; "
            "pass extver"
        )
    return matches[0][0]


def _resolve_hdu(path: str, name: str, extver: int | None) -> int:
    index = _find_hdu(path, name, extver, required=True)
    assert index is not None
    return index


def _plane_shape(path: str, index: int, name: str) -> tuple[int, int]:
    _, shape = torchfits.read_shape(path, index)
    if len(shape) != 2:
        raise ValueError(f"{name} must be a 2D image, got shape {tuple(shape)}")
    return int(shape[0]), int(shape[1])


def _clean_header(header: object) -> dict:
    return {
        str(key): value
        for key, value in dict(header).items()  # type: ignore[call-overload]
        if str(key).strip().upper() not in _DROPPED_HEADER_KEYS
    }


def _text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _band_name(primary: dict, science: dict) -> str | None:
    for header in (primary, science):
        filt = _text(header.get("FILTER"))
        if filt is None:
            continue
        pupil = _text(header.get("PUPIL"))
        if pupil is not None and pupil.upper() not in _EMPTY_PUPIL:
            return f"{filt}-{pupil}"
        return filt
    return None


def _exposure_time(primary: dict, science: dict) -> float | None:
    for header in (primary, science):
        for keyword in _EXPOSURE_TIME_KEYWORDS:
            try:
                value = float(header[keyword])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0.0:
                return value
    return None


def _variance(
    plane: Tensor | None, device: torch.device, dtype: torch.dtype
) -> Tensor | None:
    return None if plane is None else _to(plane, device, dtype)


def _world_function(header: dict, detector_to_sky: DetectorToSkyTransform | None):
    """Zero-based full-detector ``(x, y)`` to ``(lon, lat)`` degrees, float64 on the CPU."""
    if detector_to_sky is not None:
        return detector_to_sky.pixel_to_world
    wcs = WCS(header)
    return wcs.pixel_to_world


def _tangent_offsets_arcsec(
    lon_deg: Tensor, lat_deg: Tensor, centre_lon_deg: float, centre_lat_deg: float
) -> tuple[Tensor, Tensor]:
    """Gnomonic (east, north) offsets [arcsec] about a centre; NaN on the far hemisphere."""
    lon = torch.deg2rad(lon_deg.to(torch.float64))
    lat = torch.deg2rad(lat_deg.to(torch.float64))
    lon0 = math.radians(centre_lon_deg)
    lat0 = math.radians(centre_lat_deg)
    dlon = lon - lon0
    cos_c = math.sin(lat0) * torch.sin(lat) + math.cos(lat0) * torch.cos(
        lat
    ) * torch.cos(dlon)
    east = torch.cos(lat) * torch.sin(dlon) / cos_c
    north = (
        math.cos(lat0) * torch.sin(lat)
        - math.sin(lat0) * torch.cos(lat) * torch.cos(dlon)
    ) / cos_c
    far = ~(cos_c > 0)
    nan = torch.full_like(east, float("nan"))
    return (
        torch.where(far, nan, east) * _ARCSEC_PER_RAD,
        torch.where(far, nan, north) * _ARCSEC_PER_RAD,
    )


def _grid_offsets(world, xs: Tensor, ys: Tensor, box: tuple[float, float, float]):
    y, x = torch.meshgrid(ys, xs, indexing="ij")
    with torch.no_grad():
        lon, lat = world(x, y)
    return _tangent_offsets_arcsec(
        torch.as_tensor(lon).detach().cpu(),
        torch.as_tensor(lat).detach().cpu(),
        box[0],
        box[1],
    )


def _window_for_sky_box(
    world,
    height: int,
    width: int,
    box: tuple[float, float, float],
    *,
    stride: int = _COARSE_STRIDE_PIX,
) -> tuple[int, int, int, int, Tensor] | None:
    """Smallest pixel rectangle holding every pixel centre inside the sky box.

    Returns ``(x0, y0, x1, y1, inside)`` with half-open bounds and ``inside`` the boolean
    ``(y1 - y0, x1 - x0)`` box-membership plane, or ``None`` when no pixel centre is inside.

    A coarse grid (every ``stride`` pixels, plus the last row and column) is tested with the
    box inflated by ``stride`` times the largest local pixel scale seen on the grid, which
    bounds how far a pixel can be from its nearest coarse sample; the full-resolution test
    then runs only on the padded bounding rectangle of the candidates. The inflation assumes
    the pixel scale changes smoothly over one coarse cell (true for tangent-plane WCS with
    smooth distortion).
    """
    half = float(box[2])
    xs = torch.unique(
        torch.cat([torch.arange(0, width, stride), torch.tensor([width - 1])])
    ).to(torch.float64)
    ys = torch.unique(
        torch.cat([torch.arange(0, height, stride), torch.tensor([height - 1])])
    ).to(torch.float64)
    east, north = _grid_offsets(world, xs, ys, box)
    scale = _largest_scale(east, north, xs, ys)
    if scale is None:
        return None
    reach = half + float(stride) * scale
    near = (east.abs() <= reach) & (north.abs() <= reach)
    if not bool(near.any()):
        return None
    rows, cols = torch.nonzero(near, as_tuple=True)
    pad = int(stride)
    wx0 = max(int(xs[cols.min()]) - pad, 0)
    wx1 = min(int(xs[cols.max()]) + pad + 1, width)
    wy0 = max(int(ys[rows.min()]) - pad, 0)
    wy1 = min(int(ys[rows.max()]) + pad + 1, height)

    fine_x = torch.arange(wx0, wx1, dtype=torch.float64)
    fine_y = torch.arange(wy0, wy1, dtype=torch.float64)
    east, north = _grid_offsets(world, fine_x, fine_y, box)
    inside = (east.abs() <= half) & (north.abs() <= half)
    if not bool(inside.any()):
        return None
    rows, cols = torch.nonzero(inside, as_tuple=True)
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    return (
        wx0 + c0,
        wy0 + r0,
        wx0 + c1,
        wy0 + r1,
        inside[r0:r1, c0:c1].contiguous(),
    )


def _largest_scale(east: Tensor, north: Tensor, xs: Tensor, ys: Tensor) -> float | None:
    """Largest arcsec-per-pixel step between neighbouring coarse samples (None if none finite)."""
    steps: list[Tensor] = []
    if xs.numel() > 1:
        dx = xs[1:] - xs[:-1]
        steps.append(
            torch.hypot(east[:, 1:] - east[:, :-1], north[:, 1:] - north[:, :-1]) / dx
        )
    if ys.numel() > 1:
        dy = (ys[1:] - ys[:-1]).unsqueeze(1)
        steps.append(torch.hypot(east[1:] - east[:-1], north[1:] - north[:-1]) / dy)
    finite = [s[torch.isfinite(s)] for s in steps]
    finite = [s for s in finite if s.numel()]
    if not finite:
        # A single coarse sample (a 1 x 1 chip): no scale to estimate, nothing to inflate by.
        return 0.0 if bool(torch.isfinite(east).any()) else None
    return float(torch.cat(finite).max())
