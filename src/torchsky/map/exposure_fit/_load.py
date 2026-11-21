"""``load_exposures``: calibrated exposure files into ``FieldObservation`` objects (the only function of this package that opens files).

Files are read through :mod:`torchfits` by :func:`torchsky.map.field.exposure_io.field_observation_from_exposure_file`; this module adds what
the exposure fit needs on top: the instrument's data-quality policy, the conversion of HST counts to MJy/sr, the HST a priori astrometric
solution, the filter curve, and the bookkeeping header cards listed in ``_observations.py``.
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import torch
import torchfits

from ...instrument import InstrumentDescription, instrument_description
from ...wcs import WCS
from ..field.exposure_io import (
    ExposureFileData,
    _resolve_hdu,
    field_observation_from_exposure_file,
)
from ..field.types import FieldObservation
from ..quality import DQPolicy

__all__ = ["load_exposures"]

_DROP = {"", "COMMENT", "HISTORY"}


def _identify(
    primary: dict, descs: Sequence[InstrumentDescription]
) -> InstrumentDescription:
    """The description among ``descs`` that matches the INSTRUME / DETECTOR cards of the primary header."""
    ins = str(primary.get("INSTRUME", "")).strip().upper()
    det = str(primary.get("DETECTOR", "")).strip().upper()
    key = {
        "NIRCAM": "NIRCAM",
        "WFC3": "WFC3_IR" if det == "IR" else "",
        "ACS": "ACS_WFC" if det == "WFC" else "",
    }.get(ins, "")
    for d in descs:
        if d.name == key:
            return d
    raise ValueError(
        f"no instrument description for INSTRUME={ins!r} DETECTOR={det!r}; given: {[d.name for d in descs]}"
    )


def apriori_header(header: dict) -> dict:
    """Header whose CRVAL, CRPIX and CD cards are those of the a priori alternate solution (guide-star catalogue), as the JWST ``_cal`` files have.

    The alternate solution is the one whose ``WCSNAME<letter>`` starts with ``IDC_``, contains ``-GSC`` and not ``FIT`` (else the plain
    ``IDC_x``); a solution fitted to other images is never used. It must share the distortion model with the primary solution.
    Raises ``ValueError`` when there is none (fail closed).
    """
    alts = {
        k[-1]: str(header[k]) for k in header if k.startswith("WCSNAME") and len(k) == 8
    }
    gsc = [k for k, v in sorted(alts.items()) if "-GSC" in v and "FIT" not in v]
    plain = [
        k for k, v in sorted(alts.items()) if v.startswith("IDC_") and "-" not in v
    ]
    key = (gsc or plain or [None])[0]
    if key is None:
        raise ValueError(
            f"no a priori alternate solution in the header (WCSNAME {header.get('WCSNAME')!r})"
        )
    if str(header.get("WCSNAME", "")).split("-")[0] != alts[key].split("-")[0]:
        raise ValueError(
            "the a priori and primary solutions use different distortion models"
        )
    out = dict(header)
    for name in (
        "CRVAL1",
        "CRVAL2",
        "CRPIX1",
        "CRPIX2",
        "CD1_1",
        "CD1_2",
        "CD2_1",
        "CD2_2",
    ):
        if name + key not in header:
            raise ValueError(f"a priori solution {key}: card {name}{key} is missing")
        out[name] = header[name + key]
    out["WCSNAME"] = alts[key]
    return out


class _HeaderWorld:
    """Detector-to-sky transform (full-detector, zero-based pixels; degrees) from a header solution."""

    def __init__(self, header: dict):
        self._wcs = WCS(header)

    def pixel_to_world(self, x: torch.Tensor, y: torch.Tensor):
        return self._wcs.pixel_to_world(x, y)


def _curve(desc: InstrumentDescription, band: str):
    if desc.filter_curves is None:
        return None
    return desc.filter_curves.get(str(band).strip().upper())


def load_exposures(
    paths: Iterable[str | os.PathLike],
    instruments: InstrumentDescription | str | Sequence[InstrumentDescription | str],
    sky_box: tuple[float, float, float],
    *,
    dq_policy: DQPolicy | None = None,
) -> list[FieldObservation]:
    """Read calibrated exposure files into observations cut to a box on the sky. This is the only function here that opens files.

    One observation is returned per (file, chip) that has usable pixels in the box; a file that does not reach the box, or whose cut has
    fewer than ``min_valid_pixels`` usable pixels, gives none. Values and errors are converted to MJy/sr (JWST ``_cal`` files already are;
    HST counts use ``InstrumentDescription.sky_factor``), and an HST file's astrometry is its a priori (guide-star) solution.

    Args:
        paths: FITS files of the instruments named by ``instruments`` (JWST ``_cal``, HST ``_flt`` / ``_flc``).
        instruments: one description or name, or a sequence of them; each file is matched by its ``INSTRUME`` and ``DETECTOR`` cards. Give
            descriptions made with ``with_filter_curves`` to attach filter curves (needed by ``fit_sky``; nothing is fetched).
        sky_box: ``(lon_deg, lat_deg, half_size_arcsec)``: the square data region on the tangent plane at ``(lon, lat)``. Pixels outside the
            square are masked; the cut is the smallest rectangle that holds the square.
        dq_policy: data-quality policy for every file; ``None`` uses each instrument's own (``InstrumentDescription.dq_policy()``).

    Returns:
        ``FieldObservation`` objects (float32, CPU) with ``image`` and ``err`` in MJy/sr, ``mask`` of usable pixels inside the box,
        ``band_name``, the cropped header, and the bookkeeping cards of the guide (``TSINST``, ``TSNAME``, ``TSSPLIT``, ``TSBOXRA``,
        ``TSBOXDE``, ``TSBOXHS``, ``TSDETNX``, ``TSDETNY``, ``TSTOSKY``, ``INSTRUME``, ``DETECTOR``).

    Raises:
        FileNotFoundError: a path does not exist. ValueError: a file matches no given instrument, or has an unknown unit.
    """
    descs = (
        [instruments]
        if isinstance(instruments, (str, InstrumentDescription))
        else list(instruments)
    )
    descs = [
        d if isinstance(d, InstrumentDescription) else instrument_description(d)
        for d in descs
    ]
    if not descs:
        raise ValueError("load_exposures needs at least one instrument")
    ra, dec, half = (float(v) for v in sky_box)
    out: list[FieldObservation] = []
    for path in (os.fspath(p) for p in paths):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"exposure file not found: {path}")
        primary = {
            k: v for k, v in torchfits.read_header(path, 0).items() if k not in _DROP
        }
        desc = _identify(primary, descs)
        policy = dq_policy if dq_policy is not None else desc.dq_policy()
        root = os.path.basename(path)
        for suffix in ("_cal.fits", "_flt.fits", "_flc.fits", ".fits"):
            if root.endswith(suffix):
                root = root[: -len(suffix)]
                break
        for ver in desc.chip_extvers:
            hdu = _resolve_hdu(
                path, desc.science_hdu, ver if len(desc.chip_extvers) > 1 else None
            )
            raw = {
                k: v
                for k, v in torchfits.read_header(path, hdu).items()
                if k not in _DROP
            }
            kwargs = {}
            if desc.wcs_policy == "apriori":
                raw = apriori_header(raw)
                kwargs["detector_to_sky"] = _HeaderWorld(raw)
            factor, how = desc.sky_factor(primary, raw)
            res = field_observation_from_exposure_file(
                path,
                science_hdu=desc.science_hdu,
                error_hdu=desc.error_hdu,
                quality_hdu=desc.quality_hdu,
                extver=ver if len(desc.chip_extvers) > 1 else None,
                dq_policy=policy,
                sky_box=(ra, dec, half),
                mask_outside_box=True,
                min_valid_pixels=desc.min_valid_pixels,
                dtype=torch.float32,
                with_extras=True,
                **kwargs,
            )
            if res is None:
                continue
            assert isinstance(res, ExposureFileData)
            obs = res.observation
            if desc.wcs_policy == "apriori":
                x0, y0 = obs.detector_x0_pix, obs.detector_y0_pix
                h = dict(raw)
                h["NAXIS1"], h["NAXIS2"] = obs.image.shape[1], obs.image.shape[0]
                h["CRPIX1"], h["CRPIX2"] = (
                    float(raw["CRPIX1"]) - x0,
                    float(raw["CRPIX2"]) - y0,
                )
                obs.header = h
                obs.clear_coordinate_cache()
            if (
                factor != 1.0
            ):  # float32 arithmetic with a float32 factor, as the prototype did
                f32 = torch.tensor(factor, dtype=torch.float32)
                # field_observation_from_exposure_file always sets err, mask and weight (error_hdu is required)
                assert (
                    obs.err is not None
                    and obs.mask is not None
                    and obs.weight is not None
                )
                obs.image = obs.image * f32
                obs.err = obs.err * f32
                obs.weight = torch.where(
                    obs.mask,
                    (obs.err.double().reciprocal().square())
                    .clamp(max=torch.finfo(torch.float32).max)
                    .float(),
                    torch.zeros_like(obs.weight),
                )
            obs.value_unit, obs.value_semantics = "MJy/sr", "surface_brightness"
            det = str(res.metadata.get("DETECTOR", desc.name)).strip().upper()
            if desc.name != "NIRCAM":
                chip = obs.header.get("CCDCHIP")
                det = (
                    desc.detectors[0]
                    if len(desc.detectors) == 1
                    else f"{desc.name}{int(chip) if chip is not None else ver}"
                )
            suffix = "" if len(desc.chip_extvers) == 1 else f"_sci{ver}"
            hdr = obs.header
            hdr.update(
                {
                    "TSINST": desc.name,
                    "TSNAME": root + suffix,
                    "TSSPLIT": "train",
                    "TSBOXRA": ra,
                    "TSBOXDE": dec,
                    "TSBOXHS": half,
                    "TSTOSKY": factor,
                    "INSTRUME": res.metadata.get("INSTRUME", ""),
                    "DETECTOR": det,
                }
            )
            nx, ny = _full_shape(path, hdu)
            hdr["TSDETNX"], hdr["TSDETNY"] = nx, ny
            curve = _curve(desc, obs.band_name or "")
            if curve is not None:
                obs.transmission_wavelength, obs.transmission_throughput = (
                    curve[0].double(),
                    curve[1].double(),
                )
                obs.spectral_unit = "um"
            out.append(obs)
    return out


def _full_shape(path: str, hdu: int) -> tuple[int, int]:
    """(NAXIS1, NAXIS2) of the full science plane, from its header."""
    h = torchfits.read_header(path, hdu)
    return int(h["NAXIS1"]), int(h["NAXIS2"])
