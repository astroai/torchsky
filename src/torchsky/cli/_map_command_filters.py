"""Per-filter grouping and field-run helpers for ``torchsky map``."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


from ._map_command_inputs import (
    _header_filter_keyword,
    _map_read_device_str,
    _probe_image_hdus,
)


def _run_field(args: argparse.Namespace) -> int:
    inputs = _core_mod._expand_inputs(args.inputs)
    if not inputs:
        print("torchsky map: no input files matched", file=sys.stderr)
        return 2
    read_dev = _map_read_device_str(args)
    if args.per_filter:
        groups = _group_inputs_by_filter(args, inputs, read_device=read_dev)
        rc = 0
        for fk, indices in sorted(groups.items()):
            gpaths = [inputs[i] for i in indices]
            if fk == "UNKNOWN" and args.field_band is None:
                print(
                    f"torchsky map field: skip UNKNOWN filter group ({len(gpaths)} files); "
                    "set --field-band",
                    file=sys.stderr,
                )
                continue
            band = str(args.field_band).upper() if fk == "UNKNOWN" else str(fk).upper()
            sub_out = _filter_output_path(args.output, fk)
            sub_out.parent.mkdir(parents=True, exist_ok=True)
            ns = argparse.Namespace(**vars(args))
            ns.output = str(sub_out)
            ns.inputs = gpaths
            ns.weights = [args.weights[i] for i in indices]
            ns.background_offsets = [args.background_offsets[i] for i in indices]
            ns.weight_maps = [args.weight_maps[i] for i in indices]
            ns.mask_maps = [args.mask_maps[i] for i in indices]
            ns.primary_beam_maps = [args.primary_beam_maps[i] for i in indices]
            ns.per_filter = False
            ns.field_band = band
            if getattr(args, "plan_json", None) is not None:
                ns.plan_json = _per_filter_plan_path(Path(args.plan_json), fk)
            # Every band group runs even after an earlier failure; the first
            # non-zero status is reported.
            rc = _core_mod._run_field_single(ns) or rc
        return rc
    return _core_mod._run_field_single(args)


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s) or "band"


def _parse_hdu_spec(value: Any) -> Any:
    """Accept an integer HDU index spelled as a decimal string."""
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
    return value


def _per_filter_plan_path(path: Path, band: str) -> Path:
    suffix = path.suffix or ".json"
    return path.with_name(f"{path.stem}_{_safe_name(band)}{suffix}")


def _filter_output_path(output: str, band: str) -> Path:
    """Resolve the per-filter output path for one band."""
    out_path = Path(output)
    if out_path.suffix == "" or out_path.is_dir():
        return out_path / f"{_safe_name(band)}.fits"
    return out_path.parent / f"{out_path.stem}_{_safe_name(band)}{out_path.suffix}"


def _group_inputs_by_filter(
    args: argparse.Namespace,
    inputs: Sequence[str],
    *,
    read_device: str,
) -> dict[str, list[int]]:
    """Group expanded input rows by FITS filter keyword.

    Returns a mapping from filter name to the indices of the input files that
    contribute to that band. A multi-extension file whose HDUs disagree on the
    filter cannot be grouped and fails closed.
    """
    hdu = int(getattr(args, "hdu", 0))
    groups: dict[str, list[int]] = defaultdict(list)
    bands_by_input: dict[int, set[str]] = defaultdict(set)
    bunits: dict[int, str] = {}
    for input_index, path, image_hdu in _image_hdu_expansion(inputs, hdu):
        header = dict(_core_mod.read_header(str(path), image_hdu))
        band = _header_filter_keyword(header) or "UNKNOWN"
        groups[band].append(input_index)
        bands_by_input[input_index].add(band)
        raw_unit = header.get("BUNIT")
        if isinstance(raw_unit, str) and raw_unit.strip():
            bunits[input_index] = raw_unit.strip()
    mixed = sorted(
        str(inputs[index]) for index, bands in bands_by_input.items() if len(bands) > 1
    )
    if mixed:
        raise SystemExit(
            "--per-filter cannot group a multi-extension file whose HDUs carry "
            f"different filters: {mixed}; split it first"
        )
    distinct_units = sorted(set(bunits.values()))
    if len(distinct_units) > 1:
        detail = ", ".join(
            f"{Path(inputs[index]).name}: {unit}"
            for index, unit in sorted(bunits.items())
        )
        raise SystemExit(
            "torchsky map: inputs declare different BUNIT units "
            f"({detail}); convert them to one unit before combining"
        )
    return {band: sorted(set(indices)) for band, indices in groups.items()}


def _reject_mixed_filters(
    args: argparse.Namespace, inputs: Sequence[str], *, read_device: str
) -> str | None:
    """Refuse to stack different filters into one science plane."""
    groups = _group_inputs_by_filter(args, inputs, read_device=read_device)
    if len(groups) > 1:
        bands = ", ".join(sorted(groups))
        raise SystemExit(
            f"torchsky map: inputs span multiple filters ({bands}); pass "
            "--per-filter to write one product per filter"
        )
    return next(iter(groups), None)


def _run_coadd_per_filter(
    args: argparse.Namespace, inputs: Sequence[str], *, read_device: str
) -> int:
    """Coadd each FITS filter into its own product."""
    groups = _group_inputs_by_filter(args, inputs, read_device=read_device)
    if not groups:
        print("torchsky map: no input files matched", file=sys.stderr)
        return 2
    rc = 0
    for band, indices in sorted(groups.items()):
        sub = argparse.Namespace(**vars(args))
        sub.inputs = [inputs[i] for i in indices]
        sub.weights = [args.weights[i] for i in indices]
        sub.photometric_scales = [args.photometric_scales[i] for i in indices]
        sub.background_offsets = [args.background_offsets[i] for i in indices]
        sub.weight_maps = [args.weight_maps[i] for i in indices]
        sub.mask_maps = [args.mask_maps[i] for i in indices]
        sub.primary_beam_maps = [args.primary_beam_maps[i] for i in indices]
        sub.dq_maps = [args.dq_maps[i] for i in indices]
        sub.per_filter = False
        sub._resolved_plan_inputs = None
        sub._coadd_band = None if band == "UNKNOWN" else band.upper()
        sub.output = str(_filter_output_path(args.output, band))
        Path(sub.output).parent.mkdir(parents=True, exist_ok=True)
        if getattr(args, "plan_json", None) is not None:
            sub.plan_json = _per_filter_plan_path(Path(args.plan_json), band)
        print(
            f"torchsky map: filter {band} -> {sub.output} "
            f"({len(indices)} input file(s))",
            file=sys.stderr,
        )
        rc = _core_mod._run_coadd(sub) or rc
    return rc


def _image_hdu_expansion(
    paths: Sequence[str], hdu: int, *, announce: bool = False
) -> list[tuple[int, str, int]]:
    """Expand multi-extension files into ``(input_index, path, hdu)`` rows.

    One policy for every map path: an unreadable or image-less file is passed
    through so the reader reports the real error, a single-image file uses that
    image, an explicit ``--hdu`` that exists wins, and an MEF otherwise
    contributes every image HDU.
    """
    rows: list[tuple[int, str, int]] = []
    for i, path in enumerate(paths):
        image_hdus = _probe_image_hdus(path)
        if len(image_hdus) == 0:
            rows.append((i, path, hdu))
        elif len(image_hdus) == 1:
            rows.append((i, path, hdu if hdu in image_hdus else image_hdus[0]))
        elif hdu != 0 and hdu in image_hdus:
            rows.append((i, path, hdu))
        else:
            rows.extend((i, path, h) for h in image_hdus)
            if announce:
                print(
                    f"  {path}: auto-expanded {len(image_hdus)} image HDUs "
                    f"{list(image_hdus)}",
                    file=sys.stderr,
                )
    return rows


def _expand_field_inputs(
    paths: list[str],
    hdu: int,
    weights: list[float],
    background_offsets: list[float],
    weight_maps: list[str | None],
    mask_maps: list[str | None],
) -> tuple[
    list[str], list[int], list[float], list[float], list[str | None], list[str | None]
]:
    """Expand MEF files into per-HDU entries for field processing."""
    rows = _image_hdu_expansion(paths, hdu)
    return (
        [path for _, path, _ in rows],
        [hdu_i for _, _, hdu_i in rows],
        [float(weights[i]) for i, _, _ in rows],
        [float(background_offsets[i]) for i, _, _ in rows],
        [weight_maps[i] for i, _, _ in rows],
        [mask_maps[i] for i, _, _ in rows],
    )


# Late-bound so monkeypatching ``map_command`` attributes still reaches this code.
from . import map_command as _core_mod  # noqa: E402
