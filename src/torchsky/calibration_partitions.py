"""Independent calibration samples within full detector exposures."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from typing import Any


def prepare_exposure_calibration_partitions(payload: dict[str, Any]) -> dict[str, Any]:
    """Copy a manifest and assign connected sample footprints to calibration splits.

    The science ``split`` is unchanged. Closed detector rectangles include one
    pixel of interpolation support beyond the extraction radius. Components
    connect across roles through rectangle overlap or shared source IDs, within
    an exact exposure and band. Hash-ranked components are assigned cyclically
    within each role-membership class; coverage explicitly records any role
    without three independent groups or without all three realized splits.
    """
    result = deepcopy(payload)
    extraction = result["extraction"]
    settings = extraction.get("settings", extraction)
    radii = {}
    for role in ("psf", "background"):
        radius = float(settings[f"{role}_radius_px"])
        if not math.isfinite(radius) or radius < 0:
            raise ValueError(f"invalid {role} extraction radius")
        radii[role] = radius
    exposures: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in result["evidence"]:
        exposure = row.get("exposure_id", row.get("exposure"))
        band, role = row["band"], row["role"]
        if not isinstance(exposure, str) or not exposure:
            raise ValueError("each evidence row needs an exact exposure identifier")
        if not isinstance(band, str) or not band or role not in radii:
            raise ValueError("each evidence row needs a band and psf/background role")
        band = band.upper()
        x, y = float(row["detector_x"]), float(row["detector_y"])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("calibration detector coordinates must be finite")
        radius = radii[role]
        row["calibration_bounds_xy"] = [
            math.floor(x - radius) - 1,
            math.ceil(x + radius) + 1,
            math.floor(y - radius) - 1,
            math.ceil(y + radius) + 1,
        ]
        row["calibration_footprint"] = {
            "coordinate_frame": "zero-indexed detector pixels",
            "bounds_convention": "inclusive xmin,xmax,ymin,ymax",
            "extraction_radius_px": radius,
            "interpolation_margin_px": 1,
            "radius_source": f"extraction.{'settings.' if 'settings' in extraction else ''}{role}_radius_px",
        }
        exposures[exposure, band].append(row)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "assignment": "sha256-ranked components cyclically assigned per role-membership class",
        "coverage": [],
        "supported_exposures": [],
        "excluded_exposures": [],
    }
    splits = ("train", "validation", "test")
    for (exposure, band), rows in sorted(exposures.items()):
        if len(rows) > 128:
            raise ValueError(
                "calibration partitioning supports at most 128 samples per exposure/band"
            )
        parents = list(range(len(rows)))

        def root(index: int) -> int:
            while parents[index] != index:
                index = parents[index]
            return index

        # ponytail: O(n²) grouping is capped at 128 samples; use a spatial index
        # if future extraction increases the sample budget.
        for i, left in enumerate(rows):
            a = left["calibration_bounds_xy"]
            for j in range(i):
                right = rows[j]
                b = right["calibration_bounds_xy"]
                overlap = (
                    a[0] <= b[1] and b[0] <= a[1] and a[2] <= b[3] and b[2] <= a[3]
                )
                same_source = left.get("source_id") not in (None, "") and left.get(
                    "source_id"
                ) == right.get("source_id")
                if overlap or same_source:
                    parents[root(i)] = root(j)
        groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for i, row in enumerate(rows):
            groups[root(i)].append(row)
        classes: dict[tuple[str, ...], list[tuple[str, list[dict[str, Any]]]]] = (
            defaultdict(list)
        )
        for members in groups.values():
            identities = sorted(
                json.dumps(
                    [
                        row["role"],
                        row.get("source_id"),
                        float(row["detector_x"]),
                        float(row["detector_y"]),
                        row["calibration_bounds_xy"],
                    ],
                    sort_keys=True,
                )
                for row in members
            )
            group_id = hashlib.sha256(
                json.dumps([exposure, band, identities]).encode()
            ).hexdigest()
            classes[tuple(sorted({row["role"] for row in members}))].append(
                (group_id, members)
            )
        for components in classes.values():
            for rank, (group_id, members) in enumerate(
                sorted(components, key=lambda item: item[0])
            ):
                for row in members:
                    row["calibration_group_id"] = group_id
                    row["calibration_split"] = splits[rank % 3]
        reasons = []
        for role in radii:
            members = [row for row in rows if row["role"] == role]
            group_count = len({row["calibration_group_id"] for row in members})
            counts = {
                split: sum(row["calibration_split"] == split for row in members)
                for split in splits
            }
            reason = (
                "insufficient_independent_groups"
                if group_count < 3
                else "incomplete_split_coverage"
                if not all(counts.values())
                else None
            )
            summary["coverage"].append(
                {
                    "exposure": exposure,
                    "band": band,
                    "role": role,
                    "sample_count": len(members),
                    "independent_group_count": group_count,
                    "split_counts": counts,
                    "supported": reason is None,
                    "reason": reason,
                }
            )
            if reason:
                reasons.append(f"{role}: {reason}")
        record = {"exposure": exposure, "band": band}
        if reasons:
            summary["excluded_exposures"].append({**record, "reasons": reasons})
        else:
            summary["supported_exposures"].append(record)
    result["calibration_partitioning"] = summary
    return result


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepared = prepare_exposure_calibration_partitions(
        json.loads(args.input.read_text())
    )
    serialized = json.dumps(prepared, indent=2, sort_keys=True) + "\n"
    with args.output.open("x") as output:
        output.write(serialized)
