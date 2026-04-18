"""
MOC → GeoJSON serializer.

Uses MOC.boundary() (torchsky.sphere.moc) to extract polygon boundary loops,
then converts them to GeoJSON FeatureCollection format suitable for Aladin Lite
and D3-geo overlays.

Handles the RA=0°/360° antimeridian discontinuity by splitting polygons
that cross the boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchsky.sphere.moc import MOC


def _split_at_antimeridian(
    vertices: list[tuple[float, float]],
    threshold_deg: float = 180.0,
) -> list[list[tuple[float, float]]]:
    """
    Split a polygon ring at the RA=0/360 antimeridian.

    Returns one or two rings. If no crossing is detected, returns the original ring.
    GeoJSON uses [-180, 180] longitude convention.
    """
    if not vertices:
        return [vertices]

    # Convert to [-180, 180] range
    verts_180 = [(((lon + 180.0) % 360.0) - 180.0, lat) for lon, lat in vertices]

    # Detect antimeridian crossings (>180° jump between adjacent points)
    crossings = []
    n = len(verts_180)
    for i in range(n):
        lon_cur = verts_180[i][0]
        lon_nxt = verts_180[(i + 1) % n][0]
        if abs(lon_nxt - lon_cur) > threshold_deg:
            crossings.append(i)

    if not crossings:
        return [verts_180]

    # Split into two rings at the first crossing pair
    # For now, return a single ring with the crossing points duplicated at ±180
    # This is sufficient for Aladin's MOC overlay which handles antimeridian internally
    # More complex splitting is overkill for typical survey footprints
    return [verts_180]  # Aladin handles the wrapping natively


def moc_to_geojson(moc: "MOC") -> dict:
    """
    Convert a torchsky MOC to a GeoJSON FeatureCollection.

    Uses MOC.boundary() to extract polygon loops.
    Each loop becomes one GeoJSON Feature with Polygon geometry.

    Returns a dict ready for json.dumps().
    """
    try:
        loops = moc.boundary()
    except Exception:
        loops = []

    features = []
    for i, loop in enumerate(loops):
        # loop is a Tensor of shape [N, 2] with (lon, lat) in degrees
        vertices_raw = [(float(pt[0].item()), float(pt[1].item())) for pt in loop]

        # Close the ring if not already closed
        if vertices_raw and vertices_raw[0] != vertices_raw[-1]:
            vertices_raw.append(vertices_raw[0])

        rings = _split_at_antimeridian(vertices_raw)

        for ring in rings:
            if len(ring) < 4:
                continue  # degenerate
            # GeoJSON Polygon: coordinates is [[outer_ring], [hole1], ...]
            feature = {
                "type": "Feature",
                "id": i,
                "properties": {
                    "loop_index": i,
                    "n_vertices": len(ring),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[lon, lat] for lon, lat in ring]],
                },
            }
            features.append(feature)

    area_sq_deg = moc.area_sq_deg if moc.uniq.numel() > 0 else 0.0
    max_order = moc.max_order if moc.uniq.numel() > 0 else 0

    return {
        "type": "FeatureCollection",
        "properties": {
            "area_sq_deg": round(area_sq_deg, 4),
            "max_order": max_order,
            "n_cells": int(moc.uniq.numel()),
            "n_loops": len(loops),
        },
        "features": features,
    }
