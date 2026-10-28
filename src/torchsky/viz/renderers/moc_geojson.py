"""
MOC → GeoJSON serializer.

Uses MOC.boundary() (torchsky.sphere.moc) to extract polygon boundary loops,
then converts them to GeoJSON FeatureCollection format suitable for Aladin Lite
and D3-geo overlays.

Longitudes are normalized to the GeoJSON [-180, 180] convention; rings are not
split at the antimeridian because Aladin unwraps them natively.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchsky.sphere.moc import MOC


def _to_geojson_longitudes(
    vertices: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Map RA in ``[0, 360)`` onto the GeoJSON ``[-180, 180)`` convention."""
    return [(((lon + 180.0) % 360.0) - 180.0, lat) for lon, lat in vertices]


def moc_to_geojson(moc: "MOC") -> dict:
    """
    Convert a torchsky MOC to a GeoJSON FeatureCollection.

    Uses MOC.boundary() to extract polygon loops.
    Each loop becomes one GeoJSON Feature with Polygon geometry.

    Boundary extraction failures propagate: an empty overlay would silently
    misrepresent the footprint.

    Returns a dict ready for json.dumps().
    """
    loops = moc.boundary()

    features = []
    for i, loop in enumerate(loops):
        # loop is a Tensor of shape [N, 2] with (lon, lat) in degrees
        vertices_raw = [(float(pt[0].item()), float(pt[1].item())) for pt in loop]

        # Close the ring if not already closed
        if vertices_raw and vertices_raw[0] != vertices_raw[-1]:
            vertices_raw.append(vertices_raw[0])

        ring = _to_geojson_longitudes(vertices_raw)
        if len(ring) < 4:
            continue  # degenerate ring
        # GeoJSON Polygon: coordinates is [[outer_ring], [hole1], ...]
        features.append(
            {
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
        )

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
