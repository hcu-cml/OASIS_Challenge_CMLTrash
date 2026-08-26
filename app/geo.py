"""Geometry and coordinate reference system helpers."""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6371008.8


def utm_epsg(lon: float, lat: float) -> int:
    """Calculate the UTM EPSG code for a given longitude and latitude."""
    zone = int((lon + 180.0) / 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


def bbox_polygon(bbox: list[float]) -> dict:
    """Create a GeoJSON polygon from a [west, south, east, north] bounding box."""
    w, s, e, n = (float(v) for v in bbox)
    return {
        "type": "Polygon",
        "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
    }


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask using nearest-neighbor interpolation."""
    from PIL import Image

    img = Image.fromarray((mask.astype(np.uint8) * 255))
    img = img.resize((shape[1], shape[0]), Image.NEAREST)
    return (np.array(img) > 127).astype(np.uint8)
