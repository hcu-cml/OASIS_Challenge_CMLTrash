"""CityGML 2.0 export for roadside waste observations."""

from __future__ import annotations

import re
from datetime import date, timezone
from typing import Any, Iterable
from xml.etree import ElementTree as ET

CITYGML_VERSION = "2.0"
SRS_NAME = "urn:ogc:def:crs:EPSG::4326"
SRS_DIMENSION = "2"

NS = {
    "core": "http://www.opengis.net/citygml/2.0",
    "gen": "http://www.opengis.net/citygml/generics/2.0",
    "gml": "http://www.opengis.net/gml",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance",
}

SCHEMA_LOCATION = (
    "http://www.opengis.net/citygml/2.0 "
    "http://schemas.opengis.net/citygml/2.0/cityGMLBase.xsd "
    "http://www.opengis.net/citygml/generics/2.0 "
    "http://schemas.opengis.net/citygml/generics/2.0/generics.xsd"
)

FEATURE_CLASS = "roadside_waste_accumulation"
MAPILLARY_INFORMATION_SYSTEM = "https://www.mapillary.com/"

_NCNAME_BAD = re.compile(r"[^A-Za-z0-9._-]")


def _q(prefix: str, tag: str) -> str:
    return f"{{{NS[prefix]}}}{tag}"


def gml_id(raw: str, prefix: str = "trash") -> str:
    """Format a valid NCName string for gml:id."""
    cleaned = _NCNAME_BAD.sub("_", str(raw or "")).strip("_") or "unknown"
    return f"{prefix}_{cleaned}"


def _text(parent: ET.Element, prefix: str, tag: str, value: Any) -> None:
    ET.SubElement(parent, _q(prefix, tag)).text = str(value)


def _generic(parent: ET.Element, kind: str, name: str, value: Any, uom: str | None = None) -> None:
    """Add a gen:*Attribute element if value is present."""
    if value is None or value == "":
        return
    node = ET.SubElement(parent, _q("gen", kind), {"name": name})
    val = ET.SubElement(node, _q("gen", "value"))
    if uom:
        val.set("uom", uom)
    if isinstance(value, bool):
        val.text = "true" if value else "false"
    elif isinstance(value, float):
        val.text = f"{value:.6g}"
    else:
        val.text = str(value)


def _iso_date(value: Any) -> str | None:
    """Extract an ISO 8601 date string (YYYY-MM-DD)."""
    if not value:
        return None
    text = str(value).strip()
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})", text)
    return match.group(0) if match else None


def _observation_feature(
    member: ET.Element,
    observation: dict[str, Any],
    run_id: str,
    fallback_date: str,
    vision_model: str | None,
) -> None:
    obj = ET.SubElement(member, _q("gen", "GenericCityObject"))
    obj.set(_q("gml", "id"), gml_id(observation.get("detection_id")))

    image_id = observation.get("image_id")

    _text(
        obj, "gml", "description",
        "Roadside waste observation confirmed by review.",
    )
    _text(obj, "gml", "name", f"Roadside waste observed in image {image_id}")

    created = _iso_date(observation.get("captured_at")) or fallback_date
    _text(obj, "core", "creationDate", created)

    ref = ET.SubElement(obj, _q("core", "externalReference"))
    _text(ref, "core", "informationSystem", MAPILLARY_INFORMATION_SYSTEM)
    external = ET.SubElement(ref, _q("core", "externalObject"))
    _text(external, "core", "name", str(image_id))

    _generic(obj, "stringAttribute", "runId", run_id)
    _generic(obj, "stringAttribute", "detectionId", observation.get("detection_id"))
    _generic(obj, "stringAttribute", "sourceImageId", image_id)
    _generic(obj, "stringAttribute", "segmentationModel", "SAM3")
    _generic(obj, "stringAttribute", "segmentationPrompt", observation.get("sam_prompt"))
    _generic(obj, "doubleAttribute", "segmentationConfidence", observation.get("sam_score"))
    _generic(obj, "stringAttribute", "visionModel", vision_model)
    _generic(obj, "doubleAttribute", "visionConfidence", observation.get("screen_confidence"))
    _generic(obj, "intAttribute", "maskAreaPixels", observation.get("mask_area_px"))
    _generic(obj, "stringAttribute", "reviewStatus", observation.get("review_status"))
    _generic(obj, "stringAttribute", "reviewNote", observation.get("review_note"))
    _generic(obj, "dateAttribute", "imageCapturedDate", _iso_date(observation.get("captured_at")))
    _generic(obj, "stringAttribute", "positionSource", "camera_gps_observation_point")

    _text(obj, "gen", "class", FEATURE_CLASS)
    _text(obj, "gen", "function", "roadside_waste")

    geometry = ET.SubElement(obj, _q("gen", "lod0Geometry"))
    point = ET.SubElement(geometry, _q("gml", "Point"))
    point.set("srsName", SRS_NAME)
    point.set("srsDimension", SRS_DIMENSION)
    point.set(_q("gml", "id"), gml_id(observation.get("detection_id"), prefix="p"))
    _text(
        point, "gml", "pos",
        f"{float(observation['camera_lat']):.7f} {float(observation['camera_lon']):.7f}",
    )


def accepted_observations(observations: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter observations that are approved and have valid coordinates."""
    out: list[dict[str, Any]] = []
    for item in observations or []:
        if item.get("review_status") != "approved":
            continue
        lat, lon = item.get("camera_lat"), item.get("camera_lon")
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            continue
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            continue
        if lat == 0.0 and lon == 0.0:
            continue
        out.append(item)
    return out


def build(
    observations: Iterable[dict[str, Any]],
    run_id: str,
    place: str | None = None,
    vision_model: str | None = None,
    created: date | None = None,
) -> tuple[str, int]:
    """Serialize reviewed observations into a CityGML 2.0 XML string."""
    for prefix, uri in NS.items():
        ET.register_namespace(prefix, uri)

    items = accepted_observations(observations)
    fallback_date = (created or date.today()).isoformat()

    root = ET.Element(_q("core", "CityModel"))
    root.set(_q("xsi", "schemaLocation"), SCHEMA_LOCATION)

    _text(root, "gml", "name", f"Roadside trash survey{f' - {place}' if place else ''}")
    _text(
        root, "gml", "description",
        f"Roadside waste observations from run {run_id}. {len(items)} approved detection(s).",
    )

    if items:
        lats = [float(i["camera_lat"]) for i in items]
        lons = [float(i["camera_lon"]) for i in items]
        bounded = ET.SubElement(root, _q("gml", "boundedBy"))
        envelope = ET.SubElement(bounded, _q("gml", "Envelope"))
        envelope.set("srsName", SRS_NAME)
        envelope.set("srsDimension", SRS_DIMENSION)
        _text(envelope, "gml", "lowerCorner", f"{min(lats):.7f} {min(lons):.7f}")
        _text(envelope, "gml", "upperCorner", f"{max(lats):.7f} {max(lons):.7f}")

    for observation in items:
        member = ET.SubElement(root, _q("core", "cityObjectMember"))
        _observation_feature(member, observation, run_id, fallback_date, vision_model)

    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode", xml_declaration=False)
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{body}\n', len(items)


def write(
    path: str,
    observations: Iterable[dict[str, Any]],
    run_id: str,
    place: str | None = None,
    vision_model: str | None = None,
) -> int:
    """Write CityGML document to disk and return feature count."""
    from datetime import datetime
    from pathlib import Path

    text, count = build(
        observations, run_id, place=place, vision_model=vision_model,
        created=datetime.now(timezone.utc).date(),
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return count

