"""External geospatial data client with caching, retries, and rate limiting."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

CACHE_DIR = Path(os.environ.get("CACHE_DIR", "fixtures/cache"))
SECRET_PARAMS = ("access_token", "api_key", "key", "token")

NOMINATIM = "https://nominatim.openstreetmap.org/search"
MAPILLARY = "https://graph.mapillary.com"
USER_AGENT = "oasis-waste-agent/1.0 (ACM SIGSPATIAL OASIS 2026 student challenge)"


def replay_only() -> bool:
    return os.environ.get("REPLAY", "0") == "1"


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    s.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=3,
                backoff_factor=1.5,
                status_forcelist=[429, 502, 503, 504],
                allowed_methods=frozenset(["GET", "POST"]),
                connect=3,
                backoff_max=20,
            )
        ),
    )
    return s


SESSION = _session()


def _scrub(params: dict[str, Any] | None) -> dict[str, Any]:
    return {k: v for k, v in (params or {}).items() if k not in SECRET_PARAMS}


def _cache_path(kind: str, url: str, payload: Any) -> Path:
    blob = json.dumps({"url": url, "payload": payload}, sort_keys=True, default=str)
    digest = hashlib.sha256(blob.encode()).hexdigest()[:20]
    return CACHE_DIR / kind / f"{digest}.json"


def cached_json(
    kind: str,
    url: str,
    params: dict | None = None,
    data: dict | None = None,
    method: str = "GET",
    timeout: int = 90,
    session: requests.Session | None = None,
    cache_url: str | None = None,
) -> Any:
    """Fetch JSON with disk cache lookup."""
    key_payload = {"params": _scrub(params), "data": _scrub(data), "method": method}
    path = _cache_path(kind, cache_url or url, key_payload)

    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    if replay_only():
        raise FileNotFoundError(
            f"REPLAY=1 but no fixture for {kind} {url}. Expected: {path}"
        )

    http = session or SESSION
    # (connect, read). The read timeout limits the gap between bytes, not the whole call.
    req_timeout = (10, timeout)
    if method == "GET":
        resp = http.get(url, params=params, timeout=req_timeout)
    else:
        resp = http.post(url, params=params, data=data, timeout=req_timeout)
    resp.raise_for_status()
    body = resp.json()
    write_json_atomic(path, body)
    return body


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link src to dst, falling back to a copy across filesystems."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.unlink(missing_ok=True)
        try:
            os.link(src, tmp)
        except OSError:
            import shutil
            shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write via a temp file and rename, so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Nominatim: place name -> polygon
# --------------------------------------------------------------------------- #
def search_places(place: str, limit: int = 5) -> list[dict[str, Any]]:
    """Search for places matching query, returning top candidates with bbox/coordinates."""
    query = (place or "").strip()
    if not query:
        return []

    try:
        results = cached_json(
            "nominatim",
            NOMINATIM,
            params={
                "q": query,
                "format": "jsonv2",
                "polygon_geojson": 1,
                "limit": limit,
                "addressdetails": 1,
                "accept-language": "en",
            },
        )
    except Exception:
        if replay_only():
            try:
                single = geocode_area(query)
                return [single]
            except Exception:
                return []
        raise

    if not isinstance(results, list):
        return []

    hits = []
    for hit in results:
        try:
            bbox_raw = hit.get("boundingbox")
            if bbox_raw and len(bbox_raw) == 4:
                south, north, west, east = (float(v) for v in bbox_raw)
                bbox = [west, south, east, north]
            else:
                lat = float(hit["lat"])
                lon = float(hit["lon"])
                bbox = [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01]
        except (KeyError, TypeError, ValueError):
            continue

        lat = float(hit.get("lat", (bbox[1] + bbox[3]) / 2))
        lon = float(hit.get("lon", (bbox[0] + bbox[2]) / 2))
        display_name = hit.get("display_name", query)
        name = hit.get("name") or display_name.split(",")[0].strip()

        hits.append({
            "display_name": display_name,
            "name": name,
            "osm_type": hit.get("osm_type"),
            "osm_id": hit.get("osm_id"),
            "type": hit.get("type", ""),
            "class": hit.get("class", ""),
            "lat": lat,
            "lon": lon,
            "bbox": bbox,
            "geometry": hit.get("geojson"),
        })
    return hits


def geocode_area(place: str) -> dict[str, Any]:
    """Resolve a place name to a polygon and bbox. No hardcoded cities."""
    query = (place or "").strip()
    if not query:
        raise ValueError("No place name was given to geocode.")

    results = cached_json(
        "nominatim",
        NOMINATIM,
        params={
            "q": query,
            "format": "jsonv2",
            "polygon_geojson": 1,
            "limit": 1,
            "addressdetails": 1,
            # Force Latin script; otherwise names echo back in the local script.
            "accept-language": "en",
        },
    )
    if not results:
        raise ValueError(
            f"Nominatim could not resolve {query!r}. Try adding a city or country, "
            "e.g. 'Hazaribagh, Dhaka', or draw a box on the map instead."
        )

    hit = results[0]
    try:
        south, north, west, east = (float(v) for v in hit["boundingbox"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Nominatim returned no usable bounding box for {query!r}."
        ) from exc
    return {
        "display_name": hit.get("display_name", query),
        "osm_type": hit.get("osm_type"),
        "osm_id": hit.get("osm_id"),
        "bbox": [west, south, east, north],
        "geometry": hit.get("geojson"),
    }









# --------------------------------------------------------------------------- #
# Mapillary
# --------------------------------------------------------------------------- #
MAPILLARY_FIELDS = "id,captured_at,compass_angle,geometry,sequence,thumb_1024_url,thumb_2048_url"


# Mapillary returns HTTP 500 when a bbox exceeds an undocumented data budget.
# Image density drives this, not the requested `limit`, so no fixed tile size
# suits both sparse and dense areas: tiles start small and overflowing tiles
# are subdivided.
TILE_START = 0.01
TILE_FLOOR = 0.00125


MAPILLARY_TIMEOUT = 20.0  # seconds per request
MAPILLARY_TILE_RETRIES = 2  # per tile, timeouts and connection errors only
# Whole-search limits; 0 means unlimited.
MAPILLARY_MAX_TILES = int(os.environ.get("MAPILLARY_MAX_TILES", "0"))
MAPILLARY_DEADLINE_SECONDS = int(os.environ.get("MAPILLARY_DEADLINE_SECONDS", "0"))


def mapillary_search(
    bbox: list[float], token: str, limit: int = 400,
    check_cancel: Callable[[], None] | None = None,
) -> list[dict]:
    """
    Search images in a bbox, subdividing tiles that overflow the API budget.

    A tile that keeps failing is skipped, so one bad patch cannot stall the run.
    """
    west, south, east, north = bbox
    area = (east - west) * (north - south)
    pending = _tiles(bbox, TILE_START) if area > TILE_START**2 else [list(bbox)]

    seen: set[str] = set()
    out: list[dict] = []
    tiles_processed = 0
    skipped_tiles = 0
    deadline = (
        time.monotonic() + MAPILLARY_DEADLINE_SECONDS if MAPILLARY_DEADLINE_SECONDS > 0 else None
    )
    while pending and len(out) < limit:
        if check_cancel:
            check_cancel()
        if deadline is not None and time.monotonic() > deadline:
            log.warning(
                "Mapillary search hit its %d minute deadline with %d tile(s) "
                "still pending; returning %d image(s) found so far",
                MAPILLARY_DEADLINE_SECONDS // 60, len(pending), len(out),
            )
            break
        if MAPILLARY_MAX_TILES > 0 and tiles_processed >= MAPILLARY_MAX_TILES:
            log.warning(
                "Mapillary search hit the %d tile cap with %d tile(s) still "
                "pending; returning %d image(s) found so far",
                MAPILLARY_MAX_TILES, len(pending), len(out),
            )
            break

        w, s, e, n = pending.pop()
        tiles_processed += 1
        params = {
            "access_token": token,
            "fields": MAPILLARY_FIELDS,
            "bbox": f"{w},{s},{e},{n}",
            "limit": min(2000, limit - len(out)),
            "start_captured_at": "2023-01-01T00:00:00Z",
        }

        body = None
        for attempt in range(MAPILLARY_TILE_RETRIES + 1):
            try:
                body = cached_json(
                    "mapillary", f"{MAPILLARY}/images", params=params,
                    timeout=MAPILLARY_TIMEOUT,
                )
                break
            except requests.HTTPError as exc:
                if exc.response is None or exc.response.status_code != 500:
                    raise
                if min(e - w, n - s) / 2 < TILE_FLOOR:
                    log.warning(
                        "Mapillary still overflows at the %.5f degree floor for "
                        "bbox %.4f,%.4f,%.4f,%.4f; skipping this tile", TILE_FLOOR, w, s, e, n
                    )
                else:
                    log.info(
                        "Mapillary overflow at %.4f degrees, subdividing tile %.4f,%.4f",
                        e - w, w, s,
                    )
                    pending.extend(_quarters([w, s, e, n]))
                break
            except (requests.Timeout, requests.ConnectionError) as exc:
                if attempt < MAPILLARY_TILE_RETRIES:
                    log.info(
                        "Mapillary %s on tile %.4f,%.4f (attempt %d/%d), retrying",
                        type(exc).__name__, w, s, attempt + 1, MAPILLARY_TILE_RETRIES + 1,
                    )
                    continue
                log.warning(
                    "Mapillary tile %.4f,%.4f,%.4f,%.4f kept failing (%s); "
                    "skipping it rather than hanging the whole survey",
                    w, s, e, n, type(exc).__name__,
                )
                skipped_tiles += 1
        if body is None:
            continue

        for img in body.get("data", []):
            iid = str(img.get("id"))
            if iid and iid not in seen:
                seen.add(iid)
                out.append(img)

    if skipped_tiles:
        log.warning(
            "%d tile(s) were skipped after repeated timeouts/connection errors; "
            "coverage in those spots will be incomplete", skipped_tiles,
        )
    return out[:limit]


def _quarters(bbox: list[float]) -> list[list[float]]:
    west, south, east, north = bbox
    mid_lon, mid_lat = (west + east) / 2, (south + north) / 2
    return [
        [west, south, mid_lon, mid_lat],
        [mid_lon, south, east, mid_lat],
        [west, mid_lat, mid_lon, north],
        [mid_lon, mid_lat, east, north],
    ]


def _tiles(bbox: list[float], size: float) -> list[list[float]]:
    west, south, east, north = bbox
    tiles = []
    lat = south
    while lat < north:
        lon = west
        while lon < east:
            tiles.append([lon, lat, min(lon + size, east), min(lat + size, north)])
            lon += size
        lat += size
    return tiles


def parse_images(raw: list[dict]) -> list[dict]:
    """Flatten Mapillary records. Timestamps in UTC, not local time."""
    from datetime import datetime, timezone

    out = []
    for img in raw:
        coords = (img.get("geometry") or {}).get("coordinates") or [None, None]
        if coords[0] is None:
            continue
        ms = img.get("captured_at")
        captured = (
            datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            if ms else None
        )
        out.append(
            {
                "image_id": str(img.get("id")),
                "lon": float(coords[0]),
                "lat": float(coords[1]),
                "compass_angle": img.get("compass_angle"),
                "captured_at": captured,
                "thumb_url": img.get("thumb_2048_url") or img.get("thumb_1024_url"),
            }
        )
    return out


def download_thumbs(
    images: list[dict],
    dest: str,
    workers: int = 8,
    check_cancel: Callable[[], None] | None = None,
) -> int:
    """Download thumbnails, checking cooperative cancellation between image/chunk work."""
    from concurrent.futures import ThreadPoolExecutor

    Path(dest).mkdir(parents=True, exist_ok=True)

    def one(meta: dict) -> bool:
        if check_cancel:
            check_cancel()
        out = Path(dest) / f"{meta['image_id']}.jpg"
        if out.exists():
            meta["local_path"] = str(out)
            return True
            
        cached = Path("fixtures/images") / f"{meta['image_id']}.jpg"
        if cached.exists():
            _link_or_copy(cached, out)
            meta["local_path"] = str(out)
            return True
            
        if replay_only() or not meta.get("thumb_url"):
            return False
        try:
            r = SESSION.get(meta["thumb_url"], timeout=45, stream=True)
            r.raise_for_status()
            with open(out, "wb") as fh:
                for chunk in r.iter_content(8192):
                    if check_cancel:
                        check_cancel()
                    fh.write(chunk)
            meta["local_path"] = str(out)
            # Share the image across runs via the image cache.
            try:
                _link_or_copy(out, cached)
            except OSError as exc:
                log.warning("could not cache thumb %s: %s", meta["image_id"], exc)
            return True
        except Exception as exc:
            if check_cancel:
                check_cancel()
            log.warning("thumb %s failed: %s", meta["image_id"], exc)
            return False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return sum(pool.map(one, images))
