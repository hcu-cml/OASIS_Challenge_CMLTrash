"""
Optional PostGIS store for grid-wise time-series analysis of trash detections.

Enabled only when DATABASE_URL is set, so the offline demo and native runs
without a database behave exactly as before.

Grid design
-----------
Cells are fixed squares in EPSG:6933 (WGS 84 / EASE-Grid 2.0 Global), a
global equal-area projection in metres. A cell is identified by its integer
column/row, floor(x / cell_m) and floor(y / cell_m), so the same patch of
ground gets the same cell ID in every run, whatever bbox was surveyed. Cell
assignment happens at query time, so any cell size can be analysed from the
same stored points without re-ingesting.

Counting rules
--------------
- One row per Mapillary image (deduplicated by image_id): re-surveying an area
  never double counts. The latest run's verdict on an image replaces earlier
  ones, including its observations.
- Only successfully screened images are stored; they form the denominator.
- All observations are stored with their review status; the time series
  counts only human-approved ones.
- Time buckets use the image capture time (UTC), not the run time.
- Positions are the camera GPS point, not a surveyed waste position.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

GRID_SRID = 6933
DEFAULT_CELL_M = float(os.environ.get("GRID_CELL_M", "250"))
BUCKETS = ("week", "month", "quarter", "year")
MIN_CELL_M, MAX_CELL_M = 25.0, 50_000.0

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS survey_runs (
    run_id         text PRIMARY KEY,
    place          text,
    resolved_area  text,
    bbox           geometry(Polygon, 4326),
    vision_model   text,
    n_images       integer NOT NULL DEFAULT 0,
    n_screened     integer NOT NULL DEFAULT 0,
    n_observations integer NOT NULL DEFAULT 0,
    n_approved     integer NOT NULL DEFAULT 0,
    coverage       jsonb,
    detection_stats jsonb,
    privacy_stats  jsonb,
    persisted_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS trash_images (
    image_id       text PRIMARY KEY,
    geom           geometry(Point, 4326) NOT NULL,
    captured_at    timestamptz,
    compass_angle  double precision,
    trash_positive boolean,
    pile_count     integer,
    first_run_id   text REFERENCES survey_runs(run_id) ON DELETE SET NULL,
    last_run_id    text REFERENCES survey_runs(run_id) ON DELETE SET NULL,
    updated_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS trash_images_geom_idx ON trash_images USING gist (geom);
CREATE INDEX IF NOT EXISTS trash_images_captured_idx ON trash_images (captured_at);

CREATE TABLE IF NOT EXISTS trash_observations (
    detection_id      text PRIMARY KEY,
    image_id          text NOT NULL REFERENCES trash_images(image_id) ON DELETE CASCADE,
    run_id            text REFERENCES survey_runs(run_id) ON DELETE SET NULL,
    geom              geometry(Point, 4326) NOT NULL,
    captured_at       timestamptz,
    compass_angle     double precision,
    mask_area_px      integer NOT NULL DEFAULT 0,
    sam_prompt        text,
    sam_score         double precision,
    screen_confidence double precision,
    review_status     text NOT NULL
        CHECK (review_status IN ('approved', 'pending', 'no_subject', 'excluded')),
    review_note       text NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS trash_obs_geom_idx ON trash_observations USING gist (geom);
CREATE INDEX IF NOT EXISTS trash_obs_image_idx ON trash_observations (image_id);
CREATE INDEX IF NOT EXISTS trash_obs_status_idx ON trash_observations (review_status);

-- Grid-wise time series. Usable from psql/QGIS too, e.g.
--   SELECT * FROM trash_grid_timeseries(250, 'month');
-- Old 3-argument signature; replaced by the version with a status filter.
DROP FUNCTION IF EXISTS trash_grid_timeseries(double precision, text, geometry);
CREATE OR REPLACE FUNCTION trash_grid_timeseries(
    cell_m double precision DEFAULT 250,
    bucket text DEFAULT 'month',
    area   geometry DEFAULT NULL,
    statuses text[] DEFAULT ARRAY['approved']
)
RETURNS TABLE (
    cell_id           text,
    ix                bigint,
    iy                bigint,
    period            timestamptz,
    images_screened   bigint,
    images_with_trash bigint,
    detections        bigint,
    mask_area_px      bigint,
    detections_per_100_images double precision,
    cell_geom         geometry
)
LANGUAGE sql STABLE AS $$
    WITH img AS (
        SELECT i.image_id,
               date_trunc(bucket, i.captured_at, 'UTC') AS period,
               floor(ST_X(ST_Transform(i.geom, {GRID_SRID})) / cell_m)::bigint AS ix,
               floor(ST_Y(ST_Transform(i.geom, {GRID_SRID})) / cell_m)::bigint AS iy
        FROM trash_images i
        WHERE i.captured_at IS NOT NULL
          AND (area IS NULL OR ST_Intersects(i.geom, area))
    ),
    det AS (
        SELECT o.image_id, count(*) AS n, sum(o.mask_area_px) AS px
        FROM trash_observations o
        WHERE o.review_status = ANY(statuses)
        GROUP BY o.image_id
    )
    SELECT
        format('%sm_%s_%s', cell_m::bigint, img.ix, img.iy),
        img.ix, img.iy, img.period,
        count(*),
        count(det.image_id),
        coalesce(sum(det.n), 0)::bigint,
        coalesce(sum(det.px), 0)::bigint,
        round((100.0 * coalesce(sum(det.n), 0) / count(*))::numeric, 3)::double precision,
        ST_Transform(ST_MakeEnvelope(
            img.ix * cell_m, img.iy * cell_m,
            (img.ix + 1) * cell_m, (img.iy + 1) * cell_m, {GRID_SRID}
        ), 4326)
    FROM img
    LEFT JOIN det ON det.image_id = img.image_id
    GROUP BY img.ix, img.iy, img.period
    ORDER BY img.ix, img.iy, img.period
$$;
"""


def database_url() -> str:
    return os.environ.get("DATABASE_URL", "").strip()


def enabled() -> bool:
    return bool(database_url())


def _connect():
    import psycopg

    conn = psycopg.connect(database_url(), connect_timeout=10)
    # Periods are bucketed in UTC; keep returned timestamps in UTC as well.
    conn.execute("SET TIME ZONE 'UTC'")
    conn.commit()
    return conn


_schema_ready = False


def ensure_schema(conn) -> None:
    """Create tables and the time-series function. Idempotent."""
    global _schema_ready
    if _schema_ready:
        return
    with conn.cursor() as cur:
        # Serialise concurrent first-time initialisation across runs.
        cur.execute("SELECT pg_advisory_xact_lock(hashtext('trash_schema'))")
        cur.execute(SCHEMA)
    conn.commit()
    _schema_ready = True


def ping() -> str:
    """Return the PostGIS version, raising if the database is unreachable."""
    with _connect() as conn:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT postgis_lib_version()")
            return str(cur.fetchone()[0])


def _ts(value: str | None) -> datetime | None:
    """Mapillary capture times are 'YYYY-MM-DD HH:MM:SS' in UTC."""
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _valid_point(lat: Any, lon: Any) -> bool:
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return False
    # 0,0 is the placeholder used when a camera position is missing.
    return -90 <= lat <= 90 and -180 <= lon <= 180 and not (lat == 0 and lon == 0)


def persist_run(state: dict[str, Any], vision_model: str | None = None) -> dict[str, int]:
    """
    Write one finished run in a single transaction.

    Returns counts of what was written, for the manifest and the run log.
    """
    from psycopg.types.json import Jsonb

    run_id = state["run_id"]
    meta = {image["image_id"]: image for image in state.get("images", [])}
    screens = state.get("trash_screens", [])
    observations = state.get("observations", [])

    # Only images the vision model actually assessed count as screened.
    screened: list[dict[str, Any]] = []
    for row in screens:
        image = meta.get(row.get("image_id"))
        assessment = row.get("assessment")
        if not image or not assessment or row.get("model_error"):
            continue
        if not _valid_point(image.get("lat"), image.get("lon")):
            continue
        screened.append({
            "image_id": image["image_id"],
            "lon": float(image["lon"]), "lat": float(image["lat"]),
            "captured_at": _ts(image.get("captured_at")),
            "compass_angle": image.get("compass_angle"),
            "trash_positive": bool(assessment.get("excessive_roadside_trash")),
            "pile_count": int(assessment.get("pile_count") or 0),
        })
    screened_ids = {row["image_id"] for row in screened}

    obs_rows = [
        o for o in observations
        if o.get("image_id") in screened_ids
        and _valid_point(o.get("camera_lat"), o.get("camera_lon"))
    ]
    approved = sum(o.get("review_status") == "approved" for o in obs_rows)

    bbox = state.get("bbox")
    resolved = ((state.get("manifest") or {}).get("aoi") or {}).get("resolved")

    with _connect() as conn:
        ensure_schema(conn)
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO survey_runs (
                    run_id, place, resolved_area, bbox, vision_model,
                    n_images, n_screened, n_observations, n_approved,
                    coverage, detection_stats, privacy_stats
                ) VALUES (
                    %s, %s, %s,
                    CASE WHEN %s::float8 IS NULL THEN NULL
                         ELSE ST_MakeEnvelope(%s, %s, %s, %s, 4326) END,
                    %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (run_id) DO UPDATE SET
                    place = EXCLUDED.place, resolved_area = EXCLUDED.resolved_area,
                    bbox = EXCLUDED.bbox, vision_model = EXCLUDED.vision_model,
                    n_images = EXCLUDED.n_images, n_screened = EXCLUDED.n_screened,
                    n_observations = EXCLUDED.n_observations,
                    n_approved = EXCLUDED.n_approved, coverage = EXCLUDED.coverage,
                    detection_stats = EXCLUDED.detection_stats,
                    privacy_stats = EXCLUDED.privacy_stats, persisted_at = now()
                """,
                (
                    run_id, state.get("place"), resolved,
                    *( [bbox[0], *bbox] if bbox else [None] * 5 ),
                    vision_model, len(state.get("images", [])), len(screened),
                    len(obs_rows), approved,
                    Jsonb(state.get("coverage") or {}),
                    Jsonb(state.get("detection_stats") or {}),
                    Jsonb(state.get("privacy_stats") or {}),
                ),
            )

            if screened:
                cur.executemany(
                    """
                    INSERT INTO trash_images (
                        image_id, geom, captured_at, compass_angle,
                        trash_positive, pile_count, first_run_id, last_run_id
                    ) VALUES (
                        %(image_id)s, ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326),
                        %(captured_at)s, %(compass_angle)s, %(trash_positive)s,
                        %(pile_count)s, %(run_id)s, %(run_id)s
                    )
                    ON CONFLICT (image_id) DO UPDATE SET
                        geom = EXCLUDED.geom, captured_at = EXCLUDED.captured_at,
                        compass_angle = EXCLUDED.compass_angle,
                        trash_positive = EXCLUDED.trash_positive,
                        pile_count = EXCLUDED.pile_count,
                        last_run_id = EXCLUDED.last_run_id, updated_at = now()
                    """,
                    [{**row, "run_id": run_id} for row in screened],
                )
                # The latest verdict on an image replaces its old detections.
                cur.execute(
                    "DELETE FROM trash_observations WHERE image_id = ANY(%s)",
                    (list(screened_ids),),
                )

            if obs_rows:
                cur.executemany(
                    """
                    INSERT INTO trash_observations (
                        detection_id, image_id, run_id, geom, captured_at,
                        compass_angle, mask_area_px, sam_prompt, sam_score,
                        screen_confidence, review_status, review_note
                    ) VALUES (
                        %(detection_id)s, %(image_id)s, %(run_id)s,
                        ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326),
                        %(captured_at)s, %(compass_angle)s, %(mask_area_px)s,
                        %(sam_prompt)s, %(sam_score)s, %(screen_confidence)s,
                        %(review_status)s, %(review_note)s
                    )
                    """,
                    [{
                        "detection_id": o["detection_id"], "image_id": o["image_id"],
                        "run_id": run_id,
                        "lon": float(o["camera_lon"]), "lat": float(o["camera_lat"]),
                        "captured_at": _ts(o.get("captured_at")),
                        "compass_angle": o.get("compass_angle"),
                        "mask_area_px": int(o.get("mask_area_px") or 0),
                        "sam_prompt": o.get("sam_prompt"), "sam_score": o.get("sam_score"),
                        "screen_confidence": o.get("screen_confidence"),
                        "review_status": o.get("review_status", "pending"),
                        "review_note": o.get("review_note") or "",
                    } for o in obs_rows],
                )

    return {
        "screened_images": len(screened),
        "observations": len(obs_rows),
        "approved_observations": approved,
    }


def grid_timeseries(
    cell_m: float = DEFAULT_CELL_M,
    bucket: str = "month",
    bbox: list[float] | None = None,
    include_pending: bool = False,
) -> dict[str, Any]:
    """
    Per-cell time series as a GeoJSON FeatureCollection: one feature per cell,
    with its periods ordered oldest first under properties.series.

    By default only human-approved detections count; include_pending also
    counts unreviewed model disagreements.
    """
    statuses = ["approved", "pending"] if include_pending else ["approved"]
    if bucket not in BUCKETS:
        raise ValueError(f"bucket must be one of {', '.join(BUCKETS)}")
    cell_m = float(cell_m)
    if not MIN_CELL_M <= cell_m <= MAX_CELL_M:
        raise ValueError(f"cell_m must be between {MIN_CELL_M:g} and {MAX_CELL_M:g} metres")

    area_sql = "NULL::geometry"
    params: list[Any] = [cell_m, bucket]
    if bbox:
        area_sql = "ST_MakeEnvelope(%s, %s, %s, %s, 4326)"
        params.extend(float(v) for v in bbox)
    params.append(statuses)

    with _connect() as conn:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT cell_id, ix, iy, period, images_screened, images_with_trash,
                       detections, mask_area_px, detections_per_100_images,
                       ST_AsGeoJSON(cell_geom, 7)
                FROM trash_grid_timeseries(%s, %s, {area_sql}, %s::text[])
                """,
                params,
            )
            rows = cur.fetchall()

    cells: dict[str, dict[str, Any]] = {}
    for (cell_id, ix, iy, period, n_img, n_pos, n_det, px, rate, geom) in rows:
        feature = cells.setdefault(cell_id, {
            "type": "Feature",
            "id": cell_id,
            "geometry": json.loads(geom),
            "properties": {"cell_id": cell_id, "ix": ix, "iy": iy, "series": []},
        })
        feature["properties"]["series"].append({
            "period": period.date().isoformat(),
            "images_screened": n_img,
            "images_with_trash": n_pos,
            "detections": n_det,
            "mask_area_px": px,
            "detections_per_100_images": rate,
        })
    return {
        "type": "FeatureCollection",
        "grid": {"crs": f"EPSG:{GRID_SRID}", "cell_m": cell_m, "bucket": bucket,
                 "statuses": statuses},
        "notes": (
            f"Counts are detections with status {' or '.join(statuses)}, "
            "deduplicated by Mapillary image, "
            "bucketed by image capture time (UTC). Positions are camera points. "
            "Compare detections_per_100_images across periods, not raw counts."
        ),
        "features": list(cells.values()),
    }
