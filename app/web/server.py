"""Flask web server with Server-Sent Events for real-time agent workflow interaction."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from .. import sources
from ..agent.graph import Run
from ..agent.state import BUS

log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/geocode")
@app.get("/api/search")
def geocode():
    """Geocode a place query or provide search suggestions."""
    query = (request.args.get("q") or "").strip()
    if not query:
        return jsonify({"query": "", "results": []})
    limit = int(request.args.get("limit", 5))
    try:
        results = sources.search_places(query, limit=limit)
        if not results:
            try:
                single = sources.geocode_area(query)
                if single:
                    results = [single]
            except Exception:
                pass
        return jsonify({"query": query, "results": results})
    except Exception as exc:
        log.warning("Geocoding failed for %r: %s", query, exc)
        return jsonify({"error": str(exc), "query": query, "results": []}), 500


@app.post("/api/run")
def start_run():
    body = request.get_json(force=True) or {}
    place = (body.get("place") or "").strip()
    bbox = body.get("bbox")
    if not place and not bbox:
        return jsonify(
            {"error": "Give a place name, e.g. 'Hazaribagh, Dhaka', or draw a box on the map."}
        ), 400
    if bbox:
        try:
            w, s, e, n = (float(v) for v in bbox)
        except (TypeError, ValueError):
            return jsonify({"error": "bbox must be [west, south, east, north]."}), 400
        if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
            return jsonify(
                {"error": "bbox must be [west, south, east, north] with west<east and south<north."}
            ), 400

    overrides = {}
    for key in ("max_images", "confidence"):
        if body.get(key) is not None:
            overrides[key] = type(_default(key))(body[key])
    if body.get("bbox"):
        overrides["bbox"] = [float(v) for v in body["bbox"]]

    try:
        run = Run(place, overrides)
    except Exception as exc:
        # Usually a missing key or an unreachable endpoint.
        return jsonify({"error": str(exc)}), 502

    run.start()
    return jsonify({"run_id": run.run_id, "provider": run.provider.profile})


def _default(key: str):
    from ..agent.graph import DEFAULTS

    return DEFAULTS.get(key, 0)


@app.get("/api/config")
def get_config():
    """Return current configuration keys (values masked for secrets)."""
    import os as _os

    keys = {
        "MAPILLARY_TOKEN": _os.environ.get("MAPILLARY_TOKEN", ""),
        "HF_TOKEN": _os.environ.get("HF_TOKEN", ""),
        "OLLAMA_BASE_URL": _os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "SAM3_DIR": _os.environ.get("SAM3_DIR", "weights/sam3"),
    }
    # Report whether secrets are set, never their value.
    masked = {}
    secret_keys = {"MAPILLARY_TOKEN", "HF_TOKEN"}
    for key, value in keys.items():
        if key in secret_keys:
            masked[key] = {"set": bool(value), "preview": value[:6] + "…" if len(value) > 6 else ""}
        else:
            masked[key] = {"set": bool(value), "value": value}
            
    masked["REPLAY_MODE"] = {"set": True, "value": _os.environ.get("REPLAY", "0") == "1"}
    
    return jsonify(masked)


@app.get("/api/config/reveal/<key>")
def reveal_config(key: str):
    """Return the full value of a config key. Local-only server, no auth needed."""
    import os as _os

    allowed = {"MAPILLARY_TOKEN", "HF_TOKEN", "OLLAMA_BASE_URL", "SAM3_DIR"}
    if key not in allowed:
        return jsonify({"error": "unknown key"}), 404
    return jsonify({"key": key, "value": _os.environ.get(key, "")})


@app.post("/api/config")
def set_config():
    """Update runtime environment variables (session-only, not persisted to .env)."""
    import os as _os

    body = request.get_json(force=True) or {}
    allowed = {"MAPILLARY_TOKEN", "HF_TOKEN", "OLLAMA_BASE_URL"}
    applied = {}
    for key, value in body.items():
        if key in allowed and isinstance(value, str):
            _os.environ[key] = value
            applied[key] = True
    return jsonify({"applied": applied})


@app.get("/api/events/<run_id>")
def events(run_id: str):
    # Check the registry first: BUS.sse() allocates a channel, so this ordering
    # keeps unknown IDs a clean 404.
    if Run.get(run_id) is None:
        return jsonify({"error": "unknown run"}), 404
    # Last-Event-ID takes precedence over ?from=, otherwise an automatic
    # reconnect replays the run and reopens any pending human gate.
    resume_at = request.headers.get("Last-Event-ID")
    if resume_at and resume_at.isdigit():
        from_index = int(resume_at)
    else:
        from_index = int(request.args.get("from", 0))
    return Response(
        BUS.sse(run_id, from_index),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            # Stops nginx-style proxies from buffering the stream.
            "X-Accel-Buffering": "no",
            # No `Connection` header: it is hop-by-hop and PEP 3333 forbids a
            # WSGI app from setting it. Waitress raises AssertionError if set.
        },
    )


@app.get("/api/state/<run_id>")
def state(run_id: str):
    run = Run.get(run_id)
    if not run:
        return jsonify({"error": "unknown run"}), 404
    return jsonify(run.snapshot())


@app.post("/api/cancel/<run_id>")
def cancel(run_id: str):
    """Idempotently request cooperative run cancellation."""
    run = Run.get(run_id)
    if not run:
        return jsonify({"error": "unknown run"}), 404
    status = run.cancel()
    return jsonify({"ok": True, "status": status})


@app.post("/api/resume/<run_id>")
def resume(run_id: str):
    """Deliver a human decision to a paused graph."""
    run = Run.get(run_id)
    if not run:
        return jsonify({"error": "unknown run"}), 404
    if run.status != "awaiting_input":
        return jsonify({"error": f"run is {run.status}, not awaiting input"}), 409

    payload = request.get_json(force=True) or {}
    run.resume(payload)
    return jsonify({"ok": True})


@app.get("/artifact/<run_id>/<path:relpath>")
def artifact(run_id: str, relpath: str):
    """Serve sanitized outputs; reveal an original only at its active privacy gate."""
    from ..agent.graph import DEFAULTS

    runs_root = Path(DEFAULTS["runs_dir"]).resolve()
    root = (runs_root / run_id).resolve()
    target = (root / relpath).resolve()
    try:
        root.relative_to(runs_root)
        relative = target.relative_to(root)
    except ValueError:
        return jsonify({"error": "forbidden"}), 403

    parts = relative.parts
    public_dirs = {"sanitized", "trash_overlays"}
    # Published artifacts: human-approved detections only, no raw imagery.
    public_files = {"brief.md", "manifest.json", "trash_citygml.gml"}
    allowed = bool(parts) and (parts[0] in public_dirs or relative.as_posix() in public_files)

    # Unredacted imagery is exposed only for items in an active privacy review.
    if parts and parts[0] == "images":
        run = Run.get(run_id)
        awaiting = run.awaiting if run and run.status == "awaiting_input" else None
        if awaiting and awaiting.get("kind") == "human_privacy_review":
            prefix = f"/artifact/{run_id}/"
            allowed_relpaths = {
                item.get("original_url", "").split(prefix, 1)[1]
                for item in awaiting.get("items", [])
                if prefix in item.get("original_url", "")
            }
            allowed = relative.as_posix() in allowed_relpaths

    if not allowed:
        return jsonify({"error": "forbidden"}), 403
    if not target.is_file():
        return jsonify({"error": "not found"}), 404
    return send_from_directory(target.parent, target.name)


def serve(host: str = "127.0.0.1", port: int = 5000, debug: bool = False) -> None:
    """
    Serve the UI on a production WSGI server when one is available.

    Single process, many threads, always. Run state - the `Run` registry, the
    in-memory event bus, the paused graphs waiting on a human - lives in this
    process. A pre-fork server with two workers would answer /api/state for a
    run the other worker owns, so the review gate would intermittently 404.
    That is why this is waitress with threads rather than gunicorn with workers.

    Each SSE stream occupies a thread for the lifetime of the run it follows, so
    the pool has to be comfortably larger than the number of open browser tabs.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s"
    )
    profile = os.environ.get("LLM_PROFILE", "ollama")
    replay = os.environ.get("REPLAY", "0") == "1"
    log.info("LLM profile: %s | replay mode: %s", profile, replay)

    if host not in ("127.0.0.1", "localhost", "::1"):
        log.warning(
            "Binding to %s. This server has NO authentication: anyone who can "
            "reach it can start runs that spend Mapillary quota and GPU time, "
            "and can submit review decisions.", host,
        )

    if debug:
        log.info("Debug mode: Flask development server on http://%s:%d", host, port)
        app.run(host=host, port=port, debug=True, threaded=True)
        return

    try:
        from waitress import serve as waitress_serve
    except ImportError:
        log.warning(
            "waitress not installed, falling back to the Flask development "
            "server. Install it with: pip install waitress"
        )
        app.run(host=host, port=port, debug=False, threaded=True)
        return

    log.info("waitress (1 process, 16 threads) on http://%s:%d", host, port)
    waitress_serve(
        app,
        host=host,
        port=port,
        threads=16,
        # SSE streams are long-lived with a 15s keepalive; the default 120s
        # channel timeout would cut them mid-run.
        channel_timeout=3600,
        ident="oasis-waste-agent",
    )
