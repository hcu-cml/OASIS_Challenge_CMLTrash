"""Application CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("app.main")


def _logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-22s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def cmd_doctor(_args: argparse.Namespace) -> int:
    """Validate system environment, API tokens, and model endpoints."""
    from .sources import CACHE_DIR, replay_only
    from .perception import MASK_CACHE

    rows: list[tuple[str, bool, str]] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        rows.append((label, ok, detail))

    n_cache = len(list(CACHE_DIR.rglob("*.json"))) if CACHE_DIR.exists() else 0
    n_masks = len(list(MASK_CACHE.rglob("*.json"))) if MASK_CACHE.exists() else 0
    check("fixtures: cached API responses", n_cache > 0, f"{n_cache} files in {CACHE_DIR}")
    check("fixtures: cached masks (RLE)", n_masks > 0, f"{n_masks} files in {MASK_CACHE}")

    # tier 1/2 — live models
    profile = os.environ.get("LLM_PROFILE", "ollama")
    try:
        from .agent.llm import Provider

        provider = Provider(profile)
        served = sorted(provider.available())
        check(f"LLM endpoint ({profile})", True, f"{len(served)} models served")
        for role in ("vision",):
            try:
                llm = provider.get(role)
                check(f"  role '{role}'", True, f"{llm.model}  {llm.params or 'default sampling'}")
            except Exception as exc:
                check(f"  role '{role}'", False, str(exc).splitlines()[0])
    except Exception as exc:
        check(f"LLM endpoint ({profile})", False, str(exc).splitlines()[0])

    check("MAPILLARY_TOKEN set", bool(os.environ.get("MAPILLARY_TOKEN")), "needed only to record new areas")

    try:
        import torch  # noqa: F401

        import torch as _t

        check("torch", True, f"cuda={_t.cuda.is_available()}")
    except Exception:
        check("torch", False, "not installed — fine for replay mode")

    width = max(len(r[0]) for r in rows) + 2
    print()
    print("  Environment check")
    print("  " + "-" * (width + 44))
    for label, ok, detail in rows:
        print(f"  {'PASS' if ok else 'FAIL'}  {label:<{width}} {detail}")
    print("  " + "-" * (width + 44))

    can_replay = n_cache > 0 and n_masks > 0
    print(f"\n  REPLAY={os.environ.get('REPLAY', '0')}   replay-capable: {can_replay}")
    if not can_replay:
        print("  Record fixtures first:  python -m app.main record \"Kaliganj, Gazipur\"")
    print()
    return 0


# --------------------------------------------------------------------------- #
def cmd_serve(args: argparse.Namespace) -> int:
    from .web.server import serve

    serve(host=args.host, port=args.port, debug=args.debug)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """
    Headless run. Privacy and trash disagreements are auto-resolved by exclusion
    so unattended execution cannot pass uncertain subjects or detections.
    """
    from .agent.graph import Run

    overrides = {}
    if args.max_images:
        overrides["max_images"] = args.max_images

    run = Run(args.place, overrides)
    print(f"run_id={run.run_id}  profile={run.provider.profile}")
    run.start()
    run.join(auto_resolve=True, timeout=args.timeout)

    snap = run.snapshot()
    print(f"\nstatus={snap['status']}  sites={len(snap.get('sites') or [])}")
    out = Path(run.cfg["runs_dir"]) / run.run_id
    if snap["status"] == "done":
        print(f"brief:    {out / 'brief.md'}")
        print(f"geojson:  {out / 'sites.geojson'}")
        print(f"manifest: {out / 'manifest.json'}")
        return 0
    print(json.dumps(snap.get("awaiting") or {}, indent=2)[:800], file=sys.stderr)
    return 1


def cmd_record(args: argparse.Namespace) -> int:
    """Run live with caching on, so the fixtures for replay mode get written."""
    os.environ["REPLAY"] = "0"
    log.info("Recording fixtures for %r — this needs network, a key and (for masks) a GPU", args.place)
    return cmd_run(args)


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="app.main", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="check the environment and report reproducible tiers")
    d.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("serve", help="run the web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=5000)
    s.add_argument("--debug", action="store_true")
    s.set_defaults(fn=cmd_serve)

    r = sub.add_parser("run", help="headless run, auto-resolving human decisions")
    r.add_argument("place")
    r.add_argument("--max-images", type=int, dest="max_images")
    r.add_argument("--timeout", type=float, default=3600.0)
    r.set_defaults(fn=cmd_run)

    rec = sub.add_parser("record", help="live run that writes replay fixtures")
    rec.add_argument("place")
    rec.add_argument("--max-images", type=int, dest="max_images", default=60)
    rec.add_argument("--timeout", type=float, default=7200.0)
    rec.set_defaults(fn=cmd_record)

    args = p.parse_args(argv)
    _logging(args.verbose)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
