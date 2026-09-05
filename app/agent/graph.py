"""State graph construction and run execution management."""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from .. import perception
from .llm import Provider
from .nodes import Nodes
from .state import BUS, GRAPH_EDGES, NODES, AgentState, RunCancelled

log = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "runs_dir": os.environ.get("RUNS_DIR", "runs"),
    "mapillary_token": os.environ.get("MAPILLARY_TOKEN", ""),
    "sam3_dir": os.environ.get("SAM3_DIR", "weights/sam3"),
    "sam3_revision": os.environ.get("SAM3_REVISION", "unpinned"),
    "max_images": int(os.environ.get("MAX_IMAGES", "300")),
    "download_workers": 8,
    "screen_workers": int(os.environ.get("SCREEN_WORKERS", "4")),
    "screen_max_tokens": int(os.environ.get("SCREEN_MAX_TOKENS", "1200")),
    "screen_image_px": int(os.environ.get("SCREEN_IMAGE_PX", "896")),
    "confidence": perception.GATE_CONFIDENCE,
    "min_area_pct": 0.15,
    "review_confidence": 0.65,
    "privacy_blur_radius": 28.0,
    "privacy_dilation_px": 25,
}


def build_graph(nodes: Nodes):
    """Construct the linear LangGraph execution pipeline."""
    graph = StateGraph(AgentState)
    implementations = {
        "aoi": nodes.aoi,
        "coverage_audit": nodes.coverage_audit,
        "fetch": nodes.fetch,
        "screen_trash": nodes.screen_trash,
        "segment_trash": nodes.segment_trash,
        "blur_humans": nodes.blur_humans,
        "review_trash": nodes.review_trash,
        "export_citygml": nodes.export_citygml,
        "report": nodes.report_node,
    }
    for node in NODES:
        graph.add_node(node["id"], implementations[node["id"]])
    graph.add_edge(START, NODES[0]["id"])
    for edge in GRAPH_EDGES:
        graph.add_edge(edge["source"], edge["target"])
    graph.add_edge(NODES[-1]["id"], END)
    return graph


class Run:
    """Manages an individual agent workflow run with SQLite checkpointing."""

    _registry: dict[str, "Run"] = {}
    _lock = threading.Lock()

    def __init__(self, place: str, overrides: dict[str, Any] | None = None):
        self.run_id = uuid.uuid4().hex[:10]
        self.cfg = {**DEFAULTS, **(overrides or {})}
        self.place = place
        self.thread = {"configurable": {"thread_id": self.run_id}}

        Path(self.cfg["runs_dir"]).mkdir(parents=True, exist_ok=True)
        self._cm = SqliteSaver.from_conn_string(
            str(Path(self.cfg["runs_dir"]) / "checkpoints.sqlite")
        )
        self.saver = self._cm.__enter__()

        self.provider = Provider()
        self._cancel_event = threading.Event()
        self._status_lock = threading.Lock()
        self.nodes = Nodes(self.run_id, self.provider, self.cfg, self._cancel_event)
        self.app = build_graph(self.nodes).compile(checkpointer=self.saver)

        self.status = "created"
        self.awaiting: dict[str, Any] | None = None
        self._worker: threading.Thread | None = None

        with Run._lock:
            Run._registry[self.run_id] = self

    # ------------------------------------------------------------------ #
    @classmethod
    def get(cls, run_id: str) -> "Run | None":
        with cls._lock:
            return cls._registry.get(run_id)

    def start(self) -> None:
        with self._status_lock:
            if self.status == "cancelled":
                return
        BUS.emit(
            self.run_id, "run_start",
            run_id=self.run_id, place=self.place,
            nodes=NODES,
            edges=GRAPH_EDGES,
            provider=self.provider.profile,
        )
        initial: dict[str, Any] = {
            "run_id": self.run_id,
            "place": self.place,
            "prompt": self.place,
        }
        # Seed the state so a hand-drawn box takes precedence over the geocoder.
        if self.cfg.get("bbox"):
            initial["bbox"] = list(self.cfg["bbox"])
        self._spawn(initial)

    def resume(self, payload: dict[str, Any]) -> None:
        """Feed a human decision back into the paused graph."""
        from langgraph.types import Command

        with self._status_lock:
            if self.status != "awaiting_input" or self._cancel_event.is_set():
                return
            BUS.emit(self.run_id, "input_received", payload_kind=payload.get("kind"))
            self.awaiting = None
            self._spawn(Command(resume=payload))

    def cancel(self) -> str:
        """Request cooperative cancellation; safe and idempotent in every state."""
        with self._status_lock:
            if self.status in ("done", "error", "cancelled"):
                return self.status
            if self.status == "cancelling":
                return self.status
            previous = self.status
            self._cancel_event.set()
            if previous in ("created", "awaiting_input"):
                self.status = "cancelled"
                self.awaiting = None
            else:
                self.status = "cancelling"
            BUS.emit(
                self.run_id, "cancel_requested", status=self.status,
                message=(
                    "Cancellation is cooperative; an in-flight synchronous model or "
                    "network call may return before the next cancellation check."
                ),
            )
            if self.status == "cancelled":
                BUS.emit(self.run_id, "cancelled", run_id=self.run_id, status="cancelled")
            return self.status

    def _raise_if_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise RunCancelled("run cancellation requested")

    def _spawn(self, first_input: Any) -> None:
        def work() -> None:
            with self._status_lock:
                if self.status == "cancelled":
                    return
                if self._cancel_event.is_set():
                    self.status = "cancelled"
                    self.awaiting = None
                    BUS.emit(self.run_id, "cancelled", run_id=self.run_id, status="cancelled")
                    return
                self.status = "running"
            try:
                self._raise_if_cancelled()
                for chunk in self.app.stream(first_input, self.thread, stream_mode="updates"):
                    self._raise_if_cancelled()
                    if "__interrupt__" in chunk:
                        payload = chunk["__interrupt__"][0].value
                        self._raise_if_cancelled()
                        self.awaiting = payload
                        kind = payload.get("kind") if isinstance(payload, dict) else None
                        review_node = {
                            "human_privacy_review": "review_humans",
                            "trash_disagreement_review": "review_trash",
                        }.get(kind, "review")
                        with self._status_lock:
                            if self._cancel_event.is_set():
                                raise RunCancelled("run cancelled while awaiting input")
                            self.status = "awaiting_input"
                            BUS.emit(
                                self.run_id, "await_input", node=review_node, kind=kind,
                                payload=payload.get("items", []) if isinstance(payload, dict) else payload,
                            )
                        return
                self._raise_if_cancelled()
                with self._status_lock:
                    if self._cancel_event.is_set():
                        raise RunCancelled("run cancelled before completion")
                    self.status = "done"
                    BUS.emit(self.run_id, "done", run_id=self.run_id)
            except RunCancelled:
                with self._status_lock:
                    already_cancelled = self.status == "cancelled"
                    self.status = "cancelled"
                    self.awaiting = None
                if not already_cancelled:
                    BUS.emit(self.run_id, "cancelled", run_id=self.run_id, status="cancelled")
            except Exception as exc:  # noqa: BLE001 - surfaced to the UI
                log.exception("run %s failed", self.run_id)
                with self._status_lock:
                    self.status = "error"
                BUS.emit(self.run_id, "error", message=f"{type(exc).__name__}: {exc}")

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------ #
    def join(self, auto_resolve: bool = False, timeout: float = 3600.0) -> str:
        """
        Block until the run finishes, errors, or needs a human.

        With `auto_resolve`, every unresolved privacy or detection disagreement
        is explicitly excluded. Headless execution therefore fails closed and
        cannot silently promote uncertain model output.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            worker = self._worker
            if worker is not None:
                worker.join(timeout=1.0)
                if worker.is_alive():
                    continue

            if self.status != "awaiting_input":
                return self.status
            if not auto_resolve:
                return self.status

            payload = self.awaiting or {}
            kind = payload.get("kind")
            if kind in ("human_privacy_review", "trash_disagreement_review"):
                # Fail closed: every unresolved disagreement is explicitly excluded.
                self.resume({
                    "kind": kind,
                    "decisions": [
                        {"image_id": item["image_id"], "decision": "exclude",
                         "note": "Headless run: unresolved disagreement rejected."}
                        for item in payload.get("items", [])
                    ],
                    "auto": True,
                })
            else:
                log.error("cannot auto-resolve interrupt kind %r", kind)
                return self.status

        log.error("run %s timed out after %.0fs", self.run_id, timeout)
        return self.status

    def close(self) -> None:
        """Release the SQLite checkpointer. Safe to call twice."""
        if self._cm is not None:
            try:
                self._cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
            self._cm = None
        with Run._lock:
            Run._registry.pop(self.run_id, None)

    def snapshot(self) -> dict[str, Any]:
        try:
            values = self.app.get_state(self.thread).values
        except Exception:
            values = {}
        screen_progress: dict[str, dict[str, Any]] = {}
        tools: dict[str, dict[str, Any]] = {}
        for event in BUS.history(self.run_id):
            if event.get("type") == "screen_progress" and event.get("node"):
                screen_progress[event["node"]] = {
                    "image_id": event.get("image_id"),
                    "completed": event.get("completed", 0),
                    "total": event.get("total", 0),
                }
            elif event.get("type") == "tool" and event.get("node"):
                # Last call wins, so a reconnecting browser sees current state.
                tools[event["node"]] = {
                    "tool": event.get("tool"),
                    "detail": event.get("detail"),
                    "status": event.get("status"),
                    "duration_ms": event.get("duration_ms"),
                }
        aoi_manifest = (values.get("manifest") or {}).get("aoi") or {}
        return {
            "run_id": self.run_id,
            "status": self.status,
            "place": self.place,
            # Resolved area name, so a restored browser shows what was surveyed.
            "resolved_area": aoi_manifest.get("resolved"),
            "bbox": values.get("bbox"),
            "awaiting": self.awaiting,
            "nodes": NODES,
            "edges": GRAPH_EDGES,
            "screen_progress": screen_progress,
            "tools": tools,
            "coverage": values.get("coverage"),
            "privacy_stats": values.get("privacy_stats", {}),
            "detection_stats": values.get("detection_stats", {}),
            "sanitized_images": values.get("sanitized_images", []),
            "observations": values.get("observations", []),
            # Lets a reconnecting browser re-offer the download without a re-run.
            "citygml": values.get("citygml") or {},
            "report_md": values.get("report_md", ""),
        }
