"""JSON-safe graph state, graph metadata, and the run event bus."""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from typing import Annotated, Any, Iterator, Literal, TypedDict

from langgraph.errors import GraphBubbleUp

from .schemas import Observation

NODES: list[dict[str, str]] = [
    {"id": "aoi", "label": "Resolve area of interest", "phase": "acquire"},
    {"id": "coverage_audit", "label": "Audit imagery availability", "phase": "acquire"},
    {"id": "fetch", "label": "Fetch source imagery", "phase": "acquire"},
    {"id": "screen_trash", "label": "Screen original images for roadside trash", "phase": "detect"},
    {"id": "segment_trash", "label": "Segment trash and select candidates", "phase": "detect"},
    {"id": "blur_humans", "label": "Detect and blur people (SAM3)", "phase": "privacy"},
    {"id": "review_trash", "label": "Human-in-the-loop review", "phase": "review"},
    {"id": "export_citygml", "label": "Export reviewed trash to CityGML 2.0", "phase": "export"},
    {"id": "report", "label": "Write factual artifacts", "phase": "report"},
]

_NODE_IDS = [node["id"] for node in NODES]
GRAPH_EDGES: list[dict[str, str]] = [
    {"source": source, "target": target}
    for source, target in zip(_NODE_IDS, _NODE_IDS[1:])
]

EventType = Literal[
    "run_start", "node_start", "node_end", "log", "artifact",
    "screen_progress", "tool", "await_input", "input_received", "paused",
    "cancel_requested", "cancelled", "done", "error",
]


class RunCancelled(RuntimeError):
    """Cooperative cancellation signal raised at explicit safe points."""


def _merge(left: dict | None, right: dict | None) -> dict:
    return {**(left or {}), **(right or {})}


class AgentState(TypedDict, total=False):
    run_id: str
    prompt: str
    place: str
    bbox: list[float]
    aoi_geojson: dict[str, Any]
    epsg: int

    coverage: dict[str, Any]
    images: list[dict[str, Any]]

    sanitized_images: list[dict[str, Any]]
    trash_screens: list[dict[str, Any]]
    trash_candidate_ids: list[str]
    trash_review_queue: list[str]
    trash_reviews: list[dict[str, Any]]
    observations: list[dict[str, Any]]
    privacy_stats: dict[str, Any]
    detection_stats: dict[str, Any]
    citygml: dict[str, Any]
    report_md: str

    manifest: Annotated[dict[str, Any], _merge]
    errors: list[str]


class EventBus:
    """Append-only per-run event history with reconnect-safe SSE cursors."""

    def __init__(self) -> None:
        self._queues: dict[str, threading.Event] = {}
        self._history: dict[str, list[dict]] = {}
        self._lock = threading.Lock()

    def channel(self, run_id: str) -> threading.Event:
        with self._lock:
            if run_id not in self._queues:
                self._queues[run_id] = threading.Event()
                self._history[run_id] = []
            return self._queues[run_id]

    def emit(self, run_id: str, event_type: EventType, /, **payload: Any) -> None:
        event = {"type": event_type, "run_id": run_id, "ts": time.time(), **payload}
        signal = self.channel(run_id)
        with self._lock:
            self._history[run_id].append(event)
        signal.set()
        signal.clear()

    def history(self, run_id: str) -> list[dict]:
        with self._lock:
            return list(self._history.get(run_id, []))

    def sse(self, run_id: str, from_index: int = 0):
        signal = self.channel(run_id)
        cursor = from_index
        while True:
            batch = self.history(run_id)[cursor:]
            for event in batch:
                cursor += 1
                yield f"id: {cursor}\ndata: {json.dumps(event)}\n\n"
                if event["type"] in ("cancelled", "done", "error"):
                    return
            if not signal.wait(timeout=15.0):
                yield ": keepalive\n\n"


BUS = EventBus()


class NodeReporter:
    def __init__(
        self,
        run_id: str,
        bus: EventBus = BUS,
        check_cancel=None,
    ) -> None:
        self.run_id = run_id
        self.bus = bus
        self._check_cancel = check_cancel or (lambda: None)
        self._node: str | None = None
        self._t0 = 0.0

    def step(self, node: str) -> "NodeReporter":
        self._node = node
        return self

    def __enter__(self) -> "NodeReporter":
        self._check_cancel()
        self._t0 = time.time()
        self.bus.emit(self.run_id, "node_start", node=self._node)
        return self

    def __exit__(self, exc_type, exc, _tb) -> bool:
        elapsed = int((time.time() - self._t0) * 1000)
        if exc is None:
            self.bus.emit(self.run_id, "node_end", node=self._node, duration_ms=elapsed)
        elif isinstance(exc, GraphBubbleUp):
            self.bus.emit(self.run_id, "paused", node=self._node, duration_ms=elapsed)
        elif isinstance(exc, RunCancelled):
            self.bus.emit(
                self.run_id, "node_end", node=self._node,
                duration_ms=elapsed, cancelled=True,
            )
        else:
            self.bus.emit(
                self.run_id, "error", node=self._node,
                message=f"{type(exc).__name__}: {exc}", duration_ms=elapsed,
            )
        return False

    def log(self, message: str, **extra: Any) -> None:
        self.bus.emit(self.run_id, "log", node=self._node, message=message, **extra)

    @contextmanager
    def tool(self, name: str, detail: str = "", **extra: Any) -> Iterator[None]:
        """
        Announce one external call - an API, a model, a solver - around the work.

        This is the audit trail the UI renders as the agent's tool trace, so the
        name is the real callable being invoked, not a friendly paraphrase. A
        failed call is reported as failed rather than silently omitted.
        """
        node = self._node
        started = time.time()
        self.bus.emit(
            self.run_id, "tool", node=node, tool=name, detail=detail,
            status="start", **extra,
        )
        try:
            yield
        except (RunCancelled, GraphBubbleUp):
            self.bus.emit(
                self.run_id, "tool", node=node, tool=name, detail=detail,
                status="cancelled", duration_ms=int((time.time() - started) * 1000),
            )
            raise
        except Exception as exc:
            self.bus.emit(
                self.run_id, "tool", node=node, tool=name, detail=detail,
                status="error", message=f"{type(exc).__name__}: {exc}",
                duration_ms=int((time.time() - started) * 1000),
            )
            raise
        else:
            self.bus.emit(
                self.run_id, "tool", node=node, tool=name, detail=detail,
                status="ok", duration_ms=int((time.time() - started) * 1000),
            )

    def artifact(self, kind: str, data: Any) -> None:
        self.bus.emit(self.run_id, "artifact", node=self._node, kind=kind, data=data)

    def screen_progress(self, image_id: str, completed: int, total: int) -> None:
        self.bus.emit(
            self.run_id, "screen_progress", node=self._node,
            image_id=image_id, completed=completed, total=total,
        )



def as_dump(model) -> dict:
    return model.model_dump(mode="json")


def dumps(models) -> list[dict]:
    return [model.model_dump(mode="json") for model in models]


def load_observations(state: AgentState) -> list[Observation]:
    return [Observation.model_validate(item) for item in state.get("observations", [])]


