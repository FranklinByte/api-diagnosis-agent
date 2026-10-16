"""Small vendor-neutral structured trace used by the V2 API and evaluator."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TraceEvent:
    name: str
    status: str = "ok"
    started_at: float = field(default_factory=time.perf_counter)
    elapsed_ms: Optional[float] = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def finish(self, *, status: Optional[str] = None, **attributes: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self.started_at) * 1000
        if status:
            self.status = status
        self.attributes.update(attributes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "elapsed_ms": round(self.elapsed_ms, 3) if self.elapsed_ms is not None else None,
            "attributes": self.attributes,
        }


@dataclass
class TraceRecorder:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    events: list[TraceEvent] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)

    def start(self, name: str, **attributes: Any) -> TraceEvent:
        event = TraceEvent(name=name, attributes=dict(attributes))
        self.events.append(event)
        return event

    def finish(self, event: TraceEvent, *, status: Optional[str] = None, **attributes: Any) -> None:
        event.finish(status=status, **attributes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "elapsed_ms": round((time.perf_counter() - self.started_at) * 1000, 3),
            "events": [event.as_dict() for event in self.events],
        }


__all__ = ["TraceEvent", "TraceRecorder"]
