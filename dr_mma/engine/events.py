"""
Workflow event bus and event schemas.

Unified event system:
- Wraps the canonical EventBus from event_bus.py (SessionEvent based, with SSE,
  pub/sub, and history replay).
- Provides WorkflowEvent for backward compatibility and workflow-specific API.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional
import uuid

from .event_bus import EventBus as _EventBus
from .event_bus import SessionEvent


EVENT_PROGRESS = "progress"
EVENT_LOW_CONFIDENCE = "low_confidence"
EVENT_REVIEW_FAILED = "review_failed"
EVENT_NEED_REPLAN = "need_replan"
EVENT_ROLE_ASSIGNED = "role_assigned"
EVENT_WORKFLOW_MODE = "workflow_mode"


@dataclass
class WorkflowEvent:
    """Workflow event dataclass for backward compatibility."""

    event_type: str
    source: str
    task_id: str = ""
    payload: dict = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: f"EV-{uuid.uuid4().hex[:8]}")
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source": self.source,
            "task_id": self.task_id,
            "payload": self.payload,
            "created_at": self.created_at,
        }


class EventBus:
    """
    Unified event bus for the workflow engine.

    Combines the canonical EventBus (pub/sub, SSE, history replay) with
    workflow-compatible publish/query interface.
    """

    def __init__(self, max_history: int = 10000):
        self._bus = _EventBus(max_history=max_history)

    # -- Publish (workflow-compatible) --

    def publish(
        self,
        event_type: str,
        source: str = "",
        task_id: str = "",
        payload: dict | None = None,
    ) -> WorkflowEvent:
        """Publish a workflow event. Compatible with the existing workflow API."""
        event = WorkflowEvent(
            event_type=event_type,
            source=source,
            task_id=task_id,
            payload=payload or {},
        )

        # Publish to the canonical bus (single source of truth)
        self._bus.publish(
            event_type,
            data=event.to_dict(),
            session_id=task_id,
            source=source,
        )
        return event

    # -- Subscribe (delegated to underlying bus) --

    def subscribe(self, callback: Callable, event_types: list[str] | None = None):
        """Register a callback for specific event types or all events."""
        self._bus.subscribe(callback, event_types)

    def unsubscribe(self, callback: Callable, event_types: list[str] | None = None):
        """Remove a previously registered callback."""
        self._bus.unsubscribe(callback, event_types)

    def subscriber_count(self, event_type: str = "") -> int:
        """Count subscribers for a given event type."""
        return self._bus.subscriber_count(event_type)

    # -- Query (workflow-compatible) --

    @staticmethod
    def _from_session_event(se: SessionEvent) -> WorkflowEvent:
        """Reconstruct a WorkflowEvent from a canonical SessionEvent's data."""
        d = se.data if isinstance(se.data, dict) else {}
        return WorkflowEvent(
            event_type=d.get("event_type", se.event_type),
            source=d.get("source", se.source),
            task_id=d.get("task_id", se.session_id),
            payload=d.get("payload", {}),
            event_id=d.get("event_id", se.event_id),
            created_at=d.get("created_at", se.created_at),
        )

    def query(self, event_type: str = "", task_id: str = "") -> list[WorkflowEvent]:
        """Query workflow events by type and/or task_id (delegated to canonical bus)."""
        results = self._bus.query(event_type=event_type, session_id=task_id)
        return [self._from_session_event(e) for e in results]

    def all_events(self) -> list[WorkflowEvent]:
        """Return all workflow events (delegated to canonical bus)."""
        return [self._from_session_event(e) for e in self._bus.all_events()]

    # -- History / SSE (delegated to underlying bus) --

    def get_history(
        self, session_id: str = "", after_id: str = "",
        event_type: str = "", limit: int = 0,
    ) -> list[SessionEvent]:
        """Get event history from the underlying bus."""
        return self._bus.get_history(
            session_id=session_id, after_id=after_id,
            event_type=event_type, limit=limit,
        )

    def get_last_event_id(self, session_id: str = "") -> Optional[str]:
        """Get the event_id of the most recent event."""
        return self._bus.get_last_event_id(session_id)

    def clear_history(self, session_id: str = ""):
        """Clear history, optionally only for a specific session."""
        self._bus.clear_history(session_id)

    def to_sse_stream(self, events: list[SessionEvent] | None = None) -> str:
        """Convert events to SSE format string."""
        return self._bus.to_sse_stream(events)

    def replay_from(self, session_id: str, after_id: str = "") -> list[SessionEvent]:
        """Replay events from a checkpoint."""
        return self._bus.replay_from(session_id, after_id)

    # -- Stats --

    def event_count(self, session_id: str = "") -> int:
        """Count total events (optionally scoped to a session)."""
        return self._bus.event_count(session_id)

    def event_types_summary(self) -> dict[str, int]:
        """Return count of each event type."""
        return self._bus.event_types_summary()
