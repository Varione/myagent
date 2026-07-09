"""Workflow event bus and event schemas."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid


EVENT_PROGRESS = "progress"
EVENT_LOW_CONFIDENCE = "low_confidence"
EVENT_REVIEW_FAILED = "review_failed"
EVENT_NEED_REPLAN = "need_replan"
EVENT_ROLE_ASSIGNED = "role_assigned"
EVENT_WORKFLOW_MODE = "workflow_mode"


@dataclass
class WorkflowEvent:
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
    """In-memory event stream used by the workflow engine."""

    def __init__(self):
        self._events: list[WorkflowEvent] = []

    def publish(self, event_type: str, source: str, task_id: str = "", payload: dict | None = None) -> WorkflowEvent:
        event = WorkflowEvent(
            event_type=event_type,
            source=source,
            task_id=task_id,
            payload=payload or {},
        )
        self._events.append(event)
        return event

    def query(self, event_type: str = "", task_id: str = "") -> list[WorkflowEvent]:
        events = list(self._events)
        if event_type:
            events = [event for event in events if event.event_type == event_type]
        if task_id:
            events = [event for event in events if event.task_id == task_id]
        return events

    def all_events(self) -> list[WorkflowEvent]:
        return list(self._events)
