"""
EventBus - SSE-style in-memory event stream

Provides publish/subscribe with history replay, session-scoped event filtering,
and SSE-formatted output. Pure Python with zero external dependencies.
"""

import uuid
import json
from datetime import datetime, timezone
from typing import Callable, Optional


# Event type constants
EVENT_SESSION_CREATED = "SESSION_CREATED"
EVENT_MESSAGE_ADDED = "MESSAGE_ADDED"
EVENT_TODO_CHANGED = "TODO_CHANGED"
EVENT_TOOL_CALLED = "TOOL_CALLED"
EVENT_COMPACTED = "COMPACTED"


class SessionEvent:
    """Single event in the stream"""

    def __init__(self, event_type: str, data: dict, session_id: str = "",
                 source: str = ""):
        self.event_id = f"EVT-{uuid.uuid4().hex[:8]}"
        self.event_type = event_type
        self.data = data or {}
        self.session_id = session_id
        self.source = source
        self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "data": self.data,
            "session_id": self.session_id,
            "source": self.source,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionEvent":
        e = cls(event_type=d.get("event_type", ""), data=d.get("data", {}),
                session_id=d.get("session_id", ""), source=d.get("source", ""))
        e.event_id = d.get("event_id", e.event_id)
        e.created_at = d.get("created_at", e.created_at)
        return e

    def to_sse(self) -> str:
        """Format as SSE (Server-Sent Events) line"""
        lines = []
        lines.append(f"id: {self.event_id}")
        lines.append(f"event: {self.event_type}")
        if self.session_id:
            lines.append(f"session: {self.session_id}")
        lines.append(f"data: {json.dumps(self.data, ensure_ascii=False)}")
        lines.append("")
        lines.append("")
        return "\n".join(lines)


class EventBus:
    """In-memory event bus with publish/subscribe and history replay

    Features:
      - publish(event_type, data): emit events to all subscribers
      - subscribe(callback, event_types): register handlers by event type
      - get_history(session_id, after_id): replay events from a checkpoint
      - to_sse_stream(): generate SSE-formatted output for streaming
    """

    def __init__(self, max_history: int = 10000):
        self._events: list[SessionEvent] = []
        self._subscribers: dict[str, list[Callable]] = {}
        self._max_history = max_history

    # -- Publish --

    def publish(self, event_type: str, data: dict, session_id: str = "",
                source: str = "") -> SessionEvent:
        """Publish an event and notify all matching subscribers"""
        event = SessionEvent(event_type=event_type, data=data,
                             session_id=session_id, source=source)
        self._events.append(event)

        # Trim history if over limit
        if len(self._events) > self._max_history:
            excess = len(self._events) - self._max_history
            self._events = self._events[excess:]

        # Notify subscribers for this event type
        callbacks = list(self._subscribers.get(event_type, []))
        for cb in callbacks:
            try:
                cb(event)
            except Exception:
                pass

        # Notify wildcard subscribers
        wildcard = list(self._subscribers.get("*", []))
        for cb in wildcard:
            try:
                cb(event)
            except Exception:
                pass

        return event

    # -- Subscribe --

    def subscribe(self, callback: Callable, event_types: list[str] | None = None):
        """Register a callback for specific event types or all events

        Args:
            callback: function receiving a SessionEvent
            event_types: list of event type strings, or None for wildcard (*)
        """
        if event_types is None:
            self._subscribers.setdefault("*", []).append(callback)
        else:
            for et in event_types:
                self._subscribers.setdefault(et, []).append(callback)

    def unsubscribe(self, callback: Callable, event_types: list[str] | None = None):
        """Remove a previously registered callback"""
        if event_types is None:
            # Remove from wildcard
            targets = ["*"]
        else:
            targets = event_types

        for key in targets:
            subs = self._subscribers.get(key, [])
            try:
                subs.remove(callback)
            except ValueError:
                pass

    def subscriber_count(self, event_type: str = "") -> int:
        """Count subscribers for a given event type"""
        if not event_type:
            return sum(len(v) for v in self._subscribers.values())
        return len(self._subscribers.get(event_type, []))

    # -- History / Replay --

    def get_history(self, session_id: str = "", after_id: str = "",
                    event_type: str = "", limit: int = 0) -> list[SessionEvent]:
        """Get event history with optional filters

        Args:
            session_id: filter by session (empty string = all sessions)
            after_id: return events published after this event ID
            event_type: filter by event type (empty string = all types)
            limit: maximum number of events to return (0 = no limit)
        """
        results = list(self._events)

        if session_id:
            results = [e for e in results if e.session_id == session_id]
        if event_type:
            results = [e for e in results if e.event_type == event_type]

        if after_id:
            idx = -1
            for i, e in enumerate(results):
                if e.event_id == after_id:
                    idx = i
                    break
            if idx >= 0:
                results = results[idx + 1:]

        if limit > 0:
            results = results[:limit]

        return results

    def get_last_event_id(self, session_id: str = "") -> Optional[str]:
        """Get the event_id of the most recent event (optionally for a session)"""
        events = self.get_history(session_id=session_id)
        if events:
            return events[-1].event_id
        return None

    def clear_history(self, session_id: str = ""):
        """Clear history, optionally only for a specific session"""
        if session_id:
            self._events = [e for e in self._events if e.session_id != session_id]
        else:
            self._events = []

    # -- SSE Stream Generation --

    def to_sse_stream(self, events: list[SessionEvent] | None = None) -> str:
        """Convert a list of events to SSE format string"""
        if events is None:
            events = self._events
        parts = []
        for event in events:
            parts.append(event.to_sse())
        return "".join(parts)

    def replay_from(self, session_id: str, after_id: str = "") -> list[SessionEvent]:
        """Convenience alias for get_history with session scope"""
        return self.get_history(session_id=session_id, after_id=after_id)

    # -- Query (for workflow compatibility) --

    def query(self, event_type: str = "", session_id: str = "") -> list[SessionEvent]:
        """Query events by type and/or session_id."""
        return self.get_history(session_id=session_id, event_type=event_type)

    def all_events(self) -> list[SessionEvent]:
        """Return all events."""
        return list(self._events)

    # -- Stats --

    def event_count(self, session_id: str = "") -> int:
        """Count total events (optionally scoped to a session)"""
        if session_id:
            return sum(1 for e in self._events if e.session_id == session_id)
        return len(self._events)

    def event_types_summary(self) -> dict[str, int]:
        """Return count of each event type"""
        summary: dict[str, int] = {}
        for e in self._events:
            summary[e.event_type] = summary.get(e.event_type, 0) + 1
        return summary
