"""EventBus in-memory event stream tests."""

import json
import pytest
from dr_mma.engine.event_bus import (
    SessionEvent, EventBus,
    EVENT_SESSION_CREATED, EVENT_MESSAGE_ADDED, EVENT_TODO_CHANGED,
    EVENT_TOOL_CALLED, EVENT_COMPACTED,
)


# -- SessionEvent model tests --

class TestSessionEvent:
    def test_auto_generate_id(self):
        e = SessionEvent(event_type="test", data={})
        assert e.event_id.startswith("EVT-")
        assert len(e.event_id) == 12

    def test_all_fields_set(self):
        e = SessionEvent(event_type="msg", data={"key": "val"},
                         session_id="S1", source="api")
        assert e.event_type == "msg"
        assert e.data == {"key": "val"}
        assert e.session_id == "S1"
        assert e.source == "api"

    def test_default_data_is_dict(self):
        e = SessionEvent(event_type="test", data=None)
        assert isinstance(e.data, dict)

    def test_to_dict_roundtrip(self):
        e = SessionEvent(event_type="test", data={"x": 1}, session_id="S1")
        d = e.to_dict()
        e2 = SessionEvent.from_dict(d)
        assert e2.event_type == "test"
        assert e2.data == {"x": 1}
        assert e2.session_id == "S1"

    def test_to_sse_format(self):
        e = SessionEvent(event_type="msg", data={"text": "hello"}, session_id="S1")
        sse = e.to_sse()
        assert f"id: {e.event_id}" in sse
        assert "event: msg" in sse
        assert "session: S1" in sse
        assert '"text": "hello"' in sse


# -- EventBus publish tests --

class TestEventBusPublish:
    def test_publish_returns_event(self):
        bus = EventBus()
        e = bus.publish(EVENT_SESSION_CREATED, {"loc": "/test"})
        assert isinstance(e, SessionEvent)
        assert e.event_type == EVENT_SESSION_CREATED

    def test_publish_stores_event(self):
        bus = EventBus()
        bus.publish(EVENT_MESSAGE_ADDED, {"role": "user"}, session_id="S1")
        assert bus.event_count() == 1

    def test_publish_with_session_id(self):
        bus = EventBus()
        bus.publish(EVENT_MESSAGE_ADDED, {}, session_id="S1")
        assert bus.event_count(session_id="S1") == 1

    def test_publish_multiple_events(self):
        bus = EventBus()
        bus.publish(EVENT_SESSION_CREATED, {}, session_id="S1")
        bus.publish(EVENT_MESSAGE_ADDED, {}, session_id="S1")
        bus.publish(EVENT_TODO_CHANGED, {}, session_id="S1")
        assert bus.event_count(session_id="S1") == 3

    def test_publish_event_types_summary(self):
        bus = EventBus()
        bus.publish(EVENT_SESSION_CREATED, {})
        bus.publish(EVENT_MESSAGE_ADDED, {})
        bus.publish(EVENT_MESSAGE_ADDED, {})
        summary = bus.event_types_summary()
        assert summary[EVENT_SESSION_CREATED] == 1
        assert summary[EVENT_MESSAGE_ADDED] == 2

    def test_max_history_trims_old_events(self):
        bus = EventBus(max_history=3)
        for i in range(5):
            bus.publish(EVENT_MESSAGE_ADDED, {"i": i})
        assert bus.event_count() == 3


# -- EventBus subscribe tests --

class TestEventBusSubscribe:
    def test_subscribe_specific_type(self):
        bus = EventBus()
        received = []
        bus.subscribe(received.append, [EVENT_MESSAGE_ADDED])
        bus.publish(EVENT_MESSAGE_ADDED, {"text": "hi"})
        assert len(received) == 1
        assert received[0].event_type == EVENT_MESSAGE_ADDED

    def test_subscribe_wildcard(self):
        bus = EventBus()
        received = []
        bus.subscribe(received.append)
        bus.publish(EVENT_SESSION_CREATED, {})
        bus.publish(EVENT_MESSAGE_ADDED, {})
        assert len(received) == 2

    def test_subscribe_filters_other_types(self):
        bus = EventBus()
        received = []
        bus.subscribe(received.append, [EVENT_MESSAGE_ADDED])
        bus.publish(EVENT_SESSION_CREATED, {})
        assert len(received) == 0

    def test_multiple_subscribers_same_type(self):
        bus = EventBus()
        r1 = []
        r2 = []
        bus.subscribe(r1.append, [EVENT_MESSAGE_ADDED])
        bus.subscribe(r2.append, [EVENT_MESSAGE_ADDED])
        bus.publish(EVENT_MESSAGE_ADDED, {})
        assert len(r1) == 1
        assert len(r2) == 1

    def test_unsubscribe(self):
        bus = EventBus()
        received = []
        bus.subscribe(received.append, [EVENT_MESSAGE_ADDED])
        bus.unsubscribe(received.append, [EVENT_MESSAGE_ADDED])
        bus.publish(EVENT_MESSAGE_ADDED, {})
        assert len(received) == 0

    def test_subscriber_count(self):
        bus = EventBus()
        bus.subscribe(lambda e: None, [EVENT_MESSAGE_ADDED])
        bus.subscribe(lambda e: None, [EVENT_MESSAGE_ADDED])
        assert bus.subscriber_count(EVENT_MESSAGE_ADDED) == 2

    def test_subscriber_count_total(self):
        bus = EventBus()
        bus.subscribe(lambda e: None, [EVENT_MESSAGE_ADDED])
        bus.subscribe(lambda e: None, [EVENT_SESSION_CREATED])
        assert bus.subscriber_count() == 2

    def test_subscribe_callback_exception_does_not_crash(self):
        bus = EventBus()
        def bad_callback(e):
            raise ValueError("boom")
        bus.subscribe(bad_callback, [EVENT_MESSAGE_ADDED])
        bus.publish(EVENT_MESSAGE_ADDED, {})


# -- EventBus history tests --

class TestEventBusHistory:
    def test_get_history_all(self):
        bus = EventBus()
        bus.publish(EVENT_SESSION_CREATED, {}, session_id="S1")
        bus.publish(EVENT_MESSAGE_ADDED, {}, session_id="S1")
        history = bus.get_history(session_id="S1")
        assert len(history) == 2

    def test_get_history_by_session(self):
        bus = EventBus()
        bus.publish(EVENT_MESSAGE_ADDED, {}, session_id="S1")
        bus.publish(EVENT_MESSAGE_ADDED, {}, session_id="S2")
        s1_events = bus.get_history(session_id="S1")
        assert len(s1_events) == 1

    def test_get_history_by_event_type(self):
        bus = EventBus()
        bus.publish(EVENT_SESSION_CREATED, {})
        bus.publish(EVENT_MESSAGE_ADDED, {})
        bus.publish(EVENT_MESSAGE_ADDED, {})
        msgs = bus.get_history(event_type=EVENT_MESSAGE_ADDED)
        assert len(msgs) == 2

    def test_get_history_after_id(self):
        bus = EventBus()
        e1 = bus.publish(EVENT_SESSION_CREATED, {})
        e2 = bus.publish(EVENT_MESSAGE_ADDED, {})
        e3 = bus.publish(EVENT_TODO_CHANGED, {})
        after = bus.get_history(after_id=e2.event_id)
        assert len(after) == 1
        assert after[0].event_id == e3.event_id

    def test_get_history_limit(self):
        bus = EventBus()
        for i in range(5):
            bus.publish(EVENT_MESSAGE_ADDED, {"i": i})
        limited = bus.get_history(limit=2)
        assert len(limited) == 2

    def test_get_last_event_id(self):
        bus = EventBus()
        e = bus.publish(EVENT_MESSAGE_ADDED, {})
        assert bus.get_last_event_id() == e.event_id

    def test_get_last_event_id_empty(self):
        bus = EventBus()
        assert bus.get_last_event_id() is None

    def test_replay_from(self):
        bus = EventBus()
        e1 = bus.publish(EVENT_SESSION_CREATED, {}, session_id="S1")
        e2 = bus.publish(EVENT_MESSAGE_ADDED, {}, session_id="S1")
        replayed = bus.replay_from("S1", after_id=e1.event_id)
        assert len(replayed) == 1
        assert replayed[0].event_id == e2.event_id


# -- EventBus SSE tests --

class TestEventBusSSE:
    def test_to_sse_stream(self):
        bus = EventBus()
        bus.publish(EVENT_MESSAGE_ADDED, {"text": "hello"}, session_id="S1")
        sse = bus.to_sse_stream()
        assert "event: MESSAGE_ADDED" in sse
        assert '"text": "hello"' in sse

    def test_to_sse_custom_events(self):
        bus = EventBus()
        e = SessionEvent(event_type="custom", data={"x": 1})
        sse = bus.to_sse_stream([e])
        assert "event: custom" in sse

    def test_sse_contains_event_id(self):
        bus = EventBus()
        e = bus.publish(EVENT_SESSION_CREATED, {})
        sse = bus.to_sse_stream()
        assert f"id: {e.event_id}" in sse


# -- EventBus clear tests --

class TestEventBusClear:
    def test_clear_all_history(self):
        bus = EventBus()
        bus.publish(EVENT_MESSAGE_ADDED, {})
        bus.clear_history()
        assert bus.event_count() == 0

    def test_clear_session_history(self):
        bus = EventBus()
        bus.publish(EVENT_MESSAGE_ADDED, {}, session_id="S1")
        bus.publish(EVENT_MESSAGE_ADDED, {}, session_id="S2")
        bus.clear_history(session_id="S1")
        assert bus.event_count(session_id="S1") == 0
        assert bus.event_count(session_id="S2") == 1


# -- Integration: Session lifecycle simulation --

class TestEventBusSessionLifecycle:
    def test_full_session_lifecycle(self):
        """Simulate a complete session: create, add messages, compact"""
        bus = EventBus()
        events_received = []
        bus.subscribe(events_received.append)

        # Create session
        bus.publish(EVENT_SESSION_CREATED, {"location": "/project"}, session_id="S1")

        # Add messages
        bus.publish(EVENT_MESSAGE_ADDED, {"role": "user", "content": "hello"}, session_id="S1")
        bus.publish(EVENT_MESSAGE_ADDED, {"role": "assistant", "content": "hi"}, session_id="S1")

        # Todo change
        bus.publish(EVENT_TODO_CHANGED, {"action": "add", "content": "review"}, session_id="S1")

        # Tool call
        bus.publish(EVENT_TOOL_CALLED, {"tool": "search", "result": "ok"}, session_id="S1")

        # Compaction
        bus.publish(EVENT_COMPACTED, {"original_count": 4, "new_count": 2}, session_id="S1")

        assert bus.event_count(session_id="S1") == 6
        assert len(events_received) == 6

        summary = bus.event_types_summary()
        assert summary[EVENT_SESSION_CREATED] == 1
        assert summary[EVENT_MESSAGE_ADDED] == 2
        assert summary[EVENT_TODO_CHANGED] == 1
        assert summary[EVENT_TOOL_CALLED] == 1
        assert summary[EVENT_COMPACTED] == 1

    def test_multi_session_isolation(self):
        """Verify events from different sessions are properly isolated"""
        bus = EventBus()
        bus.publish(EVENT_SESSION_CREATED, {}, session_id="A")
        bus.publish(EVENT_SESSION_CREATED, {}, session_id="B")
        bus.publish(EVENT_MESSAGE_ADDED, {}, session_id="A")
        bus.publish(EVENT_MESSAGE_ADDED, {}, session_id="B")

        a_events = bus.get_history(session_id="A")
        b_events = bus.get_history(session_id="B")
        assert len(a_events) == 2
        assert len(b_events) == 2

    def test_wildcard_and_specific_subscribers(self):
        """Verify both wildcard and specific subscribers receive events"""
        bus = EventBus()
        wildcard_received = []
        specific_received = []
        bus.subscribe(wildcard_received.append)
        bus.subscribe(specific_received.append, [EVENT_MESSAGE_ADDED])

        bus.publish(EVENT_SESSION_CREATED, {}, session_id="S1")
        bus.publish(EVENT_MESSAGE_ADDED, {}, session_id="S1")

        assert len(wildcard_received) == 2
        assert len(specific_received) == 1

    def test_subscribe_and_unsubscribe_wildcard(self):
        """Verify wildcard subscribe/unsubscribe works"""
        bus = EventBus()
        received = []
        bus.subscribe(received.append)
        bus.unsubscribe(received.append)
        bus.publish(EVENT_MESSAGE_ADDED, {})
        assert len(received) == 0
