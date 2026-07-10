"""Streaming engine tests -- StreamEvent, Producer, Consumer, Session."""

import json
import queue
import threading
import time

import pytest

from dr_mma.engine.streaming import (
    StreamEvent,
    StreamEventKind,
    StreamProducer,
    StreamConsumer,
    StreamSession,
)


# ===========================================================================
# 1. StreamEvent -- creation, serialization (5 tests)
# ===========================================================================

class TestStreamEvent:
    def test_creation_defaults(self):
        e = StreamEvent(kind=StreamEventKind.CHUNK)
        assert e.kind == StreamEventKind.CHUNK
        assert e.data == {}
        assert e.seq == 0
        assert isinstance(e.timestamp, float)

    def test_creation_with_data_and_seq(self):
        e = StreamEvent(
            kind=StreamEventKind.TOOL_CALL,
            data={"tool": "calc", "args": {"x": 1}},
            seq=5,
        )
        assert e.kind == StreamEventKind.TOOL_CALL
        assert e.data["tool"] == "calc"
        assert e.seq == 5

    def test_to_sse_format(self):
        e = StreamEvent(
            kind=StreamEventKind.CHUNK,
            data={"text": "hello"},
            seq=1,
        )
        sse = e.to_sse()
        assert "event: chunk" in sse
        assert '"text": "hello"' in sse
        assert sse.endswith("\n\n")

    def test_to_sse_unicode(self):
        e = StreamEvent(kind=StreamEventKind.CHUNK, data={"text": "\u4f60\u597d\u4e16\u754c"})
        sse = e.to_sse()
        assert "\u4f60\u597d\u4e16\u754c" in sse

    def test_to_dict(self):
        e = StreamEvent(
            kind=StreamEventKind.DONE,
            data={"status": "ok"},
            seq=3,
        )
        d = e.to_dict()
        assert d["kind"] == "done"
        assert d["data"] == {"status": "ok"}
        assert d["seq"] == 3
        assert isinstance(d["timestamp"], float)


# ===========================================================================
# 2. StreamProducer -- subscribe, send, close, history (10 tests)
# ===========================================================================

class TestStreamProducer:
    def test_auto_stream_id(self):
        p = StreamProducer()
        assert len(p.stream_id) == 8

    def test_custom_stream_id(self):
        p = StreamProducer(stream_id="myid")
        assert p.stream_id == "myid"

    def test_subscribe_returns_queue(self):
        p = StreamProducer()
        q = p.subscribe()
        assert isinstance(q, queue.Queue)

    def test_subscribe_increases_consumer_count(self):
        p = StreamProducer()
        p.subscribe()
        p.subscribe()
        assert p.consumer_count == 2

    def test_send_chunk(self):
        p = StreamProducer()
        q = p.subscribe()
        e = p.send_chunk("hello")
        assert e.kind == StreamEventKind.CHUNK
        assert e.data["text"] == "hello"
        received = q.get_nowait()
        assert received.seq == 1

    def test_send_tool_call(self):
        p = StreamProducer()
        q = p.subscribe()
        p.send_tool_call("search", {"q": "test"})
        e = q.get_nowait()
        assert e.kind == StreamEventKind.TOOL_CALL
        assert e.data["tool"] == "search"

    def test_send_tool_result(self):
        p = StreamProducer()
        q = p.subscribe()
        p.send_tool_result("search", {"results": []}, success=True)
        e = q.get_nowait()
        assert e.kind == StreamEventKind.TOOL_RESULT
        assert e.data["success"] is True

    def test_send_thinking(self):
        p = StreamProducer()
        q = p.subscribe()
        p.send_thinking("Let me think...")
        e = q.get_nowait()
        assert e.kind == StreamEventKind.THINKING
        assert e.data["content"] == "Let me think..."

    def test_send_after_close_raises(self):
        p = StreamProducer()
        p.close()
        with pytest.raises(RuntimeError, match="already closed"):
            p.send_chunk("boom")

    def test_history_grows(self):
        p = StreamProducer()
        p.send_chunk("a")
        p.send_chunk("b")
        p.send_thinking("x")
        assert len(p.history) == 3


# ===========================================================================
# 3. StreamProducer -- multi-consumer broadcast (4 tests)
# ===========================================================================

class TestStreamProducerBroadcast:
    def test_two_consumers_receive_same_event(self):
        p = StreamProducer()
        q1 = p.subscribe()
        q2 = p.subscribe()
        p.send_chunk("hi")
        e1 = q1.get_nowait()
        e2 = q2.get_nowait()
        assert e1.data["text"] == "hi"
        assert e2.data["text"] == "hi"

    def test_three_consumers_all_receive(self):
        p = StreamProducer()
        qs = [p.subscribe() for _ in range(3)]
        p.send_chunk("broadcast")
        for q in qs:
            assert q.get_nowait().data["text"] == "broadcast"

    def test_events_delivered_in_order(self):
        p = StreamProducer()
        q = p.subscribe()
        for i in range(5):
            p.send_chunk(str(i))
        seqs = [q.get_nowait().seq for _ in range(5)]
        assert seqs == [1, 2, 3, 4, 5]

    def test_consumer_count_after_subscribe(self):
        p = StreamProducer()
        for _ in range(4):
            p.subscribe()
        assert p.consumer_count == 4


# ===========================================================================
# 4. StreamProducer -- full-queue eviction (3 tests)
# ===========================================================================

class TestStreamProducerEviction:
    def test_full_queue_consumer_evicted(self):
        p = StreamProducer()
        q = p.subscribe(buffer_size=2)
        p.send_chunk("1")
        p.send_chunk("2")
        # queue is full; next send should evict
        p.send_chunk("3")
        assert p.consumer_count == 0

    def test_evicted_consumer_not_receives_further_events(self):
        p = StreamProducer()
        small_q = p.subscribe(buffer_size=1)
        alive_q = p.subscribe(buffer_size=100)
        p.send_chunk("fill")
        # small_q is now full. Next send evicts small_q, only alive_q gets it.
        p.send_chunk("overflow")
        assert p.consumer_count == 1
        # Drain alive_q: first "fill", then "overflow"
        e1 = alive_q.get_nowait()
        assert e1.data["text"] == "fill"
        e2 = alive_q.get_nowait()
        assert e2.data["text"] == "overflow"

    def test_mixed_alive_and_dead(self):
        p = StreamProducer()
        small = p.subscribe(buffer_size=1)
        large = p.subscribe(buffer_size=10)
        p.send_chunk("a")  # both get it
        p.send_chunk("b")  # small is full, evicted; large gets it
        p.send_chunk("c")  # only large exists
        assert p.consumer_count == 1
        # large should have all three: a, b, c
        events = []
        while not large.empty():
            events.append(large.get_nowait())
        assert len(events) == 3
        assert [e.data["text"] for e in events] == ["a", "b", "c"]


# ===========================================================================
# 5. StreamConsumer -- receive, timeout, filter (6 tests)
# ===========================================================================

class TestStreamConsumer:
    def test_receive_immediate(self):
        p = StreamProducer()
        q = p.subscribe()
        c = StreamConsumer(q, p)
        p.send_chunk("hello")
        e = c.receive(timeout=0.1)
        assert e is not None
        assert e.data["text"] == "hello"

    def test_receive_timeout_returns_none(self):
        p = StreamProducer()
        q = p.subscribe()
        c = StreamConsumer(q, p)
        result = c.receive(timeout=0.05)
        assert result is None

    def test_receive_all(self):
        p = StreamProducer()
        q = p.subscribe()
        c = StreamConsumer(q, p)
        p.send_chunk("a")
        p.send_chunk("b")
        events = c.receive_all(timeout=0.1)
        assert len(events) == 2

    def test_filter_keeps_matching(self):
        p = StreamProducer()
        q = p.subscribe()
        c = StreamConsumer(q, p)
        c.set_filter({StreamEventKind.CHUNK})
        p.send_chunk("keep")
        p.send_thinking("drop")
        e1 = c.receive(timeout=0.1)
        assert e1 is not None
        assert e1.kind == StreamEventKind.CHUNK

    def test_filter_drops_non_matching(self):
        p = StreamProducer()
        q = p.subscribe()
        c = StreamConsumer(q, p)
        c.set_filter({StreamEventKind.DONE})
        p.send_chunk("drop me")
        e = c.receive(timeout=0.1)
        assert e is None

    def test_buffer_grows(self):
        p = StreamProducer()
        q = p.subscribe()
        c = StreamConsumer(q, p)
        p.send_chunk("x")
        p.send_chunk("y")
        c.receive(timeout=0.1)
        c.receive(timeout=0.1)
        assert len(c.buffer) == 2


# ===========================================================================
# 6. StreamConsumer -- chunks, is_done (4 tests)
# ===========================================================================

class TestStreamConsumerChunks:
    def test_chunks_concatenates(self):
        p = StreamProducer()
        q = p.subscribe()
        c = StreamConsumer(q, p)
        p.send_chunk("Hello ")
        p.send_chunk("World")
        c.receive(timeout=0.1)
        c.receive(timeout=0.1)
        assert c.chunks == "Hello World"

    def test_chunks_empty_when_no_chunk_events(self):
        p = StreamProducer()
        q = p.subscribe()
        c = StreamConsumer(q, p)
        p.send_thinking("thinking")
        c.receive(timeout=0.1)
        assert c.chunks == ""

    def test_is_done_after_done_event(self):
        p = StreamProducer()
        q = p.subscribe()
        c = StreamConsumer(q, p)
        p.close()
        c.receive(timeout=0.1)
        assert c.is_done is True

    def test_is_done_after_error_event(self):
        p = StreamProducer()
        q = p.subscribe()
        c = StreamConsumer(q, p)
        p.close(error="boom")
        c.receive(timeout=0.1)
        assert c.is_done is True


# ===========================================================================
# 7. StreamSession -- full lifecycle (5 tests)
# ===========================================================================

class TestStreamSession:
    def test_default_stream_id(self):
        s = StreamSession()
        assert len(s.stream_id) == 8

    def test_custom_stream_id(self):
        s = StreamSession(stream_id="sess1")
        assert s.stream_id == "sess1"

    def test_add_consumer_returns_consumer(self):
        s = StreamSession()
        c = s.add_consumer()
        assert isinstance(c, StreamConsumer)

    def test_send_delivers_to_consumer(self):
        s = StreamSession()
        c = s.add_consumer()
        s.send_chunk("hi")
        e = c.receive(timeout=0.1)
        assert e.data["text"] == "hi"

    def test_close_returns_history(self):
        s = StreamSession()
        s.send_chunk("a")
        s.send_chunk("b")
        history = s.close()
        assert len(history) == 3  # a, b, DONE


# ===========================================================================
# 8. StreamSession -- callbacks (4 tests)
# ===========================================================================

class TestStreamSessionCallbacks:
    def test_on_event_receives_events(self):
        s = StreamSession()
        received = []
        s.on_event(received.append)
        s.send_chunk("cb-test")
        assert len(received) == 1
        assert received[0].data["text"] == "cb-test"

    def test_multiple_callbacks(self):
        s = StreamSession()
        r1 = []
        r2 = []
        s.on_event(r1.append)
        s.on_event(r2.append)
        s.send_chunk("multi")
        assert len(r1) == 1
        assert len(r2) == 1

    def test_callback_exception_does_not_crash(self):
        s = StreamSession()
        def bad(e):
            raise ValueError("boom")
        s.on_event(bad)
        s.send_chunk("safe")  # should not raise

    def test_callback_receives_all_kinds(self):
        s = StreamSession()
        received = []
        s.on_event(received.append)
        s.send_chunk("x")
        s.send_thinking("y")
        s.close()
        assert len(received) == 3


# ===========================================================================
# 9. SSE output format (3 tests)
# ===========================================================================

class TestSSEOutput:
    def test_session_sse_output(self):
        s = StreamSession()
        s.send_chunk("hello")
        s.close()
        out = s.sse_output()
        assert "event: chunk" in out
        assert "event: done" in out
        assert '"text": "hello"' in out

    def test_sse_block_ends_with_double_newline(self):
        e = StreamEvent(kind=StreamEventKind.CHUNK, data={"t": 1})
        sse = e.to_sse()
        assert sse.endswith("\n\n")

    def test_sse_json_parsable(self):
        e = StreamEvent(kind=StreamEventKind.TOOL_CALL, data={"tool": "calc"})
        sse = e.to_sse()
        lines = sse.strip().split("\n")
        data_line = [l for l in lines if l.startswith("data: ")][0]
        payload = json.loads(data_line.split("data: ", 1)[1])
        assert payload["tool"] == "calc"


# ===========================================================================
# 10. Sequence numbering (3 tests)
# ===========================================================================

class TestSequenceNumbering:
    def test_seq_starts_at_1(self):
        p = StreamProducer()
        e = p.send_chunk("first")
        assert e.seq == 1

    def test_seq_increments(self):
        p = StreamProducer()
        e1 = p.send_chunk("a")
        e2 = p.send_chunk("b")
        e3 = p.send_chunk("c")
        assert e1.seq == 1
        assert e2.seq == 2
        assert e3.seq == 3

    def test_seq_in_history(self):
        p = StreamProducer()
        p.send_chunk("a")
        p.send_chunk("b")
        hist = p.history
        assert [e.seq for e in hist] == [1, 2]


# ===========================================================================
# 11. Thread safety (3 tests)
# ===========================================================================

class TestThreadSafety:
    def test_concurrent_sends(self):
        p = StreamProducer()
        q = p.subscribe(buffer_size=200)
        errors = []

        def sender(n):
            try:
                for i in range(10):
                    p.send_chunk(f"{n}-{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=sender, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors
        count = 0
        while not q.empty():
            q.get_nowait()
            count += 1
        assert count == 40

    def test_close_is_thread_safe(self):
        """Ensure close() eventually sets is_closed even under contention."""
        p = StreamProducer()
        barrier = threading.Event()

        def closer():
            barrier.wait()  # wait for signal
            p.close()

        closer_thread = threading.Thread(target=closer)
        closer_thread.start()

        # Let the main thread do some work first, then signal close
        time.sleep(0.05)
        barrier.set()
        closer_thread.join(timeout=2)

        assert p.is_closed

    def test_subscribe_thread_safe(self):
        p = StreamProducer()
        errors = []

        def subscriber():
            try:
                p.subscribe()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=subscriber) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2)
        assert not errors
        assert p.consumer_count == 10


# ===========================================================================
# 12. Producer close and history (3 tests)
# ===========================================================================

class TestProducerCloseAndHistory:
    def test_close_normal(self):
        p = StreamProducer()
        p.send_chunk("x")
        history = p.close()
        assert len(history) == 2  # chunk + DONE
        assert history[-1].kind == StreamEventKind.DONE

    def test_close_with_error(self):
        p = StreamProducer()
        p.send_chunk("x")
        history = p.close(error="fail")
        assert len(history) == 2
        assert history[-1].kind == StreamEventKind.ERROR
        assert history[-1].data["error"] == "fail"

    def test_history_is_copy(self):
        p = StreamProducer()
        p.send_chunk("a")
        h1 = p.history
        p.send_chunk("b")
        h2 = p.history
        assert len(h1) == 1
        assert len(h2) == 2


# ===========================================================================
# 13. Consumer edge cases (3 tests)
# ===========================================================================

class TestConsumerEdgeCases:
    def test_receive_all_empty(self):
        p = StreamProducer()
        q = p.subscribe()
        c = StreamConsumer(q, p)
        events = c.receive_all(timeout=0.05)
        assert events == []

    def test_chunks_ignores_non_chunk_events(self):
        p = StreamProducer()
        q = p.subscribe()
        c = StreamConsumer(q, p)
        p.send_thinking("think")
        p.send_chunk("A")
        p.send_tool_call("t", {})
        c.receive(timeout=0.1)
        c.receive(timeout=0.1)
        c.receive(timeout=0.1)
        assert c.chunks == "A"

    def test_is_done_false_before_done(self):
        p = StreamProducer()
        q = p.subscribe()
        c = StreamConsumer(q, p)
        p.send_chunk("not done yet")
        c.receive(timeout=0.1)
        assert c.is_done is False


# ===========================================================================
# 14. StreamSession sse_output (2 tests)
# ===========================================================================

class TestSessionSSE:
    def test_sse_contains_all_events(self):
        s = StreamSession()
        s.send_chunk("A")
        s.send_thinking("B")
        s.close()
        out = s.sse_output()
        assert "event: chunk" in out
        assert "event: thinking" in out
        assert "event: done" in out

    def test_sse_empty_session(self):
        s = StreamSession()
        s.close()
        out = s.sse_output()
        assert "event: done" in out
        assert "event: chunk" not in out


# ===========================================================================
# 15. StreamEventKind enum coverage (2 tests)
# ===========================================================================

class TestStreamEventKindEnum:
    def test_all_kinds_have_values(self):
        for kind in StreamEventKind:
            assert isinstance(kind.value, str)
            assert len(kind.value) > 0

    def test_kind_count(self):
        kinds = [e.name for e in StreamEventKind]
        expected = ["CHUNK", "TOOL_CALL", "TOOL_RESULT", "THINKING", "DONE", "ERROR"]
        assert kinds == expected
