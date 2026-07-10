"""DR-MMA Integration Tests Part A: Streaming + WindowManager + MCPClient cross-module collaboration.

Scenarios:
    1. Streaming + WindowManager linkage (chunk streaming into window, token budget, snapshots)
    2. SSE output format validation
    3. Multi-consumer + WindowManager coordination
    4. Callback auto-write to WindowManager
    5. MCPClient data classes interacting with WindowManager
    6. Error and edge-case handling across modules
    7. Concurrent streaming + window operations
    8. Window snapshot + stream history reconciliation
"""

import json
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
from dr_mma.engine.window_manager import (
    MessageRole,
    WindowConfig,
    WindowMessage,
    WindowManager,
)


# ============================================================================
# Scenario 1: Streaming + WindowManager linkage
# ============================================================================

class TestStreamingWindowIntegration:
    """StreamSession events feed into WindowManager in real time."""

    def test_stream_chunks_fill_window(self):
        """Streaming CHUNK events are written to window_manager; tokens accumulate."""
        wm = WindowManager(WindowConfig(max_tokens=100, reserve_tokens=10))
        session = StreamSession()
        consumer = session.add_consumer()

        for i in range(10):
            chunk = f"Chunk {i}: " + "x" * 20
            session.send_chunk(chunk)
            wm.add_user(chunk, importance=1.0)

        assert wm.total_tokens > 0
        assert wm.message_count == 10

    def test_window_trims_during_streaming(self):
        """Window auto-trims when streaming exceeds budget."""
        wm = WindowManager(WindowConfig(max_tokens=50, reserve_tokens=5))
        session = StreamSession()

        for i in range(20):
            chunk = f"Long segment {i}: " + "data" * 10
            session.send_chunk(chunk)
            wm.add_assistant(chunk, importance=2.0)

        assert wm.total_tokens <= wm.config.usable_tokens or wm.message_count < 20

    def test_stream_done_triggers_snapshot(self):
        """After stream close, window snapshot is consistent."""
        wm = WindowManager(WindowConfig(max_tokens=1000))
        session = StreamSession()
        consumer = session.add_consumer()

        session.send_chunk("Hello")
        wm.add_user("Hello")
        session.close()

        snap = wm.snapshot()
        assert snap.total_tokens > 0
        assert not consumer.is_done or session.producer.is_closed

    def test_streaming_with_token_budget_constraint(self):
        """Window enforces token budget while streaming large chunks."""
        cfg = WindowConfig(max_tokens=60, reserve_tokens=5, min_keep_count=3)
        wm = WindowManager(config=cfg)
        session = StreamSession()

        for i in range(15):
            chunk = "A" * 40
            session.send_chunk(chunk)
            wm.add_assistant(chunk)

        assert wm.total_tokens <= cfg.usable_tokens
        assert wm.message_count < 15

    def test_streaming_error_closes_producer(self):
        """Error close on stream sets producer closed flag."""
        wm = WindowManager()
        session = StreamSession()
        session.send_chunk("before error")
        wm.add_user("before error")

        history = session.close(error="simulated failure")
        assert session.producer.is_closed is True
        assert any(e.kind == StreamEventKind.ERROR for e in history)

    def test_window_stats_after_streaming_session(self):
        """WindowManager stats reflect streaming activity."""
        wm = WindowManager(WindowConfig(max_tokens=500, reserve_tokens=10))
        session = StreamSession()

        for i in range(5):
            session.send_chunk(f"msg{i}")
            wm.add_user(f"msg{i}")

        session.close()
        stats = wm.stats
        assert stats["message_count"] == 5
        assert stats["total_tokens"] > 0
        assert "total_dropped" in stats


# ============================================================================
# Scenario 2: SSE output format validation
# ============================================================================

class TestSSEOutputFormat:
    """Verify SSE output contains all event types and is parseable."""

    def test_sse_complete_stream(self):
        """Complete streaming session SSE output contains all event kinds."""
        session = StreamSession("test-sse")
        session.send_chunk("Part 1")
        session.send_thinking("Analyzing...")
        session.send_tool_call("search", {"query": "test"})
        session.send_tool_result("search", {"results": ["a"]}, success=True)
        session.send_chunk("Part 2")
        session.close()

        sse = session.sse_output()
        assert "event: chunk" in sse
        assert "event: thinking" in sse
        assert "event: tool_call" in sse
        assert "event: tool_result" in sse
        assert "event: done" in sse

    def test_sse_parseable(self):
        """SSE output is line-by-line parseable."""
        session = StreamSession()
        session.send_chunk("Line 1")
        session.send_chunk("Line 2")
        session.close()

        for line in session.sse_output().splitlines():
            stripped = line.strip()
            if stripped:
                assert stripped.startswith("event:") or stripped.startswith("data:")

    def test_sse_contains_all_event_kinds(self):
        """SSE output covers every StreamEventKind except ERROR."""
        session = StreamSession()
        session.send_chunk("text")
        session.send_thinking("reasoning")
        session.send_tool_call("tool1", {})
        session.send_tool_result("tool1", {}, success=True)
        session.close()

        sse = session.sse_output()
        for kind in [StreamEventKind.CHUNK, StreamEventKind.THINKING,
                     StreamEventKind.TOOL_CALL, StreamEventKind.TOOL_RESULT,
                     StreamEventKind.DONE]:
            assert f"event: {kind.value}" in sse

    def test_sse_json_data_parseable(self):
        """Each data line in SSE contains valid JSON."""
        session = StreamSession()
        session.send_chunk("Hello")
        session.send_tool_call("calc", {"expr": "1+1"})
        session.close()

        sse = session.sse_output()
        for line in sse.splitlines():
            if line.startswith("data:"):
                payload = line[len("data:"):]
                json.loads(payload.strip())

    def test_sse_empty_session_has_done(self):
        """Empty session still produces DONE event in SSE."""
        session = StreamSession()
        session.close()
        sse = session.sse_output()
        assert "event: done" in sse
        assert "event: chunk" not in sse

    def test_sse_preserves_unicode(self):
        """SSE output preserves non-ASCII characters."""
        session = StreamSession()
        session.send_chunk("\u4f60\u597d")
        session.close()
        sse = session.sse_output()
        assert "\u4f60\u597d" in sse


# ============================================================================
# Scenario 3: Multi-consumer + WindowManager
# ============================================================================

class TestMultiConsumerWindow:
    """Multiple consumers reading from same stream, writing to one window."""

    def test_multiple_consumers_same_window(self):
        """Three consumers read same stream; events written to shared window."""
        wm = WindowManager(WindowConfig(max_tokens=500))
        session = StreamSession()

        consumers = [session.add_consumer() for _ in range(3)]

        for i in range(5):
            session.send_chunk(f"Message {i}")

        for consumer in consumers:
            events = consumer.receive_all(timeout=0.1)
            for e in events:
                if e.kind == StreamEventKind.CHUNK:
                    wm.add_user(e.data["text"])

        assert wm.message_count >= 5

    def test_consumer_filter_reduces_window_entries(self):
        """Filtered consumers only pass specific event types to window."""
        wm = WindowManager(WindowConfig(max_tokens=500))
        session = StreamSession()
        consumer = session.add_consumer()
        consumer.set_filter({StreamEventKind.CHUNK})

        session.send_chunk("Visible")
        session.send_thinking("Hidden thinking")
        session.send_tool_call("hidden_tool", {})
        session.close()

        for e in consumer.receive_all(timeout=0.1):
            wm.add_user(e.data.get("text", ""))

        messages = wm.get_messages()
        contents = [m.content for m in messages if m.content]
        assert "Visible" in contents

    def test_consumer_chunks_property_matches_window(self):
        """Consumer chunks property matches window assistant content."""
        wm = WindowManager(WindowConfig(max_tokens=1000))
        session = StreamSession()
        consumer = session.add_consumer()

        parts = ["Hello ", "World", "!"]
        for part in parts:
            session.send_chunk(part)
            wm.add_assistant(part)

        consumer.receive_all(timeout=0.1)
        expected = "".join(parts)
        assert consumer.chunks == expected
        # Each chunk is stored as a separate message, so check individual parts
        prompt = wm.to_prompt()
        for part in parts:
            assert part.strip() in prompt

    def test_each_consumer_gets_independent_copy(self):
        """Each consumer maintains its own event buffer."""
        session = StreamSession()
        c1 = session.add_consumer()
        c2 = session.add_consumer()

        session.send_chunk("shared")
        session.close()

        events1 = c1.receive_all(timeout=0.1)
        events2 = c2.receive_all(timeout=0.1)
        assert len(events1) == len(events2)
        chunk1 = [e for e in events1 if e.kind == StreamEventKind.CHUNK]
        chunk2 = [e for e in events2 if e.kind == StreamEventKind.CHUNK]
        assert len(chunk1) == 1
        assert len(chunk2) == 1

    def test_multi_consumer_with_mixed_event_types(self):
        """Multiple consumers handle mixed event types correctly."""
        wm = WindowManager(WindowConfig(max_tokens=500))
        session = StreamSession()
        consumer = session.add_consumer()

        session.send_chunk("chunk1")
        session.send_thinking("think1")
        session.send_tool_call("t", {})
        session.send_tool_result("t", {}, success=True)
        session.send_chunk("chunk2")
        session.close()

        events = consumer.receive_all(timeout=0.1)
        kinds = [e.kind for e in events]
        assert StreamEventKind.CHUNK in kinds
        assert StreamEventKind.THINKING in kinds
        assert StreamEventKind.TOOL_CALL in kinds
        assert StreamEventKind.TOOL_RESULT in kinds
        assert StreamEventKind.DONE in kinds

    def test_consumer_buffer_grows_with_streaming(self):
        """Consumer buffer accumulates as stream events arrive."""
        session = StreamSession()
        consumer = session.add_consumer()

        for i in range(10):
            session.send_chunk(f"c{i}")
            consumer.receive(timeout=0.01)

        assert len(consumer.buffer) == 10


# ============================================================================
# Scenario 4: Callback + WindowManager auto-write
# ============================================================================

class TestCallbackAutoWrite:
    """Callbacks registered on StreamSession automatically write to WindowManager."""

    def test_callback_auto_writes_to_window(self):
        """Registering a callback causes stream events to auto-write into window."""
        wm = WindowManager(WindowConfig(max_tokens=1000))
        session = StreamSession()

        def auto_write(event):
            if event.kind == StreamEventKind.CHUNK:
                wm.add_assistant(event.data["text"])

        session.on_event(auto_write)
        session.send_chunk("Auto-written chunk")
        session.close()

        assert any("Auto-written" in m.content for m in wm.get_messages())

    def test_callback_writes_all_chunk_events(self):
        """Callback writes every CHUNK event to window."""
        wm = WindowManager(WindowConfig(max_tokens=1000))
        session = StreamSession()

        received_chunks = []
        def collector(event):
            if event.kind == StreamEventKind.CHUNK:
                received_chunks.append(event.data["text"])
                wm.add_assistant(event.data["text"])

        session.on_event(collector)
        for i in range(5):
            session.send_chunk(f"part_{i}")
        session.close()

        assert len(received_chunks) == 5
        assert wm.message_count == 5

    def test_callback_with_multiple_handlers(self):
        """Multiple callbacks each perform independent window writes."""
        wm_user = WindowManager(WindowConfig(max_tokens=1000))
        wm_assistant = WindowManager(WindowConfig(max_tokens=1000))
        session = StreamSession()

        def write_user(event):
            if event.kind == StreamEventKind.CHUNK:
                wm_user.add_user(event.data["text"])

        def write_assistant(event):
            if event.kind == StreamEventKind.CHUNK:
                wm_assistant.add_assistant(event.data["text"])

        session.on_event(write_user)
        session.on_event(write_assistant)
        session.send_chunk("shared")
        session.close()

        assert wm_user.message_count == 1
        assert wm_assistant.message_count == 1

    def test_callback_receives_tool_events(self):
        """Callback receives TOOL_CALL and TOOL_RESULT events."""
        wm = WindowManager(WindowConfig(max_tokens=1000))
        session = StreamSession()
        tool_events = []

        def track_tools(event):
            if event.kind in (StreamEventKind.TOOL_CALL, StreamEventKind.TOOL_RESULT):
                tool_events.append(event)
                tool_name = event.data.get("tool", "?")
                wm.add_tool(f"{event.kind.value}: {tool_name}")

        session.on_event(track_tools)
        session.send_tool_call("search", {"q": "test"})
        session.send_tool_result("search", {"hits": 3}, success=True)
        session.close()

        assert len(tool_events) == 2
        assert wm.message_count == 2

    def test_callback_exception_does_not_break_stream(self):
        """Callback exceptions do not prevent other events from flowing."""
        wm = WindowManager(WindowConfig(max_tokens=1000))
        session = StreamSession()
        call_count = [0]

        def flaky(event):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("deliberate crash")
            wm.add_assistant(event.data.get("text", ""))

        session.on_event(flaky)
        session.send_chunk("first")
        session.send_chunk("second")
        session.close()

        assert call_count[0] == 3
        assert wm.message_count >= 1

    def test_callback_with_thinking_events(self):
        """Callback can capture THINKING events into window."""
        wm = WindowManager(WindowConfig(max_tokens=1000))
        session = StreamSession()

        def capture_thinking(event):
            if event.kind == StreamEventKind.THINKING:
                content = event.data["content"]
                wm.add_assistant(f"[thinking] {content}")

        session.on_event(capture_thinking)
        session.send_thinking("Let me analyze this...")
        session.close()

        msgs = wm.get_messages(MessageRole.ASSISTANT)
        assert any("[thinking]" in m.content for m in msgs)


# ============================================================================
# Scenario 5: MCPClient data classes + WindowManager interaction
# ============================================================================

class TestMCPClientDataWithWindowManager:
    """MCPToolDefinition and MCPToolResult integrate with WindowManager."""

    def test_tool_definition_stored_in_window(self):
        """MCPToolDefinition serialized into window for context."""
        from dr_mma.engine.mcp_client import MCPToolDefinition

        wm = WindowManager(WindowConfig(max_tokens=1000))
        tool = MCPToolDefinition(
            name="search",
            description="Search the web",
            inputSchema={"type": "object"},
        )
        wm.add_tool(json.dumps(tool.to_dict()))
        assert wm.message_count == 1

    def test_tool_result_stored_in_window(self):
        """MCPToolResult written to window as tool message."""
        from dr_mma.engine.mcp_client import MCPToolResult

        wm = WindowManager(WindowConfig(max_tokens=1000))
        result = MCPToolResult(
            success=True,
            content=[{"type": "text", "text": "Found 3 results"}],
        )
        wm.add_tool(json.dumps(result.to_dict()))
        msgs = wm.get_messages(MessageRole.TOOL)
        assert len(msgs) == 1

    def test_streaming_mcp_tool_call_lifecycle(self):
        """Full lifecycle: stream tool_call, then tool_result into window."""
        wm = WindowManager(WindowConfig(max_tokens=1000))
        session = StreamSession()

        def write_to_window(event):
            if event.kind == StreamEventKind.TOOL_CALL:
                tool_name = event.data["tool"]
                args = event.data.get("args", {})
                wm.add_tool(f"Calling {tool_name} with {args}")
            elif event.kind == StreamEventKind.TOOL_RESULT:
                result_data = event.data.get("result", {})
                wm.add_tool(f"Result: {result_data}")

        session.on_event(write_to_window)
        session.send_tool_call("search", {"q": "test"})
        session.send_tool_result("search", {"hits": 5}, success=True)
        session.close()

        tool_msgs = wm.get_messages(MessageRole.TOOL)
        assert len(tool_msgs) == 2

    def test_multiple_tool_results_in_window(self):
        """Multiple MCP tool results accumulate in window."""
        from dr_mma.engine.mcp_client import MCPToolResult

        wm = WindowManager(WindowConfig(max_tokens=1000))
        for i in range(3):
            result = MCPToolResult(
                success=True,
                content=[{"type": "text", "text": f"result_{i}"}],
            )
            wm.add_tool(json.dumps(result.to_dict()))

        assert wm.message_count == 3
        assert wm.total_tokens > 0

    def test_mcp_error_result_in_window(self):
        """MCP error result tracked in window with isError flag."""
        from dr_mma.engine.mcp_client import MCPToolResult

        wm = WindowManager(WindowConfig(max_tokens=1000))
        result = MCPToolResult(success=False, isError=True)
        wm.add_tool(json.dumps(result.to_dict()))

        msgs = wm.get_messages(MessageRole.TOOL)
        data = json.loads(msgs[0].content)
        assert data["isError"] is True


# ============================================================================
# Scenario 6: Error handling and edge cases across modules
# ============================================================================

class TestCrossModuleErrors:
    """Error scenarios spanning streaming, window_manager, and consumers."""

    def test_stream_error_close_propagates_to_consumer(self):
        """Error close sends ERROR event that consumer can detect."""
        session = StreamSession()
        consumer = session.add_consumer()
        session.send_chunk("ok")
        session.close(error="fatal error")

        events = consumer.receive_all(timeout=0.1)
        assert any(e.kind == StreamEventKind.ERROR for e in events)
        assert consumer.is_done is True

    def test_window_overflows_with_streaming_flood(self):
        """Window auto-trims under heavy streaming flood."""
        wm = WindowManager(WindowConfig(max_tokens=20, reserve_tokens=0, min_keep_count=1))
        session = StreamSession()

        for i in range(50):
            chunk = "X" * 100
            session.send_chunk(chunk)
            wm.add_assistant(chunk)

        stats = wm.stats
        assert stats["total_dropped"] > 0
        # Heavy flood causes significant trimming; message count drops far below 50
        assert wm.message_count < 50

    def test_consumer_disconnect_during_stream(self):
        """Consumer queue full causes eviction; stream continues for others."""
        session = StreamSession()
        small_q = session.add_consumer(buffer_size=2)
        alive_q = session.add_consumer(buffer_size=100)

        for i in range(5):
            session.send_chunk(f"msg{i}")

        small_events = small_q.receive_all(timeout=0.1)
        alive_events = alive_q.receive_all(timeout=0.1)
        assert len(alive_events) >= len(small_events)

    def test_window_clear_resets_after_stream(self):
        """WindowManager clear() resets state after streaming session."""
        wm = WindowManager(WindowConfig(max_tokens=100))
        session = StreamSession()

        for i in range(5):
            session.send_chunk(f"msg{i}")
            wm.add_user(f"msg{i}")

        session.close()
        cleared = wm.clear()
        assert cleared == 5
        assert wm.message_count == 0
        assert wm.total_tokens == 0

    def test_session_history_matches_consumer_buffer(self):
        """Producer history length matches total consumer buffer entries."""
        session = StreamSession()
        consumer = session.add_consumer()

        for i in range(3):
            session.send_chunk(f"h{i}")
        session.close()

        events = consumer.receive_all(timeout=0.1)
        history = session.producer.history
        assert len(events) == len(history)

    def test_window_prompt_export_after_streaming(self):
        """Window to_prompt() produces valid output after streaming."""
        wm = WindowManager(WindowConfig(max_tokens=1000))
        session = StreamSession()

        def write(event):
            if event.kind == StreamEventKind.CHUNK:
                wm.add_assistant(event.data["text"])

        session.on_event(write)
        session.send_chunk("Answer part 1")
        session.send_chunk("Answer part 2")
        session.close()

        prompt = wm.to_prompt()
        assert "[ASSISTANT]" in prompt
        assert "Answer part 1" in prompt
        assert "Answer part 2" in prompt

    def test_window_dicts_export_after_streaming(self):
        """Window to_dicts() produces valid structured output."""
        wm = WindowManager(WindowConfig(max_tokens=1000))
        session = StreamSession()

        session.send_chunk("data")
        wm.add_user("data")
        session.close()

        dicts = wm.to_dicts()
        assert len(dicts) == 1
        assert dicts[0]["role"] == "user"
        assert dicts[0]["content"] == "data"

    def test_consumer_is_done_false_before_close(self):
        """Consumer is_done remains False until DONE or ERROR event arrives."""
        session = StreamSession()
        consumer = session.add_consumer()

        session.send_chunk("not done")
        consumer.receive(timeout=0.1)
        assert consumer.is_done is False

        session.close()
        consumer.receive(timeout=0.1)
        assert consumer.is_done is True

    def test_window_usage_ratio_during_streaming(self):
        """Usage ratio increases as chunks are written to window."""
        wm = WindowManager(WindowConfig(max_tokens=100, reserve_tokens=0))
        session = StreamSession()

        initial_ratio = wm.usage_ratio
        assert initial_ratio == 0.0

        session.send_chunk("A" * 40)
        wm.add_assistant("A" * 40)
        assert wm.usage_ratio > initial_ratio

    def test_multiple_sessions_independent_windows(self):
        """Two sessions each maintain independent WindowManager state."""
        wm1 = WindowManager(WindowConfig(max_tokens=500))
        wm2 = WindowManager(WindowConfig(max_tokens=500))
        s1 = StreamSession("s1")
        s2 = StreamSession("s2")

        s1.send_chunk("from s1")
        wm1.add_user("from s1")
        s2.send_chunk("from s2")
        wm2.add_user("from s2")

        assert wm1.message_count == 1
        assert wm2.message_count == 1
        assert "from s1" not in wm2.to_prompt()


# ============================================================================
# Scenario 7: Concurrent streaming + window operations
# ============================================================================

class TestConcurrentStreamingWindow:
    """Threaded streaming and window writes work correctly."""

    def test_concurrent_chunk_send_and_window_add(self):
        """Chunks sent from one thread, written to window from another."""
        wm = WindowManager(WindowConfig(max_tokens=5000))
        session = StreamSession()
        consumer = session.add_consumer()

        def sender():
            for i in range(20):
                session.send_chunk(f"concurrent_{i}")
                time.sleep(0.001)
            session.close()

        def writer():
            while not consumer.is_done:
                events = consumer.receive_all(timeout=0.05)
                for e in events:
                    if e.kind == StreamEventKind.CHUNK:
                        wm.add_assistant(e.data["text"])

        t1 = threading.Thread(target=sender)
        t2 = threading.Thread(target=writer)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert wm.message_count > 0

    def test_concurrent_multi_consumer_window_write(self):
        """Multiple consumers writing to same window concurrently."""
        wm = WindowManager(WindowConfig(max_tokens=5000))
        session = StreamSession()

        consumers = [session.add_consumer() for _ in range(3)]

        def consume_and_write(consumer):
            events = consumer.receive_all(timeout=0.5)
            for e in events:
                if e.kind == StreamEventKind.CHUNK:
                    wm.add_user(e.data["text"])

        for i in range(10):
            session.send_chunk(f"item_{i}")
        session.close()

        threads = [threading.Thread(target=consume_and_write, args=(c,)) for c in consumers]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert wm.message_count >= 10


# ============================================================================
# Scenario 8: Window snapshot + stream history reconciliation
# ============================================================================

class TestSnapshotStreamReconciliation:
    """Window snapshots match stream history at key points."""

    def test_snapshot_after_full_stream(self):
        """Window snapshot reflects all streamed chunks."""
        wm = WindowManager(WindowConfig(max_tokens=1000))
        session = StreamSession()

        for i in range(5):
            session.send_chunk(f"chunk_{i}")
            wm.add_user(f"chunk_{i}")

        session.close()
        snap = wm.snapshot()

        assert snap.total_tokens == wm.total_tokens
        assert len(snap.messages) == 5
        assert snap.strategy == "none"

    def test_snapshot_after_trim_during_stream(self):
        """Snapshot shows dropped count when trim occurred during streaming."""
        wm = WindowManager(WindowConfig(max_tokens=15, reserve_tokens=0))
        session = StreamSession()

        for i in range(20):
            chunk = "B" * 40
            session.send_chunk(chunk)
            wm.add_assistant(chunk)

        snap = wm.snapshot()
        stats = wm.stats
        assert stats["total_dropped"] > 0
        assert snap.total_tokens == wm.total_tokens

    def test_history_seq_ordering(self):
        """Stream history maintains sequential ordering."""
        session = StreamSession()
        consumer = session.add_consumer()

        for i in range(5):
            session.send_chunk(f"seq_{i}")
        session.close()

        events = consumer.receive_all(timeout=0.1)
        chunk_events = [e for e in events if e.kind == StreamEventKind.CHUNK]
        seqs = [e.seq for e in chunk_events]
        assert seqs == sorted(seqs)
