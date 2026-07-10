"""PromptQueue admission flow tests."""

import threading
import time
import pytest
from dr_mma.engine.prompt_queue import (
    AdmittedPrompt,
    PromptQueue,
    STATUS_PENDING,
    STATUS_PROMOTED,
    STATUS_EXECUTING,
    STATUS_COMPLETED,
    STATUS_CANCELLED,
)


def _result_with_continuation(has_cont: bool) -> dict:
    """Helper to create a mock executor result."""
    return {"has_continuation": has_cont, "executed": True}


# -- AdmittedPrompt model tests --

class TestAdmittedPrompt:
    def test_auto_generate_id(self):
        p = AdmittedPrompt()
        assert p.id.startswith("PMT-")

    def test_default_status_is_pending(self):
        p = AdmittedPrompt()
        assert p.status == STATUS_PENDING

    def test_default_priority_zero(self):
        p = AdmittedPrompt()
        assert p.priority == 0

    def test_custom_fields(self):
        p = AdmittedPrompt(
            id="PMT-custom",
            session_id="S1",
            content="hello",
            priority=5,
            status=STATUS_PROMOTED,
        )
        assert p.id == "PMT-custom"
        assert p.session_id == "S1"
        assert p.content == "hello"
        assert p.priority == 5
        assert p.status == STATUS_PROMOTED

    def test_to_dict_roundtrip(self):
        p = AdmittedPrompt(
            id="PMT-1",
            session_id="S1",
            content="test",
            priority=3,
            status=STATUS_PENDING,
            promoted_at=None,
        )
        d = p.to_dict()
        p2 = AdmittedPrompt.from_dict(d)
        assert p2.id == "PMT-1"
        assert p2.session_id == "S1"
        assert p2.content == "test"
        assert p2.priority == 3
        assert p2.status == STATUS_PENDING

    def test_created_at_is_set(self):
        p = AdmittedPrompt()
        assert p.created_at

    def test_promoted_at_default_none(self):
        p = AdmittedPrompt()
        assert p.promoted_at is None


# -- admit tests --

class TestAdmit:
    def test_admit_returns_pending_prompt(self):
        q = PromptQueue()
        p = q.admit("hello", session_id="S1")
        assert p.status == STATUS_PENDING

    def test_admit_stores_prompt(self):
        q = PromptQueue()
        q.admit("hello", session_id="S1")
        assert q.inbox_count() == 1

    def test_admit_with_priority(self):
        q = PromptQueue()
        p = q.admit("urgent", session_id="S1", priority=10)
        assert p.priority == 10

    def test_admit_multiple(self):
        q = PromptQueue()
        q.admit("a", session_id="S1")
        q.admit("b", session_id="S1")
        assert q.inbox_count() == 2

    def test_admit_empty_content(self):
        q = PromptQueue()
        p = q.admit("", session_id="S1")
        assert p.content == ""

    def test_admit_no_session_id(self):
        q = PromptQueue()
        q.admit("hello")
        assert q.inbox_count() == 1


# -- promote tests --

class TestPromote:
    def test_promote_single_prompt(self):
        q = PromptQueue()
        q.admit("hello", session_id="S1")
        p = q.promote()
        assert p is not None
        assert p.status == STATUS_PROMOTED
        assert p.promoted_at is not None

    def test_promote_empty_queue_returns_none(self):
        q = PromptQueue()
        assert q.promote() is None

    def test_promote_higher_priority_first(self):
        q = PromptQueue()
        q.admit("low", session_id="S1", priority=1)
        q.admit("high", session_id="S1", priority=10)
        p = q.promote()
        assert p.content == "high"

    def test_promote_removes_from_inbox(self):
        q = PromptQueue()
        q.admit("hello", session_id="S1")
        q.promote()
        assert q.inbox_count() == 0

    def test_promote_skips_cancelled(self):
        q = PromptQueue()
        p1 = q.admit("cancel-me", session_id="S1", priority=10)
        q.admit("keep", session_id="S1", priority=5)
        q.cancel(p1.id)
        p = q.promote()
        assert p.content == "keep"

    def test_promote_fifo_same_priority(self):
        q = PromptQueue()
        p1 = q.admit("first", session_id="S1", priority=5)
        p2 = q.admit("second", session_id="S1", priority=5)
        promoted = q.promote()
        assert promoted.id == p1.id

    def test_promote_status_transition(self):
        q = PromptQueue()
        q.admit("hello", session_id="S1")
        p = q.promote()
        assert p.status == STATUS_PROMOTED
        # promote again should return None (already consumed)
        assert q.promote() is None


# -- peek_next tests --

class TestPeekNext:
    def test_peek_returns_highest_priority(self):
        q = PromptQueue()
        q.admit("low", session_id="S1", priority=1)
        q.admit("high", session_id="S1", priority=10)
        p = q.peek_next()
        assert p.content == "high"

    def test_peek_does_not_consume(self):
        q = PromptQueue()
        q.admit("hello", session_id="S1")
        q.peek_next()
        assert q.inbox_count() == 1

    def test_peek_empty_queue_returns_none(self):
        q = PromptQueue()
        assert q.peek_next() is None

    def test_peek_with_session_filter(self):
        q = PromptQueue()
        q.admit("s1", session_id="S1")
        q.admit("s2", session_id="S2")
        p = q.peek_next(session_id="S1")
        assert p is not None
        assert p.session_id == "S1"

    def test_peek_session_not_found(self):
        q = PromptQueue()
        q.admit("s1", session_id="S1")
        assert q.peek_next(session_id="S2") is None


# -- cancel tests --

class TestCancel:
    def test_cancel_pending_prompt(self):
        q = PromptQueue()
        p = q.admit("cancel-me", session_id="S1")
        assert q.cancel(p.id) is True
        assert q.inbox_count() == 0

    def test_cancel_nonexistent_prompt_returns_false(self):
        q = PromptQueue()
        assert q.cancel("PMT-nope") is False

    def test_cancel_completed_prompt_fails(self):
        q = PromptQueue()
        p = q.admit("hello", session_id="S1")
        q.promote()
        # After promote, prompt is removed from inbox, so cancel should fail
        assert q.cancel(p.id) is False

    def test_cancel_does_not_affect_other_prompts(self):
        q = PromptQueue()
        p1 = q.admit("cancel-me", session_id="S1")
        q.admit("keep", session_id="S1")
        q.cancel(p1.id)
        assert q.inbox_count() == 1

    def test_cancelled_prompt_not_promotable(self):
        q = PromptQueue()
        p = q.admit("cancel-me", session_id="S1")
        q.cancel(p.id)
        assert q.promote() is None


# -- drain tests --

class TestDrain:
    def test_drain_single_turn(self):
        q = PromptQueue()
        q.admit("hello", session_id="S1")

        results = q.drain(
            max_turns=1,
            executor=lambda p: _result_with_continuation(False),
        )
        assert len(results) == 1
        assert results[0]["executed"] is True

    def test_drain_with_continuation(self):
        q = PromptQueue()
        q.admit("first", session_id="S1")
        q.admit("second", session_id="S1")
        q.admit("third", session_id="S1")

        results = q.drain(
            max_turns=3,
            executor=lambda p: _result_with_continuation(True),
        )
        assert len(results) == 3

    def test_drain_stops_on_no_continuation(self):
        q = PromptQueue()
        q.admit("first", session_id="S1")
        q.admit("second", session_id="S1")
        q.admit("third", session_id="S1")

        call_count = 0

        def executor(p):
            nonlocal call_count
            call_count += 1
            return _result_with_continuation(call_count < 2)

        results = q.drain(max_turns=3, executor=executor)
        assert len(results) == 2
        assert call_count == 2

    def test_drain_respects_max_turns(self):
        q = PromptQueue()
        for i in range(5):
            q.admit(f"prompt-{i}", session_id="S1")

        results = q.drain(
            max_turns=2,
            executor=lambda p: _result_with_continuation(True),
        )
        assert len(results) == 2

    def test_drain_empty_queue(self):
        q = PromptQueue()
        results = q.drain(max_turns=5)
        assert results == []

    def test_drain_default_executor(self):
        q = PromptQueue()
        q.admit("hello", session_id="S1")
        results = q.drain(max_turns=1)
        assert len(results) == 1
        assert "prompt_id" in results[0]

    def test_drain_custom_continuation_check(self):
        q = PromptQueue()
        q.admit("a", session_id="S1")
        q.admit("b", session_id="S1")

        results = q.drain(
            max_turns=3,
            executor=lambda p: {"content": p.content, "score": 0.5},
            continuation_check=lambda r: r.get("score", 0) > 0.4,
        )
        assert len(results) == 2

    def test_drain_prompt_status_transitions(self):
        q = PromptQueue()
        p = q.admit("hello", session_id="S1")

        # Before drain, prompt is in inbox with pending status
        assert q.inbox_count() == 1

        results = q.drain(
            max_turns=1,
            executor=lambda p: _result_with_continuation(False),
        )

        # After drain, inbox should be empty (prompt consumed and completed)
        assert q.inbox_count() == 0
        assert len(results) == 1


# -- priority scheduling tests --

class TestPriorityScheduling:
    def test_high_priority_promoted_before_low(self):
        q = PromptQueue()
        q.admit("low", session_id="S1", priority=1)
        q.admit("mid", session_id="S1", priority=5)
        q.admit("high", session_id="S1", priority=10)

        p1 = q.promote()
        p2 = q.promote()
        p3 = q.promote()

        assert p1.content == "high"
        assert p2.content == "mid"
        assert p3.content == "low"

    def test_negative_priority_works(self):
        q = PromptQueue()
        q.admit("neg", session_id="S1", priority=-5)
        q.admit("pos", session_id="S1", priority=1)
        p = q.promote()
        assert p.content == "pos"

    def test_drain_respects_priority(self):
        q = PromptQueue()
        q.admit("low", session_id="S1", priority=1)
        q.admit("high", session_id="S1", priority=10)

        results = q.drain(
            max_turns=2,
            executor=lambda p: {"content": p.content, "has_continuation": False},
        )
        # First result should be high priority
        assert results[0]["content"] == "high"


# -- session isolation tests --

class TestSessionIsolation:
    def test_inbox_count_per_session(self):
        q = PromptQueue()
        q.admit("a", session_id="S1")
        q.admit("b", session_id="S1")
        q.admit("c", session_id="S2")

        assert q.inbox_count(session_id="S1") == 2
        assert q.inbox_count(session_id="S2") == 1
        assert q.inbox_count() == 3

    def test_peek_next_per_session(self):
        q = PromptQueue()
        q.admit("s1", session_id="S1")
        q.admit("s2", session_id="S2")

        p = q.peek_next(session_id="S2")
        assert p is not None
        assert p.session_id == "S2"

    def test_cancel_per_session(self):
        q = PromptQueue()
        p1 = q.admit("s1", session_id="S1")
        q.admit("s2", session_id="S2")

        q.cancel(p1.id)
        assert q.inbox_count(session_id="S1") == 0
        assert q.inbox_count(session_id="S2") == 1

    def test_queue_summary_session_breakdown(self):
        q = PromptQueue()
        q.admit("a", session_id="S1")
        q.admit("b", session_id="S1")
        q.admit("c", session_id="S2")

        summary = q.queue_summary()
        assert summary["session_breakdown"]["S1"] == 2
        assert summary["session_breakdown"]["S2"] == 1

    def test_list_pending_per_session(self):
        q = PromptQueue()
        q.admit("a", session_id="S1")
        q.admit("b", session_id="S2")

        s1_pending = q.list_pending(session_id="S1")
        assert len(s1_pending) == 1
        assert s1_pending[0].session_id == "S1"


# -- diagnostics tests --

class TestDiagnostics:
    def test_queue_summary_empty(self):
        q = PromptQueue()
        summary = q.queue_summary()
        assert summary["total_inbox"] == 0
        assert summary["status_breakdown"] == {}
        assert summary["session_breakdown"] == {}

    def test_queue_summary_after_operations(self):
        q = PromptQueue()
        p1 = q.admit("a", session_id="S1")
        p2 = q.admit("b", session_id="S1")
        # Cancel both before promoting
        q.cancel(p1.id)
        q.cancel(p2.id)

        summary = q.queue_summary()
        assert summary["total_inbox"] == 2
        assert STATUS_CANCELLED in summary["status_breakdown"]

    def test_get_by_id(self):
        q = PromptQueue()
        p = q.admit("hello", session_id="S1")
        found = q.get_by_id(p.id)
        assert found is not None
        assert found.content == "hello"

    def test_get_by_id_not_found(self):
        q = PromptQueue()
        assert q.get_by_id("PMT-nope") is None

    def test_list_pending_sorted_by_priority(self):
        q = PromptQueue()
        q.admit("low", session_id="S1", priority=1)
        q.admit("high", session_id="S1", priority=10)
        pending = q.list_pending()
        assert pending[0].priority > pending[1].priority

    def test_clear(self):
        q = PromptQueue()
        q.admit("a", session_id="S1")
        q.admit("b", session_id="S2")
        q.clear()
        assert q.inbox_count() == 0


# -- concurrency tests --

class TestConcurrency:
    def test_concurrent_admit(self):
        """Multiple threads admitting prompts should not lose data."""
        q = PromptQueue()
        num_threads = 10
        prompts_per_thread = 100

        def admit_batch():
            for i in range(prompts_per_thread):
                q.admit(f"p-{i}", session_id="S1")

        threads = [threading.Thread(target=admit_batch) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert q.inbox_count() == num_threads * prompts_per_thread

    def test_concurrent_promote(self):
        """Multiple threads promoting should not double-consume."""
        q = PromptQueue()
        for i in range(10):
            q.admit(f"p-{i}", session_id="S1")

        promoted = []
        lock = threading.Lock()

        def promote_one():
            p = q.promote()
            if p is not None:
                with lock:
                    promoted.append(p.id)

        threads = [threading.Thread(target=promote_one) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each prompt should be promoted at most once
        assert len(promoted) == 10
        assert len(set(promoted)) == 10

    def test_concurrent_admit_and_promote(self):
        """Mixing admit and promote from different threads."""
        q = PromptQueue()
        admitted_count = 0
        promoted_count = 0
        lock = threading.Lock()

        def admit_loop():
            nonlocal admitted_count
            for i in range(50):
                q.admit(f"p-{i}", session_id="S1")
                with lock:
                    admitted_count += 1
                time.sleep(0.001)

        def promote_loop():
            nonlocal promoted_count
            for _ in range(50):
                p = q.promote()
                if p is not None:
                    with lock:
                        promoted_count += 1
                time.sleep(0.002)

        t1 = threading.Thread(target=admit_loop)
        t2 = threading.Thread(target=promote_loop)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Total admitted should be 50, promoted <= admitted
        assert admitted_count == 50
        assert promoted_count <= admitted_count
