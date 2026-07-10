"""ContextEpoch unit tests."""

import time

import pytest
from dr_mma.engine.context_epoch import (
    BaselineContext,
    ContextSnapshot,
    MidConversationUpdate,
    TerminationCondition,
    ContextEpoch,
)


class TestBaselineContext:
    def test_default_render(self):
        b = BaselineContext()
        assert b.render() == ""

    def test_render_system_prompt(self):
        b = BaselineContext(system_prompt="You are helpful")
        rendered = b.render()
        assert "System: You are helpful" in rendered

    def test_render_roles(self):
        b = BaselineContext(role_definitions={"Agent": "Do tasks", "Supervisor": "Oversee"})
        rendered = b.render()
        assert "Roles:" in rendered
        assert "Agent: Do tasks" in rendered

    def test_render_all_fields(self):
        b = BaselineContext(
            system_prompt="SYS",
            role_definitions={"R1": "Desc"},
            task_objective="Goal",
            constraints=["C1", "C2"],
            available_tools=["T1"],
        )
        rendered = b.render()
        assert "SYS" in rendered
        assert "Goal" in rendered
        assert "C1" in rendered
        assert "T1" in rendered

    def test_to_dict(self):
        b = BaselineContext(system_prompt="S", task_objective="O")
        d = b.to_dict()
        assert d["system_prompt"] == "S"
        assert d["task_objective"] == "O"


class TestContextSnapshot:
    def test_default_snapshot(self):
        s = ContextSnapshot(epoch_id="E1")
        assert s.token_count == 0
        assert s.message_count == 0

    def test_to_dict(self):
        s = ContextSnapshot(epoch_id="E1", token_count=100, error_count=2)
        d = s.to_dict()
        assert d["token_count"] == 100
        assert d["error_count"] == 2

    def test_diff_basic(self):
        s1 = ContextSnapshot(epoch_id="E1", token_count=100, message_count=5)
        s2 = ContextSnapshot(epoch_id="E1", token_count=200, message_count=10)
        diff = s2.diff(s1)
        assert diff["token_delta"] == 100
        assert diff["message_delta"] == 5

    def test_diff_negative(self):
        s1 = ContextSnapshot(epoch_id="E1", token_count=200)
        s2 = ContextSnapshot(epoch_id="E1", token_count=100)
        diff = s2.diff(s1)
        assert diff["token_delta"] == -100


class TestMidConversationUpdate:
    def test_create_update(self):
        u = MidConversationUpdate(update_id="U1", epoch_id="E1")
        assert u.update_type == "context_change"

    def test_to_dict(self):
        u = MidConversationUpdate(
            update_id="U1",
            epoch_id="E1",
            update_type="tool_update",
            content={"key": "val"},
        )
        d = u.to_dict()
        assert d["update_type"] == "tool_update"
        assert d["content"]["key"] == "val"


class TestTerminationCondition:
    def test_no_termination(self):
        tc = TerminationCondition(max_tokens=10000, max_errors=5)
        snap = ContextSnapshot(epoch_id="E1", token_count=100, error_count=0)
        should, reason = tc.should_terminate(snap, time.time())
        assert should is False

    def test_token_limit(self):
        tc = TerminationCondition(max_tokens=100)
        snap = ContextSnapshot(epoch_id="E1", token_count=200)
        should, reason = tc.should_terminate(snap, time.time())
        assert should is True
        assert "Token limit" in reason

    def test_error_limit(self):
        tc = TerminationCondition(max_errors=3)
        snap = ContextSnapshot(epoch_id="E1", error_count=5)
        should, reason = tc.should_terminate(snap, time.time())
        assert should is True
        assert "Error" in reason

    def test_duration_limit(self):
        tc = TerminationCondition(max_duration_s=1.0)
        snap = ContextSnapshot(epoch_id="E1", token_count=10)
        should, reason = tc.should_terminate(snap, time.time() - 5)
        assert should is True
        assert "Duration" in reason

    def test_custom_check(self):
        def custom(snap):
            return snap.tool_call_count > 10

        tc = TerminationCondition(custom_check=custom)
        snap = ContextSnapshot(epoch_id="E1", tool_call_count=20)
        should, reason = tc.should_terminate(snap, time.time())
        assert should is True
        assert "Custom" in reason


class TestContextEpoch:
    def test_create_epoch(self):
        epoch = ContextEpoch(epoch_id="E1")
        assert epoch.epoch_id == "E1"
        assert epoch.is_terminated is False

    def test_with_baseline(self):
        b = BaselineContext(system_prompt="SYS")
        epoch = ContextEpoch(epoch_id="E1", baseline=b)
        assert "SYS" in epoch.rendered_baseline

    def test_update_snapshot(self):
        epoch = ContextEpoch(epoch_id="E1")
        snap = epoch.update_snapshot(token_count=500, message_count=10)
        assert snap.token_count == 500
        assert snap.message_count == 10

    def test_should_terminate_false(self):
        epoch = ContextEpoch(epoch_id="E1", max_tokens=10000)
        epoch.update_snapshot(token_count=100)
        should, reason = epoch.should_terminate()
        assert should is False

    def test_should_terminate_true(self):
        epoch = ContextEpoch(epoch_id="E1", max_tokens=100)
        epoch.update_snapshot(token_count=200)
        should, reason = epoch.should_terminate()
        assert should is True

    def test_manual_terminate(self):
        epoch = ContextEpoch(epoch_id="E1")
        epoch.terminate(reason="done")
        assert epoch.is_terminated is True
        assert epoch.termination_reason == "done"

    def test_terminated_at_set_on_terminate(self):
        epoch = ContextEpoch(epoch_id="E1")
        epoch.terminate()
        assert epoch.terminated_at is not None

    def test_duration_s(self):
        epoch = ContextEpoch(epoch_id="E1")
        duration = epoch.duration_s
        assert duration >= 0

    def test_apply_update(self):
        epoch = ContextEpoch(epoch_id="E1")
        update = MidConversationUpdate(update_id="U1", epoch_id="E1")
        epoch.apply_update(update)
        updates = epoch.get_all_updates()
        assert len(updates) == 1
        assert updates[0].update_id == "U1"

    def test_apply_update_after_terminate_raises(self):
        epoch = ContextEpoch(epoch_id="E1")
        epoch.terminate()
        update = MidConversationUpdate(update_id="U1", epoch_id="E1")
        with pytest.raises(RuntimeError):
            epoch.apply_update(update)

    def test_get_updates_since(self):
        epoch = ContextEpoch(epoch_id="E1")
        u1 = MidConversationUpdate(update_id="U1", epoch_id="E1")
        epoch.apply_update(u1)
        import time as t
        t.sleep(0.01)
        u2 = MidConversationUpdate(update_id="U2", epoch_id="E1")
        epoch.apply_update(u2)

        since = u1.timestamp
        recent = epoch.get_updates_since(since)
        assert len(recent) == 1
        assert recent[0].update_id == "U2"

    def test_snapshot_diff(self):
        e1 = ContextEpoch(epoch_id="E1")
        e2 = ContextEpoch(epoch_id="E2")
        e1.update_snapshot(token_count=100)
        e2.update_snapshot(token_count=200)
        diff = e2.snapshot_diff(e1)
        assert diff["token_delta"] == 100

    def test_to_dict(self):
        epoch = ContextEpoch(epoch_id="E1")
        d = epoch.to_dict()
        assert d["epoch_id"] == "E1"
        assert d["is_terminated"] is False
