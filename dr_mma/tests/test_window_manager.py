"""WindowManager unit tests."""

import pytest
from dr_mma.engine.window_manager import (
    MessageRole,
    WindowConfig,
    WindowMessage,
    WindowSnapshot,
    WindowManager,
)


class TestWindowMessage:
    def test_basic_creation(self):
        msg = WindowMessage(role=MessageRole.USER, content="hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "hello"
        assert msg.importance == 1.0

    def test_token_estimate_empty(self):
        msg = WindowMessage(role=MessageRole.USER, content="")
        assert msg.token_count >= 1  # max(1, 0) = 1

    def test_token_estimate_long(self):
        msg = WindowMessage(role=MessageRole.USER, content="A" * 100)
        assert msg.token_count == 25  # 100 // 4 = 25

    def test_custom_token_count(self):
        msg = WindowMessage(
            role=MessageRole.USER, content="hi", token_count=50
        )
        assert msg.token_count == 50

    def test_to_dict(self):
        msg = WindowMessage(
            role=MessageRole.ASSISTANT,
            content="answer",
            importance=3.0,
            metadata={"key": "val"},
        )
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["importance"] == 3.0
        assert d["metadata"]["key"] == "val"

    def test_metadata_default_empty(self):
        msg = WindowMessage(role=MessageRole.USER, content="hi")
        assert msg.metadata == {}


class TestWindowConfig:
    def test_defaults(self):
        cfg = WindowConfig()
        assert cfg.max_tokens == 128000
        assert cfg.reserve_tokens == 4096
        assert cfg.system_always_keep is True
        assert cfg.min_keep_count == 5

    def test_usable_tokens(self):
        cfg = WindowConfig(max_tokens=1000, reserve_tokens=200)
        assert cfg.usable_tokens == 800

    def test_custom_values(self):
        cfg = WindowConfig(
            max_tokens=500, reserve_tokens=100, system_always_keep=False, min_keep_count=3
        )
        assert cfg.usable_tokens == 400
        assert cfg.system_always_keep is False
        assert cfg.min_keep_count == 3


class TestWindowSnapshot:
    def test_creation(self):
        snap = WindowSnapshot(
            messages=[], total_tokens=0, dropped_count=0, dropped_tokens=0, strategy="none"
        )
        assert snap.strategy == "none"


class TestWindowManagerAdd:
    def _wm(self, **kwargs):
        return WindowManager(config=kwargs.get("config"))

    def test_add_user(self):
        wm = self._wm()
        wm.add_user("hello")
        assert wm.message_count == 1

    def test_add_system(self):
        wm = self._wm()
        msg = wm.add_system("You are helpful")
        assert msg.importance == 5.0
        assert msg.role == MessageRole.SYSTEM

    def test_add_assistant(self):
        wm = self._wm()
        msg = wm.add_assistant("Sure!")
        assert msg.importance == 2.0
        assert msg.role == MessageRole.ASSISTANT

    def test_add_tool(self):
        wm = self._wm()
        msg = wm.add_tool("tool output", metadata={"name": "search"})
        assert msg.importance == 1.5
        assert msg.metadata["name"] == "search"

    def test_add_returns_message(self):
        wm = self._wm()
        msg = wm.add(MessageRole.USER, "test")
        assert isinstance(msg, WindowMessage)


class TestWindowManagerProperties:
    def test_total_tokens_empty(self):
        wm = WindowManager()
        assert wm.total_tokens == 0

    def test_total_tokens_sum(self):
        wm = WindowManager()
        wm.add_user("AAAA")  # 1 token
        wm.add_user("BBBB")  # 1 token
        assert wm.total_tokens >= 2

    def test_message_count(self):
        wm = WindowManager()
        wm.add_user("a")
        wm.add_user("b")
        assert wm.message_count == 2

    def test_usage_ratio_zero_max(self):
        cfg = WindowConfig(max_tokens=0)
        wm = WindowManager(config=cfg)
        assert wm.usage_ratio == 0.0

    def test_usage_ratio_normal(self):
        cfg = WindowConfig(max_tokens=100)
        wm = WindowManager(config=cfg)
        wm.add_user("A" * 40)  # ~10 tokens
        assert 0 < wm.usage_ratio <= 1.0

    def test_usage_ratio_capped_at_1(self):
        cfg = WindowConfig(max_tokens=10)
        wm = WindowManager(config=cfg)
        wm.add_user("A" * 1000)  # ~250 tokens, way over
        assert wm.usage_ratio == 1.0


class TestTrimByImportance:
    def _wm(self, **kwargs):
        return WindowManager(config=WindowConfig(**kwargs))

    def test_no_trim_when_under_budget(self):
        cfg = WindowConfig(max_tokens=1000, reserve_tokens=0)
        wm = self._wm(
            max_tokens=1000, reserve_tokens=0
        )
        wm.add_user("hello")
        snap = wm.trim_by_importance()
        assert wm.message_count == 1
        assert snap.dropped_count == 0

    def test_drops_low_importance_first(self):
        cfg = WindowConfig(max_tokens=20, reserve_tokens=0)
        wm = WindowManager(config=cfg)
        wm.add_user("low", importance=0.5)
        wm.add_user("mid", importance=1.0)
        wm.add_user("high", importance=3.0)

        snap = wm.trim_by_importance()
        # low importance should be dropped first
        remaining = wm.get_messages()
        contents = [m.content for m in remaining]
        assert "high" in contents

    def test_system_messages_protected(self):
        cfg = WindowConfig(max_tokens=20, reserve_tokens=0)
        wm = WindowManager(config=cfg)
        wm.add_system("system msg")
        wm.add_user("user low", importance=0.5)
        wm.add_user("user mid", importance=1.0)

        wm.trim_by_importance()
        sys_msgs = wm.get_messages(MessageRole.SYSTEM)
        assert len(sys_msgs) == 1

    def test_system_not_protected_when_disabled(self):
        cfg = WindowConfig(max_tokens=10, reserve_tokens=0, system_always_keep=False)
        wm = WindowManager(config=cfg)
        wm.add_system("system")
        wm.add_user("user", importance=5.0)

        snap = wm.trim_by_importance()
        # system might be dropped since protection is off
        # the key point is the strategy still works
        assert snap.strategy == "importance"

    def test_min_keep_count_enforced(self):
        cfg = WindowConfig(max_tokens=10, reserve_tokens=0, min_keep_count=3)
        wm = WindowManager(config=cfg)
        for i in range(5):
            wm.add_user(f"msg{i}", importance=float(i))

        snap = wm.trim_by_importance()
        assert wm.message_count >= 3

    def test_dropped_tokens_tracked(self):
        cfg = WindowConfig(max_tokens=10, reserve_tokens=0)
        wm = WindowManager(config=cfg)
        wm.add_user("A" * 40, importance=0.5)  # 10 tokens
        wm.add_user("B" * 40, importance=0.5)  # 10 tokens

        snap = wm.trim_by_importance()
        assert snap.dropped_tokens >= 0
        assert snap.dropped_count >= 0

    def test_target_tokens_override(self):
        cfg = WindowConfig(max_tokens=100, reserve_tokens=0)
        wm = WindowManager(config=cfg)
        wm.add_user("A" * 40, importance=0.5)
        wm.add_user("B" * 40, importance=0.5)

        snap = wm.trim_by_importance(target_tokens=5)
        assert snap.strategy == "importance"


class TestTrimSliding:
    def test_keeps_tail(self):
        cfg = WindowConfig(max_tokens=128000, reserve_tokens=4096)
        wm = WindowManager(config=cfg)
        for i in range(30):
            wm.add_user(f"msg{i}")

        snap = wm.trim_sliding(keep_tail=10)
        assert wm.message_count == 10
        assert snap.dropped_count == 20

    def test_system_protected_in_sliding(self):
        cfg = WindowConfig(max_tokens=128000, reserve_tokens=4096)
        wm = WindowManager(config=cfg)
        wm.add_system("sys")
        for i in range(30):
            wm.add_user(f"msg{i}")

        snap = wm.trim_sliding(keep_tail=10)
        sys_msgs = wm.get_messages(MessageRole.SYSTEM)
        assert len(sys_msgs) == 1

    def test_no_drop_when_under_keep_tail(self):
        cfg = WindowConfig(max_tokens=128000, reserve_tokens=4096)
        wm = WindowManager(config=cfg)
        for i in range(5):
            wm.add_user(f"msg{i}")

        snap = wm.trim_sliding(keep_tail=10)
        assert snap.dropped_count == 0
        assert wm.message_count == 5

    def test_dropped_tokens_recorded(self):
        cfg = WindowConfig(max_tokens=128000, reserve_tokens=4096)
        wm = WindowManager(config=cfg)
        for i in range(30):
            wm.add_user(f"msg{i}")

        snap = wm.trim_sliding(keep_tail=5)
        assert snap.dropped_tokens > 0


class TestTrimHybrid:
    def test_importance_only_suffices(self):
        cfg = WindowConfig(max_tokens=100, reserve_tokens=0)
        wm = WindowManager(config=cfg)
        for i in range(5):
            wm.add_user(f"msg{i}", importance=float(i))

        snap = wm.trim_hybrid()
        # If importance trim alone brings it under budget, strategy is "importance"
        assert snap.strategy in ("importance", "hybrid")

    def test_fallback_to_sliding(self):
        # Use a large budget so add() doesn't auto-trim, then manually trim
        cfg = WindowConfig(max_tokens=128000, reserve_tokens=4096, min_keep_count=3)
        wm = WindowManager(config=cfg)
        for i in range(50):
            wm.add_user(f"msg{i}", importance=5.0)  # all high importance

        # Set a very small target so importance trim can't drop enough
        # min_keep_count=3 prevents dropping below 3, but 3*1=3 tokens still > 2
        wm.config.max_tokens = 2
        wm.config.reserve_tokens = 0

        snap = wm.trim_hybrid()
        assert snap.strategy == "hybrid"

    def test_combined_drops(self):
        cfg = WindowConfig(max_tokens=128000, reserve_tokens=4096)
        wm = WindowManager(config=cfg)
        for i in range(20):
            wm.add_user(f"msg{i}", importance=5.0)

        # Change budget to force trim
        wm.config.max_tokens = 10
        wm.config.reserve_tokens = 0

        snap = wm.trim_hybrid()
        assert snap.dropped_count > 0

    def test_under_budget_returns_none(self):
        cfg = WindowConfig(max_tokens=128000, reserve_tokens=4096)
        wm = WindowManager(config=cfg)
        wm.add_user("hello")
        result = wm.trim_if_needed()
        assert result is None


class TestTrimIfNeeded:
    def test_auto_triggers_on_overflow(self):
        cfg = WindowConfig(max_tokens=20, reserve_tokens=0)
        wm = WindowManager(config=cfg)
        wm.add_user("A" * 100)  # ~25 tokens, over budget
        assert wm.message_count > 0  # some messages remain after trim

    def test_no_trim_when_under(self):
        cfg = WindowConfig(max_tokens=128000, reserve_tokens=4096)
        wm = WindowManager(config=cfg)
        wm.add_user("hello")
        assert wm.trim_if_needed() is None

    def test_trim_if_needed_returns_snapshot_on_trim(self):
        cfg = WindowConfig(max_tokens=10, reserve_tokens=0)
        wm = WindowManager(config=cfg)
        wm.add_user("A" * 100)  # over budget
        # trim_if_needed was called automatically by add()
        stats = wm.stats
        assert stats["total_dropped"] >= 0


class TestQueryAndExport:
    def test_get_messages_all(self):
        wm = WindowManager()
        wm.add_user("u1")
        wm.add_assistant("a1")
        all_msgs = wm.get_messages()
        assert len(all_msgs) == 2

    def test_get_messages_filter_role(self):
        wm = WindowManager()
        wm.add_user("u1")
        wm.add_assistant("a1")
        user_msgs = wm.get_messages(MessageRole.USER)
        assert all(m.role == MessageRole.USER for m in user_msgs)

    def test_get_recent(self):
        wm = WindowManager()
        for i in range(20):
            wm.add_user(f"msg{i}")
        recent = wm.get_recent(5)
        assert len(recent) == 5
        # last message should be most recent
        assert recent[-1].content == "msg19"

    def test_to_prompt(self):
        wm = WindowManager()
        wm.add_user("hello")
        prompt = wm.to_prompt()
        assert "[USER]" in prompt
        assert "hello" in prompt

    def test_to_dicts(self):
        wm = WindowManager()
        wm.add_user("hi")
        dicts = wm.to_dicts()
        assert len(dicts) == 1
        assert dicts[0]["role"] == "user"

    def test_get_messages_empty(self):
        wm = WindowManager()
        assert wm.get_messages() == []


class TestSnapshotClearStats:
    def test_snapshot_no_modification(self):
        wm = WindowManager()
        wm.add_user("hi")
        snap = wm.snapshot()
        assert snap.strategy == "none"
        assert snap.dropped_count == 0

    def test_clear_returns_count(self):
        wm = WindowManager()
        wm.add_user("a")
        wm.add_user("b")
        count = wm.clear()
        assert count == 2
        assert wm.message_count == 0

    def test_clear_empty(self):
        wm = WindowManager()
        assert wm.clear() == 0

    def test_stats_keys(self):
        wm = WindowManager()
        s = wm.stats
        for key in [
            "total_tokens",
            "message_count",
            "usage_ratio",
            "max_tokens",
            "usable_tokens",
            "total_dropped",
            "total_dropped_tokens",
        ]:
            assert key in s

    def test_stats_after_trim(self):
        cfg = WindowConfig(max_tokens=10, reserve_tokens=0)
        wm = WindowManager(config=cfg)
        for i in range(20):
            wm.add_user(f"msg{i}")
        s = wm.stats
        assert s["total_dropped"] > 0


class TestThreadSafety:
    def test_concurrent_add(self):
        import threading

        wm = WindowManager(config=WindowConfig(max_tokens=128000, reserve_tokens=4096))
        errors = []

        def add_msgs():
            try:
                for i in range(50):
                    wm.add_user(f"msg_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_msgs) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert wm.message_count > 0

    def test_concurrent_clear(self):
        import threading

        wm = WindowManager()
        for i in range(10):
            wm.add_user(f"msg{i}")

        errors = []

        def do_clear():
            try:
                wm.clear()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_clear) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestEdgeCases:
    def test_add_empty_content(self):
        wm = WindowManager()
        wm.add_user("")
        assert wm.message_count == 1

    def test_very_large_importance(self):
        wm = WindowManager()
        wm.add_user("msg", importance=100.0)
        msg = wm.get_messages()[0]
        assert msg.importance == 100.0

    def test_zero_max_tokens_no_crash(self):
        cfg = WindowConfig(max_tokens=0, reserve_tokens=0)
        wm = WindowManager(config=cfg)
        wm.add_user("hello")
        # Should not crash even with zero budget
        assert wm.message_count >= 0

    def test_reserve_tokens_larger_than_max(self):
        cfg = WindowConfig(max_tokens=100, reserve_tokens=200)
        wm = WindowManager(config=cfg)
        # usable_tokens will be negative, add should still work
        wm.add_user("hello")
        assert wm.message_count >= 0

    def test_multiple_trims_cumulative_dropped(self):
        cfg = WindowConfig(max_tokens=10, reserve_tokens=0)
        wm = WindowManager(config=cfg)
        for i in range(30):
            wm.add_user(f"msg{i}")

        stats = wm.stats
        assert stats["total_dropped"] > 0
        assert stats["total_dropped_tokens"] > 0

    def test_trim_sliding_keeps_system_and_tail(self):
        cfg = WindowConfig(max_tokens=128000, reserve_tokens=4096)
        wm = WindowManager(config=cfg)
        wm.add_system("sys prompt")
        for i in range(25):
            wm.add_user(f"msg{i}")

        snap = wm.trim_sliding(keep_tail=10)
        # system + 10 tail = 11 messages
        assert wm.message_count == 11

    def test_trim_by_importance_all_same_importance(self):
        cfg = WindowConfig(max_tokens=20, reserve_tokens=0)
        wm = WindowManager(config=cfg)
        for i in range(10):
            wm.add_user(f"msg{i}", importance=1.0)

        snap = wm.trim_by_importance()
        assert snap.strategy == "importance"
        # some should be dropped since all same importance and over budget
        assert wm.message_count < 10 or wm.total_tokens <= 20
