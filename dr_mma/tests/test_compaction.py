"""Compaction unit tests."""

import pytest
from dr_mma.engine.compaction import (
    SlidingWindowConfig,
    CompactionTrigger,
    TriggerMode,
    CompactionSummary,
    CompactionEngine,
)


class TestSlidingWindowConfig:
    def test_defaults(self):
        cfg = SlidingWindowConfig()
        assert cfg.max_tokens == 64_000
        assert cfg.keep_last_n_messages == 20

    def test_custom(self):
        cfg = SlidingWindowConfig(max_tokens=1000, keep_last_n_messages=5)
        assert cfg.max_tokens == 1000

    def test_to_dict(self):
        cfg = SlidingWindowConfig(max_tokens=5000)
        d = cfg.to_dict()
        assert d["max_tokens"] == 5000


class TestCompactionTrigger:
    def test_auto_below_threshold(self):
        tr = CompactionTrigger(mode=TriggerMode.AUTO, token_threshold=1000)
        assert tr.should_compact(500) is False

    def test_auto_above_threshold(self):
        tr = CompactionTrigger(mode=TriggerMode.AUTO, token_threshold=1000)
        assert tr.should_compact(1500) is True

    def test_manual_never_auto(self):
        tr = CompactionTrigger(mode=TriggerMode.MANUAL)
        assert tr.should_compact(999_999) is False

    def test_manual_trigger(self):
        tr = CompactionTrigger(mode=TriggerMode.MANUAL)
        assert tr.trigger() is True


class TestCompactionSummary:
    def test_create(self):
        s = CompactionSummary(epoch_id="E1", tokens_before=1000, tokens_after=500)
        assert s.tokens_saved == 500

    def test_compression_ratio(self):
        s = CompactionSummary(epoch_id="E1", tokens_before=1000, tokens_after=250)
        assert s.compression_ratio == 0.25

    def test_compression_ratio_zero_before(self):
        s = CompactionSummary(epoch_id="E1", tokens_before=0, tokens_after=0)
        assert s.compression_ratio == 1.0

    def test_to_dict(self):
        s = CompactionSummary(epoch_id="E1", summary_text="Sum", compressed_segments=["user"])
        d = s.to_dict()
        assert d["compressed_segments_count"] == 1


class TestCompactionEngine:
    def _make_engine(self, **kwargs):
        return CompactionEngine(**kwargs)

    def test_add_message(self):
        engine = self._make_engine()
        engine.add_message("user", "Hello")
        assert engine.message_count == 1

    def test_add_system_message(self):
        engine = self._make_engine()
        engine.add_system_message("System prompt")
        assert engine.system_message_count == 1

    def test_current_tokens_estimate(self):
        engine = self._make_engine()
        engine.add_message("user", "A" * 100)
        assert engine.current_tokens >= 25  # 100 chars / 4 = 25 tokens

    def test_should_compact_auto_true(self):
        cfg = SlidingWindowConfig()
        tr = CompactionTrigger(mode=TriggerMode.AUTO, token_threshold=10)
        engine = self._make_engine(config=cfg, trigger=tr)
        engine.add_message("user", "A" * 100)
        assert engine.should_compact() is True

    def test_should_compact_auto_false(self):
        cfg = SlidingWindowConfig()
        tr = CompactionTrigger(mode=TriggerMode.AUTO, token_threshold=999_999)
        engine = self._make_engine(config=cfg, trigger=tr)
        engine.add_message("user", "Hi")
        assert engine.should_compact() is False

    def test_compact_reduces_messages(self):
        cfg = SlidingWindowConfig(keep_last_n_messages=3)
        engine = self._make_engine(config=cfg)
        for i in range(10):
            engine.add_message("user", f"Msg {i}")

        summary = engine.compact(epoch_id="E1")
        # After compaction, only 3 messages + 1 compaction marker remain
        assert engine.message_count <= 5
        assert summary.tokens_saved >= 0

    def test_compact_preserves_system_messages(self):
        cfg = SlidingWindowConfig(keep_last_n_messages=2)
        engine = self._make_engine(config=cfg)
        engine.add_system_message("Always keep")
        for i in range(10):
            engine.add_message("user", f"Msg {i}")

        engine.compact(epoch_id="E1")
        assert engine.system_message_count == 1

    def test_compact_returns_summary(self):
        cfg = SlidingWindowConfig(keep_last_n_messages=2)
        engine = self._make_engine(config=cfg)
        for i in range(5):
            engine.add_message("user", f"Msg {i}")

        summary = engine.compact(epoch_id="E1")
        assert summary.epoch_id == "E1"
        assert len(summary.summary_text) > 0

    def test_compact_noop_when_under_window(self):
        cfg = SlidingWindowConfig(keep_last_n_messages=100)
        engine = self._make_engine(config=cfg)
        engine.add_message("user", "One")

        summary = engine.compact(epoch_id="E1")
        assert summary.summary_text == ""
        assert summary.tokens_saved == 0

    def test_compact_manual_mode(self):
        cfg = SlidingWindowConfig(keep_last_n_messages=2)
        tr = CompactionTrigger(mode=TriggerMode.MANUAL)
        engine = self._make_engine(config=cfg, trigger=tr)
        for i in range(5):
            engine.add_message("user", f"Msg {i}")

        summary = engine.compact(epoch_id="E1")
        assert summary is not None

    def test_compact_history(self):
        cfg = SlidingWindowConfig(keep_last_n_messages=2)
        engine = self._make_engine(config=cfg)
        for i in range(5):
            engine.add_message("user", f"Msg {i}")

        engine.compact(epoch_id="E1")
        history = engine.get_compaction_history()
        assert len(history) == 1

    def test_total_saved_tokens(self):
        cfg = SlidingWindowConfig(keep_last_n_messages=2)
        engine = self._make_engine(config=cfg)
        for i in range(5):
            engine.add_message("user", f"Msg {i}")

        engine.compact(epoch_id="E1")
        assert engine.get_total_saved_tokens() >= 0

    def test_get_messages(self):
        engine = self._make_engine()
        engine.add_system_message("SYS")
        engine.add_message("user", "Hi")
        msgs = engine.get_messages(include_system=True)
        assert any(m["role"] == "system" for m in msgs)

    def test_get_messages_no_system(self):
        engine = self._make_engine()
        engine.add_system_message("SYS")
        engine.add_message("user", "Hi")
        msgs = engine.get_messages(include_system=False)
        assert all(m["role"] != "system" for m in msgs)

    def test_reset(self):
        engine = self._make_engine()
        engine.add_message("user", "Hi")
        engine.reset()
        assert engine.message_count == 0
        assert engine.system_message_count == 0

    def test_to_dict(self):
        engine = self._make_engine()
        d = engine.to_dict()
        assert "current_tokens" in d
        assert "message_count" in d

    def test_custom_summary_fn(self):
        def my_fn(msgs):
            return "Custom summary"

        cfg = SlidingWindowConfig(keep_last_n_messages=1)
        engine = self._make_engine(config=cfg, custom_summary_fn=my_fn)
        for i in range(3):
            engine.add_message("user", f"Msg {i}")

        summary = engine.compact(epoch_id="E1")
        assert "Custom summary" in summary.summary_text
