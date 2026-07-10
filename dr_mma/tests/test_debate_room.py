"""DebateRoom unit tests."""

import pytest
from dr_mma.engine.debate_room import DebateRoom, DebateTurn, DebateResult
from dr_mma.engine.events import EventBus


class TestDebateInitiation:
    def test_initiate_returns_debate_result(self):
        room = DebateRoom()
        result = room.initiate(topic="A vs B", participants=["Worker", "Critic"])
        assert isinstance(result, DebateResult)
        assert result.debate_id.startswith("DEB-")
        assert result.status == "in_progress"

    def test_initiate_with_custom_id(self):
        room = DebateRoom()
        result = room.initiate(topic="X", participants=["A"], debate_id="CUSTOM-1")
        assert result.debate_id == "CUSTOM-1"

    def test_initiate_publishes_event(self):
        bus = EventBus()
        room = DebateRoom(event_bus=bus)
        room.initiate(topic="T", participants=["W", "C"])
        events = bus.query("debate_initiated")
        assert len(events) == 1
        assert events[0].payload["topic"] == "T"

    def test_initiate_without_event_bus(self):
        room = DebateRoom()
        result = room.initiate(topic="T", participants=["W"])
        assert result.status == "in_progress"


class TestDebateTurns:
    def _setup(self):
        bus = EventBus()
        room = DebateRoom(event_bus=bus)
        room.initiate(topic="Topic", participants=["Worker", "Critic"], debate_id="D1")
        return room, bus

    def test_add_turn_increments_round(self):
        room, _ = self._setup()
        t1 = room.add_turn("D1", "Worker", "m1", "arg1")
        t2 = room.add_turn("D1", "Critic", "m2", "arg2")
        assert t1.round_number == 1
        assert t2.round_number == 2

    def test_add_turn_publishes_event(self):
        room, bus = self._setup()
        room.add_turn("D1", "Worker", "m1", "x")
        events = bus.query("debate_turn")
        assert len(events) == 1
        assert events[0].payload["role"] == "Worker"

    def test_add_turn_to_unknown_debate_raises(self):
        room, _ = self._setup()
        with pytest.raises(ValueError, match="not found"):
            room.add_turn("NOPE", "W", "m1", "x")

    def test_add_turn_to_resolved_debate_raises(self):
        room, _ = self._setup()
        room.resolve("D1", ruling="done")
        with pytest.raises(ValueError, match="not active"):
            room.add_turn("D1", "W", "m1", "x")

    def test_turn_count(self):
        room, _ = self._setup()
        assert room.turn_count("D1") == 0
        room.add_turn("D1", "Worker", "m1", "a")
        room.add_turn("D1", "Critic", "m2", "b")
        assert room.turn_count("D1") == 2

    def test_has_reached_max_rounds(self):
        room, _ = self._setup()
        assert not room.has_reached_max_rounds("D1", 3)
        room.add_turn("D1", "Worker", "m1", "a")
        room.add_turn("D1", "Critic", "m2", "b")
        room.add_turn("D1", "Worker", "m1", "c")
        assert room.has_reached_max_rounds("D1", 3)


class TestDebateResolve:
    def _setup(self):
        bus = EventBus()
        room = DebateRoom(event_bus=bus)
        room.initiate(topic="T", participants=["W", "C"], debate_id="R1")
        room.add_turn("R1", "W", "m1", "yes")
        room.add_turn("R1", "C", "m2", "no")
        return room, bus

    def test_resolve_sets_status(self):
        room, _ = self._setup()
        res = room.resolve("R1", ruling="W wins", rationale="because")
        assert res.status == "resolved"
        assert res.ruling == "W wins"
        assert res.rationale == "because"
        assert res.resolved_at is not None

    def test_resolve_with_evidence(self):
        room, _ = self._setup()
        res = room.resolve("R1", ruling="ok", evidence_refs=["BB-001"])
        assert res.evidence_refs == ["BB-001"]

    def test_resolve_publishes_event(self):
        room, bus = self._setup()
        room.resolve("R1", ruling="x")
        events = bus.query("debate_resolved")
        assert len(events) == 1
        assert events[0].payload["turn_count"] == 2

    def test_resolve_unknown_debate_raises(self):
        room, _ = self._setup()
        with pytest.raises(ValueError):
            room.resolve("NOPE", ruling="x")


class TestDebateQueries:
    def _setup_many(self):
        bus = EventBus()
        room = DebateRoom(event_bus=bus)
        room.initiate(topic="T1", participants=["W"], debate_id="A")
        room.initiate(topic="T2", participants=["W"], debate_id="B")
        room.resolve("A", ruling="done")
        return room

    def test_get_debate(self):
        room = self._setup_many()
        assert room.get_debate("A").topic == "T1"
        assert room.get_debate("NOPE") is None

    def test_list_active_debates(self):
        room = self._setup_many()
        active = room.list_active_debates()
        assert len(active) == 1
        assert active[0].debate_id == "B"

    def test_list_resolved_debates(self):
        room = self._setup_many()
        resolved = room.list_resolved_debates()
        assert len(resolved) == 1
        assert resolved[0].debate_id == "A"

    def test_all_debates(self):
        room = self._setup_many()
        assert len(room.all_debates()) == 2

    def test_get_turns_by_role(self):
        bus = EventBus()
        room = DebateRoom(event_bus=bus)
        room.initiate(topic="T", participants=["Worker", "Critic"], debate_id="X")
        room.add_turn("X", "Worker", "m1", "a")
        room.add_turn("X", "Critic", "m2", "b")
        room.add_turn("X", "Worker", "m1", "c")
        w_turns = room.get_turns_by_role("X", "Worker")
        assert len(w_turns) == 2
        assert all(t.role == "Worker" for t in w_turns)

    def test_get_turns_by_role_unknown_debate(self):
        room = self._setup_many()
        assert room.get_turns_by_role("NOPE", "W") == []


class TestDebateTurnToDict:
    def test_turn_to_dict(self):
        t = DebateTurn(round_number=1, role="Worker", model_id="m1", content="hello")
        d = t.to_dict()
        assert d["round"] == 1
        assert d["role"] == "Worker"
        assert d["content"] == "hello"

    def test_result_to_dict(self):
        dr = DebateResult(debate_id="D1", topic="T", status="resolved")
        dr.turns = [DebateTurn(1, "W", "m1", "x")]
        dr.ruling = "ok"
        d = dr.to_dict()
        assert d["debate_id"] == "D1"
        assert len(d["turns"]) == 1
        assert d["ruling"] == "ok"
