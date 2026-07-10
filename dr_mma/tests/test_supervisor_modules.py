"""SupervisorModules unit tests."""

import pytest
from dr_mma.engine.supervisor_modules import (
    TaskUnderstandingModule,
    DAGPlanningModule,
    EventHandlingModule,
    DecisionModule,
    FinalReviewModule,
    SupervisorOrchestrator,
    TaskUnderstandingResult,
    DAGPlan,
    EventHandlingDecision,
    DecisionResult,
    FinalReviewResult,
)


class TestTaskUnderstandingModule:
    def _mod(self):
        return TaskUnderstandingModule()

    def test_basic_analyze(self):
        r = self._mod().analyze("T1", "写一份报告")
        assert r.task_id == "T1"
        assert r.goal == "写一份报告"
        assert r.risk_level == "low"

    def test_constraints_passed_through(self):
        r = self._mod().analyze("T1", "X", constraints=["C1", "C2"])
        assert "C1" in r.constraints
        assert "C2" in r.constraints

    def test_output_format_added(self):
        r = self._mod().analyze("T1", "X", output_format="JSON")
        assert any("JSON" in req for req in r.output_requirements)

    def test_complexity_estimation(self):
        long_task = "A" * 200
        r = self._mod().analyze("T1", long_task)
        assert r.estimated_complexity >= 1
        assert r.estimated_complexity <= 12

    def test_risk_high_keyword(self):
        r = self._mod().analyze("T1", "关键安全任务")
        assert r.risk_level == "high"

    def test_risk_medium_long_task(self):
        r = self._mod().analyze("T1", "A" * 150)
        assert r.risk_level == "medium"

    def test_to_dict(self):
        r = self._mod().analyze("T1", "X")
        d = r.to_dict()
        assert d["task_id"] == "T1"
        assert d["goal"] == "X"


class TestDAGPlanningModule:
    def _mod(self):
        return DAGPlanningModule()

    def test_plan_creates_nodes(self):
        mod = self._mod()
        understanding = TaskUnderstandingResult(task_id="WF1", goal="G")
        subtasks = [
            {"task_name": "A", "objective": "Do A"},
            {"task_name": "B", "objective": "Do B"},
        ]
        plan = mod.plan("WF1", understanding, subtasks)
        assert len(plan.nodes) == 2
        assert plan.nodes[0]["node_id"] == "WF1-T00"

    def test_plan_creates_edges(self):
        mod = self._mod()
        understanding = TaskUnderstandingResult(task_id="WF1", goal="G")
        subtasks = [
            {"task_name": "A", "depends_on": []},
            {"task_name": "B", "depends_on": [0]},
        ]
        plan = mod.plan("WF1", understanding, subtasks)
        assert ("WF1-T00", "WF1-T01") in plan.edges

    def test_plan_parallel_groups(self):
        mod = self._mod()
        understanding = TaskUnderstandingResult(task_id="WF1", goal="G")
        subtasks = [
            {"task_name": "A", "depends_on": []},
            {"task_name": "B", "depends_on": []},
            {"task_name": "C", "depends_on": []},
        ]
        plan = mod.plan("WF1", understanding, subtasks)
        assert len(plan.parallel_groups) == 1
        assert len(plan.parallel_groups[0]) == 3

    def test_plan_no_parallel_when_dependent(self):
        mod = self._mod()
        understanding = TaskUnderstandingResult(task_id="WF1", goal="G")
        subtasks = [
            {"task_name": "A", "depends_on": []},
            {"task_name": "B", "depends_on": [0]},
        ]
        plan = mod.plan("WF1", understanding, subtasks)
        assert len(plan.parallel_groups) == 0

    def test_execution_order(self):
        mod = self._mod()
        understanding = TaskUnderstandingResult(task_id="WF1", goal="G")
        subtasks = [{"task_name": "A"}, {"task_name": "B"}]
        plan = mod.plan("WF1", understanding, subtasks)
        assert len(plan.execution_order) == 2

    def test_to_dict(self):
        mod = self._mod()
        understanding = TaskUnderstandingResult(task_id="WF1", goal="G")
        plan = mod.plan("WF1", understanding, [{"task_name": "X"}])
        d = plan.to_dict()
        assert d["task_id"] == "WF1"
        assert len(d["nodes"]) == 1


class TestEventHandlingModule:
    def _mod(self):
        return EventHandlingModule()

    def test_low_confidence_escalates(self):
        d = self._mod().handle("low_confidence", {})
        assert d.action == "escalate"
        assert d.target_role == "Supervisor"

    def test_review_failed_retries(self):
        d = self._mod().handle("review_failed", {"retry": 0})
        assert d.action == "retry"
        assert d.target_role == "Worker"

    def test_review_failed_triggers_debate_after_two(self):
        d = self._mod().handle("review_failed", {"retry": 2})
        assert d.action == "trigger_debate"
        assert d.target_role == "DebateRoom"

    def test_need_replan(self):
        d = self._mod().handle("need_replan", {})
        assert d.action == "replan"
        assert d.target_role == "Planner"

    def test_tool_failed(self):
        d = self._mod().handle("tool_failed", {})
        assert d.action == "retry"

    def test_conflict_detected(self):
        d = self._mod().handle("conflict_detected", {})
        assert d.action == "trigger_debate"

    def test_unknown_event_ignored(self):
        d = self._mod().handle("unknown_xyz", {})
        assert d.action == "ignore"

    def test_event_history_recorded(self):
        mod = self._mod()
        mod.handle("low_confidence", {})
        mod.handle("tool_failed", {})
        history = mod.get_event_history()
        assert len(history) == 2

    def test_to_dict(self):
        d = self._mod().handle("low_confidence", {})
        dct = d.to_dict()
        assert dct["event_type"] == "low_confidence"
        assert dct["action"] == "escalate"


class TestDecisionModule:
    def _mod(self):
        return DecisionModule()

    def test_single_position_consensus(self):
        mod = self._mod()
        r = mod.decide("C1", "fact", [{"role": "W", "position": "A"}])
        assert r.ruling == "A"
        assert r.rationale == "唯一立场，直接采用"

    def test_consensus_by_confidence(self):
        mod = self._mod()
        positions = [
            {"role": "W", "position": "A", "confidence": 0.6},
            {"role": "C", "position": "B", "confidence": 0.9},
        ]
        r = mod.decide("C1", "logic", positions)
        assert r.ruling == "B"
        assert r.confidence == 0.9

    def test_evidence_based_decisions(self):
        mod = self._mod()
        positions = [
            {"role": "W", "position": "A", "confidence": 0.5, "has_evidence": True},
            {"role": "C", "position": "B", "confidence": 0.9, "has_evidence": False},
        ]
        r = mod.decide("C1", "fact", positions, evidence=["E1"])
        assert r.ruling == "A"
        assert r.evidence_refs == ["E1"]

    def test_dissenting_opinions_recorded(self):
        mod = self._mod()
        positions = [
            {"role": "W", "position": "A", "confidence": 0.9},
            {"role": "C", "position": "B", "confidence": 0.5},
        ]
        r = mod.decide("C1", "logic", positions)
        assert any("C" in opp for opp in r.dissenting_opinions)

    def test_empty_positions(self):
        mod = self._mod()
        r = mod.decide("C1", "fact", [])
        assert r.ruling == ""

    def test_decisions_accumulate(self):
        mod = self._mod()
        mod.decide("C1", "fact", [{"role": "W", "position": "A"}])
        mod.decide("C2", "fact", [{"role": "W", "position": "B"}])
        assert len(mod.get_decisions()) == 2

    def test_to_dict(self):
        mod = self._mod()
        r = mod.decide("C1", "fact", [{"role": "W", "position": "A"}])
        d = r.to_dict()
        assert d["conflict_id"] == "C1"
        assert d["ruling"] == "A"


class TestFinalReviewModule:
    def _mod(self):
        return FinalReviewModule()

    def test_passed_review(self):
        mod = self._mod()
        subtasks = {
            "S1": {"worker": type("W", (), {"status": "completed"})()},
            "S2": {"worker": type("W", (), {"status": "completed"})()},
        }
        r = mod.review("T1", "output", subtasks, [], [{"next_action_recommendation": "pass"}])
        assert r.status == "passed"

    def test_conditionally_passed(self):
        mod = self._mod()
        subtasks = {
            "S1": {"worker": type("W", (), {"status": "completed"})()},
            "S2": {"worker": type("W", (), {"status": "failed"})()},
            "S3": {"worker": type("W", (), {"status": "completed"})()},
        }
        r = mod.review("T1", "out", subtasks, [], [{"next_action_recommendation": "pass"}])
        assert r.status == "conditionally_passed"

    def test_failed_with_critic_issues(self):
        mod = self._mod()
        subtasks = {
            "S1": {"worker": type("W", (), {"status": "completed"})()},
        }
        critics = [{"next_action_recommendation": "revise", "summary": "Bad quality"}]
        r = mod.review("T1", "out", subtasks, critics, [])
        assert r.status == "failed"
        assert "Bad quality" in r.weaknesses

    def test_failed_low_completion(self):
        mod = self._mod()
        subtasks = {
            "S1": {"worker": type("W", (), {"status": "failed"})()},
            "S2": {"worker": type("W", (), {"status": "failed"})()},
        }
        r = mod.review("T1", "out", subtasks, [], [])
        assert r.status == "failed"
        assert any("50%" in w for w in r.weaknesses)

    def test_empty_subtasks(self):
        mod = self._mod()
        r = mod.review("T1", "out", {}, [], [])
        assert r.quality_score == 0

    def test_to_dict(self):
        mod = self._mod()
        r = mod.review("T1", "out", {}, [], [])
        d = r.to_dict()
        assert d["task_id"] == "T1"
        assert d["final_output"] == "out"


class TestSupervisorOrchestrator:
    def _orch(self):
        return SupervisorOrchestrator()

    def test_default_stage_idle(self):
        o = self._orch()
        assert o.current_stage == "idle"

    def test_enter_init_stage(self):
        o = self._orch()
        o.enter_stage("init")
        assert o.is_module_active("understanding") is True
        assert o.is_module_active("planning") is True
        assert o.is_module_active("event_handling") is False

    def test_enter_executing_stage(self):
        o = self._orch()
        o.enter_stage("executing")
        assert o.is_module_active("event_handling") is True

    def test_enter_resolving_stage(self):
        o = self._orch()
        o.enter_stage("resolving")
        assert o.is_module_active("decision") is True

    def test_enter_reviewing_stage(self):
        o = self._orch()
        o.enter_stage("reviewing")
        assert o.is_module_active("final_review") is True

    def test_understand_delegates(self):
        o = self._orch()
        r = o.understand("T1", "写报告", constraints=["C1"])
        assert isinstance(r, TaskUnderstandingResult)
        assert "C1" in r.constraints

    def test_plan_delegates(self):
        o = self._orch()
        u = o.understand("T1", "X")
        p = o.plan(u, [{"task_name": "A"}])
        assert isinstance(p, DAGPlan)
        assert len(p.nodes) == 1

    def test_handle_event_delegates(self):
        o = self._orch()
        d = o.handle_event("low_confidence", {})
        assert isinstance(d, EventHandlingDecision)
        assert d.action == "escalate"

    def test_decide_delegates(self):
        o = self._orch()
        r = o.decide("C1", "fact", [{"role": "W", "position": "A"}])
        assert isinstance(r, DecisionResult)
        assert r.ruling == "A"

    def test_review_delegates(self):
        o = self._orch()
        subtasks = {"S1": {"worker": type("W", (), {"status": "completed"})()}}
        r = o.review("T1", "out", subtasks, [], [{"next_action_recommendation": "pass"}])
        assert isinstance(r, FinalReviewResult)
        assert r.status == "passed"

    def test_full_workflow(self):
        o = self._orch()
        o.enter_stage("init")
        u = o.understand("WF1", "分析报告")
        p = o.plan(u, [{"task_name": "A"}, {"task_name": "B", "depends_on": [0]}])
        assert len(p.nodes) == 2

        o.enter_stage("executing")
        d = o.handle_event("tool_failed", {})
        assert d.action == "retry"

        o.enter_stage("resolving")
        ruling = o.decide("C1", "plan", [
            {"role": "W", "position": "X", "confidence": 0.7},
            {"role": "C", "position": "Y", "confidence": 0.4},
        ])
        assert ruling.ruling == "X"

        o.enter_stage("reviewing")
        subtasks = {"S1": {"worker": type("W", (), {"status": "completed"})()}}
        review = o.review("WF1", "final", subtasks, [], [{"next_action_recommendation": "pass"}])
        assert review.status == "passed"
