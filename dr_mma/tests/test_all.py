"""
DR-MMA 全模块测试套件

覆盖范围：
  - Schema: TaskContract, AgentResponse, BlackboardEntry
  - Models: ModelAdapter, MockModel
  - Roles: RolePromptLibrary, RoleRunner
  - Storage: Blackboard, ArtifactStore, DecisionLog
  - Engine: WorkflowEngine
"""

import sys
import os
import json
import tempfile
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

# ─── Schema Tests ─────────────────────────────────

from dr_mma.schemas.task_contract import TaskContract
from dr_mma.schemas.agent_response import AgentResponse, Claim, Risk, ArtifactRef
from dr_mma.schemas.blackboard_entry import BlackboardEntry


class TestTaskContract:
    def test_default_id_generation(self):
        c = TaskContract()
        assert c.task_id.startswith("T-")
        assert len(c.task_id) == 10  # "T-" + 8 hex chars

    def test_custom_id(self):
        c = TaskContract(task_id="T-custom")
        assert c.task_id == "T-custom"

    def test_to_dict(self):
        c = TaskContract(task_id="T-001", task_name="测试", role="Worker",
                         objective="完成测试")
        d = c.to_dict()
        assert d["task_id"] == "T-001"
        assert d["role"] == "Worker"

    def test_from_dict(self):
        d = {"task_id": "T-002", "task_name": "审查", "role": "Critic",
             "objective": "审查输出"}
        c = TaskContract.from_dict(d)
        assert c.task_id == "T-002"
        assert c.role == "Critic"

    def test_to_json(self):
        c = TaskContract(task_id="T-003", task_name="汇总", role="Supervisor")
        j = json.loads(c.to_json())
        assert j["task_id"] == "T-003"

    def test_short_summary(self):
        c = TaskContract(task_id="T-004", task_name="规划", role="Planner")
        assert c.short_summary() == "[T-004] 规划 (Planner)"

    def test_input_refs(self):
        c = TaskContract(task_id="T-005", input_refs=["BB-001", "BB-002"])
        assert len(c.input_refs) == 2


class TestAgentResponse:
    def test_default_status(self):
        r = AgentResponse()
        assert r.status == "completed"
        assert r.is_success()

    def test_status_checkers(self):
        r1 = AgentResponse(status="need_review")
        assert r1.needs_review()
        assert not r1.is_success()

        r2 = AgentResponse(status="low_confidence")
        assert r2.needs_review()

    def test_to_dict(self):
        r = AgentResponse(
            task_id="T-001",
            role="Worker",
            summary="执行完成",
            content="详细内容",
            claims=[Claim(claim="结论A", confidence=0.9)],
            risks=[Risk(risk="风险1", severity="low")],
        )
        d = r.to_dict()
        assert d["task_id"] == "T-001"
        assert len(d["claims"]) == 1
        assert d["claims"][0]["claim"] == "结论A"

    def test_from_dict(self):
        d = {
            "task_id": "T-002",
            "role": "Critic",
            "status": "completed",
            "summary": "审查通过",
            "content": "无问题",
            "claims": [{"claim": "无缺陷", "confidence": 0.95}],
        }
        r = AgentResponse.from_dict(d)
        assert r.role == "Critic"
        assert len(r.claims) == 1
        assert r.claims[0].confidence == 0.95

    def test_artifacts(self):
        r = AgentResponse(
            artifacts=[ArtifactRef(artifact_id="ART-001", version=2)]
        )
        assert r.artifacts[0].version == 2


class TestBlackboardEntry:
    def test_auto_id(self):
        e = BlackboardEntry()
        assert e.entry_id.startswith("BB-")

    def test_auto_timestamp(self):
        e = BlackboardEntry()
        assert e.created_at != ""

    def test_custom_id(self):
        e = BlackboardEntry(entry_id="BB-custom")
        assert e.entry_id == "BB-custom"

    def test_to_dict(self):
        e = BlackboardEntry(
            task_id="T-001",
            source_role="Planner",
            content_type="task_output",
            summary="规划完成",
            payload={"subtasks": [{"name": "task1"}]},
        )
        d = e.to_dict()
        assert d["source_role"] == "Planner"
        assert d["payload"]["subtasks"][0]["name"] == "task1"

    def test_roundtrip(self):
        e1 = BlackboardEntry(
            task_id="T-001",
            source_role="Worker",
            content_type="task_output",
            summary="测试",
        )
        d = e1.to_dict()
        e2 = BlackboardEntry.from_dict(d)
        assert e2.entry_id == e1.entry_id
        assert e2.source_role == "Worker"

    def test_to_json_line(self):
        e = BlackboardEntry(entry_id="BB-test")
        line = e.to_json_line()
        data = json.loads(line)
        assert data["entry_id"] == "BB-test"


# ─── Model Tests ─────────────────────────────────

from dr_mma.models.adapter import ModelAdapter, MockModel, ChatMessage, ModelResponse


class TestModelAdapter:
    def test_register_and_get(self):
        adapter = ModelAdapter()
        mock = MockModel("mock1")
        adapter.register("mock1", mock)
        assert adapter.get("mock1") is mock
        assert "mock1" in adapter.available_models

    def test_unregistered_model(self):
        adapter = ModelAdapter()
        resp = adapter.chat("unknown", [ChatMessage(role="user", content="hi")])
        assert resp.status == "error"
        assert "未注册" in resp.content

    def test_mock_chat_default(self):
        adapter = ModelAdapter()
        mock = MockModel("mock1")
        adapter.register("mock1", mock)
        resp = adapter.chat("mock1", [ChatMessage(role="user", content="hello")])
        assert resp.status == "success"
        assert "模拟响应" in resp.content

    def test_mock_chat_keyword(self):
        adapter = ModelAdapter()
        mock = MockModel("mock1", {"规划": "已规划完成"})
        adapter.register("mock1", mock)
        resp = adapter.chat("mock1", [ChatMessage(role="user", content="请规划任务")])
        assert "已规划完成" in resp.content

    def test_mock_call_count(self):
        mock = MockModel("mock1")
        adapter = ModelAdapter()
        adapter.register("mock1", mock)
        adapter.chat("mock1", [ChatMessage(role="user", content="a")])
        adapter.chat("mock1", [ChatMessage(role="user", content="b")])
        assert mock.call_count == 2


class TestChatMessage:
    def test_to_dict(self):
        m = ChatMessage(role="user", content="hello")
        assert m.to_dict() == {"role": "user", "content": "hello"}

    def test_system_message(self):
        m = ChatMessage(role="system", content="you are a bot")
        assert m.to_dict()["role"] == "system"


class TestModelResponse:
    def test_defaults(self):
        r = ModelResponse()
        assert r.status == "success"
        assert r.content == ""

    def test_str(self):
        r = ModelResponse(content="hello world")
        assert str(r) == "hello world"


# ─── Role Tests ──────────────────────────────────

from dr_mma.roles.prompts import RolePromptLibrary
from dr_mma.roles.runner import RoleRunner


class TestRolePromptLibrary:
    def test_all_prompts_nonempty(self):
        for role in ["Planner", "Worker", "Critic", "Verifier", "Supervisor"]:
            prompt = RolePromptLibrary.get_prompt(role)
            assert prompt != "", f"{role} prompt should not be empty"

    def test_unknown_role(self):
        prompt = RolePromptLibrary.get_prompt("Unknown")
        assert prompt == ""

    def test_planner_has_subtasks(self):
        prompt = RolePromptLibrary.planner()
        assert "subtasks" in prompt

    def test_worker_has_task_id(self):
        prompt = RolePromptLibrary.worker()
        assert "task_id" in prompt

    def test_critic_has_claims(self):
        prompt = RolePromptLibrary.critic()
        assert "claims" in prompt

    def test_verifier_has_pass_fail(self):
        prompt = RolePromptLibrary.verifier()
        assert "PASS" in prompt or "FAIL" in prompt

    def test_supervisor_has_final(self):
        prompt = RolePromptLibrary.supervisor()
        assert "最终" in prompt


class TestRoleRunner:
    def test_unknown_role_returns_failed(self):
        adapter = ModelAdapter()
        mock = MockModel("mock1")
        adapter.register("mock1", mock)
        runner = RoleRunner(adapter)
        contract = TaskContract(task_id="T-test", role="UnknownRole")
        resp = runner.run(contract, "mock1")
        assert resp.status == "failed"

    def test_planner_mock_run(self):
        adapter = ModelAdapter()
        mock = MockModel("mock1", {"规划": '{"subtasks": [{"task_name": "t1"}]}'})
        adapter.register("mock1", mock)
        runner = RoleRunner(adapter)
        contract = TaskContract(task_id="T-plan", role="Planner",
                                objective="测试规划")
        resp = runner.run(contract, "mock1")
        assert resp.status == "completed"

    def test_extract_json_from_codeblock(self):
        text = "Some text\n```json\n{\"key\": \"value\"}\n```\nmore text"
        result = RoleRunner._extract_json(text)
        assert result is not None
        data = json.loads(result)
        assert data["key"] == "value"

    def test_extract_json_braces(self):
        text = '前置文字{"a": 1, "b": 2}后置文字'
        result = RoleRunner._extract_json(text)
        assert result is not None
        data = json.loads(result)
        assert data["a"] == 1

    def test_extract_json_none(self):
        text = "没有任何 JSON 内容的纯文本"
        result = RoleRunner._extract_json(text)
        assert result is None

    def test_blackboard_entries_to_context(self):
        entries = [
            BlackboardEntry(
                task_id="T-001",
                source_role="Worker",
                content_type="task_output",
                summary="第一步完成",
            ),
        ]
        msgs = RoleRunner.blackboard_entries_to_context(entries)
        assert len(msgs) == 1
        assert "第一步完成" in msgs[0].content


# ─── Storage Tests ───────────────────────────────

from dr_mma.storage.blackboard import Blackboard
from dr_mma.storage.artifact_store import ArtifactStore
from dr_mma.storage.decision_log import DecisionLog


class TestBlackboard:
    @pytest.fixture
    def tmp_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_write_and_count(self, tmp_file):
        bb = Blackboard(tmp_file)
        assert bb.count() == 0
        entry = BlackboardEntry(task_id="T-001", source_role="Planner",
                                content_type="task_output", summary="test")
        bb.write(entry)
        assert bb.count() == 1

    def test_read(self, tmp_file):
        bb = Blackboard(tmp_file)
        entry = BlackboardEntry(task_id="T-001", source_role="Worker",
                                content_type="task_output", summary="read_test")
        eid = bb.write(entry)
        found = bb.read(eid)
        assert found is not None
        assert found.summary == "read_test"

    def test_read_not_found(self, tmp_file):
        bb = Blackboard(tmp_file)
        assert bb.read("nonexistent") is None

    def test_query_by_task_id(self, tmp_file):
        bb = Blackboard(tmp_file)
        bb.write(BlackboardEntry(task_id="T-001", source_role="Planner"))
        bb.write(BlackboardEntry(task_id="T-002", source_role="Worker"))
        results = bb.query(task_id="T-001")
        assert len(results) == 1

    def test_query_by_role(self, tmp_file):
        bb = Blackboard(tmp_file)
        bb.write(BlackboardEntry(task_id="T-001", source_role="Planner"))
        bb.write(BlackboardEntry(task_id="T-002", source_role="Worker"))
        bb.write(BlackboardEntry(task_id="T-003", source_role="Worker"))
        results = bb.query(source_role="Worker")
        assert len(results) == 2

    def test_query_limit(self, tmp_file):
        bb = Blackboard(tmp_file)
        for i in range(5):
            bb.write(BlackboardEntry(task_id=f"T-{i:03d}", source_role="Planner"))
        results = bb.query(limit=2)
        assert len(results) == 2

    def test_persistence(self, tmp_file):
        bb1 = Blackboard(tmp_file)
        bb1.write(BlackboardEntry(task_id="T-001", source_role="Planner",
                                   content_type="task_output", summary="persist"))
        # Re-open and verify
        bb2 = Blackboard(tmp_file)
        assert bb2.count() == 1
        assert bb2.query(task_id="T-001")[0].summary == "persist"

    def test_get_latest(self, tmp_file):
        bb = Blackboard(tmp_file)
        e1 = BlackboardEntry(task_id="T-001", summary="first")
        e2 = BlackboardEntry(task_id="T-001", summary="second")
        bb.write(e1)
        bb.write(e2)
        latest = bb.get_latest("T-001")
        assert latest.summary == "second"

    def test_clear(self, tmp_file):
        bb = Blackboard(tmp_file)
        bb.write(BlackboardEntry(task_id="T-001"))
        bb.clear()
        assert bb.count() == 0
        assert not os.path.exists(tmp_file)


class TestArtifactStore:
    @pytest.fixture
    def tmp_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_save_and_retrieve(self, tmp_file):
        store = ArtifactStore(tmp_file)
        av = store.save("ART-001", "测试内容")
        assert av.artifact_id == "ART-001"
        assert av.version == 1
        assert av.content == "测试内容"

    def test_version_increment(self, tmp_file):
        store = ArtifactStore(tmp_file)
        store.save("ART-001", "v1")
        av2 = store.save("ART-001", "v2")
        assert av2.version == 2

    def test_get_latest(self, tmp_file):
        store = ArtifactStore(tmp_file)
        store.save("ART-001", "v1")
        store.save("ART-001", "v2")
        latest = store.get_latest("ART-001")
        assert latest.content == "v2"

    def test_get_version(self, tmp_file):
        store = ArtifactStore(tmp_file)
        store.save("ART-001", "v1")
        store.save("ART-001", "v2")
        v1 = store.get_version("ART-001", 1)
        assert v1.content == "v1"

    def test_get_nonexistent(self, tmp_file):
        store = ArtifactStore(tmp_file)
        assert store.get_latest("NONEXIST") is None
        assert store.get_version("NONEXIST", 1) is None

    def test_list_versions(self, tmp_file):
        store = ArtifactStore(tmp_file)
        store.save("ART-001", "v1")
        store.save("ART-001", "v2")
        store.save("ART-001", "v3")
        versions = store.list_versions("ART-001")
        assert len(versions) == 3
        assert versions[0].version == 1
        assert versions[-1].version == 3

    def test_list_artifacts(self, tmp_file):
        store = ArtifactStore(tmp_file)
        store.save("ART-001", "a")
        store.save("ART-002", "b")
        artifacts = store.list_artifacts()
        assert "ART-001" in artifacts
        assert "ART-002" in artifacts

    def test_persistence(self, tmp_file):
        store1 = ArtifactStore(tmp_file)
        store1.save("ART-001", "persistent content", {"key": "val"})
        store2 = ArtifactStore(tmp_file)
        latest = store2.get_latest("ART-001")
        assert latest.content == "persistent content"
        assert latest.metadata["key"] == "val"

    def test_count(self, tmp_file):
        store = ArtifactStore(tmp_file)
        store.save("ART-001", "a")
        store.save("ART-001", "b")
        store.save("ART-002", "c")
        assert store.count() == 3


class TestDecisionLog:
    @pytest.fixture
    def tmp_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_log_and_count(self, tmp_file):
        log = DecisionLog(tmp_file)
        assert log.count() == 0
        log.log("T-001", "plan_created", "任务已规划")
        assert log.count() == 1

    def test_query_by_task_id(self, tmp_file):
        log = DecisionLog(tmp_file)
        log.log("T-001", "decision_a", "决策A")
        log.log("T-002", "decision_b", "决策B")
        results = log.query(task_id="T-001")
        assert len(results) == 1
        assert results[0].decision == "decision_a"

    def test_query_by_decision_type(self, tmp_file):
        log = DecisionLog(tmp_file)
        log.log("T-001", "retry", "重试")
        log.log("T-002", "complete", "完成")
        results = log.query(decision="retry")
        assert len(results) == 1

    def test_persistence(self, tmp_file):
        log1 = DecisionLog(tmp_file)
        log1.log("T-001", "test", "持久化测试", {"extra": "data"})
        log2 = DecisionLog(tmp_file)
        assert log2.count() == 1
        rec = log2.query(task_id="T-001")[0]
        assert rec.rationale == "持久化测试"
        assert rec.context["extra"] == "data"

    def test_query_limit(self, tmp_file):
        log = DecisionLog(tmp_file)
        for i in range(5):
            log.log(f"T-{i:03d}", "test", f"决策{i}")
        results = log.query(limit=2)
        assert len(results) == 2


# ─── Engine Tests ────────────────────────────────

from dr_mma.engine.workflow import WorkflowEngine
from dr_mma.engine.complexity import TaskComplexityEvaluator, MODE_DIRECT, MODE_STANDARD
from dr_mma.engine.capabilities import CapabilityRegistry, DynamicRoleAssigner
from dr_mma.engine.events import EventBus, EVENT_NEED_REPLAN


class TestComplexityEvaluator:
    def test_simple_task_routes_to_direct_mode(self):
        evaluator = TaskComplexityEvaluator()
        report = evaluator.evaluate("简单回答一句话")
        assert report.mode == MODE_DIRECT

    def test_complex_task_routes_to_standard_or_higher(self):
        evaluator = TaskComplexityEvaluator()
        report = evaluator.evaluate(
            "请设计一个支持权限、安全、数据库、API、实时同步和测试校验的多阶段系统方案，"
            "并给出步骤、风险和验证方式。"
        )
        assert report.mode in {MODE_STANDARD, "Expanded Mode"}


class TestDynamicRoleAssigner:
    def test_assigner_returns_requested_roles(self):
        registry = CapabilityRegistry()
        assigner = DynamicRoleAssigner(registry)
        assignments = assigner.assign(["Planner", "Worker", "Critic"], ["mock-a", "mock-b"])
        assert set(assignments.keys()) == {"Planner", "Worker", "Critic"}
        assert all(model in {"mock-a", "mock-b"} for model in assignments.values())


class TestEventBus:
    def test_event_bus_publish_and_query(self):
        bus = EventBus()
        bus.publish(EVENT_NEED_REPLAN, source="Critic", task_id="T-001", payload={"reason": "review_failed"})
        events = bus.query(event_type=EVENT_NEED_REPLAN, task_id="T-001")
        assert len(events) == 1
        assert events[0].payload["reason"] == "review_failed"


class TestWorkflowEngine:
    @pytest.fixture
    def engine(self):
        adapter = ModelAdapter()
        mock = MockModel("test_model")
        # Pre-configure keyword responses
        mock.add_response("Planner（规划者）", json.dumps({
            "subtasks": [
                {"task_name": "分析需求", "objective": "分析用户需求",
                 "success_criteria": ["完成分析"], "depends_on": []},
                {"task_name": "设计方案", "objective": "设计系统方案",
                 "success_criteria": ["完成设计"], "depends_on": ["分析需求"]},
            ]
        }))
        mock.add_response("Worker（执行者）", json.dumps({
            "status": "completed", "summary": "执行完成",
            "content": "任务执行结果详细内容",
        }))
        mock.add_response("Critic（审查者）", json.dumps({
            "status": "completed", "summary": "审查通过",
            "content": "没有发现问题",
            "next_action_recommendation": "PASS",
        }))
        mock.add_response("Verifier（校验者）", json.dumps({
            "status": "completed", "summary": "校验通过",
            "content": "所有问题已解决",
            "next_action_recommendation": "PASS",
        }))
        mock.add_response("Supervisor（主控者）", json.dumps({
            "status": "completed", "summary": "汇总完成",
            "content": "最终输出结果汇总",
        }))
        adapter.register("test_model", mock)

        # Temp files for dependencies
        self.tmp_bb = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        self.tmp_art = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        self.tmp_dec = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name

        engine = WorkflowEngine(
            adapter=adapter,
            blackboard=Blackboard(self.tmp_bb),
            artifact_store=ArtifactStore(self.tmp_art),
            decision_log=DecisionLog(self.tmp_dec),
            main_model="test_model",
        )
        yield engine

        # Cleanup
        for p in [self.tmp_bb, self.tmp_art, self.tmp_dec]:
            if os.path.exists(p):
                os.unlink(p)

    def test_workflow_execution(self, engine):
        result = engine.execute("测试任务: 分析需求并设计方案")
        assert result.status == "completed"
        assert len(result.subtask_results) > 0
        assert result.final_output != ""

    def test_workflow_blackboard_written(self, engine):
        result = engine.execute("测试任务")
        assert result.blackboard_count > 0

    def test_workflow_reports_mode_and_assignments(self, engine):
        result = engine.execute("请分析需求并设计一个包含权限、测试和验证步骤的方案")
        assert result.mode != ""
        assert result.complexity_score >= 0
        assert "Planner" in result.role_assignments
        assert result.event_count >= 1
        assert len(result.dag_nodes) >= 1

    def test_workflow_without_model(self):
        adapter = ModelAdapter()
        engine = WorkflowEngine(
            adapter=adapter,
            blackboard=Blackboard(self.tmp_bb if hasattr(self, 'tmp_bb') else tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name),
            artifact_store=ArtifactStore(self.tmp_art if hasattr(self, 'tmp_art') else tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name),
            decision_log=DecisionLog(self.tmp_dec if hasattr(self, 'tmp_dec') else tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name),
        )
        with pytest.raises(ValueError):
            engine.execute("test")

    def test_workflow_failed_plan(self, engine):
        # Override with a mock that returns no subtasks
        adapter = engine.adapter
        empty_mock = MockModel("empty")
        empty_mock.add_response("Planner（规划者）", '{"subtasks": []}')
        adapter.register("empty", empty_mock)
        engine.main_model = "empty"
        result = engine.execute("测试任务")
        assert result.status == "failed"


# ─── Integration: Full Round Trip ────────────────


class TestIntegration:
    def test_full_data_flow(self):
        """Schema → Model → Runner → Blackboard → Engine full round trip"""
        adapter = ModelAdapter()
        mock = MockModel("int_mock", {
            "Planner（规划者）": json.dumps({
                "subtasks": [
                    {"task_name": "子任务1", "objective": "目标1",
                     "success_criteria": ["完成"], "depends_on": []},
                ]
            }),
            "Worker（执行者）": json.dumps({
                "status": "completed", "summary": "执行成功",
                "content": "内容正文",
            }),
            "Critic（审查者）": json.dumps({
                "status": "completed", "summary": "审查通过",
                "content": "无问题",
                "next_action_recommendation": "PASS",
            }),
            "Verifier（校验者）": json.dumps({
                "status": "completed", "summary": "校验通过",
                "content": "通过",
                "next_action_recommendation": "PASS",
            }),
            "Supervisor（主控者）": json.dumps({
                "status": "completed", "summary": "最终汇总",
                "content": "这是最终输出结果",
            }),
        })
        adapter.register("int_mock", mock)

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f_bb:
            bb_path = f_bb.name
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f_art:
            art_path = f_art.name
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f_dec:
            dec_path = f_dec.name

        try:
            engine = WorkflowEngine(
                adapter=adapter,
                blackboard=Blackboard(bb_path),
                artifact_store=ArtifactStore(art_path),
                decision_log=DecisionLog(dec_path),
                main_model="int_mock",
            )
            result = engine.execute("集成测试: 完成一个完整任务")
            assert result.status == "completed"
            assert result.final_output != ""
            assert "最终输出" in result.final_output

            # Verify storage persistence
            bb2 = Blackboard(bb_path)
            assert bb2.count() > 0

            art2 = ArtifactStore(art_path)
            assert art2.count() > 0

            dec2 = DecisionLog(dec_path)
            assert dec2.count() > 0

        finally:
            for p in [bb_path, art_path, dec_path]:
                if os.path.exists(p):
                    os.unlink(p)

    def test_controller_persistence(self):
        """WorkflowController 模型配置持久化"""
        import tempfile
        from dr_mma.ui.controller import WorkflowController

        # 用临时文件模拟持久化
        ctrl = WorkflowController()

        # 注册两个模型
        ctrl.register_model("测试模型A", model_type="mock")
        ctrl.register_model("测试模型B", endpoint="http://127.0.0.1:1234/v1",
                            model_type="local")

        assert "测试模型A" in ctrl.get_available_models()
        assert "测试模型B" in ctrl.get_available_models()

        # 验证持久化文件存在且有内容
        from dr_mma.ui.controller import CONFIG_FILE
        assert CONFIG_FILE.exists()
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["models"]) >= 2

        # 清理
        if CONFIG_FILE.exists():
            CONFIG_FILE.unlink()

    def test_controller_runtime_snapshot_defaults(self):
        from dr_mma.ui.controller import WorkflowController

        ctrl = WorkflowController()
        snapshot = ctrl.get_runtime_snapshot()
        assert snapshot["events"] == []
        assert snapshot["assignments"] == {}
        assert snapshot["mode"] == ""
        assert snapshot["complexity_score"] == 0
        assert snapshot["dag_nodes"] == []

    def test_ui_modules_import(self):
        pytest.importorskip("customtkinter")

        import dr_mma.ui.chat_panel as chat_panel
        import dr_mma.ui.config_panel as config_panel
        import dr_mma.ui.log_panel as log_panel
        import dr_mma.ui.main_window as main_window
        import dr_mma.ui.pipeline_panel as pipeline_panel
        import dr_mma.ui.results_panel as results_panel
        import dr_mma.ui.task_panel as task_panel

        assert chat_panel.ChatPanel is not None
        assert config_panel.ConfigPanel is not None
        assert log_panel.LogPanel is not None
        assert main_window.MainWindow is not None
        assert pipeline_panel.PipelinePanel is not None
        assert results_panel.ResultsPanel is not None
        assert task_panel.TaskPanel is not None
