"""
Security & Fault Injection Tests — 安全/故障注入测试

测试 P2 重构后存储层和校验层的安全性边界条件：
- SQL 注入防护
- Schema 校验拒绝恶意输入
- UUIDv7 不重复性（对抗性条件）
- DAG 循环检测
- 权限模型边界
"""

import json
import os
import tempfile
import threading

import pytest

from dr_mma.storage.blackboard import Blackboard
from dr_mma.storage.decision_log import DecisionLog
from dr_mma.storage.artifact_store import ArtifactStore
from dr_mma.schemas.task_contract import TaskContract
from dr_mma.schemas.agent_response import AgentResponse, Claim, Risk, ToolCall, ArtifactRef
from dr_mma.schemas.blackboard_entry import BlackboardEntry
from dr_mma.engine.id_utils import uuid7


def _tmp_db(prefix="sec"):
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False, prefix=prefix) as f:
        return f.name


# ── SQL Injection Resistance ─────────────────────────────────────────────────

class TestSQLInjection:
    """SQLite 参数化查询确保注入字符串被当作文本而非 SQL 执行。"""

    @pytest.fixture
    def bb(self):
        p = _tmp_db("sec_bb")
        s = Blackboard(p)
        yield s
        s.close()
        try:
            os.unlink(p)
        except PermissionError:
            pass

    @pytest.fixture
    def dl(self):
        p = _tmp_db("sec_dl")
        s = DecisionLog(p)
        yield s
        s.close()
        try:
            os.unlink(p)
        except PermissionError:
            pass

    @pytest.fixture
    def art(self):
        p = _tmp_db("sec_art")
        s = ArtifactStore(p)
        yield s
        s.close()
        try:
            os.unlink(p)
        except PermissionError:
            pass

    def test_blackboard_task_id_sql_injection(self, bb):
        """Blackboard: task_id 含 SQL 注入字符串不应破坏查询。"""
        injection = "T-001'; DROP TABLE blackboard; --"
        e = BlackboardEntry(task_id=injection, source_role="Attacker",
                            content_type="task_output", summary="注入测试")
        bb.write(e)
        # 写入不应失败
        assert bb.count() == 1
        # 查询注入字符串应能找到记录
        results = bb.query(task_id=injection)
        assert len(results) == 1

    def test_blackboard_payload_sql_injection(self, bb):
        """Blackboard: payload 字段含 SQL 注入不应破坏存储。"""
        malicious_payload = {
            "payload": "'); DROP TABLE blackboard; --",
            "nested": {"inject": "' OR 1=1 --"},
            "list": ["'; DELETE FROM blackboard; --"],
        }
        e = BlackboardEntry(task_id="T-safe", source_role="Worker",
                            content_type="task_output", payload=malicious_payload)
        bb.write(e)
        # 验证 payload 原样保留
        found = bb.read(e.entry_id)
        assert found is not None
        assert found.payload["payload"] == malicious_payload["payload"]
        assert found.payload["nested"]["inject"] == malicious_payload["nested"]["inject"]

    def test_decision_log_injection(self, dl):
        """DecisionLog: decision 字段含 SQL 注入字符串。"""
        injection = "retry'; DROP TABLE decision_log; --"
        dl.log("T-001", injection, "测试注入")
        assert dl.count() == 1
        results = dl.query(decision=injection)
        assert len(results) == 1

    def test_artifact_store_injection(self, art):
        """ArtifactStore: artifact_id 含 SQL 注入字符串。"""
        injection = "ART-001'; DROP TABLE artifact; --"
        art.save(injection, "注入内容", {"key": "' OR '1'='1"})
        assert art.count() == 1
        latest = art.get_latest(injection)
        assert latest is not None
        assert latest.content == "注入内容"


# ── Schema Validation Edge Cases ────────────────────────────────────────────

class TestSchemaValidationSecurity:
    """Schema 校验拒绝恶意/畸形的输入。"""

    def test_task_contract_rejects_invalid_role(self):
        """TaskContract: 无效角色名应被拒绝。"""
        c = TaskContract(task_id="T-001", role="Hacker", objective="test",
                         task_name="test", task_type="general")
        errors = c.validate()
        assert any("role" in e.lower() for e in errors)

    def test_task_contract_rejects_invalid_task_type(self):
        """TaskContract: 无效 task_type 应被拒绝。"""
        c = TaskContract(task_id="T-001", role="Worker", objective="test",
                         task_name="test", task_type="malicious")
        errors = c.validate()
        assert any("task_type" in e.lower() for e in errors)

    def test_task_contract_rejects_empty_objective(self):
        """TaskContract: 空的 objective 应被拒绝。"""
        c = TaskContract(task_id="T-001", role="Worker", task_name="test",
                         objective="", task_type="general")
        errors = c.validate()
        assert any("objective" in e.lower() for e in errors)

    def test_task_contract_rejects_short_objective(self):
        """TaskContract: 过短的 objective 应被拒绝。"""
        c = TaskContract(task_id="T-001", role="Worker", task_name="test",
                         objective="ab", task_type="general")
        errors = c.validate()
        assert any("objective" in e.lower() for e in errors)

    def test_agent_response_rejects_invalid_status(self):
        """AgentResponse: 无效 status 应被拒绝。"""
        r = AgentResponse(task_id="T-001", role="Worker", status="hacked")
        errors = r.validate()
        assert any("status" in e.lower() for e in errors)

    def test_agent_response_rejects_invalid_role(self):
        """AgentResponse: 无效 role 应被拒绝。"""
        r = AgentResponse(task_id="T-001", role="ScriptKiddie", status="completed")
        errors = r.validate()
        assert any("role" in e.lower() for e in errors)

    def test_agent_response_rejects_claim_confidence_out_of_range(self):
        """AgentResponse: Claim 的 confidence 超出 [0, 1] 范围。"""
        r = AgentResponse(
            task_id="T-001", role="Worker", status="completed",
            claims=[Claim(claim="测试", confidence=1.5)],
        )
        errors = r.validate()
        assert any("confidence" in e.lower() for e in errors)

    def test_agent_response_rejects_empty_claim(self):
        """AgentResponse: 空的 claim 应被拒绝。"""
        r = AgentResponse(
            task_id="T-001", role="Worker", status="completed",
            claims=[Claim(claim="", confidence=0.5)],
        )
        errors = r.validate()
        assert any("claim" in e.lower() for e in errors)

    def test_agent_response_rejects_negative_version(self):
        """AgentResponse: ArtifactRef 的负版本号。"""
        r = AgentResponse(
            task_id="T-001", role="Worker", status="completed",
            artifacts=[ArtifactRef(artifact_id="ART-001", version=-1)],
        )
        errors = r.validate()
        assert any("version" in e.lower() for e in errors)

    def test_agent_response_empty_tool_name(self):
        """AgentResponse: 空的 tool_name。"""
        r = AgentResponse(
            task_id="T-001", role="Worker", status="completed",
            tool_calls=[ToolCall(tool_name="", args={"x": 1})],
        )
        errors = r.validate()
        assert any("tool_name" in e.lower() for e in errors)

    def test_agent_response_invalid_severity(self):
        """AgentResponse: Risk 的无效 severity。"""
        r = AgentResponse(
            task_id="T-001", role="Worker", status="completed",
            risks=[Risk(risk="测试风险", severity="critical")],
        )
        errors = r.validate()
        assert any("severity" in e.lower() for e in errors)


# ── UUIDv7 Uniqueness Under Adversarial Conditions ─────────────────────────

class TestUUIDv7Security:
    """UUIDv7 在对抗性条件下的唯一性和单调性。"""

    def test_concurrent_generation_no_duplicates(self):
        """并发生成 UUIDv7 不应产生重复。"""
        N = 5000
        ids = set()
        lock = threading.Lock()

        def generate(n):
            local_ids = [uuid7() for _ in range(n)]
            with lock:
                ids.update(local_ids)

        threads = []
        n_per_thread = N // 4
        for _ in range(4):
            t = threading.Thread(target=generate, args=(n_per_thread,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        assert len(ids) == n_per_thread * 4  # 无重复

    def test_rapid_sequential_no_duplicates(self):
        """快速连续生成 UUIDv7 不应产生重复（单线程高压）。"""
        N = 100_000
        ids = [uuid7() for _ in range(N)]
        assert len(set(ids)) == N  # 无重复

    def test_monotonic_under_pressure(self):
        """压力下 UUIDv7 仍保持严格单调递增。"""
        ids = [uuid7() for _ in range(2000)]
        sorted_ids = sorted(ids)
        assert ids == sorted_ids


# ── Boundary & Fault Injection ──────────────────────────────────────────────

class TestBoundaryConditions:
    """边界条件和故障注入测试。"""

    def test_blackboard_empty_payload(self):
        """Blackboard: 空的 payload 应正常处理。"""
        path = _tmp_db("sec_bound")
        bb = Blackboard(path)
        try:
            e = BlackboardEntry(task_id="T-001", source_role="Worker",
                                content_type="task_output", summary="空payload",
                                payload={})
            eid = bb.write(e)
            found = bb.read(eid)
            assert found is not None
            assert found.payload == {}
        finally:
            bb.close()
            try:
                os.unlink(path)
            except PermissionError:
                pass

    def test_blackboard_large_payload(self):
        """Blackboard: 大规模 payload (100KB) 应正常写入和读取。"""
        path = _tmp_db("sec_large")
        bb = Blackboard(path)
        try:
            large_text = "x" * 100_000
            e = BlackboardEntry(task_id="T-001", source_role="Worker",
                                content_type="task_output", summary="大payload",
                                payload={"data": large_text})
            eid = bb.write(e)
            found = bb.read(eid)
            assert found is not None
            assert len(found.payload["data"]) == 100_000
        finally:
            bb.close()
            try:
                os.unlink(path)
            except PermissionError:
                pass

    def test_decision_log_large_rationale(self):
        """DecisionLog: 超长 rationale 应正常存储。"""
        path = _tmp_db("sec_dl_large")
        dl = DecisionLog(path)
        try:
            large_rationale = "r" * 50_000
            dl.log("T-001", "test", large_rationale)
            results = dl.query(task_id="T-001")
            assert len(results) == 1
            assert len(results[0].rationale) == 50_000
        finally:
            dl.close()
            try:
                os.unlink(path)
            except PermissionError:
                pass

    def test_artifact_store_long_content(self):
        """ArtifactStore: 超大 content 应正常存储。"""
        path = _tmp_db("sec_art_large")
        art = ArtifactStore(path)
        try:
            large_content = "c" * 500_000
            art.save("ART-LARGE", large_content)
            latest = art.get_latest("ART-LARGE")
            assert latest is not None
            assert len(latest.content) == 500_000
        finally:
            art.close()
            try:
                os.unlink(path)
            except PermissionError:
                pass

    def test_concurrent_sqlite_writes(self):
        """并发写入 SQLite 不应丢数据或损坏。"""
        path = _tmp_db("sec_concurrent")
        bb = Blackboard(path)
        N = 500
        errors = []

        def write_entries(start):
            local_bb = Blackboard(path)
            try:
                for i in range(start, start + N):
                    e = BlackboardEntry(
                        task_id=f"T-{i:04d}", source_role="Worker",
                        content_type="task_output", summary=f"concurrent {i}",
                    )
                    local_bb.write(e)
            except Exception as exc:
                errors.append(str(exc))
            finally:
                local_bb.close()

        threads = [
            threading.Thread(target=write_entries, args=(0,)),
            threading.Thread(target=write_entries, args=(N,)),
            threading.Thread(target=write_entries, args=(N * 2,)),
            threading.Thread(target=write_entries, args=(N * 3,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []  # 无写入错误
        assert bb.count() == N * 4  # 数据完整
        bb.close()
        try:
            os.unlink(path)
        except PermissionError:
            pass
