"""
Benchmark Tests — 性能基准测试

测量关键路径的性能指标，确保 P2 重构（SQLite 迁移、Schema 校验、
UUIDv7 生成）不引入不可接受的性能退化。
"""

import time
import tempfile
import os
from pathlib import Path

import pytest

from dr_mma.engine.id_utils import uuid7, uuid7_hex, make_id
from dr_mma.storage.blackboard import Blackboard
from dr_mma.storage.decision_log import DecisionLog
from dr_mma.storage.artifact_store import ArtifactStore
from dr_mma.schemas.task_contract import TaskContract
from dr_mma.schemas.agent_response import AgentResponse, Claim, Risk
from dr_mma.schemas.blackboard_entry import BlackboardEntry
from dr_mma.engine.observability import DAGGraph


# ── Helpers ──────────────────────────────────────────────────────────────────

def _tmp_db(prefix: str = "bench") -> str:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False, prefix=prefix) as f:
        path = f.name
    return path


# ── UUIDv7 Generation Throughput ────────────────────────────────────────────

class TestUUID7Benchmark:
    def test_generate_10000(self):
        """Measure UUIDv7 generation throughput (target: < 10s for 100k)."""
        N = 10_000
        t0 = time.perf_counter()
        ids = [uuid7() for _ in range(N)]
        elapsed = time.perf_counter() - t0
        assert len(ids) == N
        assert len(set(ids)) == N  # No duplicates
        print(f"\n  UUIDv7 x{N}: {elapsed:.4f}s ({N/elapsed:.0f} uuid/s)")
        assert elapsed < 10.0  # Sanity threshold

    def test_generate_100000(self):
        """Stress test: 100k UUIDv7s."""
        N = 100_000
        t0 = time.perf_counter()
        ids = [uuid7() for _ in range(N)]
        elapsed = time.perf_counter() - t0
        assert len(set(ids)) == N
        print(f"\n  UUIDv7 x{N}: {elapsed:.4f}s ({N/elapsed:.0f} uuid/s)")
        assert elapsed < 30.0

    def test_make_id_throughput(self):
        """Measure make_id() throughput."""
        N = 10_000
        t0 = time.perf_counter()
        ids = [make_id("ART") for _ in range(N)]
        elapsed = time.perf_counter() - t0
        assert len(set(ids)) == N
        print(f"\n  make_id x{N}: {elapsed:.4f}s ({N/elapsed:.0f} id/s)")

    def test_uuid7_hex_throughput(self):
        """Measure uuid7_hex() throughput."""
        N = 10_000
        t0 = time.perf_counter()
        result = [uuid7_hex() for _ in range(N)]
        elapsed = time.perf_counter() - t0
        assert len(set(result)) == N
        print(f"\n  uuid7_hex x{N}: {elapsed:.4f}s ({N/elapsed:.0f} id/s)")


# ── SQLite Storage Throughput ───────────────────────────────────────────────

class TestSQLiteBenchmark:
    @pytest.fixture
    def bb(self):
        path = _tmp_db("bench_bb")
        store = Blackboard(path)
        yield store
        store.close()
        try:
            os.unlink(path)
        except PermissionError:
            pass

    @pytest.fixture
    def dl(self):
        path = _tmp_db("bench_dl")
        store = DecisionLog(path)
        yield store
        store.close()
        try:
            os.unlink(path)
        except PermissionError:
            pass

    @pytest.fixture
    def art(self):
        path = _tmp_db("bench_art")
        store = ArtifactStore(path)
        yield store
        store.close()
        try:
            os.unlink(path)
        except PermissionError:
            pass

    def test_blackboard_write_1000(self, bb):
        """Blackboard: write 1000 entries sequentially."""
        N = 1000
        t0 = time.perf_counter()
        for i in range(N):
            e = BlackboardEntry(task_id=f"T-{i:04d}", source_role="Worker",
                                content_type="task_output", summary=f"entry {i}")
            bb.write(e)
        elapsed = time.perf_counter() - t0
        assert bb.count() == N
        print(f"\n  Blackboard write x{N}: {elapsed:.4f}s ({N/elapsed:.0f} writes/s)")
        assert elapsed < 15.0

    def test_blackboard_query_1000(self, bb):
        """Blackboard: query from 1000 entries."""
        N = 1000
        for i in range(N):
            e = BlackboardEntry(task_id=f"T-{i % 10:04d}", source_role="Worker",
                                content_type="task_output", summary=f"entry {i}")
            bb.write(e)

        t0 = time.perf_counter()
        for _ in range(100):
            _ = bb.query(task_id="T-0003")
        elapsed = time.perf_counter() - t0
        print(f"\n  Blackboard query x100: {elapsed:.4f}s ({100/elapsed:.0f} queries/s)")

    def test_decision_log_write_1000(self, dl):
        """DecisionLog: write 1000 entries sequentially."""
        N = 1000
        t0 = time.perf_counter()
        for i in range(N):
            dl.log(f"T-{i:04d}", "benchmark", f"decision {i}")
        elapsed = time.perf_counter() - t0
        assert dl.count() == N
        print(f"\n  DecisionLog write x{N}: {elapsed:.4f}s ({N/elapsed:.0f} writes/s)")
        assert elapsed < 15.0

    def test_decision_log_query(self, dl):
        """DecisionLog: query by task_id from 1000 entries."""
        N = 1000
        for i in range(N):
            dl.log(f"T-{i % 50:04d}", "benchmark", f"decision {i}")

        t0 = time.perf_counter()
        for _ in range(100):
            _ = dl.query(task_id="T-0003")
        elapsed = time.perf_counter() - t0
        print(f"\n  DecisionLog query x100: {elapsed:.4f}s ({100/elapsed:.0f} queries/s)")

    def test_artifact_store_write_1000(self, art):
        """ArtifactStore: save 1000 versions sequentially."""
        N = 1000
        t0 = time.perf_counter()
        for i in range(N):
            art.save("ART-BENCH", f"content {i}", {"ver": i})
        elapsed = time.perf_counter() - t0
        assert art.count() == N
        print(f"\n  ArtifactStore write x{N}: {elapsed:.4f}s ({N/elapsed:.0f} writes/s)")
        assert elapsed < 15.0

    def test_artifact_store_version_chain(self, art):
        """ArtifactStore: verify version chain integrity on large chain."""
        N = 1000
        for i in range(N):
            art.save("ART-BENCH", f"content {i}", {"ver": i})
        t0 = time.perf_counter()
        issues = art.verify_chain("ART-BENCH")
        elapsed = time.perf_counter() - t0
        assert issues == []  # Chain must be intact
        assert art.chain_is_valid("ART-BENCH")
        print(f"\n  Version chain verify x{N}: {elapsed:.4f}s")

    def test_artifact_store_version_lookup(self, art):
        """ArtifactStore: random version lookups."""
        N = 1000
        for i in range(N):
            art.save("ART-BENCH", f"content {i}", {"ver": i})
        t0 = time.perf_counter()
        for v in [1, 500, 999, 100, 750]:
            av = art.get_version("ART-BENCH", v)
            assert av is not None
            assert av.version == v
        elapsed = time.perf_counter() - t0
        print(f"\n  Version lookup x5: {elapsed:.6f}s")


# ── Schema Validation Benchmark ─────────────────────────────────────────────

class TestSchemaValidationBenchmark:
    def test_task_contract_validate(self):
        """TaskContract.validate() throughput."""
        contract = TaskContract(
            task_id="T-bench",
            task_name="性能测试",
            role="Worker",
            objective="这是一个性能测试任务，用于测量 validate() 的吞吐量",
            task_type="general",
            timeout_seconds=120,
        )
        N = 10_000
        t0 = time.perf_counter()
        for _ in range(N):
            errs = contract.validate()
        elapsed = time.perf_counter() - t0
        assert len(contract.validate()) == 0
        print(f"\n  TaskContract.validate x{N}: {elapsed:.4f}s ({N/elapsed:.0f} calls/s)")

    def test_task_contract_validate_invalid(self):
        """TaskContract.validate() on invalid contracts."""
        contract = TaskContract()
        N = 10_000
        t0 = time.perf_counter()
        for _ in range(N):
            errs = contract.validate()
        elapsed = time.perf_counter() - t0
        assert len(contract.validate()) > 0
        print(f"\n  TaskContract.validate (invalid) x{N}: {elapsed:.4f}s ({N/elapsed:.0f} calls/s)")

    def test_agent_response_validate(self):
        """AgentResponse.validate() throughput."""
        response = AgentResponse(
            task_id="T-bench",
            role="Worker",
            status="completed",
            summary="测试摘要",
            content="x" * 1000,
            claims=[Claim(claim="论断1", confidence=0.9, evidence_refs=["ref1"])],
            risks=[Risk(risk="风险1", severity="low")],
        )
        N = 10_000
        t0 = time.perf_counter()
        for _ in range(N):
            errs = response.validate()
        elapsed = time.perf_counter() - t0
        assert len(response.validate()) == 0
        print(f"\n  AgentResponse.validate x{N}: {elapsed:.4f}s ({N/elapsed:.0f} calls/s)")


# ── DAG Schedule Benchmark ──────────────────────────────────────────────────

class TestDAGBenchmark:
    def test_dag_graph_build_large(self):
        """DAGGraph: build a large chain DAG."""
        dag = DAGGraph()
        N = 100
        t0 = time.perf_counter()
        for i in range(N):
            dag.add_node(f"task_{i}", label=f"Task {i}")
            if i > 0:
                dag.add_edge(f"task_{i-1}", f"task_{i}", label="depends")
        elapsed = time.perf_counter() - t0
        assert dag.node_count == N
        assert dag.edge_count == N - 1
        print(f"\n  DAG chain build x{N}: {elapsed:.4f}s")

    def test_dag_fan_out(self):
        """DAGGraph: wide fan-out (1 root → N leaves)."""
        dag = DAGGraph()
        N = 500
        t0 = time.perf_counter()
        dag.add_node("root", label="Root")
        for i in range(N):
            dag.add_node(f"leaf_{i}", label=f"Leaf {i}")
            dag.add_edge("root", f"leaf_{i}", label="depends")
        elapsed = time.perf_counter() - t0
        assert dag.node_count == N + 1
        assert dag.edge_count == N
        print(f"\n  DAG fan-out build x{N+1}: {elapsed:.4f}s")
