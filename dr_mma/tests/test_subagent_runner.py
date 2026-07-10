"""SubAgentRunner unit tests - spawn, run, collect, parallel, DAG."""

import time
import pytest
from dr_mma.engine.subagent_runner import (
    SubAgentRunner,
    SubAgentHandle,
    SubAgentResult,
    SubAgentStatus,
)


# ================================================================
# Helper fixtures
# ================================================================

@pytest.fixture
def runner():
    r = SubAgentRunner(max_workers=4, default_timeout=10.0)
    yield r
    r.shutdown(wait=True)


# ================================================================
# 1. SubAgentHandle tests
# ================================================================

class TestSubAgentHandle:
    def test_handle_initial_status_is_pending(self):
        h = SubAgentHandle(agent_id="A1")
        assert h.status == SubAgentStatus.PENDING
        assert h.result is None
        assert h.error is None
        assert h.started_at is None
        assert h.completed_at is None

    def test_is_done_returns_false_when_pending(self):
        h = SubAgentHandle(agent_id="A1")
        assert not h.is_done()

    def test_is_done_returns_true_when_completed(self):
        h = SubAgentHandle(agent_id="A1", status=SubAgentStatus.COMPLETED)
        assert h.is_done()

    def test_is_done_returns_true_when_failed(self):
        h = SubAgentHandle(agent_id="A1", status=SubAgentStatus.FAILED)
        assert h.is_done()

    def test_is_running_returns_true(self):
        h = SubAgentHandle(agent_id="A1", status=SubAgentStatus.RUNNING)
        assert h.is_running()

    def test_is_running_returns_false_when_pending(self):
        h = SubAgentHandle(agent_id="A1")
        assert not h.is_running()

    def test_to_dict_contains_all_fields(self):
        h = SubAgentHandle(agent_id="A1", status=SubAgentStatus.COMPLETED)
        d = h.to_dict()
        assert d["agent_id"] == "A1"
        assert d["status"] == "completed"
        assert "result" in d
        assert "error" in d


# ================================================================
# 2. SubAgentResult tests
# ================================================================

class TestSubAgentResult:
    def test_result_creation(self):
        r = SubAgentResult(agent_id="A1", success=True, content="hello")
        assert r.agent_id == "A1"
        assert r.success is True
        assert r.content == "hello"
        assert r.latency_ms == 0.0
        assert r.model_used == ""

    def test_result_to_dict(self):
        r = SubAgentResult(agent_id="A1", success=True, content="data", model_used="gpt-4")
        d = r.to_dict()
        assert d["agent_id"] == "A1"
        assert d["success"] is True
        assert d["content"] == "data"
        assert d["model_used"] == "gpt-4"
        assert "metadata" in d

    def test_result_metadata_default_empty(self):
        r = SubAgentResult(agent_id="A1", success=True)
        assert r.metadata == {}

    def test_result_latency_rounding(self):
        r = SubAgentResult(agent_id="A1", success=True, latency_ms=123.456789)
        d = r.to_dict()
        assert d["latency_ms"] == 123.46


# ================================================================
# 3. Spawn tests
# ================================================================

class TestSpawn:
    def test_spawn_returns_handle(self, runner):
        h = runner.spawn("A1", "task1", model="gpt-4")
        assert isinstance(h, SubAgentHandle)
        assert h.agent_id == "A1"
        assert h.status == SubAgentStatus.PENDING

    def test_spawn_sets_pending_status(self, runner):
        h = runner.spawn("A2", "task2")
        assert not h.is_done()
        assert not h.is_running()

    def test_spawn_deduplicates_agent_id(self, runner):
        h1 = runner.spawn("dup", "t1")
        h2 = runner.spawn("dup", "t2")
        assert h1.agent_id == "dup"
        assert h2.agent_id == "dup_1"
        assert h1.agent_id != h2.agent_id

    def test_spawn_tracks_in_runner(self, runner):
        h = runner.spawn("tracked", "task")
        found = runner.get_handle("tracked")
        assert found is h

    def test_spawn_with_priority(self, runner):
        h = runner.spawn("pri", "task", priority=5)
        assert h.agent_id == "pri"

    def test_spawn_after_shutdown_raises(self):
        r = SubAgentRunner()
        r.shutdown()
        with pytest.raises(RuntimeError, match="closed"):
            r.spawn("x", "t")

    def test_list_handles(self, runner):
        runner.spawn("L1", "t1")
        runner.spawn("L2", "t2")
        handles = runner.list_handles()
        assert len(handles) == 2


# ================================================================
# 4. Run tests (synchronous execution)
# ================================================================

class TestRun:
    def test_run_returns_result(self, runner):
        h = runner.spawn("R1", "do something")
        result = runner.run(h, task="do something", model="gpt-4")
        assert isinstance(result, SubAgentResult)
        assert result.success is True
        assert result.agent_id == "R1"

    def test_run_updates_handle_status(self, runner):
        h = runner.spawn("R2", "task")
        runner.run(h, task="task", model="m1")
        assert h.is_done()
        assert h.status == SubAgentStatus.COMPLETED
        assert h.started_at is not None
        assert h.completed_at is not None

    def test_run_records_latency(self, runner):
        h = runner.spawn("R3", "task")
        result = runner.run(h, task="task", model="m1")
        assert result.latency_ms > 0

    def test_run_records_model_used(self, runner):
        h = runner.spawn("R4", "task")
        result = runner.run(h, task="task", model="my-model")
        assert result.model_used == "my-model"

    def test_run_after_shutdown_raises(self):
        r = SubAgentRunner(default_timeout=10.0)
        h = r.spawn("X", "t")
        r.shutdown()
        with pytest.raises(RuntimeError, match="closed"):
            r.run(h, task="t", model="m")

    def test_run_cannot_rerun_completed(self, runner):
        h = runner.spawn("R5", "task")
        runner.run(h, task="task", model="m1")
        with pytest.raises(RuntimeError, match="Cannot execute"):
            runner.run(h, task="task2", model="m1")


# ================================================================
# 5. Collect tests
# ================================================================

class TestCollect:
    def test_collect_returns_result(self, runner):
        h = runner.spawn("C1", "task")
        runner.run(h, task="task", model="m1")
        result = runner.collect(h)
        assert result.success is True
        assert result.agent_id == "C1"

    def test_collect_raises_on_pending(self, runner):
        h = runner.spawn("C2", "task")
        with pytest.raises(RuntimeError, match="not done"):
            runner.collect(h)

    def test_collect_after_failure_returns_failed_result(self, runner):
        def fail_callback(agent_id, task, model, ctx):
            raise RuntimeError("boom")
        runner.set_execute_callback(fail_callback)
        h = runner.spawn("C3", "task")
        result = runner.run(h, task="task", model="m1")
        assert result.success is False
        collected = runner.collect(h)
        assert collected.success is False

    def test_collect_result_matches_run_result(self, runner):
        h = runner.spawn("C4", "task")
        run_result = runner.run(h, task="task", model="m1")
        collect_result = runner.collect(h)
        assert collect_result.agent_id == run_result.agent_id
        assert collect_result.success == run_result.success


# ================================================================
# 6. Parallel execute tests
# ================================================================

class TestParallelExecute:
    def test_parallel_returns_correct_count(self, runner):
        h1 = runner.spawn("P1", "t1")
        h2 = runner.spawn("P2", "t2")
        h3 = runner.spawn("P3", "t3")
        results = runner.parallel_execute([h1, h2, h3])
        assert len(results) == 3

    def test_parallel_all_succeed(self, runner):
        handles = [runner.spawn(f"P{i}", f"task{i}") for i in range(5)]
        results = runner.parallel_execute(handles)
        assert all(r.success for r in results)

    def test_parallel_preserves_order(self, runner):
        handles = [runner.spawn(f"PO{i}", f"t{i}") for i in range(3)]
        results = runner.parallel_execute(handles)
        for i, r in enumerate(results):
            assert r.agent_id == f"PO{i}"

    def test_parallel_empty_list(self, runner):
        results = runner.parallel_execute([])
        assert results == []

    def test_parallel_with_custom_tasks_and_models(self, runner):
        h1 = runner.spawn("PT1", "")
        h2 = runner.spawn("PT2", "")
        results = runner.parallel_execute(
            [h1, h2],
            tasks=["task A", "task B"],
            models=["model-x", "model-y"],
        )
        assert results[0].model_used == "model-x"
        assert results[1].model_used == "model-y"

    def test_parallel_error_isolation(self, runner):
        def selective_fail(agent_id, task, model, ctx):
            if agent_id == "PF2":
                raise RuntimeError("fail")
            return SubAgentResult(agent_id=agent_id, success=True, content="ok", model_used=model)
        runner.set_execute_callback(selective_fail)
        h1 = runner.spawn("PF1", "t1")
        h2 = runner.spawn("PF2", "t2")
        h3 = runner.spawn("PF3", "t3")
        results = runner.parallel_execute([h1, h2, h3])
        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True


# ================================================================
# 7. DAG execution tests
# ================================================================

class TestDAGExecution:
    def test_dag_no_edges_all_parallel(self, runner):
        nodes = [{"id": "D1"}, {"id": "D2"}, {"id": "D3"}]
        edges = []
        results = runner.execute_dag(nodes, edges)
        assert len(results) == 3
        assert all(r.success for r in results.values())

    def test_dag_linear_chain(self, runner):
        nodes = [{"id": "L1"}, {"id": "L2"}, {"id": "L3"}]
        edges = [("L1", "L2"), ("L2", "L3")]
        results = runner.execute_dag(nodes, edges)
        assert len(results) == 3
        assert all(r.success for r in results.values())

    def test_dag_fan_out(self, runner):
        nodes = [{"id": "F0"}, {"id": "F1"}, {"id": "F2"}]
        edges = [("F0", "F1"), ("F0", "F2")]
        results = runner.execute_dag(nodes, edges)
        assert len(results) == 3
        assert all(r.success for r in results.values())

    def test_dag_fan_in(self, runner):
        nodes = [{"id": "I1"}, {"id": "I2"}, {"id": "I0"}]
        edges = [("I1", "I0"), ("I2", "I0")]
        results = runner.execute_dag(nodes, edges)
        assert len(results) == 3
        assert all(r.success for r in results.values())

    def test_dag_empty_nodes(self, runner):
        results = runner.execute_dag([], [])
        assert results == {}

    def test_dag_with_tasks_map(self, runner):
        nodes = [{"id": "DT1"}, {"id": "DT2"}]
        edges = []
        tasks_map = {"DT1": "build", "DT2": "test"}
        results = runner.execute_dag(nodes, edges, tasks_map=tasks_map)
        assert len(results) == 2

    def test_dag_two_layer_dependency(self, runner):
        nodes = [{"id": "T1"}, {"id": "T2"}, {"id": "T3"}, {"id": "T4"}]
        edges = [("T1", "T3"), ("T2", "T4")]
        results = runner.execute_dag(nodes, edges)
        assert len(results) == 4
        assert all(r.success for r in results.values())


# ================================================================
# 8. Timeout tests
# ================================================================

class TestTimeout:
    def test_run_timeout_raises(self):
        r = SubAgentRunner(max_workers=2, default_timeout=0.1)
        def slow_cb(agent_id, task, model, ctx):
            time.sleep(5)
            return SubAgentResult(agent_id=agent_id, success=True, content="done", model_used=model)
        r.set_execute_callback(slow_cb)
        h = r.spawn("TO1", "slow task")
        with pytest.raises(TimeoutError):
            r.run(h, task="slow", model="m1")
        r.shutdown()

    def test_run_timeout_sets_handle_failed(self):
        r = SubAgentRunner(max_workers=2, default_timeout=0.05)
        def slow_cb(agent_id, task, model, ctx):
            time.sleep(5)
            return SubAgentResult(agent_id=agent_id, success=True, content="done", model_used=model)
        r.set_execute_callback(slow_cb)
        h = r.spawn("TO2", "slow")
        try:
            r.run(h, task="slow", model="m1")
        except TimeoutError:
            pass
        assert h.status == SubAgentStatus.FAILED
        assert h.error is not None
        r.shutdown()

    def test_parallel_timeout_on_slow_node(self):
        r = SubAgentRunner(max_workers=4, default_timeout=0.5)
        def slow_callback(agent_id, task, model, ctx):
            if agent_id == "TS1":
                time.sleep(2)
            return SubAgentResult(agent_id=agent_id, success=True, content="ok", model_used=model)
        r.set_execute_callback(slow_callback)
        h1 = r.spawn("TS1", "slow")
        h2 = r.spawn("TS2", "fast")
        results = r.parallel_execute([h1, h2], timeout_seconds=0.3)
        assert len(results) == 2
        assert results[0].success is False
        assert results[1].success is True
        r.shutdown()


# ================================================================
# 9. Error isolation tests
# ================================================================

class TestErrorIsolation:
    def test_single_failure_does_not_affect_others(self, runner):
        def fail_one(agent_id, task, model, ctx):
            if agent_id == "EI2":
                raise RuntimeError("agent EI2 crashed")
            return SubAgentResult(agent_id=agent_id, success=True, content="ok", model_used=model)
        runner.set_execute_callback(fail_one)
        h1 = runner.spawn("EI1", "t1")
        h2 = runner.spawn("EI2", "t2")
        h3 = runner.spawn("EI3", "t3")
        results = runner.parallel_execute([h1, h2, h3])
        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True

    def test_failed_handle_has_error_message(self, runner):
        def always_fail(agent_id, task, model, ctx):
            raise ValueError("test error")
        runner.set_execute_callback(always_fail)
        h = runner.spawn("ERR1", "task")
        result = runner.run(h, task="task", model="m1")
        assert result.success is False
        assert h.error == "test error"

    def test_dag_failure_propagates_without_blocking(self, runner):
        def fail_mid(agent_id, task, model, ctx):
            if agent_id == "DF2":
                raise RuntimeError("mid failure")
            return SubAgentResult(agent_id=agent_id, success=True, content="ok", model_used=model)
        runner.set_execute_callback(fail_mid)
        nodes = [{"id": "DF1"}, {"id": "DF2"}, {"id": "DF3"}]
        edges = [("DF1", "DF3")]
        results = runner.execute_dag(nodes, edges)
        assert len(results) == 3
        assert results["DF1"].success is True
        assert results["DF2"].success is False


# ================================================================
# 10. Context manager and utility tests
# ================================================================

class TestContextManager:
    def test_context_manager_closes(self):
        with SubAgentRunner(max_workers=2) as r:
            h = r.spawn("CM1", "task")
            result = r.run(h, task="t", model="m")
            assert result.success is True
        assert r._closed is True

    def test_context_manager_on_exception(self):
        try:
            with SubAgentRunner(max_workers=2) as r:
                raise ValueError("test")
        except ValueError:
            pass


class TestUtilityProperties:
    def test_active_count_increases_during_run(self, runner):
        h = runner.spawn("AC1", "slow task")
        def slow_work(agent_id, task, model, ctx):
            time.sleep(0.5)
            return SubAgentResult(agent_id=agent_id, success=True, content="done", model_used=model)
        runner.set_execute_callback(slow_work)
        import threading
        t = threading.Thread(target=lambda: runner.run(h, task="slow", model="m"))
        t.start()
        time.sleep(0.1)
        assert runner.active_count >= 1
        t.join(timeout=5)

    def test_completed_count_after_run(self, runner):
        h = runner.spawn("CC1", "task")
        runner.run(h, task="t", model="m")
        assert runner.completed_count >= 1

    def test_get_handle_returns_none_for_unknown(self, runner):
        assert runner.get_handle("nonexistent") is None


class TestCustomCallback:
    def test_custom_callback_used(self, runner):
        def custom(agent_id, task, model, ctx):
            return SubAgentResult(
                agent_id=agent_id, success=True, content="custom output",
                model_used=model, latency_ms=42.0,
            )
        runner.set_execute_callback(custom)
        h = runner.spawn("CB1", "task")
        result = runner.run(h, task="task", model="custom-model")
        assert result.content == "custom output"
        assert result.model_used == "custom-model"

    def test_custom_callback_failure(self, runner):
        def failing(agent_id, task, model, ctx):
            raise RuntimeError("callback error")
        runner.set_execute_callback(failing)
        h = runner.spawn("CBF1", "task")
        result = runner.run(h, task="t", model="m")
        assert result.success is False
