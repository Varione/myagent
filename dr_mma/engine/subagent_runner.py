"""
SubAgentRunner - SubAgent lifecycle manager for DR-MMA.

Provides spawn -> run -> collect protocol and parallel DAG execution.
Zero external dependencies: uses only threading + concurrent.futures.
"""

from __future__ import annotations

import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class SubAgentStatus(str, Enum):
    """SubAgent lifecycle status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SubAgentHandle:
    """Lightweight handle tracking a single sub-agent lifecycle."""
    agent_id: str
    status: SubAgentStatus = SubAgentStatus.PENDING
    result: Optional["SubAgentResult"] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    _future: Optional[Future] = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def is_done(self) -> bool:
        with self._lock:
            return self.status in (SubAgentStatus.COMPLETED, SubAgentStatus.FAILED)

    def is_running(self) -> bool:
        with self._lock:
            return self.status == SubAgentStatus.RUNNING

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "result": self.result.to_dict() if self.result else None,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class SubAgentResult:
    """Structured result after sub-agent execution."""
    agent_id: str
    success: bool
    content: str = ""
    latency_ms: float = 0.0
    model_used: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "success": self.success,
            "content": self.content,
            "latency_ms": round(self.latency_ms, 2),
            "model_used": self.model_used,
            "metadata": dict(self.metadata),
        }


class SubAgentRunner:
    """
    SubAgent lifecycle manager.

    Supports spawn -> run -> collect protocol and parallel DAG execution.

    Args:
        max_workers: thread pool max concurrency, default 8.
        default_timeout: default timeout in seconds, None for unlimited.
    """

    def __init__(
        self,
        max_workers: int = 8,
        default_timeout: Optional[float] = None,
    ):
        self.max_workers = max_workers
        self.default_timeout = default_timeout
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._handles: Dict[str, SubAgentHandle] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._execute_callback: Optional[Callable[[str, str, str, dict], SubAgentResult]] = None

    def spawn(
        self,
        agent_id: str,
        task: str,
        model: str = "",
        priority: int = 0,
        timeout_seconds: Optional[float] = None,
        extra_context: Optional[dict] = None,
    ) -> SubAgentHandle:
        """Spawn a sub-agent handle without executing it."""
        if self._closed:
            raise RuntimeError("SubAgentRunner is closed, cannot spawn")
        agent_id = self._deduplicate_agent_id(agent_id)
        handle = SubAgentHandle(agent_id=agent_id)
        with self._lock:
            self._handles[agent_id] = handle
        return handle

    def run(
        self,
        handle: SubAgentHandle,
        task: Optional[str] = None,
        model: str = "",
        timeout_seconds: Optional[float] = None,
        extra_context: Optional[dict] = None,
    ) -> SubAgentResult:
        """Synchronously execute a single sub-agent, blocking until done."""
        if self._closed:
            raise RuntimeError("SubAgentRunner is closed")
        effective_timeout = timeout_seconds or self.default_timeout
        future = self._submit(handle, task or "", model, extra_context or {}, effective_timeout)
        try:
            result = future.result(timeout=effective_timeout)
            return result
        except FuturesTimeoutError:
            handle.error = f"timeout: {effective_timeout}s"
            with handle._lock:
                handle.status = SubAgentStatus.FAILED
                handle.completed_at = time.time()
            raise TimeoutError(handle.error)

    def collect(self, handle: SubAgentHandle) -> SubAgentResult:
        """Extract result from a completed handle."""
        if not handle.is_done():
            raise RuntimeError(
                f"SubAgent {handle.agent_id} not done, status: {handle.status.value}"
            )
        if handle.result is not None:
            return handle.result
        return SubAgentResult(
            agent_id=handle.agent_id,
            success=False,
            content="",
            latency_ms=0.0,
            model_used="",
            metadata={"error": handle.error or "unknown"},
        )

    def parallel_execute(
        self,
        handles: List[SubAgentHandle],
        tasks: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> List[SubAgentResult]:
        """Execute a group of independent sub-agents in parallel."""
        if not handles:
            return []
        effective_timeout = timeout_seconds or self.default_timeout
        futures_list: List[tuple[SubAgentHandle, Future]] = []
        for i, handle in enumerate(handles):
            task_text = (tasks[i] if tasks else "")
            model_name = (models[i] if models else "")
            future = self._submit(handle, task_text, model_name, {}, effective_timeout)
            futures_list.append((handle, future))
        results: List[Optional[SubAgentResult]] = [None] * len(handles)
        for idx, (handle, future) in enumerate(futures_list):
            try:
                result = future.result(timeout=effective_timeout)
                results[idx] = result
            except FuturesTimeoutError:
                handle.error = f"timeout: {effective_timeout}s"
                with handle._lock:
                    handle.status = SubAgentStatus.FAILED
                    handle.completed_at = time.time()
                results[idx] = SubAgentResult(
                    agent_id=handle.agent_id, success=False, content="",
                    latency_ms=0.0, model_used=model_name,
                    metadata={"error": handle.error},
                )
            except Exception as e:
                handle.error = str(e)
                with handle._lock:
                    handle.status = SubAgentStatus.FAILED
                    handle.completed_at = time.time()
                results[idx] = SubAgentResult(
                    agent_id=handle.agent_id, success=False, content="",
                    latency_ms=0.0, model_used=model_name,
                    metadata={"error": str(e)},
                )
        return [r for r in results if r is not None]

    def execute_dag(
        self,
        nodes: List[dict],
        edges: List[tuple[str, str]],
        tasks_map: Optional[Dict[str, str]] = None,
        models_map: Optional[Dict[str, str]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, SubAgentResult]:
        """DAG-aware execution: parallelize independent nodes, serialize dependent ones."""
        if not nodes:
            return {}
        effective_timeout = timeout_seconds or self.default_timeout
        tasks_map = tasks_map or {}
        models_map = models_map or {}
        agent_ids = [n["id"] for n in nodes]
        in_degree: Dict[str, int] = {aid: 0 for aid in agent_ids}
        dependents: Dict[str, List[str]] = {aid: [] for aid in agent_ids}
        for upstream, downstream in edges:
            if upstream in in_degree and downstream in in_degree:
                in_degree[downstream] += 1
                dependents[upstream].append(downstream)
        queue = deque([aid for aid in agent_ids if in_degree[aid] == 0])
        completed: Dict[str, SubAgentResult] = {}
        all_handles: Dict[str, SubAgentHandle] = {}
        for node in nodes:
            aid = node["id"]
            h = self.spawn(agent_id=aid, task=tasks_map.get(aid, ""), model=models_map.get(aid, ""))
            all_handles[aid] = h
        while queue:
            current_layer = list(queue)
            queue.clear()
            if not current_layer:
                break
            layer_handles = [all_handles[aid] for aid in current_layer]
            layer_tasks = [tasks_map.get(aid, "") for aid in current_layer]
            layer_models = [models_map.get(aid, "") for aid in current_layer]
            layer_results = self.parallel_execute(
                layer_handles, tasks=layer_tasks, models=layer_models, timeout_seconds=effective_timeout,
            )
            for aid, result in zip(current_layer, layer_results):
                completed[aid] = result
                for dep in dependents.get(aid, []):
                    in_degree[dep] -= 1
                    if in_degree[dep] == 0:
                        queue.append(dep)
        return completed

    def shutdown(self, wait: bool = True):
        """Shutdown thread pool and release resources."""
        self._closed = True
        self._executor.shutdown(wait=wait)

    def _deduplicate_agent_id(self, agent_id: str) -> str:
        """Ensure agent_id is unique by appending counter on collision."""
        original = agent_id
        counter = 0
        while agent_id in self._handles:
            counter += 1
            agent_id = f"{original}_{counter}"
        return agent_id

    def _submit(
        self,
        handle: SubAgentHandle,
        task: str,
        model: str,
        extra_context: dict,
        timeout: Optional[float],
    ) -> Future:
        """Submit sub-agent execution to the thread pool."""
        with handle._lock:
            if handle.status != SubAgentStatus.PENDING:
                raise RuntimeError(
                    f"Cannot execute: {handle.agent_id} is {handle.status.value}"
                )
            handle.status = SubAgentStatus.RUNNING
            handle.started_at = time.time()
        future = self._executor.submit(
            self._execute_worker, handle, task, model, extra_context, timeout,
        )
        with handle._lock:
            handle._future = future
        return future

    def _execute_worker(
        self,
        handle: SubAgentHandle,
        task: str,
        model: str,
        extra_context: dict,
        timeout: Optional[float],
    ) -> SubAgentResult:
        """Worker function executed inside a thread."""
        start_time = time.time()
        try:
            if self._execute_callback is not None:
                result = self._execute_callback(handle.agent_id, task, model, extra_context)
            else:
                result = self._default_execute(handle.agent_id, task, model, extra_context, timeout)
            elapsed_ms = (time.time() - start_time) * 1000
            final_result = SubAgentResult(
                agent_id=handle.agent_id,
                success=result.success if hasattr(result, "success") else True,
                content=getattr(result, "content", str(result)),
                latency_ms=elapsed_ms,
                model_used=model,
                metadata=extra_context,
            )
            with handle._lock:
                handle.result = final_result
                handle.status = SubAgentStatus.COMPLETED
                handle.completed_at = time.time()
            return final_result
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            error_msg = str(e)
            handle.error = error_msg
            with handle._lock:
                handle.status = SubAgentStatus.FAILED
                handle.completed_at = time.time()
            return SubAgentResult(
                agent_id=handle.agent_id, success=False, content="",
                latency_ms=elapsed_ms, model_used=model,
                metadata={"error": error_msg},
            )

    @staticmethod
    def _default_execute(
        agent_id: str, task: str, model: str, extra_context: dict, timeout: Optional[float],
    ) -> SubAgentResult:
        """Default execution logic that simulates sub-agent work."""
        delay = extra_context.get("_simulated_delay", 0.01)
        if delay > 0:
            time.sleep(delay)
        should_fail = extra_context.get("_simulate_failure", False)
        if should_fail:
            raise RuntimeError(f"Simulated failure: agent {agent_id}")
        return SubAgentResult(
            agent_id=agent_id, success=True,
            content=f"[{model}] completed: {task}",
            latency_ms=0.0, model_used=model, metadata={},
        )

    def set_execute_callback(
        self, callback: Callable[[str, str, str, dict], SubAgentResult],
    ) -> None:
        """Set a custom execution callback for production use."""
        self._execute_callback = callback

    def get_handle(self, agent_id: str) -> Optional[SubAgentHandle]:
        """Get handle by agent_id."""
        with self._lock:
            return self._handles.get(agent_id)

    def list_handles(self) -> List[SubAgentHandle]:
        """List all spawned handles."""
        with self._lock:
            return list(self._handles.values())

    @property
    def active_count(self) -> int:
        """Number of currently running sub-agents."""
        count = 0
        with self._lock:
            for h in self._handles.values():
                if h.is_running():
                    count += 1
        return count

    @property
    def completed_count(self) -> int:
        """Number of completed (success + failed) sub-agents."""
        count = 0
        with self._lock:
            for h in self._handles.values():
                if h.is_done():
                    count += 1
        return count

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)
        return False
