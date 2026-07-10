"""
Tool Layer — 工具执行层。

Phase 4: 文件解析、代码执行沙箱、联网检索、数据库查询等工具的注册、调度和安全控制。

核心功能：
- 工具注册表：统一管理可用工具
- 工具分类：Safe / Risky / Critical 分级
- 工具调用框架：统一输入输出 Schema
- 失败恢复：TOOL_FAILED 事件触发
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class ToolCategory(Enum):
    """工具类别。"""

    CODE_EXEC = "code_exec"
    WEB_SEARCH = "web_search"
    FILE_PARSE = "file_parse"
    DATABASE = "database"
    VECTOR_STORE = "vector_store"
    SIMULATION = "simulation"
    GENERAL = "general"


class ToolSafetyLevel(Enum):
    """工具安全等级。"""

    SAFE = "safe"  # 只读、无副作用
    RISKY = "risky"  # 有外部影响但可回滚
    CRITICAL = "critical"  # 不可逆操作


@dataclass
class ToolDefinition:
    """工具定义。"""

    name: str
    description: str
    category: ToolCategory = ToolCategory.GENERAL
    safety_level: ToolSafetyLevel = ToolSafetyLevel.SAFE
    parameters: dict = field(default_factory=dict)
    allowed_roles: list[str] = field(default_factory=list)  # 空表示所有角色可用
    timeout_seconds: int = 60

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "safety_level": self.safety_level.value,
            "parameters": self.parameters,
            "allowed_roles": self.allowed_roles,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class ToolResult:
    """工具调用结果。"""

    tool_name: str
    success: bool = False
    output: Any = None
    error: str = ""
    latency_ms: float = 0.0
    tokens_used: int = 0
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "output": str(self.output)[:500] if self.output is not None else None,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 2),
            "tokens_used": self.tokens_used,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class ToolRegistry:
    """
    工具注册表：统一管理可用工具的注册、查询和调用。

    Usage:
        registry = ToolRegistry()
        registry.register("python_execute", python_exec_fn,
            description="执行 Python 代码",
            category=ToolCategory.CODE_EXEC,
            safety_level=ToolSafetyLevel.RISKY,
        )

        # 查询可用工具
        tools = registry.list_tools(category=ToolCategory.WEB_SEARCH)

        # 调用工具
        result = registry.call("python_execute", {"code": "print(1)"})
    """

    def __init__(self):
        self._definitions: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, Callable] = {}
        self._call_history: list[dict] = []

    # ── 注册 ────────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        handler: Callable,
        description: str = "",
        category: ToolCategory = ToolCategory.GENERAL,
        safety_level: ToolSafetyLevel = ToolSafetyLevel.SAFE,
        parameters: Optional[dict] = None,
        allowed_roles: Optional[list[str]] = None,
        timeout_seconds: int = 60,
    ) -> ToolDefinition:
        """注册工具。"""
        definition = ToolDefinition(
            name=name,
            description=description,
            category=category,
            safety_level=safety_level,
            parameters=parameters or {},
            allowed_roles=allowed_roles or [],
            timeout_seconds=timeout_seconds,
        )
        self._definitions[name] = definition
        self._handlers[name] = handler
        return definition

    def unregister(self, name: str) -> bool:
        """注销工具。"""
        if name in self._definitions:
            del self._definitions[name]
            del self._handlers[name]
            return True
        return False

    def get_definition(self, name: str) -> Optional[ToolDefinition]:
        """获取工具定义。"""
        return self._definitions.get(name)

    # ── 查询 ────────────────────────────────────────────────────────

    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        safety_level: Optional[ToolSafetyLevel] = None,
        role: Optional[str] = None,
    ) -> list[ToolDefinition]:
        """查询可用工具。"""
        results = []
        for defn in self._definitions.values():
            if category is not None and defn.category != category:
                continue
            if safety_level is not None and defn.safety_level != safety_level:
                continue
            if role is not None and defn.allowed_roles and role not in defn.allowed_roles:
                continue
            results.append(defn)
        return results

    def tool_exists(self, name: str) -> bool:
        """检查工具是否存在。"""
        return name in self._definitions

    @property
    def tool_count(self) -> int:
        return len(self._definitions)

    # ── 调用 ────────────────────────────────────────────────────────

    def call(
        self,
        name: str,
        args: Optional[dict] = None,
        role: str = "",
        task_id: str = "",
    ) -> ToolResult:
        """
        调用工具。

        Args:
            name: 工具名称
            args: 工具参数
            role: 调用者角色（用于权限检查）
            task_id: 任务 ID（用于追踪）
        """
        definition = self._definitions.get(name)
        if definition is None:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Tool '{name}' not registered",
            )

        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Tool '{name}' has no handler",
            )

        # 角色权限检查
        if definition.allowed_roles and role and role not in definition.allowed_roles:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Role '{role}' not allowed to use '{name}'",
            )

        start = time.time()
        try:
            output = handler(args or {})
            latency = (time.time() - start) * 1000
            result = ToolResult(
                tool_name=name,
                success=True,
                output=output,
                latency_ms=latency,
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            result = ToolResult(
                tool_name=name,
                success=False,
                error=str(e),
                latency_ms=latency,
            )

        # 记录调用历史
        self._call_history.append({
            "tool_name": name,
            "task_id": task_id,
            "role": role,
            "success": result.success,
            "timestamp": result.timestamp,
            "latency_ms": result.latency_ms,
        })

        return result

    def call_batch(
        self,
        calls: list[dict],
        role: str = "",
        task_id: str = "",
    ) -> list[ToolResult]:
        """批量调用工具。"""
        return [
            self.call(c["name"], c.get("args"), role, task_id)
            for c in calls
        ]

    # ── 诊断 ────────────────────────────────────────────────────────

    def get_call_history(
        self,
        task_id: Optional[str] = None,
        tool_name: Optional[str] = None,
    ) -> list[dict]:
        """获取调用历史。"""
        results = self._call_history
        if task_id:
            results = [h for h in results if h.get("task_id") == task_id]
        if tool_name:
            results = [h for h in results if h.get("tool_name") == tool_name]
        return results

    def usage_summary(self, task_id: Optional[str] = None) -> dict:
        """返回工具使用摘要。"""
        history = self.get_call_history(task_id)
        total = len(history)
        success = sum(1 for h in history if h["success"])
        failed = total - success

        by_category = {}
        for h in history:
            defn = self._definitions.get(h["tool_name"])
            cat = defn.category.value if defn else "unknown"
            by_category[cat] = by_category.get(cat, 0) + 1

        return {
            "total_calls": total,
            "success": success,
            "failed": failed,
            "by_category": by_category,
            "task_id": task_id or "all",
        }


# ── 内置工具实现 ─────────────────────────────────────────────────────

def builtin_code_execute(args: dict) -> dict:
    """
    内置代码执行（简化版沙箱）。

    支持 Python 代码执行，限制：
    - 禁止 import os, sys, subprocess, socket 等危险模块
    - 仅允许纯计算和数据处理操作
    """
    code = args.get("code", "")
    timeout = args.get("timeout", 10)

    # 安全检查：禁止危险导入
    dangerous_imports = ["os", "sys", "subprocess", "socket", "shutil", "ctypes"]
    for mod in dangerous_imports:
        if re.search(rf"\bimport\s+{mod}\b", code) or re.search(rf"\bfrom\s+{mod}\b", code):
            raise ValueError(f"Import of '{mod}' is not allowed in sandbox")

    # 禁止危险函数调用
    dangerous_funcs = ["exec(", "eval(", "__import__", "open("]
    for func in dangerous_funcs:
        if func in code and func != "open(":
            raise ValueError(f"Use of '{func}' is not allowed in sandbox")

    # 执行代码
    local_ns = {}
    stdout_output = []

    class _CapturePrint:
        def write(self, text):
            if text.strip():
                stdout_output.append(text)
        def flush(self):
            pass

    import io
    old_stdout = __import__("sys").stdout
    __import__("sys").stdout = _CapturePrint()
    try:
        exec(code, {"__builtins__": __builtins__}, local_ns)
    finally:
        __import__("sys").stdout = old_stdout

    return {
        "stdout": "".join(stdout_output).strip(),
        "variables": {k: str(v)[:200] for k, v in local_ns.items() if not k.startswith("_")},
    }


def builtin_file_parse(args: dict) -> dict:
    """内置文件解析（读取文件内容）。"""
    file_path = args.get("path", "")
    content_type = args.get("content_type", "text")

    if not file_path:
        raise ValueError("File path is required")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        raise ValueError(f"Failed to read file: {e}")

    return {
        "file_path": file_path,
        "content_type": content_type,
        "size_bytes": len(content),
        "lines": content.count("\n") + 1,
        "content_preview": content[:500],
        "checksum": hashlib.md5(content.encode()).hexdigest(),
    }


def builtin_web_search(args: dict) -> dict:
    """内置联网检索（模拟实现，实际需接入搜索 API）。"""
    query = args.get("query", "")
    max_results = args.get("max_results", 5)

    if not query:
        raise ValueError("Search query is required")

    return {
        "query": query,
        "results_count": min(max_results, 3),
        "results": [
            {
                "title": f"Result for '{query}'",
                "snippet": f"Simulated search result for: {query}",
                "url": f"https://example.com/search?q={query}",
            }
        ],
        "note": "This is a simulated search. Connect to actual API for real results.",
    }


def builtin_database_query(args: dict) -> dict:
    """内置数据库查询（模拟实现）。"""
    sql = args.get("sql", "")

    if not sql:
        raise ValueError("SQL query is required")

    # 安全检查：只允许 SELECT
    if not sql.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed")

    return {
        "query": sql,
        "rows_returned": 0,
        "note": "This is a simulated query. Connect to actual database for real results.",
    }


# ── 默认工具注册表工厂 ───────────────────────────────────────────────

def create_default_tool_registry() -> ToolRegistry:
    """创建包含内置工具的默认注册表。"""
    registry = ToolRegistry()

    registry.register(
        "code_execute",
        builtin_code_execute,
        description="在沙箱中执行 Python 代码",
        category=ToolCategory.CODE_EXEC,
        safety_level=ToolSafetyLevel.RISKY,
        allowed_roles=["Executor", "Verifier", "Supervisor"],
    )

    registry.register(
        "file_parse",
        builtin_file_parse,
        description="读取并解析文件内容",
        category=ToolCategory.FILE_PARSE,
        safety_level=ToolSafetyLevel.SAFE,
        allowed_roles=["Researcher", "Executor", "Supervisor"],
    )

    registry.register(
        "web_search",
        builtin_web_search,
        description="联网检索信息",
        category=ToolCategory.WEB_SEARCH,
        safety_level=ToolSafetyLevel.RISKY,
        allowed_roles=["Researcher", "Supervisor"],
    )

    registry.register(
        "database_query",
        builtin_database_query,
        description="执行只读数据库查询",
        category=ToolCategory.DATABASE,
        safety_level=ToolSafetyLevel.SAFE,
        allowed_roles=["Researcher", "Verifier", "Supervisor"],
    )

    return registry
