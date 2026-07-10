"""Tool Layer unit tests."""

import pytest
from dr_mma.engine.tools import (
    ToolRegistry,
    ToolDefinition,
    ToolResult,
    ToolCategory,
    ToolSafetyLevel,
    SecurityError,
    builtin_code_execute,
    builtin_file_parse,
    builtin_web_search,
    builtin_database_query,
    create_default_tool_registry,
    enable_code_execute,
)


class TestToolRegistration:
    def test_register_and_get(self):
        reg = ToolRegistry()
        d = reg.register("test_fn", lambda x: "ok")
        assert isinstance(d, ToolDefinition)
        assert reg.get_definition("test_fn") is not None

    def test_unregister(self):
        reg = ToolRegistry()
        reg.register("x", lambda: 1)
        assert reg.unregister("x") is True
        assert reg.tool_exists("x") is False

    def test_unregister_nonexistent(self):
        reg = ToolRegistry()
        assert reg.unregister("nope") is False

    def test_tool_count(self):
        reg = ToolRegistry()
        reg.register("a", lambda: 1)
        reg.register("b", lambda: 2)
        assert reg.tool_count == 2


class TestToolQuery:
    def _setup(self):
        reg = ToolRegistry()
        reg.register("safe_fn", lambda: 1, safety_level=ToolSafetyLevel.SAFE)
        reg.register("risky_fn", lambda: 2, safety_level=ToolSafetyLevel.RISKY)
        reg.register("code_fn", lambda: 3, category=ToolCategory.CODE_EXEC)
        return reg

    def test_list_all(self):
        reg = self._setup()
        assert len(reg.list_tools()) == 3

    def test_filter_by_category(self):
        reg = self._setup()
        tools = reg.list_tools(category=ToolCategory.CODE_EXEC)
        assert len(tools) == 1
        assert tools[0].name == "code_fn"

    def test_filter_by_safety(self):
        reg = self._setup()
        tools = reg.list_tools(safety_level=ToolSafetyLevel.SAFE)
        assert len(tools) == 2  # safe_fn and code_fn both default to SAFE

    def test_filter_by_role(self):
        reg = ToolRegistry()
        reg.register("admin_fn", lambda: 1, allowed_roles=["Supervisor"])
        reg.register("public_fn", lambda: 2)
        tools = reg.list_tools(role="Supervisor")
        assert len(tools) == 2
        tools = reg.list_tools(role="Worker")
        assert len(tools) == 1


class TestToolCall:
    def test_successful_call(self):
        reg = ToolRegistry()
        reg.register("add", lambda args: args["a"] + args["b"])
        r = reg.call("add", {"a": 1, "b": 2})
        assert r.success is True
        assert r.output == 3

    def test_call_unregistered_tool(self):
        reg = ToolRegistry()
        r = reg.call("nope")
        assert r.success is False
        assert "not registered" in r.error

    def test_call_with_exception(self):
        reg = ToolRegistry()
        reg.register("fail_fn", lambda args: 1 / 0)
        r = reg.call("fail_fn")
        assert r.success is False
        assert "division by zero" in r.error

    def test_call_role_restricted(self):
        reg = ToolRegistry()
        reg.register("admin_fn", lambda: 1, allowed_roles=["Supervisor"])
        r = reg.call("admin_fn", role="Worker")
        assert r.success is False
        assert "not allowed" in r.error

    def test_call_role_allowed(self):
        reg = ToolRegistry()
        reg.register("admin_fn", lambda args: 1, allowed_roles=["Supervisor"])
        r = reg.call("admin_fn", role="Supervisor")
        assert r.success is True

    def test_call_records_history(self):
        reg = ToolRegistry()
        reg.register("x", lambda: 1)
        reg.call("x", task_id="T1")
        history = reg.get_call_history(task_id="T1")
        assert len(history) == 1

    def test_to_dict(self):
        reg = ToolRegistry()
        reg.register("x", lambda args: 1)
        r = reg.call("x")
        d = r.to_dict()
        assert d["tool_name"] == "x"
        assert d["success"] is True


class TestBatchCall:
    def test_batch_calls(self):
        reg = ToolRegistry()
        reg.register("a", lambda args: 1)
        reg.register("b", lambda args: 2)
        results = reg.call_batch([{"name": "a"}, {"name": "b"}])
        assert len(results) == 2
        assert all(r.success for r in results)


class TestUsageSummary:
    def test_summary(self):
        reg = ToolRegistry()
        reg.register("code_fn", lambda args: 1, category=ToolCategory.CODE_EXEC)
        reg.register("web_fn", lambda args: 2, category=ToolCategory.WEB_SEARCH)
        reg.call("code_fn", task_id="T1")
        reg.call("web_fn", task_id="T1")
        s = reg.usage_summary(task_id="T1")
        assert s["total_calls"] == 2
        assert s["success"] == 2


class TestBuiltinCodeExecute:
    def test_disabled_by_default(self):
        """code_execute raises SecurityError when called."""
        with pytest.raises(SecurityError, match="disabled"):
            builtin_code_execute({"code": "x = 1 + 2"})

    def test_error_message_informative(self):
        """SecurityError message explains why it's disabled."""
        with pytest.raises(SecurityError) as exc_info:
            builtin_code_execute({"code": "print(1)"})
        assert "sandbox" in str(exc_info.value).lower() or "security" in str(exc_info.value).lower()


class TestBuiltinFileParse:
    def test_parse_existing_file(self):
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content\nline 2")
            path = f.name
        try:
            r = builtin_file_parse({"path": path})
            assert r["lines"] == 2
            assert "test content" in r["content_preview"]
        finally:
            os.unlink(path)

    def test_missing_path_raises(self):
        with pytest.raises(ValueError, match="required"):
            builtin_file_parse({})

    def test_nonexistent_file_raises(self):
        with pytest.raises(ValueError, match="Failed to read"):
            builtin_file_parse({"path": "/no/such/file"})


class TestBuiltinWebSearch:
    def test_search_returns_results(self):
        r = builtin_web_search({"query": "test query"})
        assert len(r["results"]) > 0
        assert "test query" in r["query"]

    def test_empty_query_raises(self):
        with pytest.raises(ValueError):
            builtin_web_search({})


class TestBuiltinDatabaseQuery:
    def test_select_allowed(self):
        r = builtin_database_query({"sql": "SELECT * FROM users"})
        assert r["rows_returned"] == 0

    def test_non_select_raises(self):
        with pytest.raises(ValueError, match="Only SELECT"):
            builtin_database_query({"sql": "DELETE FROM users"})

    def test_empty_sql_raises(self):
        with pytest.raises(ValueError):
            builtin_database_query({})


class TestDefaultRegistry:
    def test_default_tools_registered(self):
        reg = create_default_tool_registry()
        assert not reg.tool_exists("code_execute")
        assert reg.tool_exists("file_parse")
        assert reg.tool_exists("web_search")
        assert reg.tool_exists("database_query")

    def test_code_execute_not_in_default(self):
        """code_execute is disabled by default for security."""
        reg = create_default_tool_registry()
        assert not reg.tool_exists("code_execute")

    def test_code_execute_can_be_enabled(self):
        """code_execute can be explicitly enabled via enable_code_execute()."""
        reg = create_default_tool_registry()
        enable_code_execute(reg)
        assert reg.tool_exists("code_execute")
        # But it still raises SecurityError when called
        r = reg.call("code_execute", {"code": "1+1"}, role="Executor")
        assert r.success is False
        assert "disabled" in r.error.lower() or "security" in r.error.lower()
