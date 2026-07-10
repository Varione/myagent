"""
ConfigHierarchy unit tests.

Coverage:
  - Loading at each level (global, project, agent)
  - Merge priority (agent > project > global)
  - Dict deep merge and list appending
  - Dot-path access via get()
  - Runtime override() with chaining
  - Source tracking via get_source()
  - Environment variable substitution
  - Schema validation (required + type checks)
  - Export to_dict() and export()
  - Reset clears all state
  - Edge cases (missing files, empty configs, nested paths)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from dr_mma.engine.config_hierarchy import (
    ConfigHierarchy,
    ValueSource,
    ValidationError,
    _deep_get,
    _deep_set,
    _merge_two,
    _substitute_env_vars,
)


# -----------------------------------------------------------------------
# Helper: create a temp JSON config file
# -----------------------------------------------------------------------

def _write_json(path: str | Path, data: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# =========================================================================
# Test helper functions
# =========================================================================

class TestDeepGet:
    def test_simple_key(self):
        d = {"a": 1, "b": 2}
        assert _deep_get(d, "a") == 1

    def test_nested_key(self):
        d = {"a": {"b": {"c": 42}}}
        assert _deep_get(d, "a.b.c") == 42

    def test_missing_key_returns_default(self):
        d = {"a": 1}
        assert _deep_get(d, "x.y.z", default="fallback") == "fallback"

    def test_partial_path_missing(self):
        d = {"a": {"b": 1}}
        assert _deep_get(d, "a.c.d", default=None) is None

    def test_deep_get_empty_dict(self):
        assert _deep_get({}, "anything", default=99) == 99


class TestDeepSet:
    def test_simple_key(self):
        d = {}
        _deep_set(d, "a", 1)
        assert d == {"a": 1}

    def test_nested_creates_intermediates(self):
        d = {}
        _deep_set(d, "a.b.c", 42)
        assert d == {"a": {"b": {"c": 42}}}

    def test_overwrite_existing(self):
        d = {"a": {"b": 1}}
        _deep_set(d, "a.b", 99)
        assert d["a"]["b"] == 99


# =========================================================================
# Test environment variable substitution
# =========================================================================

class TestSubstituteEnvVars:
    def test_string_with_env_var(self, monkeypatch):
        monkeypatch.setenv("MY_TEST_VAR", "hello")
        result = _substitute_env_vars("${MY_TEST_VAR}")
        assert result == "hello"

    def test_string_without_env_var_unchanged(self):
        result = _substitute_env_vars("no_vars_here")
        assert result == "no_vars_here"

    def test_unset_var_keeps_original(self):
        result = _substitute_env_vars("${NONEXISTENT_VAR_12345}")
        assert result == "${NONEXISTENT_VAR_12345}"

    def test_nested_dict_substitution(self, monkeypatch):
        monkeypatch.setenv("DB_HOST", "localhost")
        data = {"db": {"host": "${DB_HOST}", "port": 5432}}
        result = _substitute_env_vars(data)
        assert result["db"]["host"] == "localhost"
        assert result["db"]["port"] == 5432

    def test_list_substitution(self, monkeypatch):
        monkeypatch.setenv("TOOL_A", "git")
        data = ["${TOOL_A}", "pytest"]
        result = _substitute_env_vars(data)
        assert result == ["git", "pytest"]

    def test_mixed_string_partial_sub(self, monkeypatch):
        monkeypatch.setenv("PORT_NUM", "8080")
        result = _substitute_env_vars("http://localhost:${PORT_NUM}/api")
        assert result == "http://localhost:8080/api"


# =========================================================================
# Test merge logic
# =========================================================================

class TestMergeTwo:
    def test_simple_override(self):
        base = {"a": 1}
        over = {"a": 2}
        assert _merge_two(base, over) == {"a": 2}

    def test_new_key_added(self):
        base = {"a": 1}
        over = {"b": 2}
        result = _merge_two(base, over)
        assert result == {"a": 1, "b": 2}

    def test_nested_dict_recursive_merge(self):
        base = {"db": {"host": "localhost", "port": 5432}}
        over = {"db": {"host": "remote"}}
        result = _merge_two(base, over)
        assert result == {"db": {"host": "remote", "port": 5432}}

    def test_list_concatenation(self):
        base = {"tools": ["git", "pytest"]}
        over = {"tools": ["rg"]}
        result = _merge_two(base, over)
        assert result["tools"] == ["git", "pytest", "rg"]

    def test_type_mismatch_override_wins(self):
        base = {"val": [1, 2]}
        over = {"val": "string"}
        result = _merge_two(base, over)
        assert result["val"] == "string"

    def test_deep_copy_isolation(self):
        base = {"a": {"b": 1}}
        over = {}
        result = _merge_two(base, over)
        result["a"]["b"] = 999
        assert base["a"]["b"] == 1  # original untouched


# =========================================================================
# Test ConfigHierarchy loading
# =========================================================================

class TestConfigHierarchyLoading:
    def test_load_global(self, tmp_path):
        cfg = _write_json(tmp_path / "global.json", {"timeout": 300})
        hier = ConfigHierarchy()
        result = hier.load_global(cfg)
        assert result is hier  # chaining
        assert hier.get("timeout") == 300

    def test_load_project(self, tmp_path):
        cfg = _write_json(tmp_path / "project.json", {"workspace": "/proj"})
        hier = ConfigHierarchy()
        hier.load_project(cfg)
        assert hier.get("workspace") == "/proj"

    def test_load_agent(self, tmp_path):
        cfg = _write_json(tmp_path / "agent.json", {"model": "gpt-4"})
        hier = ConfigHierarchy()
        hier.load_agent("coder", cfg)
        assert hier.get("model") == "gpt-4"

    def test_load_missing_file_noop(self, tmp_path):
        hier = ConfigHierarchy()
        result = hier.load_global(tmp_path / "nonexistent.json")
        assert result is hier
        assert hier.to_dict() == {}

    def test_load_empty_json(self, tmp_path):
        cfg = _write_json(tmp_path / "empty.json", {})
        hier = ConfigHierarchy()
        hier.load_global(cfg)
        assert hier.to_dict() == {}

    def test_chained_loading(self, tmp_path):
        g = _write_json(tmp_path / "g.json", {"a": 1})
        p = _write_json(tmp_path / "p.json", {"b": 2})
        a = _write_json(tmp_path / "a.json", {"c": 3})
        hier = ConfigHierarchy()
        hier.load_global(g).load_project(p).load_agent("test", a)
        merged = hier.merge()
        assert merged == {"a": 1, "b": 2, "c": 3}


# =========================================================================
# Test merge priority
# =========================================================================

class TestMergePriority:
    def test_agent_overrides_project(self, tmp_path):
        g = _write_json(tmp_path / "g.json", {"timeout": 100})
        p = _write_json(tmp_path / "p.json", {"timeout": 200})
        a = _write_json(tmp_path / "a.json", {"timeout": 300})
        hier = ConfigHierarchy()
        hier.load_global(g).load_project(p).load_agent("x", a)
        assert hier.merge()["timeout"] == 300

    def test_project_overrides_global(self, tmp_path):
        g = _write_json(tmp_path / "g.json", {"timeout": 100})
        p = _write_json(tmp_path / "p.json", {"timeout": 200})
        hier = ConfigHierarchy()
        hier.load_global(g).load_project(p)
        assert hier.merge()["timeout"] == 200

    def test_runtime_override_wins_all(self, tmp_path):
        g = _write_json(tmp_path / "g.json", {"timeout": 100})
        a = _write_json(tmp_path / "a.json", {"timeout": 300})
        hier = ConfigHierarchy()
        hier.load_global(g).load_agent("x", a)
        hier.override("timeout", 999)
        assert hier.get("timeout") == 999

    def test_nested_merge_preserves_other_keys(self, tmp_path):
        g = _write_json(tmp_path / "g.json", {"db": {"host": "localhost", "port": 5432}})
        a = _write_json(tmp_path / "a.json", {"db": {"host": "remote"}})
        hier = ConfigHierarchy()
        hier.load_global(g).load_agent("x", a)
        merged = hier.merge()
        assert merged["db"]["host"] == "remote"
        assert merged["db"]["port"] == 5432

    def test_list_appending_across_levels(self, tmp_path):
        g = _write_json(tmp_path / "g.json", {"tools": ["git"]})
        p = _write_json(tmp_path / "p.json", {"tools": ["pytest"]})
        a = _write_json(tmp_path / "a.json", {"tools": ["rg"]})
        hier = ConfigHierarchy()
        hier.load_global(g).load_project(p).load_agent("x", a)
        merged = hier.merge()
        assert merged["tools"] == ["git", "pytest", "rg"]


# =========================================================================
# Test dot-path access
# =========================================================================

class TestDotPathAccess:
    def test_get_nested_value(self, tmp_path):
        cfg = _write_json(tmp_path / "c.json", {"provider": {"baseURL": "http://api"}})
        hier = ConfigHierarchy()
        hier.load_global(cfg)
        assert hier.get("provider.baseURL") == "http://api"

    def test_get_with_default(self, tmp_path):
        cfg = _write_json(tmp_path / "c.json", {"a": 1})
        hier = ConfigHierarchy()
        hier.load_global(cfg)
        assert hier.get("nonexistent", default="fallback") == "fallback"

    def test_get_deeply_nested(self, tmp_path):
        cfg = _write_json(tmp_path / "c.json", {"a": {"b": {"c": {"d": 42}}}})
        hier = ConfigHierarchy()
        hier.load_global(cfg)
        assert hier.get("a.b.c.d") == 42

    def test_auto_merge_on_get(self, tmp_path):
        cfg = _write_json(tmp_path / "c.json", {"key": "val"})
        hier = ConfigHierarchy()
        hier.load_global(cfg)
        # get() should auto-merge without explicit merge() call
        assert hier.get("key") == "val"


# =========================================================================
# Test override
# =========================================================================

class TestOverride:
    def test_override_simple(self, tmp_path):
        cfg = _write_json(tmp_path / "c.json", {"timeout": 100})
        hier = ConfigHierarchy()
        hier.load_global(cfg)
        hier.override("timeout", 300)
        assert hier.get("timeout") == 300

    def test_override_nested(self, tmp_path):
        cfg = _write_json(tmp_path / "c.json", {"db": {"host": "localhost"}})
        hier = ConfigHierarchy()
        hier.load_global(cfg)
        hier.override("db.host", "remote")
        assert hier.get("db.host") == "remote"

    def test_override_chaining(self, tmp_path):
        cfg = _write_json(tmp_path / "c.json", {"a": 1})
        hier = ConfigHierarchy()
        hier.load_global(cfg)
        result = hier.override("a", 2).override("b", 3)
        assert result is hier

    def test_override_survives_remerge(self, tmp_path):
        cfg = _write_json(tmp_path / "c.json", {"timeout": 100})
        hier = ConfigHierarchy()
        hier.load_global(cfg)
        hier.override("timeout", 300)
        # Force re-merge
        hier.merge()
        assert hier.get("timeout") == 300


# =========================================================================
# Test source tracking
# =========================================================================

class TestSourceTracking:
    def test_source_from_global(self, tmp_path):
        cfg = _write_json(tmp_path / "g.json", {"timeout": 100})
        hier = ConfigHierarchy()
        hier.load_global(cfg)
        hier.merge()
        src = hier.get_source("timeout")
        assert src is not None
        assert src.level == "global"

    def test_source_from_agent_overrides_global(self, tmp_path):
        g = _write_json(tmp_path / "g.json", {"timeout": 100})
        a = _write_json(tmp_path / "a.json", {"timeout": 300})
        hier = ConfigHierarchy()
        hier.load_global(g).load_agent("x", a)
        hier.merge()
        src = hier.get_source("timeout")
        assert src is not None
        assert src.level == "agent"

    def test_source_from_override(self, tmp_path):
        cfg = _write_json(tmp_path / "c.json", {"timeout": 100})
        hier = ConfigHierarchy()
        hier.load_global(cfg)
        hier.override("timeout", 300)
        hier.merge()
        src = hier.get_source("timeout")
        assert src is not None
        assert src.level == "override"

    def test_source_path_contains_file(self, tmp_path):
        cfg = _write_json(tmp_path / "my_config.json", {"key": "val"})
        hier = ConfigHierarchy()
        hier.load_global(cfg)
        hier.merge()
        src = hier.get_source("key")
        assert src is not None
        assert "my_config.json" in src.path

    def test_source_missing_key_returns_none(self, tmp_path):
        cfg = _write_json(tmp_path / "c.json", {"a": 1})
        hier = ConfigHierarchy()
        hier.load_global(cfg)
        assert hier.get_source("nonexistent") is None


# =========================================================================
# Test schema validation
# =========================================================================

class TestValidation:
    def test_valid_config(self, tmp_path):
        cfg = _write_json(tmp_path / "c.json", {"timeout": 120, "name": "test"})
        hier = ConfigHierarchy()
        hier.load_global(cfg)
        schema = {
            "required": ["timeout"],
            "types": {"timeout": int, "name": str},
        }
        errors = hier.validate(schema)
        assert errors == []

    def test_missing_required_key(self, tmp_path):
        cfg = _write_json(tmp_path / "c.json", {"name": "test"})
        hier = ConfigHierarchy()
        hier.load_global(cfg)
        schema = {"required": ["timeout"]}
        errors = hier.validate(schema)
        assert len(errors) == 1
        assert errors[0].key == "timeout"

    def test_wrong_type(self, tmp_path):
        cfg = _write_json(tmp_path / "c.json", {"timeout": "not_an_int"})
        hier = ConfigHierarchy()
        hier.load_global(cfg)
        schema = {"types": {"timeout": int}}
        errors = hier.validate(schema)
        assert len(errors) == 1
        assert isinstance(errors[0], ValidationError)

    def test_nested_required_key(self, tmp_path):
        cfg = _write_json(tmp_path / "c.json", {"provider": {}})
        hier = ConfigHierarchy()
        hier.load_global(cfg)
        schema = {"required": ["provider.baseURL"]}
        errors = hier.validate(schema)
        assert len(errors) == 1

    def test_multiple_errors(self, tmp_path):
        cfg = _write_json(tmp_path / "c.json", {"a": "wrong"})
        hier = ConfigHierarchy()
        hier.load_global(cfg)
        schema = {
            "required": ["missing_key"],
            "types": {"a": int},
        }
        errors = hier.validate(schema)
        assert len(errors) == 2

    def test_empty_schema_passes(self, tmp_path):
        cfg = _write_json(tmp_path / "c.json", {"anything": 1})
        hier = ConfigHierarchy()
        hier.load_global(cfg)
        errors = hier.validate({})
        assert errors == []


# =========================================================================
# Test export
# =========================================================================

class TestExport:
    def test_to_dict(self, tmp_path):
        cfg = _write_json(tmp_path / "c.json", {"a": 1, "b": [2]})
        hier = ConfigHierarchy()
        hier.load_global(cfg)
        d = hier.to_dict()
        assert d == {"a": 1, "b": [2]}

    def test_to_dict_is_deep_copy(self, tmp_path):
        cfg = _write_json(tmp_path / "c.json", {"a": {"b": 1}})
        hier = ConfigHierarchy()
        hier.load_global(cfg)
        d = hier.to_dict()
        d["a"]["b"] = 999
        assert hier.get("a.b") == 1

    def test_export_to_file(self, tmp_path):
        cfg = _write_json(tmp_path / "c.json", {"key": "val"})
        out = tmp_path / "output" / "exported.json"
        hier = ConfigHierarchy()
        hier.load_global(cfg)
        hier.export(out)
        assert out.is_file()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data == {"key": "val"}

    def test_export_creates_parent_dirs(self, tmp_path):
        cfg = _write_json(tmp_path / "c.json", {"a": 1})
        out = tmp_path / "deep" / "nested" / "dir" / "out.json"
        hier = ConfigHierarchy()
        hier.load_global(cfg)
        hier.export(out)
        assert out.is_file()

    def test_export_sources(self, tmp_path):
        cfg = _write_json(tmp_path / "c.json", {"key": "val"})
        out = tmp_path / "sources.json"
        hier = ConfigHierarchy()
        hier.load_global(cfg)
        hier.export_sources(out)
        assert out.is_file()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "__source__" in data["key"]


# =========================================================================
# Test reset
# =========================================================================

class TestReset:
    def test_reset_clears_everything(self, tmp_path):
        cfg = _write_json(tmp_path / "c.json", {"a": 1})
        hier = ConfigHierarchy()
        hier.load_global(cfg)
        hier.merge()
        assert hier.get("a") == 1

        hier.reset()
        assert hier.to_dict() == {}
        assert hier.get("a", default="gone") == "gone"


# =========================================================================
# Test ValueSource repr
# =========================================================================

class TestDataClasses:
    def test_value_source_repr(self):
        vs = ValueSource("global", "/path/to/file.json")
        assert "global" in repr(vs)
        assert "/path/to/file.json" in repr(vs)

    def test_validation_error_repr(self):
        ve = ValidationError("key", "missing")
        assert "key" in repr(ve)
        assert "missing" in repr(ve)


# =========================================================================
# Test edge cases
# =========================================================================

class TestEdgeCases:
    def test_merge_with_no_levels_loaded(self):
        hier = ConfigHierarchy()
        result = hier.merge()
        assert result == {}

    def test_get_on_empty_hierarchy(self):
        hier = ConfigHierarchy()
        assert hier.get("anything", default=42) == 42

    def test_env_var_in_nested_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("API_KEY_VAL", "secret123")
        cfg = _write_json(tmp_path / "c.json", {"api": {"key": "${API_KEY_VAL}"}})
        hier = ConfigHierarchy()
        hier.load_global(cfg)
        assert hier.get("api.key") == "secret123"

    def test_override_before_merge(self, tmp_path):
        cfg = _write_json(tmp_path / "c.json", {"timeout": 100})
        hier = ConfigHierarchy()
        hier.load_global(cfg)
        # Override before any merge call
        hier.override("timeout", 500)
        assert hier.get("timeout") == 500

    def test_multiple_overrides_same_key(self, tmp_path):
        cfg = _write_json(tmp_path / "c.json", {"a": 1})
        hier = ConfigHierarchy()
        hier.load_global(cfg)
        hier.override("a", 2).override("a", 3)
        assert hier.get("a") == 3
