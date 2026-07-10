"""
ConfigHierarchy -- three-level configuration merging for DR-MMA.

Priority (high to low):  agent > project > global

Features:
  * JSON-based config files
  * Shallow dict merge with list appending
  * Dot-path key access
  * Env var substitution via ${VAR} syntax
  * Source tracking per value
  * Schema validation
  * Zero external dependencies
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ValueSource:
    """Track where a configuration value originated."""

    level: str          # global | project | agent | override
    path: str = ""      # file path the value was loaded from (empty for override)

    def __repr__(self) -> str:
        return f"ValueSource({self.level}, {self.path})"


@dataclass
class ValidationError:
    """Single validation error."""

    key: str
    message: str

    def __repr__(self) -> str:
        return f"ValidationError({self.key!r}: {self.message})"


# ---------------------------------------------------------------------------
# Environment variable substitution
# ---------------------------------------------------------------------------

_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _substitute_env_vars(value: Any) -> Any:
    """Recursively replace ${VAR} patterns with os.environ values."""

    if isinstance(value, str):
        def _replacer(m: re.Match) -> str:
            var_name = m.group(1)
            return os.environ.get(var_name, m.group(0))

        return _ENV_VAR_RE.sub(_replacer, value)

    if isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]

    if isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}

    return value


# ---------------------------------------------------------------------------
# Deep helpers
# ---------------------------------------------------------------------------

def _deep_get(d: dict, dot_path: str, default: Any = None) -> Any:
    """Retrieve a nested value using dot notation. Missing keys return default."""

    keys = dot_path.split(".")
    current: Any = d
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return default
    return current


def _deep_set(d: dict, dot_path: str, value: Any) -> None:
    """Set a nested value using dot notation, creating intermediate dicts."""

    keys = dot_path.split(".")
    current = d
    for k in keys[:-1]:
        if k not in current or not isinstance(current[k], dict):
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def _merge_two(base: dict, override: dict) -> dict:
    """
    Shallow merge two dicts. Rules:

    * If both values are dicts  --> recursive merge
    * If both values are lists  --> concatenate (base + override)
    * Otherwise                 --> override wins
    """

    result = copy.deepcopy(base)

    for key, o_val in override.items():
        if key in result:
            b_val = result[key]
            if isinstance(b_val, dict) and isinstance(o_val, dict):
                result[key] = _merge_two(b_val, o_val)
            elif isinstance(b_val, list) and isinstance(o_val, list):
                result[key] = b_val + o_val
            else:
                result[key] = copy.deepcopy(o_val)
        else:
            result[key] = copy.deepcopy(o_val)

    return result


def _is_leaf_source(d: Any) -> bool:
    """Check if d is a leaf source entry like {"__source__": ValueSource(...)}. """
    return isinstance(d, dict) and set(d.keys()) == {"__source__"}


def _merge_sources(base_sources: dict, override_sources: dict, level: str, path: str) -> dict:
    """Merge source-tracking dicts, higher-priority level overwrites."""

    result = copy.deepcopy(base_sources)

    for key, o_val in override_sources.items():
        if _is_leaf_source(o_val):
            # Leaf source entry always replaces at this level
            result[key] = {"__source__": ValueSource(level, path)}
        elif isinstance(o_val, dict):
            if key in result and isinstance(result[key], dict) and not _is_leaf_source(result[key]):
                result[key] = _merge_sources(result[key], o_val, level, path)
            else:
                result[key] = copy.deepcopy(o_val)
        else:
            result[key] = {"__source__": ValueSource(level, path)}

    return result


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _validate_value(key: str, value: Any, expected_type: type | tuple[type, ...]) -> list[ValidationError]:
    """Return a list of ValidationError if value does not match expected_type."""

    if isinstance(expected_type, tuple):
        if not isinstance(value, expected_type):
            return [ValidationError(key, f"expected one of {expected_type}, got {type(value).__name__}")]
    else:
        if not isinstance(value, expected_type):
            return [ValidationError(key, f"expected {expected_type.__name__}, got {type(value).__name__}")]
    return []


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ConfigHierarchy:
    """
    Three-level configuration hierarchy with merge, validation, and source tracking.

    Levels (lowest to highest priority):
      1. global   -- user-wide defaults
      2. project  -- workspace-specific
      3. agent    -- per-agent overrides

    After loading one or more levels, call merge() to produce the final
    configuration dict and source map.
    """

    def __init__(self) -> None:
        self._global: dict = {}
        self._project: dict = {}
        self._agent: dict = {}
        self._overrides: dict = {}
        self._merged: Optional[dict] = None
        self._sources: dict = {}
        self._global_path: str = ""
        self._project_path: str = ""
        self._agent_name: str = ""
        self._agent_path: str = ""

    # -- Loading --

    def load_global(self, path: str | Path) -> ConfigHierarchy:
        """Load the global-level config file. Returns self for chaining."""

        resolved = Path(path).expanduser()
        if not resolved.is_file():
            return self

        data = self._read_json(resolved)
        self._global = _substitute_env_vars(data)
        self._global_path = str(resolved)
        self._merged = None  # invalidate cache
        return self

    def load_project(self, path: str | Path) -> ConfigHierarchy:
        """Load the project-level config file. Returns self for chaining."""

        resolved = Path(path)
        if not resolved.is_file():
            return self

        data = self._read_json(resolved)
        self._project = _substitute_env_vars(data)
        self._project_path = str(resolved)
        self._merged = None
        return self

    def load_agent(self, agent_name: str, path: str | Path) -> ConfigHierarchy:
        """Load an agent-level config file. Returns self for chaining."""

        resolved = Path(path)
        if not resolved.is_file():
            return self

        data = self._read_json(resolved)
        self._agent = _substitute_env_vars(data)
        self._agent_name = agent_name
        self._agent_path = str(resolved)
        self._merged = None
        return self

    # -- Merge --

    def merge(self) -> dict:
        """
        Merge all loaded levels into a single configuration dictionary.

        Priority (highest wins):  overrides > agent > project > global

        Also builds the source-tracking map.
        """

        result: dict = {}
        sources: dict = {}

        # global (lowest)
        if self._global:
            result = _merge_two(result, self._global)
            sources = _merge_sources(
                sources,
                self._build_source_map(self._global, "global", self._global_path),
                "global",
                self._global_path,
            )

        # project
        if self._project:
            result = _merge_two(result, self._project)
            sources = _merge_sources(
                sources,
                self._build_source_map(self._project, "project", self._project_path),
                "project",
                self._project_path,
            )

        # agent
        if self._agent:
            result = _merge_two(result, self._agent)
            sources = _merge_sources(
                sources,
                self._build_source_map(self._agent, "agent", self._agent_path),
                "agent",
                self._agent_path,
            )

        # runtime overrides (highest)
        if self._overrides:
            result = _merge_two(result, self._overrides)
            sources = _merge_sources(
                sources,
                self._build_source_map(self._overrides, "override", ""),
                "override",
                "",
            )

        self._merged = result
        self._sources = sources
        return result

    # -- Access --

    def get(self, dot_path: str, default: Any = None) -> Any:
        """
        Retrieve a value using dot notation. If not yet merged, auto-merge.

        Examples:
            hier.get("provider.baseURL")   --> nested access
            hier.get("timeout", 120)       --> with default
        """

        if self._merged is None:
            self.merge()
        return _deep_get(self._merged, dot_path, default)

    def override(self, dot_path: str, value: Any) -> ConfigHierarchy:
        """Set a runtime override (highest priority). Returns self for chaining."""

        if self._merged is not None:
            _deep_set(self._merged, dot_path, value)

        _deep_set(self._overrides, dot_path, value)
        self._merged = None  # force re-merge next time
        return self

    def get_source(self, dot_path: str) -> Optional[ValueSource]:
        """Return the source tracking info for a given key path."""

        if self._sources is None or not self._sources:
            self.merge()

        src_entry = _deep_get(self._sources, dot_path)
        if isinstance(src_entry, dict) and "__source__" in src_entry:
            return src_entry["__source__"]
        return None

    # -- Validation --

    def validate(self, schema: dict) -> list[ValidationError]:
        """
        Validate the merged config against a schema.

        Schema format:

            {
                "required": ["provider", "timeout_seconds"],
                "types": {
                    "timeout_seconds": int,
                    "provider.baseURL": str,
                    "allowed_tools": list,
                },
            }

        Returns a list of ValidationError (empty means valid).
        """

        if self._merged is None:
            self.merge()

        errors: list[ValidationError] = []

        # Check required keys
        for key in schema.get("required", []):
            val = _deep_get(self._merged, key)
            if val is None:
                errors.append(ValidationError(key, "required key is missing"))

        # Check types
        for key, expected_type in schema.get("types", {}).items():
            val = _deep_get(self._merged, key)
            if val is None:
                continue  # already caught by required check (or optional)
            errors.extend(_validate_value(key, val, expected_type))

        return errors

    # -- Export --

    def to_dict(self) -> dict:
        """Return the full merged configuration as a plain dict."""

        if self._merged is None:
            self.merge()
        return copy.deepcopy(self._merged)

    def export(self, path: str | Path) -> None:
        """Write the merged configuration to a JSON file."""

        data = self.to_dict()
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def export_sources(self, path: str | Path) -> None:
        """Write the source-tracking map to a JSON file (for debugging)."""

        if self._sources is None or not self._sources:
            self.merge()

        serializable = self._serialize_sources(self._sources)
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")

    # -- Reset --

    def reset(self) -> None:
        """Clear all loaded configs and merged state."""

        self._global = {}
        self._project = {}
        self._agent = {}
        self._overrides = {}
        self._merged = None
        self._sources = {}
        self._global_path = ""
        self._project_path = ""
        self._agent_name = ""
        self._agent_path = ""

    # -- Internal helpers --

    @staticmethod
    def _read_json(path: Path) -> dict:
        text = path.read_text(encoding="utf-8")
        return json.loads(text) if text.strip() else {}

    def _build_source_map(self, data: Any, level: str, file_path: str) -> dict:
        """Build a nested source-tracking dict mirroring data structure."""

        source = ValueSource(level, file_path)

        def _walk(d: Any) -> Any:
            if isinstance(d, dict):
                return {k: _walk(v) for k, v in d.items()}
            return {"__source__": copy.copy(source)}

        return _walk(data)

    @staticmethod
    def _serialize_sources(sources: dict) -> dict:
        """Convert ValueSource objects to plain dicts for JSON serialization."""

        def _walk(d: Any) -> Any:
            if isinstance(d, dict):
                if "__source__" in d and isinstance(d["__source__"], ValueSource):
                    s = d["__source__"]
                    return {"__source__": {"level": s.level, "path": s.path}}
                return {k: _walk(r) for k, r in d.items()}
            return d

        return _walk(sources)
