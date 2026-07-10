"""UI controller that bridges the desktop shell and workflow engine."""

from __future__ import annotations

import json
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from ..engine.streaming import StreamEventKind
from ..engine.workflow import WorkflowEngine, WorkflowResult
from ..models.adapter import LocalModel, MockModel, ModelAdapter, RemoteModel
from ..storage.artifact_store import ArtifactStore
from ..storage.blackboard import Blackboard
from ..storage.decision_log import DecisionLog
from .i18n import ZH_CN, tr


CONFIG_DIR = Path("dr_mma_data")
CONFIG_FILE = CONFIG_DIR / "models_config.json"
RUNTIME_LOG_FILE = CONFIG_DIR / "runtime.log"
DEFAULT_MODEL_NAME = "Mock Model (Test)"


class WorkflowController:
    """Owns workflow runtime state and desktop-facing callbacks."""

    def __init__(self, tk_root=None):
        self._tk_root = tk_root
        self._engine: Optional[WorkflowEngine] = None
        self._worker: Optional[threading.Thread] = None
        self._last_result: Optional[WorkflowResult] = None
        self._cancel_flag = threading.Event()
        self._model_adapter = ModelAdapter()
        self._model_infos: list[dict] = []
        self.language = ZH_CN

        self.on_stage_change: Optional[Callable[[str, str], None]] = None
        self.on_subtask_done: Optional[Callable[[int, int, dict], None]] = None
        self.on_workflow_done: Optional[Callable[[WorkflowResult], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_log: Optional[Callable[[str], None]] = None

        self.config: dict = {
            "blackboard_path": "",
            "artifact_path": "",
            "decision_path": "",
            "models": {},
            "language": ZH_CN,
            "workspace_root": "",
            "allowed_tools": None,
            "permission_mode": "workspace_only",
            "assignment_mode": "primary_preferred",
            "timeout_seconds": 120,
        }

        self.load_config()

    def t(self, key: str, **kwargs) -> str:
        return tr(self.language, key, **kwargs)

    def set_language(self, language: str):
        self.language = language
        self.config["language"] = language
        self.save_config()

    def load_config(self):
        """Load persisted model registrations and storage paths."""
        self._model_adapter = ModelAdapter()
        self._model_infos = []

        if not CONFIG_FILE.exists():
            self._register_default_mock()
            return

        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            self._register_default_mock()
            return

        self.language = data.get("language", ZH_CN)
        self.config["language"] = self.language
        self._model_infos = list(data.get("models", []))
        restored = False
        for info in self._model_infos:
            try:
                model = self._build_model_from_info(info)
            except Exception:
                continue
            self._model_adapter.register(info["name"], model)
            restored = True

        for key in ("blackboard_path", "artifact_path", "decision_path"):
            if key in data.get("paths", {}):
                self.config[key] = data["paths"][key]
        runtime = data.get("runtime", {})
        for key in ("workspace_root", "allowed_tools", "permission_mode", "assignment_mode", "timeout_seconds"):
            if key in runtime:
                self.config[key] = runtime[key]

        if not restored:
            self._register_default_mock()

        # Ensure config file exists on first launch
        if not CONFIG_FILE.exists():
            self.save_config()

    def save_config(self):
        """Persist registered models and storage paths (api_keys excluded)."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "language": self.language,
            "models": self._sanitize_model_infos(self._model_infos),
            "paths": {
                key: self.config[key]
                for key in ("blackboard_path", "artifact_path", "decision_path")
                if self.config.get(key)
            },
            "runtime": {
                "workspace_root": self.config.get("workspace_root", ""),
                "allowed_tools": self.config.get("allowed_tools"),
                "permission_mode": self.config.get("permission_mode", "workspace_only"),
                "assignment_mode": self.config.get("assignment_mode", "primary_preferred"),
                "timeout_seconds": int(self.config.get("timeout_seconds", 120) or 120),
            },
        }
        try:
            CONFIG_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def register_model(
        self,
        name: str,
        endpoint: str = "",
        api_key: str = "",
        model_name: str = "",
        model_type: str = "mock",
    ):
        """Register a model instance and persist its definition (api_key excluded from persistence)."""
        info = {"name": name, "type": model_type}
        if endpoint:
            info["endpoint"] = endpoint
        # Never persist api_key to disk; keep only in runtime memory
        if model_name:
            info["model_name"] = model_name

        model = self._build_model_from_info({**info, "api_key": api_key})
        self._model_adapter.register(name, model)
        self._model_infos = [item for item in self._model_infos if item.get("name") != name]
        self._model_infos.append(info)
        self.save_config()

    def _sanitize_model_infos(self, infos: list[dict]) -> list[dict]:
        """Remove api_key from model info dicts before persistence."""
        return [{k: v for k, v in info.items() if k != "api_key"} for info in infos]

    def remove_model(self, name: str):
        """Remove a model registration and rebuild the adapter."""
        self._model_infos = [item for item in self._model_infos if item.get("name") != name]
        self._model_adapter = ModelAdapter()
        for info in self._model_infos:
            try:
                self._model_adapter.register(info["name"], self._build_model_from_info(info))
            except Exception:
                continue
        if not self._model_adapter.available_models:
            self._register_default_mock()
        self.save_config()

    def get_available_models(self) -> list[str]:
        models = self._model_adapter.available_models
        return models or [DEFAULT_MODEL_NAME]

    def get_model_infos(self) -> list[dict]:
        """Return model infos without api_key values."""
        return self._sanitize_model_infos(self._model_infos)

    def build_engine(self, model_name: str) -> WorkflowEngine:
        """Create a workflow engine with configured storage backends."""
        self._ensure_paths()
        engine = WorkflowEngine(
            adapter=self._model_adapter,
            blackboard=Blackboard(self.config["blackboard_path"]),
            artifact_store=ArtifactStore(self.config["artifact_path"]),
            decision_log=DecisionLog(self.config["decision_path"]),
            main_model=model_name,
            runtime_config={
                "workspace_root": self.config.get("workspace_root", ""),
                "allowed_tools": list(self.config.get("allowed_tools", [])),
                "permission_mode": self.config.get("permission_mode", "workspace_only"),
                "assignment_mode": self.config.get("assignment_mode", "primary_preferred"),
                "timeout_seconds": int(self.config.get("timeout_seconds", 120) or 120),
            },
        )
        self._engine = engine
        return engine

    def get_runtime_snapshot(self) -> dict:
        """Expose the latest workflow state for UI panels."""
        snapshot = {
            "result": self._last_result,
            "events": [],
            "assignments": {},
            "mode": "",
            "complexity_score": 0,
            "dag_nodes": [],
            "runtime_logs": self.read_runtime_logs(),
            "log_file": str(RUNTIME_LOG_FILE),
            "runtime_config": {
                "workspace_root": self.config.get("workspace_root", ""),
                "allowed_tools": list(self.config.get("allowed_tools")) if self.config.get("allowed_tools") else None,
                "permission_mode": self.config.get("permission_mode", "workspace_only"),
                "assignment_mode": self.config.get("assignment_mode", "primary_preferred"),
                "timeout_seconds": int(self.config.get("timeout_seconds", 120) or 120),
            },
        }
        if self._last_result:
            snapshot["assignments"] = dict(getattr(self._last_result, "role_assignments", {}))
            snapshot["mode"] = getattr(self._last_result, "mode", "")
            snapshot["complexity_score"] = getattr(self._last_result, "complexity_score", 0)
            snapshot["dag_nodes"] = list(getattr(self._last_result, "dag_nodes", []))
        if self._engine and getattr(self._engine, "event_bus", None):
            snapshot["events"] = [event.to_dict() for event in self._engine.event_bus.all_events()]
        return snapshot

    def read_runtime_logs(self, limit: int = 200) -> list[str]:
        if not RUNTIME_LOG_FILE.exists():
            return []
        try:
            lines = RUNTIME_LOG_FILE.read_text(encoding="utf-8").splitlines()
            return lines[-limit:]
        except Exception:
            return []

    def get_tool_execution_records(self) -> list[dict]:
        """Get tool execution records from the current engine's ToolExecutor."""
        if self._engine and getattr(self._engine, "tool_executor", None):
            return self._engine.tool_executor.get_records_dict()
        return []

    def append_runtime_log(self, message: str):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(RUNTIME_LOG_FILE, "a", encoding="utf-8") as file:
                file.write(f"[{stamp}] {message}\n")
        except Exception:
            pass

    def execute(self, task_text: str, model_name: str):
        """Run a workflow in a background thread."""
        if self._worker and self._worker.is_alive():
            self._log(self.t("controller_running"))
            return

        self._cancel_flag.clear()
        self._worker = threading.Thread(
            target=self._run_workflow,
            args=(task_text, model_name),
            daemon=True,
        )
        self._worker.start()
        self._log(self.t("controller_start_log", task=task_text[:50]))

    def cancel(self):
        """Request cancellation for the current workflow."""
        self._cancel_flag.set()
        self._log(self.t("controller_cancel_log"))

    @property
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def _ensure_paths(self):
        base = Path("dr_mma_data")
        base.mkdir(exist_ok=True)
        defaults = {
            "blackboard_path": str(base / "blackboard.jsonl"),
            "artifact_path": str(base / "artifacts.jsonl"),
            "decision_path": str(base / "decisions.jsonl"),
        }
        for key, value in defaults.items():
            if not self.config.get(key):
                self.config[key] = value
        if not self.config.get("workspace_root"):
            self.config["workspace_root"] = str(Path.cwd())

    def _register_default_mock(self):
        if not any(item.get("name") == DEFAULT_MODEL_NAME for item in self._model_infos):
            self._model_infos.append({"name": DEFAULT_MODEL_NAME, "type": "mock"})
        self._model_adapter.register(DEFAULT_MODEL_NAME, MockModel(DEFAULT_MODEL_NAME))

    def _build_model_from_info(self, info: dict):
        name = info.get("name", "").strip()
        model_type = info.get("type", "mock")
        model_name = info.get("model_name", "").strip()
        if not name:
            raise ValueError(self.t("controller_model_name_required"))
        if model_type == "mock":
            return MockModel(name)
        if model_type == "local":
            return LocalModel(name, info.get("endpoint", ""), model_name)
        if model_type == "remote":
            return RemoteModel(
                name,
                info.get("endpoint", ""),
                info.get("api_key", ""),
                model_name,
            )
        raise ValueError(self.t("controller_unknown_model_type", model_type=model_type))

    def _run_workflow(self, task_text: str, model_name: str):
        try:
            engine = self.build_engine(model_name)
            self._last_result = None
            self._notify_stage("Planner", self.t("controller_plan_start"))

            if "Mock" in model_name:
                self._setup_mock_responses(model_name, task_text)

            # Create streaming session BEFORE execute so callbacks can register
            from ..engine.streaming import StreamSession
            stream_session = StreamSession()
            stream_session.on_event(self._on_stream_event)

            result = engine.execute(task_text, model_name, stream_session=stream_session, cancel_token=self._cancel_flag)
            self._last_result = result
            self._log(
                self.t(
                    "controller_log_mode",
                    mode=result.mode,
                    score=result.complexity_score,
                    events=result.event_count,
                )
            )

            if self._cancel_flag.is_set():
                self._notify_stage("CANCELLED", self.t("controller_cancelled"))
                return

            self._notify_stage("DONE", self.t("controller_complete"))
            self._notify_done(result)
        except Exception as exc:
            self._notify_error(self.t("controller_failed", error=str(exc)))
            self._log(traceback.format_exc())

    def _setup_mock_responses(self, model_name: str, task_text: str):
        mock = self._model_adapter.get(model_name)
        if not mock or not hasattr(mock, "add_response"):
            return

        mock.add_response(
            "Planner",
            json.dumps(
                {
                    "subtasks": [
                        {
                            "task_name": "需求分析",
                            "objective": f"分析任务：{task_text[:80]}",
                            "success_criteria": ["识别核心需求"],
                            "depends_on": [],
                        },
                        {
                            "task_name": "方案设计",
                            "objective": "设计一个可实施的解决方案。",
                            "success_criteria": ["给出明确的实现路径"],
                            "depends_on": ["需求分析"],
                        },
                    ]
                },
                ensure_ascii=False,
            ),
        )
        mock.add_response(
            "Worker",
            json.dumps(
                {
                    "status": "completed",
                    "summary": "执行完成",
                    "content": (
                        f"任务已处理。\n\n"
                        f"任务焦点：{task_text[:120]}\n"
                        "1. 已识别核心需求。\n"
                        "2. 已提出可实施方案。\n"
                        "3. 已整理主要风险与约束。"
                    ),
                },
                ensure_ascii=False,
            ),
        )
        mock.add_response(
            "Critic",
            json.dumps(
                {
                    "status": "completed",
                    "summary": "审查通过",
                    "content": "输出结构完整，已覆盖主要需求。",
                    "next_action_recommendation": "PASS",
                },
                ensure_ascii=False,
            ),
        )
        mock.add_response(
            "Verifier",
            json.dumps(
                {
                    "status": "completed",
                    "summary": "校验通过",
                    "content": "本次模拟执行已满足验收条件。",
                    "next_action_recommendation": "PASS",
                },
                ensure_ascii=False,
            ),
        )
        mock.add_response(
            "Supervisor",
            json.dumps(
                {
                    "status": "completed",
                    "summary": "最终汇总完成",
                    "content": (
                        "最终结果\n\n"
                        f"任务：{task_text[:120]}\n"
                        "- 已完成需求分析。\n"
                        "- 已输出方案方向。\n"
                        "- 已归纳关键风险。"
                    ),
                },
                ensure_ascii=False,
            ),
        )

    def _notify_stage(self, stage: str, message: str):
        if self.on_stage_change:
            self._app_safe_call(self.on_stage_change, stage, message)

    def _notify_done(self, result: WorkflowResult):
        if self.on_workflow_done:
            self._app_safe_call(self.on_workflow_done, result)

    def _notify_error(self, msg: str):
        if self.on_error:
            self._app_safe_call(self.on_error, msg)

    def _on_stream_event(self, event):
        """Handle streaming events and push to log."""
        from ..engine.streaming import StreamEventKind

        if event.kind == StreamEventKind.CHUNK:
            text = event.data.get("text", "")
            if text:
                self._log(f"[stream] {text}")
        elif event.kind == StreamEventKind.TOOL_CALL:
            tool = event.data.get("tool", "unknown")
            args = event.data.get("args", {})
            self._log(f"[tool_call] {tool}({args})")
        elif event.kind == StreamEventKind.TOOL_RESULT:
            tool = event.data.get("tool", "unknown")
            success = event.data.get("success", False)
            result = event.data.get("result", {})
            status = "success" if success else "failed"
            self._log(f"[tool_result] {tool} -> {status}: {str(result)[:100]}")
        elif event.kind == StreamEventKind.THINKING:
            content = event.data.get("content", "")
            self._log(f"[thinking] {content[:200]}")
        elif event.kind == StreamEventKind.DONE:
            self._log("[stream] 流式输出完成")
        elif event.kind == StreamEventKind.ERROR:
            error = event.data.get("error", "unknown")
            self._log(f"[stream error] {error}")

    def _log(self, msg: str):
        self.append_runtime_log(msg)
        if self.on_log:
            self._app_safe_call(self.on_log, msg)

    def _app_safe_call(self, callback, *args):
        """Call callback safely on the Tk main thread."""
        if self._tk_root is not None:
            self._tk_root.after(0, callback, *args)
        else:
            callback(*args)
