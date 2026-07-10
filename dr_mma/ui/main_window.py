"""Main desktop window for DR-MMA."""

from __future__ import annotations

import json

import customtkinter as ctk

from .chat_panel import ChatPanel
from .config_panel import ConfigPanel
from .controller import WorkflowController
from .i18n import EN_US, ZH_CN
from .log_panel import LogPanel
from .pipeline_panel import PipelinePanel
from .results_panel import ResultsPanel
from .theme import COLORS, FONT_UI


class MainWindow(ctk.CTk):
    """Top-level desktop application window."""

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.controller = WorkflowController()
        self.chat_panel: ChatPanel | None = None
        self.pipeline_panel: PipelinePanel | None = None
        self.results_panel: ResultsPanel | None = None
        self.log_panel: LogPanel | None = None
        self.config_panel: ConfigPanel | None = None
        self._workflow_running = False
        self._active_model_name = ""
        self._nav_btns: dict[str, ctk.CTkButton] = {}
        self._nav_labels = {
            "chat": "nav_chat",
            "pipeline": "nav_pipeline",
            "results": "nav_results",
            "logs": "nav_logs",
            "config": "nav_config",
        }
        self._language_var = ctk.StringVar(value=self.controller.language)

        self.title(self.controller.t("app_title"))
        self.geometry("1480x920")
        self.minsize(1180, 780)
        self.configure(fg_color=COLORS["bg"])

        self._build_ui()
        self._register_callbacks()
        self.after(100, self._on_config_save)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.refresh_language()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=0, minsize=260)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, corner_radius=0, fg_color=COLORS["sidebar"])
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        header = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        header.pack(pady=(28, 24), padx=20, fill="x")
        self.brand_label = ctk.CTkLabel(
            header,
            text="DR-MMA",
            font=(FONT_UI, 28, "bold"),
            text_color=COLORS["accent"],
        )
        self.brand_label.pack(anchor="w")
        self.subtitle_label = ctk.CTkLabel(
            header,
            text="",
            font=(FONT_UI, 12),
            text_color=COLORS["muted"],
        )
        self.subtitle_label.pack(anchor="w", pady=(4, 0))

        self.lang_label = ctk.CTkLabel(
            self.sidebar,
            text="",
            font=(FONT_UI, 12),
            text_color=COLORS["muted"],
        )
        self.lang_label.pack(anchor="w", padx=20)
        self.lang_menu = ctk.CTkOptionMenu(
            self.sidebar,
            variable=self._language_var,
            values=[ZH_CN, EN_US],
            command=self._on_language_change,
            width=200,
            fg_color=COLORS["panel_alt"],
            button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_alt"],
        )
        self.lang_menu.pack(anchor="w", padx=20, pady=(6, 20))

        for page_id in ("chat", "pipeline", "results", "logs", "config"):
            btn = ctk.CTkButton(
                self.sidebar,
                text="",
                font=(FONT_UI, 15, "bold"),
                anchor="w",
                height=46,
                corner_radius=14,
                fg_color="transparent",
                text_color=COLORS["muted"],
                hover_color=COLORS["sidebar_hover"],
                command=lambda pid=page_id: self._switch_page(pid),
            )
            btn.pack(pady=4, padx=14, fill="x")
            self._nav_btns[page_id] = btn

        self.version_label = ctk.CTkLabel(
            self.sidebar,
            text="v0.3",
            font=(FONT_UI, 12),
            text_color="#5c7395",
        )
        self.version_label.pack(side="bottom", pady=22, padx=20, anchor="w")

        self.content = ctk.CTkFrame(self, corner_radius=0, fg_color=COLORS["bg"])
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)

        self.chat_panel = ChatPanel(self.content, self.controller)
        self.chat_panel.set_on_send(self._on_send)
        self.pipeline_panel = PipelinePanel(self.content, self.controller)
        self.results_panel = ResultsPanel(self.content, self.controller)
        self.log_panel = LogPanel(self.content, self.controller)
        self.config_panel = ConfigPanel(self.content, self.controller, on_save=self._on_config_save)

        self._switch_page("chat")

    def _register_callbacks(self):
        self.controller.on_stage_change = self._on_stage_change
        self.controller.on_workflow_done = self._on_workflow_done
        self.controller.on_error = self._on_error
        self.controller.on_log = self._on_log

    def _on_language_change(self, language: str):
        self.controller.set_language(language)
        self.refresh_language()

    def refresh_language(self):
        self.title(self.controller.t("app_title"))
        self.subtitle_label.configure(text=self.controller.t("app_subtitle"))
        self.lang_label.configure(text=self.controller.t("lang_label"))
        self.lang_menu.configure(
            values=[self.controller.t("lang_zh"), self.controller.t("lang_en")]
            if self._language_var.get() in {self.controller.t("lang_zh"), self.controller.t("lang_en")}
            else [ZH_CN, EN_US]
        )
        current = self.controller.language
        display = self.controller.t("lang_zh") if current == ZH_CN else self.controller.t("lang_en")
        self.lang_menu.set(display)
        for page_id, btn in self._nav_btns.items():
            btn.configure(text=self.controller.t(self._nav_labels[page_id]))

        for panel in (self.chat_panel, self.pipeline_panel, self.results_panel, self.log_panel, self.config_panel):
            if hasattr(panel, "refresh_language"):
                panel.refresh_language()

    def _switch_page(self, page_id: str):
        for current_id, btn in self._nav_btns.items():
            if current_id == page_id:
                btn.configure(fg_color=COLORS["sidebar_active"], text_color=COLORS["text"])
            else:
                btn.configure(fg_color="transparent", text_color=COLORS["muted"])

        for panel, panel_id in [
            (self.chat_panel, "chat"),
            (self.pipeline_panel, "pipeline"),
            (self.results_panel, "results"),
            (self.log_panel, "logs"),
            (self.config_panel, "config"),
        ]:
            if panel_id == page_id:
                panel.grid(row=0, column=0, sticky="nsew")
                panel.lift()
            else:
                panel.grid_remove()

    def _on_send(self, task: str, model: str):
        if self._workflow_running or not task:
            return

        self._active_model_name = model
        self._workflow_running = True
        self.chat_panel.set_sending(True)
        self.chat_panel.set_status(self.controller.t("status_running"), COLORS["warning"])
        self.pipeline_panel.reset()
        self.results_panel.reset()
        self.pipeline_panel.set_stage("Planner", self.controller.t("controller_plan_start"))
        self.pipeline_panel.update_dag([])
        self.chat_panel.add_system_bubble(self.controller.t("workflow_started", model=model), "info")
        self.controller.execute(task, model)

    def _on_stage_change(self, stage: str, message: str):
        self.after(0, self._do_stage_change, stage, message)

    def _do_stage_change(self, stage: str, message: str):
        model_name = self._active_model_name or self.chat_panel.get_selected_model()
        if stage in getattr(self.pipeline_panel, "cards", {}):
            self.pipeline_panel.set_stage(stage, message or stage)
        if stage == "Planner":
            self.chat_panel.add_agent_bubble("Planner", self.controller.t("stage_planner_progress"), message, model_name=model_name)
        elif stage == "DONE":
            self.pipeline_panel.set_stage("DONE", message or self.controller.t("controller_complete"))
            self.chat_panel.add_system_bubble(self.controller.t("workflow_done"), "success")
        elif stage == "CANCELLED":
            self.pipeline_panel.set_cancelled()
            self.chat_panel.add_system_bubble(self.controller.t("workflow_cancelled"), "info")

    def _on_workflow_done(self, result):
        self.after(0, self._do_workflow_done, result)

    def _do_workflow_done(self, result):
        self._workflow_running = False
        self.chat_panel.set_sending(False)
        self.chat_panel.set_status(self.controller.t("status_completed"), COLORS["success"])

        summary_lines = [
            f"{self.controller.t('main_mode')}: {getattr(result, 'mode', '')}",
            f"{self.controller.t('main_score')}: {getattr(result, 'complexity_score', 0)}",
            f"{self.controller.t('main_status')}: {getattr(result, 'status', '')}",
            f"{self.controller.t('main_events')}: {getattr(result, 'event_count', 0)}",
            "",
            f"{self.controller.t('main_assignments')}:",
        ]
        for role, model in getattr(result, "role_assignments", {}).items():
            summary_lines.append(f"- {role}: {model}")
        summary_lines.extend(
            [
                "",
                f"{self.controller.t('main_runtime')}:",
                f"- workspace_root: {getattr(result, 'runtime_config', {}).get('workspace_root', '')}",
                f"- permission_mode: {getattr(result, 'runtime_config', {}).get('permission_mode', '')}",
                f"- assignment_mode: {getattr(result, 'runtime_config', {}).get('assignment_mode', '')}",
                f"- timeout_seconds: {getattr(result, 'runtime_config', {}).get('timeout_seconds', 120)}",
                f"- allowed_tools: {', '.join(getattr(result, 'runtime_config', {}).get('allowed_tools', []))}",
            ]
        )
        self.results_panel.set_summary("\n".join(summary_lines))
        self.results_panel.set_final_output(result.final_output)
        self.pipeline_panel.update_dag(getattr(result, "dag_nodes", []))

        model_name = self._active_model_name or self.chat_panel.get_selected_model()
        self.chat_panel.add_agent_bubble(
            "Supervisor",
            self.controller.t("main_summary_supervisor"),
            detail=result.final_output,
            status=result.status,
            model_name=model_name,
        )

        for subtask_id, subtask_result in result.subtask_results.items():
            self.results_panel.add_subtask(subtask_id)
            worker = subtask_result.get("worker")
            critic = subtask_result.get("critic")
            verifier = subtask_result.get("verifier")

            worker_text = getattr(worker, "content", "") if worker else ""
            critic_text = getattr(critic, "content", "") if critic else ""
            verifier_text = getattr(verifier, "content", "") if verifier else ""
            contract_text = json.dumps(subtask_result.get("contracts", {}), ensure_ascii=False, indent=2)
            response_text = json.dumps(
                {
                    "worker": getattr(worker, "to_dict", lambda: {})(),
                    "critic": getattr(critic, "to_dict", lambda: {})(),
                    "verifier": getattr(verifier, "to_dict", lambda: {})(),
                },
                ensure_ascii=False,
                indent=2,
            )

            self.results_panel.set_subtask_content(
                subtask_id,
                worker=worker_text,
                critic=critic_text,
                verifier=verifier_text,
                contract=contract_text,
                response=response_text,
            )

            if worker:
                self.chat_panel.add_agent_bubble(
                    "Worker",
                    getattr(worker, "summary", self.controller.t("main_summary_worker")),
                    detail=worker_text,
                    status=getattr(worker, "status", "completed"),
                    claims=getattr(worker, "claims", []),
                    risks=getattr(worker, "risks", []),
                    model_name=model_name,
                )
            if critic:
                self.chat_panel.add_agent_bubble(
                    "Critic",
                    getattr(critic, "summary", self.controller.t("main_summary_critic")),
                    detail=critic_text,
                    claims=getattr(critic, "claims", []),
                    model_name=model_name,
                )
            if verifier:
                self.chat_panel.add_agent_bubble(
                    "Verifier",
                    getattr(verifier, "summary", self.controller.t("main_summary_verifier")),
                    detail=verifier_text,
                    status=getattr(verifier, "status", "completed"),
                    model_name=model_name,
                )

        self._refresh_blackboard()
        self.log_panel._refresh_current()
        self.chat_panel.add_system_bubble(
            self.controller.t(
                "workflow_complete_metrics",
                latency=result.total_latency_ms,
                count=result.blackboard_count,
            ),
            "success",
        )
        self._active_model_name = ""

    def _on_error(self, msg: str):
        self.after(0, self._do_error, msg)

    def _do_error(self, msg: str):
        self._workflow_running = False
        self.chat_panel.set_sending(False)
        self.chat_panel.set_status(self.controller.t("status_error"), COLORS["danger"])
        self.pipeline_panel.set_error(msg)
        self.chat_panel.add_system_bubble(msg, "error")
        self.log_panel._refresh_current()
        self._active_model_name = ""

    def _on_log(self, msg: str):
        self.after(0, self.chat_panel.set_status, msg, COLORS["muted"])
        self.after(0, self.log_panel.append_log, msg)

    def _on_config_save(self):
        self.chat_panel.refresh_models()
        self.config_panel.refresh_language()

    def _refresh_blackboard(self):
        bb_path = self.controller.config.get("blackboard_path", "")
        art_path = self.controller.config.get("artifact_path", "")
        if not bb_path:
            return
        try:
            from ..storage.artifact_store import ArtifactStore
            from ..storage.blackboard import Blackboard

            blackboard = Blackboard(bb_path)
            entries = blackboard.query()
            self.chat_panel.update_blackboard(entries)

            artifact_store = ArtifactStore(art_path) if art_path else None
            if artifact_store:
                self.chat_panel.update_artifacts(artifact_store.list_artifacts())
        except Exception:
            pass

    def _on_close(self):
        self.controller.save_config()
        self.controller.cancel()
        self.destroy()
