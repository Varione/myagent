"""Log and event viewer panel."""

from __future__ import annotations

import customtkinter as ctk

from .theme import COLORS, FONT_MONO, FONT_UI


class LogPanel(ctk.CTkFrame):
    """Displays runtime logs, event stream, blackboard entries, and decisions."""

    def __init__(self, parent, controller, **kwargs):
        super().__init__(parent, fg_color=COLORS["bg"], **kwargs)
        self.controller = controller
        self._build_ui()

    def _build_ui(self):
        self.title_label = ctk.CTkLabel(self, text="", font=(FONT_UI, 22, "bold"), text_color=COLORS["text"])
        self.title_label.pack(pady=(20, 8), padx=22, anchor="w")

        self.log_type = ctk.CTkSegmentedButton(
            self,
            values=[],
            command=self._on_switch_log,
            fg_color=COLORS["panel_alt"],
            selected_color=COLORS["accent"],
            selected_hover_color=COLORS["accent_alt"],
            text_color=COLORS["text"],
        )
        self.log_type.pack(pady=(0, 16), padx=22, anchor="w")

        self.log_text = ctk.CTkTextbox(
            self,
            font=(FONT_MONO, 12),
            fg_color=COLORS["surface"],
            text_color=COLORS["text"],
            border_width=1,
            border_color=COLORS["border"],
            corner_radius=20,
        )
        self.log_text.pack(pady=(0, 18), padx=22, fill="both", expand=True)
        self.log_text.configure(state="disabled")

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=(0, 22), padx=22, fill="x")

        self.clear_btn = ctk.CTkButton(
            btn_frame,
            text="",
            width=90,
            fg_color=COLORS["panel_alt"],
            hover_color=COLORS["sidebar_active"],
            command=self._clear_logs,
        )
        self.clear_btn.pack(side="left")
        self.refresh_btn = ctk.CTkButton(
            btn_frame,
            text="",
            width=120,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_alt"],
            text_color="#07213a",
            command=self._refresh_current,
        )
        self.refresh_btn.pack(side="left", padx=(10, 0))
        self.refresh_language()

    def refresh_language(self):
        self.title_label.configure(text=self.controller.t("logs_title"))
        values = [
            self.controller.t("logs_runtime"),
            self.controller.t("logs_events"),
            self.controller.t("logs_blackboard"),
            self.controller.t("logs_decisions"),
            self.controller.t("logs_tools"),
        ]
        self.log_type.configure(values=values)
        self.log_type.set(values[0])
        self.clear_btn.configure(text=self.controller.t("logs_clear"))
        self.refresh_btn.configure(text=self.controller.t("logs_refresh"))
        self._render_runtime_logs()

    def append_log(self, message: str):
        if self.log_type.get() == self.controller.t("logs_runtime"):
            self.log_text.configure(state="normal")
            self.log_text.insert("end", f"{message}\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")

    def _refresh_current(self):
        self._on_switch_log(self.log_type.get())

    def _on_switch_log(self, value: str):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        if value == self.controller.t("logs_runtime"):
            self._render_runtime_logs()
        elif value == self.controller.t("logs_events"):
            self._refresh_events()
        elif value == self.controller.t("logs_blackboard"):
            self._refresh_blackboard()
        elif value == self.controller.t("logs_decisions"):
            self._refresh_decision_log()
        elif value == self.controller.t("logs_tools"):
            self._refresh_tools()
        self.log_text.configure(state="disabled")

    def _render_runtime_logs(self):
        logs = self.controller.read_runtime_logs()
        if not logs:
            self.log_text.insert("1.0", f"{self.controller.t('logs_ready')}\n{self.controller.t('logs_runtime_missing')}\n")
            return
        self.log_text.insert("1.0", "\n".join(logs) + "\n")

    def _refresh_events(self):
        snapshot = self.controller.get_runtime_snapshot()
        mode = snapshot.get("mode", "")
        score = snapshot.get("complexity_score", 0)
        assignments = snapshot.get("assignments", {})
        events = snapshot.get("events", [])

        if mode:
            self.log_text.insert("end", f"[mode] {mode} | score={score}\n\n")
        if assignments:
            self.log_text.insert("end", "[role assignments]\n")
            for role, model in assignments.items():
                self.log_text.insert("end", f"- {role}: {model}\n")
            self.log_text.insert("end", "\n")
        if not events:
            self.log_text.insert("end", f"{self.controller.t('logs_events_none')}\n")
            return
        self.log_text.insert("end", "[events]\n")
        for event in events[-50:]:
            self.log_text.insert(
                "end",
                f"- {event.get('event_type')} task={event.get('task_id')} source={event.get('source')} payload={event.get('payload', {})}\n",
            )

    def _refresh_blackboard(self):
        bb_path = self.controller.config.get("blackboard_path", "")
        if not bb_path:
            self.log_text.insert("1.0", f"{self.controller.t('logs_blackboard_missing')}\n")
            return
        from ..storage.blackboard import Blackboard

        try:
            bb = Blackboard(bb_path)
            entries = bb.query()
            if not entries:
                self.log_text.insert("1.0", f"{self.controller.t('logs_blackboard_empty')}\n")
            for entry in entries[-20:]:
                self.log_text.insert("end", f"[{entry.content_type}] {entry.source_role}: {entry.summary[:120]}\n")
        except Exception as exc:
            self.log_text.insert("1.0", f"{self.controller.t('logs_blackboard_failed', error=exc)}\n")

    def _refresh_decision_log(self):
        dec_path = self.controller.config.get("decision_path", "")
        if not dec_path:
            self.log_text.insert("1.0", f"{self.controller.t('logs_decisions_missing')}\n")
            return
        from ..storage.decision_log import DecisionLog

        try:
            log = DecisionLog(dec_path)
            records = log.query()
            if not records:
                self.log_text.insert("1.0", f"{self.controller.t('logs_decisions_empty')}\n")
            for record in records[-20:]:
                self.log_text.insert("end", f"[{record.decision}] Task {record.task_id}: {record.rationale[:120]}\n")
        except Exception as exc:
            self.log_text.insert("1.0", f"{self.controller.t('logs_decisions_failed', error=exc)}\n")

    def _refresh_tools(self):
        records = self.controller.get_tool_execution_records()
        if not records:
            self.log_text.insert("1.0", f"{self.controller.t('tools_none')}\n")
            return
        for record in records[-50:]:
            tool_name = record.get("tool_name", "unknown")
            role = record.get("role", "")
            latency_ms = record.get("latency_ms", 0)
            permission_allowed = record.get("permission_allowed", True)
            success = record.get("success", False)
            error = record.get("error", "")
            output = record.get("output", "")
            permission_reason = record.get("permission_reason", "")

            if not permission_allowed:
                status_text = self.controller.t("tools_denied")
                detail = f"[{tool_name}] role={role} status={status_text} latency={latency_ms:.0f}ms | reason: {permission_reason}"
            elif success:
                preview = (output or "")[:200] if output else ""
                status_text = self.controller.t("tools_success")
                detail = f"[{tool_name}] role={role} status={status_text} latency={latency_ms:.0f}ms | {preview}"
            else:
                status_text = self.controller.t("tools_failed")
                detail = f"[{tool_name}] role={role} status={status_text} latency={latency_ms:.0f}ms | error: {error}"

            self.log_text.insert("end", f"{detail}\n")

    def _clear_logs(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.insert("1.0", f"{self.controller.t('logs_cleared')}\n")
        self.log_text.configure(state="disabled")
