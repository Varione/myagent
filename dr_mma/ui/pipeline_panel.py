"""Workflow pipeline and DAG summary visualization."""

from __future__ import annotations

import customtkinter as ctk

from .theme import COLORS, FONT_MONO, FONT_UI


STAGES = ["Planner", "Worker", "Critic", "Verifier", "Supervisor", "DONE"]

STATUS_COLORS = {
    "pending": COLORS["muted"],
    "active": COLORS["warning"],
    "done": COLORS["success"],
    "error": COLORS["danger"],
    "cancelled": "#7b8798",
    "completed": COLORS["success"],
    "partial": COLORS["warning"],
    "failed": COLORS["danger"],
}

STATUS_SYMBOLS = {
    "pending": "○",
    "active": "◉",
    "done": "●",
    "error": "×",
    "cancelled": "–",
    "completed": "●",
    "partial": "~",
    "failed": "×",
}


class StageCard(ctk.CTkFrame):
    """Single workflow stage card."""

    def __init__(self, parent, controller, stage_id: str, **kwargs):
        super().__init__(
            parent,
            corner_radius=18,
            fg_color=COLORS["surface"],
            border_width=1,
            border_color=COLORS["border"],
            **kwargs,
        )
        self.controller = controller
        self.stage_id = stage_id
        self._status = "pending"
        self.dot = ctk.CTkLabel(self, text=STATUS_SYMBOLS["pending"], font=(FONT_MONO, 22), text_color=COLORS["muted"])
        self.dot.pack(pady=(12, 0))
        self.name_label = ctk.CTkLabel(self, text=stage_id, font=(FONT_UI, 15, "bold"), text_color=COLORS["text"])
        self.name_label.pack(pady=(4, 0))
        self.id_label = ctk.CTkLabel(self, text=stage_id, font=(FONT_UI, 11), text_color=COLORS["muted"])
        self.id_label.pack(pady=(2, 12))
        self.configure(height=108)

    def refresh_language(self):
        self.name_label.configure(text=self.stage_id)

    def set_status(self, status: str):
        self._status = status
        color = STATUS_COLORS.get(status, COLORS["muted"])
        self.configure(border_color=color if status != "pending" else COLORS["border"])
        self.dot.configure(text=STATUS_SYMBOLS.get(status, "○"), text_color=color)
        self.update_idletasks()


class PipelinePanel(ctk.CTkFrame):
    """Visual summary of workflow stages and planned subtasks."""

    def __init__(self, parent, controller, **kwargs):
        super().__init__(parent, fg_color=COLORS["bg"], **kwargs)
        self.controller = controller
        self.cards: dict[str, StageCard] = {}
        self._build_ui()

    def _build_ui(self):
        self.title_label = ctk.CTkLabel(self, text="", font=(FONT_UI, 22, "bold"), text_color=COLORS["text"])
        self.title_label.pack(pady=(20, 4), padx=22, anchor="w")
        self.subtitle_label = ctk.CTkLabel(self, text="", font=(FONT_UI, 13), text_color=COLORS["muted"])
        self.subtitle_label.pack(pady=(0, 18), padx=22, anchor="w")

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(pady=6, padx=22, fill="both", expand=True)
        body.grid_columnconfigure(0, weight=0, minsize=300)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self.pipeline_frame = ctk.CTkFrame(
            body,
            fg_color=COLORS["surface"],
            corner_radius=24,
            border_width=1,
            border_color=COLORS["border"],
        )
        self.pipeline_frame.grid(row=0, column=0, sticky="nsw", padx=(0, 18))

        for index, stage_id in enumerate(STAGES):
            card = StageCard(self.pipeline_frame, self.controller, stage_id)
            card.pack(pady=6, padx=12, fill="x", ipady=4)
            self.cards[stage_id] = card
            if index < len(STAGES) - 1:
                ctk.CTkLabel(self.pipeline_frame, text="│", font=(FONT_MONO, 18), text_color=COLORS["border"]).pack()

        dag_container = ctk.CTkFrame(
            body,
            fg_color=COLORS["surface"],
            corner_radius=24,
            border_width=1,
            border_color=COLORS["border"],
        )
        dag_container.grid(row=0, column=1, sticky="nsew")
        dag_container.grid_rowconfigure(1, weight=1)
        dag_container.grid_columnconfigure(0, weight=1)

        self.dag_title_label = ctk.CTkLabel(
            dag_container, text="", font=(FONT_UI, 17, "bold"), text_color=COLORS["accent"]
        )
        self.dag_title_label.grid(row=0, column=0, sticky="w", padx=18, pady=(16, 8))

        self.dag_scroll = ctk.CTkScrollableFrame(dag_container, fg_color="transparent")
        self.dag_scroll.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        self._show_empty_dag()

        self.status_bar = ctk.CTkFrame(
            self,
            fg_color=COLORS["surface"],
            corner_radius=18,
            border_width=1,
            border_color=COLORS["border"],
        )
        self.status_bar.pack(pady=(18, 22), padx=22, fill="x")
        self.status_label = ctk.CTkLabel(
            self.status_bar,
            text="",
            font=(FONT_UI, 14),
            anchor="w",
            justify="left",
            text_color=COLORS["text"],
        )
        self.status_label.pack(pady=10, padx=16, fill="x")
        self.refresh_language()

    def refresh_language(self):
        self.title_label.configure(text=self.controller.t("pipeline_title"))
        self.subtitle_label.configure(text=self.controller.t("pipeline_subtitle"))
        self.dag_title_label.configure(text=self.controller.t("pipeline_dag_title"))
        if self.status_label.cget("text") in {"", self.controller.t("pipeline_waiting")}:
            self.status_label.configure(text=self.controller.t("pipeline_waiting"))
        for card in self.cards.values():
            card.refresh_language()

    def _show_empty_dag(self):
        for child in self.dag_scroll.winfo_children():
            child.destroy()
        ctk.CTkLabel(
            self.dag_scroll,
            text=self.controller.t("pipeline_empty"),
            font=(FONT_UI, 13),
            text_color=COLORS["muted"],
            justify="left",
            wraplength=660,
        ).pack(pady=24, padx=14, anchor="w")

    def reset(self):
        for card in self.cards.values():
            card.set_status("pending")
        self.status_label.configure(text=self.controller.t("pipeline_waiting"), text_color=COLORS["text"])
        self._show_empty_dag()

    def set_stage(self, stage_id: str, message: str = ""):
        found = False
        for current_id, card in self.cards.items():
            if current_id == stage_id:
                card.set_status("done" if stage_id == "DONE" else "active")
                found = True
            elif found or current_id == "DONE":
                card.set_status("pending")
            else:
                card.set_status("done")
        if message:
            self.status_label.configure(text=message, text_color=COLORS["text"])

    def update_dag(self, nodes: list[dict]):
        for child in self.dag_scroll.winfo_children():
            child.destroy()
        if not nodes:
            self._show_empty_dag()
            return
        for node in nodes:
            status = node.get("status", "pending")
            color = STATUS_COLORS.get(status, COLORS["muted"])
            deps = ", ".join(node.get("depends_on", [])) or self.controller.t("pipeline_depends_none")
            card = ctk.CTkFrame(
                self.dag_scroll,
                fg_color=COLORS["card"],
                corner_radius=16,
                border_width=1,
                border_color=color,
            )
            card.pack(pady=8, padx=6, fill="x")
            ctk.CTkLabel(
                card,
                text=f"{node.get('task_id', '')}  [{status}]",
                font=(FONT_MONO, 12, "bold"),
                text_color=color,
            ).pack(pady=(10, 4), padx=14, anchor="w")
            ctk.CTkLabel(
                card,
                text=node.get("task_name", ""),
                font=(FONT_UI, 14, "bold"),
                text_color=COLORS["text"],
            ).pack(pady=(0, 4), padx=14, anchor="w")
            ctk.CTkLabel(
                card,
                text=node.get("objective", "")[:220],
                font=(FONT_UI, 12),
                text_color=COLORS["muted"],
                justify="left",
                wraplength=700,
            ).pack(pady=(0, 4), padx=14, anchor="w")
            ctk.CTkLabel(
                card,
                text=f"{self.controller.t('pipeline_depends_on')}: {deps}",
                font=(FONT_MONO, 11),
                text_color=COLORS["accent"],
            ).pack(pady=(0, 10), padx=14, anchor="w")

    def set_error(self, message: str):
        self.status_label.configure(text=self.controller.t("pipeline_error", message=message), text_color=COLORS["danger"])
        for card in self.cards.values():
            if card._status == "active":
                card.set_status("error")

    def set_cancelled(self):
        self.status_label.configure(text=self.controller.t("workflow_cancelled"), text_color=STATUS_COLORS["cancelled"])
        for card in self.cards.values():
            if card._status == "pending":
                card.set_status("cancelled")
