"""Chat-style workflow view."""

from __future__ import annotations

from typing import Optional

import customtkinter as ctk

from .theme import COLORS, FONT_MONO, FONT_UI


ROLE_COLORS = {
    "Planner": ("#1f8bff", "#58d6ff"),
    "Worker": ("#19b37d", "#89f0c4"),
    "Critic": ("#ff9f43", "#ffd180"),
    "Verifier": ("#b084f9", "#d3b7ff"),
    "Supervisor": ("#ff6b8a", "#ff9bb0"),
    "User": ("#1d3557", "#457b9d"),
    "System": ("#3b4f73", "#7ea7d8"),
}


class AgentBubble(ctk.CTkFrame):
    """Collapsible agent output card."""

    def __init__(
        self,
        parent,
        controller,
        role: str,
        summary: str,
        detail: str = "",
        status: str = "completed",
        claims: list | None = None,
        risks: list | None = None,
        model_name: str = "",
        **kwargs,
    ):
        super().__init__(
            parent,
            corner_radius=18,
            fg_color=COLORS["card"],
            border_width=1,
            border_color=COLORS["border"],
            **kwargs,
        )
        self.controller = controller
        self.role = role
        self._expanded = False
        self._toggle_btn: ctk.CTkButton | None = None
        self._claims_label: ctk.CTkLabel | None = None
        self._risks_label: ctk.CTkLabel | None = None
        self._role_desc_label: ctk.CTkLabel | None = None
        bg, _ = ROLE_COLORS.get(role, ("#334155", "#64748b"))

        row1 = ctk.CTkFrame(self, fg_color="transparent")
        row1.pack(pady=(12, 4), padx=16, fill="x")

        self.role_label = ctk.CTkLabel(
            row1,
            text=f"  {role}  ",
            font=(FONT_UI, 13, "bold"),
            fg_color=bg,
            text_color="white",
            corner_radius=8,
        )
        self.role_label.pack(side="left")

        self.model_label = ctk.CTkLabel(
            row1,
            text=f"  {model_name}  " if model_name else "",
            font=(FONT_UI, 11),
            fg_color=COLORS["panel_alt"],
            text_color=COLORS["accent"],
            corner_radius=8,
        )
        if model_name:
            self.model_label.pack(side="left", padx=(8, 0))

        status_symbol = {
            "completed": "●",
            "failed": "×",
            "need_review": "!",
            "low_confidence": "?",
            "partial": "~",
        }.get(status, "•")
        status_color = {
            "completed": COLORS["success"],
            "failed": COLORS["danger"],
            "need_review": COLORS["warning"],
            "low_confidence": COLORS["warning"],
            "partial": COLORS["warning"],
        }.get(status, COLORS["muted"])
        self.status_label = ctk.CTkLabel(row1, text=status_symbol, font=(FONT_UI, 15, "bold"), text_color=status_color)
        self.status_label.pack(side="left", padx=(10, 0))

        self.summary_label = ctk.CTkLabel(
            row1,
            text=summary[:80],
            font=(FONT_UI, 13, "bold"),
            text_color=COLORS["text"],
            anchor="w",
        )
        self.summary_label.pack(side="left", padx=(12, 0), fill="x", expand=True)

        self._toggle_btn = ctk.CTkButton(
            row1,
            text="",
            font=(FONT_UI, 12),
            width=96,
            height=28,
            fg_color=COLORS["panel_alt"],
            hover_color=COLORS["sidebar_active"],
            text_color=COLORS["text"],
            command=self._toggle_detail,
        )
        self._toggle_btn.pack(side="right")

        self._role_desc_label = ctk.CTkLabel(
            self,
            text="",
            font=(FONT_UI, 11),
            text_color=COLORS["muted"],
            justify="left",
            wraplength=720,
        )
        self._role_desc_label.pack(pady=(0, 8), padx=16, anchor="w")

        self._detail_frame = ctk.CTkFrame(self, fg_color=COLORS["surface"], corner_radius=14)
        self._detail_text = ctk.CTkTextbox(
            self._detail_frame,
            height=132,
            font=(FONT_MONO, 12),
            fg_color=COLORS["surface"],
            text_color=COLORS["text"],
            border_width=0,
        )
        self._detail_text.pack(pady=(10, 6), padx=12, fill="x")
        self._detail_text.insert("1.0", detail)
        self._detail_text.configure(state="disabled")

        if claims:
            self._claims_label = ctk.CTkLabel(
                self._detail_frame,
                text="",
                font=(FONT_UI, 12, "bold"),
                text_color=COLORS["accent_alt"],
            )
            self._claims_label.pack(pady=(0, 2), padx=12, anchor="w")
            for claim in claims:
                text = getattr(claim, "claim", str(claim))
                conf = getattr(claim, "confidence", 0)
                ctk.CTkLabel(
                    self._detail_frame,
                    text=f"- {text} ({conf})",
                    font=(FONT_UI, 11),
                    text_color=COLORS["text"],
                    anchor="w",
                    justify="left",
                    wraplength=720,
                ).pack(pady=1, padx=16, anchor="w")

        if risks:
            self._risks_label = ctk.CTkLabel(
                self._detail_frame,
                text="",
                font=(FONT_UI, 12, "bold"),
                text_color=COLORS["warning"],
            )
            self._risks_label.pack(pady=(6, 2), padx=12, anchor="w")
            for risk in risks:
                text = getattr(risk, "risk", str(risk))
                sev = getattr(risk, "severity", "")
                ctk.CTkLabel(
                    self._detail_frame,
                    text=f"- {text} [{sev}]",
                    font=(FONT_UI, 11),
                    text_color=COLORS["text"],
                    anchor="w",
                    justify="left",
                    wraplength=720,
                ).pack(pady=1, padx=16, anchor="w")

        self.refresh_language()

    def refresh_language(self):
        desc_key = {
            "Planner": "chat_role_planner_desc",
            "Worker": "chat_role_worker_desc",
            "Critic": "chat_role_critic_desc",
            "Verifier": "chat_role_verifier_desc",
            "Supervisor": "chat_role_supervisor_desc",
        }.get(self.role, "")
        if desc_key:
            self._role_desc_label.configure(text=self.controller.t(desc_key))
        if self._toggle_btn:
            self._toggle_btn.configure(
                text=self.controller.t("chat_hide_details") if self._expanded else self.controller.t("chat_show_details")
            )
        if self._claims_label:
            self._claims_label.configure(text=self.controller.t("chat_claims"))
        if self._risks_label:
            self._risks_label.configure(text=self.controller.t("chat_risks"))

    def _toggle_detail(self):
        self._expanded = not self._expanded
        if self._expanded:
            self._detail_frame.pack(pady=(0, 12), padx=16, fill="x")
        else:
            self._detail_frame.pack_forget()
        self.refresh_language()


class UserBubble(ctk.CTkFrame):
    def __init__(self, parent, controller, text: str, **kwargs):
        super().__init__(
            parent,
            corner_radius=18,
            fg_color="#113153",
            border_width=1,
            border_color="#2b5f8a",
            **kwargs,
        )
        self.controller = controller
        self.title_label = ctk.CTkLabel(self, text="", font=(FONT_UI, 13, "bold"), text_color=COLORS["accent"])
        self.title_label.pack(pady=(8, 2), padx=16, anchor="w")
        self.body_label = ctk.CTkLabel(
            self,
            text=text,
            font=(FONT_UI, 13),
            text_color=COLORS["text"],
            anchor="w",
            justify="left",
            wraplength=700,
        )
        self.body_label.pack(pady=(0, 10), padx=16, fill="x")
        self.refresh_language()

    def refresh_language(self):
        self.title_label.configure(text=self.controller.t("chat_user"))


class SystemBubble(ctk.CTkFrame):
    def __init__(self, parent, controller, text: str, msg_type: str = "info", **kwargs):
        palette = {
            "info": ("#17304f", "#9fd8ff"),
            "error": ("#421f2a", "#ffadbb"),
            "success": ("#153628", "#9ff3bf"),
        }
        fg, text_color = palette.get(msg_type, ("#1f2937", COLORS["text"]))
        super().__init__(parent, corner_radius=14, fg_color=fg, **kwargs)
        self.label = ctk.CTkLabel(
            self,
            text=text,
            font=(FONT_UI, 12),
            text_color=text_color,
            anchor="w",
            justify="left",
            wraplength=760,
        )
        self.label.pack(pady=10, padx=16, fill="x")


class BlackboardSidebar(ctk.CTkFrame):
    def __init__(self, parent, controller, **kwargs):
        super().__init__(
            parent,
            corner_radius=24,
            fg_color=COLORS["surface"],
            border_width=1,
            border_color=COLORS["border"],
            **kwargs,
        )
        self.controller = controller
        self.title_label = ctk.CTkLabel(self, text="", font=(FONT_UI, 17, "bold"), text_color=COLORS["accent"])
        self.title_label.pack(pady=(16, 4), padx=16, anchor="w")
        self._summary_label = ctk.CTkLabel(self, text="", font=(FONT_UI, 12), text_color=COLORS["muted"])
        self._summary_label.pack(pady=(0, 10), padx=16, anchor="w")

        self._list_frame = ctk.CTkScrollableFrame(self, fg_color="transparent", height=340)
        self._list_frame.pack(pady=(0, 12), padx=12, fill="both", expand=True)

        self.artifacts_label = ctk.CTkLabel(self, text="", font=(FONT_UI, 14, "bold"), text_color=COLORS["accent_alt"])
        self.artifacts_label.pack(pady=(0, 4), padx=16, anchor="w")
        self._artifact_frame = ctk.CTkScrollableFrame(self, fg_color="transparent", height=120)
        self._artifact_frame.pack(pady=(0, 12), padx=12, fill="x")
        self.refresh_language()

    def refresh_language(self):
        self.title_label.configure(text=self.controller.t("chat_blackboard"))
        self.artifacts_label.configure(text=self.controller.t("chat_artifacts"))

    def update_entries(self, entries: list):
        for child in self._list_frame.winfo_children():
            child.destroy()
        if not entries:
            ctk.CTkLabel(self._list_frame, text=self.controller.t("chat_empty"), font=(FONT_UI, 11), text_color=COLORS["muted"]).pack(pady=10)
            self._summary_label.configure(text=self.controller.t("chat_entries", count=0))
            return
        self._summary_label.configure(text=self.controller.t("chat_entries", count=len(entries)))
        for entry in entries[-30:]:
            item = ctk.CTkFrame(self._list_frame, fg_color=COLORS["card_alt"], corner_radius=12)
            item.pack(pady=4, fill="x")
            ctk.CTkLabel(
                item,
                text=f"[{getattr(entry, 'content_type', '?')}] {getattr(entry, 'source_role', '?')}",
                font=(FONT_UI, 11, "bold"),
                text_color=COLORS["text"],
            ).pack(pady=(6, 0), padx=10, anchor="w")
            ctk.CTkLabel(
                item,
                text=getattr(entry, "summary", "")[:60] or self.controller.t("chat_empty"),
                font=(FONT_UI, 10),
                text_color=COLORS["muted"],
                wraplength=260,
                justify="left",
            ).pack(pady=(0, 6), padx=10, anchor="w")

    def update_artifacts(self, artifact_ids: list):
        for child in self._artifact_frame.winfo_children():
            child.destroy()
        if not artifact_ids:
            ctk.CTkLabel(self._artifact_frame, text=self.controller.t("chat_empty"), font=(FONT_UI, 11), text_color=COLORS["muted"]).pack(pady=5)
            return
        for artifact_id in artifact_ids:
            ctk.CTkLabel(self._artifact_frame, text=f"- {artifact_id}", font=(FONT_UI, 11), text_color=COLORS["text"]).pack(
                pady=2, padx=8, anchor="w"
            )


class ChatPanel(ctk.CTkFrame):
    """Primary chat-style workflow panel."""

    def __init__(self, parent, controller, **kwargs):
        super().__init__(parent, fg_color=COLORS["bg"], **kwargs)
        self.controller = controller
        self._on_send_callback: Optional[callable] = None
        self._agent_bubbles: list[AgentBubble] = []
        self._user_bubbles: list[UserBubble] = []
        self._intro_label: ctk.CTkLabel | None = None
        self._input_entry: ctk.CTkEntry | None = None
        self._status_label: ctk.CTkLabel | None = None
        self._model_menu: ctk.CTkOptionMenu | None = None
        self._send_btn: ctk.CTkButton | None = None
        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0, minsize=320)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)

        msg_container = ctk.CTkFrame(self, fg_color="transparent")
        msg_container.grid(row=0, column=0, sticky="nsew", padx=(18, 10), pady=18)
        msg_container.grid_rowconfigure(1, weight=1)
        msg_container.grid_columnconfigure(0, weight=1)

        hero = ctk.CTkFrame(
            msg_container,
            fg_color=COLORS["surface"],
            corner_radius=28,
            border_width=1,
            border_color=COLORS["border"],
        )
        hero.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        ctk.CTkLabel(
            hero,
            text="DR-MMA",
            font=(FONT_UI, 28, "bold"),
            text_color=COLORS["accent"],
        ).pack(anchor="w", padx=18, pady=(18, 4))
        self._intro_label = ctk.CTkLabel(
            hero,
            text="",
            font=(FONT_UI, 13),
            text_color=COLORS["muted"],
            justify="left",
            wraplength=760,
        )
        self._intro_label.pack(anchor="w", padx=18, pady=(0, 18))

        self._msg_scroll = ctk.CTkScrollableFrame(msg_container, fg_color="transparent")
        self._msg_scroll.grid(row=1, column=0, sticky="nsew")

        self._blackboard_sidebar = BlackboardSidebar(self, self.controller)
        self._blackboard_sidebar.grid(row=0, column=1, sticky="nsew", padx=(8, 18), pady=18)

        self._build_input_bar()
        self.add_system_bubble(self.controller.t("chat_ready"), "info")
        self.refresh_language()

    def _build_input_bar(self):
        input_frame = ctk.CTkFrame(
            self,
            fg_color=COLORS["surface"],
            corner_radius=24,
            border_width=1,
            border_color=COLORS["border"],
        )
        input_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=18, pady=(0, 18))
        input_frame.grid_columnconfigure(1, weight=1)
        input_frame.grid_columnconfigure(2, weight=0)

        self._model_label = ctk.CTkLabel(input_frame, text="", font=(FONT_UI, 13), text_color=COLORS["muted"])
        self._model_label.grid(row=0, column=0, padx=(16, 8), pady=(16, 6), sticky="w")
        self._model_var = ctk.StringVar(value="Mock Model (Test)")
        self._model_menu = ctk.CTkOptionMenu(
            input_frame,
            variable=self._model_var,
            values=self.controller.get_available_models(),
            width=240,
            fg_color=COLORS["panel_alt"],
            button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_alt"],
            text_color=COLORS["text"],
        )
        self._model_menu.grid(row=0, column=1, padx=(0, 10), pady=(16, 6), sticky="w")

        self._status_label = ctk.CTkLabel(input_frame, text="", font=(FONT_UI, 12), text_color=COLORS["muted"])
        self._status_label.grid(row=0, column=2, padx=(10, 18), pady=(16, 6), sticky="e")

        self._input_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="",
            font=(FONT_UI, 14),
            height=46,
            corner_radius=16,
            fg_color=COLORS["card_alt"],
            border_color=COLORS["border"],
            text_color=COLORS["text"],
        )
        self._input_entry.grid(row=1, column=0, columnspan=2, sticky="ew", padx=(16, 10), pady=(0, 16))
        self._input_entry.bind("<Return>", lambda _event: self._on_send())

        self._send_btn = ctk.CTkButton(
            input_frame,
            text="",
            font=(FONT_UI, 14, "bold"),
            height=46,
            width=120,
            corner_radius=16,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_alt"],
            text_color="#07213a",
            command=self._on_send,
        )
        self._send_btn.grid(row=1, column=2, padx=(0, 16), pady=(0, 16), sticky="e")

    def refresh_language(self):
        self._intro_label.configure(text=self.controller.t("chat_intro"))
        self._model_label.configure(text=self.controller.t("chat_model"))
        self._input_entry.configure(placeholder_text=self.controller.t("chat_placeholder"))
        self._send_btn.configure(text=self.controller.t("chat_running") if self._send_btn.cget("state") == "disabled" else self.controller.t("chat_send"))
        if self._status_label and not self._status_label.cget("text"):
            self._status_label.configure(text=self.controller.t("status_ready"))
        self._blackboard_sidebar.refresh_language()
        for bubble in self._agent_bubbles:
            bubble.refresh_language()
        for bubble in self._user_bubbles:
            bubble.refresh_language()

    def set_on_send(self, callback):
        self._on_send_callback = callback

    def get_task_text(self) -> str:
        return self._input_entry.get().strip()

    def get_selected_model(self) -> str:
        return self._model_var.get()

    def refresh_models(self):
        models = self.controller.get_available_models()
        self._model_menu.configure(values=models)
        if models and self._model_var.get() not in models:
            self._model_var.set(models[0])

    def set_status(self, text: str, color: str = COLORS["muted"]):
        self._status_label.configure(text=text, text_color=color)

    def set_sending(self, sending: bool):
        if sending:
            self._send_btn.configure(state="disabled", text=self.controller.t("chat_running"))
            self._input_entry.configure(state="disabled")
            self._model_menu.configure(state="disabled")
        else:
            self._send_btn.configure(state="normal", text=self.controller.t("chat_send"))
            self._input_entry.configure(state="normal")
            self._model_menu.configure(state="normal")

    def clear_input(self):
        self._input_entry.delete(0, "end")

    def add_user_bubble(self, text: str):
        bubble = UserBubble(self._msg_scroll, self.controller, text)
        bubble.pack(pady=(0, 10), padx=6, fill="x")
        self._user_bubbles.append(bubble)
        self._scroll_to_bottom()

    def add_agent_bubble(
        self,
        role: str,
        summary: str,
        detail: str = "",
        status: str = "completed",
        claims: list | None = None,
        risks: list | None = None,
        model_name: str = "",
    ):
        bubble = AgentBubble(
            self._msg_scroll,
            self.controller,
            role=role,
            summary=summary,
            detail=detail,
            status=status,
            claims=claims or [],
            risks=risks or [],
            model_name=model_name,
        )
        bubble.pack(pady=(0, 10), padx=6, fill="x")
        self._agent_bubbles.append(bubble)
        self._scroll_to_bottom()

    def add_system_bubble(self, text: str, msg_type: str = "info"):
        SystemBubble(self._msg_scroll, self.controller, text, msg_type).pack(pady=(0, 8), padx=6, fill="x")
        self._scroll_to_bottom()

    def update_blackboard(self, entries: list):
        self._blackboard_sidebar.update_entries(entries)

    def update_artifacts(self, artifact_ids: list):
        self._blackboard_sidebar.update_artifacts(artifact_ids)

    def clear_chat(self):
        self._agent_bubbles.clear()
        self._user_bubbles.clear()
        for child in self._msg_scroll.winfo_children():
            child.destroy()

    def _scroll_to_bottom(self):
        self._msg_scroll._parent_canvas.yview_moveto(1.0)
        self._msg_scroll.update()

    def _on_send(self):
        text = self.get_task_text()
        if not text:
            return
        model_name = self.get_selected_model()
        self.clear_input()
        self.add_user_bubble(text)
        if self._on_send_callback:
            self._on_send_callback(text, model_name)
