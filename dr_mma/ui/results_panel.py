"""Results display panels for workflow output."""

from __future__ import annotations

import customtkinter as ctk

from .theme import COLORS, FONT_MONO, FONT_UI


class SubtaskTab(ctk.CTkFrame):
    """Single subtask result view."""

    def __init__(self, parent, controller, subtask_id: str, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.controller = controller
        self.subtask_id = subtask_id
        self._build_ui()

    def _build_ui(self):
        self.worker_label = ctk.CTkLabel(self, text="", font=(FONT_UI, 15, "bold"), text_color=COLORS["text"])
        self.worker_label.pack(pady=(15, 6), padx=15, anchor="w")
        self.worker_text = ctk.CTkTextbox(self, height=120, font=(FONT_UI, 13), fg_color=COLORS["surface"], text_color=COLORS["text"])
        self.worker_text.pack(pady=(0, 15), padx=15, fill="x")
        self.worker_text.insert("1.0", "")
        self.worker_text.configure(state="disabled")

        self.critic_label = ctk.CTkLabel(self, text="", font=(FONT_UI, 15, "bold"), text_color=COLORS["text"])
        self.critic_label.pack(pady=(5, 6), padx=15, anchor="w")
        self.critic_text = ctk.CTkTextbox(self, height=100, font=(FONT_UI, 13), fg_color=COLORS["surface"], text_color=COLORS["text"])
        self.critic_text.pack(pady=(0, 15), padx=15, fill="x")
        self.critic_text.insert("1.0", "")
        self.critic_text.configure(state="disabled")

        self.verifier_label = ctk.CTkLabel(self, text="", font=(FONT_UI, 15, "bold"), text_color=COLORS["text"])
        self.verifier_label.pack(pady=(5, 6), padx=15, anchor="w")
        self.verifier_text = ctk.CTkTextbox(self, height=80, font=(FONT_UI, 13), fg_color=COLORS["surface"], text_color=COLORS["text"])
        self.verifier_text.pack(pady=(0, 15), padx=15, fill="x")
        self.verifier_text.insert("1.0", "")
        self.verifier_text.configure(state="disabled")

        self.contract_label = ctk.CTkLabel(self, text="", font=(FONT_UI, 15, "bold"), text_color=COLORS["text"])
        self.contract_label.pack(pady=(5, 6), padx=15, anchor="w")
        self.contract_text = ctk.CTkTextbox(self, height=120, font=(FONT_MONO, 12), fg_color=COLORS["surface"], text_color=COLORS["text"])
        self.contract_text.pack(pady=(0, 15), padx=15, fill="x")
        self.contract_text.insert("1.0", "")
        self.contract_text.configure(state="disabled")

        self.response_label = ctk.CTkLabel(self, text="", font=(FONT_UI, 15, "bold"), text_color=COLORS["text"])
        self.response_label.pack(pady=(5, 6), padx=15, anchor="w")
        self.response_text = ctk.CTkTextbox(self, height=120, font=(FONT_MONO, 12), fg_color=COLORS["surface"], text_color=COLORS["text"])
        self.response_text.pack(pady=(0, 15), padx=15, fill="x")
        self.response_text.insert("1.0", "")
        self.response_text.configure(state="disabled")
        self.refresh_language()

    def refresh_language(self):
        self.worker_label.configure(text=self.controller.t("results_worker"))
        self.critic_label.configure(text=self.controller.t("results_critic"))
        self.verifier_label.configure(text=self.controller.t("results_verifier"))
        self.contract_label.configure(text=self.controller.t("results_contract"))
        self.response_label.configure(text=self.controller.t("results_response"))
        if not self.worker_text.get("1.0", "end-1c").strip():
            self.set_content(worker=self.controller.t("results_wait_worker"))
        if not self.critic_text.get("1.0", "end-1c").strip():
            self.set_content(critic=self.controller.t("results_wait_critic"))
        if not self.verifier_text.get("1.0", "end-1c").strip():
            self.set_content(verifier=self.controller.t("results_wait_verifier"))
        if not self.contract_text.get("1.0", "end-1c").strip():
            self.set_content(contract=self.controller.t("results_wait_contract"))
        if not self.response_text.get("1.0", "end-1c").strip():
            self.set_content(response=self.controller.t("results_wait_response"))

    def set_content(self, worker: str = "", critic: str = "", verifier: str = "", contract: str = "", response: str = ""):
        if worker:
            self.worker_text.configure(state="normal")
            self.worker_text.delete("1.0", "end")
            self.worker_text.insert("1.0", worker)
            self.worker_text.configure(state="disabled")
        if critic:
            self.critic_text.configure(state="normal")
            self.critic_text.delete("1.0", "end")
            self.critic_text.insert("1.0", critic)
            self.critic_text.configure(state="disabled")
        if verifier:
            self.verifier_text.configure(state="normal")
            self.verifier_text.delete("1.0", "end")
            self.verifier_text.insert("1.0", verifier)
            self.verifier_text.configure(state="disabled")
        if contract:
            self.contract_text.configure(state="normal")
            self.contract_text.delete("1.0", "end")
            self.contract_text.insert("1.0", contract)
            self.contract_text.configure(state="disabled")
        if response:
            self.response_text.configure(state="normal")
            self.response_text.delete("1.0", "end")
            self.response_text.insert("1.0", response)
            self.response_text.configure(state="disabled")


class ResultsPanel(ctk.CTkFrame):
    """Shows workflow summary, final output, and per-subtask results."""

    def __init__(self, parent, controller, **kwargs):
        super().__init__(parent, fg_color=COLORS["bg"], **kwargs)
        self.controller = controller
        self.tab_view: ctk.CTkTabview | None = None
        self.tabs: dict[str, SubtaskTab] = {}
        self._summary_box = None
        self._final_output_box = None
        self._placeholder_tab_name = ""
        self._build_ui()

    def _build_ui(self):
        self.title_label = ctk.CTkLabel(self, text="", font=(FONT_UI, 22, "bold"), text_color=COLORS["text"])
        self.title_label.pack(pady=(20, 6), padx=22, anchor="w")

        summary_frame = ctk.CTkFrame(self, fg_color=COLORS["surface"], corner_radius=20, border_width=1, border_color=COLORS["border"])
        summary_frame.pack(pady=(0, 15), padx=22, fill="x")
        self.summary_label = ctk.CTkLabel(summary_frame, text="", font=(FONT_UI, 15, "bold"), text_color=COLORS["accent"])
        self.summary_label.pack(pady=(12, 6), padx=16, anchor="w")
        self._summary_box = ctk.CTkTextbox(summary_frame, height=90, font=(FONT_MONO, 12), fg_color=COLORS["surface"], text_color=COLORS["text"])
        self._summary_box.pack(pady=(0, 12), padx=16, fill="x")
        self._summary_box.configure(state="disabled")

        final_frame = ctk.CTkFrame(self, fg_color=COLORS["surface"], corner_radius=20, border_width=1, border_color=COLORS["border"])
        final_frame.pack(pady=(0, 15), padx=22, fill="x")
        self.final_label = ctk.CTkLabel(final_frame, text="", font=(FONT_UI, 15, "bold"), text_color=COLORS["accent_alt"])
        self.final_label.pack(pady=(12, 6), padx=16, anchor="w")
        self._final_output_box = ctk.CTkTextbox(final_frame, height=170, font=(FONT_UI, 13), fg_color=COLORS["surface"], text_color=COLORS["text"])
        self._final_output_box.pack(pady=(0, 12), padx=16, fill="x")
        self._final_output_box.configure(state="disabled")

        self.tab_view = ctk.CTkTabview(
            self,
            fg_color=COLORS["surface"],
            segmented_button_fg_color=COLORS["panel_alt"],
            segmented_button_selected_color=COLORS["accent"],
            segmented_button_selected_hover_color=COLORS["accent_alt"],
            text_color=COLORS["text"],
        )
        self.tab_view.pack(pady=(0, 22), padx=22, fill="both", expand=True)
        self.refresh_language()
        self._show_placeholder_tab()

    def refresh_language(self):
        self.title_label.configure(text=self.controller.t("results_title"))
        self.summary_label.configure(text=self.controller.t("results_summary"))
        self.final_label.configure(text=self.controller.t("results_final"))
        if not self.tabs:
            self.set_summary(self.controller.t("results_waiting_start"))
            self.set_final_output(self.controller.t("results_waiting_done"))

    def _show_placeholder_tab(self):
        self._placeholder_tab_name = self.controller.t("results_waiting_tab")
        try:
            self.tab_view.add(self._placeholder_tab_name)
            placeholder = ctk.CTkLabel(
                self.tab_view.tab(self._placeholder_tab_name),
                text=self.controller.t("results_waiting_tab_body"),
                font=(FONT_UI, 14),
                text_color=COLORS["muted"],
            )
            placeholder.pack(pady=40)
        except Exception:
            pass

    def reset(self):
        self.set_summary(self.controller.t("results_waiting_start"))
        self.set_final_output(self.controller.t("results_waiting_done"))
        if self.tab_view:
            for tab_name in list(self.tabs.keys()):
                try:
                    self.tab_view.delete(self.controller.t("results_subtask_tab", suffix=tab_name[-4:]))
                except Exception:
                    pass
            try:
                self.tab_view.delete(self._placeholder_tab_name)
            except Exception:
                pass
        self.tabs.clear()
        self._show_placeholder_tab()

    def add_subtask(self, subtask_id: str):
        try:
            self.tab_view.delete(self._placeholder_tab_name)
        except Exception:
            pass
        if subtask_id in self.tabs:
            return
        tab_name = self.controller.t("results_subtask_tab", suffix=subtask_id[-4:])
        self.tab_view.add(tab_name)
        tab_content = SubtaskTab(self.tab_view.tab(tab_name), self.controller, subtask_id)
        tab_content.pack(fill="both", expand=True)
        self.tabs[subtask_id] = tab_content

    def set_subtask_content(
        self,
        subtask_id: str,
        worker: str = "",
        critic: str = "",
        verifier: str = "",
        contract: str = "",
        response: str = "",
    ):
        tab = self.tabs.get(subtask_id)
        if tab:
            tab.set_content(
                worker or self.controller.t("results_wait_worker"),
                critic or self.controller.t("results_wait_critic"),
                verifier or self.controller.t("results_wait_verifier"),
                contract or self.controller.t("results_wait_contract"),
                response or self.controller.t("results_wait_response"),
            )

    def set_summary(self, text: str):
        self._summary_box.configure(state="normal")
        self._summary_box.delete("1.0", "end")
        self._summary_box.insert("1.0", text)
        self._summary_box.configure(state="disabled")

    def set_final_output(self, text: str):
        self._final_output_box.configure(state="normal")
        self._final_output_box.delete("1.0", "end")
        self._final_output_box.insert("1.0", text)
        self._final_output_box.configure(state="disabled")
