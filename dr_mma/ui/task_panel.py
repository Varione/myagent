"""Legacy task launcher panel kept for compatibility tests."""

from typing import Optional

import customtkinter as ctk


class TaskPanel(ctk.CTkFrame):
    """Collects task text and selected model with the same send semantics as ChatPanel."""

    def __init__(self, parent, controller, **kwargs):
        super().__init__(parent, **kwargs)
        self.controller = controller
        self._callback_on_execute: Optional[callable] = None
        self._build_ui()

    def _build_ui(self):
        ctk.CTkLabel(self, text="Task Input", font=("", 20, "bold")).pack(
            pady=(20, 5), padx=20, anchor="w"
        )
        ctk.CTkLabel(
            self,
            text="Describe the task that DR-MMA should plan, execute, review, and verify.",
            font=("", 13),
            text_color="gray",
        ).pack(pady=(0, 15), padx=20, anchor="w")

        model_frame = ctk.CTkFrame(self, fg_color="transparent")
        model_frame.pack(pady=(0, 10), padx=20, fill="x")
        ctk.CTkLabel(model_frame, text="Model:", font=("", 14)).pack(side="left")

        models = self.controller.get_available_models()
        default_model = models[0] if models else "Mock Model (Test)"
        self.model_var = ctk.StringVar(value=default_model)
        self.model_menu = ctk.CTkOptionMenu(
            model_frame,
            variable=self.model_var,
            values=models,
            width=220,
        )
        self.model_menu.pack(side="left", padx=(10, 0))

        ctk.CTkButton(model_frame, text="Refresh", width=80, command=self._refresh_models).pack(
            side="left", padx=(10, 0)
        )

        self.task_text = ctk.CTkTextbox(self, height=220, font=("", 14))
        self.task_text.pack(pady=(0, 15), padx=20, fill="x")
        self.task_text.insert(
            "1.0",
            "Analyze the requirement and design a technical solution for a multi-user collaborative editor "
            "with real-time sync, version history, and permissions.",
        )

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=(0, 20), padx=20, fill="x")

        self.execute_btn = ctk.CTkButton(
            btn_frame,
            text="Run Workflow",
            font=("", 15, "bold"),
            height=40,
            command=self._on_execute,
        )
        self.execute_btn.pack(side="left")

        self.cancel_btn = ctk.CTkButton(
            btn_frame,
            text="Cancel",
            font=("", 14),
            height=40,
            fg_color="#888888",
            hover_color="#666666",
            state="disabled",
            command=self._on_cancel,
        )
        self.cancel_btn.pack(side="left", padx=(15, 0))

        self.status_label = ctk.CTkLabel(btn_frame, text="Ready", font=("", 13), text_color="gray")
        self.status_label.pack(side="right")

    def set_on_execute(self, callback):
        self._callback_on_execute = callback

    def set_status(self, text: str, color: str = "gray"):
        self.status_label.configure(text=text, text_color=color)

    def set_executing(self, executing: bool):
        if executing:
            self.execute_btn.configure(state="disabled")
            self.cancel_btn.configure(state="normal")
            self.task_text.configure(state="disabled")
            self.set_status("Running...", "#f0a000")
        else:
            self.execute_btn.configure(state="normal")
            self.cancel_btn.configure(state="disabled")
            self.task_text.configure(state="normal")

    def get_task_text(self) -> str:
        return self.task_text.get("1.0", "end-1c").strip()

    def get_selected_model(self) -> str:
        return self.model_var.get()

    def _refresh_models(self):
        models = self.controller.get_available_models()
        self.model_menu.configure(values=models)
        if models and self.model_var.get() not in models:
            self.model_var.set(models[0])

    def _on_execute(self):
        task = self.get_task_text()
        if not task:
            self.set_status("Task text is required", "#ff4444")
            return
        self.set_executing(True)
        if self._callback_on_execute:
            self._callback_on_execute(task, self.get_selected_model())

    def _on_cancel(self):
        self.controller.cancel()
        self.set_executing(False)
        self.set_status("Cancelled", "#888888")
