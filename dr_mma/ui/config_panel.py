"""Model configuration panel."""

from __future__ import annotations

from typing import Callable, Optional

import customtkinter as ctk

from .theme import COLORS, FONT_UI


TYPE_LABELS = {
    "mock": "config_type_mock",
    "local": "config_type_local",
    "remote": "config_type_remote",
}


class ModelCard(ctk.CTkFrame):
    """Compact model summary card."""

    def __init__(
        self,
        parent,
        controller,
        info: dict,
        on_edit: Callable | None = None,
        on_delete: Callable | None = None,
        **kwargs,
    ):
        super().__init__(
            parent,
            corner_radius=18,
            fg_color=COLORS["surface"],
            border_width=1,
            border_color=COLORS["border"],
            **kwargs,
        )
        self.controller = controller
        self.info = info
        self._on_edit = on_edit
        self._on_delete = on_delete
        self._build_ui()

    def _build_ui(self):
        info_frame = ctk.CTkFrame(self, fg_color="transparent")
        info_frame.pack(side="left", fill="x", expand=True, padx=(14, 5), pady=14)

        name_frame = ctk.CTkFrame(info_frame, fg_color="transparent")
        name_frame.pack(fill="x")
        self.name_label = ctk.CTkLabel(name_frame, text=self.info.get("name", "?"), font=(FONT_UI, 15, "bold"), text_color=COLORS["text"])
        self.name_label.pack(side="left")

        model_type = self.info.get("type", "mock")
        tag_color = {
            "mock": "#5f6c7b",
            "local": "#2c7be5",
            "remote": "#db7c26",
        }.get(model_type, "#5f6c7b")
        self.type_label = ctk.CTkLabel(
            name_frame,
            text="",
            font=(FONT_UI, 11),
            fg_color=tag_color,
            text_color="white",
            corner_radius=6,
        )
        self.type_label.pack(side="left", padx=(8, 0))

        self.endpoint_label = ctk.CTkLabel(info_frame, text="", font=(FONT_UI, 12), text_color=COLORS["muted"])
        self.endpoint_label.pack(pady=(4, 0), anchor="w")
        self.model_id_label = ctk.CTkLabel(info_frame, text="", font=(FONT_UI, 11), text_color=COLORS["muted"])
        self.model_id_label.pack(anchor="w")
        self.key_label = ctk.CTkLabel(info_frame, text="", font=(FONT_UI, 11), text_color=COLORS["muted"])
        self.key_label.pack(anchor="w")

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(side="right", padx=(0, 10), pady=10)

        self.edit_btn = ctk.CTkButton(
            btn_frame,
            text="",
            font=(FONT_UI, 12),
            width=70,
            height=30,
            fg_color=COLORS["panel_alt"],
            hover_color=COLORS["sidebar_active"],
            command=lambda: self._on_edit(self.info.get("name", "")) if self._on_edit else None,
        )
        self.edit_btn.pack(side="top", pady=(4, 4))
        self.delete_btn = ctk.CTkButton(
            btn_frame,
            text="",
            font=(FONT_UI, 12),
            width=70,
            height=30,
            fg_color="#7c2d36",
            hover_color="#5c2028",
            command=lambda: self._on_delete(self.info.get("name", "")) if self._on_delete else None,
        )
        self.delete_btn.pack(side="top", pady=(4, 4))
        self.refresh_language()

    def refresh_language(self):
        model_type = self.info.get("type", "mock")
        self.type_label.configure(text=f"  {self.controller.t(TYPE_LABELS.get(model_type, 'config_unknown_type'))}  ")
        endpoint = self.info.get("endpoint", "")
        model_id = self.info.get("model_name", "")
        api_key = self.info.get("api_key", "")
        self.endpoint_label.configure(text=self.controller.t("config_endpoint_label", endpoint=endpoint) if endpoint else "")
        self.model_id_label.configure(text=self.controller.t("config_model_id_label", model_id=model_id) if model_id else "")
        if api_key:
            masked = api_key[:6] + "****" if len(api_key) > 6 else "****"
            self.key_label.configure(text=self.controller.t("config_key_label", key=masked))
        else:
            self.key_label.configure(text="")
        self.edit_btn.configure(text=self.controller.t("config_edit"))
        self.delete_btn.configure(text=self.controller.t("config_delete"))


class ConfigPanel(ctk.CTkFrame):
    """Configure models and storage paths."""

    def __init__(self, parent, controller, on_save: Callable | None = None, **kwargs):
        super().__init__(parent, fg_color=COLORS["bg"], **kwargs)
        self.controller = controller
        self._on_save_cb = on_save
        self._editing_name: Optional[str] = None
        self._model_card_frame: Optional[ctk.CTkScrollableFrame] = None
        self._model_cards: list[ModelCard] = []
        self._build_ui()
        self._refresh_list()

    def _build_ui(self):
        self.title_label = ctk.CTkLabel(self, text="", font=(FONT_UI, 22, "bold"), text_color=COLORS["text"])
        self.title_label.pack(pady=(20, 4), padx=24, anchor="w")
        self.subtitle_label = ctk.CTkLabel(self, text="", font=(FONT_UI, 13), text_color=COLORS["muted"])
        self.subtitle_label.pack(pady=(0, 18), padx=24, anchor="w")

        self.registered_label = ctk.CTkLabel(self, text="", font=(FONT_UI, 15, "bold"), text_color=COLORS["text"])
        self.registered_label.pack(pady=(0, 8), padx=24, anchor="w")
        self._model_card_frame = ctk.CTkScrollableFrame(self, fg_color="transparent", height=200)
        self._model_card_frame.pack(pady=(0, 20), padx=24, fill="x")

        form_frame = ctk.CTkFrame(self, corner_radius=22, fg_color=COLORS["surface"], border_width=1, border_color=COLORS["border"])
        form_frame.pack(pady=(0, 20), padx=24, fill="x")

        self._form_title = ctk.CTkLabel(form_frame, text="", font=(FONT_UI, 17, "bold"), text_color=COLORS["accent"])
        self._form_title.pack(pady=(16, 12), padx=18, anchor="w")

        self._name_entry = self._labeled_entry(form_frame, "config_name", "example-local-model")

        type_frame = ctk.CTkFrame(form_frame, fg_color="transparent")
        type_frame.pack(pady=(0, 10), padx=18, fill="x")
        self._type_label = ctk.CTkLabel(type_frame, text="", width=110, font=(FONT_UI, 14), text_color=COLORS["muted"])
        self._type_label.pack(side="left")
        self._type_var = ctk.StringVar(value="local")
        self._type_menu = ctk.CTkOptionMenu(
            type_frame,
            values=["local", "remote", "mock"],
            variable=self._type_var,
            width=180,
            command=self._on_type_change,
            fg_color=COLORS["panel_alt"],
            button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_alt"],
        )
        self._type_menu.pack(side="left", padx=(8, 0))

        self._endpoint_frame, self._endpoint_entry = self._labeled_entry_with_frame(form_frame, "config_endpoint", "http://127.0.0.1:1234/v1")
        self._model_id_frame, self._model_id_entry = self._labeled_entry_with_frame(form_frame, "config_model_id", "qwopus-27b")
        self._key_frame, self._key_entry = self._labeled_entry_with_frame(form_frame, "config_api_key", "")
        self._key_entry.configure(show="*")
        self._path_entry = self._labeled_entry(form_frame, "config_data_path", "dr_mma_data")
        self.runtime_title = ctk.CTkLabel(form_frame, text="", font=(FONT_UI, 16, "bold"), text_color=COLORS["accent_alt"])
        self.runtime_title.pack(pady=(4, 12), padx=18, anchor="w")
        self._workspace_entry = self._labeled_entry(form_frame, "config_workspace_root", "E:/remote/myagent")
        self._tools_entry = self._labeled_entry(form_frame, "config_tools", "rg, git, pytest")
        self._timeout_entry = self._labeled_entry(form_frame, "config_timeout", "120")

        permission_frame = ctk.CTkFrame(form_frame, fg_color="transparent")
        permission_frame.pack(pady=(0, 10), padx=18, fill="x")
        self._permission_label = ctk.CTkLabel(permission_frame, text="", width=110, font=(FONT_UI, 14), text_color=COLORS["muted"])
        self._permission_label.pack(side="left")
        self._permission_var = ctk.StringVar(value=self.controller.config.get("permission_mode", "workspace_only"))
        self._permission_menu = ctk.CTkOptionMenu(
            permission_frame,
            values=["workspace_only", "full_access"],
            variable=self._permission_var,
            width=180,
            fg_color=COLORS["panel_alt"],
            button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_alt"],
        )
        self._permission_menu.pack(side="left", padx=(8, 0))

        assign_frame = ctk.CTkFrame(form_frame, fg_color="transparent")
        assign_frame.pack(pady=(0, 10), padx=18, fill="x")
        self._assign_label = ctk.CTkLabel(assign_frame, text="", width=110, font=(FONT_UI, 14), text_color=COLORS["muted"])
        self._assign_label.pack(side="left")
        self._assign_var = ctk.StringVar(value=self.controller.config.get("assignment_mode", "primary_preferred"))
        self._assign_menu = ctk.CTkOptionMenu(
            assign_frame,
            values=["primary_preferred", "balanced", "single_model"],
            variable=self._assign_var,
            width=180,
            fg_color=COLORS["panel_alt"],
            button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_alt"],
        )
        self._assign_menu.pack(side="left", padx=(8, 0))

        btn_frame = ctk.CTkFrame(form_frame, fg_color="transparent")
        btn_frame.pack(pady=(6, 16), padx=18, fill="x")

        self._add_btn = ctk.CTkButton(
            btn_frame,
            text="",
            font=(FONT_UI, 14, "bold"),
            height=38,
            corner_radius=14,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_alt"],
            text_color="#07213a",
            command=self._on_submit,
        )
        self._add_btn.pack(side="left")

        self._cancel_edit_btn = ctk.CTkButton(
            btn_frame,
            text="",
            font=(FONT_UI, 13),
            height=38,
            corner_radius=14,
            fg_color=COLORS["panel_alt"],
            hover_color=COLORS["sidebar_active"],
            command=self._cancel_edit,
        )

        self._save_btn = ctk.CTkButton(
            btn_frame,
            text="",
            font=(FONT_UI, 14),
            height=38,
            corner_radius=14,
            fg_color=COLORS["success"],
            hover_color="#37b66d",
            text_color="#072316",
            command=self._on_save,
        )
        self._save_btn.pack(side="left", padx=(10, 0))

        self._status_label = ctk.CTkLabel(btn_frame, text="", font=(FONT_UI, 13), text_color=COLORS["success"])
        self._status_label.pack(side="left", padx=(16, 0))
        self._on_type_change(self._type_var.get())
        self.refresh_language()

    def refresh_language(self):
        self.title_label.configure(text=self.controller.t("config_title"))
        self.subtitle_label.configure(text=self.controller.t("config_subtitle"))
        self.registered_label.configure(text=self.controller.t("config_registered"))
        form_title = self.controller.t("config_edit_model", name=self._editing_name) if self._editing_name else self.controller.t("config_add_model")
        self._form_title.configure(text=form_title)
        self.runtime_title.configure(text=self.controller.t("config_runtime"))
        self._type_label.configure(text=self.controller.t("config_type"))
        self._permission_label.configure(text=self.controller.t("config_permissions"))
        self._assign_label.configure(text=self.controller.t("config_assignment"))
        self._permission_menu.configure(
            values=[
                self.controller.t("config_permissions_workspace"),
                self.controller.t("config_permissions_full"),
            ]
        )
        self._permission_menu.set(
            self.controller.t("config_permissions_workspace")
            if self._permission_var.get() == "workspace_only"
            else self.controller.t("config_permissions_full")
        )
        self._assign_menu.configure(
            values=[
                self.controller.t("config_assignment_primary"),
                self.controller.t("config_assignment_balanced"),
                self.controller.t("config_assignment_single"),
            ]
        )
        assign_label = {
            "primary_preferred": self.controller.t("config_assignment_primary"),
            "balanced": self.controller.t("config_assignment_balanced"),
            "single_model": self.controller.t("config_assignment_single"),
        }.get(self._assign_var.get(), self.controller.t("config_assignment_primary"))
        self._assign_menu.set(assign_label)
        self._add_btn.configure(text=self.controller.t("config_update_btn") if self._editing_name else self.controller.t("config_add_btn"))
        self._cancel_edit_btn.configure(text=self.controller.t("config_cancel_edit"))
        self._save_btn.configure(text=self.controller.t("config_save_paths"))
        for widget in (
            self._name_entry,
            self._path_entry,
            self._workspace_entry,
            self._tools_entry,
            self._timeout_entry,
        ):
            widget._i18n_label.configure(text=self.controller.t(widget._i18n_key))
        for frame in (self._endpoint_frame, self._model_id_frame, self._key_frame):
            frame._i18n_label.configure(text=self.controller.t(frame._i18n_key))
        for card in self._model_cards:
            card.refresh_language()

    def _labeled_entry(self, parent, label_key: str, placeholder: str):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(pady=(0, 10), padx=18, fill="x")
        label = ctk.CTkLabel(frame, text=self.controller.t(label_key), width=110, font=(FONT_UI, 14), text_color=COLORS["muted"])
        label.pack(side="left")
        entry = ctk.CTkEntry(frame, placeholder_text=placeholder, fg_color=COLORS["card_alt"], border_color=COLORS["border"], text_color=COLORS["text"])
        entry.pack(side="left", padx=(8, 0), fill="x", expand=True)
        entry._i18n_label = label
        entry._i18n_key = label_key
        return entry

    def _labeled_entry_with_frame(self, parent, label_key: str, placeholder: str):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(pady=(0, 10), padx=18, fill="x")
        label = ctk.CTkLabel(frame, text=self.controller.t(label_key), width=110, font=(FONT_UI, 14), text_color=COLORS["muted"])
        label.pack(side="left")
        entry = ctk.CTkEntry(frame, placeholder_text=placeholder, fg_color=COLORS["card_alt"], border_color=COLORS["border"], text_color=COLORS["text"])
        entry.pack(side="left", padx=(8, 0), fill="x", expand=True)
        frame._i18n_label = label
        frame._i18n_key = label_key
        return frame, entry

    def _on_type_change(self, value: str):
        if value == "mock":
            self._endpoint_frame.pack_forget()
            self._model_id_frame.pack_forget()
            self._key_frame.pack_forget()
            return
        self._endpoint_frame.pack(pady=(0, 10), padx=18, fill="x")
        self._model_id_frame.pack(pady=(0, 10), padx=18, fill="x")
        if value == "remote":
            self._key_frame.pack(pady=(0, 10), padx=18, fill="x")
        else:
            self._key_frame.pack_forget()

    def _populate_form(self, info: dict):
        self._editing_name = info.get("name", "")
        self._name_entry.delete(0, "end")
        self._name_entry.insert(0, info.get("name", ""))
        self._type_var.set(info.get("type", "local"))
        self._endpoint_entry.delete(0, "end")
        self._endpoint_entry.insert(0, info.get("endpoint", ""))
        self._model_id_entry.delete(0, "end")
        self._model_id_entry.insert(0, info.get("model_name", ""))
        self._key_entry.delete(0, "end")
        self._key_entry.insert(0, info.get("api_key", ""))
        self._workspace_entry.delete(0, "end")
        self._workspace_entry.insert(0, self.controller.config.get("workspace_root", ""))
        self._tools_entry.delete(0, "end")
        self._tools_entry.insert(0, ", ".join(self.controller.config.get("allowed_tools", [])))
        self._timeout_entry.delete(0, "end")
        self._timeout_entry.insert(0, str(self.controller.config.get("timeout_seconds", 120)))
        self._on_type_change(self._type_var.get())
        self.refresh_language()
        self._cancel_edit_btn.pack(side="left", padx=(10, 0))

    def _cancel_edit(self):
        self._editing_name = None
        self._cancel_edit_btn.pack_forget()
        for entry in (self._name_entry, self._endpoint_entry, self._model_id_entry, self._key_entry):
            entry.delete(0, "end")
        self.refresh_language()

    def _collect_form(self) -> dict:
        model_type = self._type_var.get()
        return {
            "name": self._name_entry.get().strip(),
            "type": model_type,
            "endpoint": self._endpoint_entry.get().strip() if model_type != "mock" else "",
            "model_name": self._model_id_entry.get().strip() if model_type != "mock" else "",
            "api_key": self._key_entry.get().strip() if model_type == "remote" else "",
        }

    def _on_submit(self):
        data = self._collect_form()
        if not data["name"]:
            self._set_status(self.controller.t("config_name_required"), COLORS["warning"])
            return
        try:
            self.controller.register_model(
                data["name"],
                endpoint=data.get("endpoint", ""),
                api_key=data.get("api_key", ""),
                model_name=data.get("model_name", ""),
                model_type=data["type"],
            )
            self._set_status(self.controller.t("config_saved", name=data["name"]), COLORS["success"])
            self._refresh_list()
            if self._editing_name:
                self._cancel_edit()
            else:
                for entry in (self._name_entry, self._endpoint_entry, self._model_id_entry, self._key_entry):
                    entry.delete(0, "end")
            if self._on_save_cb:
                self._on_save_cb()
        except Exception as exc:
            self._set_status(self.controller.t("config_save_failed", error=exc), COLORS["danger"])

    def _on_save(self):
        path = self._path_entry.get().strip()
        workspace_root = self._workspace_entry.get().strip()
        tools_text = self._tools_entry.get().strip()
        timeout_text = self._timeout_entry.get().strip()
        if path:
            path = path.replace("\\", "/")
            self.controller.config["blackboard_path"] = f"{path}/blackboard.jsonl"
            self.controller.config["artifact_path"] = f"{path}/artifacts.jsonl"
            self.controller.config["decision_path"] = f"{path}/decisions.jsonl"
        if workspace_root:
            self.controller.config["workspace_root"] = workspace_root
        self.controller.config["allowed_tools"] = [item.strip() for item in tools_text.split(",") if item.strip()]
        self.controller.config["timeout_seconds"] = int(timeout_text) if timeout_text.isdigit() else 120
        permission_label = self._permission_menu.get()
        self.controller.config["permission_mode"] = (
            "full_access" if permission_label == self.controller.t("config_permissions_full") else "workspace_only"
        )
        assign_label = self._assign_menu.get()
        if assign_label == self.controller.t("config_assignment_balanced"):
            self.controller.config["assignment_mode"] = "balanced"
        elif assign_label == self.controller.t("config_assignment_single"):
            self.controller.config["assignment_mode"] = "single_model"
        else:
            self.controller.config["assignment_mode"] = "primary_preferred"
        self.controller.save_config()
        self._set_status(self.controller.t("config_paths_saved"), COLORS["success"])
        if self._on_save_cb:
            self._on_save_cb()

    def _edit_model(self, name: str):
        for info in self.controller.get_model_infos():
            if info.get("name") == name:
                self._populate_form(info)
                break

    def _delete_model(self, name: str):
        if name == "Mock Model (Test)":
            self._set_status(self.controller.t("config_default_protected"), COLORS["warning"])
            return
        self.controller.remove_model(name)
        self._set_status(self.controller.t("config_removed", name=name), COLORS["warning"])
        self._refresh_list()
        if self._on_save_cb:
            self._on_save_cb()

    def _refresh_list(self):
        self._model_cards.clear()
        if not self._model_card_frame:
            return
        for child in self._model_card_frame.winfo_children():
            child.destroy()

        infos = self.controller.get_model_infos()
        if not infos:
            ctk.CTkLabel(
                self._model_card_frame,
                text=self.controller.t("config_no_models"),
                font=(FONT_UI, 13),
                text_color=COLORS["muted"],
            ).pack(pady=10)
            return

        for info in infos:
            card = ModelCard(
                self._model_card_frame,
                self.controller,
                info=info,
                on_edit=self._edit_model,
                on_delete=self._delete_model,
            )
            card.pack(pady=(0, 8), fill="x")
            self._model_cards.append(card)
        self._workspace_entry.delete(0, "end")
        self._workspace_entry.insert(0, self.controller.config.get("workspace_root", ""))
        self._tools_entry.delete(0, "end")
        self._tools_entry.insert(0, ", ".join(self.controller.config.get("allowed_tools", [])))
        self._timeout_entry.delete(0, "end")
        self._timeout_entry.insert(0, str(self.controller.config.get("timeout_seconds", 120)))
        self._permission_var.set(self.controller.config.get("permission_mode", "workspace_only"))
        self._assign_var.set(self.controller.config.get("assignment_mode", "primary_preferred"))
        self.refresh_language()

    def _set_status(self, text: str, color: str = COLORS["muted"]):
        self._status_label.configure(text=text, text_color=color)
        self.after(4000, lambda: self._status_label.configure(text=""))
