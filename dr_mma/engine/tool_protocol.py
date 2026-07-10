"""
Tool Protocol — 工具 Schema 校验与 materialize/settle 协议。

Phase X: 定义标准化工具协议，包括输入/输出 Schema 校验、
tool 调用准备（materialize）和结果处理（settle）流程。

核心概念：
- ToolDefinition: 含 input_schema / output_schema 的工具定义
- validate_input(): 基于 Python 内置类型检查的输入校验
- materialize(): 准备工具调用参数
- settle(): 处理工具返回结果并校验输出 schema
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class SchemaType(Enum):
    """Schema 字段类型。"""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class SchemaField:
    """Schema 字段定义。"""

    name: str
    field_type: SchemaType
    required: bool = True
    description: str = ""
    default: Any = None
    allowed_values: Optional[list] = None
    min_length: int = 0
    max_length: int = 10_000
    items_type: Optional[SchemaType] = None  # 用于 array 的元素类型
    properties: Optional[dict[str, SchemaField]] = None  # 用于 object 的子字段

    def to_dict(self) -> dict:
        result: dict[str, Any] = {
            "name": self.name,
            "type": self.field_type.value,
            "required": self.required,
            "description": self.description,
        }
        if self.default is not None:
            result["default"] = self.default
        if self.allowed_values is not None:
            result["allowed_values"] = self.allowed_values
        if self.min_length > 0:
            result["min_length"] = self.min_length
        if self.max_length < 10_000:
            result["max_length"] = self.max_length
        if self.items_type is not None:
            result["items_type"] = self.items_type.value
        if self.properties is not None:
            result["properties"] = {k: v.to_dict() for k, v in self.properties.items()}
        return result


class SchemaValidationError(Exception):
    """Schema 校验失败。"""

    def __init__(self, field_name: str, message: str):
        self.field_name = field_name
        super().__init__(f"Field '{field_name}': {message}")


@dataclass
class ToolInputSchema:
    """工具输入 Schema。"""

    fields: list[SchemaField] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "fields": [f.to_dict() for f in self.fields],
        }


class ToolOutputSchema:
    """工具输出 Schema。"""

    def __init__(self, fields: Optional[list[SchemaField]] = None):
        self.fields = fields or []

    def to_dict(self) -> dict:
        return {
            "fields": [f.to_dict() for f in self.fields],
        }


@dataclass
class ToolMaterializeResult:
    """materialize 的结果。"""

    tool_name: str
    is_valid: bool = True
    resolved_args: dict[str, Any] = field(default_factory=dict)
    validation_errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "is_valid": self.is_valid,
            "resolved_args": self.resolved_args,
            "validation_errors": self.validation_errors,
        }


@dataclass
class ToolSettleResult:
    """settle 的结果。"""

    tool_name: str
    is_valid: bool = True
    output: Any = None
    validation_errors: list[str] = field(default_factory=list)
    settled_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "is_valid": self.is_valid,
            "output": str(self.output)[:500] if self.output is not None else None,
            "validation_errors": self.validation_errors,
        }


class ToolProtocol:
    """
    标准化工具协议：Schema 校验 + materialize/settle 流程。

    Usage:
        protocol = ToolProtocol(
            name="code_execute",
            input_schema=ToolInputSchema(fields=[
                SchemaField("code", SchemaType.STRING, required=True),
                SchemaField("timeout", SchemaType.INTEGER, default=10),
            ]),
            output_schema=ToolOutputSchema(fields=[
                SchemaField("stdout", SchemaType.STRING, required=False),
                SchemaField("variables", SchemaType.OBJECT, required=False),
            ]),
        )

        # Materialize: 准备并校验输入
        mat = protocol.materialize({"code": "x=1"})
        if not mat.is_valid:
            print(mat.validation_errors)

        # Settle: 处理并校验输出
        stl = protocol.settle({"stdout": "", "variables": {}})
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        input_schema: Optional[ToolInputSchema] = None,
        output_schema: Optional[ToolOutputSchema] = None,
        handler: Optional[Callable[[dict[str, Any]], Any]] = None,
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema or ToolInputSchema()
        self.output_schema = output_schema or ToolOutputSchema()
        self.handler = handler

    def validate_input(self, args: dict[str, Any]) -> list[str]:
        """
        校验输入参数是否符合 input_schema。

        Returns:
            错误列表，空列表表示校验通过。
        """
        errors: list[str] = []

        # 检查必需字段
        for fdef in self.input_schema.fields:
            if not fdef.required:
                continue
            if fdef.name not in args:
                errors.append(f"Missing required field '{fdef.name}'")
                continue

            value = args[fdef.name]
            type_errors = self._validate_field_type(fdef, value)
            errors.extend(type_errors)

        # 检查多余字段（可选：如果不想严格限制，可注释掉）
        allowed_names = {fdef.name for fdef in self.input_schema.fields}
        for key in args:
            if key not in allowed_names:
                errors.append(f"Unknown field '{key}'")

        return errors

    def materialize(self, args: dict[str, Any]) -> ToolMaterializeResult:
        """
        准备工具调用：应用默认值、校验输入。

        流程：
        1. 填充缺失的可选字段默认值
        2. 校验输入 schema
        3. 返回 resolved_args 和校验结果
        """
        resolved = dict(args)

        # 应用默认值
        for fdef in self.input_schema.fields:
            if fdef.name not in resolved and fdef.default is not None:
                resolved[fdef.name] = fdef.default

        # 校验
        errors = self.validate_input(resolved)

        return ToolMaterializeResult(
            tool_name=self.name,
            is_valid=len(errors) == 0,
            resolved_args=resolved,
            validation_errors=errors,
        )

    def settle(self, output: Any) -> ToolSettleResult:
        """
        处理工具返回结果并校验输出 schema。

        流程：
        1. 如果 output_schema 有字段定义，校验输出结构
        2. 返回校验结果
        """
        errors: list[str] = []

        if self.output_schema.fields and isinstance(output, dict):
            for fdef in self.output_schema.fields:
                if not fdef.required:
                    continue
                if fdef.name not in output:
                    errors.append(f"Output missing required field '{fdef.name}'")
                    continue

                value = output[fdef.name]
                type_errors = self._validate_field_type(fdef, value)
                errors.extend(type_errors)

        return ToolSettleResult(
            tool_name=self.name,
            is_valid=len(errors) == 0,
            output=output,
            validation_errors=errors,
        )

    def execute(self, args: dict[str, Any]) -> ToolSettleResult:
        """
        完整执行流程：materialize -> handler -> settle。

        Returns:
            ToolSettleResult
        """
        mat = self.materialize(args)
        if not mat.is_valid:
            return ToolSettleResult(
                tool_name=self.name,
                is_valid=False,
                validation_errors=mat.validation_errors,
            )

        if self.handler is None:
            return ToolSettleResult(
                tool_name=self.name,
                is_valid=False,
                validation_errors=["No handler registered for tool"],
            )

        try:
            result = self.handler(mat.resolved_args)
        except Exception as e:
            return ToolSettleResult(
                tool_name=self.name,
                is_valid=False,
                validation_errors=[f"Handler error: {str(e)}"],
            )

        return self.settle(result)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema.to_dict(),
            "output_schema": self.output_schema.to_dict(),
        }

    @staticmethod
    def _validate_field_type(fdef: SchemaField, value: Any) -> list[str]:
        """校验单个字段的类型和约束。"""
        errors: list[str] = []
        type_map = {
            SchemaType.STRING: str,
            SchemaType.INTEGER: int,
            SchemaType.FLOAT: (int, float),
            SchemaType.BOOLEAN: bool,
            SchemaType.ARRAY: list,
            SchemaType.OBJECT: dict,
        }

        # 类型检查
        expected_type = type_map.get(fdef.field_type)
        if expected_type and not isinstance(value, expected_type):
            errors.append(
                f"Expected {fdef.field_type.value}, got {type(value).__name__}"
            )
            return errors

        # 字符串长度检查
        if fdef.field_type == SchemaType.STRING:
            if len(value) < fdef.min_length:
                errors.append(
                    f"String too short: min={fdef.min_length}, got={len(value)}"
                )
            if len(value) > fdef.max_length:
                errors.append(
                    f"String too long: max={fdef.max_length}, got={len(value)}"
                )

        # 枚举值检查
        if fdef.allowed_values is not None and value not in fdef.allowed_values:
            errors.append(
                f"Value '{value}' not in allowed: {fdef.allowed_values}"
            )

        # 数组元素类型检查
        if fdef.field_type == SchemaType.ARRAY and fdef.items_type is not None:
            type_map_items = {
                SchemaType.STRING: str,
                SchemaType.INTEGER: int,
                SchemaType.FLOAT: (int, float),
                SchemaType.BOOLEAN: bool,
            }
            item_type = type_map_items.get(fdef.items_type)
            if item_type:
                for i, item in enumerate(value):
                    if not isinstance(item, item_type):
                        errors.append(
                            f"Array item [{i}] expected {fdef.items_type.value}, "
                            f"got {type(item).__name__}"
                        )

        # 对象子字段检查
        if (
            fdef.field_type == SchemaType.OBJECT
            and fdef.properties is not None
            and isinstance(value, dict)
        ):
            for prop_name, prop_def in fdef.properties.items():
                if prop_def.required and prop_name not in value:
                    errors.append(
                        f"Object missing required property '{prop_name}'"
                    )
                elif prop_name in value:
                    sub_errors = ToolProtocol._validate_field_type(prop_def, value[prop_name])
                    for err in sub_errors:
                        errors.append(f"Property '{prop_name}': {err}")

        return errors
