"""ToolProtocol unit tests."""

import pytest
from dr_mma.engine.tool_protocol import (
    SchemaType,
    SchemaField,
    SchemaValidationError,
    ToolInputSchema,
    ToolOutputSchema,
    ToolMaterializeResult,
    ToolSettleResult,
    ToolProtocol,
)


class TestSchemaField:
    def test_required_field(self):
        f = SchemaField("name", SchemaType.STRING, required=True)
        assert f.required is True

    def test_to_dict(self):
        f = SchemaField("age", SchemaType.INTEGER, default=0)
        d = f.to_dict()
        assert d["type"] == "integer"
        assert d["default"] == 0

    def test_to_dict_with_allowed_values(self):
        f = SchemaField("status", SchemaType.STRING, allowed_values=["open", "closed"])
        d = f.to_dict()
        assert d["allowed_values"] == ["open", "closed"]

    def test_to_dict_with_items_type(self):
        f = SchemaField("tags", SchemaType.ARRAY, items_type=SchemaType.STRING)
        d = f.to_dict()
        assert d["items_type"] == "string"

    def test_to_dict_with_properties(self):
        prop = SchemaField("id", SchemaType.INTEGER)
        f = SchemaField("config", SchemaType.OBJECT, properties={"id": prop})
        d = f.to_dict()
        assert "properties" in d


class TestToolInputSchema:
    def test_empty_schema(self):
        s = ToolInputSchema()
        assert len(s.fields) == 0

    def test_to_dict(self):
        f = SchemaField("code", SchemaType.STRING)
        s = ToolInputSchema(fields=[f])
        d = s.to_dict()
        assert len(d["fields"]) == 1


class TestToolOutputSchema:
    def test_empty_schema(self):
        s = ToolOutputSchema()
        assert len(s.fields) == 0

    def test_to_dict(self):
        f = SchemaField("result", SchemaType.STRING)
        s = ToolOutputSchema(fields=[f])
        d = s.to_dict()
        assert len(d["fields"]) == 1


class TestToolMaterializeResult:
    def test_valid_result(self):
        r = ToolMaterializeResult(tool_name="test", is_valid=True, resolved_args={"a": 1})
        assert r.is_valid is True

    def test_invalid_result(self):
        r = ToolMaterializeResult(
            tool_name="test",
            is_valid=False,
            validation_errors=["Missing field"],
        )
        assert r.is_valid is False
        assert len(r.validation_errors) == 1

    def test_to_dict(self):
        r = ToolMaterializeResult(tool_name="test", resolved_args={"x": 1})
        d = r.to_dict()
        assert d["tool_name"] == "test"


class TestToolSettleResult:
    def test_valid_result(self):
        r = ToolSettleResult(tool_name="test", output={"ok": True})
        assert r.is_valid is True

    def test_output_truncation(self):
        r = ToolSettleResult(tool_name="test", output="A" * 1000)
        d = r.to_dict()
        assert len(d["output"]) <= 500

    def test_none_output(self):
        r = ToolSettleResult(tool_name="test")
        assert r.output is None


class TestToolProtocolValidateInput:
    def _make_protocol(self, fields=None):
        return ToolProtocol(
            name="test_tool",
            input_schema=ToolInputSchema(fields=fields or []),
        )

    def test_no_fields_any_input_ok(self):
        p = self._make_protocol()
        errors = p.validate_input({})
        assert len(errors) == 0

    def test_missing_required_field(self):
        f = SchemaField("code", SchemaType.STRING, required=True)
        p = self._make_protocol(fields=[f])
        errors = p.validate_input({})
        assert any("Missing required" in e for e in errors)

    def test_wrong_type(self):
        f = SchemaField("age", SchemaType.INTEGER, required=True)
        p = self._make_protocol(fields=[f])
        errors = p.validate_input({"age": "not_int"})
        assert any("Expected integer" in e for e in errors)

    def test_correct_type(self):
        f = SchemaField("code", SchemaType.STRING, required=True)
        p = self._make_protocol(fields=[f])
        errors = p.validate_input({"code": "x=1"})
        assert len(errors) == 0

    def test_string_min_length(self):
        f = SchemaField("name", SchemaType.STRING, min_length=3)
        p = self._make_protocol(fields=[f])
        errors = p.validate_input({"name": "ab"})
        assert any("too short" in e for e in errors)

    def test_string_max_length(self):
        f = SchemaField("name", SchemaType.STRING, max_length=3)
        p = self._make_protocol(fields=[f])
        errors = p.validate_input({"name": "abcd"})
        assert any("too long" in e for e in errors)

    def test_allowed_values(self):
        f = SchemaField("status", SchemaType.STRING, allowed_values=["open", "closed"])
        p = self._make_protocol(fields=[f])
        errors = p.validate_input({"status": "invalid"})
        assert any("not in allowed" in e for e in errors)

    def test_allowed_values_ok(self):
        f = SchemaField("status", SchemaType.STRING, allowed_values=["open", "closed"])
        p = self._make_protocol(fields=[f])
        errors = p.validate_input({"status": "open"})
        assert len(errors) == 0

    def test_array_item_type(self):
        f = SchemaField("tags", SchemaType.ARRAY, items_type=SchemaType.STRING)
        p = self._make_protocol(fields=[f])
        errors = p.validate_input({"tags": ["a", 123]})
        assert any("expected string" in e.lower() for e in errors)

    def test_object_required_property(self):
        prop = SchemaField("id", SchemaType.INTEGER, required=True)
        f = SchemaField("config", SchemaType.OBJECT, properties={"id": prop})
        p = self._make_protocol(fields=[f])
        errors = p.validate_input({"config": {}})
        assert any("missing required property" in e.lower() for e in errors)

    def test_object_property_type(self):
        prop = SchemaField("id", SchemaType.INTEGER, required=True)
        f = SchemaField("config", SchemaType.OBJECT, properties={"id": prop})
        p = self._make_protocol(fields=[f])
        errors = p.validate_input({"config": {"id": "not_int"}})
        assert any("Expected integer" in e for e in errors)

    def test_optional_field_not_required(self):
        f = SchemaField("extra", SchemaType.STRING, required=False)
        p = self._make_protocol(fields=[f])
        errors = p.validate_input({})
        assert len(errors) == 0

    def test_unknown_field_error(self):
        f = SchemaField("code", SchemaType.STRING)
        p = self._make_protocol(fields=[f])
        errors = p.validate_input({"code": "x=1", "unknown": 1})
        assert any("Unknown field" in e for e in errors)


class TestToolProtocolMaterialize:
    def _make_protocol(self, fields=None):
        return ToolProtocol(
            name="test_tool",
            input_schema=ToolInputSchema(fields=fields or []),
        )

    def test_materialize_valid(self):
        f = SchemaField("code", SchemaType.STRING)
        p = self._make_protocol(fields=[f])
        result = p.materialize({"code": "x=1"})
        assert result.is_valid is True
        assert result.resolved_args["code"] == "x=1"

    def test_materialize_applies_default(self):
        f = SchemaField("timeout", SchemaType.INTEGER, default=30)
        p = self._make_protocol(fields=[f])
        result = p.materialize({})
        assert result.resolved_args["timeout"] == 30

    def test_materialize_invalid(self):
        f = SchemaField("code", SchemaType.STRING, required=True)
        p = self._make_protocol(fields=[f])
        result = p.materialize({})
        assert result.is_valid is False
        assert len(result.validation_errors) > 0

    def test_materialize_result_to_dict(self):
        f = SchemaField("code", SchemaType.STRING)
        p = self._make_protocol(fields=[f])
        result = p.materialize({"code": "x=1"})
        d = result.to_dict()
        assert d["tool_name"] == "test_tool"


class TestToolProtocolSettle:
    def _make_protocol(self, out_fields=None):
        return ToolProtocol(
            name="test_tool",
            output_schema=ToolOutputSchema(fields=out_fields or []),
        )

    def test_settle_no_schema(self):
        p = self._make_protocol()
        result = p.settle({"anything": True})
        assert result.is_valid is True

    def test_settle_missing_required_output(self):
        f = SchemaField("stdout", SchemaType.STRING, required=True)
        p = self._make_protocol(out_fields=[f])
        result = p.settle({})
        assert result.is_valid is False
        assert any("missing required" in e.lower() for e in result.validation_errors)

    def test_settle_valid_output(self):
        f = SchemaField("stdout", SchemaType.STRING, required=True)
        p = self._make_protocol(out_fields=[f])
        result = p.settle({"stdout": "hello"})
        assert result.is_valid is True

    def test_settle_wrong_output_type(self):
        f = SchemaField("count", SchemaType.INTEGER, required=True)
        p = self._make_protocol(out_fields=[f])
        result = p.settle({"count": "not_int"})
        assert result.is_valid is False

    def test_settle_non_dict_output(self):
        f = SchemaField("stdout", SchemaType.STRING, required=True)
        p = self._make_protocol(out_fields=[f])
        result = p.settle("plain string")
        # Non-dict output with schema fields: should be valid since we can't check
        assert result.is_valid is True

    def test_settle_result_to_dict(self):
        p = self._make_protocol()
        result = p.settle({"ok": True})
        d = result.to_dict()
        assert d["tool_name"] == "test_tool"


class TestToolProtocolExecute:
    def _make_protocol(self, handler=None, in_fields=None, out_fields=None):
        return ToolProtocol(
            name="exec_tool",
            input_schema=ToolInputSchema(fields=in_fields or []),
            output_schema=ToolOutputSchema(fields=out_fields or []),
            handler=handler,
        )

    def test_execute_success(self):
        p = self._make_protocol(handler=lambda args: {"result": 42})
        result = p.execute({})
        assert result.is_valid is True
        assert result.output == {"result": 42}

    def test_execute_input_validation_fails(self):
        f = SchemaField("code", SchemaType.STRING, required=True)
        p = self._make_protocol(handler=lambda args: "ok", in_fields=[f])
        result = p.execute({})
        assert result.is_valid is False

    def test_execute_no_handler(self):
        p = self._make_protocol()
        result = p.execute({})
        assert result.is_valid is False
        assert any("No handler" in e for e in result.validation_errors)

    def test_execute_handler_raises(self):
        p = self._make_protocol(handler=lambda args: 1 / 0)
        result = p.execute({})
        assert result.is_valid is False
        assert any("Handler error" in e for e in result.validation_errors)

    def test_full_pipeline(self):
        in_f = SchemaField("code", SchemaType.STRING, required=True)
        out_f = SchemaField("stdout", SchemaType.STRING, required=False)
        p = self._make_protocol(
            handler=lambda args: {"stdout": "hello"},
            in_fields=[in_f],
            out_fields=[out_f],
        )
        result = p.execute({"code": "print('hi')"})
        assert result.is_valid is True

    def test_to_dict(self):
        p = self._make_protocol()
        d = p.to_dict()
        assert d["name"] == "exec_tool"
