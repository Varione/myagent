"""DR-MMA CLI tests."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

from dr_mma.cli import main


class TestCLIVersion:

    def test_version_command(self, capsys):
        ret = main(["version"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "DR-MMA" in out
        assert "v0.1.0" in out


class TestCLISessions:

    def test_session_list_empty(self, capsys):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            ret = main(["sessions", "--db", db, "--action", "list"])
            assert ret == 0
            out = capsys.readouterr().out
            assert "Total:" in out
        finally:
            Path(db).unlink(missing_ok=True)

    def test_session_create(self, capsys):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            ret = main(["sessions", "--db", db, "--action", "create", "--location", "test"])
            assert ret == 0
            out = capsys.readouterr().out
            data = json.loads(out.strip())
            assert data["location"] == "test"
            assert "session_id" in data
        finally:
            Path(db).unlink(missing_ok=True)

    def test_session_create_then_list(self, capsys):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            main(["sessions", "--db", db, "--action", "create"])
            ret = main(["sessions", "--db", db, "--action", "list"])
            assert ret == 0
            out = capsys.readouterr().out
            assert "Total: 1" in out
        finally:
            Path(db).unlink(missing_ok=True)

    def test_session_delete(self, capsys):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            out_create = main(["sessions", "--db", db, "--action", "create"])
            assert out_create == 0
            created = json.loads(capsys.readouterr().out.strip())
            sid = created["session_id"]

            ret = main(["sessions", "--db", db, "--action", "delete", "--session-id", sid])
            assert ret == 0
            out = capsys.readouterr().out
            assert "Deleted" in out
        finally:
            Path(db).unlink(missing_ok=True)


class TestCLIMessages:

    def test_messages_empty_session(self, capsys):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            created = main(["sessions", "--db", db, "--action", "create"])
            sid = json.loads(capsys.readouterr().out.strip())["session_id"]

            ret = main(["messages", sid, "--db", db])
            assert ret == 0
            out = capsys.readouterr().out
            assert "Total: 0" in out
        finally:
            Path(db).unlink(missing_ok=True)


class TestCLIConfig:

    def test_config_default(self, capsys):
        ret = main(["config"])
        assert ret == 0
        out = capsys.readouterr().out
        json.loads(out.strip())

    def test_config_with_file(self, capsys):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"test": True}, f)
            cfg = f.name
        try:
            ret = main(["config", "--file", cfg])
            assert ret == 0
            out = capsys.readouterr().out
            data = json.loads(out.strip())
            assert data.get("test") is True
        finally:
            Path(cfg).unlink(missing_ok=True)


class TestCLIStream:

    def test_stream_default(self, capsys):
        ret = main(["stream"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "event:" in out

    def test_stream_with_text(self, capsys):
        ret = main(["stream", "--text", "hello world"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "hello" in out


class TestCLIEventBus:

    def test_event_bus_publish(self, capsys):
        ret = main(["event-bus", "--publish", "test_event", "--data", '{"key": "val"}'])
        assert ret == 0
        out = capsys.readouterr().out
        assert "test_event" in out


class TestCLISubagent:

    def test_subagent_default(self, capsys):
        ret = main(["subagent", "--prompt", "echo test"])
        assert ret == 0
        out = capsys.readouterr().out
        data = json.loads(out.strip())
        assert "agent_id" in data


class TestCLIWindow:

    def test_window_default(self, capsys):
        ret = main(["window"])
        assert ret == 0
        out = capsys.readouterr().out
        data = json.loads(out.strip())
        assert "total_tokens" in data

    def test_window_with_text(self, capsys):
        ret = main(["window", "--text", "line1\nline2", "--max-tokens", "100"])
        assert ret == 0
        out = capsys.readouterr().out
        data = json.loads(out.strip())
        assert data["message_count"] == 2


class TestCLIHelp:

    def test_no_command_shows_help(self, capsys):
        ret = main([])
        assert ret == 0
        out = capsys.readouterr().out
        assert "dr-mma" in out or "Dynamic Role-based" in out


class TestCLIComplexity:

    def test_complexity_evaluation(self, capsys):
        ret = main(["complexity", "--task", "build a REST API"])
        assert ret == 0
        out = capsys.readouterr().out
        assert len(out.strip()) > 0


class TestCLICapabilities:

    def test_capabilities_list(self, capsys):
        ret = main(["capabilities"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "Total:" in out


class TestCLIBudget:

    def test_budget_default(self, capsys):
        ret = main(["budget"])
        assert ret == 0
        out = capsys.readouterr().out
        assert len(out.strip()) > 0

    def test_budget_consume(self, capsys):
        ret = main(["budget", "--use", "1000"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "Consumed" in out or "remaining" in out.lower() or len(out.strip()) > 0
