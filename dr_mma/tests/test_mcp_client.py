"""MCP Client unit and integration tests."""

import json
import os
import subprocess
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from threading import Thread
from unittest.mock import MagicMock, patch

import pytest

from dr_mma.engine.mcp_client import (
    MCPClient,
    MCPConnectionError,
    MCPCapabilities,
    MCPErrors,
    MCPProtocolError,
    MCPToolDefinition,
    MCPToolResult,
    MCPTimedOutError,
    MCPTransportMode,
)

class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _wait_for_http(url: str, timeout: float = 5.0) -> bool:
    """Wait for an HTTP server to become reachable."""
    import time as _time
    import urllib.request as _urllib
    deadline = _time.time() + timeout
    while _time.time() < deadline:
        try:
            with _urllib.urlopen(url, timeout=1.0):
                return True
        except Exception:
            _time.sleep(0.1)
    return False


def _make_stdio_server_script() -> str:
    """Generate a minimal MCP stdio server script for testing."""
    return """
import sys, json

def handle_request(msg):
    method = msg.get("method", "")
    req_id = msg.get("id", 0)

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "test-server", "version": "0.1.0"},
                "capabilities": {"tools": {}},
            },
        }
    elif method == "notifications/initialized":
        return None
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {"name": "echo", "description": "Echo back input", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}},
                    {"name": "calc", "description": "Calculator", "inputSchema": {"type": "object", "properties": {"expr": {"type": "string"}}}},
                ],
            },
        }
    elif method == "tools/call":
        tool_name = msg.get("params", {}).get("name", "")
        args = msg.get("params", {}).get("arguments", {})
        if tool_name == "echo":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": args.get("text", "")}],
                    "isError": False,
                },
            }
        elif tool_name == "calc":
            try:
                val = eval(args.get("expr", "0"))
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": str(val)}],
                        "isError": False,
                    },
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": str(e)}],
                        "isError": True,
                    },
                }
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }
    elif method.startswith("notifications/"):
        return None
    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Unknown method: {method}"},
        }

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle_request(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
"""


def _start_stdio_server() -> tuple[str, subprocess.Popen]:
    """Start a real stdio MCP server for integration tests."""
    script = _make_stdio_server_script()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        f.flush()
        path = f.name

    proc = subprocess.Popen(
        [sys.executable, path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0,
    )
    return path, proc


def _start_http_server(port: int) -> _ThreadingHTTPServer:
    """Start a minimal HTTP MCP server for integration tests."""

    class MCPRouter(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # suppress logs

        def do_HEAD(self):
            self.send_response(200)
            self.end_headers()

        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8") if length else ""
            try:
                msg = json.loads(body)
            except json.JSONDecodeError:
                self.send_response(400)
                self.end_headers()
                return

            method = msg.get("method", "")
            req_id = msg.get("id", 0)

            if method == "initialize":
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {"name": "http-test-server", "version": "0.1.0"},
                        "capabilities": {"tools": {}},
                    },
                }
            elif method == "tools/list":
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": [
                            {
                                "name": "http_echo",
                                "description": "HTTP echo tool",
                                "inputSchema": {"type": "object"},
                            },
                        ],
                    },
                }
            elif method == "tools/call":
                args = msg.get("params", {}).get("arguments", {})
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(args)}],
                        "isError": False,
                    },
                }
            elif method.startswith("notifications/"):
                resp = None
            else:
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Unknown method: {method}"},
                }

            if resp is not None:
                raw = json.dumps(resp).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            else:
                self.send_response(204)
                self.end_headers()

    server = _ThreadingHTTPServer(("127.0.0.1", port), MCPRouter)
    return server


class TestMCPToolDefinition:
    def test_from_dict(self):
        data = {
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}},
        }
        t = MCPToolDefinition.from_dict(data)
        assert t.name == "test_tool"
        assert t.description == "A test tool"
        assert t.inputSchema["type"] == "object"

    def test_to_dict(self):
        t = MCPToolDefinition(name="echo", description="Echo tool")
        d = t.to_dict()
        assert d["name"] == "echo"
        assert d["description"] == "Echo tool"
        assert d["inputSchema"] == {}

    def test_defaults(self):
        t = MCPToolDefinition(name="x")
        assert t.description == ""
        assert t.inputSchema == {}


class TestMCPToolResult:
    def test_success(self):
        r = MCPToolResult(success=True, content=[{"type": "text", "text": "ok"}])
        assert r.success is True
        assert r.isError is False

    def test_error(self):
        r = MCPToolResult(success=False, isError=True)
        assert r.success is False
        assert r.isError is True

    def test_to_dict(self):
        r = MCPToolResult(success=True, content=[{"t": "x"}], metadata={"k": 1})
        d = r.to_dict()
        assert d["success"] is True
        assert d["metadata"]["k"] == 1

    def test_empty_content(self):
        r = MCPToolResult()
        assert r.content == []


class TestMCPClientDataClasses:
    def test_mcp_errors_constants(self):
        assert MCPErrors.PARSE_ERROR == -32700
        assert MCPErrors.INTERNAL_ERROR == -32603

    def test_mcp_capabilities_enum(self):
        assert MCPCapabilities.TOOLS.value == "tools"
        assert MCPCapabilities.RESSOURCES.value if hasattr(MCPCapabilities, "RESSOURCES") else True


class TestMCPClientStdioIntegration:
    """Integration tests against a real stdio MCP server."""

    @pytest.fixture(autouse=True)
    def _server(self):
        self._path, self._proc = _start_stdio_server()
        yield
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()
            self._proc.wait()
        finally:
            if os.path.exists(self._path):
                os.unlink(self._path)

    def test_connect_local_and_disconnect(self):
        client = MCPClient(connect_timeout=5)
        client.connect_local(sys.executable, [self._path])
        assert client.is_connected is True
        assert client.mode == MCPTransportMode.STDIO
        client.disconnect()
        assert client.is_connected is False

    def test_server_info(self):
        client = MCPClient(connect_timeout=5)
        client.connect_local(sys.executable, [self._path])
        info = client.server_info
        assert info["name"] == "test-server"
        client.disconnect()

    def test_server_capabilities(self):
        client = MCPClient(connect_timeout=5)
        client.connect_local(sys.executable, [self._path])
        caps = client.server_capabilities
        assert "tools" in caps
        assert client.has_capability("tools") is True
        client.disconnect()

    def test_discover_tools(self):
        client = MCPClient(connect_timeout=5)
        client.connect_local(sys.executable, [self._path])
        tools = client.discover_tools()
        names = [t.name for t in tools]
        assert "echo" in names
        assert "calc" in names
        client.disconnect()

    def test_discover_tools_caching(self):
        client = MCPClient(connect_timeout=5)
        client.connect_local(sys.executable, [self._path])
        tools1 = client.discover_tools()
        tools2 = client.discover_tools()
        assert len(tools1) == len(tools2)
        # Force refresh
        tools3 = client.discover_tools(force_refresh=True)
        assert len(tools3) == len(tools1)
        client.disconnect()

    def test_call_tool_echo(self):
        client = MCPClient(connect_timeout=5, call_timeout=10)
        client.connect_local(sys.executable, [self._path])
        result = client.call_tool("echo", {"text": "hello"})
        assert result.success is True
        assert len(result.content) > 0
        assert result.content[0]["text"] == "hello"
        client.disconnect()

    def test_call_tool_calc(self):
        client = MCPClient(connect_timeout=5, call_timeout=10)
        client.connect_local(sys.executable, [self._path])
        result = client.call_tool("calc", {"expr": "2+3*4"})
        assert result.success is True
        assert result.content[0]["text"] == "14"
        client.disconnect()

    def test_call_tool_unknown_raises(self):
        client = MCPClient(connect_timeout=5, call_timeout=10)
        client.connect_local(sys.executable, [self._path])
        with pytest.raises(MCPProtocolError):
            client.call_tool("nonexistent", {})
        client.disconnect()

    def test_double_connect_raises(self):
        client = MCPClient(connect_timeout=5)
        client.connect_local(sys.executable, [self._path])
        with pytest.raises(MCPConnectionError):
            client.connect_local(sys.executable, [self._path])
        client.disconnect()

    def test_connect_invalid_command(self):
        client = MCPClient(connect_timeout=5)
        with pytest.raises(MCPConnectionError):
            client.connect_local("__nonexistent_cmd__", [])

    def test_call_tool_not_connected_raises(self):
        client = MCPClient()
        with pytest.raises(MCPConnectionError):
            client.call_tool("echo", {})

    def test_discover_tools_not_connected_raises(self):
        client = MCPClient()
        with pytest.raises(MCPConnectionError):
            client.discover_tools()


class TestMCPClientHttpIntegration:
    """Integration tests against a real HTTP MCP server."""

    @pytest.fixture(autouse=True)
    def _server(self):
        self._server = _start_http_server(0)
        thread = Thread(target=self._server.serve_forever, daemon=True)
        thread.start()
        port = self._server.server_address[1]
        if not _wait_for_http(f"http://127.0.0.1:{port}"):
            pytest.skip("HTTP server did not start in time")
        yield
        self._server.shutdown()

    def _url(self):
        port = self._server.server_address[1]
        return f"http://127.0.0.1:{port}"

    def test_connect_remote_and_disconnect(self):
        client = MCPClient(connect_timeout=5)
        client.connect_remote(self._url())
        assert client.is_connected is True
        assert client.mode == MCPTransportMode.HTTP
        client.disconnect()
        assert client.is_connected is False

    def test_server_info(self):
        client = MCPClient(connect_timeout=5)
        client.connect_remote(self._url())
        info = client.server_info
        assert info["name"] == "http-test-server"
        client.disconnect()

    def test_discover_tools(self):
        client = MCPClient(connect_timeout=5)
        client.connect_remote(self._url())
        tools = client.discover_tools()
        assert len(tools) >= 1
        assert tools[0].name == "http_echo"
        client.disconnect()

    def test_call_tool(self):
        client = MCPClient(connect_timeout=5, call_timeout=10)
        client.connect_remote(self._url())
        result = client.call_tool("http_echo", {"msg": "hi"})
        assert result.success is True
        assert len(result.content) > 0
        client.disconnect()

    def test_connect_unreachable(self):
        client = MCPClient(connect_timeout=2)
        with pytest.raises(MCPConnectionError):
            client.connect_remote("http://127.0.0.1:65432")

    def test_double_connect_raises(self):
        client = MCPClient(connect_timeout=5)
        client.connect_remote(self._url())
        with pytest.raises(MCPConnectionError):
            client.connect_remote(self._url())
        client.disconnect()

    def test_call_tool_not_connected(self):
        client = MCPClient()
        with pytest.raises(MCPConnectionError):
            client.call_tool("x", {})


class TestMCPClientMixedTransport:
    """Test switching between transport modes."""

    @pytest.fixture(autouse=True)
    def _servers(self):
        self._stdio_path, self._stdio_proc = _start_stdio_server()
        self._http_server = _start_http_server(0)
        thread = Thread(target=self._http_server.serve_forever, daemon=True)
        thread.start()
        port = self._http_server.server_address[1]
        if not _wait_for_http(f"http://127.0.0.1:{port}"):
            pytest.skip("HTTP server did not start in time")
        yield
        try:
            self._stdio_proc.terminate()
            self._stdio_proc.wait(timeout=5)
        except Exception:
            self._stdio_proc.kill()
            self._stdio_proc.wait()
        finally:
            if os.path.exists(self._stdio_path):
                os.unlink(self._stdio_path)
            self._http_server.shutdown()

    def _url(self):
        port = self._http_server.server_address[1]
        return f"http://127.0.0.1:{port}"

    def test_switch_from_stdio_to_http(self):
        client = MCPClient(connect_timeout=5)
        client.connect_local(sys.executable, [self._stdio_path])
        assert client.mode == MCPTransportMode.STDIO
        client.disconnect()
        client.connect_remote(self._url())
        assert client.mode == MCPTransportMode.HTTP
        client.disconnect()

    def test_switch_from_http_to_stdio(self):
        client = MCPClient(connect_timeout=5)
        client.connect_remote(self._url())
        assert client.mode == MCPTransportMode.HTTP
        client.disconnect()
        client.connect_local(sys.executable, [self._stdio_path])
        assert client.mode == MCPTransportMode.STDIO
        client.disconnect()

    def test_disconnect_clears_tools_cache(self):
        client = MCPClient(connect_timeout=5)
        client.connect_local(sys.executable, [self._stdio_path])
        tools = client.discover_tools()
        assert len(tools) > 0
        client.disconnect()
        # After disconnect, cache should be empty
        with pytest.raises(MCPConnectionError):
            client.discover_tools()

    def test_disconnect_clears_mode(self):
        client = MCPClient(connect_timeout=5)
        client.connect_local(sys.executable, [self._stdio_path])
        assert client.mode is not None
        client.disconnect()
        assert client.mode is None


class TestMCPClientErrorHandling:
    """Test error handling and edge cases."""

    @pytest.fixture(autouse=True)
    def _server(self):
        self._path, self._proc = _start_stdio_server()
        yield
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()
            self._proc.wait()
        finally:
            if os.path.exists(self._path):
                os.unlink(self._path)

    def test_initialize_not_connected_raises(self):
        client = MCPClient()
        with pytest.raises(MCPConnectionError):
            client.initialize()

    def test_call_tool_with_custom_timeout(self):
        client = MCPClient(connect_timeout=5, call_timeout=10)
        client.connect_local(sys.executable, [self._path])
        result = client.call_tool("echo", {"text": "timeout_test"}, timeout=5)
        assert result.success is True
        client.disconnect()

    def test_call_tool_with_empty_params(self):
        client = MCPClient(connect_timeout=5, call_timeout=10)
        client.connect_local(sys.executable, [self._path])
        result = client.call_tool("echo", {})
        assert result.success is True
        client.disconnect()

    def test_server_info_is_copy(self):
        client = MCPClient(connect_timeout=5)
        client.connect_local(sys.executable, [self._path])
        info1 = client.server_info
        info2 = client.server_info
        assert info1 is not info2
        client.disconnect()

    def test_server_capabilities_is_copy(self):
        client = MCPClient(connect_timeout=5)
        client.connect_local(sys.executable, [self._path])
        caps1 = client.server_capabilities
        caps2 = client.server_capabilities
        assert caps1 is not caps2
        client.disconnect()

    def test_has_capability_false(self):
        client = MCPClient(connect_timeout=5)
        client.connect_local(sys.executable, [self._path])
        assert client.has_capability("nonexistent") is False
        client.disconnect()

    def test_mcp_protocol_error_message(self):
        err = MCPProtocolError("test error")
        assert "test error" in str(err)

    def test_mcp_connection_error_message(self):
        err = MCPConnectionError("conn failed")
        assert "conn failed" in str(err)

    def test_mcp_timed_out_error_message(self):
        err = MCPTimedOutError("timeout")
        assert "timeout" in str(err)
