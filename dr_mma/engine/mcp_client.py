"""
MCP Client — Model Context Protocol client for DR-MMA.

Supports two transport modes:
- Local stdio: subprocess.Popen with JSON-RPC over stdin/stdout
- Remote HTTP/SSE: urllib.request POST + Server-Sent Events parsing

Protocol flow (JSON-RPC 2.0):
  initialize → initialized notification → tools/list → tools/call
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class MCPErrors:
    """MCP JSON-RPC error codes."""

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


class MCPCapabilities(Enum):
    """MCP server capabilities."""

    TOOLS = "tools"
    RESOURCES = "resources"
    PROMPTS = "prompts"


@dataclass
class MCPToolDefinition:
    """MCP tool definition returned by tools/list."""

    name: str
    description: str = ""
    inputSchema: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.inputSchema,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MCPToolDefinition:
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            inputSchema=data.get("inputSchema", {}),
        )


@dataclass
class MCPToolResult:
    """Result of an MCP tool call."""

    success: bool = True
    content: list[dict] = field(default_factory=list)
    isError: bool = False
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "content": self.content,
            "isError": self.isError,
            "metadata": self.metadata,
        }


class MCPTransportMode(Enum):
    """Transport mode for MCP connection."""

    STDIO = "stdio"
    HTTP = "http"


class MCPConnectionError(Exception):
    """MCP connection failed."""

    pass


class MCPProtocolError(Exception):
    """MCP protocol violation detected."""

    pass


class MCPTimedOutError(Exception):
    """MCP operation timed out."""

    pass


class _StdioTransport:
    """
    Local stdio transport via subprocess.Popen.

    Sends JSON-RPC messages over stdin, reads responses from stdout.
    Each message is terminated by a newline.
    """

    def __init__(self, command: str, args: list[str], env: Optional[dict] = None):
        self._process: Optional[subprocess.Popen] = None
        self._command = command
        self._args = args
        self._env = env or {}
        self._lock = threading.Lock()

    def connect(self, timeout: float = 10.0) -> None:
        """Start the subprocess and wait for it to be ready."""
        try:
            self._process = subprocess.Popen(
                [self._command] + self._args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self._env,
                text=True,
                bufsize=0,
            )
        except FileNotFoundError:
            raise MCPConnectionError(f"Command not found: {self._command}")
        except Exception as e:
            raise MCPConnectionError(f"Failed to start process: {e}")

    def send(self, message: dict, timeout: float = 30.0) -> dict:
        """
        Send a JSON-RPC request and wait for the corresponding response.

        Returns the parsed response dict.
        Raises MCPProtocolError if the server returns an error.
        Raises MCPTimedOutError if no response within timeout.
        """
        if self._process is None or self._process.poll() is not None:
            raise MCPConnectionError("Transport not connected")

        raw = json.dumps(message) + "\n"

        with self._lock:
            try:
                self._process.stdin.write(raw)
                self._process.stdin.flush()
            except Exception as e:
                raise MCPConnectionError(f"Write failed: {e}")

            # Read response line by line until we get a valid JSON response
            start = time.time()
            buffer = ""
            while True:
                elapsed = time.time() - start
                if elapsed > timeout:
                    raise MCPTimedOutError(
                        f"No response within {timeout}s for method {message.get('method')}"
                    )

                try:
                    line = self._process.stdout.readline()
                except Exception as e:
                    raise MCPConnectionError(f"Read failed: {e}")

                if not line:
                    # stdout closed
                    raise MCPConnectionError("Server process closed unexpectedly")

                buffer += line.strip()

                try:
                    response = json.loads(buffer)
                    if isinstance(response, dict) and "id" in response:
                        return response
                    buffer = ""
                except json.JSONDecodeError:
                    # Incomplete JSON, keep reading
                    continue

    def disconnect(self) -> None:
        """Terminate the subprocess."""
        if self._process is not None:
            try:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
            except Exception:
                pass
            finally:
                self._process = None

    @property
    def is_connected(self) -> bool:
        return self._process is not None and self._process.poll() is None


class _HttpTransport:
    """
    Remote HTTP/SSE transport via urllib.request.

    Sends JSON-RPC POST requests to the server URL.
    Supports SSE event stream for notifications.
    """

    def __init__(self, url: str, headers: Optional[dict] = None):
        self._url = url.rstrip("/")
        self._headers = {"Content-Type": "application/json"}
        if headers:
            self._headers.update(headers)
        self._connected = False

    def connect(self, timeout: float = 10.0) -> None:
        """Verify the server is reachable."""
        # Try HEAD first, fall back to GET if HEAD fails
        for method in ("HEAD", "GET"):
            try:
                req = urllib.request.Request(
                    self._url,
                    method=method,
                    headers=self._headers,
                )
                with urllib.request.urlopen(req, timeout=timeout):
                    self._connected = True
                    return
            except Exception:
                continue
        raise MCPConnectionError(f"Cannot reach server at {self._url}")

    def send(self, message: dict, timeout: float = 30.0) -> dict:
        """
        Send a JSON-RPC POST request and return the parsed response.

        Raises MCPProtocolError if the server returns an error.
        Raises MCPTimedOutError if no response within timeout.
        """
        if not self._connected:
            raise MCPConnectionError("HTTP transport not connected")

        raw = json.dumps(message).encode("utf-8")

        try:
            req = urllib.request.Request(
                self._url,
                data=raw,
                headers=self._headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            raise MCPConnectionError(f"HTTP error {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            raise MCPConnectionError(f"URL error: {e.reason}")
        except Exception as e:
            raise MCPConnectionError(f"Request failed: {e}")

        # Parse response — may be a single JSON object or SSE stream
        try:
            response = json.loads(body)
            if isinstance(response, dict) and "id" in response:
                return response
        except json.JSONDecodeError:
            pass

        # Try SSE parsing: look for data: lines
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("data: "):
                try:
                    data = json.loads(stripped[6:])
                    if isinstance(data, dict) and "id" in data:
                        return data
                except json.JSONDecodeError:
                    continue

        raise MCPProtocolError(f"Could not parse server response: {body[:200]}")

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected


class MCPClient:
    """
    Model Context Protocol client supporting stdio and HTTP transports.

    Lifecycle:
        1. connect_local() or connect_remote()
        2. initialize() (called automatically on connect)
        3. discover_tools()
        4. call_tool(name, params)
        5. disconnect()

    Usage (stdio):
        client = MCPClient()
        client.connect_local("python", ["-m", "mcp_server"])
        tools = client.discover_tools()
        result = client.call_tool("my_tool", {"arg": "value"})
        client.disconnect()

    Usage (HTTP):
        client = MCPClient()
        client.connect_remote("http://localhost:3000/mcp")
        tools = client.discover_tools()
        result = client.call_tool("my_tool", {"arg": "value"})
        client.disconnect()
    """

    def __init__(self, connect_timeout: float = 10.0, call_timeout: float = 30.0):
        self._transport: Optional[_StdioTransport] = None
        self._http_transport: Optional[_HttpTransport] = None
        self._mode: Optional[MCPTransportMode] = None
        self._server_info: dict = {}
        self._server_capabilities: list[str] = []
        self._initialized = False
        self._request_id = 0
        self._lock = threading.Lock()
        self._connect_timeout = connect_timeout
        self._call_timeout = call_timeout
        self._tools_cache: list[MCPToolDefinition] = []

    # ── Connection ────────────────────────────────────────────────────

    def connect_local(
        self,
        command: str,
        args: Optional[list[str]] = None,
        env: Optional[dict] = None,
    ) -> None:
        """
        Connect to an MCP server via local stdio.

        Args:
            command: Executable path (e.g., "python")
            args: Command arguments (e.g., ["-m", "mcp_server"])
            env: Environment variables for the subprocess
        """
        if self._mode is not None:
            raise MCPConnectionError("Already connected. Call disconnect() first.")

        transport = _StdioTransport(command, args or [], env)
        transport.connect(timeout=self._connect_timeout)
        self._transport = transport
        self._mode = MCPTransportMode.STDIO

        # Auto-initialize
        self._do_initialize()

    def connect_remote(
        self,
        url: str,
        headers: Optional[dict] = None,
    ) -> None:
        """
        Connect to an MCP server via HTTP.

        Args:
            url: Server URL (e.g., "http://localhost:3000/mcp")
            headers: Additional HTTP headers
        """
        if self._mode is not None:
            raise MCPConnectionError("Already connected. Call disconnect() first.")

        transport = _HttpTransport(url, headers)
        transport.connect(timeout=self._connect_timeout)
        self._http_transport = transport
        self._mode = MCPTransportMode.HTTP

        # Auto-initialize
        self._do_initialize()

    def disconnect(self) -> None:
        """Close the connection and release resources."""
        if self._transport is not None:
            self._transport.disconnect()
            self._transport = None
        if self._http_transport is not None:
            self._http_transport.disconnect()
            self._http_transport = None
        self._mode = None
        self._initialized = False
        self._tools_cache = []

    @property
    def is_connected(self) -> bool:
        """Check if the client is connected to a server."""
        if self._transport is not None:
            return self._transport.is_connected
        if self._http_transport is not None:
            return self._http_transport.is_connected
        return False

    @property
    def mode(self) -> Optional[MCPTransportMode]:
        """Current transport mode."""
        return self._mode

    # ── Protocol ──────────────────────────────────────────────────────

    def _next_id(self) -> int:
        with self._lock:
            self._request_id += 1
            return self._request_id

    def _send_request(
        self,
        method: str,
        params: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> dict:
        """
        Send a JSON-RPC 2.0 request and return the result.

        Raises MCPProtocolError if the server returns an error response.
        """
        req_id = self._next_id()
        message: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
        }
        if params is not None:
            message["params"] = params

        transport_timeout = timeout or self._call_timeout

        if self._mode == MCPTransportMode.STDIO and self._transport is not None:
            response = self._transport.send(message, transport_timeout)
        elif self._mode == MCPTransportMode.HTTP and self._http_transport is not None:
            response = self._http_transport.send(message, transport_timeout)
        else:
            raise MCPConnectionError("Not connected")

        # Check for protocol errors
        if "error" in response:
            error = response["error"]
            code = error.get("code", -1)
            msg = error.get("message", "Unknown error")
            raise MCPProtocolError(f"MCP error {code}: {msg}")

        return response

    def _send_notification(self, method: str, params: Optional[dict] = None) -> None:
        """
        Send a JSON-RPC 2.0 notification (no response expected).

        For stdio mode, we send and don't wait.
        For HTTP mode, we fire-and-forget via POST.
        """
        message: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            message["params"] = params

        # Notifications don't have an id, so we can't wait for response
        # For stdio, send without waiting
        if self._mode == MCPTransportMode.STDIO and self._transport is not None:
            raw = json.dumps(message) + "\n"
            try:
                self._transport._process.stdin.write(raw)
                self._transport._process.stdin.flush()
            except Exception as e:
                raise MCPConnectionError(f"Notification send failed: {e}")
        elif self._mode == MCPTransportMode.HTTP and self._http_transport is not None:
            raw = json.dumps(message).encode("utf-8")
            try:
                req = urllib.request.Request(
                    self._http_transport._url,
                    data=raw,
                    headers=self._http_transport._headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self._call_timeout):
                    pass
            except Exception:
                # Notifications are best-effort
                pass
        else:
            raise MCPConnectionError("Not connected")

    def _do_initialize(self) -> None:
        """
        Perform the MCP initialization handshake.

        1. Send initialize request with client info
        2. Wait for server capabilities response
        3. Send initialized notification
        """
        try:
            response = self._send_request(
                "initialize",
                params={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                    },
                    "clientInfo": {
                        "name": "dr-mma",
                        "version": "1.0.0",
                    },
                },
            )
        except MCPProtocolError:
            # Some servers may not support initialize gracefully
            pass

        result = response.get("result", {})
        self._server_info = result.get("serverInfo", {})
        server_caps = result.get("capabilities", {})
        self._server_capabilities = list(server_caps.keys())
        self._initialized = True

        # Send initialized notification
        try:
            self._send_notification("notifications/initialized")
        except Exception:
            pass

    def initialize(
        self,
        client_name: str = "dr-mma",
        client_version: str = "1.0.0",
    ) -> None:
        """
        Manually trigger initialization (useful if auto-init was skipped).

        Args:
            client_name: Client identifier name
            client_version: Client identifier version
        """
        if not self.is_connected:
            raise MCPConnectionError("Not connected. Call connect_local() or connect_remote() first.")

        response = self._send_request(
            "initialize",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                },
                "clientInfo": {
                    "name": client_name,
                    "version": client_version,
                },
            },
        )

        result = response.get("result", {})
        self._server_info = result.get("serverInfo", {})
        server_caps = result.get("capabilities", {})
        self._server_capabilities = list(server_caps.keys())
        self._initialized = True

        # Send initialized notification
        try:
            self._send_notification("notifications/initialized")
        except Exception:
            pass

    # ── Tools ─────────────────────────────────────────────────────────

    def discover_tools(self, force_refresh: bool = False) -> list[MCPToolDefinition]:
        """
        Discover tools available from the server.

        If force_refresh is False and tools were already cached, returns the cache.

        Returns:
            List of MCPToolDefinition objects
        """
        if not self.is_connected:
            raise MCPConnectionError("Not connected")

        if not force_refresh and self._tools_cache:
            return list(self._tools_cache)

        response = self._send_request("tools/list")
        result = response.get("result", {})
        tools_data = result.get("tools", [])

        self._tools_cache = [
            MCPToolDefinition.from_dict(t) for t in tools_data
        ]
        return list(self._tools_cache)

    def call_tool(
        self,
        name: str,
        params: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> MCPToolResult:
        """
        Call an MCP tool by name.

        Args:
            name: Tool name (must match a discovered tool)
            params: Tool parameters
            timeout: Override default call timeout

        Returns:
            MCPToolResult with success, content, isError, and metadata
        """
        if not self.is_connected:
            raise MCPConnectionError("Not connected")

        response = self._send_request(
            "tools/call",
            params={
                "name": name,
                "arguments": params or {},
            },
            timeout=timeout,
        )

        result = response.get("result", {})
        content = result.get("content", [])
        is_error = result.get("isError", False)

        return MCPToolResult(
            success=not is_error,
            content=content if isinstance(content, list) else [],
            isError=is_error,
            metadata={"tool_name": name},
        )

    # ── Server info ───────────────────────────────────────────────────

    @property
    def server_info(self) -> dict:
        """Server information returned during initialization."""
        return self._server_info.copy()

    @property
    def server_capabilities(self) -> list[str]:
        """Server capabilities returned during initialization."""
        return self._server_capabilities.copy()

    def has_capability(self, capability: str) -> bool:
        """Check if the server supports a given capability."""
        return capability in self._server_capabilities
