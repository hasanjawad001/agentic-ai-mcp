"""AgenticAIServer - Simple MCP server using FastMCP."""

import asyncio
import contextlib
import functools
import inspect
import multiprocessing
import os
import platform
import signal
import socket
import subprocess
import time
from collections.abc import Callable
from typing import Any

import cloudpickle
from fastmcp import FastMCP


def _wrap_tool_result(func: Callable[..., Any]) -> Callable[..., dict[str, Any]]:
    """Wrap function to return {"result": <original_return>}."""

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = await func(*args, **kwargs)
        return {"result": result}

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = func(*args, **kwargs)
        return {"result": result}

    if inspect.iscoroutinefunction(func):
        async_wrapper.__annotations__ = {**func.__annotations__, "return": dict}
        return async_wrapper  # type: ignore[return-value]
    else:
        sync_wrapper.__annotations__ = {**func.__annotations__, "return": dict}
        return sync_wrapper


def _run_server_process(name: str, host: str, port: int, pickled_funcs: bytes) -> None:
    """Run MCP server in a subprocess.

    Args:
        name: Server name
        host: Host address
        port: Port number
        pickled_funcs: Cloudpickle-serialized list of functions to register as tools
    """
    funcs: list[Callable[..., Any]] = cloudpickle.loads(pickled_funcs)

    mcp = FastMCP(name)
    for func in funcs:
        wrapped = _wrap_tool_result(func)
        mcp.tool()(wrapped)
    asyncio.run(mcp.run_http_async(host=host, port=port, stateless_http=True))


class AgenticAIServer:
    """Simple MCP server using FastMCP.

    Example:
        from agentic_ai_mcp import AgenticAIServer

        def add(a: int, b: int) -> int:
            '''Add two numbers.'''
            return a + b

        server = AgenticAIServer(host="0.0.0.0", port=8888)
        server.register_tool(add)

        print(f"Tools: {server.tools}")
        print(f"URL: {server.mcp_url}")

        # Start server in background
        server.start()

        # ... do other things ...

        # Stop server when done
        server.stop()
    """

    def __init__(
        self,
        name: str = "agentic-ai-server",
        host: str = "127.0.0.1",
        port: int = 8888,
        verbose: bool = False,
    ) -> None:
        """Initialize AgenticAIServer.

        Args:
            name: Name for the MCP server
            host: Host address for MCP server
            port: Port for MCP server
            verbose: Enable verbose output
        """
        self.name = name
        self.host = host
        self.port = port
        self.verbose = verbose

        # Track registered functions
        self._registered_funcs: list[Callable[..., Any]] = []
        self._tool_names: list[str] = []

        # Server process
        self._server_process: multiprocessing.Process | None = None
        self._running = False

    @property
    def tools(self) -> list[str]:
        """List of registered tool names."""
        return self._tool_names.copy()

    @property
    def mcp_url(self) -> str:
        """Get the MCP server URL."""
        return f"http://{self.host}:{self.port}/mcp"

    @property
    def is_running(self) -> bool:
        """Check if the server is currently running."""
        return self._running

    def register_tool(self, func: Callable[..., Any]) -> None:
        """Register a function as an MCP tool.

        Args:
            func: Function with type hints and docstring
        """
        # store original function name
        self._tool_names.append(func.__name__)

        # store original function
        self._registered_funcs.append(func)

        if self.verbose:
            print(f"Registered tool: {func.__name__}")

    def start(self) -> None:
        """Start the MCP server in background.

        The server runs in a separate process and can be stopped with stop().
        """
        if self._running:
            if self.verbose:
                print("Server is already running.")
            return

        # serialize functions using cloudpickle (handles notebook-defined functions)
        pickled_funcs = cloudpickle.dumps(self._registered_funcs)

        # start server process
        self._server_process = multiprocessing.Process(
            target=_run_server_process,
            args=(self.name, self.host, self.port, pickled_funcs),
            daemon=True,
        )
        self._server_process.start()

        # Wait for server to be ready
        self._wait_for_server()
        self._running = True

        if self.verbose:
            print(f"MCP Server running at {self.mcp_url}")

    def stop(self) -> None:
        """Stop the MCP server."""
        if not self._running and not self._get_pids_on_port():
            if self.verbose:
                print("Server is not running.")
            return

        self._kill_process_on_port()

        # Clean up process reference
        if self._server_process is not None:
            with contextlib.suppress(Exception):
                self._server_process.join(timeout=1)
            self._server_process = None

        self._running = False

        if self.verbose:
            print("MCP Server stopped.")

    def _get_pids_on_port(self) -> list[int]:
        """Get PIDs of processes using the server port."""
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["netstat", "-ano"],
                    capture_output=True,
                    text=True,
                )
                pids = []
                for line in result.stdout.splitlines():
                    if f":{self.port}" in line and "LISTENING" in line:
                        parts = line.strip().split()
                        if parts:
                            pid = int(parts[-1])
                            if pid not in pids:
                                pids.append(pid)
                return pids
            else:
                result = subprocess.run(
                    ["lsof", "-ti", f":{self.port}"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return [int(pid) for pid in result.stdout.strip().split("\n")]
        except (subprocess.SubprocessError, ValueError, FileNotFoundError):
            pass
        return []

    def _kill_process_on_port(self) -> None:
        """Kill any process using the server port."""
        pids = self._get_pids_on_port()

        if platform.system() == "Windows":
            for pid in pids:
                with contextlib.suppress(Exception):
                    subprocess.run(
                        ["taskkill", "/F", "/PID", str(pid)],
                        capture_output=True,
                    )
        else:
            for pid in pids:
                with contextlib.suppress(ProcessLookupError, OSError):
                    os.kill(pid, signal.SIGKILL)

        time.sleep(0.5)

    def _wait_for_server(self, timeout: float = 10.0) -> None:
        """Wait for server to be ready.

        Args:
            timeout: Maximum time to wait in seconds

        Raises:
            TimeoutError: If server doesn't start within timeout
        """
        start = time.time()
        while time.time() - start < timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                if sock.connect_ex((self.host, self.port)) == 0:
                    sock.close()
                    time.sleep(0.5)
                    return
                sock.close()
            except OSError:
                pass
            time.sleep(0.2)
        raise TimeoutError(f"Server did not start within {timeout}s")
