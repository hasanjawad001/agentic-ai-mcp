"""MCP Server management."""

import asyncio
import contextlib
import multiprocessing
import os
import signal
import socket
import subprocess
import time
from collections.abc import Callable
from typing import Any

import cloudpickle
from fastmcp import FastMCP


def _run_server_process(name: str, host: str, port: int, pickled_funcs: bytes) -> None:
    """Run MCP server in a subprocess.

    Args:
        name: Server name
        host: Host address
        port: Port number
        pickled_funcs: Cloudpickle-serialized list of functions to register as tools
    """
    # Deserialize functions using cloudpickle (handles notebook-defined functions)
    funcs: list[Callable[..., Any]] = cloudpickle.loads(pickled_funcs)

    mcp = FastMCP(name)
    for func in funcs:
        mcp.tool()(func)
    asyncio.run(mcp.run_http_async(host=host, port=port))


class MCPServerManager:
    """Manages the lifecycle of an MCP server.

    Handles starting, stopping, and monitoring the MCP server process.
    """

    def __init__(
        self,
        name: str = "agentic-ai",
        host: str = "127.0.0.1",
        port: int = 8888,
        verbose: bool = False,
    ) -> None:
        """Initialize the server manager.

        Args:
            name: Server name for identification
            host: Host address to bind to
            port: Port number to listen on
            verbose: Enable verbose output
        """
        self.name = name
        self.host = host
        self.port = port
        self.verbose = verbose
        self._server_process: multiprocessing.Process | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        """Check if the server is currently running."""
        return self._running

    @property
    def mcp_url(self) -> str:
        """Get the MCP server URL."""
        return f"http://{self.host}:{self.port}/mcp"

    def start(self, funcs: list[Callable[..., Any]]) -> None:
        """Start the MCP server with registered functions.

        Args:
            funcs: List of functions to register as tools

        Raises:
            TimeoutError: If server doesn't start within timeout
        """
        if self._running:
            return

        # Serialize functions using cloudpickle (handles notebook-defined functions on Windows)
        pickled_funcs = cloudpickle.dumps(funcs)

        # Start server process
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
        """Get PIDs of processes using the server port.

        Returns:
            List of process IDs using the port
        """
        try:
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
