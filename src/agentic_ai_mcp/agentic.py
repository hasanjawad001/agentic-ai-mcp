"""Unified AgenticAI interface - simple one-class solution."""

import asyncio
import inspect
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from typing import Any

from .config import get_default_model
from .workflow import AgenticWorkflow


class AgenticAI:
    """Unified interface for agentic AI with automatic MCP server management.

    Example:
        from agentic_ai_mcp import AgenticAI

        ai = AgenticAI()

        def add(a: int, b: int) -> int:
            '''Add two numbers.'''
            return a + b

        def greet(name: str, times: int = 1) -> str:
            '''Greet someone.'''
            return ("Hello, " + name + "! ") * times

        ai.register_tool(add)
        ai.register_tool(greet)

        result = await ai.run("Calculate 2+3 and greet Tom the result times")
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8888,
        model: str | None = None,
        max_iterations: int = 10,
    ):
        """Initialize AgenticAI.

        Args:
            host: Host for the MCP server (default: 127.0.0.1)
            port: Port for the MCP server (default: 8888)
            model: LLM model to use (default: from env or claude-sonnet-4-20250514)
            max_iterations: Maximum agent iterations (default: 10)
        """
        self.host = host
        self.port = port
        self.model = model or get_default_model()
        self.max_iterations = max_iterations
        self._tools: dict[str, tuple[Callable[..., Any], str]] = {}
        self._server_process: subprocess.Popen[bytes] | None = None
        self._temp_file: str | None = None

    @property
    def tools(self) -> list[str]:
        """List of registered tool names."""
        return list(self._tools.keys())

    def register_tool(self, func: Callable[..., Any]) -> None:
        """Register a function as a tool.

        Args:
            func: The function to register. Must have type hints and a docstring.

        Example:
            def multiply(a: int, b: int) -> int:
                '''Multiply two numbers.'''
                return a * b

            ai.register_tool(multiply)
        """
        source = inspect.getsource(func)
        self._tools[func.__name__] = (func, source)

    def _generate_server_code(self) -> str:
        """Generate the MCP server code with registered tools."""
        tool_sources = "\n\n".join(source for _, source in self._tools.values())
        tool_registrations = "\n".join(
            f"server.add_tool({name})" for name in self._tools
        )

        return f'''"""Auto-generated MCP server."""
from agentic_ai_mcp import MCPServer

server = MCPServer(host="{self.host}", port={self.port})

{tool_sources}

{tool_registrations}

if __name__ == "__main__":
    server.run()
'''

    def _start_server(self) -> None:
        """Start the MCP server in a subprocess."""
        if not self._tools:
            raise RuntimeError("No tools registered. Use ai.register_tool(func) first.")

        # Create temp file with server code
        code = self._generate_server_code()

        fd, self._temp_file = tempfile.mkstemp(suffix=".py", prefix="agentic_server_")
        try:
            os.write(fd, code.encode())
        finally:
            os.close(fd)

        # Start subprocess
        self._server_process = subprocess.Popen(
            [sys.executable, self._temp_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to be ready
        self._wait_for_server()

    def _wait_for_server(self, timeout: float = 10.0) -> None:
        """Wait for the MCP server to be ready."""
        import socket

        start = time.time()
        while time.time() - start < timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex((self.host, self.port))
                sock.close()
                if result == 0:
                    # Give it a moment to fully initialize
                    time.sleep(0.5)
                    return
            except OSError:
                pass
            time.sleep(0.2)

        raise TimeoutError(f"MCP server did not start within {timeout} seconds")

    def _stop_server(self) -> None:
        """Stop the MCP server subprocess."""
        if self._server_process:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
                self._server_process.wait()
            self._server_process = None

        # Clean up temp file
        if self._temp_file and os.path.exists(self._temp_file):
            os.unlink(self._temp_file)
            self._temp_file = None

    async def run(self, prompt: str) -> str:
        """Run an agentic workflow with the given prompt.

        This method:
        1. Starts an MCP server with registered tools (subprocess)
        2. Runs the agentic workflow with the prompt
        3. Returns the result
        4. Cleans up the server

        Args:
            prompt: The task/prompt for the agent

        Returns:
            The agent's final response
        """
        try:
            self._start_server()

            mcp_url = f"http://{self.host}:{self.port}/mcp"
            workflow = AgenticWorkflow(
                mcp_url=mcp_url,
                model=self.model,
                max_iterations=self.max_iterations,
            )

            result = await workflow.run(prompt)
            return result
        finally:
            self._stop_server()

    def run_sync(self, prompt: str) -> str:
        """Synchronous version of run().

        Args:
            prompt: The task/prompt for the agent

        Returns:
            The agent's final response
        """
        return asyncio.run(self.run(prompt))
