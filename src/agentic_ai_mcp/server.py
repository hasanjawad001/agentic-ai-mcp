"""AgenticAIServer - Simple MCP server using FastMCP."""

import functools
import inspect
from collections.abc import Callable
from typing import Any

from fastmcp import FastMCP


class AgenticAIServer:
    """Simple MCP server using FastMCP directly.

    Example:
        from agentic_ai_mcp import AgenticAIServer

        def add(a: int, b: int) -> int:
            '''Add two numbers.'''
            return a + b

        server = AgenticAIServer(host="0.0.0.0", port=8888)
        server.register_tool(add)

        print(f"Tools: {server.tools}")
        print(f"URL: {server.mcp_url}")

        # Runs until Ctrl+C
        server.run()
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

        # FastMCP instance
        self._mcp = FastMCP(name)

        # Track original function names (before wrapping)
        self._tool_names: list[str] = []

    @property
    def tools(self) -> list[str]:
        """List of registered tool names."""
        return self._tool_names.copy()

    @property
    def mcp_url(self) -> str:
        """Get the MCP server URL."""
        return f"http://{self.host}:{self.port}/mcp"

    def _wrap_tool_result(self, func: Callable[..., Any]) -> Callable[..., dict[str, Any]]:
        """Wrap function to return {"result": <original_return>}.

        This ensures all tool returns are dicts, which is required for
        MCP structured_content to work correctly with non-dict types
        like lists, arrays, and scalars.

        Args:
            func: Function to wrap

        Returns:
            Wrapped function that returns dict
        """

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            result = await func(*args, **kwargs)
            return {"result": result}

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            result = func(*args, **kwargs)
            return {"result": result}

        if inspect.iscoroutinefunction(func):
            async_wrapper.__annotations__["return"] = dict
            return async_wrapper  # type: ignore[return-value]
        else:
            sync_wrapper.__annotations__["return"] = dict
            return sync_wrapper

    def register_tool(self, func: Callable[..., Any]) -> None:
        """Register a function as an MCP tool.

        Args:
            func: Function with type hints and docstring
        """
        # Store original function name
        self._tool_names.append(func.__name__)

        # Wrap to ensure dict returns
        wrapped_func = self._wrap_tool_result(func)

        # Register with FastMCP
        self._mcp.tool()(wrapped_func)

        if self.verbose:
            print(f"Registered tool: {func.__name__}")

    def run(self) -> None:
        """Start the MCP server.

        Runs until interrupted (e.g., Ctrl+C or kernel restart).
        """
        if self.verbose:
            print(f"Starting MCP server at {self.mcp_url}")
            print(f"Tools: {self.tools}")

        # Run FastMCP with streamable-http transport (blocking)
        self._mcp.run(transport="streamable-http", host=self.host, port=self.port)
