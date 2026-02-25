"""MCP Server - Register functions and serve them as MCP tools.

Example:
    from agentic_ai_mcp import MCPServer

    server = MCPServer()

    @server.tool()
    def add(a: int, b: int) -> int:
        '''Add two numbers.'''
        return a + b

    @server.tool()
    def greet(name: str) -> str:
        '''Greet someone.'''
        return f"Hello, {name}!"

    server.run()
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


class MCPServer:
    """
    Simple MCP Server for registering and serving tools.

    Usage:
        server = MCPServer()

        @server.tool()
        def my_function(x: int) -> int:
            return x * 2

        server.run()
    """

    def __init__(
        self,
        name: str = "agentic-ai-mcp",
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        """
        Initialize the MCP server.

        Args:
            name: Server name
            host: Host address (default: from env or 0.0.0.0)
            port: Port number (default: from env or 8888)
        """
        self.host = host or os.getenv("MCP_SERVER_HOST", "0.0.0.0")
        self.port = port or int(os.getenv("MCP_SERVER_PORT", "8888"))
        self._mcp = FastMCP(name, host=self.host, port=self.port)
        self._tools: list[str] = []

    def tool(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Decorator to register a function as an MCP tool.

        Example:
            @server.tool()
            def add(a: int, b: int) -> int:
                '''Add two numbers.'''
                return a + b
        """
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._mcp.tool()(func)
            self._tools.append(func.__name__)
            logger.debug(f"Registered tool: {func.__name__}")
            return func
        return decorator

    def add_tool(self, func: Callable[..., Any]) -> None:
        """
        Register a function as an MCP tool (non-decorator style).

        Args:
            func: Function to register

        Example:
            def add(a: int, b: int) -> int:
                return a + b

            server.add_tool(add)
        """
        self._mcp.tool()(func)
        self._tools.append(func.__name__)
        logger.debug(f"Registered tool: {func.__name__}")

    def run(self, transport: str = "streamable-http") -> None:
        """
        Start the MCP server.

        Args:
            transport: Transport protocol (default: streamable-http)
        """
        logger.info(f"Starting MCP server on {self.host}:{self.port}")
        logger.info(f"Registered tools: {self._tools}")
        print(f"MCP Server running at http://{self.host}:{self.port}/mcp")
        print(f"Tools available: {self._tools}")

        self._mcp.run(transport=transport)

    @property
    def tools(self) -> list[str]:
        """Get list of registered tool names."""
        return self._tools.copy()


def create_server(
    name: str = "agentic-ai-mcp",
    host: str | None = None,
    port: int | None = None,
) -> MCPServer:
    """Factory function to create an MCP server."""
    return MCPServer(name=name, host=host, port=port)


def main() -> None:
    """Entry point - prints usage instructions."""
    print("Usage:")
    print("  from agentic_ai_mcp import MCPServer")
    print()
    print("  server = MCPServer()")
    print()
    print("  @server.tool()")
    print("  def add(a: int, b: int) -> int:")
    print("      return a + b")
    print()
    print("  server.run()")


if __name__ == "__main__":
    main()
