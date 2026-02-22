"""MCP Client for connecting to and invoking tools on MCP servers."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from agentic_ai.config.settings import get_settings

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Client for interacting with MCP servers.

    Provides methods to:
    - Connect to MCP servers
    - Discover available tools
    - Invoke tools with arguments
    """

    def __init__(self, server_url: str | None = None) -> None:
        """
        Initialize the MCP client.

        Args:
            server_url: URL of the MCP server (defaults to settings)
        """
        settings = get_settings()
        self.server_url = server_url or settings.mcp_server_url
        self._client: ClientSession | None = None
        self._tools: list[dict[str, Any]] = []

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[MCPClient]:
        """
        Connect to the MCP server.

        Yields:
            Connected MCPClient instance
        """
        logger.info(f"Connecting to MCP server at {self.server_url}")

        async with streamablehttp_client(self.server_url) as (read, write, _):
            async with ClientSession(read, write) as client:
                self._client = client
                await client.initialize()

                # Fetch available tools
                tools_response = await client.list_tools()
                self._tools = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    }
                    for tool in tools_response.tools
                ]
                logger.info(f"Connected. Found {len(self._tools)} tools")

                try:
                    yield self
                finally:
                    self._client = None
                    self._tools = []

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Get list of available tools."""
        return self._tools

    def get_tool_names(self) -> list[str]:
        """Get names of all available tools."""
        return [tool["name"] for tool in self._tools]

    def get_tool_by_name(self, name: str) -> dict[str, Any] | None:
        """Get tool definition by name."""
        for tool in self._tools:
            if tool["name"] == name:
                return tool
        return None

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> str:
        """
        Call a tool on the MCP server.

        Args:
            name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result as string

        Raises:
            RuntimeError: If not connected to server
            ValueError: If tool not found
        """
        if self._client is None:
            raise RuntimeError("Not connected to MCP server. Use 'async with client.connect():'")

        tool = self.get_tool_by_name(name)
        if tool is None:
            available = self.get_tool_names()
            raise ValueError(f"Tool '{name}' not found. Available: {available}")

        logger.debug(f"Calling tool '{name}' with args: {arguments}")

        result = await self._client.call_tool(name, arguments or {})

        # Extract text content from result
        content_parts = []
        for content in result.content:
            if hasattr(content, "text"):
                content_parts.append(content.text)

        result_text = "\n".join(content_parts)
        logger.debug(f"Tool '{name}' result: {result_text}")

        return result_text

    async def batch_call_tools(
        self,
        calls: list[tuple[str, dict[str, Any]]],
    ) -> list[str]:
        """
        Call multiple tools in sequence.

        Args:
            calls: List of (tool_name, arguments) tuples

        Returns:
            List of results in the same order as calls
        """
        results = []
        for name, args in calls:
            result = await self.call_tool(name, args)
            results.append(result)
        return results


async def get_available_tools(server_url: str | None = None) -> list[dict[str, Any]]:
    """
    Get list of available tools from an MCP server.

    Args:
        server_url: URL of the MCP server

    Returns:
        List of tool definitions
    """
    client = MCPClient(server_url)
    async with client.connect():
        return client.tools


async def invoke_tool(
    name: str,
    arguments: dict[str, Any],
    server_url: str | None = None,
) -> str:
    """
    Invoke a single tool on an MCP server.

    Args:
        name: Tool name
        arguments: Tool arguments
        server_url: MCP server URL

    Returns:
        Tool execution result
    """
    client = MCPClient(server_url)
    async with client.connect():
        return await client.call_tool(name, arguments)
