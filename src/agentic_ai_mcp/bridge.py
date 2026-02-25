"""MCP Tool Bridge - Connect to MCP server and get tools as LangChain StructuredTools.

Example:
    from agentic_ai_mcp import MCPToolBridge

    bridge = MCPToolBridge("http://localhost:8888/mcp")
    await bridge.load_tools()
    langchain_tools = bridge.get_langchain_tools()
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastmcp import Client
from langchain_core.tools import StructuredTool
from pydantic import create_model

logger = logging.getLogger(__name__)


class MCPToolBridge:
    """
    Bridge between MCP server and LangChain.
    Connects to MCP server and converts tools to LangChain StructuredTools.
    """

    def __init__(self, mcp_url: str = "http://localhost:8888/mcp") -> None:
        """
        Initialize the bridge.

        Args:
            mcp_url: URL of the MCP server
        """
        self.mcp_url = mcp_url
        self.mcp_tools: list[Any] = []
        self.langchain_tools: list[StructuredTool] = []

    async def load_tools(self) -> list[Any]:
        """
        Load tools from MCP server.

        Returns:
            List of MCP tool definitions
        """
        async with Client(self.mcp_url) as client:
            self.mcp_tools = await client.list_tools()

        logger.info(f"Loaded {len(self.mcp_tools)} tools from MCP server")
        return self.mcp_tools

    def _create_tool_executor(self, tool_name: str):
        """Create a sync executor that calls the MCP tool."""
        mcp_url = self.mcp_url

        async def async_execute(**kwargs) -> str:
            async with Client(mcp_url) as client:
                result = await client.call_tool(tool_name, kwargs)
                if result.content and hasattr(result.content[0], "text"):
                    return str(result.content[0].text)
                return str(result.content)

        def sync_execute(**kwargs) -> str:
            """Sync wrapper for async executor."""
            return asyncio.run(async_execute(**kwargs))

        return sync_execute

    def _convert_schema(self, mcp_schema: dict | None) -> type | None:
        """Convert MCP JSON schema to Pydantic model."""
        if not mcp_schema or "properties" not in mcp_schema:
            return None

        fields: dict[str, Any] = {}
        properties = mcp_schema.get("properties", {})
        required = set(mcp_schema.get("required", []))

        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        for name, prop in properties.items():
            prop_type = type_map.get(prop.get("type", "string"), str)
            if name in required:
                fields[name] = (prop_type, ...)
            else:
                fields[name] = (prop_type | None, None)

        return create_model("ToolArgs", **fields)

    def get_langchain_tools(
        self,
        tool_names: list[str] | None = None,
    ) -> list[StructuredTool]:
        """
        Convert MCP tools to LangChain StructuredTools.

        Args:
            tool_names: Optional list of tool names to convert. If None, converts all.

        Returns:
            List of LangChain StructuredTool objects
        """
        if not self.mcp_tools:
            raise RuntimeError("No tools loaded. Call load_tools() first.")

        tools = []

        for mcp_tool in self.mcp_tools:
            if tool_names and mcp_tool.name not in tool_names:
                continue

            lc_tool = StructuredTool(
                name=mcp_tool.name,
                description=mcp_tool.description or f"Execute {mcp_tool.name}",
                func=self._create_tool_executor(mcp_tool.name),
                args_schema=self._convert_schema(mcp_tool.inputSchema),
            )
            tools.append(lc_tool)

        self.langchain_tools = tools
        logger.info(f"Converted {len(tools)} tools to LangChain format")
        return tools

    async def get_tools(self) -> list[StructuredTool]:
        """
        Convenience method: Load tools and convert to LangChain format.

        Returns:
            List of LangChain StructuredTool objects
        """
        await self.load_tools()
        return self.get_langchain_tools()
