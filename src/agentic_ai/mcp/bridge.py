"""Bridge between MCP tools and LangChain tools."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from agentic_ai.mcp.client import MCPClient

logger = logging.getLogger(__name__)


class MCPToolBridge:
    """
    Bridge that converts MCP tools to LangChain StructuredTools.

    This allows MCP-served tools to be used seamlessly with
    LangChain agents and LangGraph workflows.
    """

    def __init__(self, server_url: str | None = None) -> None:
        """
        Initialize the bridge.

        Args:
            server_url: URL of the MCP server
        """
        self.client = MCPClient(server_url)
        self._tools: list[StructuredTool] = []
        self._tool_map: dict[str, StructuredTool] = {}

    async def load_tools(self) -> list[StructuredTool]:
        """
        Load tools from MCP server and convert to LangChain tools.

        Returns:
            List of LangChain StructuredTools
        """
        async with self.client.connect() as connected_client:
            mcp_tools = connected_client.tools
            self._tools = []
            self._tool_map = {}

            for mcp_tool in mcp_tools:
                lc_tool = self._convert_mcp_to_langchain(
                    mcp_tool,
                    connected_client,
                )
                self._tools.append(lc_tool)
                self._tool_map[lc_tool.name] = lc_tool

            logger.info(f"Loaded {len(self._tools)} tools from MCP server")
            return self._tools

    def _convert_mcp_to_langchain(
        self,
        mcp_tool: dict[str, Any],
        client: MCPClient,
    ) -> StructuredTool:
        """
        Convert an MCP tool definition to a LangChain StructuredTool.

        Args:
            mcp_tool: MCP tool definition
            client: Connected MCP client

        Returns:
            LangChain StructuredTool
        """
        name = mcp_tool["name"]
        description = mcp_tool.get("description", f"Tool: {name}")
        input_schema = mcp_tool.get("input_schema", {})

        # Build Pydantic model from JSON schema
        args_schema = self._schema_to_pydantic(name, input_schema)

        # Create sync wrapper for async tool call
        def create_tool_func(tool_name: str) -> Callable[..., str]:
            def tool_func(**kwargs: Any) -> str:
                return asyncio.run(
                    self._call_tool_async(tool_name, kwargs)
                )
            return tool_func

        return StructuredTool(
            name=name,
            description=description,
            func=create_tool_func(name),
            args_schema=args_schema,
        )

    async def _call_tool_async(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Call a tool asynchronously."""
        async with self.client.connect() as connected_client:
            return await connected_client.call_tool(name, arguments)

    def _schema_to_pydantic(
        self,
        name: str,
        schema: dict[str, Any],
    ) -> type[BaseModel]:
        """
        Convert JSON schema to Pydantic model.

        Args:
            name: Name for the model
            schema: JSON schema definition

        Returns:
            Pydantic model class
        """
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        fields: dict[str, Any] = {}
        for prop_name, prop_schema in properties.items():
            prop_type = self._json_type_to_python(prop_schema.get("type", "string"))
            prop_description = prop_schema.get("description", "")

            if prop_name in required:
                fields[prop_name] = (prop_type, Field(..., description=prop_description))
            else:
                default = prop_schema.get("default")
                fields[prop_name] = (prop_type, Field(default=default, description=prop_description))

        model_name = f"{name.title().replace('_', '')}Args"
        return create_model(model_name, **fields)

    def _json_type_to_python(self, json_type: str) -> type:
        """Convert JSON schema type to Python type."""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        return type_mapping.get(json_type, str)

    def get_tools(self) -> list[StructuredTool]:
        """Get loaded tools."""
        return self._tools

    def get_tool(self, name: str) -> StructuredTool | None:
        """Get a specific tool by name."""
        return self._tool_map.get(name)

    def get_tools_by_category(self, category: str) -> list[StructuredTool]:
        """
        Get tools by category based on naming convention.

        Math tools: add, subtract, multiply, divide, power, sqrt
        Text tools: to_uppercase, to_lowercase, reverse_text, count_chars, etc.

        Args:
            category: "math" or "text"

        Returns:
            List of tools in the category
        """
        math_tools = {"add", "subtract", "multiply", "divide", "power", "sqrt"}
        text_tools = {
            "to_uppercase", "to_lowercase", "reverse_text", "count_chars",
            "count_words", "capitalize", "strip_whitespace", "search_replace"
        }

        if category == "math":
            return [t for t in self._tools if t.name in math_tools]
        elif category == "text":
            return [t for t in self._tools if t.name in text_tools]
        else:
            return self._tools


async def load_mcp_tools(server_url: str | None = None) -> list[StructuredTool]:
    """
    Load tools from MCP server.

    Args:
        server_url: MCP server URL

    Returns:
        List of LangChain StructuredTools
    """
    bridge = MCPToolBridge(server_url)
    return await bridge.load_tools()


def load_mcp_tools_sync(server_url: str | None = None) -> list[StructuredTool]:
    """
    Load tools from MCP server (sync wrapper).

    Args:
        server_url: MCP server URL

    Returns:
        List of LangChain StructuredTools
    """
    return asyncio.run(load_mcp_tools(server_url))
