"""Tool registry for managing and discovering tools."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from agentic_ai_mcp.tools.math_tools import get_math_tools
from agentic_ai_mcp.tools.text_tools import get_text_tools

logger = logging.getLogger(__name__)


class ToolMetadata(BaseModel):
    """Metadata about a registered tool."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    category: str = Field(default="general", description="Tool category")
    tags: list[str] = Field(default_factory=list, description="Tool tags for discovery")


class ToolRegistry:
    """
    Central registry for managing tools in the framework.

    The registry provides:
    - Tool registration and discovery
    - Category-based tool organization
    - Tool lookup by name or category
    """

    _instance: ToolRegistry | None = None

    def __new__(cls) -> ToolRegistry:
        """Singleton pattern for global registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
            cls._instance._metadata = {}
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if not self._initialized:
            self._tools: dict[str, BaseTool] = {}
            self._metadata: dict[str, ToolMetadata] = {}
            self._initialized = True

    def register(
        self,
        tool: BaseTool,
        category: str = "general",
        tags: list[str] | None = None,
    ) -> None:
        """
        Register a tool in the registry.

        Args:
            tool: Tool to register
            category: Category for the tool
            tags: Tags for discovery
        """
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting")

        self._tools[tool.name] = tool
        self._metadata[tool.name] = ToolMetadata(
            name=tool.name,
            description=tool.description,
            category=category,
            tags=tags or [],
        )
        logger.info(f"Registered tool: {tool.name} (category: {category})")

    def register_function(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
        category: str = "general",
        tags: list[str] | None = None,
        args_schema: type[BaseModel] | None = None,
    ) -> BaseTool:
        """
        Register a function as a tool.

        Args:
            func: Function to register
            name: Tool name (defaults to function name)
            description: Tool description (defaults to docstring)
            category: Category for the tool
            tags: Tags for discovery
            args_schema: Pydantic model for argument validation

        Returns:
            The created tool
        """
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Tool: {tool_name}"

        tool = StructuredTool(
            name=tool_name,
            description=tool_description,
            func=func,
            args_schema=args_schema,
        )

        self.register(tool, category=category, tags=tags)
        return tool

    def get(self, name: str) -> BaseTool | None:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool if found, None otherwise
        """
        return self._tools.get(name)

    def get_by_category(self, category: str) -> list[BaseTool]:
        """
        Get all tools in a category.

        Args:
            category: Category to filter by

        Returns:
            List of tools in the category
        """
        return [
            tool for name, tool in self._tools.items() if self._metadata[name].category == category
        ]

    def get_by_tags(self, tags: list[str]) -> list[BaseTool]:
        """
        Get all tools matching any of the given tags.

        Args:
            tags: Tags to search for

        Returns:
            List of tools matching the tags
        """
        tag_set = set(tags)
        return [
            tool for name, tool in self._tools.items() if tag_set & set(self._metadata[name].tags)
        ]

    def list_all(self) -> list[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def list_names(self) -> list[str]:
        """Get names of all registered tools."""
        return list(self._tools.keys())

    def list_categories(self) -> list[str]:
        """Get all unique categories."""
        return list({m.category for m in self._metadata.values()})

    def get_metadata(self, name: str) -> ToolMetadata | None:
        """Get metadata for a tool."""
        return self._metadata.get(name)

    def unregister(self, name: str) -> bool:
        """
        Remove a tool from the registry.

        Args:
            name: Tool name to remove

        Returns:
            True if tool was removed, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            del self._metadata[name]
            logger.info(f"Unregistered tool: {name}")
            return True
        return False

    def clear(self) -> None:
        """Remove all tools from the registry."""
        self._tools.clear()
        self._metadata.clear()
        logger.info("Cleared all tools from registry")

    def load_default_tools(self) -> None:
        """Load the default set of math and text tools."""
        # Register math tools
        for tool in get_math_tools():
            self.register(tool, category="math", tags=["math", "calculation", "numeric"])

        # Register text tools
        for tool in get_text_tools():
            self.register(tool, category="text", tags=["text", "string", "manipulation"])

        logger.info(f"Loaded {len(self._tools)} default tools")

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"ToolRegistry({len(self._tools)} tools, {len(self.list_categories())} categories)"


def get_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    return ToolRegistry()


def register_default_tools() -> ToolRegistry:
    """Initialize registry with default tools and return it."""
    registry = get_registry()
    registry.load_default_tools()
    return registry
