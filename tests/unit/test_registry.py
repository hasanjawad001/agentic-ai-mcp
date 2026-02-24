"""Unit tests for tool registry."""

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class EmptyInput(BaseModel):
    """Empty input schema for test tools that take no arguments."""

    pass


class SingleArgInput(BaseModel):
    """Input schema for test tools that take a single argument."""

    x: int = Field(description="Input value")


def create_test_tool(name: str, description: str = "", func=None) -> StructuredTool:
    """Helper to create test tools with proper args_schema."""
    if func is None:
        func = lambda: None
    return StructuredTool(
        name=name,
        description=description or f"Test tool {name}",
        func=func,
        args_schema=EmptyInput,
    )


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_register_tool(self, tool_registry):
        """Test registering a tool."""
        tool = StructuredTool(
            name="test_tool",
            description="A test tool",
            func=lambda x: x,
            args_schema=SingleArgInput,
        )

        tool_registry.register(tool, category="test")

        assert "test_tool" in tool_registry
        assert len(tool_registry) == 1

    def test_register_function(self, tool_registry):
        """Test registering a function as a tool."""

        def my_func(x: int) -> int:
            """Double the input."""
            return x * 2

        tool = tool_registry.register_function(
            my_func,
            name="doubler",
            category="math",
        )

        assert tool.name == "doubler"
        assert "doubler" in tool_registry

    def test_get_tool(self, tool_registry):
        """Test getting a tool by name."""
        tool = create_test_tool("my_tool", "Test")
        tool_registry.register(tool)

        retrieved = tool_registry.get("my_tool")
        assert retrieved is not None
        assert retrieved.name == "my_tool"

    def test_get_nonexistent_tool(self, tool_registry):
        """Test getting a tool that doesn't exist."""
        assert tool_registry.get("nonexistent") is None

    def test_get_by_category(self, tool_registry):
        """Test getting tools by category."""
        tool1 = create_test_tool("t1")
        tool2 = create_test_tool("t2")
        tool3 = create_test_tool("t3")

        tool_registry.register(tool1, category="math")
        tool_registry.register(tool2, category="math")
        tool_registry.register(tool3, category="text")

        math_tools = tool_registry.get_by_category("math")
        assert len(math_tools) == 2

    def test_get_by_tags(self, tool_registry):
        """Test getting tools by tags."""
        tool1 = create_test_tool("t1")
        tool2 = create_test_tool("t2")

        tool_registry.register(tool1, tags=["numeric", "fast"])
        tool_registry.register(tool2, tags=["string"])

        numeric_tools = tool_registry.get_by_tags(["numeric"])
        assert len(numeric_tools) == 1
        assert numeric_tools[0].name == "t1"

    def test_unregister_tool(self, tool_registry):
        """Test unregistering a tool."""
        tool = create_test_tool("to_remove")
        tool_registry.register(tool)

        assert "to_remove" in tool_registry

        result = tool_registry.unregister("to_remove")
        assert result is True
        assert "to_remove" not in tool_registry

    def test_unregister_nonexistent(self, tool_registry):
        """Test unregistering a tool that doesn't exist."""
        result = tool_registry.unregister("nonexistent")
        assert result is False

    def test_clear(self, tool_registry):
        """Test clearing all tools."""
        tool_registry.register(create_test_tool("t1"))
        tool_registry.register(create_test_tool("t2"))

        assert len(tool_registry) == 2

        tool_registry.clear()

        assert len(tool_registry) == 0

    def test_list_all(self, tool_registry):
        """Test listing all tools."""
        tool1 = create_test_tool("t1")
        tool2 = create_test_tool("t2")

        tool_registry.register(tool1)
        tool_registry.register(tool2)

        all_tools = tool_registry.list_all()
        assert len(all_tools) == 2

    def test_list_names(self, tool_registry):
        """Test listing tool names."""
        tool_registry.register(create_test_tool("alpha"))
        tool_registry.register(create_test_tool("beta"))

        names = tool_registry.list_names()
        assert "alpha" in names
        assert "beta" in names

    def test_list_categories(self, tool_registry):
        """Test listing categories."""
        tool_registry.register(create_test_tool("t1"), category="cat1")
        tool_registry.register(create_test_tool("t2"), category="cat2")

        categories = tool_registry.list_categories()
        assert "cat1" in categories
        assert "cat2" in categories

    def test_load_default_tools(self, tool_registry):
        """Test loading default tools."""
        tool_registry.load_default_tools()

        # Should have both math and text tools
        assert len(tool_registry) > 0
        assert "add" in tool_registry
        assert "to_uppercase" in tool_registry
