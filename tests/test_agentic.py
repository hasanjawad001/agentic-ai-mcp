"""Tests for AgenticAI."""

from agentic_ai_mcp import AgenticAI


class TestAgenticAI:
    """Tests for AgenticAI."""

    def test_create(self):
        """Test creating AgenticAI."""
        ai = AgenticAI()
        assert ai.host == "127.0.0.1"
        assert ai.port == 8888
        assert ai.tools == []

    def test_register_tool(self):
        """Test registering a tool."""
        ai = AgenticAI()

        def add(a: int, b: int) -> int:
            """Add numbers."""
            return a + b

        ai.register_tool(add)
        assert "add" in ai.tools

    def test_register_multiple_tools(self):
        """Test registering multiple tools."""
        ai = AgenticAI()

        def add(a: int, b: int) -> int:
            return a + b

        def greet(name: str) -> str:
            return f"Hello, {name}!"

        ai.register_tool(add)
        ai.register_tool(greet)

        assert len(ai.tools) == 2
        assert "add" in ai.tools
        assert "greet" in ai.tools
