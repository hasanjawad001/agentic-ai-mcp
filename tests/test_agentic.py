"""Tests for AgenticAI unified interface."""

import pytest

from agentic_ai_mcp import AgenticAI


class TestAgenticAI:
    """Tests for AgenticAI class."""

    def test_create_agentic_ai(self):
        """Test creating AgenticAI instance."""
        ai = AgenticAI()
        assert ai.host == "127.0.0.1"
        assert ai.port == 8888
        assert ai.max_iterations == 10
        assert ai.tools == []

    def test_create_with_custom_config(self):
        """Test creating with custom configuration."""
        ai = AgenticAI(host="localhost", port=9000, max_iterations=5)
        assert ai.host == "localhost"
        assert ai.port == 9000
        assert ai.max_iterations == 5

    def test_register_tool(self):
        """Test registering a tool."""
        ai = AgenticAI()

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        ai.register_tool(add)

        assert "add" in ai.tools
        assert len(ai.tools) == 1

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

    def test_generate_server_code(self):
        """Test server code generation."""
        ai = AgenticAI()

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        ai.register_tool(add)

        code = ai._generate_server_code()

        assert "from agentic_ai_mcp import MCPServer" in code
        assert "def add(a: int, b: int) -> int:" in code
        assert "server.add_tool(add)" in code
        assert 'host="127.0.0.1"' in code
        assert "port=8888" in code

    def test_run_without_tools_raises_error(self):
        """Test that running without tools raises an error."""
        ai = AgenticAI()

        with pytest.raises(RuntimeError, match="No tools registered"):
            ai._start_server()
