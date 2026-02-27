"""Tests for AgenticAI core methods."""

import os

import pytest

from agentic_ai_mcp import AgenticAI


class TestAgenticAI:
    """Tests for the 5 core AgenticAI methods."""

    def test_register_tool(self):
        """Test ai.register_tool(func)"""
        ai = AgenticAI()

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        ai.register_tool(add)
        ai.register_tool(multiply)

        assert "add" in ai.tools
        assert "multiply" in ai.tools
        assert len(ai.tools) == 2

    def test_run_mcp_server(self):
        """Test ai.run_mcp_server()"""
        ai = AgenticAI(port=8895)

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        ai.register_tool(add)
        ai.run_mcp_server()

        assert ai._server_running is True

        # Cleanup
        ai.stop_mcp_server()

    def test_stop_mcp_server(self):
        """Test ai.stop_mcp_server()"""
        ai = AgenticAI(port=8896)

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        ai.register_tool(add)
        ai.run_mcp_server()
        assert ai._server_running is True

        ai.stop_mcp_server()
        assert ai._server_running is False

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set",
    )
    async def test_run(self):
        """Test ai.run(prompt)"""
        ai = AgenticAI(port=8897)

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        ai.register_tool(add)

        try:
            result = await ai.run("What is 2 + 3? Use the add tool.")
            assert "5" in result
        finally:
            ai.stop_mcp_server()

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set",
    )
    async def test_run_with_planning(self):
        """Test ai.run_with_planning(prompt)"""
        ai = AgenticAI(port=8898)

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        ai.register_tool(add)
        ai.register_tool(multiply)

        try:
            result = await ai.run_with_planning(
                "First multiply 3 and 4, then add 5 to the result."
            )
            # 3 * 4 = 12, 12 + 5 = 17
            assert "17" in result
        finally:
            ai.stop_mcp_server()
