"""Integration tests for agentic-ai-mcp."""

import os

import pytest

from agentic_ai_mcp import AgenticAI, Settings


class TestAgenticAIIntegration:
    """Integration tests for AgenticAI."""

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set",
    )
    async def test_run_simple_calculation(self):
        """Test run() with a simple calculation tool."""
        ai = AgenticAI(port=8890, verbose=False)

        def add(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        ai.register_tool(add)

        try:
            result = await ai.run("What is 2 + 3? Use the add tool.")
            assert "5" in result
        finally:
            ai.stop_mcp_server()

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set",
    )
    async def test_run_with_planning_multi_step(self):
        """Test run_with_planning() with multi-step operations."""
        ai = AgenticAI(port=8891, verbose=False)

        def add(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        def multiply(a: int, b: int) -> int:
            """Multiply two numbers together."""
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

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )
    async def test_openai_provider(self):
        """Test using OpenAI provider."""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            pytest.skip("langchain-openai not installed")

        ai = AgenticAI(
            port=8892,
            provider="openai",
            model="gpt-4o-mini",
            verbose=False,
        )

        def add(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        ai.register_tool(add)

        try:
            result = await ai.run("What is 5 + 7? Use the add tool.")
            assert "12" in result
        finally:
            ai.stop_mcp_server()

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set",
    )
    async def test_client_only_mode(self):
        """Test client-only mode connecting to existing server."""
        # First, start a server
        server_ai = AgenticAI(port=8893, verbose=False)

        def add(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        server_ai.register_tool(add)
        server_ai.run_mcp_server()

        try:
            # Now create a client-only instance
            client_ai = AgenticAI(mcp_url="http://127.0.0.1:8893/mcp")

            result = await client_ai.run("What is 10 + 20? Use the add tool.")
            assert "30" in result
        finally:
            server_ai.stop_mcp_server()

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set",
    )
    async def test_custom_settings(self):
        """Test using custom Settings."""
        settings = Settings(max_retries=3)
        ai = AgenticAI(port=8894, settings=settings, verbose=False)

        def add(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        ai.register_tool(add)

        try:
            result = await ai.run("What is 1 + 1? Use the add tool.")
            assert "2" in result
        finally:
            ai.stop_mcp_server()


class TestToolRegistry:
    """Tests for tool registration and management."""

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

        def multiply(a: int, b: int) -> int:
            return a * b

        ai.register_tool(add)
        ai.register_tool(multiply)

        assert len(ai.tools) == 2
        assert "add" in ai.tools
        assert "multiply" in ai.tools

    def test_client_only_cannot_register(self):
        """Test that client-only mode cannot register tools."""
        ai = AgenticAI(mcp_url="http://127.0.0.1:8888/mcp")

        def add(a: int, b: int) -> int:
            return a + b

        with pytest.raises(RuntimeError, match="client-only mode"):
            ai.register_tool(add)

    def test_client_only_cannot_start_server(self):
        """Test that client-only mode cannot start server."""
        ai = AgenticAI(mcp_url="http://127.0.0.1:8888/mcp")

        with pytest.raises(RuntimeError, match="client-only mode"):
            ai.run_mcp_server()
