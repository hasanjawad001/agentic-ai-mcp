"""Pytest fixtures for agentic-ai-mcp tests."""

import os

import pytest

from agentic_ai_mcp import AgenticAI, Settings


@pytest.fixture
def sample_tools():
    """Provide sample tool functions for testing."""

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    def greet(name: str, times: int = 1) -> str:
        """Greet someone."""
        return f"Hello, {name}! " * times

    return {"add": add, "multiply": multiply, "greet": greet}


@pytest.fixture
def agentic_ai():
    """Provide a basic AgenticAI instance."""
    return AgenticAI()


@pytest.fixture
def agentic_ai_with_tools(sample_tools):
    """Provide an AgenticAI instance with registered tools."""
    ai = AgenticAI(port=8889)  # Use different port to avoid conflicts
    for func in sample_tools.values():
        ai.register_tool(func)
    yield ai
    # Cleanup
    ai.stop_mcp_server()


@pytest.fixture
def settings():
    """Provide a Settings instance."""
    return Settings()


@pytest.fixture
def has_anthropic_key():
    """Check if Anthropic API key is available."""
    return bool(os.getenv("ANTHROPIC_API_KEY"))


@pytest.fixture
def has_openai_key():
    """Check if OpenAI API key is available."""
    return bool(os.getenv("OPENAI_API_KEY"))
