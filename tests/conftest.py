"""Pytest configuration and fixtures."""

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        "ANTHROPIC_API_KEY": "sk-test-key-for-testing",
        "MCP_SERVER_HOST": "localhost",
        "MCP_SERVER_PORT": "8888",
    }):
        yield


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    from langchain_core.messages import AIMessage, HumanMessage

    return [
        HumanMessage(content="Calculate 5 + 3"),
        AIMessage(content="I'll help you calculate 5 + 3."),
    ]


@pytest.fixture
def mock_llm():
    """Mock LLM for testing without API calls."""
    mock = MagicMock()
    mock.ainvoke = MagicMock(return_value=MagicMock(content="Mock response"))
    return mock


@pytest.fixture
def math_tools():
    """Get math tools for testing."""
    from agentic_ai.tools.math_tools import get_math_tools
    return get_math_tools()


@pytest.fixture
def text_tools():
    """Get text tools for testing."""
    from agentic_ai.tools.text_tools import get_text_tools
    return get_text_tools()


@pytest.fixture
def tool_registry():
    """Get a fresh tool registry for testing."""
    from agentic_ai.tools.registry import ToolRegistry

    # Create a new instance (bypass singleton for testing)
    registry = object.__new__(ToolRegistry)
    registry._tools = {}
    registry._metadata = {}
    registry._initialized = True
    return registry
