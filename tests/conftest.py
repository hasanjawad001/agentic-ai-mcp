"""Pytest configuration and fixtures."""

import os
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(
        os.environ,
        {
            "ANTHROPIC_API_KEY": "sk-test-key-for-testing",
            "MCP_SERVER_HOST": "localhost",
            "MCP_SERVER_PORT": "8888",
        },
    ):
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
    """Get math tools for testing (directly, for unit tests)."""
    from agentic_ai_mcp.tools.math_tools import get_math_tools

    return get_math_tools()


@pytest.fixture
def text_tools():
    """Get text tools for testing (directly, for unit tests)."""
    from agentic_ai_mcp.tools.text_tools import get_text_tools

    return get_text_tools()


@pytest.fixture
def tool_registry():
    """Get a fresh tool registry for testing."""
    from agentic_ai_mcp.tools.registry import ToolRegistry

    # Create a new instance (bypass singleton for testing)
    registry = object.__new__(ToolRegistry)
    registry._tools = {}
    registry._metadata = {}
    registry._initialized = True
    return registry


@pytest.fixture
def mock_mcp_bridge():
    """
    Mock MCPToolBridge for workflow tests.

    Creates a mock bridge that provides math and text tools
    without requiring an actual MCP server connection.
    """
    from agentic_ai_mcp.tools.math_tools import get_math_tools
    from agentic_ai_mcp.tools.text_tools import get_text_tools

    mock_bridge = MagicMock()

    # Get actual tools for realistic testing
    math_tools = get_math_tools()
    text_tools = get_text_tools()
    all_tools = math_tools + text_tools

    mock_bridge._tools = all_tools
    mock_bridge._tool_map = {t.name: t for t in all_tools}

    def get_tools_by_category(category):
        if category == "math":
            return math_tools
        elif category == "text":
            return text_tools
        return all_tools

    mock_bridge.get_tools_by_category = get_tools_by_category
    mock_bridge.get_tools = MagicMock(return_value=all_tools)

    @asynccontextmanager
    async def mock_connect():
        yield mock_bridge

    mock_bridge.connect = mock_connect

    return mock_bridge


@pytest.fixture
def mock_workflow_with_mcp(mock_mcp_bridge):
    """
    Create a workflow with mocked MCP bridge.

    Use this fixture for workflow integration tests that don't need
    a real MCP server connection.
    """
    from agentic_ai_mcp.orchestration.workflow import AgenticWorkflow

    workflow = AgenticWorkflow()
    workflow._bridge = mock_mcp_bridge
    return workflow
