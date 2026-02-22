"""Unit tests for state management."""

import pytest
from datetime import datetime

from agentic_ai.core.state import AgentState, create_initial_state, GraphState


class TestAgentState:
    """Tests for AgentState class."""

    def test_create_agent_state(self):
        """Test creating an agent state."""
        state = AgentState(name="test_agent")

        assert state.name == "test_agent"
        assert state.messages == []
        assert state.tool_results == []
        assert state.iteration_count == 0
        assert state.is_complete is False
        assert state.error is None

    def test_add_message(self):
        """Test adding a message to state."""
        from langchain_core.messages import HumanMessage

        state = AgentState(name="test")
        msg = HumanMessage(content="Hello")

        state.add_message(msg)

        assert len(state.messages) == 1
        assert state.messages[0].content == "Hello"

    def test_add_tool_result(self):
        """Test adding a tool result."""
        state = AgentState(name="test")

        result = {"tool": "add", "output": 5}
        state.add_tool_result(result)

        assert len(state.tool_results) == 1
        assert state.tool_results[0]["output"] == 5

    def test_increment_iteration(self):
        """Test incrementing iteration count."""
        state = AgentState(name="test")

        assert state.iteration_count == 0

        count = state.increment_iteration()
        assert count == 1
        assert state.iteration_count == 1

        count = state.increment_iteration()
        assert count == 2

    def test_mark_complete(self):
        """Test marking state as complete."""
        state = AgentState(name="test")
        state.started_at = datetime.now()

        state.mark_complete()

        assert state.is_complete is True
        assert state.completed_at is not None
        assert state.error is None

    def test_mark_complete_with_error(self):
        """Test marking state as complete with error."""
        state = AgentState(name="test")

        state.mark_complete(error="Something went wrong")

        assert state.is_complete is True
        assert state.error == "Something went wrong"


class TestCreateInitialState:
    """Tests for create_initial_state function."""

    def test_creates_valid_state(self):
        """Test that initial state is valid."""
        state = create_initial_state("Hello, how are you?")

        assert "messages" in state
        assert len(state["messages"]) == 1
        assert state["messages"][0].content == "Hello, how are you?"

    def test_includes_metadata(self):
        """Test that initial state includes metadata."""
        state = create_initial_state("Test query")

        assert "metadata" in state
        assert "started_at" in state["metadata"]
        assert "user_query" in state["metadata"]
        assert state["metadata"]["user_query"] == "Test query"

    def test_initial_values(self):
        """Test initial state values."""
        state = create_initial_state("Query")

        assert state["next_agent"] == ""
        assert state["execution_path"] == []
        assert state["final_answer"] is None
        assert state["error"] is None
