"""State management for agents and workflows."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, Sequence

from langchain_core.messages import BaseMessage


@dataclass
class AgentState:
    """State maintained by an individual agent during execution."""

    name: str
    messages: list[BaseMessage] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    iteration_count: int = 0
    is_complete: bool = False
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the agent's conversation history."""
        self.messages.append(message)

    def add_tool_result(self, result: dict[str, Any]) -> None:
        """Record a tool execution result."""
        self.tool_results.append(result)

    def increment_iteration(self) -> int:
        """Increment and return the iteration count."""
        self.iteration_count += 1
        return self.iteration_count

    def mark_complete(self, error: str | None = None) -> None:
        """Mark the agent execution as complete."""
        self.is_complete = True
        self.completed_at = datetime.now()
        self.error = error


class WorkflowState:
    """
    State schema for LangGraph workflow.

    This class defines the state that flows through the workflow graph.
    Uses Annotated types with reducers for proper state accumulation.
    """

    def __init__(self) -> None:
        self._state: dict[str, Any] = {
            "messages": [],
            "next_agent": "",
            "execution_path": [],
            "final_answer": None,
            "error": None,
            "metadata": {},
        }

    @property
    def messages(self) -> list[BaseMessage]:
        """Get accumulated messages."""
        return self._state["messages"]

    @messages.setter
    def messages(self, value: list[BaseMessage]) -> None:
        """Set messages."""
        self._state["messages"] = value

    @property
    def next_agent(self) -> str:
        """Get next agent to execute."""
        return self._state["next_agent"]

    @next_agent.setter
    def next_agent(self, value: str) -> None:
        """Set next agent."""
        self._state["next_agent"] = value

    @property
    def execution_path(self) -> list[str]:
        """Get execution path trace."""
        return self._state["execution_path"]

    @property
    def final_answer(self) -> str | None:
        """Get final answer if complete."""
        return self._state["final_answer"]

    @final_answer.setter
    def final_answer(self, value: str | None) -> None:
        """Set final answer."""
        self._state["final_answer"] = value

    def add_to_path(self, agent_name: str) -> None:
        """Add an agent to the execution path."""
        self._state["execution_path"].append(agent_name)

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary."""
        return self._state.copy()


# TypedDict version for LangGraph compatibility
from typing import TypedDict


class GraphState(TypedDict):
    """
    TypedDict state schema for LangGraph StateGraph.

    Attributes:
        messages: Accumulated conversation messages (uses add reducer)
        next_agent: Name of the next agent to execute
        execution_path: List of agents that have executed (uses add reducer)
        final_answer: Final response when workflow completes
        error: Error message if workflow fails
        metadata: Additional workflow metadata
    """

    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str
    execution_path: Annotated[list[str], operator.add]
    final_answer: str | None
    error: str | None
    metadata: dict[str, Any]


def create_initial_state(user_message: str) -> dict[str, Any]:
    """Create initial state for a new workflow execution."""
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content=user_message)],
        "next_agent": "",
        "execution_path": [],
        "final_answer": None,
        "error": None,
        "metadata": {
            "started_at": datetime.now().isoformat(),
            "user_query": user_message,
        },
    }
