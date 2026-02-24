"""Core type definitions for the Agentic AI Framework."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentRole(str, Enum):
    """Enumeration of agent roles in the system."""

    SUPERVISOR = "supervisor"
    SPECIALIST = "specialist"
    EXECUTOR = "executor"


class ToolCall(BaseModel):
    """Represents a tool call made by an agent."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Unique identifier for this tool call")
    name: str = Field(..., description="Name of the tool to invoke")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments to pass to the tool",
    )


class ToolResult(BaseModel):
    """Represents the result of a tool execution."""

    model_config = ConfigDict(frozen=True)

    tool_call_id: str = Field(..., description="ID of the tool call this result belongs to")
    name: str = Field(..., description="Name of the tool that was invoked")
    content: str = Field(..., description="Result content from the tool")
    is_error: bool = Field(default=False, description="Whether the result is an error")


class AgentResponse(BaseModel):
    """Represents a response from an agent."""

    agent_name: str = Field(..., description="Name of the responding agent")
    content: str | None = Field(default=None, description="Text content of the response")
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="Tool calls requested by the agent",
    )
    is_final: bool = Field(
        default=False,
        description="Whether this is the final response",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the response",
    )


class RouteDecision(BaseModel):
    """Decision made by the supervisor about which agent to route to."""

    next_agent: str = Field(
        ...,
        description="Name of the next agent to handle the task, or 'FINISH' to complete",
    )
    reasoning: str = Field(
        ...,
        description="Explanation for why this routing decision was made",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence level in the routing decision",
    )


class ExecutionStatus(str, Enum):
    """Status of workflow execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
