"""Base agent class defining the interface for all agents."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from agentic_ai_mcp.core.types import AgentResponse, AgentRole

logger = logging.getLogger(__name__)


class BaseAgent(BaseModel, ABC):
    """
    Abstract base class for all agents in the framework.

    Agents are autonomous entities that can:
    - Process messages and generate responses
    - Invoke tools to perform actions
    - Communicate with other agents through the orchestrator

    Subclasses must implement the `process` method to define agent behavior.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="Unique name identifying this agent")
    description: str = Field(..., description="Description of the agent's capabilities")
    role: AgentRole = Field(
        default=AgentRole.SPECIALIST,
        description="Role of this agent in the system",
    )
    tools: list[BaseTool] = Field(
        default_factory=list,
        description="Tools available to this agent",
    )
    system_prompt: str = Field(
        default="You are a helpful AI assistant.",
        description="System prompt defining agent behavior",
    )
    max_iterations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum iterations for agent execution",
    )

    @abstractmethod
    async def process(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> AgentResponse:
        """
        Process incoming messages and generate a response.

        Args:
            messages: List of conversation messages
            **kwargs: Additional arguments for processing

        Returns:
            AgentResponse containing the agent's response
        """
        raise NotImplementedError

    async def invoke_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> str:
        """
        Invoke a tool by name with given input.

        Args:
            tool_name: Name of the tool to invoke
            tool_input: Input arguments for the tool

        Returns:
            Tool execution result as string

        Raises:
            ValueError: If tool is not found
        """
        for tool in self.tools:
            if tool.name == tool_name:
                logger.debug(f"Agent {self.name} invoking tool: {tool_name}")
                try:
                    result = await tool.ainvoke(tool_input)
                    logger.debug(f"Tool {tool_name} result: {result}")
                    return str(result)
                except Exception as e:
                    logger.error(f"Tool {tool_name} failed: {e}")
                    raise

        available_tools = [t.name for t in self.tools]
        raise ValueError(
            f"Tool '{tool_name}' not found. Available tools: {available_tools}"
        )

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """
        Get JSON schemas for all tools available to this agent.

        Returns:
            List of tool schemas in OpenAI function calling format
        """
        schemas = []
        for tool in self.tools:
            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.args_schema.model_json_schema()
                    if tool.args_schema
                    else {"type": "object", "properties": {}},
                },
            }
            schemas.append(schema)
        return schemas

    def create_tool_message(
        self,
        tool_call_id: str,
        content: str,
    ) -> ToolMessage:
        """
        Create a tool message for the conversation history.

        Args:
            tool_call_id: ID of the tool call this responds to
            content: Result content from the tool

        Returns:
            ToolMessage for the conversation
        """
        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"role={self.role.value}, "
            f"tools={[t.name for t in self.tools]})"
        )
