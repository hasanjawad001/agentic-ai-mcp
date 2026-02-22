"""Math Agent specialized for mathematical operations."""

from __future__ import annotations

import logging
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from agentic_ai_mcp.config.settings import get_settings
from agentic_ai_mcp.core.base_agent import BaseAgent
from agentic_ai_mcp.core.types import AgentResponse, AgentRole
from agentic_ai_mcp.tools.math_tools import get_math_tools

logger = logging.getLogger(__name__)


MATH_AGENT_PROMPT = """You are a specialized Math Agent responsible for mathematical calculations.

Your capabilities include:
- Addition, subtraction, multiplication, and division
- Exponentiation (power operations)
- Square root calculations

Guidelines:
1. Use the appropriate tool for each mathematical operation
2. Show your work by explaining what calculation you're performing
3. Return numerical results clearly
4. If a calculation cannot be performed (e.g., division by zero), explain why

You have access to the following tools: add, subtract, multiply, divide, power, sqrt

Always use the tools for calculations rather than computing mentally."""


class MathAgent(BaseAgent):
    """
    Specialized agent for mathematical operations.

    This agent has access to math tools (add, subtract, multiply, divide, power, sqrt)
    and is designed to handle numerical computation tasks.
    """

    name: str = "math_agent"
    description: str = "Specialized agent for mathematical calculations and numerical operations"
    role: AgentRole = AgentRole.SPECIALIST
    system_prompt: str = MATH_AGENT_PROMPT

    def __init__(self, **data: Any) -> None:
        """Initialize the Math Agent with math tools."""
        if "tools" not in data:
            data["tools"] = get_math_tools()
        super().__init__(**data)
        self._llm: ChatAnthropic | None = None
        self._react_agent: Any | None = None

    def _get_llm(self) -> ChatAnthropic:
        """Get or create the LLM instance."""
        if self._llm is None:
            settings = get_settings()
            self._llm = ChatAnthropic(
                model=settings.default_model,
                api_key=settings.anthropic_api_key,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
            )
        return self._llm

    def _get_react_agent(self) -> Any:
        """Get or create the ReAct agent."""
        if self._react_agent is None:
            self._react_agent = create_react_agent(
                self._get_llm(),
                self.tools,
            )
        return self._react_agent

    async def process(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> AgentResponse:
        """
        Process mathematical queries.

        Args:
            messages: Conversation messages
            **kwargs: Additional arguments

        Returns:
            AgentResponse with calculation results
        """
        logger.info(f"Math Agent processing request")

        agent = self._get_react_agent()

        # Prepare input state
        input_state = {
            "messages": [
                HumanMessage(content=self.system_prompt),
                *messages,
            ]
        }

        # Execute agent
        try:
            result = await agent.ainvoke(input_state)
            final_messages = result.get("messages", [])

            # Extract final response
            content = None
            for msg in reversed(final_messages):
                if isinstance(msg, AIMessage) and msg.content:
                    content = msg.content
                    break

            logger.info(f"Math Agent completed processing")

            return AgentResponse(
                agent_name=self.name,
                content=content,
                is_final=True,
                metadata={"tool_calls_count": len(final_messages) - len(messages)},
            )

        except Exception as e:
            logger.error(f"Math Agent error: {e}")
            return AgentResponse(
                agent_name=self.name,
                content=f"Error performing calculation: {str(e)}",
                is_final=True,
                metadata={"error": str(e)},
            )

    def get_tool_descriptions(self) -> str:
        """Get formatted descriptions of available tools."""
        descriptions = []
        for tool in self.tools:
            descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(descriptions)


def create_math_agent() -> MathAgent:
    """Factory function to create a MathAgent instance."""
    return MathAgent()


def get_math_react_agent() -> Any:
    """
    Get a pre-configured ReAct agent for math operations.

    Returns:
        LangGraph ReAct agent with math tools
    """
    settings = get_settings()
    llm = ChatAnthropic(
        model=settings.default_model,
        api_key=settings.anthropic_api_key,
        max_tokens=settings.max_tokens,
    )
    return create_react_agent(llm, get_math_tools())
