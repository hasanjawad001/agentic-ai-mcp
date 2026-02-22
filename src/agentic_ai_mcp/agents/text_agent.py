"""Text Agent specialized for text manipulation operations."""

from __future__ import annotations

import logging
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from agentic_ai_mcp.config.settings import get_settings
from agentic_ai_mcp.core.base_agent import BaseAgent
from agentic_ai_mcp.core.types import AgentResponse, AgentRole
from agentic_ai_mcp.tools.text_tools import get_text_tools

logger = logging.getLogger(__name__)


TEXT_AGENT_PROMPT = """You are a specialized Text Agent responsible for text manipulation operations.

Your capabilities include:
- Case conversion (uppercase, lowercase, capitalize)
- Text transformation (reverse, strip whitespace)
- Text analysis (count characters, count words)
- Search and replace operations

Guidelines:
1. Use the appropriate tool for each text operation
2. Explain what transformation you're applying
3. Return the transformed text clearly
4. Handle empty strings and edge cases gracefully

You have access to the following tools: to_uppercase, to_lowercase, capitalize,
reverse_text, strip_whitespace, count_chars, count_words, search_replace

Always use the tools for transformations rather than doing them mentally."""


class TextAgent(BaseAgent):
    """
    Specialized agent for text manipulation operations.

    This agent has access to text tools (uppercase, lowercase, reverse, count, etc.)
    and is designed to handle string processing tasks.
    """

    name: str = "text_agent"
    description: str = "Specialized agent for text manipulation and string operations"
    role: AgentRole = AgentRole.SPECIALIST
    system_prompt: str = TEXT_AGENT_PROMPT

    def __init__(self, **data: Any) -> None:
        """Initialize the Text Agent with text tools."""
        if "tools" not in data:
            data["tools"] = get_text_tools()
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
        Process text manipulation queries.

        Args:
            messages: Conversation messages
            **kwargs: Additional arguments

        Returns:
            AgentResponse with transformed text
        """
        logger.info(f"Text Agent processing request")

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

            logger.info(f"Text Agent completed processing")

            return AgentResponse(
                agent_name=self.name,
                content=content,
                is_final=True,
                metadata={"tool_calls_count": len(final_messages) - len(messages)},
            )

        except Exception as e:
            logger.error(f"Text Agent error: {e}")
            return AgentResponse(
                agent_name=self.name,
                content=f"Error performing text operation: {str(e)}",
                is_final=True,
                metadata={"error": str(e)},
            )

    def get_tool_descriptions(self) -> str:
        """Get formatted descriptions of available tools."""
        descriptions = []
        for tool in self.tools:
            descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(descriptions)


def create_text_agent() -> TextAgent:
    """Factory function to create a TextAgent instance."""
    return TextAgent()


def get_text_react_agent() -> Any:
    """
    Get a pre-configured ReAct agent for text operations.

    Returns:
        LangGraph ReAct agent with text tools
    """
    settings = get_settings()
    llm = ChatAnthropic(
        model=settings.default_model,
        api_key=settings.anthropic_api_key,
        max_tokens=settings.max_tokens,
    )
    return create_react_agent(llm, get_text_tools())
