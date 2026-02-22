"""Router for directing tasks to appropriate agents."""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agentic_ai.config.settings import get_settings

logger = logging.getLogger(__name__)


class RouterDecision(BaseModel):
    """Structured output for routing decisions."""

    next: Literal["math_agent", "text_agent", "FINISH"] = Field(
        ...,
        description="Next agent to route to, or FINISH if complete",
    )
    reasoning: str = Field(
        ...,
        description="Explanation for the routing decision",
    )


ROUTER_PROMPT = """You are a routing component in a multi-agent system.

Your job is to analyze the conversation and decide which specialist agent should handle the next step.

Available agents:
- math_agent: For mathematical operations (addition, subtraction, multiplication, division, power, sqrt)
- text_agent: For text manipulation (uppercase, lowercase, reverse, count characters/words, etc.)
- FINISH: When the task is complete and no more agents are needed

Routing guidelines:
1. If the task involves numbers and arithmetic → math_agent
2. If the task involves text/string operations → text_agent
3. If both math and text are needed, route to one at a time based on order of operations
4. If the original request has been fully addressed → FINISH

Analyze the conversation and make your routing decision."""


class AgentRouter:
    """
    Router that determines which agent should handle a task.

    Uses structured output from Claude to make routing decisions.
    """

    def __init__(self) -> None:
        """Initialize the router."""
        self._llm: ChatAnthropic | None = None

    def _get_llm(self) -> ChatAnthropic:
        """Get or create the LLM instance."""
        if self._llm is None:
            settings = get_settings()
            self._llm = ChatAnthropic(
                model=settings.default_model,
                api_key=settings.anthropic_api_key,
                max_tokens=512,
                temperature=0,  # Deterministic routing
            )
        return self._llm

    async def route(
        self,
        messages: list[BaseMessage],
    ) -> RouterDecision:
        """
        Determine which agent should handle the next step.

        Args:
            messages: Conversation history

        Returns:
            RouterDecision with next agent and reasoning
        """
        logger.debug("Router making decision")

        llm = self._get_llm()
        structured_llm = llm.with_structured_output(RouterDecision)

        routing_messages = [
            SystemMessage(content=ROUTER_PROMPT),
            *messages,
            HumanMessage(content="Make your routing decision."),
        ]

        try:
            decision: RouterDecision = await structured_llm.ainvoke(routing_messages)
            logger.info(f"Router decision: {decision.next} ({decision.reasoning})")
            return decision

        except Exception as e:
            logger.error(f"Router error: {e}")
            # Default to FINISH on error
            return RouterDecision(
                next="FINISH",
                reasoning=f"Routing error: {str(e)}",
            )

    def route_sync(self, messages: list[BaseMessage]) -> RouterDecision:
        """Synchronous version of route."""
        import asyncio
        return asyncio.run(self.route(messages))


def create_router() -> AgentRouter:
    """Factory function to create an AgentRouter."""
    return AgentRouter()


def get_conditional_edge_function() -> Any:
    """
    Get a function for LangGraph conditional edges.

    Returns:
        Function that takes state and returns next node name
    """
    router = create_router()

    def route_to_agent(state: dict[str, Any]) -> str:
        """Routing function for LangGraph."""
        import asyncio

        messages = state.get("messages", [])
        decision = asyncio.run(router.route(messages))

        # Map decision to graph nodes
        if decision.next == "FINISH":
            return "__end__"
        return decision.next

    return route_to_agent


async def async_route_function(state: dict[str, Any]) -> str:
    """
    Async routing function for LangGraph workflows.

    Args:
        state: Current workflow state

    Returns:
        Name of next node to execute
    """
    router = create_router()
    messages = state.get("messages", [])
    decision = await router.route(messages)

    if decision.next == "FINISH":
        return "__end__"
    return decision.next
