"""Supervisor Agent for orchestrating multiple specialist agents."""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agentic_ai.config.settings import get_settings
from agentic_ai.core.base_agent import BaseAgent
from agentic_ai.core.types import AgentResponse, AgentRole, RouteDecision

logger = logging.getLogger(__name__)


SUPERVISOR_PROMPT = """You are a Supervisor Agent responsible for routing tasks to the appropriate specialist agents.

Available agents:
- math_agent: Handles mathematical calculations (add, subtract, multiply, divide, power, sqrt)
- text_agent: Handles text manipulation (uppercase, lowercase, reverse, count chars/words, etc.)

Your job is to:
1. Analyze the user's request
2. Determine which agent(s) should handle it
3. Route to the appropriate agent
4. If the task requires multiple agents, route to them sequentially

Decision rules:
- Mathematical operations (numbers, calculations, arithmetic) → math_agent
- Text operations (string manipulation, case conversion, counting) → text_agent
- If the task is complete or doesn't require any agent → FINISH

Always explain your routing decision briefly."""


class RoutingDecision(BaseModel):
    """Structured output for supervisor routing decisions."""

    next_agent: Literal["math_agent", "text_agent", "FINISH"] = Field(
        ...,
        description="The next agent to route to, or FINISH if complete",
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of why this routing decision was made",
    )


class SupervisorAgent(BaseAgent):
    """
    Supervisor agent that orchestrates multiple specialist agents.

    The supervisor analyzes incoming requests and routes them to
    the appropriate specialist agent (math_agent or text_agent).
    """

    name: str = "supervisor"
    description: str = "Orchestrates tasks by routing to appropriate specialist agents"
    role: AgentRole = AgentRole.SUPERVISOR
    system_prompt: str = SUPERVISOR_PROMPT

    def __init__(self, **data: Any) -> None:
        """Initialize the Supervisor Agent."""
        super().__init__(**data)
        self._llm: ChatAnthropic | None = None

    def _get_llm(self) -> ChatAnthropic:
        """Get or create the LLM instance with structured output."""
        if self._llm is None:
            settings = get_settings()
            self._llm = ChatAnthropic(
                model=settings.default_model,
                api_key=settings.anthropic_api_key,
                max_tokens=settings.max_tokens,
                temperature=0,  # Deterministic routing
            )
        return self._llm

    async def process(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> AgentResponse:
        """
        Process messages and decide which agent to route to.

        Args:
            messages: Conversation messages
            **kwargs: Additional arguments

        Returns:
            AgentResponse with routing decision
        """
        logger.info("Supervisor processing routing decision")

        llm = self._get_llm()
        structured_llm = llm.with_structured_output(RoutingDecision)

        # Build prompt for routing decision
        routing_messages = [
            SystemMessage(content=self.system_prompt),
            *messages,
            HumanMessage(content="Based on the conversation, which agent should handle this next? Reply with your routing decision."),
        ]

        try:
            decision: RoutingDecision = await structured_llm.ainvoke(routing_messages)

            logger.info(
                f"Supervisor decision: {decision.next_agent} "
                f"(reason: {decision.reasoning})"
            )

            return AgentResponse(
                agent_name=self.name,
                content=decision.reasoning,
                is_final=(decision.next_agent == "FINISH"),
                metadata={
                    "next_agent": decision.next_agent,
                    "reasoning": decision.reasoning,
                },
            )

        except Exception as e:
            logger.error(f"Supervisor routing error: {e}")
            return AgentResponse(
                agent_name=self.name,
                content=f"Error making routing decision: {str(e)}",
                is_final=True,
                metadata={"error": str(e)},
            )

    async def route(
        self,
        messages: list[BaseMessage],
    ) -> RouteDecision:
        """
        Make a routing decision for the given messages.

        Args:
            messages: Conversation messages to route

        Returns:
            RouteDecision indicating next agent or FINISH
        """
        response = await self.process(messages)

        next_agent = response.metadata.get("next_agent", "FINISH")
        reasoning = response.metadata.get("reasoning", "")

        return RouteDecision(
            next_agent=next_agent,
            reasoning=reasoning,
        )

    def get_available_agents(self) -> list[str]:
        """Get list of agents this supervisor can route to."""
        return ["math_agent", "text_agent"]


def create_supervisor() -> SupervisorAgent:
    """Factory function to create a SupervisorAgent instance."""
    return SupervisorAgent()


def get_routing_function() -> Any:
    """
    Get a routing function for use in LangGraph workflows.

    Returns:
        Function that takes state and returns next node name
    """
    supervisor = create_supervisor()

    async def router(state: dict[str, Any]) -> str:
        messages = state.get("messages", [])
        decision = await supervisor.route(messages)
        return decision.next_agent

    return router
