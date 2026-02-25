"""Agentic Workflow - Run agentic tasks using MCP tools.

Example:
    from agentic_ai_mcp import AgenticWorkflow

    workflow = AgenticWorkflow()
    result = await workflow.run("Calculate 2+3 and greet Tom with the result")
    print(result)
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from agentic_ai_mcp.bridge import MCPToolBridge
from agentic_ai_mcp.config import get_anthropic_api_key, get_default_model, get_mcp_url

logger = logging.getLogger(__name__)


class AgenticWorkflow:
    """
    Simple agentic workflow that uses MCP tools.

    Connects to MCP server, loads tools, and executes tasks using a ReAct agent.

    Example:
        workflow = AgenticWorkflow()
        result = await workflow.run("Calculate 2+3 and greet Tom")
        print(result)
    """

    def __init__(
        self,
        mcp_url: str | None = None,
        model: str | None = None,
        max_iterations: int = 10,
    ) -> None:
        """
        Initialize the workflow.

        Args:
            mcp_url: MCP server URL (default: from env or localhost:8888)
            model: LLM model to use (default: from env or claude-sonnet)
            max_iterations: Max iterations for the agent
        """
        self.mcp_url = mcp_url or get_mcp_url()
        self.model = model or get_default_model()
        self.max_iterations = max_iterations
        self._bridge: MCPToolBridge | None = None
        self._agent: Any = None
        self._tools: list = []

    async def _setup(self) -> None:
        """Setup the bridge and agent."""
        if self._agent is not None:
            return

        # Load tools from MCP server
        self._bridge = MCPToolBridge(self.mcp_url)
        self._tools = await self._bridge.get_tools()

        if not self._tools:
            raise RuntimeError(f"No tools found at {self.mcp_url}")

        logger.info(f"Loaded {len(self._tools)} tools: {[t.name for t in self._tools]}")

        # Create LLM
        llm = ChatAnthropic(
            model=self.model,
            api_key=get_anthropic_api_key(),
        )

        # Create ReAct agent
        self._agent = create_react_agent(llm, self._tools)

    async def run(self, prompt: str) -> str:
        """
        Run the workflow with a given prompt.

        Args:
            prompt: The task/question for the agent

        Returns:
            The agent's response
        """
        await self._setup()

        logger.info(f"Running workflow: {prompt[:100]}...")

        # Run the agent
        result = await self._agent.ainvoke({
            "messages": [HumanMessage(content=prompt)]
        })

        # Extract final response
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return str(msg.content)

        return "No response generated"

    def run_sync(self, prompt: str) -> str:
        """
        Synchronous version of run().

        Args:
            prompt: The task/question for the agent

        Returns:
            The agent's response
        """
        import asyncio
        return asyncio.run(self.run(prompt))

    @property
    def tools(self) -> list:
        """Get loaded tools."""
        return self._tools


async def run_workflow(prompt: str, mcp_url: str | None = None) -> str:
    """
    Convenience function to run a workflow.

    Args:
        prompt: The task/question
        mcp_url: Optional MCP server URL

    Returns:
        The agent's response
    """
    workflow = AgenticWorkflow(mcp_url=mcp_url)
    return await workflow.run(prompt)
