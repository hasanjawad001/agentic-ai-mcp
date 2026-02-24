"""LangGraph workflow for multi-agent orchestration."""

from __future__ import annotations

import logging
import operator
from collections.abc import Sequence
from typing import Annotated, Any, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from agentic_ai_mcp.agents.supervisor import RoutingDecision
from agentic_ai_mcp.config.settings import get_settings
from agentic_ai_mcp.mcp.bridge import MCPToolBridge

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """
    State schema for the multi-agent workflow.

    Attributes:
        messages: Accumulated conversation messages
        next_agent: Name of the next agent to execute
        execution_path: Trace of agents that have executed
        iteration_count: Number of routing iterations
    """

    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str
    execution_path: Annotated[list[str], operator.add]
    iteration_count: int


class AgenticWorkflow:
    """
    Multi-agent workflow orchestrator using LangGraph.

    This workflow:
    1. Receives a user request
    2. Connects to MCP server and loads tools
    3. Routes to the appropriate specialist agent
    4. Executes the agent's tools via MCP
    5. Routes to next agent or finishes

    All tools are accessed exclusively through the MCP server.

    Example:
        workflow = AgenticWorkflow()
        result = await workflow.execute("Calculate 5 + 3, then convert to uppercase")
    """

    def __init__(
        self,
        max_iterations: int = 10,
        server_url: str | None = None,
    ) -> None:
        """
        Initialize the workflow.

        Args:
            max_iterations: Maximum number of agent routing iterations
            server_url: MCP server URL (defaults to settings)
        """
        self.max_iterations = max_iterations
        self._bridge = MCPToolBridge(server_url)
        self._graph: StateGraph | None = None
        self._compiled: Any = None
        self._llm: ChatAnthropic | None = None
        self._math_agent: Any = None
        self._text_agent: Any = None
        self._math_tools: list = []
        self._text_tools: list = []

    def _get_llm(self) -> ChatAnthropic:
        """Get or create the LLM instance."""
        if self._llm is None:
            settings = get_settings()
            self._llm = ChatAnthropic(
                model=settings.default_model,
                api_key=settings.anthropic_api_key,
                max_tokens=settings.max_tokens,
            )
        return self._llm

    def _get_math_agent(self) -> Any:
        """Get or create the math ReAct agent using MCP-sourced tools."""
        if self._math_agent is None:
            self._math_agent = create_react_agent(
                self._get_llm(),
                self._math_tools,
            )
        return self._math_agent

    def _get_text_agent(self) -> Any:
        """Get or create the text ReAct agent using MCP-sourced tools."""
        if self._text_agent is None:
            self._text_agent = create_react_agent(
                self._get_llm(),
                self._text_tools,
            )
        return self._text_agent

    async def _load_tools_from_mcp(self) -> None:
        """Load tools from MCP server and categorize them."""
        self._math_tools = self._bridge.get_tools_by_category("math")
        self._text_tools = self._bridge.get_tools_by_category("text")
        logger.info(
            f"Loaded {len(self._math_tools)} math tools and "
            f"{len(self._text_tools)} text tools from MCP"
        )

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        if self._graph is not None:
            return self._graph

        graph = StateGraph(WorkflowState)

        # Add nodes
        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("math_agent", self._math_agent_node)
        graph.add_node("text_agent", self._text_agent_node)

        # Add edges from START to supervisor
        graph.add_edge(START, "supervisor")

        # Add conditional edges from supervisor
        graph.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "math_agent": "math_agent",
                "text_agent": "text_agent",
                "FINISH": END,
            },
        )

        # Agents return to supervisor
        graph.add_edge("math_agent", "supervisor")
        graph.add_edge("text_agent", "supervisor")

        self._graph = graph
        return graph

    async def _supervisor_node(
        self,
        state: WorkflowState,
    ) -> dict[str, Any]:
        """
        Supervisor node that decides routing.

        Args:
            state: Current workflow state

        Returns:
            State update with next_agent decision
        """
        logger.info("Supervisor analyzing request")

        # Check iteration limit
        if state.get("iteration_count", 0) >= self.max_iterations:
            logger.warning("Max iterations reached, finishing")
            return {"next_agent": "FINISH"}

        llm = self._get_llm()
        structured_llm = llm.with_structured_output(RoutingDecision)

        supervisor_prompt = """Analyze the conversation and decide which agent to route to next.

Available agents:
- math_agent: For mathematical operations (add, subtract, multiply, divide, power, sqrt)
- text_agent: For text manipulation (uppercase, lowercase, reverse, count, etc.)
- FINISH: When the task is complete

Consider what has already been done and what still needs to be done."""

        messages = list(state["messages"])
        messages.append(HumanMessage(content=supervisor_prompt))

        try:
            decision: RoutingDecision = await structured_llm.ainvoke(messages)
            logger.info(f"Supervisor routes to: {decision.next_agent}")

            return {
                "next_agent": decision.next_agent,
                "execution_path": ["supervisor"],
                "iteration_count": state.get("iteration_count", 0) + 1,
            }

        except Exception as e:
            logger.error(f"Supervisor error: {e}")
            return {"next_agent": "FINISH"}

    def _route_from_supervisor(self, state: WorkflowState) -> str:
        """Route from supervisor to next node."""
        return state.get("next_agent", "FINISH")

    async def _math_agent_node(
        self,
        state: WorkflowState,
    ) -> dict[str, Any]:
        """
        Math agent node that performs calculations.

        Args:
            state: Current workflow state

        Returns:
            State update with agent response
        """
        logger.info("Math Agent processing")

        agent = self._get_math_agent()
        result = await agent.ainvoke({"messages": state["messages"]})

        # Extract new messages (excluding input)
        new_messages = result["messages"][len(state["messages"]) :]

        return {
            "messages": new_messages,
            "execution_path": ["math_agent"],
        }

    async def _text_agent_node(
        self,
        state: WorkflowState,
    ) -> dict[str, Any]:
        """
        Text agent node that performs text operations.

        Args:
            state: Current workflow state

        Returns:
            State update with agent response
        """
        logger.info("Text Agent processing")

        agent = self._get_text_agent()
        result = await agent.ainvoke({"messages": state["messages"]})

        # Extract new messages (excluding input)
        new_messages = result["messages"][len(state["messages"]) :]

        return {
            "messages": new_messages,
            "execution_path": ["text_agent"],
        }

    def compile(self) -> Any:
        """Compile the workflow graph."""
        if self._compiled is None:
            graph = self._build_graph()
            self._compiled = graph.compile()
        return self._compiled

    async def execute(
        self,
        query: str,
    ) -> dict[str, Any]:
        """
        Execute the workflow with a user query.

        Connects to MCP server, loads tools, and runs the workflow.

        Args:
            query: User's request

        Returns:
            Final workflow state with results

        Raises:
            agentic_ai_mcp.mcp.bridge.MCPConnectionError: If unable to connect to MCP server
        """
        logger.info(f"Executing workflow for: {query[:100]}...")

        # Connect to MCP and run workflow within the connection context
        async with self._bridge.connect():
            # Load and categorize tools from MCP
            await self._load_tools_from_mcp()

            # Reset agents to use newly loaded tools
            self._math_agent = None
            self._text_agent = None

            compiled = self.compile()

            initial_state: WorkflowState = {
                "messages": [HumanMessage(content=query)],
                "next_agent": "",
                "execution_path": [],
                "iteration_count": 0,
            }

            result = await compiled.ainvoke(initial_state)
            return result

    async def execute_stream(
        self,
        query: str,
    ) -> Any:
        """
        Execute the workflow with streaming output.

        Connects to MCP server, loads tools, and streams workflow execution.

        Args:
            query: User's request

        Yields:
            Execution steps as they happen

        Raises:
            agentic_ai_mcp.mcp.bridge.MCPConnectionError: If unable to connect to MCP server
        """
        logger.info(f"Streaming workflow for: {query[:100]}...")

        # Connect to MCP and run workflow within the connection context
        async with self._bridge.connect():
            # Load and categorize tools from MCP
            await self._load_tools_from_mcp()

            # Reset agents to use newly loaded tools
            self._math_agent = None
            self._text_agent = None

            compiled = self.compile()

            initial_state: WorkflowState = {
                "messages": [HumanMessage(content=query)],
                "next_agent": "",
                "execution_path": [],
                "iteration_count": 0,
            }

            async for step in compiled.astream(initial_state):
                logger.debug(f"Step: {step}")
                yield step

    def execute_sync(self, query: str) -> dict[str, Any]:
        """Synchronous execution wrapper."""
        import asyncio

        return asyncio.run(self.execute(query))

    def get_final_response(self, result: dict[str, Any]) -> str:
        """Extract the final response from workflow result."""
        messages = result.get("messages", [])

        # Find last AI message
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return str(msg.content)

        return "No response generated"

    def visualize(self) -> bytes | None:
        """
        Generate a visualization of the workflow graph.

        Returns:
            PNG image bytes if visualization succeeds, None otherwise
        """
        try:
            compiled = self.compile()
            return compiled.get_graph().draw_mermaid_png()
        except Exception as e:
            logger.warning(f"Could not generate visualization: {e}")
            return None


def create_workflow(
    max_iterations: int = 10,
    server_url: str | None = None,
) -> AgenticWorkflow:
    """
    Factory function to create an AgenticWorkflow.

    Args:
        max_iterations: Maximum routing iterations
        server_url: MCP server URL (defaults to settings)

    Returns:
        Configured AgenticWorkflow instance
    """
    return AgenticWorkflow(max_iterations=max_iterations, server_url=server_url)


async def run_workflow(query: str) -> str:
    """
    Convenience function to run a workflow and get the response.

    Args:
        query: User's request

    Returns:
        Final response string
    """
    workflow = create_workflow()
    result = await workflow.execute(query)
    return workflow.get_final_response(result)
