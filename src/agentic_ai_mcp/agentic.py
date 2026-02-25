"""AgenticAI - Simple agentic AI with MCP tools."""

import asyncio
import operator
import threading
import time
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from fastmcp import Client, FastMCP
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import create_model

from .config import get_anthropic_api_key, get_default_model


class PlanningState(TypedDict):
    """State for the planning workflow."""

    task: str
    plan: list[str]
    current_step: int
    step_results: Annotated[list[str], operator.add]
    final_result: str


class AgenticAI:
    """Simple agentic AI with MCP tool serving.

    Example:
        ai = AgenticAI()

        def add(a: int, b: int) -> int:
            '''Add two numbers.'''
            return a + b

        ai.register_tool(add)
        ai.run_mcp_server()

        # Simple execution
        result = await ai.run("Calculate 2+3")

        # Complex tasks with planning
        result = await ai.run_with_planning("Research X, then do Y based on results")
    """

    def __init__(
        self,
        name: str = "agentic-ai",
        host: str = "127.0.0.1",
        port: int = 8888,
        model: str | None = None,
        verbose: bool = False,
    ):
        self.host = host
        self.port = port
        self.model = model or get_default_model()
        self.verbose = verbose

        # MCP server
        self.mcp = FastMCP(name)
        self._tools: list[str] = []
        self._server_thread: threading.Thread | None = None
        self._server_running = False

        # Agents
        self._agent: Any = None
        self._planning_workflow: Any = None
        self._langchain_tools: list[StructuredTool] = []
        self._llm: ChatAnthropic | None = None

    @property
    def tools(self) -> list[str]:
        """List of registered tool names."""
        return self._tools.copy()

    def register_tool(self, func: Callable[..., Any]) -> None:
        """Register a function as an MCP tool.

        Args:
            func: Function with type hints and docstring
        """
        self.mcp.tool()(func)
        self._tools.append(func.__name__)

    def run_mcp_server(self) -> None:
        """Start the MCP server in background."""
        if self._server_running:
            return

        def _run() -> None:
            asyncio.run(self.mcp.run_http_async(host=self.host, port=self.port))

        self._server_thread = threading.Thread(target=_run, daemon=True)
        self._server_thread.start()

        # Wait for server to be ready
        self._wait_for_server()
        self._server_running = True

        if self.verbose:
            print(f"MCP Server running at http://{self.host}:{self.port}/mcp")
            print(f"Tools: {self.tools}")

    def _wait_for_server(self, timeout: float = 10.0) -> None:
        """Wait for server to be ready."""
        import socket

        start = time.time()
        while time.time() - start < timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                if sock.connect_ex((self.host, self.port)) == 0:
                    sock.close()
                    time.sleep(0.5)
                    return
                sock.close()
            except OSError:
                pass
            time.sleep(0.2)
        raise TimeoutError(f"Server did not start within {timeout}s")

    async def _load_tools(self) -> None:
        """Load tools from MCP server as LangChain tools."""
        mcp_url = f"http://{self.host}:{self.port}/mcp"

        async with Client(mcp_url) as client:
            mcp_tools = await client.list_tools()

        self._langchain_tools = []
        for tool in mcp_tools:
            lc_tool = self._convert_to_langchain(tool, mcp_url)
            self._langchain_tools.append(lc_tool)

    def _convert_to_langchain(self, mcp_tool: Any, mcp_url: str) -> StructuredTool:
        """Convert MCP tool to LangChain StructuredTool."""
        schema = mcp_tool.inputSchema if hasattr(mcp_tool, "inputSchema") else {}
        args_model = self._create_args_model(schema)

        async def acall_tool(**kwargs: Any) -> Any:
            async with Client(mcp_url) as client:
                return await client.call_tool(mcp_tool.name, kwargs)

        def call_tool(**kwargs: Any) -> Any:
            return asyncio.run(acall_tool(**kwargs))

        return StructuredTool(
            name=mcp_tool.name,
            description=mcp_tool.description or mcp_tool.name,
            func=call_tool,
            coroutine=acall_tool,
            args_schema=args_model,
        )

    def _create_args_model(self, schema: dict[str, Any]) -> Any:
        """Create Pydantic model from JSON schema."""
        if not schema or "properties" not in schema:
            return None

        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
        }

        fields: dict[str, Any] = {}
        for name, prop in properties.items():
            prop_type = type_map.get(prop.get("type", "string"), str)
            if name in required:
                fields[name] = (prop_type, ...)
            else:
                fields[name] = (prop_type | None, None)

        return create_model("ToolArgs", **fields)

    def _get_llm(self) -> ChatAnthropic:
        """Get or create LLM instance."""
        if self._llm is None:
            self._llm = ChatAnthropic(
                model=self.model,  # type: ignore[call-arg]
                api_key=get_anthropic_api_key(),  # type: ignore[arg-type]
            )
        return self._llm

    async def run(self, prompt: str) -> str:
        """Run the agent with a prompt (simple ReAct).

        Args:
            prompt: Task for the agent

        Returns:
            Agent's response
        """
        # Start server if not running
        if not self._server_running:
            self.run_mcp_server()

        # Load tools if not loaded
        if not self._langchain_tools:
            await self._load_tools()

        # Create agent if not created
        if self._agent is None:
            self._agent = create_react_agent(self._get_llm(), self._langchain_tools)

        if self.verbose:
            print(f"\n{'=' * 50}")
            print(f"PROMPT: {prompt}")
            print(f"{'=' * 50}\n")

        # Run agent
        result = await self._agent.ainvoke({"messages": [HumanMessage(content=prompt)]})

        # Process and return response
        messages = result.get("messages", [])
        final_response = "No response"
        step = 0

        for msg in messages:
            if isinstance(msg, AIMessage):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        step += 1
                        if self.verbose:
                            print(f"STEP {step}: {tc['name']}({tc['args']})")
                if msg.content and not (hasattr(msg, "tool_calls") and msg.tool_calls):
                    final_response = str(msg.content)
            elif isinstance(msg, ToolMessage) and self.verbose:
                print(f"  → {msg.content}\n")

        if self.verbose:
            print(f"{'=' * 50}")
            print(f"RESULT: {final_response}")
            print(f"{'=' * 50}\n")

        return final_response

    async def run_with_planning(self, prompt: str) -> str:
        """Run the agent with planning for complex tasks.

        Uses LangGraph StateGraph to:
        1. Plan: Break down the task into steps
        2. Execute: Run each step with tools
        3. Synthesize: Combine results into final response

        Args:
            prompt: Complex task for the agent

        Returns:
            Agent's response
        """
        # Start server if not running
        if not self._server_running:
            self.run_mcp_server()

        # Load tools if not loaded
        if not self._langchain_tools:
            await self._load_tools()

        # Create planning workflow if not created
        if self._planning_workflow is None:
            self._planning_workflow = self._create_planning_workflow()

        if self.verbose:
            print(f"\n{'=' * 50}")
            print("PLANNING MODE")
            print(f"TASK: {prompt}")
            print(f"{'=' * 50}\n")

        # Run planning workflow
        initial_state: PlanningState = {
            "task": prompt,
            "plan": [],
            "current_step": 0,
            "step_results": [],
            "final_result": "",
        }

        result = await self._planning_workflow.ainvoke(initial_state)

        if self.verbose:
            print(f"\n{'=' * 50}")
            print(f"FINAL RESULT: {result['final_result']}")
            print(f"{'=' * 50}\n")

        return str(result["final_result"])

    def _create_planning_workflow(self) -> Any:
        """Create the planning workflow using LangGraph."""
        llm = self._get_llm()
        tools = self._langchain_tools
        verbose = self.verbose

        # Planner node: breaks down the task
        async def planner(state: PlanningState) -> dict[str, Any]:
            task = state["task"]

            plan_prompt = f"""Break down this task into clear, executable steps.
Each step should be a single action that can be done with the available tools.

Available tools: {[t.name + ": " + t.description for t in tools]}

Task: {task}

Respond with ONLY a numbered list of steps, nothing else. Example:
1. First step
2. Second step
3. Third step"""

            response = await llm.ainvoke([HumanMessage(content=plan_prompt)])
            plan_text = str(response.content)

            # Parse steps
            steps = []
            for line in plan_text.strip().split("\n"):
                line = line.strip()
                if line and line[0].isdigit():
                    # Remove numbering
                    step = line.lstrip("0123456789.").strip()
                    if step:
                        steps.append(step)

            if verbose:
                print("PLAN:")
                for i, step in enumerate(steps, 1):
                    print(f"  {i}. {step}")
                print()

            return {"plan": steps, "current_step": 0}

        # Executor node: executes one step at a time
        async def executor(state: PlanningState) -> dict[str, Any]:
            plan = state["plan"]
            current_step = state["current_step"]

            if current_step >= len(plan):
                return {"current_step": current_step}

            step = plan[current_step]

            if verbose:
                print(f"EXECUTING STEP {current_step + 1}/{len(plan)}: {step}")
                print("-" * 40)

            # Create a mini ReAct agent for this step
            step_agent = create_react_agent(llm, tools)
            result = await step_agent.ainvoke({"messages": [HumanMessage(content=step)]})

            # Extract result and show all messages if verbose
            messages = result.get("messages", [])
            step_result = "No result"
            tool_call_count = 0

            for msg in messages:
                if isinstance(msg, AIMessage):
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tool_call_count += 1
                            if verbose:
                                print(f"  TOOL CALL {tool_call_count}: {tc['name']}({tc['args']})")
                    elif msg.content:
                        step_result = str(msg.content)
                        if verbose:
                            print(f"  AI RESPONSE: {msg.content}")
                elif isinstance(msg, ToolMessage) and verbose:
                    print(f"    → {msg.content}")

            if verbose:
                print("-" * 40)
                print(f"STEP {current_step + 1} COMPLETE\n")

            return {
                "step_results": [f"Step {current_step + 1}: {step_result}"],
                "current_step": current_step + 1,
            }

        # Check if more steps to execute
        def should_continue(state: PlanningState) -> str:
            if state["current_step"] < len(state["plan"]):
                return "executor"
            return "synthesizer"

        # Synthesizer node: combines all results
        async def synthesizer(state: PlanningState) -> dict[str, Any]:
            task = state["task"]
            step_results = state["step_results"]

            if verbose:
                print("=" * 50)
                print("SYNTHESIZING RESULTS")
                print("=" * 50)
                print(f"Original task: {task}")
                print()
                print("Step results collected:")
                for result in step_results:
                    print(f"  • {result}")
                print()

            synth_prompt = f"""You completed a multi-step task. Synthesize the results into a final response.

Original task: {task}

Step results:
{chr(10).join(step_results)}

Provide a clear, concise final response that addresses the original task."""

            if verbose:
                print("Sending synthesis prompt...")

            response = await llm.ainvoke([HumanMessage(content=synth_prompt)])

            if verbose:
                print(f"Synthesis complete.")
                print("=" * 50)

            return {"final_result": str(response.content)}

        # Build the graph
        workflow = StateGraph(PlanningState)

        # Add nodes
        workflow.add_node("planner", planner)
        workflow.add_node("executor", executor)
        workflow.add_node("synthesizer", synthesizer)

        # Add edges
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "executor")
        workflow.add_conditional_edges("executor", should_continue)
        workflow.add_edge("synthesizer", END)

        return workflow.compile()

    def run_sync(self, prompt: str) -> str:
        """Synchronous version of run()."""
        return asyncio.run(self.run(prompt))

    def run_with_planning_sync(self, prompt: str) -> str:
        """Synchronous version of run_with_planning()."""
        return asyncio.run(self.run_with_planning(prompt))
