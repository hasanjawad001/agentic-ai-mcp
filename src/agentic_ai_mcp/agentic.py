"""AgenticAI - Simple/Complex agentic AI workflow with MCP tools."""

import asyncio
import contextlib
import multiprocessing
import operator
import os
import signal
import subprocess
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


def _run_server_process(
    name: str, host: str, port: int, funcs: list[Callable[..., Any]]
) -> None:
    """Run MCP server in a subprocess."""
    mcp = FastMCP(name)
    for func in funcs:
        mcp.tool()(func)
    asyncio.run(mcp.run_http_async(host=host, port=port))


class PlanningState(TypedDict):
    """State for the planning workflow."""

    task: str
    plan: list[str]
    current_step: int
    step_results: Annotated[list[str], operator.add]
    final_result: str


class AgenticAI:
    """Simple/Complex agentic AI workflow with MCP tool serving.

    Example (Server + Agent mode):
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

    Example (Client-only mode - connect to existing MCP server):
        ai = AgenticAI(mcp_url="http://192.168.1.100:8888/mcp")
        result = await ai.run("Calculate 2+3")
    """

    def __init__(
        self,
        name: str = "agentic-ai",
        host: str = "127.0.0.1",
        port: int = 8888,
        model: str | None = None,
        verbose: bool = False,
        mcp_url: str | None = None,
    ):
        self.host = host
        self.port = port
        self.model = model or get_default_model()
        self.verbose = verbose

        # client-only mode: connect to existing MCP server
        self._mcp_url = mcp_url
        self._client_only = mcp_url is not None

        # MCP server config
        self._name = name
        self._registered_funcs: list[Callable[..., Any]] = []  # store funcs for subprocess
        self._server_process: multiprocessing.Process | None = None
        self._server_running = False

        # agentic
        self._agent: Any = None
        self._planning_workflow: Any = None
        self._langchain_tools: list[StructuredTool] = []
        self._llm: ChatAnthropic | None = None

    @property
    def tools(self) -> list[str]:
        """List of registered tool names."""
        rf = self._registered_funcs.copy()
        tools = [f.__name__ for f in rf]
        return tools

    def register_tool(self, func: Callable[..., Any]) -> None:
        """Register a function as an MCP tool.

        Args:
            func: Function with type hints and docstring

        Raises:
            RuntimeError: If called in client-only mode
        """
        if self._client_only:
            raise RuntimeError("Cannot register tools in client-only mode. Use server mode instead.")
        self._registered_funcs.append(func)

    def run_mcp_server(self) -> None:
        """Start the MCP server in background.

        Raises:
            RuntimeError: If called in client-only mode
        """
        if self._client_only:
            raise RuntimeError("Cannot start server in client-only mode.")

        if self._server_running:
            return

        # start server
        self._server_process = multiprocessing.Process(
            target=_run_server_process,
            args=(self._name, self.host, self.port, self._registered_funcs),
            daemon=True,
        )
        self._server_process.start()

        # wait for server to be ready
        self._wait_for_server()
        self._server_running = True

        if self.verbose:
            print(f"MCP Server running at http://{self.host}:{self.port}/mcp")
            print(f"Tools: {self.tools}")

    def stop_mcp_server(self) -> None:
        """Stop the MCP server."""
        if not self._server_running and not self._get_pids_on_port():
            if self.verbose:
                print("Server is not running.")
            return

        self._kill_process_on_port()

        # clean up process reference
        if self._server_process is not None:
            with contextlib.suppress(Exception):
                self._server_process.join(timeout=1)
            self._server_process = None

        self._server_running = False

        if self.verbose:
            print("MCP Server stopped.")

    def _get_pids_on_port(self) -> list[int]:
        """Get PIDs of processes using the server port."""
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{self.port}"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                return [int(pid) for pid in result.stdout.strip().split("\n")]
        except (subprocess.SubprocessError, ValueError, FileNotFoundError):
            pass
        return []

    def _kill_process_on_port(self) -> None:
        """Kill any process using the server port."""
        pids = self._get_pids_on_port()
        for pid in pids:
            with contextlib.suppress(ProcessLookupError, OSError):
                os.kill(pid, signal.SIGKILL)

        time.sleep(0.5)

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
        mcp_url = self._mcp_url if self._client_only else f"http://{self.host}:{self.port}/mcp"

        if self.verbose:
            print(f"Loading tools from: {mcp_url}")

        async with Client(mcp_url) as client:
            mcp_tools = await client.list_tools()

        self._langchain_tools = []
        for tool in mcp_tools:
            lc_tool = self._convert_to_langchain(tool, mcp_url)
            self._langchain_tools.append(lc_tool)

        if self.verbose:
            print(f"Loaded tools: {[t.name for t in self._langchain_tools]}")

    def _convert_to_langchain(self, mcp_tool: Any, mcp_url: str) -> StructuredTool:
        """Convert MCP tool to LangChain StructuredTool."""
        schema = mcp_tool.inputSchema if hasattr(mcp_tool, "inputSchema") else {}
        args_model = self._create_args_model(schema)

        async def acall_tool(**kwargs: Any) -> Any:
            # filter out None values so MCP tool can use its defaults
            filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
            async with Client(mcp_url) as client:
                return await client.call_tool(mcp_tool.name, filtered_kwargs)

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
            elif "default" in prop:
                # use actual default value from schema
                fields[name] = (prop_type, prop["default"])
            else:
                # optional with no default - allow None
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
        # start server if not running (skip in client-only mode)
        if not self._client_only and not self._server_running:
            self.run_mcp_server()

        # load tools if not loaded
        if not self._langchain_tools:
            await self._load_tools()

        # create agent if not created
        if self._agent is None:
            self._agent = create_react_agent(self._get_llm(), self._langchain_tools)

        if self.verbose:
            print(f"\n{'=' * 50}")
            print(f"PROMPT: {prompt}")
            print(f"{'=' * 50}\n")

        # run
        result = await self._agent.ainvoke({"messages": [HumanMessage(content=prompt)]})

        # process
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
        # start server if not running (skip in client-only mode)
        if not self._client_only and not self._server_running:
            self.run_mcp_server()

        # load tools if not loaded
        if not self._langchain_tools:
            await self._load_tools()

        # create planning workflow if not created
        if self._planning_workflow is None:
            self._planning_workflow = self._create_planning_workflow()

        if self.verbose:
            print(f"\n{'=' * 50}")
            print("PLANNING MODE")
            print(f"TASK: {prompt}")
            print(f"{'=' * 50}\n")

        # run
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

    def _format_tool_signature(self, tool: StructuredTool) -> str:
        """Format tool with its argument signature."""
        if tool.args_schema:
            schema = tool.args_schema.model_json_schema()
            props = schema.get("properties", {})
            required = set(schema.get("required", []))
            args = []
            for name, prop in props.items():
                arg_type = prop.get("type", "any")
                if name in required:
                    args.append(f"{name}: {arg_type}")
                else:
                    default = prop.get("default", "None")
                    args.append(f"{name}: {arg_type} = {default}")
            args_str = ", ".join(args)
            return f"{tool.name}({args_str}): {tool.description}"
        return f"{tool.name}(): {tool.description}"

    def _create_planning_workflow(self) -> Any:
        """Create the planning workflow using LangGraph."""
        llm = self._get_llm()
        tools = self._langchain_tools
        verbose = self.verbose

        # format tools with signatures
        tool_signatures = [self._format_tool_signature(t) for t in tools]

        # planner
        async def planner(state: PlanningState) -> dict[str, Any]:
            task = state["task"]

            plan_prompt = f"""Break down this task into clear, executable steps.
Each step should use the available tools efficiently.

Available tools:
{chr(10).join(f"  - {sig}" for sig in tool_signatures)}

Task: {task}

Respond with ONLY a numbered list of steps. Reference previous step results when needed.

Example for task "calculate 2+3, then greet Bob that many times":
1. Use add(a=2, b=3) to calculate the sum
2. Use greet(name="Bob", times=result from step 1) to greet Bob

Example for task "multiply 4 and 5, then add 10":
1. Use multiply(a=4, b=5) to get the product
2. Use add(a=result from step 1, b=10) to add 10"""


            response = await llm.ainvoke([HumanMessage(content=plan_prompt)])
            plan_text = str(response.content)

            # parse
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

        # executor
        async def executor(state: PlanningState) -> dict[str, Any]:
            plan = state["plan"]
            current_step = state["current_step"]
            step_results = state.get("step_results", [])

            if current_step >= len(plan):
                return {"current_step": current_step}

            step = plan[current_step]

            if verbose:
                print(f"EXECUTING STEP {current_step + 1}/{len(plan)}: {step}")
                print("-" * 40)

            # build context with previous results
            if step_results:
                context = "Previous results:\n" + "\n".join(step_results) + "\n\n"
                step_prompt = f"{context}Current task: {step}"
            else:
                step_prompt = step

            # mini agent
            step_agent = create_react_agent(llm, tools)
            result = await step_agent.ainvoke({"messages": [HumanMessage(content=step_prompt)]})

            # extract result
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

        # conditional check
        def should_continue(state: PlanningState) -> str:
            if state["current_step"] < len(state["plan"]):
                return "executor"
            return "synthesizer"

        # synthesizer
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
                print("Synthesis complete.")
                print("=" * 50)

            return {"final_result": str(response.content)}

        # workflow setup
        workflow = StateGraph(PlanningState)
        workflow.add_node("planner", planner)
        workflow.add_node("executor", executor)
        workflow.add_node("synthesizer", synthesizer)
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
