"""AgenticAI - Simple/Complex agentic AI workflow with MCP tools."""

import asyncio
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from agentic_ai_mcp.config import Settings, get_default_model, get_settings
from agentic_ai_mcp.providers import get_provider
from agentic_ai_mcp.server import MCPServerManager
from agentic_ai_mcp.tools import ToolRegistry
from agentic_ai_mcp.workflows import PlanningWorkflow


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

    Example (OpenAI provider):
        ai = AgenticAI(provider="openai", model="gpt-4o-mini")
    """

    def __init__(
        self,
        name: str = "agentic-ai",
        host: str = "127.0.0.1",
        port: int = 8888,
        model: str | None = None,
        verbose: bool = False,
        mcp_url: str | None = None,
        provider: str = "anthropic",
        settings: Settings | None = None,
    ) -> None:
        """Initialize AgenticAI.

        Args:
            name: Name for the MCP server
            host: Host address for MCP server
            port: Port for MCP server
            model: LLM model name (defaults to settings)
            verbose: Enable verbose output
            mcp_url: URL of existing MCP server (client-only mode)
            provider: LLM provider ('anthropic' or 'openai')
            settings: Optional Settings override
        """
        # Settings
        self._settings = settings or get_settings()

        # Basic config
        self.host = host
        self.port = port
        self.model = model or get_default_model()
        self.verbose = verbose

        # Client-only mode
        self._mcp_url = mcp_url
        self._client_only = mcp_url is not None

        # Components
        self._server_manager = MCPServerManager(
            name=name,
            host=host,
            port=port,
            verbose=verbose,
        )
        self._tool_registry = ToolRegistry(verbose=verbose)

        # Provider
        self._provider = get_provider(
            provider_type=provider,
            model=self.model,
            settings=self._settings,
        )

        # Agent state
        self._agent: Any = None
        self._planning_workflow: PlanningWorkflow | None = None

    @property
    def tools(self) -> list[str]:
        """List of registered tool names."""
        return self._tool_registry.tool_names

    @property
    def _registered_funcs(self) -> list[Any]:
        """Backward compatibility: access registered functions."""
        return self._tool_registry.registered_funcs

    @property
    def _server_running(self) -> bool:
        """Backward compatibility: check if server is running."""
        return self._server_manager.is_running

    @property
    def _langchain_tools(self) -> list[Any]:
        """Backward compatibility: access LangChain tools."""
        return self._tool_registry.langchain_tools

    def register_tool(self, func: Any) -> None:
        """Register a function as an MCP tool.

        Args:
            func: Function with type hints and docstring

        Raises:
            RuntimeError: If called in client-only mode
        """
        if self._client_only:
            raise RuntimeError(
                "Cannot register tools in client-only mode. Use server mode instead."
            )
        self._tool_registry.register(func)

    def run_mcp_server(self) -> None:
        """Start the MCP server in background.

        Raises:
            RuntimeError: If called in client-only mode
        """
        if self._client_only:
            raise RuntimeError("Cannot start server in client-only mode.")

        self._server_manager.start(self._tool_registry.registered_funcs)

        if self.verbose:
            print(f"Tools: {self.tools}")

    def stop_mcp_server(self) -> None:
        """Stop the MCP server."""
        self._server_manager.stop()

    def _get_llm(self) -> Any:
        """Get the LLM instance from the provider."""
        return self._provider.get_chat_model()

    async def _load_tools(self) -> None:
        """Load tools from MCP server as LangChain tools."""
        if self._client_only:
            if self._mcp_url is None:
                raise RuntimeError("MCP URL not configured for client-only mode")
            mcp_url = self._mcp_url
        else:
            mcp_url = self._server_manager.mcp_url
        await self._tool_registry.load_from_mcp(mcp_url)

    async def run(self, prompt: str) -> str:
        """Run the agent with a prompt (simple ReAct).

        Args:
            prompt: Task for the agent

        Returns:
            Agent's response
        """
        # Start server if not running (skip in client-only mode)
        if not self._client_only and not self._server_manager.is_running:
            self.run_mcp_server()

        # Load tools if not loaded
        if not self._tool_registry.langchain_tools:
            await self._load_tools()

        # Create agent if not created
        if self._agent is None:
            self._agent = create_react_agent(
                self._get_llm(),
                self._tool_registry.langchain_tools,
            )

        if self.verbose:
            print(f"\n{'=' * 50}")
            print(f"PROMPT: {prompt}")
            print(f"{'=' * 50}\n")

        # Run
        result = await self._agent.ainvoke({"messages": [HumanMessage(content=prompt)]})

        # Process
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
        # Start server if not running (skip in client-only mode)
        if not self._client_only and not self._server_manager.is_running:
            self.run_mcp_server()

        # Load tools if not loaded
        if not self._tool_registry.langchain_tools:
            await self._load_tools()

        # Create planning workflow if not created
        if self._planning_workflow is None:
            self._planning_workflow = PlanningWorkflow(
                llm=self._get_llm(),
                tools=self._tool_registry.langchain_tools,
                tool_registry=self._tool_registry,
                max_retries=self._settings.max_retries,
                verbose=self.verbose,
            )

        return await self._planning_workflow.run(prompt)

    def run_sync(self, prompt: str) -> str:
        """Synchronous version of run()."""
        return asyncio.run(self.run(prompt))

    def run_with_planning_sync(self, prompt: str) -> str:
        """Synchronous version of run_with_planning()."""
        return asyncio.run(self.run_with_planning(prompt))
