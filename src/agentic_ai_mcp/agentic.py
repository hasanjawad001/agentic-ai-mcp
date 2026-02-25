"""AgenticAI - Simple agentic AI with MCP tools."""

import asyncio
import threading
import time
from collections.abc import Callable
from typing import Any

from fastmcp import Client, FastMCP
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from pydantic import create_model

from .config import get_anthropic_api_key, get_default_model


class AgenticAI:
    """Simple agentic AI with MCP tool serving.

    Example:
        ai = AgenticAI()

        def add(a: int, b: int) -> int:
            '''Add two numbers.'''
            return a + b

        ai.register_tool(add)
        ai.run_mcp_server()

        result = await ai.run("Calculate 2+3")
        print(result)
    """

    def __init__(
        self,
        name: str = "agentic-ai",
        host: str = "127.0.0.1",
        port: int = 8888,
        model: str | None = None,
        verbose: bool = True,
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

        # Agent
        self._agent: Any = None
        self._langchain_tools: list[StructuredTool] = []

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

        def _run():
            import asyncio
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

    async def run(self, prompt: str) -> str:
        """Run the agent with a prompt.

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
            llm = ChatAnthropic(
                model=self.model,
                api_key=get_anthropic_api_key(),
            )
            self._agent = create_react_agent(llm, self._langchain_tools)

        if self.verbose:
            print(f"\n{'='*50}")
            print(f"PROMPT: {prompt}")
            print(f"{'='*50}\n")

        # Run agent
        result = await self._agent.ainvoke({
            "messages": [HumanMessage(content=prompt)]
        })

        # Process and return response
        messages = result.get("messages", [])
        final_response = "No response"
        step = 0

        for msg in messages:
            if isinstance(msg, HumanMessage):   
                pass
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
            print(f"{'='*50}")
            print(f"RESULT: {final_response}")
            print(f"{'='*50}\n")

        return final_response

    def run_sync(self, prompt: str) -> str:
        """Synchronous version of run()."""
        return asyncio.run(self.run(prompt))
