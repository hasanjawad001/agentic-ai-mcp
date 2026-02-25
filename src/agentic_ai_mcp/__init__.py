"""
Agentic AI MCP - A lightweight agentic AI framework with MCP tool serving.

Simple usage:

    from agentic_ai_mcp import AgenticAI

    ai = AgenticAI()

    def add(a: int, b: int) -> int:
        '''Add two numbers.'''
        return a + b

    def greet(name: str, times: int = 1) -> str:
        '''Greet someone.'''
        return ("Hello, " + name + "! ") * times

    ai.register_tool(add)
    ai.register_tool(greet)

    result = await ai.run("Calculate 2+3 and greet Tom the result times")
    print(result)

For advanced usage (separate server/client), see MCPServer and AgenticWorkflow.
"""

from agentic_ai_mcp.agentic import AgenticAI
from agentic_ai_mcp.bridge import MCPToolBridge
from agentic_ai_mcp.server import MCPServer, create_server
from agentic_ai_mcp.workflow import AgenticWorkflow, run_workflow

__version__ = "0.5.0"

__all__ = [
    # Main interface
    "AgenticAI",
    # Advanced: Server
    "MCPServer",
    "create_server",
    # Advanced: Bridge
    "MCPToolBridge",
    # Advanced: Workflow
    "AgenticWorkflow",
    "run_workflow",
    # Version
    "__version__",
]
