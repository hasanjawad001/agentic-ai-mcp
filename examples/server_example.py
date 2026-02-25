#!/usr/bin/env python3
"""
Server Example - How to define and serve tools.

Run this file to start the MCP server:
    python examples/server_example.py

Then in another terminal, run the workflow example.
"""

from agentic_ai_mcp import MCPServer

# Create server
server = MCPServer()


# Define tools using decorator
@server.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@server.tool()
def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b


@server.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@server.tool()
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}! Nice to meet you."


@server.tool()
def farewell(name: str) -> str:
    """Say goodbye to someone."""
    return f"Goodbye, {name}! See you next time."


if __name__ == "__main__":
    print("Starting MCP Server with custom tools...")
    print(f"Tools registered: {server.tools}")
    print()
    server.run()
