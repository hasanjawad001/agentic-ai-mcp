#!/usr/bin/env python3
"""
MCP Server Example

Demonstrates how to run the MCP server and connect to it with a client.

Usage:
    # Terminal 1: Start the server
    python examples/mcp_server_example.py --server

    # Terminal 2: Run the client
    python examples/mcp_server_example.py --client
"""

import argparse
import asyncio
import logging
import sys

from agentic_ai_mcp.mcp.client import MCPClient
from agentic_ai_mcp.mcp.server import run_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def run_client_example():
    """Example: Connecting to MCP server and calling tools."""
    print("\n" + "=" * 60)
    print("MCP Client Example")
    print("=" * 60)

    client = MCPClient()

    async with client.connect() as connected_client:
        # List available tools
        print("\nAvailable tools:")
        for tool in connected_client.tools:
            print(f"  - {tool['name']}: {tool.get('description', 'No description')}")

        # Call math tools
        print("\n" + "-" * 40)
        print("Math Tool Calls")
        print("-" * 40)

        result = await connected_client.call_tool("add", {"a": 10, "b": 20})
        print(f"add(10, 20) = {result}")

        result = await connected_client.call_tool("multiply", {"a": 7, "b": 8})
        print(f"multiply(7, 8) = {result}")

        result = await connected_client.call_tool("sqrt", {"number": 144})
        print(f"sqrt(144) = {result}")

        # Call text tools
        print("\n" + "-" * 40)
        print("Text Tool Calls")
        print("-" * 40)

        result = await connected_client.call_tool("to_uppercase", {"text": "hello world"})
        print(f"to_uppercase('hello world') = {result}")

        result = await connected_client.call_tool("reverse_text", {"text": "python"})
        print(f"reverse_text('python') = {result}")

        result = await connected_client.call_tool("count_chars", {"text": "agentic ai"})
        print(f"count_chars('agentic ai') = {result}")

        # Batch calls
        print("\n" + "-" * 40)
        print("Batch Tool Calls")
        print("-" * 40)

        calls = [
            ("add", {"a": 1, "b": 2}),
            ("add", {"a": 3, "b": 4}),
            ("add", {"a": 5, "b": 6}),
        ]
        results = await connected_client.batch_call_tools(calls)
        print(f"Batch add results: {results}")


def run_server_mode():
    """Run the MCP server."""
    print("\n" + "=" * 60)
    print("Starting MCP Server")
    print("=" * 60)
    print("\nPress Ctrl+C to stop the server.")
    print("Connect using: http://localhost:8888/mcp")
    print("=" * 60 + "\n")

    run_server()


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="MCP Server/Client Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start the server
    python mcp_server_example.py --server

    # Run the client (server must be running)
    python mcp_server_example.py --client
        """,
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run in server mode",
    )
    parser.add_argument(
        "--client",
        action="store_true",
        help="Run in client mode",
    )

    args = parser.parse_args()

    if args.server:
        run_server_mode()
    elif args.client:
        asyncio.run(run_client_example())
    else:
        parser.print_help()
        print("\nPlease specify --server or --client mode.")
        sys.exit(1)


if __name__ == "__main__":
    main()
