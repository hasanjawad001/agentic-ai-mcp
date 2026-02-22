"""MCP Server implementation for serving tools over HTTP."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from agentic_ai_mcp.config.settings import get_settings
from agentic_ai_mcp.tools.math_tools import MathTools
from agentic_ai_mcp.tools.text_tools import TextTools

logger = logging.getLogger(__name__)


def create_mcp_server(
    name: str = "agentic-ai-tools",
    host: str = "0.0.0.0",
    port: int = 8888,
) -> FastMCP:
    """
    Create and configure an MCP server with all available tools.

    Args:
        name: Name for the MCP server
        host: Host address to bind to
        port: Port number to listen on

    Returns:
        Configured FastMCP server instance
    """
    mcp = FastMCP(name, host=host, port=port)

    # Register Math Tools
    @mcp.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers together and return the sum."""
        return MathTools.add(a, b)

    @mcp.tool()
    def subtract(a: int, b: int) -> int:
        """Subtract the second number from the first and return the difference."""
        return MathTools.subtract(a, b)

    @mcp.tool()
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers together and return the product."""
        return MathTools.multiply(a, b)

    @mcp.tool()
    def divide(a: float, b: float) -> float:
        """Divide the first number by the second and return the quotient."""
        return MathTools.divide(a, b)

    @mcp.tool()
    def power(base: float, exponent: float) -> float:
        """Raise base to the power of exponent."""
        return MathTools.power(base, exponent)

    @mcp.tool()
    def sqrt(number: float) -> float:
        """Calculate the square root of a non-negative number."""
        return MathTools.sqrt(number)

    # Register Text Tools
    @mcp.tool()
    def to_uppercase(text: str) -> str:
        """Convert text to uppercase letters."""
        return TextTools.to_uppercase(text)

    @mcp.tool()
    def to_lowercase(text: str) -> str:
        """Convert text to lowercase letters."""
        return TextTools.to_lowercase(text)

    @mcp.tool()
    def reverse_text(text: str) -> str:
        """Reverse the order of characters in the text."""
        return TextTools.reverse_text(text)

    @mcp.tool()
    def count_chars(text: str) -> int:
        """Count the number of characters in the text."""
        return TextTools.count_chars(text)

    @mcp.tool()
    def count_words(text: str) -> int:
        """Count the number of words in the text."""
        return TextTools.count_words(text)

    @mcp.tool()
    def capitalize(text: str) -> str:
        """Capitalize the first letter of each word."""
        return TextTools.capitalize(text)

    @mcp.tool()
    def strip_whitespace(text: str) -> str:
        """Remove leading and trailing whitespace from text."""
        return TextTools.strip_whitespace(text)

    @mcp.tool()
    def search_replace(text: str, search: str, replace: str) -> str:
        """Replace all occurrences of search pattern with replacement string."""
        return TextTools.search_replace(text, search, replace)

    logger.info(f"Created MCP server '{name}' with math and text tools")
    return mcp


def run_server(
    host: str | None = None,
    port: int | None = None,
    transport: str = "streamable-http",
) -> None:
    """
    Run the MCP server.

    Args:
        host: Host address (defaults to settings)
        port: Port number (defaults to settings)
        transport: Transport protocol to use
    """
    settings = get_settings()
    host = host or settings.mcp_server_host
    port = port or settings.mcp_server_port

    mcp = create_mcp_server(host=host, port=port)

    logger.info(f"Starting MCP server on {host}:{port}")
    print(f"MCP Server running at http://{host}:{port}/mcp")

    mcp.run(transport=transport)


def main() -> None:
    """Entry point for the MCP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        run_server()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
