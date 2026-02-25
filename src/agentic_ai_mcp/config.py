"""Simple configuration for Agentic AI MCP."""

import os
from functools import lru_cache


@lru_cache
def get_anthropic_api_key() -> str:
    """Get Anthropic API key from environment."""
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    return key


@lru_cache
def get_mcp_url() -> str:
    """Get MCP server URL from environment."""
    host = os.getenv("MCP_SERVER_HOST", "localhost")
    port = os.getenv("MCP_SERVER_PORT", "8888")
    return f"http://{host}:{port}/mcp"


@lru_cache
def get_default_model() -> str:
    """Get default LLM model."""
    return os.getenv("DEFAULT_MODEL", "claude-sonnet-4-20250514")
