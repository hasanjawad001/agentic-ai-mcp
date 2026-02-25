"""Configuration."""

import os

from dotenv import load_dotenv

load_dotenv()


def get_anthropic_api_key() -> str:
    """Get Anthropic API key from environment."""
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    return key


def get_default_model() -> str:
    """Get default model from environment."""
    return os.getenv("DEFAULT_MODEL", "claude-haiku-4-5-20251001")  ## claude-sonnet-4-20250514
