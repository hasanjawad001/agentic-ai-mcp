"""Configuration with Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable support.

    Environment variables:
        ANTHROPIC_API_KEY: API key for Anthropic
        OPENAI_API_KEY: API key for OpenAI
        DEFAULT_MODEL: Default model to use (default: claude-haiku-4-5-20251001)
        DEFAULT_PROVIDER: Default LLM provider (default: anthropic)
        MCP_HOST: Default MCP server host (default: 127.0.0.1)
        MCP_PORT: Default MCP server port (default: 8888)
        MAX_RETRIES: Maximum retry attempts for API calls (default: 5)
        RETRY_BASE_DELAY: Base delay for exponential backoff (default: 1.0)
        RETRY_MAX_DELAY: Maximum delay for retries (default: 60.0)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")

    # Model settings
    default_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Default model to use",
    )
    default_provider: Literal["anthropic", "openai"] = Field(
        default="anthropic",
        description="Default LLM provider",
    )

    # MCP Server settings
    mcp_host: str = Field(default="127.0.0.1", description="MCP server host")
    mcp_port: int = Field(default=8888, ge=1, le=65535, description="MCP server port")

    # Retry settings
    max_retries: int = Field(default=5, ge=0, le=20, description="Maximum retry attempts")
    retry_base_delay: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Base delay for exponential backoff"
    )
    retry_max_delay: float = Field(
        default=60.0, ge=1.0, le=300.0, description="Maximum delay for retries"
    )

    @field_validator("mcp_port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port is in valid range."""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v

    def get_api_key(self, provider: str) -> str:
        """Get API key for a specific provider.

        Args:
            provider: Provider name ('anthropic' or 'openai')

        Returns:
            API key for the provider

        Raises:
            ValueError: If API key is not set
        """
        if provider == "anthropic":
            if not self.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            return self.anthropic_api_key
        elif provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY not set")
            return self.openai_api_key
        else:
            raise ValueError(f"Unknown provider: {provider}")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Backward compatibility functions
def get_anthropic_api_key() -> str:
    """Get Anthropic API key from environment.

    Backward compatible function for existing code.
    """
    return get_settings().get_api_key("anthropic")


def get_default_model() -> str:
    """Get default model from environment.

    Backward compatible function for existing code.
    """
    return get_settings().default_model
