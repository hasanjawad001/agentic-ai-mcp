"""Application settings and configuration management."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys
    anthropic_api_key: str = Field(
        ...,
        description="Anthropic API key for Claude access",
    )

    # MCP Server Configuration
    mcp_server_host: str = Field(
        default="0.0.0.0",
        description="Host address for MCP server",
    )
    mcp_server_port: int = Field(
        default=8888,
        ge=1,
        le=65535,
        description="Port number for MCP server",
    )

    # LLM Configuration
    default_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Default Claude model to use",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=8192,
        description="Maximum tokens for LLM responses",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Temperature for LLM sampling",
    )

    # Agent Configuration
    max_agent_iterations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum iterations per agent execution",
    )
    agent_timeout_seconds: int = Field(
        default=300,
        ge=10,
        le=3600,
        description="Timeout for agent execution in seconds",
    )

    # Logging Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    log_format: Literal["json", "console"] = Field(
        default="console",
        description="Log output format",
    )

    @field_validator("anthropic_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate that API key is not empty and has expected format."""
        if not v or not v.strip():
            raise ValueError("ANTHROPIC_API_KEY cannot be empty")
        if not v.startswith("sk-"):
            raise ValueError("ANTHROPIC_API_KEY should start with 'sk-'")
        return v.strip()

    @property
    def mcp_server_url(self) -> str:
        """Get the full MCP server URL."""
        return f"http://{self.mcp_server_host}:{self.mcp_server_port}/mcp"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
