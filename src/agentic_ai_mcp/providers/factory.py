"""Provider factory for creating LLM providers."""

from typing import TYPE_CHECKING

from agentic_ai_mcp.providers.anthropic import AnthropicProvider
from agentic_ai_mcp.providers.base import LLMProvider, ProviderType
from agentic_ai_mcp.providers.openai import OpenAIProvider

if TYPE_CHECKING:
    from agentic_ai_mcp.config import Settings


def get_provider(
    provider_type: str | ProviderType,
    model: str,
    settings: "Settings",
) -> LLMProvider:
    """Factory function to create an LLM provider.

    Args:
        provider_type: Provider type ('anthropic', 'openai', or ProviderType enum)
        model: Model name/identifier
        settings: Application settings

    Returns:
        Configured LLM provider instance

    Raises:
        ValueError: If provider_type is not supported
    """
    # Convert string to enum if needed
    if isinstance(provider_type, str):
        try:
            provider_type = ProviderType(provider_type.lower())
        except ValueError as e:
            raise ValueError(
                f"Unknown provider: {provider_type}. "
                f"Supported providers: {[p.value for p in ProviderType]}"
            ) from e

    if provider_type == ProviderType.ANTHROPIC:
        return AnthropicProvider(model=model, settings=settings)
    elif provider_type == ProviderType.OPENAI:
        return OpenAIProvider(model=model, settings=settings)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
