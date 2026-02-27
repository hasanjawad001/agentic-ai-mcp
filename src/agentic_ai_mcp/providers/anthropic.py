"""Anthropic LLM provider implementation."""

from typing import TYPE_CHECKING

from langchain_anthropic import ChatAnthropic

from agentic_ai_mcp.providers.base import LLMProvider, ProviderType

if TYPE_CHECKING:
    pass


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider using LangChain."""

    @property
    def provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.ANTHROPIC

    def get_api_key(self) -> str:
        """Get the Anthropic API key.

        Returns:
            Anthropic API key

        Raises:
            ValueError: If ANTHROPIC_API_KEY is not set
        """
        return self.settings.get_api_key("anthropic")

    def get_chat_model(self) -> ChatAnthropic:
        """Get the ChatAnthropic instance.

        Returns:
            Configured ChatAnthropic model
        """
        if self._chat_model is None:
            self._chat_model = ChatAnthropic(
                model=self.model,  # type: ignore[call-arg]
                api_key=self.get_api_key(),  # type: ignore[arg-type]
            )
        result: ChatAnthropic = self._chat_model
        return result
