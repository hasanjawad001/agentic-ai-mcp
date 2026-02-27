"""OpenAI LLM provider implementation."""

from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI

from agentic_ai_mcp.providers.base import LLMProvider, ProviderType

if TYPE_CHECKING:
    pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider using LangChain."""

    @property
    def provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.OPENAI

    def get_api_key(self) -> str:
        """Get the OpenAI API key.

        Returns:
            OpenAI API key

        Raises:
            ValueError: If OPENAI_API_KEY is not set
        """
        return self.settings.get_api_key("openai")

    def get_chat_model(self) -> ChatOpenAI:
        """Get the ChatOpenAI instance.

        Returns:
            Configured ChatOpenAI model
        """
        if self._chat_model is None:
            self._chat_model = ChatOpenAI(
                model=self.model,
                api_key=self.get_api_key(),  # type: ignore[arg-type]
            )
        result: ChatOpenAI = self._chat_model
        return result
