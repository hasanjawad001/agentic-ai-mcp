"""OpenAI LLM provider implementation."""

from typing import TYPE_CHECKING, Any

from agentic_ai_mcp.providers.base import LLMProvider, ProviderType

if TYPE_CHECKING:
    pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider using LangChain.

    Requires the 'openai' optional dependency:
        pip install agentic-ai-mcp[openai]
    """

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

    def get_chat_model(self) -> Any:
        """Get the ChatOpenAI instance.

        Returns:
            Configured ChatOpenAI model

        Raises:
            ImportError: If langchain-openai is not installed
        """
        if self._chat_model is None:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as e:
                raise ImportError(
                    "OpenAI provider requires langchain-openai. "
                    "Install with: pip install agentic-ai-mcp[openai]"
                ) from e

            self._chat_model = ChatOpenAI(
                model=self.model,
                api_key=self.get_api_key(),  # type: ignore[arg-type]
            )
        return self._chat_model
