"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentic_ai_mcp.config import Settings


class ProviderType(StrEnum):
    """Supported LLM provider types."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Provides a unified interface for different LLM backends.
    """

    def __init__(self, model: str, settings: "Settings") -> None:
        """Initialize the provider.

        Args:
            model: Model name/identifier
            settings: Application settings
        """
        self.model = model
        self.settings = settings
        self._chat_model: Any = None

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Get the provider type."""
        ...

    @abstractmethod
    def get_chat_model(self) -> Any:
        """Get the LangChain chat model instance.

        Returns:
            A LangChain chat model (ChatAnthropic, ChatOpenAI, etc.)
        """
        ...

    @abstractmethod
    def get_api_key(self) -> str:
        """Get the API key for this provider.

        Returns:
            API key string

        Raises:
            ValueError: If API key is not configured
        """
        ...
