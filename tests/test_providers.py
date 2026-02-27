"""Tests for providers module."""

import os
from unittest.mock import patch

import pytest

from agentic_ai_mcp.config import Settings
from agentic_ai_mcp.providers import LLMProvider, ProviderType, get_provider
from agentic_ai_mcp.providers.anthropic import AnthropicProvider
from agentic_ai_mcp.providers.openai import OpenAIProvider


class TestProviderType:
    """Tests for ProviderType enum."""

    def test_provider_types(self):
        """Test provider type values."""
        assert ProviderType.ANTHROPIC.value == "anthropic"
        assert ProviderType.OPENAI.value == "openai"


class TestGetProvider:
    """Tests for get_provider factory function."""

    def test_get_anthropic_provider(self):
        """Test getting Anthropic provider."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            settings = Settings()
            provider = get_provider("anthropic", "claude-3-haiku", settings)
            assert isinstance(provider, AnthropicProvider)
            assert provider.model == "claude-3-haiku"
            assert provider.provider_type == ProviderType.ANTHROPIC

    def test_get_openai_provider(self):
        """Test getting OpenAI provider."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            settings = Settings()
            provider = get_provider("openai", "gpt-4o-mini", settings)
            assert isinstance(provider, OpenAIProvider)
            assert provider.model == "gpt-4o-mini"
            assert provider.provider_type == ProviderType.OPENAI

    def test_get_provider_with_enum(self):
        """Test getting provider with ProviderType enum."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            settings = Settings()
            provider = get_provider(ProviderType.ANTHROPIC, "claude-3-haiku", settings)
            assert isinstance(provider, AnthropicProvider)

    def test_get_provider_unknown(self):
        """Test error for unknown provider."""
        settings = Settings()
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("unknown", "model", settings)


class TestAnthropicProvider:
    """Tests for AnthropicProvider."""

    def test_provider_type(self):
        """Test provider type."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            settings = Settings()
            provider = AnthropicProvider("claude-3-haiku", settings)
            assert provider.provider_type == ProviderType.ANTHROPIC

    def test_get_api_key(self):
        """Test getting API key."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            settings = Settings()
            provider = AnthropicProvider("claude-3-haiku", settings)
            assert provider.get_api_key() == "test-key"

    def test_get_api_key_missing(self):
        """Test error when API key is missing."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=True):
            settings = Settings()
            provider = AnthropicProvider("claude-3-haiku", settings)
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY not set"):
                provider.get_api_key()


class TestOpenAIProvider:
    """Tests for OpenAIProvider."""

    def test_provider_type(self):
        """Test provider type."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            settings = Settings()
            provider = OpenAIProvider("gpt-4o-mini", settings)
            assert provider.provider_type == ProviderType.OPENAI

    def test_get_api_key(self):
        """Test getting API key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            settings = Settings()
            provider = OpenAIProvider("gpt-4o-mini", settings)
            assert provider.get_api_key() == "test-key"

    def test_get_api_key_missing(self):
        """Test error when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            provider = OpenAIProvider("gpt-4o-mini", settings)
            with pytest.raises(ValueError, match="OPENAI_API_KEY not set"):
                provider.get_api_key()
