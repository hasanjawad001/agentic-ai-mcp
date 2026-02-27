"""Tests for config module."""

import os
from unittest.mock import patch

import pytest

from agentic_ai_mcp.config import Settings, get_anthropic_api_key, get_default_model, get_settings


class TestSettings:
    """Tests for Settings class."""

    def test_default_values(self):
        """Test default settings values."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.default_model == "claude-haiku-4-5-20251001"
            assert settings.default_provider == "anthropic"
            assert settings.mcp_host == "127.0.0.1"
            assert settings.mcp_port == 8888
            assert settings.max_retries == 5
            assert settings.retry_base_delay == 1.0
            assert settings.retry_max_delay == 60.0

    def test_env_override(self):
        """Test environment variable override."""
        with patch.dict(
            os.environ,
            {
                "DEFAULT_MODEL": "custom-model",
                "MCP_PORT": "9999",
                "MAX_RETRIES": "10",
            },
        ):
            settings = Settings()
            assert settings.default_model == "custom-model"
            assert settings.mcp_port == 9999
            assert settings.max_retries == 10

    def test_port_validation(self):
        """Test port validation."""
        with patch.dict(os.environ, {"MCP_PORT": "0"}):
            with pytest.raises(ValueError):
                Settings()

        with patch.dict(os.environ, {"MCP_PORT": "99999"}):
            with pytest.raises(ValueError):
                Settings()

    def test_get_api_key_anthropic(self):
        """Test getting Anthropic API key."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            settings = Settings()
            assert settings.get_api_key("anthropic") == "test-key"

    def test_get_api_key_openai(self):
        """Test getting OpenAI API key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}):
            settings = Settings()
            assert settings.get_api_key("openai") == "test-openai-key"

    def test_get_api_key_missing(self):
        """Test error when API key is missing."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": ""}, clear=True):
            settings = Settings()
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY not set"):
                settings.get_api_key("anthropic")

    def test_get_api_key_unknown_provider(self):
        """Test error for unknown provider."""
        settings = Settings()
        with pytest.raises(ValueError, match="Unknown provider"):
            settings.get_api_key("unknown")


class TestBackwardCompatibility:
    """Tests for backward compatibility functions."""

    def test_get_anthropic_api_key(self):
        """Test backward compatible get_anthropic_api_key function."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            # Clear the cached settings
            get_settings.cache_clear()
            assert get_anthropic_api_key() == "test-key"

    def test_get_default_model(self):
        """Test backward compatible get_default_model function."""
        with patch.dict(os.environ, {"DEFAULT_MODEL": "custom-model"}):
            get_settings.cache_clear()
            assert get_default_model() == "custom-model"

    def test_get_default_model_default(self):
        """Test default model when not set."""
        with patch.dict(os.environ, {}, clear=True):
            get_settings.cache_clear()
            assert get_default_model() == "claude-haiku-4-5-20251001"
