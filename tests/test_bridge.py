"""Tests for MCPToolBridge."""

import pytest

from agentic_ai_mcp import MCPToolBridge


class TestMCPToolBridge:
    """Tests for MCPToolBridge class."""

    def test_create_bridge(self):
        """Test creating a bridge."""
        bridge = MCPToolBridge()
        assert bridge.mcp_url == "http://localhost:8888/mcp"
        assert bridge.mcp_tools == []
        assert bridge.langchain_tools == []

    def test_create_bridge_custom_url(self):
        """Test creating bridge with custom URL."""
        bridge = MCPToolBridge("http://example.com:9000/mcp")
        assert bridge.mcp_url == "http://example.com:9000/mcp"

    def test_get_langchain_tools_without_loading(self):
        """Test that getting tools without loading raises error."""
        bridge = MCPToolBridge()
        with pytest.raises(RuntimeError, match="No tools loaded"):
            bridge.get_langchain_tools()

    def test_convert_schema_empty(self):
        """Test converting empty schema."""
        bridge = MCPToolBridge()
        result = bridge._convert_schema(None)
        assert result is None

        result = bridge._convert_schema({})
        assert result is None

    def test_convert_schema_with_properties(self):
        """Test converting schema with properties."""
        bridge = MCPToolBridge()
        schema = {
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["name"],
        }
        model = bridge._convert_schema(schema)
        assert model is not None
        assert "name" in model.model_fields
        assert "count" in model.model_fields
