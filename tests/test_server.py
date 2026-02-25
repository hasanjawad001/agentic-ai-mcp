"""Tests for MCPServer."""


from agentic_ai_mcp import MCPServer


class TestMCPServer:
    """Tests for MCPServer class."""

    def test_create_server(self):
        """Test creating a server."""
        server = MCPServer()
        assert server.host == "0.0.0.0"
        assert server.port == 8888
        assert server.tools == []

    def test_create_server_custom_port(self):
        """Test creating server with custom port."""
        server = MCPServer(port=9000)
        assert server.port == 9000

    def test_register_tool_decorator(self):
        """Test registering a tool with decorator."""
        server = MCPServer()

        @server.tool()
        def add(a: int, b: int) -> int:
            return a + b

        assert "add" in server.tools

    def test_register_tool_method(self):
        """Test registering a tool with add_tool method."""
        server = MCPServer()

        def multiply(a: int, b: int) -> int:
            return a * b

        server.add_tool(multiply)

        assert "multiply" in server.tools

    def test_register_multiple_tools(self):
        """Test registering multiple tools."""
        server = MCPServer()

        @server.tool()
        def add(a: int, b: int) -> int:
            return a + b

        @server.tool()
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        assert len(server.tools) == 2
        assert "add" in server.tools
        assert "greet" in server.tools
