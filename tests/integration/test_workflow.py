"""Integration tests for workflow orchestration."""

import pytest

from agentic_ai_mcp.orchestration.workflow import AgenticWorkflow, create_workflow


class TestAgenticWorkflow:
    """Tests for AgenticWorkflow class."""

    def test_create_workflow(self):
        """Test workflow creation."""
        workflow = create_workflow(max_iterations=5)

        assert isinstance(workflow, AgenticWorkflow)
        assert workflow.max_iterations == 5

    def test_create_workflow_with_server_url(self):
        """Test workflow creation with custom server URL."""
        workflow = create_workflow(server_url="http://custom:9999/mcp")

        assert workflow._bridge.client.server_url == "http://custom:9999/mcp"

    def test_workflow_default_iterations(self):
        """Test default max iterations."""
        workflow = AgenticWorkflow()
        assert workflow.max_iterations == 10

    def test_workflow_has_mcp_bridge(self):
        """Test that workflow has MCP bridge."""
        from agentic_ai_mcp.mcp.bridge import MCPToolBridge

        workflow = AgenticWorkflow()
        assert isinstance(workflow._bridge, MCPToolBridge)

    def test_compile_workflow(self, mock_workflow_with_mcp):
        """Test workflow compilation with mocked MCP."""
        # Compilation should not raise
        compiled = mock_workflow_with_mcp.compile()

        assert compiled is not None

    def test_compile_caches_result(self, mock_workflow_with_mcp):
        """Test that compile caches the compiled graph."""
        compiled1 = mock_workflow_with_mcp.compile()
        compiled2 = mock_workflow_with_mcp.compile()

        assert compiled1 is compiled2

    def test_get_final_response_empty(self):
        """Test getting final response from empty result."""
        workflow = AgenticWorkflow()

        result = {"messages": []}
        response = workflow.get_final_response(result)

        assert response == "No response generated"

    def test_get_final_response_with_messages(self):
        """Test getting final response with messages."""
        from langchain_core.messages import AIMessage, HumanMessage

        workflow = AgenticWorkflow()

        result = {
            "messages": [
                HumanMessage(content="Calculate 2+2"),
                AIMessage(content="The result is 4"),
            ]
        }

        response = workflow.get_final_response(result)
        assert response == "The result is 4"


class TestWorkflowGraph:
    """Tests for workflow graph structure."""

    def test_graph_has_required_nodes(self, mock_workflow_with_mcp):
        """Test that the graph has all required nodes."""
        mock_workflow_with_mcp._build_graph()

        graph = mock_workflow_with_mcp._graph
        assert graph is not None

        # Check nodes exist (by checking the graph was built)
        nodes = list(graph.nodes.keys())
        assert "supervisor" in nodes
        assert "math_agent" in nodes
        assert "text_agent" in nodes

    def test_workflow_state_schema(self):
        """Test workflow state schema."""
        from agentic_ai_mcp.orchestration.workflow import WorkflowState

        # WorkflowState should have required keys
        assert "messages" in WorkflowState.__annotations__
        assert "next_agent" in WorkflowState.__annotations__
        assert "execution_path" in WorkflowState.__annotations__


class TestMCPIntegration:
    """Tests for MCP integration in workflow."""

    @pytest.mark.asyncio
    async def test_load_tools_from_mcp(self, mock_workflow_with_mcp):
        """Test that tools are loaded from MCP bridge."""
        async with mock_workflow_with_mcp._bridge.connect():
            await mock_workflow_with_mcp._load_tools_from_mcp()

            assert len(mock_workflow_with_mcp._math_tools) > 0
            assert len(mock_workflow_with_mcp._text_tools) > 0

    @pytest.mark.asyncio
    async def test_math_tools_loaded_correctly(self, mock_workflow_with_mcp):
        """Test that math tools include expected tools."""
        async with mock_workflow_with_mcp._bridge.connect():
            await mock_workflow_with_mcp._load_tools_from_mcp()

            tool_names = [t.name for t in mock_workflow_with_mcp._math_tools]
            assert "add" in tool_names
            assert "subtract" in tool_names
            assert "multiply" in tool_names

    @pytest.mark.asyncio
    async def test_text_tools_loaded_correctly(self, mock_workflow_with_mcp):
        """Test that text tools include expected tools."""
        async with mock_workflow_with_mcp._bridge.connect():
            await mock_workflow_with_mcp._load_tools_from_mcp()

            tool_names = [t.name for t in mock_workflow_with_mcp._text_tools]
            assert "to_uppercase" in tool_names
            assert "to_lowercase" in tool_names
