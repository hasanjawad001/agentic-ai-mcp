"""Integration tests for workflow orchestration."""

from agentic_ai_mcp.orchestration.workflow import AgenticWorkflow, create_workflow


class TestAgenticWorkflow:
    """Tests for AgenticWorkflow class."""

    def test_create_workflow(self):
        """Test workflow creation."""
        workflow = create_workflow(max_iterations=5)

        assert isinstance(workflow, AgenticWorkflow)
        assert workflow.max_iterations == 5

    def test_workflow_default_iterations(self):
        """Test default max iterations."""
        workflow = AgenticWorkflow()
        assert workflow.max_iterations == 10

    def test_compile_workflow(self):
        """Test workflow compilation."""
        workflow = AgenticWorkflow()

        # Compilation should not raise
        compiled = workflow.compile()

        assert compiled is not None

    def test_compile_caches_result(self):
        """Test that compile caches the compiled graph."""
        workflow = AgenticWorkflow()

        compiled1 = workflow.compile()
        compiled2 = workflow.compile()

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

    def test_graph_has_required_nodes(self):
        """Test that the graph has all required nodes."""
        workflow = AgenticWorkflow()
        workflow._build_graph()

        graph = workflow._graph
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
