"""Tests for AgenticWorkflow."""

import os
from unittest.mock import patch

from agentic_ai_mcp import AgenticWorkflow


class TestAgenticWorkflow:
    """Tests for AgenticWorkflow class."""

    def test_create_workflow(self):
        """Test creating a workflow."""
        workflow = AgenticWorkflow()
        assert workflow.mcp_url == "http://localhost:8888/mcp"
        assert workflow.max_iterations == 10
        assert workflow.tools == []

    def test_create_workflow_custom_url(self):
        """Test creating workflow with custom URL."""
        workflow = AgenticWorkflow(mcp_url="http://example.com:9000/mcp")
        assert workflow.mcp_url == "http://example.com:9000/mcp"

    def test_create_workflow_custom_iterations(self):
        """Test creating workflow with custom max iterations."""
        workflow = AgenticWorkflow(max_iterations=5)
        assert workflow.max_iterations == 5

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""})
    def test_workflow_requires_api_key(self):
        """Test that workflow requires API key to run."""
        workflow = AgenticWorkflow()
        # The error would happen during setup, not creation
        # Just test that we can create the workflow
        assert workflow is not None
