"""
Agentic AI Framework - A production-grade multi-agent orchestration system.

This framework provides:
- Multi-agent orchestration using LangGraph
- MCP (Model Context Protocol) integration for tool serving
- Extensible agent and tool architectures
- Type-safe, async-first design
"""

from agentic_ai.core.base_agent import BaseAgent
from agentic_ai.core.state import AgentState, WorkflowState
from agentic_ai.orchestration.workflow import AgenticWorkflow
from agentic_ai.tools.registry import ToolRegistry

__version__ = "0.1.0"

__all__ = [
    "BaseAgent",
    "AgentState",
    "WorkflowState",
    "AgenticWorkflow",
    "ToolRegistry",
    "__version__",
]
