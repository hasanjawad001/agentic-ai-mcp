# Architecture

This document describes the architecture of the Agentic AI Framework.

## Overview

The framework follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface                         │
│               (CLI, API, Notebooks, etc.)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Orchestration Layer                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │               LangGraph Workflow                     │   │
│  │   ┌───────────┐  ┌──────────┐  ┌──────────────┐    │   │
│  │   │Supervisor │──│  Router  │──│  Executor    │    │   │
│  │   └───────────┘  └──────────┘  └──────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Agent Layer                            │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │  Math Agent   │  │  Text Agent   │  │ Custom Agent  │   │
│  │  (ReAct Loop) │  │  (ReAct Loop) │  │  (ReAct Loop) │   │
│  └───────────────┘  └───────────────┘  └───────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Tool Execution Layer                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              MCP Server (HTTP Transport)             │   │
│  │   ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │   │
│  │   │Math Tools│  │Text Tools│  │  Custom Tools    │  │   │
│  │   └──────────┘  └──────────┘  └──────────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Orchestration Layer

The orchestration layer manages the flow of execution through the multi-agent system.

**Components:**
- **Supervisor**: Makes routing decisions based on task requirements
- **Router**: Implements conditional routing logic
- **Executor**: Manages workflow execution and state

**Technology:** LangGraph StateGraph

### 2. Agent Layer

Agents are autonomous entities that can reason, plan, and execute actions.

**Agent Types:**
- **Supervisor Agent**: Coordinates specialist agents
- **Specialist Agents**: Domain-specific agents (Math, Text, etc.)

**Pattern:** ReAct (Reason-Act-Observe) loop

### 3. Tool Layer

Tools are atomic operations that agents can invoke.

**Features:**
- Type-safe input/output schemas (Pydantic)
- Served via MCP protocol
- Organized in registry by category/tags

## Data Flow

```
User Query
     │
     ▼
┌─────────────┐
│  Supervisor │──── Analyzes task requirements
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Router    │──── Decides which agent to call
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Agent     │──── Executes tools via ReAct loop
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    MCP      │──── Tool execution via MCP protocol
└──────┬──────┘
       │
       ▼
Tool Result → Agent → Supervisor → Next Agent or Finish
```

## State Management

The workflow state is managed using LangGraph's TypedDict pattern:

```python
class WorkflowState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str
    execution_path: Annotated[list[str], operator.add]
    iteration_count: int
```

**Reducers:**
- `messages`: Accumulated using `operator.add`
- `execution_path`: Accumulated using `operator.add`
- `next_agent`: Overwritten on each update

## Design Patterns

### 1. Singleton Registry
Tool registry uses singleton pattern for global access.

### 2. Factory Functions
Agents and workflows are created via factory functions.

### 3. Async-First
All I/O operations are async for optimal performance.

### 4. Structured Output
Routing decisions use Pydantic models for type safety.

## Extension Points

### Adding Custom Agents

```python
from agentic_ai.core import BaseAgent

class MyAgent(BaseAgent):
    name = "my_agent"
    description = "Custom agent"

    async def process(self, messages, **kwargs):
        # Implementation
        pass
```

### Adding Custom Tools

```python
from agentic_ai.tools import tool

@tool(name="my_tool", description="Does something")
def my_tool(input: str) -> str:
    return f"Processed: {input}"
```

### Adding Custom Workflows

```python
from langgraph.graph import StateGraph

def build_custom_workflow():
    graph = StateGraph(WorkflowState)
    # Add nodes and edges
    return graph.compile()
```

## Security Considerations

1. **API Key Management**: Keys stored in environment variables
2. **Input Validation**: Pydantic schemas validate all inputs
3. **Iteration Limits**: Prevent infinite loops in agent execution
4. **Error Handling**: Graceful error propagation and logging
