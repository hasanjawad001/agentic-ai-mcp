# Getting Started

This guide will help you get up and running with the Agentic AI Framework.

## Prerequisites

- Python 3.11 or higher
- An Anthropic API key

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/hasanjawad001/code_agentic_ai.git
cd code_agentic_ai
```

### 2. Create Virtual Environment

```bash
# Using uv (recommended)
uv venv env --python 3.13
source env/bin/activate

# Or using standard venv
python -m venv env
source env/bin/activate  # Linux/Mac
# or
.\env\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
# Install in development mode
pip install -e ".[dev]"

# Or install just the main dependencies
pip install -e .
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Quick Start

### Running a Simple Workflow

```python
import asyncio
from agentic_ai import AgenticWorkflow

async def main():
    workflow = AgenticWorkflow()
    result = await workflow.execute("What is 15 + 27?")
    print(workflow.get_final_response(result))

asyncio.run(main())
```

### Using Individual Agents

```python
import asyncio
from agentic_ai.agents import MathAgent
from langchain_core.messages import HumanMessage

async def main():
    agent = MathAgent()
    messages = [HumanMessage(content="Calculate 100 divided by 4")]
    response = await agent.process(messages)
    print(response.content)

asyncio.run(main())
```

### Starting the MCP Server

```bash
# Start the server
python -m agentic_ai.mcp.server

# Or using the entry point
agentic-server
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agentic_ai

# Run only unit tests
pytest tests/unit -v
```

## Running Examples

```bash
# Basic workflow example
python examples/basic_workflow.py

# Multi-agent workflow
python examples/multi_agent_workflow.py

# Custom tools example
python examples/custom_tools.py

# MCP server example
python examples/mcp_server_example.py --server  # Terminal 1
python examples/mcp_server_example.py --client  # Terminal 2
```

## Project Structure

```
code_agentic_ai/
├── src/agentic_ai/       # Main package
│   ├── core/             # Base classes and types
│   ├── tools/            # Tool implementations
│   ├── mcp/              # MCP server and client
│   ├── agents/           # Agent implementations
│   ├── orchestration/    # Workflow orchestration
│   └── config/           # Configuration
├── tests/                # Test suite
├── examples/             # Usage examples
└── docs/                 # Documentation
```

## Common Tasks

### Creating a Custom Tool

```python
from agentic_ai.tools.registry import get_registry

def my_tool(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

registry = get_registry()
registry.register_function(my_tool, category="math")
```

### Creating a Custom Agent

```python
from agentic_ai.core import BaseAgent
from agentic_ai.core.types import AgentResponse

class MyAgent(BaseAgent):
    name = "my_agent"
    description = "My custom agent"

    async def process(self, messages, **kwargs):
        # Your implementation
        return AgentResponse(
            agent_name=self.name,
            content="Response",
            is_final=True,
        )
```

## Troubleshooting

### API Key Issues

If you get authentication errors:
1. Verify your API key is set: `echo $ANTHROPIC_API_KEY`
2. Check the key starts with `sk-`
3. Ensure `.env` file is in the project root

### Import Errors

If you get import errors:
1. Make sure you've activated the virtual environment
2. Reinstall with `pip install -e ".[dev]"`

### MCP Connection Issues

If you can't connect to the MCP server:
1. Make sure the server is running: `python -m agentic_ai.mcp.server`
2. Check the port is available: `lsof -i :8888`
3. Verify the URL matches: `http://localhost:8888/mcp`

## Next Steps

- Read the [Architecture](architecture.md) documentation
- Explore the [examples](../examples/) directory
- Check out the test files for usage patterns
