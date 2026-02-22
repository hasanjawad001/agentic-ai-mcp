# Agentic AI Framework

A multi-agent AI orchestration framework built with Python, featuring MCP (Model Context Protocol) integration, LangGraph workflows, and extensible agent architectures.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agentic AI Framework                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Orchestration Layer (LangGraph)             │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  │  Supervisor │──│   Router     │──│   Executor   │   │   │
│  │  └─────────────┘  └──────────────┘  └──────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Agent Layer                           │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  │ Math Agent  │  │  Text Agent  │  │ Custom Agent │   │   │
│  │  └─────────────┘  └──────────────┘  └──────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    MCP Layer                             │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  │ MCP Server  │  │  MCP Client  │  │ Tool Bridge  │   │   │
│  │  └─────────────┘  └──────────────┘  └──────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Tools Layer                           │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  │ Math Tools  │  │  Text Tools  │  │   Registry   │   │   │
│  │  └─────────────┘  └──────────────┘  └──────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **Multi-Agent Orchestration**: Coordinate multiple specialized agents using LangGraph state machines
- **MCP Protocol Support**: Standard Model Context Protocol for tool serving
- **Extensible Tool System**: Plugin-based architecture for adding custom tools
- **Type-Safe Design**: Full type hints with Pydantic validation
- **Logging**: Structured logging with configurable levels
- **Async-First**: Built on asyncio for high-performance concurrent execution
- **Comprehensive Testing**: Unit and integration tests with pytest

## Installation

### From PyPI (Recommended)

```bash
pip install agentic-ai-mcp
```

### From Source (For Development)

```bash
# Clone the repository
git clone https://github.com/hasanjawad001/agentic-ai-mcp.git
cd agentic-ai-mcp

# Create virtual environment
uv venv env --python 3.13
source env/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Quick Start

### 1. Start the MCP Server

```bash
python -m agentic_ai_mcp.mcp.server
```

### 2. Run a Multi-Agent Workflow

```python
from agentic_ai_mcp import AgenticWorkflow

async def main():
    workflow = AgenticWorkflow()
    result = await workflow.execute(
        "Calculate 15 + 27, then convert the result to uppercase"
    )
    print(result)

import asyncio
asyncio.run(main())
```

### 3. Create Custom Agents

```python
from agentic_ai_mcp.core import BaseAgent
from agentic_ai_mcp.tools import tool

class MyCustomAgent(BaseAgent):
    name = "custom_agent"
    description = "A custom agent for specific tasks"

    @tool
    def my_custom_tool(self, input_data: str) -> str:
        """Process input data in a custom way."""
        return f"Processed: {input_data}"
```

## Project Structure

```
code_agentic_ai_mcp/
├── src/
│   └── agentic_ai_mcp/
│       ├── core/           # Core abstractions and base classes
│       ├── tools/          # Tool definitions and registry
│       ├── mcp/            # MCP server, client, and bridge
│       ├── agents/         # Specialized agent implementations
│       ├── orchestration/  # LangGraph workflows and routing
│       └── config/         # Configuration management
├── tests/
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── examples/              # Usage examples
└── docs/                  # Documentation
```

## Key Concepts

### Agents

Agents are autonomous entities that can reason, plan, and execute actions. Each agent has:
- A set of tools it can use
- A specific domain of expertise
- The ability to communicate with other agents through the orchestrator

### Tools

Tools are atomic operations that agents can invoke. They are:
- Defined with type hints and descriptions
- Served via MCP protocol for language-agnostic access
- Validated at runtime using Pydantic schemas

### Orchestration

The orchestration layer coordinates multiple agents:
- **Supervisor**: Decides which agent should handle a task
- **Router**: Routes requests based on task requirements
- **Executor**: Manages the execution flow and state

## Configuration

```python
# config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str
    mcp_server_host: str = "0.0.0.0"
    mcp_server_port: int = 8888
    log_level: str = "INFO"
    max_agent_iterations: int = 10

    class Config:
        env_file = ".env"
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agentic_ai_mcp --cov-report=html

# Run specific test file
pytest tests/unit/test_tools.py -v
```

## Examples

See the `examples/` directory for comprehensive usage examples:

- `basic_workflow.py`: Simple single-agent workflow
- `multi_agent_workflow.py`: Multi-agent coordination
- `custom_tools.py`: Creating and registering custom tools

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
