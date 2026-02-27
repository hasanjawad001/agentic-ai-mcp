# Agentic AI MCP

Lightweight agentic AI with MCP tools. Supports multiple LLM providers (Anthropic, OpenAI) and distributed setups where tools run on one machine and agents on another.

## Install

```bash
pip install agentic-ai-mcp
```

## Setup

Set your API key in `.env` file (only needed on the client/agent machine):

```bash
# For Anthropic (default)
ANTHROPIC_API_KEY=sk-ant-...

# For OpenAI
OPENAI_API_KEY=sk-...
```

## Quick Start

See the example notebooks:
- [`examples/quickstart_server.ipynb`](examples/quickstart_server.ipynb) - Run on machine exposing tools
- [`examples/quickstart_client.ipynb`](examples/quickstart_client.ipynb) - Run on machine executing agents

## Usage

### Server Mode (expose tools)

Run this on the machine where you want to host tools:

```python
from agentic_ai_mcp import AgenticAI

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def greet(name: str, times: int = 1) -> str:
    """Greet someone."""
    return ("Hello, " + name + "! ") * times

# Create agentic instance and register tools
ai = AgenticAI()
ai.register_tool(add)
ai.register_tool(greet)

# Start server
ai.run_mcp_server()

# Stop when done
ai.stop_mcp_server()
```

### Client Mode (run agents)

Run this on another machine to connect to the server and execute agents:

```python
from agentic_ai_mcp import AgenticAI

# Connect to remote MCP server 
ai = AgenticAI(mcp_url="http://<server-ip>:8888/mcp") ## default provider: Anthropic, use provider = 'openai' to use OpenAI 

# Simple agent workflow
result = await ai.run("Calculate 2+1, then use the result as the number of times to greet 'Alice'.")
print(result)

# Planning-based workflow for complex tasks
result = await ai.run_with_planning("Calculate ((0+2) + (1+1) + 1), then use the result as the number of times to greet 'Bob'.")
print(result)
```

### Using OpenAI

```python
from agentic_ai_mcp import AgenticAI

# Use OpenAI instead of Anthropic
ai = AgenticAI(
    mcp_url="http://<server-ip>:8888/mcp",
    provider="openai",
    model="gpt-4o-mini" ## or 'gpt-4o'/'gpt-4-turbo' etc.
)

result = await ai.run("Calculate -1+2")
```

## Methods

| Method | Description |
|--------|-------------|
| `ai.register_tool(func)` | Register a function as an MCP tool |
| `ai.run_mcp_server()` | Start MCP server in background |
| `ai.stop_mcp_server()` | Stop the MCP server |
| `ai.run(prompt)` | Simple agent workflow |
| `ai.run_with_planning(prompt)` | Complex agent workflow |

## License

MIT
