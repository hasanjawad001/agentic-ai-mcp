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

### Server (expose tools)

Run this on the machine where you want to host tools:

```python
from agentic_ai_mcp import AgenticAIServer

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def greet(name: str, times: int = 1) -> str:
    """Greet someone."""
    return ("Hello, " + name + "! ") * times

# Create server and register tools
server = AgenticAIServer(host="0.0.0.0", port=8888)
server.register_tool(add)
server.register_tool(greet)

print(f"Tools: {server.tools}")
print(f"URL: {server.mcp_url}")

# Start server 
server.run()
```

### Client (run agents)

Run this on another machine to connect to the server and execute agents:

```python
from agentic_ai_mcp import AgenticAIClient

# Connect to MCP server
client = AgenticAIClient(mcp_url="http://<server-ip>:8888/mcp")

# Simple agent workflow
result = await client.run("Calculate 2+1, then greet 'Alice' that many times.")
print(result)

# Planning-based workflow for complex tasks
result = await client.run_with_planning("Calculate ((0+2) + (1+1) + 1), then greet 'Bob' that many times.")
print(result)
```

### Using OpenAI

```python
from agentic_ai_mcp import AgenticAIClient

# Use OpenAI instead of Anthropic
client = AgenticAIClient(
    mcp_url="http://<server-ip>:8888/mcp",
    provider="openai",
    model="gpt-4o-mini"
)

result = await client.run("Calculate -1+2")
```

### Passing API Key Directly

```python
from agentic_ai_mcp import AgenticAIClient

# Pass API key directly (instead of using .env)
client = AgenticAIClient(
    mcp_url="http://<server-ip>:8888/mcp",
    api_key="sk-ant-..."
)
```

## API Reference

### AgenticAIServer

| Property/Method | Description |
|-----------------|-------------|
| `server.tools` | List of registered tool names |
| `server.mcp_url` | Server URL |
| `server.register_tool(func)` | Register a function as an MCP tool |
| `server.run()` | Start MCP server |

### AgenticAIClient

| Property/Method | Description |
|-----------------|-------------|
| `client.tools` | List of loaded tool names |
| `client.run(prompt)` | Simple agentic workflow |
| `client.run_with_planning(prompt)` | Planning-based workflow for complex tasks |

## License

MIT
