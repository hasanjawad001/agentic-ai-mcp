# Agentic AI MCP

Lightweight agentic AI with MCP tools. Supports multiple LLM providers (Anthropic, OpenAI) and distributed setups where tools run on one machine and agents on another.

## Install

```bash
pip install agentic-ai-mcp
```

Both Anthropic and OpenAI providers are included. Choose which to use at runtime.

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

# Create and register tools
ai = AgenticAI(host="0.0.0.0", port=8888)
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

# Connect to remote MCP server (default: Anthropic)
ai = AgenticAI(mcp_url="http://<server-ip>:8888/mcp")

# Simple agent workflow
result = await ai.run("Calculate 2+3 and greet Tom the result times")
print(result)

# Planning-based workflow for complex tasks
result = await ai.run_with_planning("First calculate ((1+2)+(1+1)+3), then greet Alice that many times")
print(result)
```

### Using OpenAI

```python
from agentic_ai_mcp import AgenticAI

# Use OpenAI instead of Anthropic
ai = AgenticAI(
    mcp_url="http://<server-ip>:8888/mcp",
    provider="openai",
    model="gpt-4o-mini"
)

result = await ai.run("Calculate 2+3")
```

### Custom Settings

```python
from agentic_ai_mcp import AgenticAI, Settings

# Configure retry behavior, timeouts, etc.
settings = Settings(
    max_retries=10,
    retry_base_delay=2.0,
)

ai = AgenticAI(
    mcp_url="http://<server-ip>:8888/mcp",
    settings=settings
)
```

## Providers

| Provider | Model Examples |
|----------|---------------|
| Anthropic (default) | `claude-sonnet-4-20250514`, `claude-haiku-4-5-20251001` |
| OpenAI | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo` |

## Methods

| Method | Description |
|--------|-------------|
| `ai.register_tool(func)` | Register a function as an MCP tool |
| `ai.run_mcp_server()` | Start MCP server in background |
| `ai.stop_mcp_server()` | Stop the MCP server |
| `ai.run(prompt)` | Simple agent workflow |
| `ai.run_with_planning(prompt)` | Complex agent workflow |

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `provider` | LLM provider (`"anthropic"` or `"openai"`) | `"anthropic"` |
| `model` | Model name | `"claude-haiku-4-5-20251001"` |
| `mcp_url` | URL of existing MCP server (client-only mode) | `None` |
| `host` | Host for MCP server | `"127.0.0.1"` |
| `port` | Port for MCP server | `8888` |
| `settings` | Custom Settings instance | `None` |
| `verbose` | Enable verbose output | `False` |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `DEFAULT_MODEL` | Default model name |
| `DEFAULT_PROVIDER` | Default provider (`anthropic` or `openai`) |

## License

MIT
