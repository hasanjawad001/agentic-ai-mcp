# Agentic AI MCP

Lightweight agentic AI with MCP tools.

## Install

```bash
pip install agentic-ai-mcp
```

## Setup

Set your Anthropic API key in `.env` file:
```bash
ANTHROPIC_API_KEY=sk-...
```

## Usage

```python
from agentic_ai_mcp import AgenticAI

# 1. Define functions
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def greet(name: str, times: int = 1) -> str:
    """Greet someone."""
    return ("Hello, " + name + "! ") * times

# 2. Create AgenticAI and register tools
ai = AgenticAI()
ai.register_tool(add)
ai.register_tool(greet)

# 3. Run MCP server
ai.run_mcp_server()

# 4. Execute agentic workflow
result = await ai.run("Calculate 2+3 and greet Tom the result times")
print(result)
```

## License

MIT
