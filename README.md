# Agentic AI MCP

Lightweight agentic AI with MCP tools.

## Install

```bash
pip install agentic-ai-mcp
```

## Usage

```python
from agentic_ai_mcp import AgenticAI

ai = AgenticAI()

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def greet(name: str, times: int = 1) -> str:
    """Greet someone."""
    return ("Hello, " + name + "! ") * times

ai.register_tool(add)
ai.register_tool(greet)
ai.run_mcp_server()

result = await ai.run("Calculate 2+3 and greet Tom the result times")
```

## Environment

```bash
ANTHROPIC_API_KEY=sk-...
```

## License

MIT
