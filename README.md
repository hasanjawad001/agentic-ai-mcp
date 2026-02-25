# Agentic AI MCP

A lightweight agentic AI framework with MCP (Model Context Protocol) tool serving.

## Quick Start

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

result = await ai.run("Calculate 2+3 and greet Tom the result times")
print(result)  # Hello, Tom! Hello, Tom! Hello, Tom! Hello, Tom! Hello, Tom!
```

## Installation

```bash
pip install agentic-ai-mcp
```

## Environment Variables

```bash
ANTHROPIC_API_KEY=sk-...  # Required
```

## API

### AgenticAI

```python
from agentic_ai_mcp import AgenticAI

ai = AgenticAI(
    host="127.0.0.1",     # MCP server host
    port=8888,            # MCP server port
    model="claude-sonnet-4-20250514",
    max_iterations=10,
)

def my_function(x: int) -> int:
    """Description."""
    return x * 2

ai.register_tool(my_function)

result = await ai.run("Double 5")
# or sync: result = ai.run_sync("Double 5")
```

## License

MIT
