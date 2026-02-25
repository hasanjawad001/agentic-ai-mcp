"""Simple example - the easiest way to use agentic-ai-mcp.

Usage:
1. Define functions with type hints and docstrings
2. Register them with ai.register_tool()
3. Run prompts with ai.run()
"""

import asyncio

from agentic_ai_mcp import AgenticAI

# Create AI instance
ai = AgenticAI()


# Define your functions
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


def greet(name: str, times: int = 1) -> str:
    """Greet someone a specified number of times."""
    return ("Hello, " + name + "! ") * times


# Register them as tools
ai.register_tool(add)
ai.register_tool(multiply)
ai.register_tool(greet)


async def main():
    # Run prompts - the agent uses tools as needed
    result = await ai.run("Calculate 2+3 and greet Tom the result times")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
