#!/usr/bin/env python3
"""
Basic Workflow Example

Demonstrates how to use the Agentic AI Framework for simple single-agent tasks.
"""

import asyncio
import logging

from agentic_ai.agents.math_agent import MathAgent
from agentic_ai.agents.text_agent import TextAgent
from langchain_core.messages import HumanMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def run_math_agent_example():
    """Example: Using the Math Agent directly."""
    print("\n" + "=" * 60)
    print("Math Agent Example")
    print("=" * 60)

    agent = MathAgent()

    # Simple calculation
    messages = [HumanMessage(content="What is 15 + 27?")]
    response = await agent.process(messages)

    print(f"\nQuery: {messages[0].content}")
    print(f"Response: {response.content}")
    print(f"Metadata: {response.metadata}")


async def run_text_agent_example():
    """Example: Using the Text Agent directly."""
    print("\n" + "=" * 60)
    print("Text Agent Example")
    print("=" * 60)

    agent = TextAgent()

    # Text transformation
    messages = [HumanMessage(content="Convert 'hello world' to uppercase")]
    response = await agent.process(messages)

    print(f"\nQuery: {messages[0].content}")
    print(f"Response: {response.content}")
    print(f"Metadata: {response.metadata}")


async def main():
    """Run all basic examples."""
    print("\n" + "#" * 60)
    print("# Agentic AI Framework - Basic Examples")
    print("#" * 60)

    await run_math_agent_example()
    await run_text_agent_example()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
