#!/usr/bin/env python3
"""
Workflow Example - How to run agentic workflows.

First start the server:
    python examples/server_example.py

Then run this file:
    python examples/workflow_example.py
"""

import asyncio
import os

from agentic_ai_mcp import AgenticWorkflow


async def main():
    # Make sure API key is set
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY=your-key-here")
        return

    # Create workflow (connects to MCP server)
    workflow = AgenticWorkflow()

    # Example prompts
    prompts = [
        "Calculate 5 + 3",
        "What is 10 multiplied by 7?",
        "Greet Alice and then say farewell to Bob",
        "Add 15 and 25, then greet someone named 'Calculator'",
    ]

    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print("=" * 60)

        result = await workflow.run(prompt)
        print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
