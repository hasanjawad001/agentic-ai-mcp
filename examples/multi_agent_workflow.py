#!/usr/bin/env python3
"""
Multi-Agent Workflow Example

Demonstrates the full multi-agent orchestration workflow using LangGraph.
Tasks that require multiple agents are automatically routed through the supervisor.
"""

import asyncio
import logging

from agentic_ai_mcp.orchestration.workflow import AgenticWorkflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def run_single_agent_task():
    """Example: Task requiring only one agent."""
    print("\n" + "=" * 60)
    print("Single Agent Task (Math Only)")
    print("=" * 60)

    workflow = AgenticWorkflow()

    query = "Calculate the square root of 144"
    print(f"\nQuery: {query}")

    result = await workflow.execute(query)
    response = workflow.get_final_response(result)

    print(f"\nResponse: {response}")
    print(f"Execution Path: {result.get('execution_path', [])}")


async def run_multi_agent_task():
    """Example: Task requiring multiple agents."""
    print("\n" + "=" * 60)
    print("Multi-Agent Task (Math + Text)")
    print("=" * 60)

    workflow = AgenticWorkflow()

    query = "Calculate 7 * 8, then convert the result to uppercase text"
    print(f"\nQuery: {query}")

    result = await workflow.execute(query)
    response = workflow.get_final_response(result)

    print(f"\nResponse: {response}")
    print(f"Execution Path: {result.get('execution_path', [])}")


async def run_complex_task():
    """Example: Complex task with multiple operations."""
    print("\n" + "=" * 60)
    print("Complex Multi-Agent Task")
    print("=" * 60)

    workflow = AgenticWorkflow()

    query = """
    I need you to:
    1. Add 100 and 200
    2. Take that result and multiply by 2
    3. Convert the final number to a string and reverse it
    """
    print(f"\nQuery: {query.strip()}")

    result = await workflow.execute(query)
    response = workflow.get_final_response(result)

    print(f"\nResponse: {response}")
    print(f"Execution Path: {result.get('execution_path', [])}")
    print(f"Total Iterations: {result.get('iteration_count', 0)}")


async def run_streaming_example():
    """Example: Streaming workflow execution."""
    print("\n" + "=" * 60)
    print("Streaming Execution Example")
    print("=" * 60)

    workflow = AgenticWorkflow()

    query = "What is 25 + 75?"
    print(f"\nQuery: {query}")
    print("\nStreaming execution steps:")

    async for step in workflow.execute_stream(query):
        # Print each step as it happens
        for key, value in step.items():
            if key == "messages":
                print(f"  [{key}]: {len(value)} messages")
            else:
                print(f"  [{key}]: {value}")
        print()


async def main():
    """Run all multi-agent workflow examples."""
    print("\n" + "#" * 60)
    print("# Agentic AI Framework - Multi-Agent Workflow Examples")
    print("#" * 60)

    await run_single_agent_task()
    await run_multi_agent_task()
    await run_complex_task()
    # Note: Streaming example commented out as it requires different handling
    # await run_streaming_example()

    print("\n" + "=" * 60)
    print("All workflow examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
