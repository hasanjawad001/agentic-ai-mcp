#!/usr/bin/env python3
"""
Custom Tools Example

Demonstrates how to create and register custom tools with the framework.
"""

import asyncio
import logging
from datetime import datetime

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agentic_ai_mcp.tools.registry import ToolRegistry, get_registry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Example 1: Simple function-based tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Example 2: Tool with input schema
class WeatherInput(BaseModel):
    """Input schema for weather tool."""

    city: str = Field(..., description="Name of the city")
    unit: str = Field(default="celsius", description="Temperature unit (celsius/fahrenheit)")


def get_weather(city: str, unit: str = "celsius") -> str:
    """
    Get the weather for a city.

    This is a mock implementation for demonstration.
    In production, this would call a real weather API.
    """
    # Mock weather data
    mock_temps = {
        "new york": 22,
        "london": 15,
        "tokyo": 28,
        "sydney": 20,
    }

    temp_c = mock_temps.get(city.lower(), 20)

    if unit.lower() == "fahrenheit":
        temp = temp_c * 9 / 5 + 32
        return f"Weather in {city}: {temp:.1f}°F"
    else:
        return f"Weather in {city}: {temp_c}°C"


# Example 3: Class-based tool collection
class StringUtilities:
    """Collection of string utility tools."""

    @staticmethod
    def truncate(text: str, max_length: int = 50) -> str:
        """Truncate text to a maximum length with ellipsis."""
        if len(text) <= max_length:
            return text
        return text[: max_length - 3] + "..."

    @staticmethod
    def repeat(text: str, times: int = 2) -> str:
        """Repeat text a specified number of times."""
        return text * times

    @staticmethod
    def word_frequency(text: str) -> dict:
        """Count the frequency of each word in text."""
        words = text.lower().split()
        freq = {}
        for word in words:
            freq[word] = freq.get(word, 0) + 1
        return freq


def register_custom_tools():
    """Register all custom tools with the registry."""
    registry = get_registry()

    # Register simple function tool
    registry.register_function(
        get_current_time,
        name="get_current_time",
        description="Get the current date and time",
        category="utility",
        tags=["time", "date", "utility"],
    )

    # Register tool with input schema
    weather_tool = StructuredTool(
        name="get_weather",
        description="Get the weather for a city. Returns temperature in specified unit.",
        func=get_weather,
        args_schema=WeatherInput,
    )
    registry.register(weather_tool, category="utility", tags=["weather", "api"])

    # Register string utility tools
    registry.register_function(
        StringUtilities.truncate,
        name="truncate_text",
        description="Truncate text to a maximum length with ellipsis",
        category="text",
        tags=["text", "string", "truncate"],
    )

    registry.register_function(
        StringUtilities.repeat,
        name="repeat_text",
        description="Repeat text a specified number of times",
        category="text",
        tags=["text", "string", "repeat"],
    )

    registry.register_function(
        StringUtilities.word_frequency,
        name="word_frequency",
        description="Count the frequency of each word in text",
        category="text",
        tags=["text", "analysis", "frequency"],
    )

    return registry


def demonstrate_registry():
    """Demonstrate registry functionality."""
    print("\n" + "=" * 60)
    print("Tool Registry Demonstration")
    print("=" * 60)

    # Load default tools and register custom ones
    registry = get_registry()
    registry.load_default_tools()
    register_custom_tools()

    print(f"\nTotal tools registered: {len(registry)}")

    # List all categories
    print(f"\nCategories: {registry.list_categories()}")

    # List tools by category
    print("\nTools by category:")
    for category in registry.list_categories():
        tools = registry.get_by_category(category)
        print(f"  {category}: {[t.name for t in tools]}")

    # Get tools by tag
    print("\nText-related tools:")
    text_tools = registry.get_by_tags(["text"])
    for tool in text_tools:
        print(f"  - {tool.name}: {tool.description}")

    # Test a custom tool
    print("\n" + "=" * 60)
    print("Testing Custom Tools")
    print("=" * 60)

    time_tool = registry.get("get_current_time")
    if time_tool:
        result = time_tool.invoke({})
        print(f"\nCurrent time: {result}")

    weather_tool = registry.get("get_weather")
    if weather_tool:
        result = weather_tool.invoke({"city": "Tokyo", "unit": "celsius"})
        print(f"\n{result}")

    truncate_tool = registry.get("truncate_text")
    if truncate_tool:
        long_text = "This is a very long text that needs to be truncated for display purposes."
        result = truncate_tool.invoke({"text": long_text, "max_length": 30})
        print(f"\nTruncated: {result}")


def main():
    """Run custom tools example."""
    print("\n" + "#" * 60)
    print("# Agentic AI Framework - Custom Tools Example")
    print("#" * 60)

    demonstrate_registry()

    print("\n" + "=" * 60)
    print("Custom tools example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
