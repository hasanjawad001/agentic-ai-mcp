"""Mathematical operation tools for the Math Agent."""

from __future__ import annotations

import logging
import math
from typing import Annotated

from langchain_core.tools import StructuredTool, tool as langchain_tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AddInput(BaseModel): ## BaseModel enforces type hints as pydantic validates it
    """Input schema for add operation."""

    a: Annotated[int | float, Field(description="First number to add")] ## Annotated combines type and Field metadata
    b: Annotated[int | float, Field(description="Second number to add")]


class MultiplyInput(BaseModel):
    """Input schema for multiply operation."""

    a: Annotated[int | float, Field(description="First number to multiply")]
    b: Annotated[int | float, Field(description="Second number to multiply")]


class SubtractInput(BaseModel):
    """Input schema for subtract operation."""

    a: Annotated[int | float, Field(description="Number to subtract from")]
    b: Annotated[int | float, Field(description="Number to subtract")]


class DivideInput(BaseModel):
    """Input schema for divide operation."""

    a: Annotated[int | float, Field(description="Dividend (number to divide)")]
    b: Annotated[int | float, Field(description="Divisor (number to divide by)")]


class PowerInput(BaseModel):
    """Input schema for power operation."""

    base: Annotated[int | float, Field(description="Base number")]
    exponent: Annotated[int | float, Field(description="Exponent to raise base to")]


class SqrtInput(BaseModel):
    """Input schema for square root operation."""

    number: Annotated[int | float, Field(description="Number to find square root of", ge=0)]


class MathTools:
    """
    Collection of mathematical operation tools.

    These tools perform basic and advanced mathematical operations
    and are designed to be used by the Math Agent.
    """

    @staticmethod
    def add(a: int | float, b: int | float) -> int | float:
        """
        Add two numbers together.

        Args:
            a: First number to add
            b: Second number to add

        Returns:
            Sum of a and b
        """
        logger.debug(f"Adding {a} + {b}")
        result = a + b
        logger.debug(f"Result: {result}")
        return result

    @staticmethod
    def subtract(a: int | float, b: int | float) -> int | float:
        """
        Subtract the second number from the first.

        Args:
            a: Number to subtract from
            b: Number to subtract

        Returns:
            Difference of a - b
        """
        logger.debug(f"Subtracting {a} - {b}")
        result = a - b
        logger.debug(f"Result: {result}")
        return result

    @staticmethod
    def multiply(a: int | float, b: int | float) -> int | float:
        """
        Multiply two numbers together.

        Args:
            a: First number to multiply
            b: Second number to multiply

        Returns:
            Product of a * b
        """
        logger.debug(f"Multiplying {a} * {b}")
        result = a * b
        logger.debug(f"Result: {result}")
        return result

    @staticmethod
    def divide(a: int | float, b: int | float) -> float:
        """
        Divide the first number by the second.

        Args:
            a: Dividend (number to divide)
            b: Divisor (number to divide by)

        Returns:
            Quotient of a / b

        Raises:
            ValueError: If b is zero
        """
        logger.debug(f"Dividing {a} / {b}")
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        logger.debug(f"Result: {result}")
        return result

    @staticmethod
    def power(base: int | float, exponent: int | float) -> float:
        """
        Raise base to the power of exponent.

        Args:
            base: Base number
            exponent: Exponent to raise base to

        Returns:
            base raised to the power of exponent
        """
        logger.debug(f"Computing {base} ^ {exponent}")
        result = math.pow(base, exponent)
        logger.debug(f"Result: {result}")
        return result

    @staticmethod
    def sqrt(number: int | float) -> float:
        """
        Calculate the square root of a number.

        Args:
            number: Non-negative number to find square root of

        Returns:
            Square root of the number

        Raises:
            ValueError: If number is negative
        """
        logger.debug(f"Computing sqrt({number})")
        if number < 0:
            raise ValueError("Cannot compute square root of negative number")
        result = math.sqrt(number)
        logger.debug(f"Result: {result}")
        return result


def get_math_tools() -> list[StructuredTool]:
    """
    Get all math tools as LangChain StructuredTools.

    Returns:
        List of StructuredTool instances for math operations
    """
    return [
        StructuredTool(
            name="add",
            description="Add two numbers together. Use this for addition operations.",
            func=MathTools.add,
            args_schema=AddInput,
        ),
        StructuredTool(
            name="subtract",
            description="Subtract the second number from the first. Use this for subtraction.",
            func=MathTools.subtract,
            args_schema=SubtractInput,
        ),
        StructuredTool(
            name="multiply",
            description="Multiply two numbers together. Use this for multiplication.",
            func=MathTools.multiply,
            args_schema=MultiplyInput,
        ),
        StructuredTool(
            name="divide",
            description="Divide the first number by the second. Use this for division.",
            func=MathTools.divide,
            args_schema=DivideInput,
        ),
        StructuredTool(
            name="power",
            description="Raise a base number to an exponent power. Use for exponentiation.",
            func=MathTools.power,
            args_schema=PowerInput,
        ),
        StructuredTool(
            name="sqrt",
            description="Calculate the square root of a non-negative number.",
            func=MathTools.sqrt,
            args_schema=SqrtInput,
        ),
    ]


# Module-level tool definitions using LangChain decorator
@langchain_tool(args_schema=AddInput)
def add_tool(a: int | float, b: int | float) -> int | float:
    """Add two numbers together."""
    return MathTools.add(a, b)


@langchain_tool(args_schema=MultiplyInput)
def multiply_tool(a: int | float, b: int | float) -> int | float:
    """Multiply two numbers together."""
    return MathTools.multiply(a, b)


@langchain_tool(args_schema=SubtractInput)
def subtract_tool(a: int | float, b: int | float) -> int | float:
    """Subtract the second number from the first."""
    return MathTools.subtract(a, b)
