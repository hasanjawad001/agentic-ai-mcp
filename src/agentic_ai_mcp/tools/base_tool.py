"""Base tool class and decorators for tool definition."""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable, ParamSpec, TypeVar

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class BaseTool(BaseModel):
    """
    Base class for tool definitions.

    Provides a structured way to define tools with:
    - Type-safe input/output schemas
    - Automatic validation
    - Consistent error handling
    """

    name: str = Field(..., description="Unique name for the tool")
    description: str = Field(..., description="Description of what the tool does")
    func: Callable[..., Any] = Field(..., description="The function to execute")
    args_schema: type[BaseModel] | None = Field(
        default=None,
        description="Pydantic model for input validation",
    )

    class Config:
        arbitrary_types_allowed = True

    def invoke(self, **kwargs: Any) -> Any:
        """Invoke the tool with given arguments."""
        if self.args_schema:
            validated = self.args_schema(**kwargs)
            kwargs = validated.model_dump()
        return self.func(**kwargs)

    async def ainvoke(self, **kwargs: Any) -> Any:
        """Async invoke the tool with given arguments."""
        if self.args_schema:
            validated = self.args_schema(**kwargs)
            kwargs = validated.model_dump()

        if inspect.iscoroutinefunction(self.func):
            return await self.func(**kwargs)
        return self.func(**kwargs)

    def to_langchain_tool(self) -> StructuredTool:
        """Convert to LangChain StructuredTool."""
        return StructuredTool(
            name=self.name,
            description=self.description,
            func=self.func,
            args_schema=self.args_schema,
        )


def tool(
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to convert a function into a tool.

    Usage:
        @tool(name="add", description="Add two numbers")
        def add(a: int, b: int) -> int:
            return a + b

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)

    Returns:
        Decorated function with tool metadata
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Tool: {tool_name}"

        # Extract type hints for args schema
        sig = inspect.signature(func)
        hints = func.__annotations__

        fields: dict[str, Any] = {}
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            param_type = hints.get(param_name, Any)
            if param.default is inspect.Parameter.empty:
                fields[param_name] = (param_type, ...)
            else:
                fields[param_name] = (param_type, param.default)

        # Create args schema dynamically
        args_schema = create_model(
            f"{tool_name.title().replace('_', '')}Args",
            **fields,
        )

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return func(*args, **kwargs)

        # Attach metadata
        wrapper._tool_name = tool_name  # type: ignore
        wrapper._tool_description = tool_description  # type: ignore
        wrapper._tool_args_schema = args_schema  # type: ignore
        wrapper._is_tool = True  # type: ignore

        return wrapper

    return decorator


def create_langchain_tool(
    name: str,
    description: str,
    func: Callable[..., Any],
    args_schema: type[BaseModel] | None = None,
) -> StructuredTool:
    """
    Create a LangChain StructuredTool from a function.

    Args:
        name: Tool name
        description: Tool description
        func: Function to wrap
        args_schema: Optional Pydantic model for input validation

    Returns:
        LangChain StructuredTool
    """
    if args_schema is None:
        # Auto-generate schema from function signature
        sig = inspect.signature(func)
        hints = func.__annotations__

        fields: dict[str, Any] = {}
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            param_type = hints.get(param_name, Any)
            if param.default is inspect.Parameter.empty:
                fields[param_name] = (param_type, Field(...))
            else:
                fields[param_name] = (param_type, Field(default=param.default))

        args_schema = create_model(f"{name.title().replace('_', '')}Schema", **fields)

    return StructuredTool(
        name=name,
        description=description,
        func=func,
        args_schema=args_schema,
    )
