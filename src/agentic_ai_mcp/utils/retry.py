"""Retry logic with exponential backoff."""

import asyncio
import random
from collections.abc import Callable
from typing import Any

import anthropic


async def retry_with_backoff(
    coro_func: Callable[..., Any],
    *args: Any,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    verbose: bool = False,
    **kwargs: Any,
) -> Any:
    """Retry an async function with exponential backoff on API overload errors.

    Args:
        coro_func: Async function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        verbose: Print retry information
        *args, **kwargs: Arguments to pass to coro_func

    Returns:
        Result from the successful call

    Raises:
        The last exception if all retries fail
    """
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await coro_func(*args, **kwargs)
        except anthropic.InternalServerError as e:
            last_exception = e
            if "overloaded" in str(e).lower() or "529" in str(e):
                if attempt < max_retries:
                    # Exponential backoff with jitter
                    delay = min(base_delay * (2**attempt) + random.uniform(0, 1), max_delay)
                    if verbose:
                        print(
                            f"  [RETRY] API overloaded. Waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}..."
                        )
                    await asyncio.sleep(delay)
                else:
                    raise
            else:
                raise
        except Exception as e:
            # Check if it's an overload error wrapped in another exception
            if "overloaded" in str(e).lower() or "529" in str(e):
                last_exception = e
                if attempt < max_retries:
                    delay = min(base_delay * (2**attempt) + random.uniform(0, 1), max_delay)
                    if verbose:
                        print(
                            f"  [RETRY] API overloaded. Waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}..."
                        )
                    await asyncio.sleep(delay)
                else:
                    raise
            else:
                raise

    raise last_exception  # type: ignore[misc]
