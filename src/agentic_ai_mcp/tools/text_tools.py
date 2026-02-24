"""Text manipulation tools for the Text Agent."""

from __future__ import annotations

import logging
import re
from typing import Annotated

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TextInput(BaseModel):
    """Input schema for single text operations."""

    text: Annotated[str, Field(description="Text string to process")]


class SearchReplaceInput(BaseModel):
    """Input schema for search and replace operations."""

    text: Annotated[str, Field(description="Text to search in")]
    search: Annotated[str, Field(description="Pattern to search for")]
    replace: Annotated[str, Field(description="Replacement string")]


class SplitInput(BaseModel):
    """Input schema for split operations."""

    text: Annotated[str, Field(description="Text to split")]
    delimiter: Annotated[str, Field(description="Delimiter to split on", default=" ")]


class JoinInput(BaseModel):
    """Input schema for join operations."""

    texts: Annotated[list[str], Field(description="List of texts to join")]
    separator: Annotated[str, Field(description="Separator between texts", default=" ")]


class TextTools:
    """
    Collection of text manipulation tools.

    These tools perform various text operations and are
    designed to be used by the Text Agent.
    """

    @staticmethod
    def to_uppercase(text: str) -> str:
        """
        Convert text to uppercase.

        Args:
            text: Text to convert

        Returns:
            Uppercase version of the text
        """
        logger.debug(f"Converting to uppercase: {text[:50]}...")
        result = text.upper()
        logger.debug(f"Result: {result[:50]}...")
        return result

    @staticmethod
    def to_lowercase(text: str) -> str:
        """
        Convert text to lowercase.

        Args:
            text: Text to convert

        Returns:
            Lowercase version of the text
        """
        logger.debug(f"Converting to lowercase: {text[:50]}...")
        result = text.lower()
        logger.debug(f"Result: {result[:50]}...")
        return result

    @staticmethod
    def reverse_text(text: str) -> str:
        """
        Reverse the order of characters in text.

        Args:
            text: Text to reverse

        Returns:
            Reversed text
        """
        logger.debug(f"Reversing: {text[:50]}...")
        result = text[::-1]
        logger.debug(f"Result: {result[:50]}...")
        return result

    @staticmethod
    def count_chars(text: str) -> int:
        """
        Count the number of characters in text.

        Args:
            text: Text to count characters in

        Returns:
            Number of characters
        """
        logger.debug(f"Counting characters in: {text[:50]}...")
        result = len(text)
        logger.debug(f"Result: {result}")
        return result

    @staticmethod
    def count_words(text: str) -> int:
        """
        Count the number of words in text.

        Args:
            text: Text to count words in

        Returns:
            Number of words
        """
        logger.debug(f"Counting words in: {text[:50]}...")
        words = text.split()
        result = len(words)
        logger.debug(f"Result: {result}")
        return result

    @staticmethod
    def capitalize(text: str) -> str:
        """
        Capitalize the first letter of each word.

        Args:
            text: Text to capitalize

        Returns:
            Text with each word capitalized
        """
        logger.debug(f"Capitalizing: {text[:50]}...")
        result = text.title()
        logger.debug(f"Result: {result[:50]}...")
        return result

    @staticmethod
    def strip_whitespace(text: str) -> str:
        """
        Remove leading and trailing whitespace.

        Args:
            text: Text to strip

        Returns:
            Text with whitespace removed
        """
        logger.debug(f"Stripping whitespace from: {repr(text[:50])}...")
        result = text.strip()
        logger.debug(f"Result: {repr(result[:50])}...")
        return result

    @staticmethod
    def search_replace(text: str, search: str, replace: str) -> str:
        """
        Replace all occurrences of search pattern with replacement.

        Args:
            text: Text to search in
            search: Pattern to search for
            replace: Replacement string

        Returns:
            Text with replacements made
        """
        logger.debug(f"Replacing '{search}' with '{replace}' in: {text[:50]}...")
        result = text.replace(search, replace)
        logger.debug(f"Result: {result[:50]}...")
        return result

    @staticmethod
    def regex_search(text: str, pattern: str) -> list[str]:
        """
        Find all matches of a regex pattern.

        Args:
            text: Text to search in
            pattern: Regex pattern to match

        Returns:
            List of all matches
        """
        logger.debug(f"Regex search for '{pattern}' in: {text[:50]}...")
        matches = re.findall(pattern, text)
        logger.debug(f"Found {len(matches)} matches")
        return matches


def get_text_tools() -> list[StructuredTool]:
    """
    Get all text tools as LangChain StructuredTools.

    Returns:
        List of StructuredTool instances for text operations
    """
    return [
        StructuredTool(
            name="to_uppercase",
            description="Convert text to uppercase letters.",
            func=TextTools.to_uppercase,
            args_schema=TextInput,
        ),
        StructuredTool(
            name="to_lowercase",
            description="Convert text to lowercase letters.",
            func=TextTools.to_lowercase,
            args_schema=TextInput,
        ),
        StructuredTool(
            name="reverse_text",
            description="Reverse the order of characters in the text.",
            func=TextTools.reverse_text,
            args_schema=TextInput,
        ),
        StructuredTool(
            name="count_chars",
            description="Count the number of characters in the text.",
            func=TextTools.count_chars,
            args_schema=TextInput,
        ),
        StructuredTool(
            name="count_words",
            description="Count the number of words in the text.",
            func=TextTools.count_words,
            args_schema=TextInput,
        ),
        StructuredTool(
            name="capitalize",
            description="Capitalize the first letter of each word in the text.",
            func=TextTools.capitalize,
            args_schema=TextInput,
        ),
        StructuredTool(
            name="strip_whitespace",
            description="Remove leading and trailing whitespace from text.",
            func=TextTools.strip_whitespace,
            args_schema=TextInput,
        ),
        StructuredTool(
            name="search_replace",
            description="Replace all occurrences of a search string with a replacement.",
            func=TextTools.search_replace,
            args_schema=SearchReplaceInput,
        ),
    ]
