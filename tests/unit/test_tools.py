"""Unit tests for tools module."""

import pytest

from agentic_ai_mcp.tools.math_tools import MathTools, get_math_tools
from agentic_ai_mcp.tools.text_tools import TextTools, get_text_tools


class TestMathTools:
    """Tests for MathTools class."""

    def test_add(self):
        """Test addition operation."""
        assert MathTools.add(2, 3) == 5
        assert MathTools.add(-1, 1) == 0
        assert MathTools.add(0, 0) == 0

    def test_subtract(self):
        """Test subtraction operation."""
        assert MathTools.subtract(5, 3) == 2
        assert MathTools.subtract(3, 5) == -2
        assert MathTools.subtract(0, 0) == 0

    def test_multiply(self):
        """Test multiplication operation."""
        assert MathTools.multiply(3, 4) == 12
        assert MathTools.multiply(-2, 3) == -6
        assert MathTools.multiply(0, 100) == 0

    def test_divide(self):
        """Test division operation."""
        assert MathTools.divide(10, 2) == 5.0
        assert MathTools.divide(7, 2) == 3.5
        assert MathTools.divide(-6, 2) == -3.0

    def test_divide_by_zero(self):
        """Test division by zero raises error."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            MathTools.divide(10, 0)

    def test_power(self):
        """Test power operation."""
        assert MathTools.power(2, 3) == 8.0
        assert MathTools.power(10, 0) == 1.0
        assert MathTools.power(2, -1) == 0.5

    def test_sqrt(self):
        """Test square root operation."""
        assert MathTools.sqrt(4) == 2.0
        assert MathTools.sqrt(9) == 3.0
        assert MathTools.sqrt(0) == 0.0

    def test_sqrt_negative(self):
        """Test square root of negative raises error."""
        with pytest.raises(ValueError, match="Cannot compute square root of negative"):
            MathTools.sqrt(-4)

    def test_get_math_tools_returns_list(self):
        """Test that get_math_tools returns a list of tools."""
        tools = get_math_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_get_math_tools_names(self):
        """Test that math tools have expected names."""
        tools = get_math_tools()
        names = {t.name for t in tools}
        assert "add" in names
        assert "subtract" in names
        assert "multiply" in names
        assert "divide" in names


class TestTextTools:
    """Tests for TextTools class."""

    def test_to_uppercase(self):
        """Test uppercase conversion."""
        assert TextTools.to_uppercase("hello") == "HELLO"
        assert TextTools.to_uppercase("Hello World") == "HELLO WORLD"
        assert TextTools.to_uppercase("") == ""

    def test_to_lowercase(self):
        """Test lowercase conversion."""
        assert TextTools.to_lowercase("HELLO") == "hello"
        assert TextTools.to_lowercase("Hello World") == "hello world"
        assert TextTools.to_lowercase("") == ""

    def test_reverse_text(self):
        """Test text reversal."""
        assert TextTools.reverse_text("hello") == "olleh"
        assert TextTools.reverse_text("12345") == "54321"
        assert TextTools.reverse_text("") == ""

    def test_count_chars(self):
        """Test character counting."""
        assert TextTools.count_chars("hello") == 5
        assert TextTools.count_chars("") == 0
        assert TextTools.count_chars("a b c") == 5

    def test_count_words(self):
        """Test word counting."""
        assert TextTools.count_words("hello world") == 2
        assert TextTools.count_words("one") == 1
        assert TextTools.count_words("") == 0

    def test_capitalize(self):
        """Test title case capitalization."""
        assert TextTools.capitalize("hello world") == "Hello World"
        assert TextTools.capitalize("HELLO") == "Hello"

    def test_strip_whitespace(self):
        """Test whitespace stripping."""
        assert TextTools.strip_whitespace("  hello  ") == "hello"
        assert TextTools.strip_whitespace("\t\nhello\n\t") == "hello"

    def test_search_replace(self):
        """Test search and replace."""
        assert TextTools.search_replace("hello world", "world", "there") == "hello there"
        assert TextTools.search_replace("aaa", "a", "b") == "bbb"

    def test_get_text_tools_returns_list(self):
        """Test that get_text_tools returns a list of tools."""
        tools = get_text_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_get_text_tools_names(self):
        """Test that text tools have expected names."""
        tools = get_text_tools()
        names = {t.name for t in tools}
        assert "to_uppercase" in names
        assert "to_lowercase" in names
        assert "reverse_text" in names
        assert "count_chars" in names
