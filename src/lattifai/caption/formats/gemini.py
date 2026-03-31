"""Backward-compatible aliases for the markdown format (formerly gemini).

All classes have been moved to markdown.py. Import from there for new code.
"""

from .markdown import MarkdownFormat, MarkdownReader, MarkdownSegment, MarkdownWriter

# Backward-compatible aliases
GeminiFormat = MarkdownFormat
GeminiReader = MarkdownReader
GeminiSegment = MarkdownSegment
GeminiWriter = MarkdownWriter

# Register "gemini" as an alias format pointing to the same implementation
from . import register_format as _register_format


@_register_format("gemini")
class _GeminiFormatAlias(MarkdownFormat):
    """Backward-compatible alias for MarkdownFormat."""

    pass


__all__ = ["GeminiReader", "GeminiWriter", "GeminiSegment", "GeminiFormat"]
