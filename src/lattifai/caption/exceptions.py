"""Custom exception types for caption processing.

Provides structured error reporting so callers can distinguish between
format detection failures, unsupported formats, and parse-time errors.
"""


class CaptionError(Exception):
    """Base exception for all caption-related errors."""


class FormatDetectionError(CaptionError):
    """Raised when caption format cannot be auto-detected from content."""


class FormatNotSupportedError(CaptionError):
    """Raised when the requested format has no registered reader/writer."""


class CaptionParseError(CaptionError):
    """Raised when caption content cannot be parsed by the selected reader."""
