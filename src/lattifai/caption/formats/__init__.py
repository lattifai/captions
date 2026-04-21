"""Caption format handlers registry.

This module provides a central registry for all caption format readers and writers.
Formats are registered using decorators and can be looked up by format ID.

Example:
    >>> from lattifai.caption.formats import get_reader, get_writer
    >>> reader = get_reader("srt")
    >>> result = reader.parse("input.srt")
    >>> writer = get_writer("vtt")
    >>> writer.write(result.supervisions, "output.vtt")
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Type

# Import base classes first (no dependencies on this module)
from .base import FormatHandler, FormatReader, FormatWriter

# Global registries - must be defined before format module imports
_READERS: Dict[str, Type[FormatReader]] = {}
_WRITERS: Dict[str, Type[FormatWriter]] = {}


def register_reader(format_id: str):
    """Decorator to register a format reader.

    Args:
        format_id: Unique identifier for the format (e.g., "srt", "vtt")

    Example:
        @register_reader("srt")
        class SRTReader(FormatReader):
            ...
    """

    def decorator(cls: Type[FormatReader]) -> Type[FormatReader]:
        cls.format_id = format_id
        _READERS[format_id.lower()] = cls
        return cls

    return decorator


def register_writer(format_id: str):
    """Decorator to register a format writer.

    Args:
        format_id: Unique identifier for the format

    Example:
        @register_writer("srt")
        class SRTWriter(FormatWriter):
            ...
    """

    def decorator(cls: Type[FormatWriter]) -> Type[FormatWriter]:
        cls.format_id = format_id
        _WRITERS[format_id.lower()] = cls
        return cls

    return decorator


def register_format(format_id: str):
    """Decorator to register both reader and writer for a format.

    Use this for classes that implement both FormatReader and FormatWriter.

    Args:
        format_id: Unique identifier for the format

    Example:
        @register_format("srt")
        class SRTFormat(FormatHandler):
            ...
    """

    def decorator(cls: Type[FormatHandler]) -> Type[FormatHandler]:
        cls.format_id = format_id
        _READERS[format_id.lower()] = cls
        _WRITERS[format_id.lower()] = cls
        return cls

    return decorator


def get_reader(format_id: str) -> Optional[Type[FormatReader]]:
    """Get a reader class by format ID.

    Args:
        format_id: Format identifier (case-insensitive)

    Returns:
        Reader class or None if not found
    """
    return _READERS.get(format_id.lower())


def get_writer(format_id: str) -> Optional[Type[FormatWriter]]:
    """Get a writer class by format ID.

    Args:
        format_id: Format identifier (case-insensitive)

    Returns:
        Writer class or None if not found
    """
    return _WRITERS.get(format_id.lower())


def list_readers() -> List[str]:
    """Get list of all registered reader format IDs."""
    return sorted(_READERS.keys())


def list_writers() -> List[str]:
    """Get list of all registered writer format IDs."""
    return sorted(_WRITERS.keys())


def detect_format(path: str) -> Optional[str]:
    """Detect format from file path by checking registered readers.

    Args:
        path: File path to check

    Returns:
        Format ID or None if no match found
    """
    path_str = str(path)

    # Check if it's content instead of a path
    is_content = "\n" in path_str or len(path_str) > 500

    # Prioritize specific formats that can detect by content
    # These often use shared extensions like .vtt, .txt, or .xml
    priority_formats = ["vtt", "markdown", "premiere_xml"]
    for format_id in priority_formats:
        reader_cls = _READERS.get(format_id)
        if reader_cls and reader_cls.can_read(path_str):
            return format_id

    if is_content:
        return None

    # Check each reader's extensions
    path_lower = path_str.lower()
    for format_id, reader_cls in _READERS.items():
        if format_id in priority_formats:
            continue
        if reader_cls.can_read(path_lower):
            return format_id

    # Fallback: try extension directly
    from pathlib import Path

    try:
        ext = Path(path_lower).suffix.lstrip(".")
        if ext in _READERS:
            return ext
    except (OSError, ValueError):
        # Likely content, not a path
        pass

    return None


def detect_format_from_content(content: str) -> Optional[str]:
    """Detect caption format by inspecting content signatures.

    Designed for the "paste into textarea" scenario where no file extension
    is available. Checks distinctive markers in priority order.

    Args:
        content: Raw caption text (e.g., pasted from clipboard).

    Returns:
        Format ID string, or None if unrecognizable.
    """
    import re

    stripped = content.lstrip("\ufeff").lstrip()

    # VTT — must start with "WEBVTT"
    if stripped.startswith("WEBVTT"):
        return "vtt"

    # ASS / SSA — "[Script Info]" header
    if stripped.startswith("[Script Info]"):
        if "v4.00+" in stripped[:200]:
            return "ass"
        return "ssa"

    # SRV3 / YouTube timed text — XML with <timedtext
    if stripped.startswith("<") and "<timedtext" in stripped[:500]:
        return "srv3"

    # TTML — XML with <tt namespace (also matches <tt:tt prefix)
    if stripped.startswith("<") and re.search(r"<tt[\s>:]", stripped[:500]):
        return "ttml"

    # JSON3 — YouTube native format with "wireMagic" or "events"+"tStartMs"
    if stripped.startswith("{") and (
        '"wireMagic"' in stripped[:500]
        or ('"events"' in stripped[:2000] and '"tStartMs"' in stripped[:2000])
    ):
        return "json3"

    # JSON — array or object with "text" key
    if stripped.startswith(("{", "[")) and '"text"' in stripped[:2000]:
        return "json"

    # Markdown transcript — [HH:MM:SS] timestamps, **Speaker:** patterns, or
    # bare timestamp headings (#### HH:MM:SS Title) with plain speaker labels.
    # Must check BEFORE LRC because [MM:SS] overlaps with LRC timestamps.
    has_md_ts = bool(re.search(r"\[\d{1,2}:\d{2}(?::\d{2})?\]", stripped[:4096]))
    has_md_spk = bool(re.search(r"\*\*.+?[:：]\*\*", stripped[:4096]))
    has_bare_ts = bool(re.search(r"^#{2,4}\s+\d{1,2}:\d{2}:\d{2}", stripped[:4096], re.MULTILINE))
    has_plain_spk = bool(
        re.search(r"^[A-Z][a-zA-Z'\-]+(?:\s+[A-Za-z'\-]+){0,3}:\s+\S", stripped[:4096], re.MULTILINE)
    )
    if has_md_spk or (has_md_ts and has_md_spk) or (has_bare_ts and has_plain_spk):
        return "markdown"

    # LRC — lines starting with [mm:ss.xx] immediately followed by lyric text.
    # Requires the closing bracket pattern to avoid false positives with markdown [MM:SS].
    if re.search(r"^\[\d{1,3}:\d{2}[.\d]*\]", stripped, re.MULTILINE):
        # If also has markdown speaker labels, prefer markdown
        if has_md_ts:
            return "markdown"
        return "lrc"

    # Markdown (timestamp-only, no speaker labels) — after LRC is ruled out
    if has_md_ts:
        return "markdown"

    # SRT — "index\ntimestamp --> timestamp" pattern
    if re.search(r"^\d+\s*\n\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->", stripped, re.MULTILINE):
        return "srt"

    # SBV — "timestamp,timestamp" pattern (YouTube legacy)
    if re.search(r"^\d{1,2}:\d{2}:\d{2}\.\d{3},\d{1,2}:\d{2}:\d{2}\.\d{3}", stripped, re.MULTILINE):
        return "sbv"

    # FCPXML — XML with <fcpxml
    if stripped.startswith("<") and "<fcpxml" in stripped[:500]:
        return "fcpxml"

    # Premiere XML — XML with <xmeml
    if stripped.startswith("<") and "<xmeml" in stripped[:500]:
        return "premiere_xml"

    # CSV — header line with "start" and "text" columns, comma-separated
    first_line = stripped.split("\n", 1)[0].lower()
    if "," in first_line and "start" in first_line and "text" in first_line:
        return "csv"

    # TSV — header line with "start" and "text" columns, tab-separated
    if "\t" in first_line and "start" in first_line and "text" in first_line:
        return "tsv"

    # Audacity labels — "start\tend\ttext" numeric pattern
    if re.search(r"^\d+\.?\d*\t\d+\.?\d*\t", stripped, re.MULTILINE):
        return "aud"

    # TextGrid — Praat header
    if stripped.startswith('File type = "ooTextFile"') or stripped.startswith("Object class"):
        return "textgrid"

    # MicroDVD SUB — {frame}{frame}text
    if re.search(r"^\{\d+\}\{\d+\}", stripped, re.MULTILINE):
        return "sub"

    # SAMI — <SAMI> tag
    if "<SAMI>" in stripped[:500].upper():
        return "sami"

    # Fallback: treat as plain text (e.g., raw audio transcript without timestamps)
    return "txt"


# Import all format modules to trigger registration
# These imports MUST come after register_* functions are defined
# Standard formats
from . import gemini  # noqa: E402  # Backward-compatible aliases for markdown format
from . import lrc  # noqa: E402  # Enhanced LRC with word-level timestamps
from . import markdown  # noqa: E402  # Markdown transcript format
from . import pysubs2  # noqa: E402  # SRT, ASS, SSA, SUB, SAMI
from . import sbv  # noqa: E402  # SubViewer
from . import srv3  # noqa: E402  # YouTube SRV3/YTT format
from . import tabular  # noqa: E402  # CSV, TSV, AUD, TXT
from . import textgrid  # noqa: E402  # Praat TextGrid
from . import ttml  # noqa: E402  # TTML, IMSC1, EBU-TT-D
from . import vtt  # noqa: E402  # WebVTT with YouTube VTT word-level timestamp support
from . import json as json_format  # noqa: E402  # JSON structured format
from . import json3  # noqa: E402  # YouTube JSON3 timed text format

# Professional NLE formats
from .nle import audition  # noqa: E402  # Adobe Audition / Pro Tools markers
from .nle import avid  # noqa: E402  # Avid DS
from .nle import fcpxml  # noqa: E402  # Final Cut Pro XML
from .nle import premiere  # noqa: E402  # Adobe Premiere Pro XML

__all__ = [
    # Base classes
    "FormatReader",
    "FormatWriter",
    "FormatHandler",
    # Registration
    "register_reader",
    "register_writer",
    "register_format",
    # Lookup
    "get_reader",
    "get_writer",
    "list_readers",
    "list_writers",
    "detect_format",
    "detect_format_from_content",
]
