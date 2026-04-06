"""Base classes for caption format readers and writers.

This module provides abstract base classes that all format handlers must implement,
ensuring a consistent interface across different caption formats.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from ..supervision import Pathlike

if TYPE_CHECKING:
    from ..supervision import Supervision


@dataclass
class ParseResult:
    """Result of parsing caption content.

    Combines supervisions with metadata in a single return value,
    eliminating the need for separate read() + extract_metadata() calls.

    Attributes:
        supervisions: Parsed caption segments with timing and text.
        language: Source language code (e.g., 'en', 'zh-Hans').
        kind: Caption kind (e.g., 'captions', 'subtitles', 'descriptions').
        format_metadata: Format-specific roundtrip data (e.g., ass_info,
            ass_styles, ttml_profile). Keyed by format-prefixed names.
    """

    supervisions: List["Supervision"] = field(default_factory=list)
    language: Optional[str] = None
    kind: Optional[str] = None
    format_metadata: Dict[str, Any] = field(default_factory=dict)


class FormatReader(ABC):
    """Abstract base class for caption format readers.

    All format readers must implement the `read` method to parse caption content
    and return a list of Supervision objects.

    Class Attributes:
        format_id: Unique identifier for the format (e.g., "srt", "vtt")
        extensions: List of file extensions this reader handles (e.g., [".srt"])
        description: Human-readable description of the format
    """

    format_id: str = ""
    extensions: List[str] = []
    description: str = ""

    @classmethod
    @abstractmethod
    def read(
        cls,
        source: Union[Pathlike, str],
        normalize_text: bool = True,
        **kwargs,
    ) -> List["Supervision"]:
        """Read caption content and return list of Supervision objects.

        Args:
            source: File path or string content
            normalize_text: Whether to normalize text (strip HTML, etc.)
            **kwargs: Format-specific options

        Returns:
            List of Supervision objects with timing and text
        """
        pass

    @classmethod
    def extract_metadata(cls, source: Union[Pathlike, str]) -> Dict[str, str]:
        """Extract metadata from caption file or content.

        Deprecated: override parse() instead, which returns ParseResult
        with metadata included.

        Args:
            source: File path or string content

        Returns:
            Dictionary of metadata key-value pairs
        """
        return {}

    @classmethod
    def parse(
        cls,
        source: Union[Pathlike, str],
        normalize_text: bool = True,
        **kwargs,
    ) -> ParseResult:
        """Parse caption content in a single pass.

        Returns ParseResult containing supervisions, language, kind,
        and format-specific metadata. Subclasses should override this
        to fold metadata extraction into parsing.

        Default implementation calls read() + extract_metadata() for
        backward compatibility.

        Args:
            source: File path or string content
            normalize_text: Whether to normalize text
            **kwargs: Format-specific options

        Returns:
            ParseResult with supervisions and metadata
        """
        supervisions = cls.read(source, normalize_text=normalize_text, **kwargs)
        metadata = cls.extract_metadata(source)
        return ParseResult(
            supervisions=supervisions,
            language=metadata.pop("language", None),
            kind=metadata.pop("kind", None),
            format_metadata=metadata,
        )

    @classmethod
    def can_read(cls, path: Union[Pathlike, str]) -> bool:
        """Check if this reader can handle the given file.

        Args:
            path: File path to check

        Returns:
            True if this reader supports the file format
        """
        path_str = str(path).lower()
        return any(path_str.endswith(ext.lower()) for ext in cls.extensions)

    @classmethod
    def is_content(cls, source: Union[Pathlike, str]) -> bool:
        """Check if source is string content rather than a file path.

        Args:
            source: Source to check

        Returns:
            True if source appears to be content, not a path
        """
        if not isinstance(source, str):
            return False
        # If it has newlines or is very long, it's likely content
        return "\n" in source or len(source) > 500


class FormatWriter(ABC):
    """Abstract base class for caption format writers.

    All format writers must implement `write` and `to_bytes` methods.

    Class Attributes:
        format_id: Unique identifier for the format (e.g., "srt", "vtt")
        extensions: List of file extensions for this format
        description: Human-readable description of the format
    """

    format_id: str = ""
    extensions: List[str] = []
    description: str = ""

    @classmethod
    @abstractmethod
    def write(
        cls,
        supervisions: List["Supervision"],
        output_path: Pathlike,
        **kwargs,
    ) -> Path:
        """Write supervisions to a file.

        Args:
            supervisions: List of Supervision objects to write
            output_path: Path to output file
            **kwargs: Format-specific options (style, karaoke, metadata, etc.)

        Returns:
            Path to the written file
        """
        pass

    @classmethod
    @abstractmethod
    def to_bytes(
        cls,
        supervisions: List["Supervision"],
        **kwargs,
    ) -> bytes:
        """Convert supervisions to bytes in this format.

        Args:
            supervisions: List of Supervision objects
            **kwargs: Format-specific options (style, karaoke, metadata, etc.)

        Returns:
            Caption content as bytes
        """
        pass

    @classmethod
    def _unpack_render(cls, render=None, **kwargs):
        """Extract output render flags from a RenderConfig instance.

        Args:
            render: RenderConfig instance or None (uses defaults)
            **kwargs: Remaining kwargs; 'render' is consumed if present.

        Returns:
            Tuple of (render, include_speaker, word_level)
        """
        from ..config import RenderConfig

        # Accept render from kwargs (when passed via Caption.write())
        if render is None:
            render = kwargs.pop("render", None)
        else:
            kwargs.pop("render", None)
        render = render or RenderConfig()
        return render, render.include_speaker_in_text, render.word_level

    @classmethod
    def _should_include_speaker(cls, sup: Any, include_speaker: bool) -> bool:
        """Check if speaker should be included in output text.

        Considers both the global include_speaker flag and the segment-level
        'original_speaker' flag in custom metadata.
        """
        if not include_speaker or not getattr(sup, "speaker", None):
            return False
        custom = getattr(sup, "custom", None)
        if custom and not custom.get("original_speaker", True):
            return False
        return True

    @staticmethod
    def _format_speaker_prefix(speaker: str) -> str:
        """Format speaker name with a colon separator for text prepending.

        Special case: ">>" (anonymous speaker change marker) gets a trailing
        space only — no colon — so output reads ">> text" not ">>: text".

        If the speaker name already ends with ':' or '：', a trailing space
        is appended. Otherwise ': ' is appended so the reader's
        parse_speaker_text() can reliably extract the speaker on read-back.
        """
        stripped = speaker.rstrip()
        if stripped == ">>":
            return ">> "
        if stripped and stripped[-1] in (":", "："):
            return f"{stripped} "
        return f"{speaker}: "


class FormatHandler(FormatReader, FormatWriter):
    """Combined reader and writer for formats that support both.

    Most caption formats support both reading and writing. This class
    combines both interfaces for convenience.
    """

    pass


# Type aliases for registration
ReaderType = type[FormatReader]
WriterType = type[FormatWriter]


def render_bilingual_text(sup: "Supervision", separator: str = "\n", translation_first: bool = False) -> str:
    """Render supervision text with translation.

    Args:
        sup: Supervision object
        separator: Separator between original text and translation
        translation_first: If True, place translation above original text

    Returns:
        Combined text string
    """
    text = sup.text or ""
    if sup.translation:
        if translation_first:
            text = f"{sup.translation}{separator}{text}"
        else:
            text = f"{text}{separator}{sup.translation}"
    return text


def strip_standard_kwargs(kwargs: dict) -> None:
    """Remove standard Caption.write() kwargs that NLE/professional formats don't support.

    This avoids repetitive kwargs.pop() blocks in every NLE format wrapper.
    Mutates the dict in-place.
    """
    for key in ("metadata", "render", "config"):
        kwargs.pop(key, None)


def expand_to_word_supervisions(supervisions: List["Supervision"]) -> List["Supervision"]:
    """Expand supervisions with word alignment to one supervision per word.

    Used for word-per-segment output when word_level=True.

    Args:
        supervisions: List of Supervision objects with optional alignment data

    Returns:
        List of Supervision objects, one per word if alignment exists,
        otherwise returns original supervisions unchanged.
    """
    from ..supervision import Supervision

    result = []
    for sup in supervisions:
        if sup.alignment and "word" in sup.alignment:
            for word in sup.alignment["word"]:
                result.append(
                    Supervision(
                        text=word.symbol,
                        start=word.start,
                        duration=word.duration,
                        speaker=sup.speaker,
                        id=f"{sup.id}_word" if sup.id else "",
                        recording_id=sup.recording_id if hasattr(sup, "recording_id") else "",
                    )
                )
        else:
            result.append(sup)
    return result
