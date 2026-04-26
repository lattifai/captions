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
        target_lang: Target language code for translated captions.
        kind: Caption kind (e.g., 'captions', 'subtitles', 'descriptions').
        format_metadata: Format-specific roundtrip data (e.g., ass_info,
            ass_styles, ttml_profile). Keyed by format-prefixed names.
    """

    supervisions: List["Supervision"] = field(default_factory=list)
    language: Optional[str] = None
    target_lang: Optional[str] = None
    kind: Optional[str] = None
    format_metadata: Dict[str, Any] = field(default_factory=dict)


class FormatReader(ABC):
    """Abstract base class for caption format readers.

    Subclasses must implement :meth:`parse`, which returns a
    :class:`ParseResult` holding supervisions plus language / kind /
    format-specific metadata in a single pass. There is no separate
    ``read`` or ``extract_metadata`` hook — callers that only need the
    supervision list should use ``cls.parse(source).supervisions``.

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
    def parse(
        cls,
        source: Union[Pathlike, str],
        normalize_text: bool = True,
        **kwargs,
    ) -> ParseResult:
        """Parse caption content in a single pass.

        Args:
            source: File path or string content.
            normalize_text: Whether to normalize text (strip HTML, collapse
                whitespace, etc.). Formats that do not apply normalization
                should accept and ignore this flag.
            **kwargs: Format-specific options.

        Returns:
            ParseResult with supervisions and any language/kind/format
            metadata the format can surface.
        """
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for banned in ("read", "extract_metadata"):
            if banned in cls.__dict__:
                raise TypeError(
                    f"{cls.__name__}: override parse(), not {banned}(). "
                    "FormatReader exposes a single parse() entry point; "
                    "use cls.parse(source).supervisions when you only "
                    "need the supervision list."
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
            Tuple of (render, include_speaker, word_level) where
            ``word_level`` is ``Optional[bool]`` (tri-state, see RenderConfig).
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
    def _should_include_speaker(
        cls,
        sup: Any,
        include_speaker: bool,
        tracker: Optional["SpeakerTracker"] = None,
    ) -> bool:
        """Check if speaker should be included in output text.

        Semantics:

        * ``include_speaker=False`` or ``sup.speaker`` empty → never include.
        * ``custom["original_speaker"]`` is ``True`` (the default) → the
          speaker was *explicitly* tagged by the source author or upstream
          producer. Emit the prefix unconditionally, even when consecutive
          cues share the same speaker. This preserves user intent and
          guarantees lossless round-trips for formats that recover speaker
          from cue text (e.g. SRT, VTT).
        * ``custom["original_speaker"]`` is ``False`` → the speaker was
          inherited or back-filled (e.g. by a sentence splitter that broke
          one author's utterance into multiple cues). When a *tracker* is
          provided, suppress the prefix on consecutive duplicates so the
          rendered output stays readable.

        In both branches the *tracker* (if provided) is advanced so a later
        inherited cue can correctly detect a speaker change.
        """
        if not include_speaker or not getattr(sup, "speaker", None):
            return False
        custom = getattr(sup, "custom", None)
        # Legacy semantic: an explicit ``original_speaker=False`` flag means
        # the caller has already determined this cue's speaker should not be
        # rendered (e.g. it was back-filled by a sentence splitter and is
        # inherited from a neighbour). Suppress unconditionally.
        if custom and not custom.get("original_speaker", True):
            return False
        # The speaker was explicitly tagged on this cue. Emit the prefix
        # unconditionally, even when it equals the previous cue's speaker —
        # tracker dedup is a display optimisation that must not erase the
        # user's intent and would otherwise break read-back fidelity for
        # text-encoded formats (VTT, etc.). Advance tracker state so a
        # later inherited cue still sees the correct previous speaker.
        if tracker is not None:
            tracker.is_new(sup.speaker)
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


class SpeakerTracker:
    """Track previous speaker to suppress consecutive duplicate labels.

    Use in writer loops to avoid repeating the same speaker prefix when
    standardization splits one speaker's text into multiple supervisions.

    Usage::

        tracker = SpeakerTracker()
        for sup in supervisions:
            if tracker.is_new(sup.speaker) and include_speaker:
                # emit speaker prefix
    """

    __slots__ = ("_prev",)

    def __init__(self) -> None:
        self._prev: Optional[str] = None

    def is_new(self, speaker: Optional[str]) -> bool:
        """Return True if *speaker* differs from the previous call's value."""
        if not speaker or speaker == self._prev:
            self._prev = speaker
            return False
        self._prev = speaker
        return True


class FormatHandler(FormatReader, FormatWriter):
    """Combined reader and writer for formats that support both.

    Most caption formats support both reading and writing. This class
    combines both interfaces for convenience.
    """

    pass


# Type aliases for registration
ReaderType = type[FormatReader]
WriterType = type[FormatWriter]


def format_text_with_translation(sup: "Supervision", separator: str = "\n", translation_first: bool = False) -> str:
    """Format supervision text, joining the translation if present.

    When ``sup.translation`` is empty, returns ``sup.text`` unchanged.

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


def has_word_alignment(sup: "Supervision") -> bool:
    """True if the supervision has a non-empty word-level alignment list.

    Empty lists (``alignment={"word": []}``) are treated as "no data" — they
    are a producer artifact that would otherwise crash code that does
    ``alignment["word"][0]``.
    """
    alignment = getattr(sup, "alignment", None)
    return bool(alignment and alignment.get("word"))


def count_supervisions_with_words(supervisions: List["Supervision"]) -> int:
    """Count supervisions in the batch that carry usable word alignment."""
    return sum(1 for s in supervisions if has_word_alignment(s))


def resolve_word_level(
    word_level: Optional[bool],
    *,
    n_with_words: int,
    n_total: int,
    format_id: str,
    smart_default: bool = False,
) -> bool:
    """Resolve the tri-state ``word_level`` flag into an effective boolean.

    Args:
        word_level: Tri-state from ``RenderConfig.word_level`` (None/True/False).
        n_with_words: Number of supervisions in the batch with usable word
            alignment (use :func:`count_supervisions_with_words`).
        n_total: Total number of supervisions in the batch.
        format_id: Format identifier used in the warning message.
        smart_default: For ``word_level=None``, the per-format default when
            no other signal applies. ``True`` means "data-driven word level
            when data is present"; ``False`` means "stay segment level".

    Returns:
        ``True`` if the writer should emit word-level output for this batch,
        ``False`` otherwise.

    Warnings emitted (via the ``lattifai.caption`` logger):
        - ``word_level=True`` and ``n_with_words == 0``: full fallback warning;
          returns ``False``.
        - ``word_level=True`` and ``0 < n_with_words < n_total``: partial
          warning naming the unaligned count; returns ``True`` (the per-cue
          loop is responsible for falling back on the unaligned segments).
    """
    import logging

    if word_level is True:
        logger = logging.getLogger("lattifai.caption")
        if n_with_words == 0:
            logger.warning(
                "%s: word_level=True requested but no word alignment available; "
                "falling back to segment-level output for this batch",
                format_id,
            )
            return False
        if n_with_words < n_total:
            logger.warning(
                "%s: word_level=True but %d/%d supervisions lack word alignment; "
                "those segments will fall back to segment-level output",
                format_id,
                n_total - n_with_words,
                n_total,
            )
        return True

    if word_level is False:
        return False

    # word_level is None — apply smart default, but only if data exists.
    return smart_default and n_with_words > 0


def maybe_expand_to_word_supervisions(
    supervisions: List["Supervision"],
    *,
    word_level: Optional[bool],
    format_id: str,
) -> List["Supervision"]:
    """Batch-level wrapper around :func:`expand_to_word_supervisions`.

    Honors tri-state ``word_level``:
        - ``True``: expand to one supervision per word; warn on empty or
          partially-aligned batches.
        - ``False`` / ``None``: return supervisions unchanged.
    """
    if word_level is not True:
        return supervisions
    n_total = len(supervisions)
    n_with = count_supervisions_with_words(supervisions)
    use = resolve_word_level(
        True,
        n_with_words=n_with,
        n_total=n_total,
        format_id=format_id,
        smart_default=False,
    )
    if not use:
        return supervisions
    return expand_to_word_supervisions(supervisions)


def expand_to_word_supervisions(supervisions: List["Supervision"]) -> List["Supervision"]:
    """Expand supervisions with word alignment to one supervision per word.

    Used for word-per-segment output when word_level=True.

    Args:
        supervisions: List of Supervision objects with optional alignment data

    Returns:
        List of Supervision objects, one per word if alignment exists (and is
        non-empty), otherwise the original supervision is preserved.
    """
    from ..supervision import Supervision

    result = []
    for sup in supervisions:
        if has_word_alignment(sup):
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
