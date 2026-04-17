"""Standard subtitle formats using pysubs2 library.

Handles: SRT, VTT, ASS, SSA, SUB (MicroDVD), SAMI/SMI
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pysubs2

from ..colors import SPEAKER_PALETTE, hex_rgb_to_bgr, resolve_speaker_color
from ..config import ASSConfig
from ..parsers.text_parser import classify_line_type, detect_speaker_candidates
from ..parsers.text_parser import normalize_text as normalize_text_fn
from ..parsers.text_parser import parse_speaker_text, set_speaker_candidates
from ..supervision import Supervision
from . import register_format
from .base import FormatHandler

logger = logging.getLogger(__name__)


def detect_file_encoding(path: Path) -> Tuple[str, str]:
    """Detect file encoding via BOM sniffing then chardet fallback.

    Fallback chain: BOM -> chardet -> utf-8 -> gbk -> latin-1.

    Args:
        path: Path to the file.

    Returns:
        Tuple of (decoded_content_string, detected_encoding_name).
    """
    raw = path.read_bytes()

    # BOM detection (most reliable)
    if raw[:3] == b"\xef\xbb\xbf":
        return raw[3:].decode("utf-8"), "utf-8-sig"
    if raw[:2] == b"\xff\xfe":
        return raw.decode("utf-16-le"), "utf-16-le"
    if raw[:2] == b"\xfe\xff":
        return raw.decode("utf-16-be"), "utf-16-be"

    # chardet detection (only trust high-confidence results)
    try:
        import chardet

        detected = chardet.detect(raw)
        if detected and detected.get("encoding") and detected.get("confidence", 0) > 0.5:
            enc = detected["encoding"].lower()
            # Normalize chardet names
            if enc in ("gb2312", "gbk", "gb18030"):
                enc = "gb18030"  # gb18030 is superset of gbk/gb2312
            try:
                return raw.decode(enc), enc
            except (UnicodeDecodeError, LookupError):
                pass
    except ImportError:
        pass

    # Manual fallback chain
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue

    # Last resort
    return raw.decode("utf-8", errors="replace"), "utf-8"


class Pysubs2Format(FormatHandler):
    """Base class for formats handled by pysubs2."""

    # Subclasses should set these
    pysubs2_format: str = ""
    # Whether to deduplicate consecutive same-speaker labels in output.
    # SRT disables this: speaker prefix is the only way to recover speaker
    # on read-back, so every cue must retain its prefix for roundtrip fidelity.
    _dedup_speaker: bool = True

    # Patterns to filter Gemini-style thinking/meta blocks
    FRONTMATTER_PATTERN = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
    THINKING_PATTERN = re.compile(r"<thinking>.*?</thinking>", re.DOTALL)

    @classmethod
    def _preprocess_content(cls, content: str) -> str:
        """Remove Gemini-style thinking/meta blocks from content.

        Filters out:
        - YAML front matter (---\\n...\\n---)
        - <thinking>...</thinking> blocks
        """
        content = cls.FRONTMATTER_PATTERN.sub("", content)
        content = cls.THINKING_PATTERN.sub("", content)
        return content

    @classmethod
    def read(
        cls,
        source,
        normalize_text: bool = True,
        **kwargs,
    ) -> List[Supervision]:
        """Read caption using pysubs2.

        Preprocesses content to remove Gemini-style thinking/meta blocks
        before parsing with pysubs2.
        """
        # Preprocess content to remove thinking/meta blocks
        if cls.is_content(source):
            content = cls._preprocess_content(source)
        else:
            path = Path(source)
            content, _ = detect_file_encoding(path)
            content = cls._preprocess_content(content)

        try:
            subs = pysubs2.SSAFile.from_string(content, format_=cls.pysubs2_format)
        except Exception:
            # Fallback: auto-detect format
            subs = pysubs2.SSAFile.from_string(content)

        # Auto-detect title-case speaker candidates from all lines
        all_texts = [e.text for e in subs.events]
        candidates = detect_speaker_candidates(all_texts)
        if candidates:
            set_speaker_candidates(candidates)

        supervisions = []
        for event in subs.events:
            text = event.text
            # pysubs2 uses \N internally for line breaks (even in SRT/VTT).
            # Convert to \n to preserve multiline structure; normalize_text
            # will collapse to space if enabled. When normalize_text=False,
            # callers get true newlines for round-tripping or custom reflow.
            text = text.replace("\\N", "\n")
            if normalize_text:
                text = normalize_text_fn(text)

            speaker, text = parse_speaker_text(text)

            # Strip speaker prefix from text when event.name matches but
            # parse_speaker_text couldn't extract it (title-case with <3 occurrences)
            if not speaker and event.name:
                for sep in (": ", "： "):
                    prefix = event.name + sep
                    if text.startswith(prefix):
                        text = text[len(prefix) :]
                        break

            supervisions.append(
                Supervision(
                    text=text,
                    speaker=speaker or event.name or None,
                    start=event.start / 1000.0 if event.start is not None else 0,
                    duration=(event.end - event.start) / 1000.0
                    if event.end is not None
                    else 0,
                )
            )

        # Clear candidates to avoid leaking into subsequent reads
        if candidates:
            set_speaker_candidates(set())

        return supervisions

    @classmethod
    def extract_metadata(cls, source, **kwargs) -> Dict[str, str]:
        """Extract metadata from VTT or SRT."""
        metadata = {}
        if cls.is_content(source):
            content = source[:4096]
        else:
            path = Path(str(source))
            if not path.exists():
                return {}
            try:
                content, _ = detect_file_encoding(path)
                content = content[:4096]
            except Exception:
                return {}

        # WebVTT metadata extraction
        if cls.pysubs2_format == "vtt" or (
            isinstance(source, str) and source.startswith("WEBVTT")
        ):
            lines = content.split("\n")
            for line in lines[:10]:
                line = line.strip()
                if line.startswith("Kind:"):
                    metadata["kind"] = line.split(":", 1)[1].strip()
                elif line.startswith("Language:"):
                    metadata["language"] = line.split(":", 1)[1].strip()
                elif line.startswith("NOTE"):
                    match = re.search(r"NOTE\s+(\w+):\s*(.+)", line)
                    if match:
                        key, value = match.groups()
                        metadata[key.lower()] = value.strip()

        # SRT doesn't have standard metadata, but check for BOM
        elif cls.pysubs2_format == "srt":
            if content.startswith("\ufeff"):
                metadata["encoding"] = "utf-8-sig"

        return metadata

    @classmethod
    def parse(cls, source, normalize_text: bool = True, **kwargs) -> "ParseResult":
        """Parse pysubs2-based format in a single pass."""
        from .base import ParseResult

        supervisions = cls.read(source, normalize_text=normalize_text, **kwargs)
        metadata = cls.extract_metadata(source)
        return ParseResult(
            supervisions=supervisions,
            language=metadata.pop("language", None),
            kind=metadata.pop("kind", None),
            format_metadata=metadata,
        )

    @classmethod
    def write(
        cls,
        supervisions: List[Supervision],
        output_path,
        fps: float = 25.0,
        **kwargs,
    ) -> Path:
        """Write caption using pysubs2."""
        output_path = Path(output_path)
        content = cls.to_bytes(supervisions, fps=fps, **kwargs)
        output_path.write_bytes(content)
        return output_path

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        fps: float = 25.0,
        **kwargs,
    ) -> bytes:
        """Convert to bytes using pysubs2.

        Args:
            supervisions: List of Supervision objects
            fps: Frames per second (for MicroDVD format)

        Returns:
            Subtitle content as bytes
        """
        from .base import maybe_expand_to_word_supervisions

        behavior, include_speaker, word_level = cls._unpack_render(**kwargs)

        # Tri-state: only word_level=True triggers expansion; None/False stay segment.
        supervisions = maybe_expand_to_word_supervisions(
            supervisions, word_level=word_level, format_id=cls.format_id
        )

        from .base import render_bilingual_text

        from .base import SpeakerTracker

        subs = pysubs2.SSAFile()

        tf = behavior.translation_first
        tracker = SpeakerTracker() if cls._dedup_speaker else None
        for sup in supervisions:
            text = render_bilingual_text(sup, translation_first=tf)
            if cls._should_include_speaker(sup, include_speaker, tracker):
                text = f"{cls._format_speaker_prefix(sup.speaker)}{text}"

            subs.append(
                pysubs2.SSAEvent(
                    start=round(sup.start * 1000),
                    end=round(sup.end * 1000),
                    text=text,
                    name=sup.speaker or "",
                )
            )

        # MicroDVD format requires framerate
        if cls.pysubs2_format == "microdvd":
            return subs.to_string(format_=cls.pysubs2_format, fps=fps).encode("utf-8")

        return subs.to_string(format_=cls.pysubs2_format).encode("utf-8")


@register_format("srt")
class SRTFormat(Pysubs2Format):
    """SRT (SubRip) format - the most widely used subtitle format.

    Supports limited HTML tags in most modern players:
    <b>, <i>, <u>, <font color="#RRGGBB">
    """

    extensions = [".srt"]
    pysubs2_format = "srt"
    description = "SubRip Subtitle format - universal compatibility"
    _dedup_speaker = False  # SRT needs every cue's speaker prefix for roundtrip

    # Font-wrapped speaker: <font color="...">Name: </font>
    _FONT_SPEAKER_RE = re.compile(r'<font\s+color="[^"]*">([^<]+?):\s*</font>')

    # SRT cue header: "<idx>\n<HH:MM:SS,mmm> --> <HH:MM:SS,mmm>"
    _SRT_CUE_HEADER_RE = re.compile(
        r"^\s*(\d+)\s*\r?\n"
        r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})[^\r\n]*\r?\n",
        re.MULTILINE,
    )

    @staticmethod
    def _extract_srt_raw_texts(content: str) -> List[str]:
        """Extract raw text body for each SRT cue (preserving ASS override tags).

        pysubs2's SRT parser strips inline ``{\\an1}{\\pos(...)}`` override tags
        and collapses whitespace on read. Subtitle groups rely on these tags to
        position sign/credit/title lines. We capture the original text body per
        cue here so the writer can splice it back verbatim on roundtrip.

        Returns one raw-text string per cue, in source order. Multi-line bodies
        keep their ``\\n`` separators.
        """
        # Strip BOM if present (only at file start)
        if content.startswith("\ufeff"):
            content = content[1:]

        raw_texts: List[str] = []
        matches = list(SRTFormat._SRT_CUE_HEADER_RE.finditer(content))
        for i, m in enumerate(matches):
            body_start = m.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            body = content[body_start:body_end]
            # Strip the trailing blank-line delimiter (one blank line between cues)
            body = re.sub(r"(\r?\n)+\s*$", "", body)
            # Normalize CRLF → LF within body; writer restores terminator from metadata
            body = body.replace("\r\n", "\n")
            raw_texts.append(body)
        return raw_texts

    @classmethod
    def read(cls, source, normalize_text: bool = True, **kwargs) -> List[Supervision]:
        """Read SRT with font-wrapped speaker color + raw-text preservation.

        Preserves ``{\\an1}{\\pos(...)}`` style override tags (used by subtitle
        groups for sign/credit/title positioning) in ``supervision.custom['srt_raw_text']``
        so roundtrip writes can restore them verbatim.
        """
        # Get raw content to extract font-wrapped speakers + raw cue texts
        if cls.is_content(source):
            raw = source
        else:
            raw, _ = detect_file_encoding(Path(str(source)))

        # Extract speaker names from <font color="...">Name: </font> patterns
        font_speakers = set(cls._FONT_SPEAKER_RE.findall(raw))
        if font_speakers:
            # Inject these as speaker candidates so parse_speaker_text recognizes
            # title-case names like "Alice:" even with fewer than 3 occurrences
            set_speaker_candidates(font_speakers)

        # Extract raw cue texts before pysubs2 strips override tags
        raw_texts = cls._extract_srt_raw_texts(raw)

        result = super().read(source, normalize_text=normalize_text, **kwargs)

        if font_speakers:
            set_speaker_candidates(set())

        # Attach raw_text per-cue (order-aligned with pysubs2 events)
        if len(raw_texts) == len(result):
            for sup, raw_text in zip(result, raw_texts):
                if sup.custom is None:
                    sup.custom = {}
                sup.custom["srt_raw_text"] = raw_text

        return result

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        use_bom: bool = False,
        metadata: Optional[Dict] = None,
        **kwargs,
    ) -> bytes:
        """Generate SRT with proper formatting (comma for milliseconds).

        Args:
            supervisions: List of supervision segments
            use_bom: Whether to add BOM for Windows compatibility
            metadata: Optional metadata dict. If encoding is 'utf-8-sig', adds BOM.

        Keyword Args:
            config: SRTConfig with speaker_color for HTML <font color> tags.
            render: RenderConfig with speaker_color fallback.
        """
        from ..config import RenderConfig, SRTConfig

        config = kwargs.pop("config", None)
        render = kwargs.get("render", None)

        # Priority: SRTConfig.speaker_color > RenderConfig.speaker_color
        speaker_color = ""
        if isinstance(config, SRTConfig) and config.speaker_color:
            speaker_color = config.speaker_color
        elif isinstance(render, RenderConfig) and render.speaker_color:
            speaker_color = render.speaker_color

        if speaker_color:
            content = cls._to_bytes_with_speaker_color(
                supervisions, speaker_color, **kwargs
            )
        else:
            content = super().to_bytes(supervisions, **kwargs)

        # Splice back raw cue texts (preserves {\an1}{\pos(...)} override tags
        # from subtitle-group SRTs). Only applied when cue count matches (i.e.
        # word-level expansion didn't split supervisions).
        content = cls._splice_raw_cue_texts(content, supervisions)

        # Restore original line terminator if metadata requests CRLF
        if metadata and metadata.get("line_terminator") == "\r\n":
            content = content.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")

        # Add BOM if requested or if original had BOM
        add_bom = use_bom
        if metadata and metadata.get("encoding") == "utf-8-sig":
            add_bom = True

        if add_bom:
            content = b"\xef\xbb\xbf" + content

        return content

    @classmethod
    def _splice_raw_cue_texts(
        cls, content: bytes, supervisions: List[Supervision]
    ) -> bytes:
        """Restore per-cue raw text (with override tags) from supervision.custom.

        pysubs2's SRT writer emits plain text (override tags stripped on read).
        For supervisions that carry ``custom['srt_raw_text']`` (set by the SRT
        reader), replace the pysubs2-emitted text block with the original raw
        text. No-op when cue count mismatches (e.g., word-level expansion).
        """
        raw_texts = [
            (sup.custom or {}).get("srt_raw_text") if sup.custom else None
            for sup in supervisions
        ]
        if not any(raw_texts):
            return content

        text = content.decode("utf-8")
        # pysubs2 emits LF; split cues on blank-line delimiter
        blocks = re.split(r"\n\n+", text)
        # Trailing empty block from trailing blank line
        trailing = "\n\n" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "")

        cue_blocks = [b for b in blocks if b.strip()]
        if len(cue_blocks) != len(supervisions):
            # Word-level expansion or other mismatch — skip splicing
            return content

        out_blocks = []
        for block, raw in zip(cue_blocks, raw_texts):
            if raw is None:
                out_blocks.append(block)
                continue
            lines = block.split("\n")
            # Expect: idx, timestamp, text..., (more text)
            if len(lines) < 3:
                out_blocks.append(block)
                continue
            # Replace everything after timestamp line with raw_text
            new_block = lines[0] + "\n" + lines[1] + "\n" + raw
            out_blocks.append(new_block)

        return ("\n\n".join(out_blocks) + trailing).encode("utf-8")

    @classmethod
    def _to_bytes_with_speaker_color(
        cls,
        supervisions: List[Supervision],
        speaker_color: str,
        **kwargs,
    ) -> bytes:
        """Generate SRT with speaker names wrapped in <font color> tags."""
        from .base import maybe_expand_to_word_supervisions, render_bilingual_text
        from ..colors import resolve_speaker_color_rgb

        behavior, include_speaker, word_level = cls._unpack_render(**kwargs)
        supervisions = maybe_expand_to_word_supervisions(
            supervisions, word_level=word_level, format_id=cls.format_id
        )

        from .base import SpeakerTracker

        color_cache: Dict[str, str] = {}
        subs = pysubs2.SSAFile()
        tf = behavior.translation_first

        tracker = SpeakerTracker() if cls._dedup_speaker else None
        for sup in supervisions:
            text = render_bilingual_text(sup, translation_first=tf)
            if cls._should_include_speaker(sup, include_speaker, tracker) and sup.speaker:
                color = resolve_speaker_color_rgb(
                    sup.speaker, speaker_color, color_cache
                )
                prefix = cls._format_speaker_prefix(sup.speaker)
                if color:
                    text = f'<font color="{color}">{prefix}</font>{text}'
                else:
                    text = f"{prefix}{text}"

            subs.append(
                pysubs2.SSAEvent(
                    start=round(sup.start * 1000),
                    end=round(sup.end * 1000),
                    text=text,
                    name=sup.speaker or "",
                )
            )

        return subs.to_string(format_=cls.pysubs2_format).encode("utf-8")


@register_format("ass")
class ASSFormat(Pysubs2Format):
    """Advanced SubStation Alpha format with karaoke support."""

    extensions = [".ass"]
    pysubs2_format = "ass"
    description = "Advanced SubStation Alpha - rich styling support"

    @classmethod
    def _should_include_speaker(cls, sup, include_speaker, tracker=None):
        """ASS has a dedicated ``Name`` event field that already carries the
        speaker. When the supervision was read from ASS (``ass_raw_text`` is
        present), roundtripping ``include_speaker_in_text=True`` would
        produce a duplicate ``Name: text`` prefix inside the text body.
        Suppress the text-level prefix for ASS-sourced supervisions;
        downstream callers that genuinely want an inline prefix can clear
        ``custom['ass_raw_text']`` before writing.
        """
        custom = getattr(sup, "custom", None) or {}
        if "ass_raw_text" in custom:
            return False
        return super()._should_include_speaker(sup, include_speaker, tracker)

    @classmethod
    def read(
        cls,
        source,
        normalize_text: bool = True,
        **kwargs,
    ) -> List[Supervision]:
        """Read ASS format with style and event metadata preservation.

        Preserves ASS-specific event attributes in Supervision.custom:
        - ass_style: Style name reference
        - ass_layer: Layer number
        - ass_margin_l/r/v: Margin overrides
        - ass_effect: Effect string
        """
        if cls.is_content(source):
            content = source
        else:
            content, _ = detect_file_encoding(Path(source))

        try:
            subs = pysubs2.SSAFile.from_string(content, format_=cls.pysubs2_format)
        except Exception:
            subs = pysubs2.SSAFile.from_string(content)

        # Auto-detect title-case speaker candidates from text content
        all_texts = [e.text for e in subs.events]
        candidates = detect_speaker_candidates(all_texts)
        if candidates:
            set_speaker_candidates(candidates)

        supervisions = []
        for event in subs.events:
            # Use plaintext for sup.text (strips override tags, converts \N to \n)
            plaintext = event.plaintext
            if normalize_text:
                # Light normalization: collapse non-newline whitespace only.
                # Generic normalize_text() collapses \n to space, destroying
                # bilingual line breaks. ASS plaintext is already clean.
                plaintext = re.sub(r"[^\S\n]+", " ", plaintext)
                plaintext = re.sub(r" *\n *", "\n", plaintext)
                plaintext = plaintext.strip()

            speaker, plaintext = parse_speaker_text(plaintext)

            # When parse_speaker_text can't extract speaker from text but
            # the Name field has one, strip the matching prefix from text
            # so roundtripping include_speaker_in_text=True doesn't duplicate.
            if not speaker and event.name:
                for sep in (": ", "： "):
                    prefix = event.name + sep
                    if plaintext.startswith(prefix):
                        plaintext = plaintext[len(prefix) :]
                        break

            # Preserve ASS-specific event attributes + raw text for roundtrip
            custom = {
                "ass_style": event.style,
                "ass_layer": event.layer,
                "ass_margin_l": event.marginl,
                "ass_margin_r": event.marginr,
                "ass_margin_v": event.marginv,
                "ass_effect": event.effect,
                "ass_raw_text": event.text,
                "ass_is_comment": event.is_comment,
            }

            # Detect drawing commands (\p1 in override tags or "m X Y" start)
            if r"\p1" in event.text or re.match(r"^\s*m\s+\d+", event.plaintext):
                custom["line_type"] = "drawing"
            else:
                # Detect staff credits and branding (early short lines)
                evt_start = event.start / 1000.0 if event.start is not None else 0
                lt = classify_line_type(plaintext, start=evt_start)
                if lt:
                    custom["line_type"] = lt

            supervisions.append(
                Supervision(
                    text=plaintext,
                    speaker=speaker or event.name or None,
                    start=event.start / 1000.0 if event.start is not None else 0,
                    duration=(event.end - event.start) / 1000.0
                    if event.end is not None
                    else 0,
                    custom=custom,
                )
            )

        if candidates:
            set_speaker_candidates(set())

        return supervisions

    @classmethod
    def extract_metadata(cls, source, **kwargs) -> Dict:
        """Extract ASS metadata. Deprecated: use parse() instead."""
        return cls.parse(source, normalize_text=False).format_metadata

    @classmethod
    def parse(cls, source, normalize_text: bool = True, **kwargs) -> "ParseResult":
        """Parse ASS in a single pass: supervisions + styles/info metadata.

        Previously read() and extract_metadata() each parsed the file with
        pysubs2 independently. This method parses once and returns both.

        Format metadata returned in ParseResult.format_metadata:
        - ass_info: Script Info section as dict
        - ass_styles: Style definitions as dict of dicts
        - ass_info_raw: Raw [Script Info] lines for roundtrip preservation
        - encoding: Detected file encoding (present when loaded from file)
        """
        from .base import ParseResult

        detected_encoding = None
        try:
            if cls.is_content(source):
                content = source
            else:
                content, detected_encoding = detect_file_encoding(Path(source))
            subs = pysubs2.SSAFile.from_string(content, format_=cls.pysubs2_format)
        except Exception:
            if cls.is_content(source):
                content = source
                subs = pysubs2.SSAFile.from_string(source)
            else:
                # Last-resort fallback when strict format parse fails; re-read
                # bytes with utf-8 so content is still available for raw
                # [Script Info] extraction below.
                subs = pysubs2.load(str(source), encoding="utf-8")
                try:
                    content = Path(source).read_text(encoding="utf-8", errors="replace")
                except Exception:
                    content = ""

        # --- Extract supervisions from events ---
        all_texts = [e.text for e in subs.events]
        candidates = detect_speaker_candidates(all_texts)
        if candidates:
            set_speaker_candidates(candidates)

        # Capture raw Dialogue/Comment line bodies before pysubs2 normalization
        # (see _extract_raw_event_bodies). Pair by position with subs.events.
        raw_event_bodies = cls._extract_raw_event_bodies(content)
        if len(raw_event_bodies) != len(subs.events):
            # Unexpected mismatch → skip splice; pysubs2 output wins
            raw_event_bodies = []

        supervisions = []
        for idx, event in enumerate(subs.events):
            # Use plaintext to keep parity with read(): strips override tags
            # and converts \N to \n so bilingual line breaks survive.
            plaintext = event.plaintext
            if normalize_text:
                plaintext = re.sub(r"[^\S\n]+", " ", plaintext)
                plaintext = re.sub(r" *\n *", "\n", plaintext)
                plaintext = plaintext.strip()

            speaker, plaintext = parse_speaker_text(plaintext)
            if not speaker and event.name:
                for sep in (": ", "： "):
                    prefix = event.name + sep
                    if plaintext.startswith(prefix):
                        plaintext = plaintext[len(prefix) :]
                        break

            custom = {
                "ass_style": event.style,
                "ass_layer": event.layer,
                "ass_margin_l": event.marginl,
                "ass_margin_r": event.marginr,
                "ass_margin_v": event.marginv,
                "ass_effect": event.effect,
                "ass_raw_text": event.text,
                "ass_is_comment": event.is_comment,
            }
            if idx < len(raw_event_bodies):
                custom["ass_raw_event_type"] = raw_event_bodies[idx][0]
                custom["ass_raw_event_body"] = raw_event_bodies[idx][1]

            if r"\p1" in event.text or re.match(r"^\s*m\s+\d+", event.plaintext):
                custom["line_type"] = "drawing"
            else:
                evt_start = event.start / 1000.0 if event.start is not None else 0
                lt = classify_line_type(plaintext, start=evt_start)
                if lt:
                    custom["line_type"] = lt

            supervisions.append(
                Supervision(
                    text=plaintext,
                    speaker=speaker or event.name or None,
                    start=event.start / 1000.0 if event.start is not None else 0,
                    duration=(event.end - event.start) / 1000.0
                    if event.end is not None
                    else 0,
                    custom=custom,
                )
            )

        if candidates:
            set_speaker_candidates(set())

        # Extract raw section bodies for byte-faithful roundtrip preservation.
        # pysubs2 re-serializes Script Info / Styles / Events headers and drops:
        #   - non-standard comment lines in Script Info
        #   - trailing ``.0`` on Style floats (100.0 → 100)
        #   - original Format line ordering in [Events]
        # Capturing the raw text lets writers splice it back in verbatim.
        raw_sections = cls._extract_raw_sections(content)
        ass_info_raw = raw_sections.get("script_info", "")
        ass_styles_raw = raw_sections.get("styles", "")
        ass_events_format_raw = raw_sections.get("events_format", "")

        # --- Extract styles/info metadata from the SAME subs object ---
        styles_dict = {}
        for name, style in subs.styles.items():
            styles_dict[name] = {
                "fontname": style.fontname,
                "fontsize": style.fontsize,
                "primarycolor": cls._color_to_str(style.primarycolor),
                "secondarycolor": cls._color_to_str(style.secondarycolor),
                "tertiarycolor": cls._color_to_str(style.tertiarycolor),
                "outlinecolor": cls._color_to_str(style.outlinecolor),
                "backcolor": cls._color_to_str(style.backcolor),
                "bold": style.bold,
                "italic": style.italic,
                "underline": style.underline,
                "strikeout": style.strikeout,
                "scalex": style.scalex,
                "scaley": style.scaley,
                "spacing": style.spacing,
                "angle": style.angle,
                "borderstyle": style.borderstyle,
                "outline": style.outline,
                "shadow": style.shadow,
                "alignment": style.alignment,
                "marginl": style.marginl,
                "marginr": style.marginr,
                "marginv": style.marginv,
                "alphalevel": style.alphalevel,
                "encoding": style.encoding,
            }

        format_metadata = {
            "ass_info": dict(subs.info),
            "ass_styles": styles_dict,
        }
        if ass_info_raw:
            format_metadata["ass_info_raw"] = ass_info_raw
        if ass_styles_raw:
            format_metadata["ass_styles_raw"] = ass_styles_raw
        if ass_events_format_raw:
            format_metadata["ass_events_format_raw"] = ass_events_format_raw
        if detected_encoding:
            format_metadata["encoding"] = detected_encoding

        # Record the exact trailing newline count so the writer matches the
        # source file byte-for-byte. Real-world ASS files land in three states:
        #   * 0 trailing newlines — no final EOL at all (some hand-authored files)
        #   * 1 trailing newline  — standard "line ends with EOL"
        #   * 2+ trailing newlines — Aegisub habit: extra blank line after last event
        # pysubs2.to_string always emits exactly 1 trailing ``\n``; capturing
        # the source count lets us restore either extreme.
        if content:
            tail = content.replace("\r\n", "\n")
            trailing_nl = len(tail) - len(tail.rstrip("\n"))
            format_metadata["trailing_newlines"] = trailing_nl

        return ParseResult(
            supervisions=supervisions,
            format_metadata=format_metadata,
        )

    # Matches raw Dialogue/Comment lines in source order. Group 1 = event type
    # ("Dialogue" or "Comment"), group 2 = the body (everything after the
    # leading "Dialogue: " / "Comment: ", preserving trailing whitespace). CR
    # is excluded from the body so CRLF sources still pair cleanly.
    _ASS_EVENT_LINE_RE = re.compile(
        r"^(Dialogue|Comment):\s?([^\r\n]*)",
        re.MULTILINE,
    )

    @classmethod
    def _extract_raw_event_bodies(cls, content: str) -> List[Tuple[str, str]]:
        """Extract raw ``(event_type, body)`` pairs for every Dialogue/Comment.

        pysubs2 normalizes several Dialogue-line artefacts that subtitle-group
        files actually contain:
          * zero-padded margin fields (``0000,0000,0000`` → ``0,0,0``)
          * trailing whitespace inside the Text field
        The captured raw body preserves both. Writer splices them back,
        updating only Start/End fields from the mutated supervision.
        """
        return [
            (m.group(1), m.group(2))
            for m in cls._ASS_EVENT_LINE_RE.finditer(content)
        ]

    @staticmethod
    def _extract_raw_sections(content: str) -> Dict[str, str]:
        """Extract raw body text for each ASS section for roundtrip fidelity.

        Returns a dict with up to three keys:
          - ``script_info``: full ``[Script Info]`` block (header + body)
          - ``styles``: full ``[V4+ Styles]`` or ``[V4 Styles]`` block
          - ``events_format``: only the ``Format:`` line from ``[Events]``
            (Dialogue/Comment lines are NOT captured — they will be
            re-serialized by the writer after supervisions are mutated)

        Section bodies preserve each line's trailing ``\\r`` when the source
        uses CRLF, so callers can splice them back verbatim.
        """
        # Split on ``\n`` only so CRLF sources keep ``\r`` at EOL. This is
        # intentional: preserving the carriage return lets roundtrip output
        # match byte-for-byte.
        lines = content.split("\n")

        out: Dict[str, str] = {}
        current: Optional[str] = None
        buf: list = []

        def flush() -> None:
            if current and buf:
                out[current] = "\n".join(buf)
            buf.clear()

        for line in lines:
            stripped = line.strip()
            low = stripped.lower()
            # Section boundaries
            if low == "[script info]":
                flush()
                current = "script_info"
                buf.append(line)
                continue
            if low in ("[v4+ styles]", "[v4 styles]"):
                flush()
                current = "styles"
                buf.append(line)
                continue
            if low == "[events]":
                flush()
                current = "events_header"  # collect only until Format: line
                buf.append(line)
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                # Unknown section — stop capturing
                flush()
                current = None
                continue

            if current == "events_header":
                # In [Events], only capture up to and including the Format: line.
                # Dialogue/Comment will be re-serialized, so stop after Format.
                buf.append(line)
                if stripped.lower().startswith("format:"):
                    out["events_format"] = "\n".join(buf)
                    buf.clear()
                    current = None
                continue

            if current:
                buf.append(line)

        flush()
        return out

    @staticmethod
    def _extract_raw_script_info(content: str) -> str:
        """Backward-compatible wrapper: delegate to _extract_raw_sections."""
        return Pysubs2Format._extract_raw_sections(content).get("script_info", "")

    @staticmethod
    def _color_to_str(color: pysubs2.Color) -> str:
        """Convert pysubs2.Color to ASS color string &HAABBGGRR."""
        return f"&H{color.a:02X}{color.b:02X}{color.g:02X}{color.r:02X}"

    @staticmethod
    def _str_to_color(color_str: str) -> pysubs2.Color:
        """Convert ASS color string &HAABBGGRR to pysubs2.Color."""
        color_str = color_str.lstrip("&H").lstrip("&h")
        if len(color_str) == 8:
            a = int(color_str[0:2], 16)
            b = int(color_str[2:4], 16)
            g = int(color_str[4:6], 16)
            r = int(color_str[6:8], 16)
        elif len(color_str) == 6:
            a = 0
            b = int(color_str[0:2], 16)
            g = int(color_str[2:4], 16)
            r = int(color_str[4:6], 16)
        else:
            return pysubs2.Color(r=255, g=255, b=255, a=0)
        return pysubs2.Color(r=r, g=g, b=b, a=a)

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        fps: float = 25.0,
        metadata: Optional[Dict] = None,
        config: Optional[ASSConfig] = None,
        **kwargs,
    ) -> bytes:
        """Convert to ASS bytes with style preservation and optional karaoke tags.

        Args:
            supervisions: List of supervision segments
            fps: Frames per second (not used for ASS)
            metadata: Optional metadata dict containing ass_info and ass_styles
                to restore original ASS formatting
            render: RenderConfig controlling output behavior (via **kwargs)
            config: ASSConfig controlling ASS-specific rendering context and style

        Returns:
            ASS content as bytes
        """
        import logging

        from .base import count_supervisions_with_words, maybe_expand_to_word_supervisions

        behavior, include_speaker, word_level = cls._unpack_render(**kwargs)
        config = config if isinstance(config, ASSConfig) else ASSConfig()

        karaoke_effect = config.karaoke_effect

        # Tri-state semantics for ASS:
        #   word_level=False  → force segment; karaoke_effect is disabled with a warning.
        #   word_level=True   → force word level. With karaoke → emit \k tags;
        #                       without karaoke → expand to one Dialogue per word.
        #   word_level=None   → karaoke_effect existence implies word level;
        #                       otherwise stay segment-level.
        _ass_logger = logging.getLogger("lattifai.caption")

        if word_level is False and karaoke_effect is not None:
            _ass_logger.warning(
                "ass: word_level=False overrides karaoke_effect=%s; karaoke disabled",
                karaoke_effect,
            )
            karaoke_effect = None

        if word_level is True and not karaoke_effect:
            supervisions = maybe_expand_to_word_supervisions(
                supervisions, word_level=True, format_id="ass"
            )

        # Karaoke + missing data: warn explicitly. The inner loop falls back
        # per-supervision, but without this guard the user gets silent
        # segment output instead of the karaoke they requested.
        if karaoke_effect and (word_level is True or word_level is None):
            n_total = len(supervisions)
            n_with = count_supervisions_with_words(supervisions)
            if n_with == 0:
                _ass_logger.warning(
                    "ass: karaoke_effect=%s requested but no word alignment available; "
                    "falling back to segment-level output (no \\k tags)",
                    karaoke_effect,
                )
            elif n_with < n_total:
                _ass_logger.warning(
                    "ass: karaoke_effect=%s but %d/%d supervisions lack word alignment; "
                    "those segments output without \\k tags",
                    karaoke_effect,
                    n_total - n_with,
                    n_total,
                )

        # Create ASS file from config + metadata
        subs = cls._create_ass_file(metadata, config)

        # Priority: ASSConfig.speaker_color > RenderConfig.speaker_color
        speaker_color = config.speaker_color
        if not speaker_color and behavior.speaker_color:
            speaker_color = behavior.speaker_color

        # Add karaoke style
        has_metadata_styles = metadata and "ass_styles" in metadata
        if karaoke_effect and "Karaoke" not in subs.styles:
            if has_metadata_styles and "Default" in metadata["ass_styles"]:
                subs.styles["Karaoke"] = subs.styles["Default"].copy()
            else:
                subs.styles["Karaoke"] = cls._build_ass_style(config)

        # Apply style to Default (when no metadata styles override)
        has_custom_default = has_metadata_styles and "Default" in (
            metadata.get("ass_styles") or {}
        )
        if not has_custom_default and "Default" in subs.styles:
            subs.styles["Default"] = cls._build_ass_style(config)

        from .base import SpeakerTracker

        # Speaker color cache: maps speaker name → BBGGRR color string (assigned on first appearance)
        _speaker_color_cache = {}

        tracker = SpeakerTracker()
        for sup in supervisions:
            alignment = getattr(sup, "alignment", None)
            word_items = alignment.get("word") if alignment else None

            # Karaoke mode: presence of karaoke_effect implies word-level rendering.
            # word_level no longer gates this branch — it has been resolved above
            # into karaoke_effect (False sets it to None) or expansion.
            if karaoke_effect and word_items:
                karaoke_text = cls._build_karaoke_text(
                    word_items,
                    karaoke_effect,
                    original_text=sup.text,
                    kinetic_style=config.kinetic_style,
                    scaley=config.scaley,
                    angle=config.angle,
                )
                karaoke_text = karaoke_text.replace("\n", "\\N")
                if cls._should_include_speaker(sup, include_speaker, tracker):
                    prefix = cls._format_speaker_prefix(sup.speaker)
                    spk_color = cls._resolve_speaker_color(
                        sup.speaker, speaker_color, _speaker_color_cache
                    )
                    if spk_color:
                        karaoke_text = (
                            f"{{\\c&H{spk_color}&}}{prefix}{{\\c}}{karaoke_text}"
                        )
                    else:
                        karaoke_text = f"{prefix}{karaoke_text}"
                # Bilingual karaoke: append the translation as a second plain
                # line below (or above) the karaoke target line. Without this
                # branch, translation would be silently dropped — only the
                # non-karaoke path called render_bilingual_text.
                #
                # COLOR JUMP FIX: \N is just a forced line break — it does
                # NOT reset tag state. A \k tag applies to all following text
                # until the next karaoke tag, \r, or end of line. So plain
                # text after the last {\kf...} is still inside that karaoke
                # run, and its color tracks the \k tween between
                # SecondaryColour → PrimaryColour, then snaps when the sweep
                # ends. The fix is two-pronged:
                #   1) {\rKaraoke}      cancel active karaoke state, reset
                #                       to the same Karaoke style
                #   2) {\1c&Hxxxx&}{\2c&Hxxxx&}  lock both fill colors to
                #                       the same value so any residual tween
                #                       is invisible.
                # Default color is Karaoke style PrimaryColour (the "sung/end"
                # state) — eye reads the translation as the visual rest
                # position next to the karaoke's destination color.
                if sup.translation:
                    trans_bgr = cls._resolve_translation_bgr(config)
                    trans_body = sup.translation.replace("\n", "\\N")
                    trans_span = (
                        f"{{\\rKaraoke\\1c&H{trans_bgr}&\\2c&H{trans_bgr}&}}{trans_body}"
                    )
                    if behavior.translation_first:
                        karaoke_text = f"{trans_span}\\N{karaoke_text}"
                    else:
                        karaoke_text = f"{karaoke_text}\\N{trans_span}"
                event_start = round(word_items[0].start * 1000)
                event_end = round(word_items[-1].end * 1000)

                event = pysubs2.SSAEvent(
                    start=event_start,
                    end=event_end,
                    text=karaoke_text,
                    style="Karaoke",
                )
                if sup.speaker:
                    event.name = sup.speaker
                subs.append(event)
            else:
                # Standard mode (no per-word \k tags). If kinetic_style is set,
                # apply line-scope kinetic as a single {override} at text start.
                from .base import render_bilingual_text

                text = render_bilingual_text(
                    sup, separator="\\N", translation_first=behavior.translation_first
                )
                if cls._should_include_speaker(sup, include_speaker, tracker):
                    prefix = cls._format_speaker_prefix(sup.speaker)
                    spk_color = cls._resolve_speaker_color(
                        sup.speaker, speaker_color, _speaker_color_cache
                    )
                    if spk_color:
                        text = f"{{\\c&H{spk_color}&}}{prefix}{{\\c}}{text}"
                    else:
                        text = f"{prefix}{text}"

                if config.kinetic_style is not None:
                    from ..kinetic import build_line_override, rebase_kinetic_impl, resolve_kinetic

                    resolved = resolve_kinetic(config.kinetic_style, word_level=False)
                    if resolved is not None:
                        scope, impl = resolved
                        impl = rebase_kinetic_impl(impl, scaley=config.scaley, angle=config.angle)
                        # resolve_kinetic with word_level=False only returns
                        # scope="line" or raises — so we are safe to prepend.
                        override = build_line_override(impl)
                        if override:
                            text = f"{{{override}}}{text}"

                event = cls._create_event_from_supervision(sup, text)
                subs.append(event)

        output = subs.to_string(format_="ass")

        # Restore original section bodies for byte-faithful roundtrip.
        # Order matters: splice sections first, then normalize line endings
        # and BOM as a post-pass so every substituted region ends up with the
        # source file's terminator style.
        if metadata:
            if metadata.get("ass_info_raw"):
                output = cls._replace_script_info(output, metadata["ass_info_raw"])
            if metadata.get("ass_styles_raw"):
                output = cls._replace_styles_section(output, metadata["ass_styles_raw"])
            if metadata.get("ass_events_format_raw"):
                output = cls._replace_events_format(output, metadata["ass_events_format_raw"])

        # Splice raw Dialogue/Comment bodies (preserves margin padding and
        # trailing whitespace inside the Text field). pysubs2 normalization:
        #   * ``0000,0000,0000`` → ``0,0,0``
        #   * ``"hello "`` → ``"hello"`` (event.text trimmed on parse)
        # Writer takes the raw body captured at read time and only rewrites
        # the Start/End fields from the mutated supervision.
        output = cls._splice_raw_event_bodies(output, supervisions)

        # Normalize to LF internally, then apply source terminator uniformly.
        # Raw section bodies may already contain CR (when source was CRLF);
        # strip them first so we never emit "\r\r\n" downstream.
        output = output.replace("\r\n", "\n")

        # Restore exact trailing newline count (0 / 1 / N) from source.
        # Aegisub often appends an extra blank after the last event (N=2+);
        # hand-authored files sometimes omit the final EOL entirely (N=0).
        trailing_newlines = (metadata or {}).get("trailing_newlines")
        if trailing_newlines is not None:
            output = output.rstrip("\n") + "\n" * trailing_newlines

        line_terminator = (metadata or {}).get("line_terminator", "\n")
        if line_terminator == "\r\n":
            output = output.replace("\n", "\r\n")

        raw_bytes = output.encode("utf-8")

        # Re-emit UTF-8 BOM when source file had one (``encoding='utf-8-sig'``).
        # Other encodings are handled by the caller when writing to disk.
        if (metadata or {}).get("encoding") == "utf-8-sig":
            raw_bytes = b"\xef\xbb\xbf" + raw_bytes

        return raw_bytes

    @staticmethod
    def _splice_raw_event_bodies(
        output: str, supervisions: List[Supervision]
    ) -> str:
        """Replace each pysubs2-emitted Dialogue/Comment body with its raw copy.

        Only the Start/End fields (positions 1 and 2) from the pysubs2 output
        are kept — those reflect current supervision timings. Everything else
        (Layer, Style, Name, MarginL/R/V, Effect, Text) uses the raw body so
        zero-padded margins and trailing whitespace inside Text survive.

        Skipped when:
          * no supervision has ``ass_raw_event_body`` (nothing to splice)
          * output Dialogue/Comment count != supervision count (word-level
            expansion or other reflow happened)
        """
        raw_pairs = [
            (
                (sup.custom or {}).get("ass_raw_event_type"),
                (sup.custom or {}).get("ass_raw_event_body"),
            )
            for sup in supervisions
        ]
        if not any(body for _, body in raw_pairs):
            return output

        lines = output.split("\n")
        event_line_indices = [
            i for i, ln in enumerate(lines)
            if ln.startswith("Dialogue:") or ln.startswith("Comment:")
        ]
        if len(event_line_indices) != len(supervisions):
            # Mismatch — bail out rather than corrupt alignment
            return output

        for idx, line_idx in enumerate(event_line_indices):
            raw_type, raw_body = raw_pairs[idx]
            if raw_body is None:
                continue
            line = lines[line_idx]
            # Split off the prefix ("Dialogue: " / "Comment: ")
            prefix, _, pysubs_body = line.partition(": ")
            if not pysubs_body:
                continue
            pysubs_fields = pysubs_body.split(",", 9)
            raw_fields = raw_body.split(",", 9)
            if len(pysubs_fields) < 10 or len(raw_fields) < 10:
                continue
            # Preserve pysubs2's Start/End (reflects current sup timing);
            # everything else comes from the raw body.
            raw_fields[1] = pysubs_fields[1]
            raw_fields[2] = pysubs_fields[2]
            final_prefix = raw_type or prefix
            lines[line_idx] = f"{final_prefix}: " + ",".join(raw_fields)

        return "\n".join(lines)

    DEFAULT_FONTSIZE = 64.0

    @staticmethod
    def _replace_script_info(ass_output: str, raw_info: str) -> str:
        """Replace pysubs2-generated [Script Info] with original raw text."""
        lines = ass_output.split("\n")
        result = []
        in_section = False
        replaced = False
        for line in lines:
            stripped = line.strip()
            if stripped.lower() == "[script info]":
                in_section = True
                result.append(raw_info)
                replaced = True
                continue
            if in_section:
                if stripped.startswith("[") and stripped.endswith("]"):
                    in_section = False
                    result.append(line)
                # Skip pysubs2-generated Script Info lines (replaced by raw)
                continue
            result.append(line)
        return "\n".join(result) if replaced else ass_output

    @staticmethod
    def _replace_styles_section(ass_output: str, raw_styles: str) -> str:
        """Replace pysubs2-generated [V4+ Styles] block with the original raw text.

        Skips every pysubs2-emitted line from ``[V4+ Styles]`` (or ``[V4 Styles]``)
        up to the next section header, substituting the captured raw block.
        The raw block already carries its own section header.

        If pysubs2 emits additional styles not present in the raw block (e.g.
        a newly-added ``Karaoke`` style via ``ASSConfig.karaoke_effect``), those
        lines are appended after the raw block to avoid silent loss.
        """
        lines = ass_output.split("\n")
        result = []
        in_section = False
        replaced = False
        extra_styles = []
        raw_style_names = set()
        # Parse style names already present in raw_styles to detect extras.
        for ln in raw_styles.split("\n"):
            s = ln.strip()
            if s.lower().startswith("style:"):
                # Style: Name,...
                name = s[len("Style:") :].split(",", 1)[0].strip()
                if name:
                    raw_style_names.add(name)

        for line in lines:
            stripped = line.strip()
            low = stripped.lower()
            if low in ("[v4+ styles]", "[v4 styles]"):
                in_section = True
                result.append(raw_styles)
                replaced = True
                continue
            if in_section:
                if stripped.startswith("[") and stripped.endswith("]"):
                    in_section = False
                    # Flush any extra styles before the next section header.
                    for es in extra_styles:
                        result.append(es)
                    extra_styles.clear()
                    result.append(line)
                    continue
                # Inside the replaced block: skip pysubs2-emitted Format/Style
                # lines, but capture any Style whose name is NOT in raw so we
                # can append it (e.g. pysubs2 added "Karaoke" at render time).
                if low.startswith("style:"):
                    name = stripped[len("Style:") :].split(",", 1)[0].strip()
                    if name and name not in raw_style_names:
                        extra_styles.append(line)
                continue
            result.append(line)

        # If the file ended while still inside the replaced section.
        if in_section:
            for es in extra_styles:
                result.append(es)

        return "\n".join(result) if replaced else ass_output

    @staticmethod
    def _replace_events_format(ass_output: str, raw_events_format: str) -> str:
        """Replace pysubs2-generated [Events] header + Format line with raw.

        Only touches the section header and its Format line — Dialogue /
        Comment lines are left in place (they reflect mutated supervisions).
        """
        lines = ass_output.split("\n")
        result = []
        in_section = False
        replaced = False
        header_emitted = False
        for line in lines:
            stripped = line.strip()
            low = stripped.lower()
            if low == "[events]":
                in_section = True
                result.append(raw_events_format)
                replaced = True
                header_emitted = True
                continue
            if in_section and not header_emitted:
                # Already emitted via raw block above; defensive branch.
                header_emitted = True
                continue
            if in_section and low.startswith("format:"):
                # Skip pysubs2's Format line — already in raw.
                continue
            result.append(line)
        return "\n".join(result) if replaced else ass_output

    @classmethod
    def _create_ass_file(
        cls, metadata: Optional[Dict], config: ASSConfig
    ) -> pysubs2.SSAFile:
        """Create SSAFile from ASSConfig defaults, then overlay metadata (roundtrip).

        Priority: metadata (roundtrip from existing ASS) > ASSConfig > pysubs2 defaults.

        Args:
            metadata: Dict containing ass_info, ass_styles for roundtrip preservation
            config: ASSConfig with rendering context (PlayRes, wrap_style, etc.)

        Returns:
            pysubs2.SSAFile with configured styles
        """
        subs = pysubs2.SSAFile()

        # 1. Apply ASSConfig rendering context
        subs.info["PlayResX"] = str(config.play_res_x)
        subs.info["PlayResY"] = str(config.play_res_y)
        subs.info["ScaledBorderAndShadow"] = (
            "yes" if config.scaled_border_and_shadow else "no"
        )
        subs.info["WrapStyle"] = str(config.wrap_style)
        subs.styles["Default"].fontsize = cls.DEFAULT_FONTSIZE

        if not metadata:
            return subs

        # 2. Metadata roundtrip overrides (from reading an existing ASS file)
        if "ass_info" in metadata:
            subs.info.update(metadata["ass_info"])

        if "ass_styles" in metadata:
            for name, style_dict in metadata["ass_styles"].items():
                subs.styles[name] = cls._dict_to_style(style_dict)

        return subs

    @classmethod
    def _dict_to_style(cls, style_dict: Dict) -> pysubs2.SSAStyle:
        """Convert style dict back to pysubs2.SSAStyle."""
        return pysubs2.SSAStyle(
            fontname=style_dict.get("fontname", "Arial"),
            fontsize=style_dict.get("fontsize", cls.DEFAULT_FONTSIZE),
            primarycolor=cls._str_to_color(
                style_dict.get("primarycolor", "&H00FFFFFF")
            ),
            secondarycolor=cls._str_to_color(
                style_dict.get("secondarycolor", "&H000000FF")
            ),
            tertiarycolor=cls._str_to_color(
                style_dict.get("tertiarycolor", "&H00000000")
            ),
            outlinecolor=cls._str_to_color(
                style_dict.get("outlinecolor", "&H00000000")
            ),
            backcolor=cls._str_to_color(style_dict.get("backcolor", "&H00000000")),
            bold=style_dict.get("bold", False),
            italic=style_dict.get("italic", False),
            underline=style_dict.get("underline", False),
            strikeout=style_dict.get("strikeout", False),
            scalex=style_dict.get("scalex", 100.0),
            scaley=style_dict.get("scaley", 100.0),
            spacing=style_dict.get("spacing", 0.0),
            angle=style_dict.get("angle", 0.0),
            borderstyle=style_dict.get("borderstyle", 1),
            outline=style_dict.get("outline", 2.0),
            shadow=style_dict.get("shadow", 2.0),
            alignment=pysubs2.Alignment(style_dict.get("alignment", 2)),
            marginl=style_dict.get("marginl", 10),
            marginr=style_dict.get("marginr", 10),
            marginv=style_dict.get("marginv", 10),
            alphalevel=style_dict.get("alphalevel", 0),
            encoding=style_dict.get("encoding", 1),
        )

    @classmethod
    def _create_event_from_supervision(
        cls, sup: Supervision, text: str
    ) -> pysubs2.SSAEvent:
        """Create SSAEvent from Supervision, restoring custom attributes.

        If ass_raw_text exists and the text hasn't been modified (matches the
        plaintext derived from ass_raw_text), the original ASS text with
        override tags is restored for roundtrip fidelity.

        Args:
            sup: Supervision with optional custom dict containing ass_* attributes
            text: Processed text content

        Returns:
            pysubs2.SSAEvent with restored attributes
        """
        custom = getattr(sup, "custom", None) or {}

        # Roundtrip: restore original ASS text with override tags if unchanged
        ass_raw = custom.get("ass_raw_text")
        if ass_raw:
            # Derive plaintext from raw to compare with current text.
            # pysubs2 uses \N for line breaks; plaintext converts to \n.
            # Collapse runs of inner whitespace on BOTH sides so that the
            # reader's ``normalize_text`` pass (which folds "  " into " ")
            # does not block the raw-text restoration path. Without this,
            # YYeTs-style double-spaced dialogue would lose the original
            # spacing on every roundtrip.
            def _norm(s: str) -> str:
                # Mirror the ASS reader's ``normalize_text`` pipeline so
                # roundtrip comparison ignores cosmetic whitespace drift
                # that the reader itself introduced:
                #   1) drop override tag blocks
                #   2) treat \N as a logical newline
                #   3) collapse runs of non-newline whitespace to one space
                #   4) strip whitespace around newlines
                #   5) strip start/end whitespace
                s = re.sub(r"\{[^}]*\}", "", s)
                s = s.replace("\\N", "\n")
                s = re.sub(r"[^\S\n]+", " ", s)
                s = re.sub(r" *\n *", "\n", s)
                return s.strip()

            raw_plain = _norm(ass_raw)
            current_plain = _norm(text)
            if raw_plain == current_plain:
                text = ass_raw
            else:
                # Text was modified: try to preserve tag prefix from original
                tag_prefix = cls._extract_ass_tag_prefix(ass_raw)
                text = text.replace("\n", "\\N")
                if tag_prefix:
                    text = tag_prefix + text
        else:
            # No raw text: just convert \n to \N for ASS format
            text = text.replace("\n", "\\N")

        event = pysubs2.SSAEvent(
            start=round(sup.start * 1000),
            end=round(sup.end * 1000),
            text=text,
            name=sup.speaker or "",
            style=custom.get("ass_style", "Default"),
            layer=custom.get("ass_layer", 0),
            marginl=custom.get("ass_margin_l", 0),
            marginr=custom.get("ass_margin_r", 0),
            marginv=custom.get("ass_margin_v", 0),
            effect=custom.get("ass_effect", ""),
        )
        # Restore Comment type for roundtrip
        if custom.get("ass_is_comment"):
            event.is_comment = True
        return event

    @staticmethod
    def _extract_ass_tag_prefix(raw_text: str) -> str:
        """Extract leading ASS override tag block from raw text.

        Example: '{\\an8\\fad(0,500)}Hello' -> '{\\an8\\fad(0,500)}'
        """
        match = re.match(r"^(\{[^}]*\})+", raw_text)
        return match.group(0) if match else ""

    @classmethod
    def _build_ass_style(cls, config: ASSConfig) -> pysubs2.SSAStyle:
        """Build pysubs2 SSAStyle from self-contained ASSConfig.

        Args:
            config: ASSConfig with all visual, positioning, and rendering fields

        Returns:
            pysubs2.SSAStyle object
        """
        alignment = pysubs2.Alignment(config.alignment)

        # Resolve borderstyle: explicit config value takes precedence,
        # otherwise auto-derive from background_color.
        #
        # ASS borderstyle=3 (opaque box) semantics:
        #   OutlineColour = box FILL color
        #   BackColour    = box shadow color
        #   Outline       = box padding (libass won't render box if 0)
        _BOX_PADDING_DEFAULT = 4

        has_bg = bool(config.background_color)

        if config.borderstyle is not None:
            # Explicit borderstyle — user controls low-level ASS behavior
            borderstyle = config.borderstyle
        else:
            # Auto-derive: borderstyle=3 when background_color is set
            borderstyle = 3 if has_bg else 1

        is_box_mode = borderstyle == 3

        if is_box_mode and has_bg:
            # High-level path: background_color drives box fill
            outline_clr = cls._hex_to_ass_color(config.background_color)
        else:
            # Low-level path: outline_color maps directly to OutlineColour
            outline_clr = cls._hex_to_ass_color(config.outline_color)

        back = cls._hex_to_ass_color(config.back_color)

        if is_box_mode:
            shadow = 0
            outline = config.outline_width if config.outline_width > 0 else _BOX_PADDING_DEFAULT
        else:
            shadow = config.shadow_depth
            outline = config.outline_width

        return pysubs2.SSAStyle(
            fontname=config.font_name,
            fontsize=config.font_size,
            primarycolor=cls._hex_to_ass_color(config.primary_color),
            secondarycolor=cls._hex_to_ass_color(config.secondary_color),
            outlinecolor=outline_clr,
            backcolor=back,
            bold=config.bold,
            italic=config.italic,
            underline=config.underline,
            strikeout=config.strikeout,
            scalex=config.scalex,
            scaley=config.scaley,
            spacing=config.spacing,
            angle=config.angle,
            outline=outline,
            shadow=shadow,
            borderstyle=borderstyle,
            alignment=alignment,
            marginl=config.margin_l,
            marginr=config.margin_r,
            marginv=config.margin_v,
        )

    @staticmethod
    def _hex_to_ass_color(hex_color: str) -> pysubs2.Color:
        """Convert #RRGGBB or #RRGGBBAA to pysubs2 Color.

        ASS uses &HAABBGGRR format (reversed RGB with INVERTED alpha).
        Standard hex: AA=FF means opaque, 00 means transparent.
        ASS alpha:    00 means opaque, FF means transparent (inverted).

        Args:
            hex_color: Color in #RRGGBB or #RRGGBBAA format

        Returns:
            pysubs2.Color object
        """
        hex_color = hex_color.lstrip("#")

        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        # Parse alpha if present (#RRGGBBAA), then invert for ASS
        if len(hex_color) >= 8:
            standard_alpha = int(hex_color[6:8], 16)  # FF=opaque, 00=transparent
            ass_alpha = 255 - standard_alpha  # 00=opaque, FF=transparent
        else:
            ass_alpha = 0  # Fully opaque (no alpha = opaque in ASS)

        return pysubs2.Color(r=r, g=g, b=b, a=ass_alpha)

    @staticmethod
    def _build_karaoke_text(
        words: list,
        effect: str = "sweep",
        original_text: str = "",
        kinetic_style: Optional[str] = None,
        scaley: float = 100.0,
        angle: float = 0.0,
    ) -> str:
        """Build karaoke tag text with gap-aware timing and optional kinetic motion.

        Each word's karaoke duration covers from its start to the next word's start,
        so silence gaps between words are absorbed into the preceding word's highlight.
        Separators between words are derived from the original text to preserve
        correct spacing for all languages (CJK, Latin, mixed).

        Kinetic scope is resolved for word_level=True here (callers gate on
        word_level). If the preset is word-scope, each word gets \\t() overrides
        offset to its cumulative activation time. If the preset is line-only
        (e.g. rise), a single line-scope override is prepended to the text so
        the whole block animates while \\k sweep still runs per word. Stagger is
        handled via expand_stagger_word().

        Args:
            words: List of AlignmentItem objects (must have start, duration, symbol)
            effect: Karaoke effect type ("sweep", "instant", "outline")
            original_text: Original supervision text — used to derive word separators
            kinetic_style: Optional kinetic motion preset (see kinetic.py)

        Returns:
            Text with karaoke + kinetic tags.
        """
        from ..kinetic import (
            build_line_override,
            build_word_overrides,
            expand_stagger_word,
            is_char_level_style,
            rebase_kinetic_impl,
            resolve_kinetic,
        )

        tag_map = {"sweep": "kf", "instant": "k", "outline": "ko"}
        tag = tag_map.get(effect, "kf")

        # Build separator map from original text by finding each word's position
        separators = [""] * len(words)
        if original_text:
            search_from = 0
            for i, word in enumerate(words):
                pos = original_text.find(word.symbol, search_from)
                if pos >= 0 and i > 0:
                    # Separator = characters between previous word end and this word start
                    separators[i] = original_text[search_from:pos]
                if pos >= 0:
                    search_from = pos + len(word.symbol)
        else:
            # Fallback: space-separated (Latin default)
            for i in range(1, len(words)):
                separators[i] = " "

        # Resolve kinetic scope. word_level is hardcoded True here because the
        # caller gated on `karaoke_effect and word_items` — reaching this branch
        # already means we have per-word data and we are in the karaoke
        # renderer path, so word-scope kinetic is the only valid choice.
        line_prefix = ""
        word_impl = None
        char_level = False
        resolved = resolve_kinetic(kinetic_style, word_level=True)
        if resolved is not None:
            scope, impl = resolved
            impl = rebase_kinetic_impl(impl, scaley=scaley, angle=angle)
            if scope == "line":
                override = build_line_override(impl)
                if override:
                    line_prefix = "{" + override + "}"
            else:  # scope == "word"
                word_impl = impl
                char_level = is_char_level_style(kinetic_style)

        parts: List[str] = []
        if line_prefix:
            parts.append(line_prefix)

        cumulative_ms = 0
        for i, word in enumerate(words):
            # Duration: span from this word to next word (absorbs silence gaps)
            if i < len(words) - 1:
                duration = words[i + 1].start - word.start
            else:
                duration = word.duration
            centiseconds = max(1, int(duration * 100))

            word_start_ms = cumulative_ms
            cumulative_ms += centiseconds * 10

            if char_level:
                # Stagger — per-char alpha reveal replaces the word body.
                body = expand_stagger_word(word.symbol, word_start_ms)
                parts.append(f"{separators[i]}{{\\{tag}{centiseconds}}}{body}")
            elif word_impl is not None:
                # \rKaraoke resets all inherited \t() animations from the
                # previous word, preventing animation "bleeding" in libass.
                # Must come before \kf and kinetic tags so the word starts
                # from a clean Karaoke style baseline.
                kinetic_override = build_word_overrides(word_impl, word_start_ms)
                parts.append(
                    f"{separators[i]}{{\\rKaraoke\\{tag}{centiseconds}{kinetic_override}}}{word.symbol}"
                )
            else:
                parts.append(f"{separators[i]}{{\\{tag}{centiseconds}}}{word.symbol}")

        return "".join(parts)

    # Backward-compatible alias — canonical source is colors.SPEAKER_PALETTE
    _SPEAKER_PALETTE = SPEAKER_PALETTE

    @classmethod
    def _resolve_speaker_color(
        cls, speaker: str, speaker_color_spec: str, cache: dict
    ) -> str:
        """Resolve ASS BBGGRR color string for a speaker.

        Delegates to colors.resolve_speaker_color() — kept as thin wrapper
        for backward compatibility with tests referencing ASSFormat._resolve_speaker_color.
        """
        return resolve_speaker_color(speaker, speaker_color_spec, cache)

    @staticmethod
    def _resolve_translation_bgr(config: ASSConfig) -> str:
        """Resolve the translation line's locked color to ASS BBGGRR hex.

        Accepts ``config.translation_color`` as one of:
            None | "primary"   → config.primary_color   (Karaoke "sung" end)
            "secondary"        → config.secondary_color (Karaoke "unsung" start)
            "#RRGGBB"          → explicit hex

        Returns the BBGGRR string (no prefix), ready for inline ASS overrides.
        """
        spec = (config.translation_color or "primary").strip().lower()
        if spec == "primary":
            return hex_rgb_to_bgr(config.primary_color)
        if spec == "secondary":
            return hex_rgb_to_bgr(config.secondary_color)
        return hex_rgb_to_bgr(config.translation_color)


@register_format("ssa")
class SSAFormat(ASSFormat):
    """SubStation Alpha format (predecessor to ASS).

    Inherits ASS metadata preservation - SSA and ASS share the same structure.
    """

    extensions = [".ssa"]
    pysubs2_format = "ssa"
    description = "SubStation Alpha - legacy format"

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        fps: float = 25.0,
        metadata: Optional[Dict] = None,
        config: Optional[ASSConfig] = None,
        **kwargs,
    ) -> bytes:
        """Convert to SSA bytes with style preservation."""
        from .base import maybe_expand_to_word_supervisions

        behavior, include_speaker, word_level = cls._unpack_render(**kwargs)
        config = config if isinstance(config, ASSConfig) else ASSConfig()

        supervisions = maybe_expand_to_word_supervisions(
            supervisions, word_level=word_level, format_id="ssa"
        )

        from .base import SpeakerTracker, render_bilingual_text

        subs = cls._create_ass_file(metadata, config)

        tf = behavior.translation_first
        tracker = SpeakerTracker()
        for sup in supervisions:
            text = render_bilingual_text(sup, separator="\\N", translation_first=tf)
            if cls._should_include_speaker(sup, include_speaker, tracker):
                text = f"{cls._format_speaker_prefix(sup.speaker)}{text}"
            event = cls._create_event_from_supervision(sup, text)
            subs.append(event)

        return subs.to_string(format_="ssa").encode("utf-8")


@register_format("sub")
class MicroDVDFormat(Pysubs2Format):
    """MicroDVD format (frame-based)."""

    extensions = [".sub"]
    pysubs2_format = "microdvd"
    description = "MicroDVD - frame-based subtitle format"


@register_format("sami")
class SAMIFormat(Pysubs2Format):
    """SAMI (Synchronized Accessible Media Interchange) format."""

    extensions = [".smi", ".sami"]
    pysubs2_format = "sami"
    description = "SAMI - Microsoft format for accessibility"


# Register alias for SMI extension
@register_format("smi")
class SMIFormat(SAMIFormat):
    """SMI format (alias for SAMI)."""

    pass
