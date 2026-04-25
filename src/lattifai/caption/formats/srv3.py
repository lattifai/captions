"""YouTube SRV3 (YTT v3) format handler.

SRV3 is YouTube's proprietary timed text format, a TTML variant with:
- Time values in milliseconds
- Word-level timing via <s> (span) elements
- Window positioning via <wp> and styling via <ws>
- Relative word timing (offset from paragraph start)

Example structure:
```xml
<timedtext format="3">
  <head>
    <ws id="1" mh="2" ju="0" sd="3"/>
    <wp id="1" ap="6" ah="20" av="100" rc="2" cc="40"/>
  </head>
  <body>
    <w t="0" id="1" wp="1" ws="1"/>
    <p t="240" d="6559" w="1">
      <s ac="0">Does</s>
      <s t="320" ac="0"> fast</s>
      <s t="560" ac="0"> charging</s>
    </p>
  </body>
</timedtext>
```

Key attributes:
- p.t: Paragraph start time (ms)
- p.d: Paragraph duration (ms)
- p.a="1": Empty/continuation line (skip)
- s.t: Word offset from paragraph start (ms), 0 if missing
"""

import html
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..parsers.text_parser import normalize_text as normalize_text_fn
from ..supervision import AlignmentItem, Pathlike, Supervision
from . import register_format
from .base import FormatHandler, ParseResult


@register_format("srv3")
class SRV3Format(FormatHandler):
    """YouTube SRV3 format reader and writer.

    Parses and generates YouTube's proprietary timed text format (YTT v3) with:
    - Millisecond-precision timing
    - Word-level alignment extraction/generation
    - HTML entity encoding/decoding
    """

    extensions = [".srv3", ".ytt"]
    description = "YouTube SRV3/YTT - YouTube timed text format with word-level timing"

    @classmethod
    def is_content(cls, source: Union[Pathlike, str]) -> bool:
        """Check if source is SRV3 XML content rather than a file path.

        Overrides base class to detect SRV3 XML content even without newlines.
        """
        if not isinstance(source, str):
            return False
        # Check for SRV3 markers or standard XML/content indicators
        return "<timedtext" in source or 'format="3"' in source or "\n" in source or len(source) > 500

    @classmethod
    def can_read(cls, source: Union[Pathlike, str]) -> bool:
        """Check if source is SRV3 format."""
        source_str = str(source)

        # Check if it's content with SRV3 markers
        if cls.is_content(source_str):
            return "<timedtext" in source_str and 'format="3"' in source_str

        # Check by extension
        path_lower = source_str.lower()
        return any(path_lower.endswith(ext) for ext in cls.extensions)

    @classmethod
    def _parse_supervisions(cls, body: ET.Element, normalize_text: bool) -> List[Supervision]:
        """Extract supervisions from a pre-located SRV3 ``<body>`` element."""
        supervisions: List[Supervision] = []

        for p in body.findall("p"):
            # Skip empty/continuation lines (a="1")
            if p.get("a") == "1":
                continue

            # Get paragraph timing (milliseconds)
            p_start_ms = int(p.get("t", "0"))
            p_duration_ms = int(p.get("d", "0"))

            if p_duration_ms <= 0:
                continue

            # Convert to seconds
            p_start = p_start_ms / 1000.0
            p_duration = p_duration_ms / 1000.0

            # Extract words and their timing
            word_items = []
            text_parts = []

            for s in p.findall("s"):
                word_text = s.text or ""
                if not word_text:
                    continue

                # Decode HTML entities (e.g., &#39; -> ')
                word_text = html.unescape(word_text)

                # Get word offset (relative to paragraph start, in ms)
                s_offset_ms = int(s.get("t", "0"))
                word_start = p_start + (s_offset_ms / 1000.0)

                text_parts.append(word_text)

                # We'll calculate duration later based on next word
                word_items.append(
                    {
                        "text": word_text,
                        "start": word_start,
                    }
                )

            if not text_parts:
                continue

            # Calculate word durations based on next word start
            alignment_items = []
            for i, item in enumerate(word_items):
                if i < len(word_items) - 1:
                    # Duration until next word
                    word_duration = word_items[i + 1]["start"] - item["start"]
                else:
                    # Last word: duration until paragraph end
                    word_duration = (p_start + p_duration) - item["start"]

                # Ensure non-negative duration
                word_duration = max(0.0, word_duration)

                alignment_items.append(
                    AlignmentItem(
                        symbol=item["text"].strip(),
                        start=item["start"],
                        duration=word_duration,
                    )
                )

            # Build full text
            full_text = "".join(text_parts)
            if normalize_text:
                full_text = normalize_text_fn(full_text)

            # Create alignment dict if we have word timing
            alignment = {"word": alignment_items} if alignment_items else None

            supervisions.append(
                Supervision(
                    start=p_start,
                    duration=p_duration,
                    text=full_text,
                    alignment=alignment,
                )
            )

        return supervisions

    @classmethod
    def _extract_srv3_metadata(cls, root: ET.Element) -> dict:
        """Extract SRV3 metadata from an already-parsed XML root."""
        metadata: dict = {}
        fmt = root.get("format")
        if fmt:
            metadata["srv3_format"] = fmt
        head = root.find("head")
        if head is not None:
            ws_ids = [ws.get("id") for ws in head.findall("ws") if ws.get("id")]
            if ws_ids:
                metadata["srv3_window_styles"] = ws_ids
            wp_ids = [wp.get("id") for wp in head.findall("wp") if wp.get("id")]
            if wp_ids:
                metadata["srv3_window_positions"] = wp_ids
        return metadata

    @classmethod
    def parse(cls, source, normalize_text: bool = True, **kwargs) -> ParseResult:
        """Parse SRV3 in a single pass: supervisions + head metadata."""
        if cls.is_content(source):
            content = source
        else:
            with open(source, "r", encoding="utf-8") as f:
                content = f.read()

        try:
            root = ET.fromstring(content)
        except ET.ParseError:
            return ParseResult(supervisions=[])

        body = root.find("body")
        if body is None:
            return ParseResult(
                supervisions=[],
                format_metadata=cls._extract_srv3_metadata(root),
            )

        supervisions = cls._parse_supervisions(body, normalize_text=normalize_text)
        metadata = cls._extract_srv3_metadata(root)
        return ParseResult(supervisions=supervisions, format_metadata=metadata)

    @classmethod
    def write(
        cls,
        supervisions: List[Supervision],
        output_path,
        **kwargs,
    ) -> Path:
        """Write SRV3 format.

        Args:
            supervisions: List of Supervision objects
            output_path: Output file path
            **kwargs: style, metadata, etc.

        Returns:
            Path to written file
        """
        output_path = Path(output_path)
        content = cls.to_bytes(supervisions, **kwargs)
        output_path.write_bytes(content)
        return output_path

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        metadata: Optional[Dict] = None,
        **kwargs,
    ) -> bytes:
        """Convert supervisions to SRV3 format bytes.

        Args:
            supervisions: List of Supervision objects
            metadata: Optional metadata dict

        Returns:
            SRV3 XML content as UTF-8 encoded bytes
        """
        from .base import count_supervisions_with_words, has_word_alignment, resolve_word_level

        behavior, include_speaker, word_level = cls._unpack_render(**kwargs)
        # SRV3 defaults to segment output (smart_default=False) for backward
        # compatibility. RenderConfig.word_level=True opts into per-word <s> timing.
        _n_with = count_supervisions_with_words(supervisions)
        _use_word = resolve_word_level(
            word_level,
            n_with_words=_n_with,
            n_total=len(supervisions),
            format_id="srv3",
            smart_default=False,
        )
        # Create root element
        root = ET.Element("timedtext")
        root.set("format", "3")

        # Create head section with default window style and position
        head = ET.SubElement(root, "head")
        ws = ET.SubElement(head, "ws")
        ws.set("id", "0")
        ws1 = ET.SubElement(head, "ws")
        ws1.set("id", "1")
        ws1.set("mh", "2")
        ws1.set("ju", "0")
        ws1.set("sd", "3")

        wp = ET.SubElement(head, "wp")
        wp.set("id", "0")
        wp1 = ET.SubElement(head, "wp")
        wp1.set("id", "1")
        wp1.set("ap", "6")
        wp1.set("ah", "20")
        wp1.set("av", "100")
        wp1.set("rc", "2")
        wp1.set("cc", "40")

        # Create body section
        body = ET.SubElement(root, "body")

        # Add window element
        w = ET.SubElement(body, "w")
        w.set("t", "0")
        w.set("id", "1")
        w.set("wp", "1")
        w.set("ws", "1")

        from .base import SpeakerTracker

        # Process supervisions
        tracker = SpeakerTracker()
        for sup in supervisions:
            # Convert timing to milliseconds
            p_start_ms = int(sup.start * 1000)
            p_duration_ms = int(sup.duration * 1000)

            p = ET.SubElement(body, "p")
            p.set("t", str(p_start_ms))
            p.set("d", str(p_duration_ms))
            p.set("w", "1")

            # Check if we should output word-level timing for THIS supervision
            has_words = _use_word and has_word_alignment(sup)

            if has_words:
                # Output each word as <s> element with timing
                for i, word_item in enumerate(sup.alignment["word"]):
                    s = ET.SubElement(p, "s")
                    s.set("ac", "0")

                    # Calculate offset from paragraph start (in ms)
                    word_offset_ms = int((word_item.start - sup.start) * 1000)

                    # First word has implicit t=0, others need explicit offset
                    if i > 0 or word_offset_ms > 0:
                        s.set("t", str(max(0, word_offset_ms)))

                    # Escape text for XML and preserve leading space
                    word_text = word_item.symbol
                    # Add leading space for non-first words (SRV3 convention)
                    if i > 0 and not word_text.startswith(" "):
                        word_text = " " + word_text
                    s.text = word_text
            else:
                # Output full text without word-level timing
                text = sup.text
                if cls._should_include_speaker(sup, include_speaker, tracker):
                    text = f"[{sup.speaker}] {text}"

                # Split text into words and output as <s> elements without timing
                words = text.split()
                for i, word in enumerate(words):
                    s = ET.SubElement(p, "s")
                    s.set("ac", "0")
                    # Add leading space for non-first words
                    if i > 0:
                        s.text = " " + word
                    else:
                        s.text = word

        # Convert to pretty-printed XML string
        xml_str = cls._prettify_xml(root)
        return xml_str.encode("utf-8")

    @classmethod
    def _prettify_xml(cls, element: ET.Element) -> str:
        """Convert XML element to formatted string."""
        rough_string = ET.tostring(element, encoding="unicode")
        # Add XML declaration
        xml_declaration = '<?xml version="1.0" encoding="utf-8" ?>'
        return xml_declaration + rough_string
