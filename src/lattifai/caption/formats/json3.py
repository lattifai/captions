"""YouTube JSON3 format handler.

JSON3 is YouTube's native timed text format with the highest precision among
all YouTube caption formats. Available via yt-dlp ``--sub-format json3``.

Example source: https://www.youtube.com/watch?v=dvt_74kV-RM (The a16z Show)

It provides:
- Word-level timing via segment offset (`tOffsetMs`)
- ASR confidence scores (`acAsrConf`)
- Window positioning and styling metadata
- Speaker change markers (`>>` prefix)

Example structure:
```json
{
  "wireMagic": "pb3",
  "pens": [{}],
  "wsWinStyles": [{}, {"mhModeHint": 2, "juJustifCode": 0, "sdScrollDir": 3}],
  "wpWinPositions": [{}, {"apPoint": 6, "ahHorPos": 20, "avVerPos": 100, "rcRows": 2, "ccCols": 40}],
  "events": [
    {"tStartMs": 0, "dDurationMs": 100000, "id": 1, "wpWinPosId": 1, "wsWinStyleId": 1},
    {"tStartMs": 80, "dDurationMs": 3839, "wWinId": 1, "segs": [
      {"utf8": "Hello", "acAsrConf": 0},
      {"utf8": " world", "tOffsetMs": 500, "acAsrConf": 0}
    ]},
    {"tStartMs": 2550, "dDurationMs": 1369, "wWinId": 1, "aAppend": 1, "segs": [{"utf8": "\\n"}]}
  ]
}
```

Key attributes:
- events[].tStartMs: Event start time (ms)
- events[].dDurationMs: Event duration (ms)
- events[].aAppend: 1 = continuation/append event (skip for caption extraction)
- events[].segs[].utf8: Word text
- events[].segs[].tOffsetMs: Word offset from event start (ms), 0 if absent
- events[].segs[].acAsrConf: ASR confidence (0 = normal)
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..parsers.text_parser import normalize_text as normalize_text_fn
from ..supervision import AlignmentItem, Pathlike, Supervision
from . import register_format
from .base import FormatHandler

# Speaker change marker in YouTube auto-generated captions
_SPEAKER_CHANGE_RE = re.compile(r"^(?:>>|&gt;&gt;)\s*")


@register_format("json3")
class JSON3Format(FormatHandler):
    """YouTube JSON3 format reader and writer.

    Parses and generates YouTube's native JSON timed text format with:
    - Millisecond-precision timing
    - Word-level alignment extraction/generation
    - ASR confidence score preservation
    - Speaker change marker handling
    """

    extensions = [".json3"]
    description = "YouTube JSON3 - YouTube native timed text format with word-level timing"

    @classmethod
    def is_content(cls, source: Union[Pathlike, str]) -> bool:
        """Check if source is JSON3 content rather than a file path.

        Overrides base class to detect JSON3 content that starts with '{'.
        """
        if not isinstance(source, str):
            return False
        stripped = source.lstrip()
        return stripped.startswith("{") or "\n" in source or len(source) > 500

    @classmethod
    def can_read(cls, source: Union[Pathlike, str]) -> bool:
        """Check if source is JSON3 format."""
        source_str = str(source)

        # Check if it's content with JSON3 markers
        if cls.is_content(source_str):
            return '"wireMagic"' in source_str[:500] or (
                '"events"' in source_str[:2000] and '"tStartMs"' in source_str[:2000]
            )

        # Check by extension
        return source_str.lower().endswith(".json3")

    @classmethod
    def read(
        cls,
        source: Union[Pathlike, str],
        normalize_text: bool = True,
        **kwargs,
    ) -> List[Supervision]:
        """Read JSON3 format and extract supervisions with word-level alignment.

        Args:
            source: File path or JSON string content
            normalize_text: Whether to normalize text content

        Returns:
            List of Supervision objects with word-level alignment in the
            'alignment' field when word timing is available.
        """
        if cls.is_content(source):
            data = json.loads(source)
        else:
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)

        events = data.get("events", [])
        supervisions = []

        for event in events:
            # Skip window/config events (no segs)
            segs = event.get("segs")
            if not segs:
                continue

            # Skip append/continuation events
            if event.get("aAppend"):
                continue

            start_ms = event.get("tStartMs", 0)
            duration_ms = event.get("dDurationMs", 0)

            if duration_ms <= 0:
                continue

            start = start_ms / 1000.0
            duration = duration_ms / 1000.0

            # Extract words and their timing
            word_items: List[Dict[str, Any]] = []
            text_parts: List[str] = []
            has_speaker_change = False

            for i, seg in enumerate(segs):
                word_text = seg.get("utf8", "")
                if not word_text:
                    continue

                # Skip bare newline segments
                if word_text.strip() == "":
                    continue

                # Detect and strip >> speaker change marker on first segment
                if i == 0:
                    stripped = _SPEAKER_CHANGE_RE.sub("", word_text)
                    if stripped != word_text:
                        has_speaker_change = True
                        word_text = stripped
                        if not word_text.strip():
                            continue

                # Get word offset (relative to event start, in ms)
                offset_ms = seg.get("tOffsetMs", 0)
                word_start = start + (offset_ms / 1000.0)

                text_parts.append(word_text)
                word_items.append({
                    "text": word_text,
                    "start": word_start,
                    "score": seg.get("acAsrConf"),
                })

            if not text_parts:
                continue

            # Calculate word durations based on next word start
            alignment_items = []
            for i, item in enumerate(word_items):
                if i < len(word_items) - 1:
                    word_duration = word_items[i + 1]["start"] - item["start"]
                else:
                    word_duration = (start + duration) - item["start"]

                word_duration = max(0.0, word_duration)

                alignment_items.append(
                    AlignmentItem(
                        symbol=item["text"].strip(),
                        start=item["start"],
                        duration=word_duration,
                        score=item.get("score"),
                    )
                )

            # Build full text
            full_text = "".join(text_parts)
            if normalize_text:
                full_text = normalize_text_fn(full_text)

            alignment = {"word": alignment_items} if alignment_items else None

            supervisions.append(
                Supervision(
                    start=start,
                    duration=duration,
                    text=full_text,
                    alignment=alignment,
                    speaker=">>" if has_speaker_change else None,
                )
            )

        return supervisions

    @classmethod
    def extract_metadata(cls, source: Union[Pathlike, str], **kwargs) -> dict:
        """Extract JSON3 metadata.

        Returns:
            Dict containing window styles, positions, and format info.
        """
        if cls.is_content(source):
            data = json.loads(source)
        else:
            try:
                with open(source, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                return {}

        metadata = {"source_format": "json3"}

        if "wireMagic" in data:
            metadata["json3_wire_magic"] = data["wireMagic"]
        if "wsWinStyles" in data:
            metadata["json3_window_styles"] = len(data["wsWinStyles"])
        if "wpWinPositions" in data:
            metadata["json3_window_positions"] = len(data["wpWinPositions"])

        return metadata

    @classmethod
    def write(
        cls,
        supervisions: List[Supervision],
        output_path,
        include_speaker: bool = True,
        word_level: bool = False,
        **kwargs,
    ) -> Path:
        """Write JSON3 format.

        Args:
            supervisions: List of Supervision objects
            output_path: Output file path
            include_speaker: Whether to include speaker change markers
            word_level: If True, include word-level timing in segments
        """
        output_path = Path(output_path)
        content = cls.to_bytes(
            supervisions,
            include_speaker=include_speaker,
            word_level=word_level,
            **kwargs,
        )
        output_path.write_bytes(content)
        return output_path

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        include_speaker: bool = True,
        word_level: bool = False,
        metadata: Optional[Dict] = None,
        **kwargs,
    ) -> bytes:
        """Convert supervisions to JSON3 format bytes.

        Args:
            supervisions: List of Supervision objects
            include_speaker: Whether to include speaker change markers (>>)
            word_level: If True, include word-level timing offsets
            metadata: Optional metadata dict
        """
        # Default window styles and positions
        data: Dict[str, Any] = {
            "wireMagic": "pb3",
            "pens": [{}],
            "wsWinStyles": [
                {},
                {"mhModeHint": 2, "juJustifCode": 0, "sdScrollDir": 3},
            ],
            "wpWinPositions": [
                {},
                {"apPoint": 6, "ahHorPos": 20, "avVerPos": 100, "rcRows": 2, "ccCols": 40},
            ],
            "events": [],
        }

        events = data["events"]

        # Compute total duration for window event
        if supervisions:
            total_end_ms = int(max(s.end for s in supervisions) * 1000)
        else:
            total_end_ms = 0

        # Window definition event
        events.append({
            "tStartMs": 0,
            "dDurationMs": total_end_ms,
            "id": 1,
            "wpWinPosId": 1,
            "wsWinStyleId": 1,
        })

        for sup in supervisions:
            start_ms = int(sup.start * 1000)
            duration_ms = int(sup.duration * 1000)

            has_words = (
                word_level
                and sup.alignment
                and "word" in sup.alignment
                and len(sup.alignment["word"]) > 0
            )

            event: Dict[str, Any] = {
                "tStartMs": start_ms,
                "dDurationMs": duration_ms,
                "wWinId": 1,
            }

            if has_words:
                segs = []
                for i, word in enumerate(sup.alignment["word"]):
                    seg: Dict[str, Any] = {}

                    word_text = word.symbol
                    # Add leading space for non-first words
                    if i > 0 and not word_text.startswith(" "):
                        word_text = " " + word_text

                    # Add >> prefix for first word if speaker change
                    if i == 0 and include_speaker and sup.speaker == ">>":
                        word_text = ">> " + word_text

                    seg["utf8"] = word_text

                    # Offset from event start
                    offset_ms = int((word.start - sup.start) * 1000)
                    if offset_ms > 0:
                        seg["tOffsetMs"] = offset_ms

                    seg["acAsrConf"] = 0
                    segs.append(seg)
            else:
                text = sup.text or ""
                words = text.split()
                segs = []
                for i, word in enumerate(words):
                    word_text = (" " + word) if i > 0 else word
                    # Prepend >> to first word (YouTube convention)
                    if i == 0 and include_speaker and sup.speaker == ">>":
                        word_text = ">> " + word_text
                    seg = {"utf8": word_text, "acAsrConf": 0}
                    segs.append(seg)

            event["segs"] = segs
            events.append(event)

            # Add append event (newline) after each content event
            append_start_ms = start_ms + duration_ms - 10
            events.append({
                "tStartMs": max(0, append_start_ms),
                "dDurationMs": max(10, duration_ms),
                "wWinId": 1,
                "aAppend": 1,
                "segs": [{"utf8": "\n"}],
            })

        return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
