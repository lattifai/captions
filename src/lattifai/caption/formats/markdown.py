"""Markdown transcript format handler.

Handles markdown transcript format with timestamps like [HH:MM:SS].
Supports reading and writing transcript files with speaker labels, events, and sections.
This format is commonly produced by the Gemini API for YouTube transcription,
but can represent any markdown-based transcript with timestamp annotations.
"""

import io
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..supervision import Pathlike, Supervision
from . import _READERS, _WRITERS, register_format
from .base import FormatHandler, FormatReader


@dataclass
class MarkdownSegment:
    """Represents a segment in a markdown transcript with metadata."""

    text: str
    timestamp: Optional[float] = None  # For backward compatibility (start time)
    end_timestamp: Optional[float] = None  # End time when timestamp is at the end
    speaker: Optional[str] = None
    section: Optional[str] = None
    segment_type: str = "dialogue"  # 'dialogue', 'event', or 'section_header'
    line_number: int = 0
    translation: Optional[str] = None  # Bilingual translation text (from > [lang] blockquote)

    @property
    def start(self) -> float:
        """Return start time in seconds."""
        return self.timestamp if self.timestamp is not None else 0.0

    @property
    def end(self) -> Optional[float]:
        """Return end time in seconds if available."""
        return self.end_timestamp


class MarkdownReader:
    """Parser for markdown transcript format with speaker labels and timestamps."""

    # Pattern to remove YAML front matter (---\n...\n---)
    FRONTMATTER_PATTERN = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)

    # Pattern to remove <thinking>...</thinking> blocks (including nested content)
    THINKING_PATTERN = re.compile(r"<thinking>.*?</thinking>", re.DOTALL)

    # Regex patterns for parsing (supports [HH:MM:SS], [HH:MM:SS.mmm], [MM:SS], [MM:SS.mmm])
    SECTION_HEADER_PATTERN = re.compile(r"^##\s*\[(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?\]\s*(.+)$")
    SPEAKER_PATTERN = re.compile(r"^\*\*(.+?[:：])\*\*\s*(.+)$")
    # Multi-event pattern: [Event1] [Event2] ... [TIMESTAMP] - two or more events before timestamp
    # Each event bracket must contain at least one letter (to exclude timestamps like [00:00:00])
    MULTI_EVENT_PATTERN = re.compile(
        r"^(\[[^\]]*[a-zA-Z\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff][^\]]*\](?:\s+\[[^\]]*[a-zA-Z\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff][^\]]*\])+)"
        r"\s+\[(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\.(\d{1,3}))?\]$"
    )
    # Event pattern: [Event] [HH:MM:SS.mmm] or [Event] [MM:SS.mmm] - prioritize HH:MM:SS format
    EVENT_PATTERN = re.compile(r"^\[([^\]]+)\]\s*\[(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\.(\d{1,3}))?\]$")
    # Timestamp at the end indicates end time (with optional milliseconds)
    INLINE_TIMESTAMP_END_PATTERN = re.compile(
        r"^(.+?)\s*\[(?:(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?|(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?)\]$"
    )
    # Timestamp at the beginning indicates start time (with optional milliseconds)
    INLINE_TIMESTAMP_START_PATTERN = re.compile(
        r"^\[(?:(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?|(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?)\]\s*(.+)$"
    )
    # Both start and end timestamps: [HH:MM:SS.mmm] text [HH:MM:SS.mmm] or mixed formats
    INLINE_BOTH_TIMESTAMPS_PATTERN = re.compile(
        r"^\[(?:(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?|(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?)\]\s*"
        r"(.+?)\s*"
        r"\[(?:(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?|(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?)\]$"
    )
    # Standalone timestamp on its own line (with optional milliseconds)
    STANDALONE_TIMESTAMP_PATTERN = re.compile(
        r"^\[(?:(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?|(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?)\]$"
    )

    # Section header with trailing timestamp: ## Title [HH:MM:SS] or ## Title [MM:SS]
    SECTION_HEADER_TRAILING_PATTERN = re.compile(r"^##\s+(.+?)\s+\[(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\.(\d{1,3}))?\]$")

    # Time range inline at end: text [HH:MM:SS → HH:MM:SS] (supports MM:SS and milliseconds)
    _TS = r"(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\.(\d{1,3}))?"
    INLINE_TIME_RANGE_PATTERN = re.compile(rf"^(.+?)\s*\[{_TS}\s*→\s*{_TS}\]$")

    # Markdown image line: ![alt](url)
    IMAGE_PATTERN = re.compile(r"^!\[.*?\]\(.*?\)")

    @classmethod
    def extract_frontmatter(cls, content: str) -> Dict[str, Any]:
        """Parse YAML frontmatter from markdown content.

        Returns:
            Dictionary of frontmatter fields. Empty dict if no frontmatter found.
        """
        m = cls.FRONTMATTER_PATTERN.match(content)
        if not m:
            return {}

        fm_block = m.group(0)
        # Strip the --- delimiters
        fm_lines = fm_block.strip().split("\n")[1:-1]  # skip first and last ---

        metadata: Dict[str, Any] = {}
        current_key = None
        current_value_lines = []

        for line in fm_lines:
            # Simple YAML key: value parsing (no nested objects)
            kv_match = re.match(r"^(\w[\w_-]*)\s*:\s*(.*)$", line)
            if kv_match:
                # Flush previous key
                if current_key is not None:
                    metadata[current_key] = "\n".join(current_value_lines).strip()
                current_key = kv_match.group(1)
                val = kv_match.group(2).strip()
                # YAML block scalar indicator (| or >) — value is in continuation lines
                if val in ("|", ">"):
                    current_value_lines = []
                    continue
                # Strip surrounding quotes
                if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                    val = val[1:-1]
                current_value_lines = [val]
            elif current_key is not None and line.startswith("  "):
                # Continuation line for multiline values
                current_value_lines.append(line.strip())

        if current_key is not None:
            metadata[current_key] = "\n".join(current_value_lines).strip()

        # Convert numeric fields
        for key in ("duration",):
            if key in metadata:
                try:
                    metadata[key] = float(metadata[key])
                except (ValueError, TypeError):
                    pass

        return metadata

    # New patterns for YouTube link format: [[MM:SS](URL&t=seconds)]
    YOUTUBE_SECTION_PATTERN = re.compile(r"^##\s*\[\[(\d{1,2}):(\d{2})\]\([^)]*&t=(\d+)\)\]\s*(.+)$")
    YOUTUBE_INLINE_PATTERN = re.compile(r"^(.+?)\s*\[\[(\d{1,2}):(\d{2})\]\([^)]*&t=(\d+)\)\]$")

    @classmethod
    def parse_timestamp(cls, *args, ms: str = None) -> float:
        """Convert timestamp to seconds.

        Supports [HH:MM:SS], [HH:MM:SS.mmm], [MM:SS], and [MM:SS.mmm] formats.
        Args can be (hours, minutes, seconds) or (minutes, seconds).
        Can also accept a single argument which is seconds.
        Optional ms parameter for milliseconds (as string, e.g., "500" for 0.5s).
        """
        ms_value = 0.0
        if ms is not None:
            # Normalize milliseconds: "5" -> 0.5, "50" -> 0.5, "500" -> 0.5
            ms_str = ms.ljust(3, "0")[:3]  # Pad to 3 digits or truncate
            ms_value = int(ms_str) / 1000.0

        if len(args) == 4:
            # HH:MM:SS.mmm format (ms passed as 4th arg)
            hours, minutes, seconds, ms_arg = args
            if ms_arg is not None:
                ms_str = str(ms_arg).ljust(3, "0")[:3]
                ms_value = int(ms_str) / 1000.0
            return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + ms_value
        elif len(args) == 3:
            # HH:MM:SS format (or MM:SS.mmm with ms as 3rd arg)
            hours, minutes, seconds = args
            return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + ms_value
        elif len(args) == 2:
            # MM:SS format
            minutes, seconds = args
            return int(minutes) * 60 + int(seconds) + ms_value
        elif len(args) == 1:
            # Direct seconds (from YouTube &t= parameter)
            return float(args[0])
        else:
            raise ValueError(f"Invalid timestamp args: {args}")

    @classmethod
    def _parse_flexible_timestamp(cls, h_or_m, m_or_s, s_opt=None, ms=None):
        """Parse timestamp from flexible groups supporting both HH:MM:SS and MM:SS formats."""
        if s_opt is not None:
            return cls.parse_timestamp(h_or_m, m_or_s, s_opt, ms)
        else:
            return cls.parse_timestamp(h_or_m, m_or_s, ms=ms)

    @classmethod
    def _extract_timestamps(cls, text: str):
        """Extract start/end timestamps from inline text.

        Tries patterns in priority order: both timestamps, time range,
        start only, end only, YouTube inline. Returns (clean_text, start, end).
        """
        # Both start and end: [HH:MM:SS] text [HH:MM:SS]
        m = cls.INLINE_BOTH_TIMESTAMPS_PATTERN.match(text)
        if m:
            g = m.groups()
            start = (
                cls.parse_timestamp(g[0], g[1], g[2], g[3])
                if g[0] is not None
                else cls.parse_timestamp(g[4], g[5], ms=g[6])
            )
            end = (
                cls.parse_timestamp(g[8], g[9], g[10], g[11])
                if g[8] is not None
                else cls.parse_timestamp(g[12], g[13], ms=g[14])
            )
            return g[7].strip(), start, end

        # Time range: text [HH:MM:SS → HH:MM:SS]
        m = cls.INLINE_TIME_RANGE_PATTERN.match(text)
        if m:
            g = m.groups()
            return (
                g[0].strip(),
                cls._parse_flexible_timestamp(g[1], g[2], g[3], g[4]),
                cls._parse_flexible_timestamp(g[5], g[6], g[7], g[8]),
            )

        # Start timestamp: [HH:MM:SS] text
        m = cls.INLINE_TIMESTAMP_START_PATTERN.match(text)
        if m:
            g = m.groups()
            start = (
                cls.parse_timestamp(g[0], g[1], g[2], g[3])
                if g[0] is not None
                else (cls.parse_timestamp(g[4], g[5], ms=g[6]) if g[4] is not None else None)
            )
            return g[7].strip(), start, None

        # End timestamp: text [HH:MM:SS]
        m = cls.INLINE_TIMESTAMP_END_PATTERN.match(text)
        if m:
            g = m.groups()
            end = (
                cls.parse_timestamp(g[1], g[2], g[3], g[4])
                if g[1] is not None
                else (cls.parse_timestamp(g[5], g[6], ms=g[7]) if g[5] is not None else None)
            )
            return g[0].strip(), None, end

        # YouTube inline: text [[MM:SS](url)]
        m = cls.YOUTUBE_INLINE_PATTERN.match(text)
        if m:
            return m.group(1).strip(), None, cls.parse_timestamp(m.group(4))

        return text.strip(), None, None

    @staticmethod
    def _estimate_duration(text: str, wps: float = 3.3, cps: float = 4.0) -> float:
        """Estimate speech duration from text length.

        Uses word-based estimation for Latin scripts and character-based
        estimation for CJK scripts (Chinese, Japanese, Korean) where
        whitespace splitting severely underestimates length.
        """
        if not text:
            return 0.3
        cjk_count = sum(
            1
            for c in text
            if "\u4e00" <= c <= "\u9fff"  # CJK Unified Ideographs
            or "\u3040" <= c <= "\u30ff"  # Hiragana + Katakana
            or "\uac00" <= c <= "\ud7af"  # Hangul
            or "\u3400" <= c <= "\u4dbf"  # CJK Extension A
            or "\uf900" <= c <= "\ufaff"  # CJK Compatibility
        )
        if cjk_count > len(text.replace(" ", "")) * 0.3:
            # CJK-dominant: ~4 characters per second
            return max(len(text.replace(" ", "")) / cps, 0.3)
        else:
            # Latin-dominant: ~3.3 words per second
            return max(len(text.split()) / wps, 0.3)

    # Pattern for bilingual translation lines: > [lang] text
    TRANSLATION_LINE_PATTERN = re.compile(r"^>\s*\[([\w-]+)\]\s+(.+)$")

    @classmethod
    def read(
        cls,
        transcript_path: Union[Pathlike, str],
        include_events: bool = True,
        include_sections: bool = False,
        target_lang: Optional[str] = None,
    ) -> List[MarkdownSegment]:
        """Parse markdown transcript file or content and return list of transcript segments.

        Args:
                transcript_path: Path to the transcript file or raw string content
                include_events: Whether to include event descriptions like [Applause]
                include_sections: Whether to include section headers

        Returns:
                List of MarkdownSegment objects with all metadata
        """
        content = ""
        # Check if transcript_path is a multi-line string (content) or a short string (likely path)
        is_content = "\n" in str(transcript_path) or len(str(transcript_path)) > 1000

        if is_content:
            content = str(transcript_path)
        else:
            p = Path(transcript_path).expanduser().resolve()
            if p.exists() and p.is_file():
                with open(p, "r", encoding="utf-8") as f:
                    content = f.read()
            else:
                # Fallback: treat as content if path doesn't exist
                content = str(transcript_path)

        # Extract target_lang from frontmatter before stripping it
        if target_lang is None:
            frontmatter = cls.extract_frontmatter(content)
            target_lang = frontmatter.get("target_lang")

        # Remove YAML front matter (---\n...\n---)
        content = cls.FRONTMATTER_PATTERN.sub("", content)

        # Remove <thinking>...</thinking> blocks (e.g., from Gemini thinking models)
        content = cls.THINKING_PATTERN.sub("", content)

        segments: List[MarkdownSegment] = []
        current_section = None
        current_speaker = None

        lines = content.splitlines()
        for line_num, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue

            # Bilingual translation line: > [lang] text
            # Only parsed when target_lang is set (via frontmatter or explicit parameter)
            # and the tag matches target_lang to avoid consuming unrelated blockquotes
            if target_lang and line.startswith(">"):
                tr_match = cls.TRANSLATION_LINE_PATTERN.match(line)
                if tr_match and tr_match.group(1) == target_lang:
                    if segments and segments[-1].segment_type == "dialogue":
                        segments[-1].translation = tr_match.group(2).strip()
                    continue  # Consume matching > [target_lang] lines

            # Skip table of contents
            if line.startswith("* ["):
                continue
            if line.startswith("## Table of Contents"):
                continue

            # Skip markdown images (e.g., ![cover](imgs/cover.jpg))
            if cls.IMAGE_PATTERN.match(line):
                continue

            # Parse section headers
            section_match = cls.SECTION_HEADER_PATTERN.match(line)
            if section_match:
                hours, minutes, seconds, ms, section_title = section_match.groups()
                timestamp = cls.parse_timestamp(hours, minutes, seconds, ms)
                current_section = section_title.strip()
                if include_sections:
                    segments.append(
                        MarkdownSegment(
                            text=section_title.strip(),
                            timestamp=timestamp,
                            section=current_section,
                            segment_type="section_header",
                            line_number=line_num,
                        )
                    )
                continue

            # Parse YouTube format section headers
            youtube_section_match = cls.YOUTUBE_SECTION_PATTERN.match(line)
            if youtube_section_match:
                minutes, seconds, url_seconds, section_title = youtube_section_match.groups()
                timestamp = cls.parse_timestamp(url_seconds)
                current_section = section_title.strip()
                if include_sections:
                    segments.append(
                        MarkdownSegment(
                            text=section_title.strip(),
                            timestamp=timestamp,
                            section=current_section,
                            segment_type="section_header",
                            line_number=line_num,
                        )
                    )
                continue

            # Parse section headers with trailing timestamp: ## Title [HH:MM:SS]
            trailing_section_match = cls.SECTION_HEADER_TRAILING_PATTERN.match(line)
            if trailing_section_match:
                groups = trailing_section_match.groups()
                section_title = groups[0].strip()
                timestamp = cls._parse_flexible_timestamp(groups[1], groups[2], groups[3], groups[4])
                current_section = section_title
                if include_sections:
                    segments.append(
                        MarkdownSegment(
                            text=section_title,
                            timestamp=timestamp,
                            section=current_section,
                            segment_type="section_header",
                            line_number=line_num,
                        )
                    )
                continue

            # Parse standalone timestamp [HH:MM:SS] or [HH:MM:SS.mmm]
            # Often used as an end timestamp for the preceding block
            standalone_match = cls.STANDALONE_TIMESTAMP_PATTERN.match(line)
            if standalone_match:
                groups = standalone_match.groups()
                # Groups: (h, m, s, ms1, m2, s2, ms2)
                if groups[0] is not None:
                    ts = cls.parse_timestamp(groups[0], groups[1], groups[2], groups[3])
                else:
                    ts = cls.parse_timestamp(groups[4], groups[5], ms=groups[6])

                # Assign to previous dialogue segment if it doesn't have an end time
                if segments and segments[-1].segment_type == "dialogue":
                    if segments[-1].end_timestamp is None:
                        segments[-1].end_timestamp = ts
                    elif segments[-1].timestamp is None:
                        # If it has an end but no start, this standalone might be its start?
                        # Usually standalone is end, but let's be flexible
                        segments[-1].timestamp = ts
                continue

            # Parse multi-event lines: [Event1] [Event2] [HH:MM:SS]
            multi_event_match = cls.MULTI_EVENT_PATTERN.match(line)
            if multi_event_match:
                groups = multi_event_match.groups()
                event_group = groups[0]
                hours_or_minutes = groups[1]
                minutes_or_seconds = groups[2]
                seconds_optional = groups[3]
                ms = groups[4]

                if seconds_optional is not None:
                    timestamp = cls.parse_timestamp(hours_or_minutes, minutes_or_seconds, seconds_optional, ms)
                else:
                    timestamp = cls.parse_timestamp(hours_or_minutes, minutes_or_seconds, ms=ms)

                if include_events and timestamp is not None:
                    individual_events = re.findall(r"\[[^\]]+\]", event_group)
                    for event_text in individual_events:
                        segments.append(
                            MarkdownSegment(
                                text=event_text.strip(),
                                timestamp=timestamp,
                                section=current_section,
                                segment_type="event",
                                line_number=line_num,
                            )
                        )
                continue

            # Parse event descriptions [event] [HH:MM:SS] or [event] [HH:MM:SS.mmm]
            event_match = cls.EVENT_PATTERN.match(line)
            if event_match:
                groups = event_match.groups()
                # Groups: (event_text, h_or_m, m_or_s, s_opt, ms)
                event_text = groups[0]
                hours_or_minutes = groups[1]
                minutes_or_seconds = groups[2]
                seconds_optional = groups[3]
                ms = groups[4]

                if seconds_optional is not None:
                    timestamp = cls.parse_timestamp(hours_or_minutes, minutes_or_seconds, seconds_optional, ms)
                else:
                    timestamp = cls.parse_timestamp(hours_or_minutes, minutes_or_seconds, ms=ms)

                if include_events and timestamp is not None:
                    segments.append(
                        MarkdownSegment(
                            text=f"[{event_text.strip()}]",
                            timestamp=timestamp,
                            section=current_section,
                            segment_type="event",
                            line_number=line_num,
                        )
                    )
                continue

            # Parse speaker dialogue: **Speaker:** [START] Text [END] or **Speaker:** Text [END]
            speaker_match = cls.SPEAKER_PATTERN.match(line)
            if speaker_match:
                speaker, text_with_timestamp = speaker_match.groups()
                current_speaker = speaker.strip()

                text, start_timestamp, end_timestamp = cls._extract_timestamps(text_with_timestamp.strip())

                segments.append(
                    MarkdownSegment(
                        text=text.strip(),
                        timestamp=start_timestamp,
                        end_timestamp=end_timestamp,
                        speaker=current_speaker,
                        section=current_section,
                        segment_type="dialogue",
                        line_number=line_num,
                    )
                )
                current_speaker = None
                continue

            # Parse plain text (might contain inline timestamps or be a continuation)
            text, start_timestamp, end_timestamp = cls._extract_timestamps(line)
            if start_timestamp is not None or end_timestamp is not None:
                segments.append(
                    MarkdownSegment(
                        text=text,
                        timestamp=start_timestamp,
                        end_timestamp=end_timestamp,
                        speaker=current_speaker,
                        section=current_section,
                        segment_type="dialogue",
                        line_number=line_num,
                    )
                )
            else:
                # Plain text without any recognized markers
                # If it follows a speaker line or another dialogue line without end timestamp,
                # merge it into the last segment to support multi-line text blocks.
                if segments and segments[-1].segment_type == "dialogue" and segments[-1].end_timestamp is None:
                    segments[-1].text += " " + line.strip()
                else:
                    # Skip markdown headers, TOC items, and other non-dialogue formatting
                    if line.startswith("#") or line.startswith("- "):
                        continue

                    segments.append(
                        MarkdownSegment(
                            text=line.strip(),
                            speaker=current_speaker,
                            section=current_section,
                            segment_type="dialogue",
                            line_number=line_num,
                        )
                    )

        return segments

    @classmethod
    def extract_for_alignment(
        cls,
        transcript_path: Pathlike,
        merge_consecutive: bool = False,
        min_duration: float = 0.1,
        merge_max_gap: float = 2.0,
        **kwargs,
    ) -> List[Supervision]:
        """Extract text segments for forced alignment.

        This extracts only dialogue segments (not events or section headers)
        and converts them to Supervision objects suitable for alignment.

        Args:
                transcript_path: Path to the transcript file
                merge_consecutive: Whether to merge consecutive segments from same speaker
                min_duration: Minimum duration for a segment
                merge_max_gap: Maximum time gap (seconds) to merge consecutive segments

        Returns:
                List of Supervision objects ready for alignment
        """
        segments = cls.read(transcript_path, include_events=True, include_sections=False, **kwargs)

        # Determine target_lang for Supervision propagation
        # Prefer explicit kwarg, then fall back to frontmatter
        target_lang = kwargs.get("target_lang")
        if target_lang is None and any(s.translation for s in segments):
            is_content = "\n" in str(transcript_path) or len(str(transcript_path)) > 1000
            if is_content:
                fm = cls.extract_frontmatter(str(transcript_path))
            else:
                p = Path(transcript_path).expanduser().resolve()
                fm = cls.extract_frontmatter(p.read_text(encoding="utf-8")) if p.exists() else {}
            target_lang = fm.get("target_lang")

        # Filter to dialogue and event segments (with or without timestamps)
        dialogue_segments = [
            s
            for s in segments
            if s.segment_type == "event"
            or (s.segment_type == "dialogue" and (s.speaker or s.timestamp is not None or s.end_timestamp is not None))
        ]

        if not dialogue_segments:
            raise ValueError(f"No dialogue segments found in {transcript_path}")

        # Convert to Supervision objects
        supervisions: List[Supervision] = []
        prev_end_time = 0.0

        for i, segment in enumerate(dialogue_segments):
            seg_start = None
            seg_end = None

            # Determine start and end times based on available timestamps
            if segment.timestamp is not None:
                # Has start time
                seg_start = segment.timestamp
                if segment.end_timestamp is not None:
                    # Has both start and end
                    seg_end = segment.end_timestamp
                else:
                    # Only has start, estimate end
                    if i < len(dialogue_segments) - 1:
                        # Use next segment's time
                        next_seg = dialogue_segments[i + 1]
                        if next_seg.timestamp is not None:
                            seg_end = next_seg.timestamp
                        elif next_seg.end_timestamp is not None:
                            # Next has only end, estimate its start and use that
                            estimated_duration_next = cls._estimate_duration(next_seg.text)
                            seg_end = next_seg.end_timestamp - estimated_duration_next

                    if seg_end is None:
                        # Estimate based on text length
                        seg_end = seg_start + cls._estimate_duration(segment.text)

            elif segment.end_timestamp is not None:
                # Only has end time, need to infer start from previous segment's end
                seg_end = segment.end_timestamp
                # Use previous segment's end time as start (prev_end_time starts at 0.0)
                seg_start = prev_end_time

            else:
                # No timestamps at all: estimate from text length
                seg_start = prev_end_time
                seg_end = seg_start + cls._estimate_duration(segment.text)

            if seg_start is not None and seg_end is not None:
                duration = max(seg_end - seg_start, min_duration)
                sup_kwargs = dict(
                    text=segment.text.strip(),
                    start=seg_start,
                    duration=duration,
                    id=f"segment_{i:05d}",
                    speaker=segment.speaker,
                )
                if segment.translation:
                    sup_kwargs["translation"] = segment.translation
                    if target_lang:
                        sup_kwargs["target_lang"] = target_lang
                supervisions.append(Supervision(**sup_kwargs))
                prev_end_time = seg_start + duration

        # Optionally merge consecutive segments from same speaker
        if merge_consecutive:
            merged = []
            current_speaker = None
            current_texts = []
            current_start = None
            last_end_time = None

            for i, (segment, sup) in enumerate(zip(dialogue_segments, supervisions)):
                # Check if we should merge with previous segment
                should_merge = False
                if segment.speaker == current_speaker and current_start is not None:
                    # Same speaker - check time gap
                    time_gap = sup.start - last_end_time if last_end_time else 0
                    if time_gap <= merge_max_gap:
                        should_merge = True

                if should_merge:
                    # Same speaker within time threshold, accumulate
                    current_texts.append(segment.text)
                    last_end_time = sup.start + sup.duration
                else:
                    # Different speaker or gap too large, save previous segment
                    if current_texts:
                        merged_text = " ".join(current_texts)
                        merged.append(
                            Supervision(
                                text=merged_text,
                                start=current_start,
                                duration=last_end_time - current_start,
                                id=f"merged_{len(merged):05d}",
                            )
                        )
                    current_speaker = segment.speaker
                    current_texts = [segment.text]
                    current_start = sup.start
                    last_end_time = sup.start + sup.duration

            # Add final segment
            if current_texts:
                merged_text = " ".join(current_texts)
                merged.append(
                    Supervision(
                        text=merged_text,
                        start=current_start,
                        duration=last_end_time - current_start,
                        id=f"merged_{len(merged):05d}",
                    )
                )

            supervisions = merged

        return supervisions


class MarkdownWriter:
    """Writer for markdown transcript files with aligned timestamps."""

    # Fields to include in YAML frontmatter (in order)
    FRONTMATTER_FIELDS = [
        "title",
        "channel",
        "url",
        "date",
        "duration",
        "language",
        "target_lang",
        "transcript_source",
        "description",
    ]

    @classmethod
    def _write_frontmatter(cls, f, metadata: Optional[Dict[str, Any]] = None):
        """Write YAML frontmatter block if metadata contains relevant fields.

        Description is written last as it can be multiline. Fields not in
        FRONTMATTER_FIELDS are silently skipped to keep the frontmatter clean.
        """
        if not metadata:
            return

        # Collect fields that have values
        entries = []
        for key in cls.FRONTMATTER_FIELDS:
            val = metadata.get(key)
            # Also check common aliases
            if val is None and key == "channel":
                val = metadata.get("uploader") or metadata.get("channel_name")
            if val is None and key == "url":
                val = metadata.get("video_url") or metadata.get("webpage_url")
            if val is None or val == "":
                continue
            entries.append((key, val))

        if not entries:
            return

        f.write("---\n")
        for key, val in entries:
            if key == "description":
                # Multiline: use YAML literal block scalar
                desc = str(val).replace("\n\n", "\n")  # collapse double newlines
                # Truncate to first meaningful section (before SPONSORS/LINKS etc.)
                for marker in ["*SPONSORS:", "*CONTACT ", "*EPISODE LINKS:", "*PODCAST LINKS:", "*SOCIAL LINKS:"]:
                    pos = desc.find(marker)
                    if pos > 0:
                        desc = desc[:pos].rstrip()
                        break
                if "\n" in desc:
                    f.write("description: |\n")
                    for line in desc.split("\n"):
                        f.write(f"  {line}\n")
                else:
                    f.write(f'description: "{desc}"\n')
            elif isinstance(val, (int, float)):
                f.write(f"{key}: {val}\n")
            else:
                val_str = str(val)
                if any(c in val_str for c in (":", '"', "'", "#", "[", "]", "{", "}")):
                    f.write(f'{key}: "{val_str}"\n')
                else:
                    f.write(f"{key}: {val_str}\n")
        f.write("---\n\n")

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Convert seconds to [HH:MM:SS] format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"

    @classmethod
    def update_timestamps(
        cls,
        original_transcript: Pathlike,
        aligned_supervisions: List[Supervision],
        output_path: Pathlike,
        timestamp_mapping: Optional[Dict[int, float]] = None,
    ) -> Pathlike:
        """Update transcript file with corrected timestamps from alignment.

        Args:
                original_transcript: Path to the original transcript file
                aligned_supervisions: List of aligned Supervision objects with corrected timestamps
                output_path: Path to write the updated transcript
                timestamp_mapping: Optional manual mapping from line_number to new timestamp

        Returns:
                Path to the output file
        """
        original_path = Path(original_transcript)
        output_path = Path(output_path)

        # Read original file
        with open(original_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Parse original segments to get line numbers
        original_segments = MarkdownReader.read(original_transcript, include_events=True, include_sections=True)

        # Create mapping from line number to new timestamp
        if timestamp_mapping is None:
            timestamp_mapping = cls._create_timestamp_mapping(original_segments, aligned_supervisions)

        # Update timestamps in lines
        updated_lines = []
        for line_num, line in enumerate(lines, start=1):
            if line_num in timestamp_mapping:
                ts_value = timestamp_mapping[line_num]
                if isinstance(ts_value, tuple):
                    new_start, new_end = ts_value
                    updated_line = cls._replace_timestamp(line, new_start, new_end)
                else:
                    # Backward compatibility: single float value
                    updated_line = cls._replace_timestamp(line, ts_value)
                updated_lines.append(updated_line)
            else:
                updated_lines.append(line)

        # Write updated content
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(updated_lines)

        return output_path

    @classmethod
    def _create_timestamp_mapping(
        cls, original_segments: List[MarkdownSegment], aligned_supervisions: List[Supervision]
    ) -> Dict[int, tuple]:
        """Create mapping from line numbers to new timestamps based on alignment.

        This performs text matching between original segments and aligned supervisions
        to determine which timestamps should be updated.

        Returns:
            Dict mapping line_number to (start, end) timestamp tuples.
        """
        mapping = {}

        # Create a simple text-based matching
        dialogue_segments = [s for s in original_segments if s.segment_type == "dialogue"]

        # Try to match based on text content
        for aligned_sup in aligned_supervisions:
            aligned_text = aligned_sup.text.strip()

            # Find best matching original segment
            best_match = None
            best_score = 0

            for orig_seg in dialogue_segments:
                orig_text = orig_seg.text.strip()

                # Simple text similarity (could be improved with fuzzy matching)
                if aligned_text == orig_text:
                    best_match = orig_seg
                    best_score = 1.0
                    break
                elif aligned_text in orig_text or orig_text in aligned_text:
                    score = min(len(aligned_text), len(orig_text)) / max(len(aligned_text), len(orig_text))
                    if score > best_score:
                        best_score = score
                        best_match = orig_seg

            # If we found a good match, update the mapping
            if best_match and best_score > 0.8:
                end_time = aligned_sup.start + aligned_sup.duration
                mapping[best_match.line_number] = (aligned_sup.start, end_time)

        return mapping

    # Pre-compiled patterns for _replace_timestamp (avoid re-compiling per call)
    _TIME_RANGE_RE = re.compile(
        r"\[\d{1,2}:\d{2}(?::\d{2})?(?:\.\d{1,3})?\s*→\s*\d{1,2}:\d{2}(?::\d{2})?(?:\.\d{1,3})?\]"
    )
    _SINGLE_TS_RE = re.compile(r"\[\d{1,2}:\d{2}:\d{2}(?:\.\d{1,3})?\]")

    @classmethod
    def _replace_timestamp(cls, line: str, new_start: float, new_end: float = None) -> str:
        """Replace timestamp(s) in a line with new values.

        Handles time ranges [HH:MM:SS → HH:MM:SS], dual timestamps
        [start] text [end], and single timestamps [HH:MM:SS].
        """
        # Handle time range format: [HH:MM:SS → HH:MM:SS]
        range_match = cls._TIME_RANGE_RE.search(line)
        if range_match and new_end is not None:
            start_str = cls.format_timestamp(new_start)[1:-1]
            end_str = cls.format_timestamp(new_end)[1:-1]
            new_range = f"[{start_str} → {end_str}]"
            return line[: range_match.start()] + new_range + line[range_match.end() :]

        # Find all [HH:MM:SS] timestamps in the line
        matches = list(cls._SINGLE_TS_RE.finditer(line))
        if not matches:
            return line

        # Dual timestamps [start] text [end]: replace first and last separately
        if len(matches) >= 2 and new_end is not None:
            last = matches[-1]
            line = line[: last.start()] + cls.format_timestamp(new_end) + line[last.end() :]
            first = matches[0]
            line = line[: first.start()] + cls.format_timestamp(new_start) + line[first.end() :]
        else:
            # Single timestamp
            first = matches[0]
            line = line[: first.start()] + cls.format_timestamp(new_start) + line[first.end() :]

        return line

    @classmethod
    def write_aligned_transcript(
        cls,
        aligned_supervisions: List[Supervision],
        output_path: Pathlike,
        include_word_timestamps: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,  # Accept extra kwargs for Caption.write() compatibility
    ) -> Pathlike:
        """Write a new transcript file from aligned supervisions.

        This creates a simplified markdown transcript format with accurate timestamps.
        If metadata is provided, writes YAML frontmatter at the top.

        Args:
                aligned_supervisions: List of aligned Supervision objects
                output_path: Path to write the transcript
                include_word_timestamps: Whether to include word-level timestamps if available
                metadata: Optional metadata dict to write as YAML frontmatter

        Returns:
                Path to the output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            cls._write_frontmatter(f, metadata)

            title = (metadata or {}).get("title", "Aligned Transcript")
            f.write(f"# {title}\n\n")

            for i, sup in enumerate(aligned_supervisions):
                # Write segment with timestamp
                start_ts = cls.format_timestamp(sup.start)
                end_ts = cls.format_timestamp(sup.start + sup.duration) if sup.duration else None
                text = sup.text or ""

                # Speaker label
                speaker_prefix = f"**{sup.speaker}:** " if sup.speaker else ""

                # Include end timestamp when translation/word data follows
                # to prevent Reader continuation-merge from polluting re-reads
                has_followup = sup.translation or (
                    include_word_timestamps and hasattr(sup, "alignment") and sup.alignment and "word" in sup.alignment
                )
                if has_followup and end_ts:
                    f.write(f"{speaker_prefix}{start_ts} {text} {end_ts}\n")
                else:
                    f.write(f"{speaker_prefix}{start_ts} {text}\n")

                if sup.translation:
                    lang_tag = sup.target_lang or "translation"
                    f.write(f"> [{lang_tag}] {sup.translation}\n")

                # Optionally write word-level timestamps
                if include_word_timestamps and hasattr(sup, "alignment") and sup.alignment:
                    if "word" in sup.alignment:
                        f.write("  Words: ")
                        word_parts = []
                        for word_info in sup.alignment["word"]:
                            word_ts = cls.format_timestamp(word_info["start"])
                            word_parts.append(f'{word_info["symbol"]}{word_ts}')
                        f.write(" ".join(word_parts))
                        f.write("\n")

                f.write("\n")

        return output_path

    @classmethod
    def write(
        cls,
        supervisions: List[Supervision],
        output_path: Pathlike,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Path:
        """Alias for write_aligned_transcript for Caption API compatibility."""
        return Path(cls.write_aligned_transcript(supervisions, output_path, metadata=metadata, **kwargs))

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> bytes:
        """Convert aligned supervisions to markdown format bytes (in-memory)."""
        buf = io.StringIO()
        cls._write_frontmatter(buf, metadata)

        title = (metadata or {}).get("title", "Aligned Transcript")
        buf.write(f"# {title}\n\n")

        include_word_timestamps = kwargs.get("include_word_timestamps", False)
        for sup in supervisions:
            start_ts = cls.format_timestamp(sup.start)
            end_ts = cls.format_timestamp(sup.start + sup.duration) if sup.duration else None
            text = sup.text or ""
            speaker_prefix = f"**{sup.speaker}:** " if sup.speaker else ""

            has_followup = sup.translation or (
                include_word_timestamps and hasattr(sup, "alignment") and sup.alignment and "word" in sup.alignment
            )
            if has_followup and end_ts:
                buf.write(f"{speaker_prefix}{start_ts} {text} {end_ts}\n")
            else:
                buf.write(f"{speaker_prefix}{start_ts} {text}\n")

            if sup.translation:
                lang_tag = sup.target_lang or "translation"
                buf.write(f"> [{lang_tag}] {sup.translation}\n")

            if include_word_timestamps and hasattr(sup, "alignment") and sup.alignment:
                if "word" in sup.alignment:
                    buf.write("  Words: ")
                    word_parts = [f'{w["symbol"]}{cls.format_timestamp(w["start"])}' for w in sup.alignment["word"]]
                    buf.write(" ".join(word_parts))
                    buf.write("\n")

            buf.write("\n")

        return buf.getvalue().encode("utf-8")


@register_format("markdown")
class MarkdownFormat(FormatHandler):
    """Markdown transcript format with timestamps.

    Supports markdown files containing transcript content with timestamp patterns
    such as [HH:MM:SS], [MM:SS], and **Speaker:** labels. This format was originally
    associated with Gemini API transcription output but applies to any markdown-based
    transcript with these conventions.
    """

    extensions = [".md"]
    description = "Markdown transcript format with timestamps"

    @classmethod
    def can_read(cls, path) -> bool:
        """Check if this is a markdown transcript file.

        Accepts files that:
        - Have "gemini" in the filename (legacy convention)
        - Are .md files containing transcript timestamp patterns like [HH:MM:SS],
          [MM:SS], or **Speaker:** labels
        """
        path_str = str(path)
        path_lower = path_str.lower()

        # Legacy: accept files with "gemini" in the name
        if "gemini" in path_lower and path_lower.endswith(".md"):
            return True

        # Only consider .md files for content sniffing
        if not path_lower.endswith(".md"):
            return False

        # Content sniffing: check for transcript timestamp patterns
        try:
            p = Path(path_str)  # Use original path (case-sensitive filesystems)
            if p.exists() and p.is_file():
                with open(p, "r", encoding="utf-8") as f:
                    # Read first 4KB for sniffing
                    content = f.read(4096)

                # Check for timestamp patterns: [HH:MM:SS] or [MM:SS]
                timestamp_re = re.compile(r"\[\d{1,2}:\d{2}(?::\d{2})?\]")
                # Check for speaker patterns: **Speaker:**
                speaker_re = re.compile(r"\*\*.+?[:：]\*\*")

                has_timestamps = bool(timestamp_re.search(content))
                has_speakers = bool(speaker_re.search(content))

                # Accept if we find timestamps (the core marker of this format)
                return has_timestamps or has_speakers
        except (OSError, ValueError):
            pass

        return False

    @classmethod
    def extract_metadata(cls, source: Union[Pathlike, str]) -> Dict[str, Any]:
        """Extract metadata from YAML frontmatter in markdown transcript."""
        if FormatReader.is_content(source):
            content = str(source)
        else:
            p = Path(source).expanduser().resolve()
            if p.exists() and p.is_file():
                content = p.read_text(encoding="utf-8")
            else:
                return {}
        return MarkdownReader.extract_frontmatter(content)

    @classmethod
    def read(cls, path: Pathlike, normalize_text: bool = True, **kwargs) -> List[Supervision]:
        """Read markdown transcript file."""
        supervisions = MarkdownReader.extract_for_alignment(path, **kwargs)
        if normalize_text:
            from ..parsers.text_parser import normalize_text as normalize_text_fn

            for sup in supervisions:
                sup.text = normalize_text_fn(sup.text)
        return supervisions

    @classmethod
    def write(
        cls,
        supervisions: List[Supervision],
        output_path: Pathlike,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Path:
        """Write markdown transcript file."""
        return MarkdownWriter.write(supervisions, output_path, metadata=metadata, **kwargs)

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> bytes:
        """Convert to markdown format bytes."""
        return MarkdownWriter.to_bytes(supervisions, metadata=metadata, **kwargs)


# Register "md" as an alias so get_reader("md") and Caption.read("file.md") work
_READERS["md"] = MarkdownFormat
_WRITERS["md"] = MarkdownFormat

__all__ = ["MarkdownFormat", "MarkdownReader", "MarkdownSegment", "MarkdownWriter"]
