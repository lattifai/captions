"""Podcast transcript format reader.

Parses human-generated podcast transcripts with speaker labels and timestamps.

Expected format (each block):

    Speaker Name
    [(HH:MM:SS)](youtube_url&t=seconds)
    Transcript text spanning one or more lines...

Supports transcripts from Lex Fridman and similar podcast sites where each
utterance is tagged with a speaker name and a clickable timestamp link.
"""

import re
from pathlib import Path
from typing import List, Union

from ..supervision import Pathlike, Supervision
from . import register_reader
from .base import FormatReader

# Matches: [(HH:MM:SS)](url&t=N) or [(HH:MM:SS)](url)
_TIMESTAMP_RE = re.compile(
    r"\[\((\d{1,2}:\d{2}:\d{2})\)\]"  # capture HH:MM:SS
    r"\([^)]+\)"  # link target: youtube URL or hash anchor (not captured)
)

# Quick content-sniff: speaker line followed by timestamp line
_SNIFF_RE = re.compile(
    r"^\s*[A-Z][a-z]+(?: [A-Z][a-z]+)+\s*$"  # capitalized multi-word name
    r".*?"
    r"\[\(\d{1,2}:\d{2}:\d{2}\)\]\(",
    re.MULTILINE | re.DOTALL,
)


def _parse_hms(hms: str) -> float:
    """Convert HH:MM:SS to seconds."""
    parts = hms.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return float(parts[0])


@register_reader("podcast-transcript")
class PodcastTranscriptReader(FormatReader):
    """Reader for podcast transcript pages with speaker labels and timestamps."""

    format_id = "podcast-transcript"
    extensions = [".md"]
    description = "Podcast transcript with speaker labels and clickable timestamps"

    @classmethod
    def can_read(cls, path: Union[Pathlike, str]) -> bool:
        """Detect podcast transcript by filename pattern or content sniffing."""
        path_str = str(path).lower()
        if not path_str.endswith(".md"):
            return False
        # Filename heuristics
        if "transcript" in path_str and "gemini" not in path_str:
            return True
        # Content sniff for small files
        try:
            p = Path(path)
            if p.exists() and p.stat().st_size < 10 * 1024 * 1024:
                head = p.read_text(encoding="utf-8")[:2000]
                return bool(_SNIFF_RE.search(head))
        except (OSError, ValueError):
            pass
        return False

    @classmethod
    def read(
        cls,
        source: Union[Pathlike, str],
        normalize_text: bool = True,
        **kwargs,
    ) -> List[Supervision]:
        """Parse podcast transcript into Supervision segments.

        Args:
            source: File path or raw text content.
            normalize_text: Unused (kept for interface compatibility).

        Returns:
            List of Supervision objects with speaker, timing, and text.
        """
        if cls.is_content(source):
            text = source
        else:
            text = Path(source).read_text(encoding="utf-8")

        lines = text.split("\n")
        raw_segments: list = []
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines, markdown headers, TOC links, YAML front matter
            if not line or line.startswith("#") or line.startswith("[") or line.startswith("---"):
                i += 1
                continue

            # Look ahead: is the next non-empty line a timestamp?
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1

            if j >= len(lines):
                i += 1
                continue

            ts_match = _TIMESTAMP_RE.search(lines[j].strip())
            if not ts_match:
                i += 1
                continue

            # Line i = speaker, line j = timestamp
            speaker = line
            start = _parse_hms(ts_match.group(1))

            # Collect text after timestamp until next speaker block
            text_lines = []
            after_ts = _TIMESTAMP_RE.sub("", lines[j]).strip()
            if after_ts:
                text_lines.append(after_ts)

            k = j + 1
            while k < len(lines):
                curr = lines[k].strip()

                if curr and not curr.startswith("#") and not curr.startswith("[") and not curr.startswith("---"):
                    # Peek ahead for timestamp → next speaker block
                    m = k + 1
                    while m < len(lines) and not lines[m].strip():
                        m += 1
                    if m < len(lines) and _TIMESTAMP_RE.search(lines[m].strip()):
                        break

                if curr:
                    # Stop at footer metadata
                    if curr.startswith("This entry was posted") or curr.startswith("View all posts"):
                        break
                    if curr.startswith("About ") and len(curr.split()) <= 4:
                        break
                    text_lines.append(curr)

                k += 1

            seg_text = " ".join(text_lines).strip()
            if seg_text:
                raw_segments.append({"speaker": speaker, "start": start, "text": seg_text})

            i = k

        # Build Supervisions with end times inferred from next segment
        supervisions: List[Supervision] = []
        for idx, seg in enumerate(raw_segments):
            if idx + 1 < len(raw_segments):
                end = raw_segments[idx + 1]["start"]
            else:
                end = seg["start"] + 30.0
            duration = max(end - seg["start"], 0.0)
            supervisions.append(
                Supervision(
                    text=seg["text"],
                    start=seg["start"],
                    duration=duration,
                    speaker=seg["speaker"],
                )
            )

        return supervisions
