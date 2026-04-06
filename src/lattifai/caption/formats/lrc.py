"""Enhanced LRC format handler.

LRC (LyRiCs) is a file format for synchronized song lyrics. Enhanced LRC
adds word-level timestamps for karaoke applications.

Standard LRC:
    [00:15.20]Hello beautiful world

Enhanced LRC (word-level):
    [00:15.20]<00:15.20>Hello <00:15.65>beautiful <00:16.40>world

Metadata tags:
    [ar:Artist Name]
    [ti:Song Title]
    [al:Album Name]
    [offset:±milliseconds]
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..config import LRCConfig
from ..supervision import AlignmentItem, Pathlike, Supervision
from . import register_format
from .base import FormatHandler


@register_format("lrc")
class LRCFormat(FormatHandler):
    """Enhanced LRC format with word-level timing support."""

    extensions = [".lrc"]
    description = "Enhanced LRC - karaoke lyrics format"

    @classmethod
    def is_content(cls, source) -> bool:
        """Check if source is LRC content rather than a file path.

        Overrides base class to also detect LRC content by timestamp pattern.
        """
        if not isinstance(source, str):
            return False
        # If it has newlines or is very long, it's likely content
        if "\n" in source or len(source) > 500:
            return True
        # LRC-specific: check for timestamp pattern at start
        if source.strip().startswith("[") and re.match(r"\[\d+:\d+", source):
            return True
        return False

    @classmethod
    def extract_metadata(cls, source: Union[Pathlike, str], **kwargs) -> Dict[str, str]:
        """Extract LRC metadata tags.

        Extracts standard LRC metadata:
        - ar: Artist name
        - ti: Title
        - al: Album
        - by: Creator
        - offset: Time offset in milliseconds
        - length: Song length

        Returns:
            Dict with lrc_* prefixed keys for metadata preservation
        """
        if cls.is_content(source):
            content = source
        else:
            try:
                content = Path(str(source)).read_text(encoding="utf-8")
            except Exception:
                return {}

        metadata = {}
        # Pattern to match [key:value] metadata tags
        meta_pattern = re.compile(r"^\[([a-z]+):(.+)\]$", re.IGNORECASE)

        for line in content.split("\n")[:50]:  # Only check first 50 lines
            line = line.strip()
            match = meta_pattern.match(line)
            if match:
                key, value = match.groups()
                key = key.lower()
                # Store with lrc_ prefix to avoid conflicts
                if key in ("ar", "ti", "al", "by", "offset", "length", "re", "ve"):
                    metadata[f"lrc_{key}"] = value.strip()

        return metadata

    @classmethod
    def read(
        cls,
        source,
        normalize_text: bool = True,
        **kwargs,
    ) -> List[Supervision]:
        """Read LRC file and return list of Supervision objects.

        Parses both standard LRC and enhanced LRC with word-level timestamps.

        Args:
            source: File path or string content
            normalize_text: Whether to normalize text (currently unused)
            **kwargs: Additional options

        Returns:
            List of Supervision objects with timing and optional word alignment
        """
        if cls.is_content(source):
            content = source
        else:
            content = Path(source).read_text(encoding="utf-8")

        supervisions = []
        # Match line timestamp: [mm:ss.xx] or [mm:ss.xxx]
        line_pattern = re.compile(r"\[(\d+):(\d+)\.(\d+)\](.+)")
        # Match word timestamp: <mm:ss.xx> or <mm:ss.xxx>
        word_pattern = re.compile(r"<(\d+):(\d+)\.(\d+)>([^<]+)")

        for line in content.split("\n"):
            line = line.strip()
            # Skip empty lines and metadata
            if not line or line.startswith("[ar:") or line.startswith("[ti:"):
                continue
            if line.startswith("[al:") or line.startswith("[offset:"):
                continue
            if line.startswith("[by:") or line.startswith("[length:"):
                continue

            match = line_pattern.match(line)
            if match:
                mins, secs, frac, text = match.groups()
                # Handle centisecond vs millisecond
                if len(frac) == 2:
                    start = int(mins) * 60 + int(secs) + int(frac) / 100
                else:
                    start = int(mins) * 60 + int(secs) + int(frac) / 1000

                # Extract word-level alignment
                words = word_pattern.findall(text)
                alignment = None
                if words:
                    alignment = {"word": []}
                    for w_mins, w_secs, w_frac, w_text in words:
                        if len(w_frac) == 2:
                            w_start = int(w_mins) * 60 + int(w_secs) + int(w_frac) / 100
                        else:
                            w_start = int(w_mins) * 60 + int(w_secs) + int(w_frac) / 1000
                        alignment["word"].append(
                            AlignmentItem(
                                symbol=w_text.strip(),
                                start=w_start,
                                duration=0,  # LRC doesn't store duration
                            )
                        )
                    # Clean text (remove timestamp tags)
                    text = re.sub(r"<\d+:\d+\.\d+>", "", text)

                supervisions.append(
                    Supervision(
                        text=text.strip(),
                        start=start,
                        duration=0,  # Will calculate below
                        alignment=alignment,
                    )
                )

        # Calculate duration from next segment
        for i, sup in enumerate(supervisions):
            if i + 1 < len(supervisions):
                sup.duration = supervisions[i + 1].start - sup.start
            else:
                sup.duration = 5.0  # Default 5 seconds for last line

        return supervisions

    @classmethod
    def write(
        cls,
        supervisions: List[Supervision],
        output_path,
        **kwargs,
    ) -> Path:
        """Write supervisions to LRC file.

        Args:
            supervisions: List of Supervision objects to write
            output_path: Path to output file
            **kwargs: Additional options (config=LRCConfig, render=RenderConfig)

        Returns:
            Path to the written file
        """
        output_path = Path(output_path)
        content = cls.to_bytes(
            supervisions,
            **kwargs,
        )
        output_path.write_bytes(content)
        return output_path

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        metadata: Optional[Dict] = None,
        config: Optional[LRCConfig] = None,
        **kwargs,
    ) -> bytes:
        """Convert supervisions to LRC format bytes.

        Args:
            supervisions: List of Supervision objects
            metadata: Optional metadata dict containing lrc_* keys to restore
            config: LRC format configuration (precision, metadata)

        Returns:
            Caption content as bytes
        """
        behavior, include_speaker, word_level = cls._unpack_render(**kwargs)
        lrc_config = config if isinstance(config, LRCConfig) else LRCConfig()
        precision = lrc_config.precision
        lines = []

        # Restore metadata from Caption.metadata (lrc_* keys)
        if metadata:
            lrc_meta_keys = ["ar", "ti", "al", "by", "offset", "length", "re", "ve"]
            for key in lrc_meta_keys:
                value = metadata.get(f"lrc_{key}")
                if value:
                    lines.append(f"[{key}:{value}]")

        # Also add LRCConfig metadata
        for key, value in lrc_config.metadata.items():
            existing_line = f"[{key}:"
            if not any(line.startswith(existing_line) for line in lines):
                lines.append(f"[{key}:{value}]")

        if lines:
            lines.append("")

        for sup in supervisions:
            if word_level and sup.alignment and "word" in sup.alignment:
                # Enhanced LRC mode: each word has inline timestamp
                word_items = sup.alignment["word"]
                line_time = cls._format_time(word_items[0].start, precision)
                word_parts = []
                for word in word_items:
                    word_time = cls._format_time(word.start, precision)
                    word_parts.append(f"<{word_time}>{word.symbol}")
                lines.append(f"[{line_time}]{' '.join(word_parts)}")
            else:
                # Standard LRC mode: only line timestamp
                from .base import render_bilingual_text

                line_time = cls._format_time(sup.start, precision)
                text = render_bilingual_text(sup, translation_first=behavior.translation_first)
                if cls._should_include_speaker(sup, include_speaker):
                    text = f"{cls._format_speaker_prefix(sup.speaker)}{text}"
                lines.append(f"[{line_time}]{text}")

        return "\n".join(lines).encode("utf-8")

    @staticmethod
    def _format_time(seconds: float, precision: str) -> str:
        """Format time for LRC timestamp.

        Args:
            seconds: Time in seconds
            precision: "centisecond" for [mm:ss.xx] or "millisecond" for [mm:ss.xxx]

        Returns:
            Formatted time string
        """
        if seconds < 0:
            seconds = 0
        minutes = int(seconds // 60)
        secs = seconds % 60
        if precision == "millisecond":
            return f"{minutes:02d}:{secs:06.3f}"  # 00:15.200
        return f"{minutes:02d}:{secs:05.2f}"  # 00:15.23


__all__ = ["LRCFormat"]
