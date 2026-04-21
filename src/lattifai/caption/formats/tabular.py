"""Tabular and plain text format handlers.

Handles: CSV, TSV, AUD (Audacity labels), TXT, JSON
"""

import csv
from io import StringIO
from pathlib import Path
from typing import List

from ..parsers.text_parser import normalize_text as normalize_text_fn
from ..parsers.text_parser import parse_speaker_text, parse_timestamp_text
from ..supervision import Supervision
from . import register_format
from .base import FormatHandler, ParseResult


@register_format("csv")
class CSVFormat(FormatHandler):
    """CSV (Comma-Separated Values) format.

    Format: speaker,start,end,text (with header)
    Times are in milliseconds.
    """

    extensions = [".csv"]
    description = "CSV - tabular subtitle format"

    @classmethod
    def parse(
        cls,
        source,
        normalize_text: bool = True,
        **kwargs,
    ) -> ParseResult:
        """Parse CSV format."""
        if cls.is_content(source):
            lines = list(csv.reader(StringIO(source)))
        else:
            with open(source, "r", encoding="utf-8", newline="") as f:
                lines = list(csv.reader(f))

        if not lines:
            return ParseResult(supervisions=[])

        # Check for header
        first_line = [col.strip().lower() for col in lines[0]]
        has_header = "start" in first_line and "end" in first_line and "text" in first_line
        has_speaker = "speaker" in first_line

        supervisions = []
        start_idx = 1 if has_header else 0

        for parts in lines[start_idx:]:
            if len(parts) < 3:
                continue
            try:
                if has_speaker and len(parts) >= 4:
                    speaker = parts[0].strip() or None
                    start = float(parts[1]) / 1000.0
                    end = float(parts[2]) / 1000.0
                    text = ",".join(parts[3:]).strip()
                else:
                    start = float(parts[0]) / 1000.0
                    end = float(parts[1]) / 1000.0
                    text = ",".join(parts[2:]).strip()
                    speaker = None

                if normalize_text:
                    text = normalize_text_fn(text)

                if end > start:
                    supervisions.append(Supervision(text=text, start=start, duration=end - start, speaker=speaker))
            except (ValueError, IndexError):
                continue

        return ParseResult(supervisions=supervisions)

    @classmethod
    def write(cls, supervisions: List[Supervision], output_path, **kwargs) -> Path:
        """Write CSV format."""
        output_path = Path(output_path)
        content = cls.to_bytes(supervisions, **kwargs)
        output_path.write_bytes(content)
        return output_path

    @classmethod
    def to_bytes(cls, supervisions: List[Supervision], **kwargs) -> bytes:
        """Convert to CSV format bytes."""
        render, include_speaker, _ = cls._unpack_render(**kwargs)
        has_translations = any(sup.translation for sup in supervisions)
        output = StringIO()
        writer = csv.writer(output)

        if include_speaker:
            header = ["speaker", "start", "end", "text"]
            if has_translations:
                header.append("translation")
            writer.writerow(header)
            for sup in supervisions:
                if cls._should_include_speaker(sup, include_speaker):
                    text = f"{sup.speaker} {sup.text.strip()}"
                else:
                    text = sup.text.strip()
                row = [sup.speaker or "", round(1000 * sup.start), round(1000 * sup.end), text]
                if has_translations:
                    row.append(sup.translation or "")
                writer.writerow(row)
        else:
            header = ["start", "end", "text"]
            if has_translations:
                header.append("translation")
            writer.writerow(header)
            for sup in supervisions:
                row = [round(1000 * sup.start), round(1000 * sup.end), sup.text.strip()]
                if has_translations:
                    row.append(sup.translation or "")
                writer.writerow(row)

        return output.getvalue().encode("utf-8")


@register_format("tsv")
class TSVFormat(FormatHandler):
    """TSV (Tab-Separated Values) format.

    Format: speaker\tstart\tend\ttext (with header)
    Times are in milliseconds.
    """

    extensions = [".tsv"]
    description = "TSV - tab-separated subtitle format"

    @classmethod
    def parse(cls, source, normalize_text: bool = True, **kwargs) -> ParseResult:
        """Parse TSV format."""
        if cls.is_content(source):
            lines = source.strip().split("\n")
        else:
            with open(source, "r", encoding="utf-8") as f:
                lines = f.readlines()

        if not lines:
            return ParseResult(supervisions=[])

        first_line = lines[0].strip().lower()
        has_header = "start" in first_line and "end" in first_line and "text" in first_line
        has_speaker = "speaker" in first_line

        supervisions = []
        start_idx = 1 if has_header else 0

        for line in lines[start_idx:]:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                continue

            try:
                if has_speaker and len(parts) >= 4:
                    speaker = parts[0].strip() or None
                    start = float(parts[1]) / 1000.0
                    end = float(parts[2]) / 1000.0
                    text = "\t".join(parts[3:]).strip()
                else:
                    start = float(parts[0]) / 1000.0
                    end = float(parts[1]) / 1000.0
                    text = "\t".join(parts[2:]).strip()
                    speaker = None

                if normalize_text:
                    text = normalize_text_fn(text)

                if end > start:
                    supervisions.append(Supervision(text=text, start=start, duration=end - start, speaker=speaker))
            except (ValueError, IndexError):
                continue

        return ParseResult(supervisions=supervisions)

    @classmethod
    def write(cls, supervisions: List[Supervision], output_path, **kwargs) -> Path:
        """Write TSV format."""
        output_path = Path(output_path)
        content = cls.to_bytes(supervisions, **kwargs)
        output_path.write_bytes(content)
        return output_path

    @classmethod
    def to_bytes(cls, supervisions: List[Supervision], **kwargs) -> bytes:
        """Convert to TSV format bytes."""
        render, include_speaker, _ = cls._unpack_render(**kwargs)
        has_translations = any(sup.translation for sup in supervisions)
        lines = []
        if include_speaker:
            header = "speaker\tstart\tend\ttext"
            if has_translations:
                header += "\ttranslation"
            lines.append(header)
            for sup in supervisions:
                speaker = sup.speaker if cls._should_include_speaker(sup, include_speaker) else ""
                text = sup.text.strip().replace("\t", " ")
                line = f"{speaker}\t{round(1000 * sup.start)}\t{round(1000 * sup.end)}\t{text}"
                if has_translations:
                    line += f"\t{(sup.translation or '').replace(chr(9), ' ')}"
                lines.append(line)
        else:
            header = "start\tend\ttext"
            if has_translations:
                header += "\ttranslation"
            lines.append(header)
            for sup in supervisions:
                text = sup.text.strip().replace("\t", " ")
                line = f"{round(1000 * sup.start)}\t{round(1000 * sup.end)}\t{text}"
                if has_translations:
                    line += f"\t{(sup.translation or '').replace(chr(9), ' ')}"
                lines.append(line)

        return "\n".join(lines).encode("utf-8")


@register_format("aud")
class AUDFormat(FormatHandler):
    """Audacity Labels format.

    Format: start\tend\t[[speaker]]text
    Times are in seconds.
    """

    extensions = [".aud", ".txt"]
    description = "Audacity Labels format"

    @classmethod
    def can_read(cls, path) -> bool:
        """Only handle .aud extension for reading."""
        return str(path).lower().endswith(".aud")

    @classmethod
    def parse(cls, source, normalize_text: bool = True, **kwargs) -> ParseResult:
        """Parse AUD format."""
        import re

        from ..parsers.text_parser import detect_speaker_candidates, set_speaker_candidates

        if cls.is_content(source):
            lines = source.strip().split("\n")
        else:
            with open(source, "r", encoding="utf-8") as f:
                lines = f.readlines()

        # Extract text column for candidate detection
        text_parts = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                text_parts.append("\t".join(parts[2:]).strip())

        candidates = detect_speaker_candidates(text_parts)
        if candidates:
            set_speaker_candidates(candidates)

        supervisions = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                continue

            try:
                start = float(parts[0])
                end = float(parts[1])
                text = "\t".join(parts[2:]).strip()

                # Extract speaker: try [[speaker]]text first, then Speaker: text
                speaker = None
                speaker_match = re.match(r"^\[\[([^\]]+)\]\]\s*(.*)$", text)
                if speaker_match:
                    speaker = speaker_match.group(1)
                    text = speaker_match.group(2)
                else:
                    speaker, text = parse_speaker_text(text)

                if normalize_text:
                    text = normalize_text_fn(text)

                if end > start:
                    supervisions.append(Supervision(text=text, start=start, duration=end - start, speaker=speaker))
            except (ValueError, IndexError):
                continue

        if candidates:
            set_speaker_candidates(set())

        return ParseResult(supervisions=supervisions)

    @classmethod
    def write(cls, supervisions: List[Supervision], output_path, **kwargs) -> Path:
        """Write AUD format."""
        output_path = Path(output_path)
        content = cls.to_bytes(supervisions, **kwargs)
        output_path.write_bytes(content)
        return output_path

    @classmethod
    def to_bytes(cls, supervisions: List[Supervision], **kwargs) -> bytes:
        """Convert to AUD format bytes."""
        from .base import SpeakerTracker

        render, include_speaker, _ = cls._unpack_render(**kwargs)
        lines = []
        tracker = SpeakerTracker()
        for sup in supervisions:
            text = sup.text.strip().replace("\t", " ")
            if cls._should_include_speaker(sup, include_speaker, tracker):
                text = f"{cls._format_speaker_prefix(sup.speaker)}{text}"
            lines.append(f"{sup.start}\t{sup.end}\t{text}")

        return "\n".join(lines).encode("utf-8")


@register_format("txt")
class TXTFormat(FormatHandler):
    """Plain text format with optional timestamps.

    Format: [start-end] text or [start-end] [speaker]: text
    """

    extensions = [".txt"]
    description = "Plain text with optional timestamps"

    @classmethod
    def parse(cls, source, normalize_text: bool = True, **kwargs) -> ParseResult:
        """Parse TXT format."""
        from ..parsers.text_parser import detect_speaker_candidates, set_speaker_candidates

        if cls.is_content(source):
            lines = source.strip().split("\n")
        else:
            with open(source, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines()]

        if normalize_text:
            lines = [normalize_text_fn(line) for line in lines]

        # Strip timestamps first, then detect speaker candidates on the text portion
        text_lines = []
        for line in lines:
            if not line:
                continue
            _, _, remaining = parse_timestamp_text(line)
            text_lines.append(remaining if remaining != line else line)

        candidates = detect_speaker_candidates(text_lines)
        if candidates:
            set_speaker_candidates(candidates)

        supervisions = []
        for line in lines:
            if not line:
                continue

            start, end, remaining_text = parse_timestamp_text(line)
            if start is not None and end is not None:
                speaker, text = parse_speaker_text(remaining_text)
                supervisions.append(Supervision(text=text, start=start, duration=end - start, speaker=speaker))
            else:
                speaker, text = parse_speaker_text(line)
                supervisions.append(Supervision(text=text, speaker=speaker))

        if candidates:
            set_speaker_candidates(set())

        return ParseResult(supervisions=supervisions)

    @classmethod
    def write(cls, supervisions: List[Supervision], output_path, **kwargs) -> Path:
        """Write TXT format."""
        output_path = Path(output_path)
        content = cls.to_bytes(supervisions, **kwargs)
        output_path.write_bytes(content)
        return output_path

    @classmethod
    def to_bytes(cls, supervisions: List[Supervision], **kwargs) -> bytes:
        """Convert to TXT format bytes."""
        from .base import SpeakerTracker

        render, include_speaker, _ = cls._unpack_render(**kwargs)
        lines = []
        tracker = SpeakerTracker()
        for sup in supervisions:
            text = sup.text or ""
            if cls._should_include_speaker(sup, include_speaker, tracker):
                text = f"{cls._format_speaker_prefix(sup.speaker)}{text}"
            lines.append(f"[{sup.start:.2f}-{sup.end:.2f}] {text}")

        return "\n".join(lines).encode("utf-8")
