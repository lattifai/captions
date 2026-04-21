"""WebVTT format with full W3C standard compliance.

Supports:
- Cue settings: align, line, position, size, vertical, region
- Inline tags: <v> voice, <b> <i> <u> formatting, <c> class, <ruby>, <lang>
- Regions: REGION blocks with scroll positioning
- Styles: STYLE blocks with ::cue pseudo-elements
- YouTube VTT: word-level timestamps (<timestamp><c> tags)
- Metadata: Kind, Language, multi-line NOTE comments
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..parsers.text_parser import normalize_text as normalize_text_fn
from ..parsers.text_parser import parse_speaker_text
from ..supervision import AlignmentItem, Supervision
from . import register_format
from .base import FormatHandler, ParseResult


@register_format("vtt")
class VTTFormat(FormatHandler):
    """WebVTT format handler with full W3C standard compliance.

    Reading:
        - Auto-detects YouTube VTT format (with word-level timestamps)
        - Parses cue settings (align, line, position, size, vertical, region)
        - Extracts <v> voice tags as speaker
        - Preserves inline formatting tags (<b>, <i>, <u>, <c>, <ruby>, <lang>)
        - Parses REGION blocks, STYLE blocks, multi-line NOTE comments

    Writing:
        - Standard VTT by default with cue settings from Supervision.custom
        - YouTube VTT style when word_level=True
        - Region and STYLE block roundtrip
        - Optional <v> voice tags via VTTConfig.voice_tag
    """

    extensions = [".vtt"]
    description = "Web Video Text Tracks - W3C standard with full feature support"

    # -- Class-level regex patterns --

    # YouTube VTT word-level timestamp detection
    YOUTUBE_VTT_PATTERN = re.compile(r"<\d{2}:\d{2}:\d{2}[.,]\d{3}><c>")

    # Speaker change marker in YouTube auto-generated VTTs
    SPEAKER_CHANGE_RE = re.compile(r"^(?:>>|&gt;&gt;)\s*")

    # Timestamp line: HH:MM:SS.mmm --> HH:MM:SS.mmm [settings]
    _TIMESTAMP_RE = re.compile(
        r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})(.*)"
    )

    # Cue setting key:value pairs
    _CUE_SETTING_RE = re.compile(r"(vertical|line|position|size|align|region):(\S+)")

    # Voice tag patterns
    _VOICE_OPEN_RE = re.compile(r"<v\s+([^>]+)>")
    _VOICE_CLOSE_RE = re.compile(r"</v>")

    # Font-wrapped speaker pattern: <font color="...">Speaker: </font>text
    _FONT_SPEAKER_RE = re.compile(r'^<font\s+color="[^"]*">([^<]+?):\s*</font>\s*(.*)', re.DOTALL)

    # Word-level timestamp patterns (YouTube VTT)
    _WORD_TS_RE = re.compile(r"<(\d{2}:\d{2}:\d{2}[.,]\d{3})><c>\s*([^<]+)</c>")
    _FIRST_WORD_RE = re.compile(r"^([^<\n]+?)<(\d{2}:\d{2}:\d{2}[.,]\d{3})>")

    # =========================================================================
    # Shared utilities
    # =========================================================================

    @staticmethod
    def _parse_timestamp(ts: str) -> float:
        """Convert VTT timestamp string to seconds."""
        ts = ts.replace(",", ".")
        parts = ts.split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format seconds into HH:MM:SS.mmm.

        Rounds total milliseconds first, then decomposes with divmod
        to correctly cascade carry across second/minute/hour boundaries.
        """
        total_ms = round(seconds * 1000)
        ms = total_ms % 1000
        total_s = total_ms // 1000
        s = total_s % 60
        total_m = total_s // 60
        m = total_m % 60
        h = total_m // 60
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    @classmethod
    def _parse_cue_settings(cls, settings_str: str) -> Dict[str, str]:
        """Parse cue settings from the remainder of a timestamp line."""
        settings = {}
        for match in cls._CUE_SETTING_RE.finditer(settings_str):
            settings[match.group(1)] = match.group(2)
        return settings

    @classmethod
    def _extract_voice_tag(cls, text: str) -> Tuple[Optional[str], str]:
        """Extract <v name> voice tag from cue text.

        Returns (speaker, cleaned_text). Handles both closed (<v name>text</v>)
        and unclosed (<v name>text) forms.
        """
        voice_match = cls._VOICE_OPEN_RE.search(text)
        if not voice_match:
            return None, text
        speaker = voice_match.group(1).strip()
        # Remove voice tags from text, keep inner content
        text = cls._VOICE_OPEN_RE.sub("", text)
        text = cls._VOICE_CLOSE_RE.sub("", text)
        return speaker, text.strip()

    # =========================================================================
    # Reader: can_read / detection
    # =========================================================================

    @classmethod
    def can_read(cls, source) -> bool:
        """Check if source is a VTT file."""
        if cls.is_content(source):
            return source.strip().startswith("WEBVTT")
        try:
            return str(source).lower().endswith(".vtt")
        except Exception:
            return False

    @classmethod
    def _is_youtube_vtt(cls, content: str) -> bool:
        """Check if content is YouTube VTT format with word-level timestamps."""
        return bool(cls.YOUTUBE_VTT_PATTERN.search(content))

    # =========================================================================
    # Reader helpers
    # =========================================================================

    @classmethod
    def _parse_supervisions(cls, content: str, normalize_text: bool) -> List[Supervision]:
        """Dispatch to the YouTube or standard VTT supervision extractor."""
        if cls._is_youtube_vtt(content):
            return cls._read_youtube_vtt(content, normalize_text)
        return cls._read_standard_vtt(content, normalize_text)

    # =========================================================================
    # Reader: standard VTT (custom parser, replaces pysubs2)
    # =========================================================================

    @classmethod
    def _read_standard_vtt(cls, content: str, normalize_text: bool = True) -> List[Supervision]:
        """Read standard VTT with custom parser.

        Extracts cue settings, voice tags, and preserves inline formatting
        tags (<b>, <i>, <u>, <c>, <ruby>, <lang>).
        """
        from ..parsers.text_parser import detect_speaker_candidates, set_speaker_candidates

        # Split into blocks separated by blank lines
        blocks = re.split(r"\n\s*\n", content)
        if not blocks:
            return []

        # Collect cues
        raw_texts: List[str] = []
        cues: List[Dict[str, Any]] = []

        for block in blocks:
            block = block.strip()
            if not block:
                continue
            # Skip header / REGION / STYLE / NOTE blocks
            if block.startswith(("WEBVTT", "REGION", "STYLE", "NOTE")):
                continue

            block_lines = block.split("\n")

            # Find the timestamp line within this block
            ts_line_idx = None
            for li, line in enumerate(block_lines):
                if cls._TIMESTAMP_RE.search(line):
                    ts_line_idx = li
                    break
            if ts_line_idx is None:
                continue

            ts_match = cls._TIMESTAMP_RE.match(block_lines[ts_line_idx].strip())
            if not ts_match:
                continue

            cue_start = cls._parse_timestamp(ts_match.group(1))
            cue_end = cls._parse_timestamp(ts_match.group(2))
            cue_settings = cls._parse_cue_settings(ts_match.group(3))

            # Cue ID is any line before the timestamp line
            cue_id = block_lines[ts_line_idx - 1].strip() if ts_line_idx > 0 else None

            # Cue text is everything after the timestamp line
            text_lines = [l for l in block_lines[ts_line_idx + 1 :] if l.strip()]
            raw_text = "\n".join(text_lines)

            raw_texts.append(raw_text)
            cues.append(
                {
                    "start": cue_start,
                    "end": cue_end,
                    "settings": cue_settings,
                    "id": cue_id,
                    "raw_text": raw_text,
                }
            )

        # Auto-detect title-case speaker candidates from raw text
        candidates = detect_speaker_candidates(raw_texts)
        if candidates:
            set_speaker_candidates(candidates)

        supervisions = []
        for cue in cues:
            text = cue["raw_text"]

            # Extract <v> voice tag as speaker
            voice_speaker, text = cls._extract_voice_tag(text)

            # Normalize: convert newlines to space, then apply text normalization
            if normalize_text:
                text = text.replace("\n", " ")
                text = normalize_text_fn(text)

            # Fall back to text-based speaker detection if no voice tag
            if voice_speaker:
                speaker = voice_speaker
            else:
                # Try font-wrapped speaker: <font color="...">Name: </font>text
                font_match = cls._FONT_SPEAKER_RE.match(text)
                if font_match:
                    speaker = font_match.group(1).strip()
                    text = font_match.group(2).strip()
                else:
                    speaker, text = parse_speaker_text(text)

            # Build custom fields for cue settings
            custom: Optional[Dict[str, Any]] = None
            if cue["settings"]:
                custom = {f"vtt_{k}": v for k, v in cue["settings"].items()}

            supervisions.append(
                Supervision(
                    text=text.strip(),
                    speaker=speaker or None,
                    start=cue["start"],
                    duration=max(0.0, cue["end"] - cue["start"]),
                    id=cue["id"] or "",
                    custom=custom,
                )
            )

        if candidates:
            set_speaker_candidates(set())

        return supervisions

    # =========================================================================
    # Reader: YouTube VTT (word-level timestamps)
    # =========================================================================

    @classmethod
    def _read_youtube_vtt(cls, content: str, normalize_text: bool = True) -> List[Supervision]:
        """Parse YouTube VTT format with word-level timestamps.

        Extended to capture cue settings from timestamp lines.
        """
        supervisions = []

        def has_word_timestamps(text: str) -> bool:
            return bool(cls._WORD_TS_RE.search(text) or cls._FIRST_WORD_RE.match(text))

        lines = content.split("\n")
        i = 0

        # First pass: collect all cues with their content
        all_cues: List[Dict[str, Any]] = []
        while i < len(lines):
            line = lines[i]
            ts_match = cls._TIMESTAMP_RE.search(line)
            if ts_match:
                cue_start = cls._parse_timestamp(ts_match.group(1))
                cue_end = cls._parse_timestamp(ts_match.group(2))
                cue_settings = cls._parse_cue_settings(ts_match.group(3))

                cue_lines: List[str] = []
                i += 1
                while i < len(lines):
                    if cls._TIMESTAMP_RE.search(lines[i]):
                        break
                    stripped = lines[i].strip()
                    if not stripped and cue_lines and not lines[i - 1].strip():
                        break
                    if stripped:
                        cue_lines.append(lines[i])
                    i += 1

                all_cues.append(
                    {
                        "start": cue_start,
                        "end": cue_end,
                        "lines": cue_lines,
                        "settings": cue_settings,
                    }
                )
                continue
            i += 1

        # Second pass: identify cues to skip and merge
        cues_to_skip: set = set()
        cues_to_merge_text: Dict[int, str] = {}

        for idx in range(len(all_cues) - 1):
            cue = all_cues[idx]
            duration = cue["end"] - cue["start"]

            if abs(duration - 0.010) < 0.001:
                cue_text = "\n".join(cue["lines"])
                if not has_word_timestamps(cue_text):
                    next_cue = all_cues[idx + 1]
                    if abs(next_cue["start"] - cue["end"]) < 0.001:
                        cues_to_skip.add(idx)

                        next_cue_text = "\n".join(next_cue["lines"])
                        if not has_word_timestamps(next_cue_text):
                            for prev_idx in range(idx - 1, -1, -1):
                                if prev_idx not in cues_to_skip:
                                    if len(next_cue["lines"]) > 1:
                                        append_text = cls.SPEAKER_CHANGE_RE.sub(
                                            "", next_cue["lines"][-1].strip()
                                        ).strip()
                                        if append_text:
                                            cues_to_merge_text[prev_idx] = append_text
                                    cues_to_skip.add(idx + 1)
                                    break

        # Third pass: process remaining cues
        for idx, cue in enumerate(all_cues):
            if idx in cues_to_skip:
                continue

            cue_start = cue["start"]
            cue_end = cue["end"]
            cue_lines = cue["lines"]
            cue_settings = cue["settings"]

            word_alignments: List[AlignmentItem] = []
            text_parts: List[str] = []
            has_speaker_change = False

            for cue_line in cue_lines:
                cue_line = cue_line.strip()
                if not cue_line:
                    continue

                # Only process NEW lines (those carrying inline word-level
                # timestamps). OLD rolling lines are discarded here — they
                # duplicate words already captured in the previous cue and
                # carry stale >> markers that would falsely flag speaker
                # changes on every subsequent cue.
                if not has_word_timestamps(cue_line):
                    continue

                # Detect and strip >> speaker change markers
                stripped_line = cls.SPEAKER_CHANGE_RE.sub("", cue_line, count=1)
                if stripped_line != cue_line:
                    has_speaker_change = True
                    cue_line = stripped_line.strip()
                    if not cue_line:
                        continue

                word_matches = cls._WORD_TS_RE.findall(cue_line)
                first_match = cls._FIRST_WORD_RE.match(cue_line)

                if word_matches or first_match:
                    if first_match:
                        first_word = first_match.group(1).strip()
                        first_word_next_ts = cls._parse_timestamp(first_match.group(2))
                        if first_word:
                            text_parts.append(first_word)
                            word_alignments.append(
                                AlignmentItem(
                                    symbol=first_word,
                                    start=cue_start,
                                    duration=max(0.01, first_word_next_ts - cue_start),
                                )
                            )

                    for word_idx, (ts, word) in enumerate(word_matches):
                        word_start = cls._parse_timestamp(ts)
                        word = word.strip()
                        if not word:
                            continue

                        text_parts.append(word)

                        if word_idx + 1 < len(word_matches):
                            next_ts = cls._parse_timestamp(word_matches[word_idx + 1][0])
                            duration = next_ts - word_start
                        else:
                            duration = cue_end - word_start

                        word_alignments.append(
                            AlignmentItem(
                                symbol=word,
                                start=word_start,
                                duration=max(0.01, duration),
                            )
                        )

            if not text_parts:
                continue

            full_text = " ".join(text_parts)
            if idx in cues_to_merge_text:
                full_text += " " + cues_to_merge_text[idx]

            if normalize_text:
                full_text = normalize_text_fn(full_text)

            if word_alignments:
                sup_start = word_alignments[0].start
                sup_end = word_alignments[-1].start + word_alignments[-1].duration
            else:
                sup_start = cue_start
                sup_end = cue_end

            # Build custom fields for cue settings
            custom: Optional[Dict[str, Any]] = None
            if cue_settings:
                custom = {f"vtt_{k}": v for k, v in cue_settings.items()}

            supervisions.append(
                Supervision(
                    text=full_text,
                    start=sup_start,
                    duration=max(0.0, sup_end - sup_start),
                    alignment={"word": word_alignments} if word_alignments else None,
                    speaker=">>" if has_speaker_change else None,
                    custom=custom,
                )
            )

        return supervisions

    # =========================================================================
    # Header / metadata parsing
    # =========================================================================

    @classmethod
    def _extract_header_metadata(cls, content: str) -> Dict[str, Any]:
        """Extract all VTT metadata: Kind, Language, NOTE, REGION, STYLE blocks."""
        metadata: Dict[str, Any] = {}
        regions: List[Dict[str, str]] = []
        styles: List[str] = []
        notes: List[str] = []

        # Split into blocks separated by blank lines
        blocks = re.split(r"\n\s*\n", content)

        for block in blocks:
            block_stripped = block.strip()
            if not block_stripped:
                continue

            # Header block (WEBVTT line + metadata)
            if block_stripped.startswith("WEBVTT"):
                for line in block_stripped.split("\n"):
                    line = line.strip()
                    if line.startswith("Kind:"):
                        metadata["kind"] = line.split(":", 1)[1].strip()
                    elif line.startswith("Language:"):
                        metadata["language"] = line.split(":", 1)[1].strip()
                continue

            # REGION block
            if block_stripped.startswith("REGION"):
                region: Dict[str, str] = {}
                for line in block_stripped.split("\n"):
                    line = line.strip()
                    if line == "REGION":
                        continue
                    if ":" in line:
                        key, _, value = line.partition(":")
                        region[key.strip()] = value.strip()
                if region:
                    regions.append(region)
                continue

            # STYLE block — store everything after "STYLE\n"
            if block_stripped.startswith("STYLE"):
                style_content = block_stripped[len("STYLE") :].strip()
                if style_content:
                    styles.append(style_content)
                continue

            # NOTE block (multi-line support)
            if block_stripped.startswith("NOTE"):
                note_content = block_stripped[len("NOTE") :].strip()
                if note_content:
                    # Also extract key:value from single-line notes for backward compat
                    kv_match = re.match(r"(\w+):\s*(.+)", note_content)
                    if kv_match and "\n" not in note_content:
                        metadata[kv_match.group(1).lower()] = kv_match.group(2).strip()
                    notes.append(note_content)
                continue

            # Once we hit a cue block (contains -->), stop metadata scan
            if "-->" in block_stripped:
                break

        if regions:
            metadata["vtt_regions"] = regions
        if styles:
            metadata["vtt_styles"] = styles
        if notes:
            metadata["vtt_notes"] = notes

        return metadata

    @classmethod
    def parse(cls, source, normalize_text: bool = True, **kwargs) -> ParseResult:
        """Parse VTT (YouTube or standard) in a single pass.

        Auto-detects YouTube VTT word-level timestamps vs standard VTT and
        returns a :class:`ParseResult` with supervisions plus header-level
        ``language`` / ``kind`` / format metadata.
        """
        if cls.is_content(source):
            content = source
        else:
            with open(source, "r", encoding="utf-8") as f:
                content = f.read()

        supervisions = cls._parse_supervisions(content, normalize_text=normalize_text)
        header_meta = cls._extract_header_metadata(content)

        return ParseResult(
            supervisions=supervisions,
            language=header_meta.pop("language", None),
            kind=header_meta.pop("kind", None),
            format_metadata=header_meta,
        )

    # =========================================================================
    # Writer: entry points
    # =========================================================================

    @classmethod
    def write(cls, supervisions: List[Supervision], output_path, **kwargs) -> Path:
        """Write VTT to file."""
        output_path = Path(output_path)
        content = cls.to_bytes(supervisions, **kwargs)
        output_path.write_bytes(content)
        return output_path

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        metadata: Optional[Dict] = None,
        config=None,
        **kwargs,
    ) -> bytes:
        """Convert to VTT bytes with optional word-level and metadata preservation.

        Args:
            supervisions: List of supervision segments
            metadata: Optional metadata dict (kind, language, vtt_regions, vtt_styles, etc.)
            config: Optional VTTConfig for output control

        Returns:
            VTT content as bytes
        """
        from .base import count_supervisions_with_words, resolve_word_level

        behavior, include_speaker, word_level = cls._unpack_render(**kwargs)
        tf = behavior.translation_first

        # Resolve VTTConfig
        from ..config import VTTConfig

        if config is None:
            config = kwargs.pop("config", None)
        if config is not None and not isinstance(config, VTTConfig):
            config = None
        vtt_config = config or VTTConfig()

        n_with = count_supervisions_with_words(supervisions)
        use_word = resolve_word_level(
            word_level,
            n_with_words=n_with,
            n_total=len(supervisions),
            format_id="vtt",
            smart_default=False,
        )

        # Priority: VTTConfig.speaker_color > RenderConfig.speaker_color
        speaker_color = vtt_config.speaker_color
        if not speaker_color and behavior.speaker_color:
            speaker_color = behavior.speaker_color

        if use_word:
            return cls._to_youtube_vtt_bytes(supervisions, include_speaker, metadata, tf, vtt_config, speaker_color)
        return cls._to_vtt_bytes_with_metadata(supervisions, include_speaker, metadata, tf, vtt_config, speaker_color)

    # =========================================================================
    # Writer: header blocks (REGION, STYLE, NOTE)
    # =========================================================================

    @classmethod
    def _write_header(cls, metadata: Optional[Dict], lines: List[str]) -> None:
        """Write WEBVTT header with Kind, Language, REGION, STYLE, NOTE blocks."""
        lines.append("WEBVTT")

        if metadata:
            if metadata.get("kind"):
                lines.append(f"Kind: {metadata['kind']}")
            if metadata.get("language"):
                lines.append(f"Language: {metadata['language']}")

        # REGION blocks
        if metadata and metadata.get("vtt_regions"):
            for region in metadata["vtt_regions"]:
                lines.append("")
                lines.append("REGION")
                for key, value in region.items():
                    lines.append(f"{key}:{value}")

        # STYLE blocks (roundtrip from vtt_styles OR generate from ass_styles)
        if metadata and metadata.get("vtt_styles"):
            for style in metadata["vtt_styles"]:
                lines.append("")
                lines.append("STYLE")
                lines.append(style)
        elif metadata and metadata.get("ass_styles"):
            style_lines = cls._build_vtt_style_block(metadata)
            if style_lines:
                lines.append("")
                lines.extend(style_lines)

        # NOTE blocks
        if metadata and metadata.get("vtt_notes"):
            for note in metadata["vtt_notes"]:
                lines.append("")
                if "\n" in note:
                    lines.append(f"NOTE\n{note}")
                else:
                    lines.append(f"NOTE {note}")

        lines.append("")

    # =========================================================================
    # Writer: cue settings
    # =========================================================================

    @classmethod
    def _format_cue_settings(cls, sup: Supervision, vtt_config: Any) -> str:
        """Build cue settings string from Supervision.custom and VTTConfig defaults.

        Returns empty string or ' key:val key:val ...' (leading space included).
        """
        settings: Dict[str, str] = {}

        # Apply defaults from VTTConfig
        if vtt_config.default_vertical:
            settings["vertical"] = vtt_config.default_vertical
        if vtt_config.default_line:
            settings["line"] = vtt_config.default_line
        if vtt_config.default_position:
            settings["position"] = vtt_config.default_position
        if vtt_config.default_size:
            settings["size"] = vtt_config.default_size
        if vtt_config.default_align:
            settings["align"] = vtt_config.default_align

        # Override with per-supervision custom fields (vtt_align, vtt_line, etc.)
        custom = getattr(sup, "custom", None)
        if custom:
            for key in ("vertical", "line", "position", "size", "align", "region"):
                vtt_key = f"vtt_{key}"
                if vtt_key in custom:
                    settings[key] = custom[vtt_key]

        if not settings:
            return ""

        # Assemble in W3C spec order
        parts = []
        for key in ("vertical", "line", "position", "size", "align", "region"):
            if key in settings:
                parts.append(f"{key}:{settings[key]}")
        return " " + " ".join(parts)

    # =========================================================================
    # Writer: standard VTT (custom generator, replaces pysubs2)
    # =========================================================================

    @classmethod
    def _to_vtt_bytes_with_metadata(
        cls,
        supervisions: List[Supervision],
        include_speaker: bool = True,
        metadata: Optional[Dict] = None,
        translation_first: bool = False,
        vtt_config: Any = None,
        speaker_color: str = "",
    ) -> bytes:
        """Generate standard VTT with full feature support."""
        from ..colors import resolve_speaker_color_rgb
        from ..config import VTTConfig
        from .base import SpeakerTracker, render_bilingual_text

        vtt_config = vtt_config or VTTConfig()
        lines: List[str] = []
        color_cache: Dict[str, str] = {}

        cls._write_header(metadata, lines)

        # Skip speaker dedup when voice_tag or speaker_color is active —
        # each cue needs its own speaker marker for roundtrip fidelity
        skip_dedup = vtt_config.voice_tag or bool(speaker_color)
        tracker = None if skip_dedup else SpeakerTracker()
        for idx, sup in enumerate(supervisions, 1):
            text = render_bilingual_text(sup, translation_first=translation_first)

            # Speaker: voice tag or prefix
            show_speaker = (
                cls._should_include_speaker(sup, include_speaker, tracker)
                if tracker is not None
                else (include_speaker and bool(getattr(sup, "speaker", None)))
            )
            if show_speaker:
                if vtt_config.voice_tag:
                    text = f"<v {sup.speaker}>{text}</v>"
                else:
                    prefix = cls._format_speaker_prefix(sup.speaker)
                    if speaker_color and sup.speaker:
                        color = resolve_speaker_color_rgb(sup.speaker, speaker_color, color_cache)
                        if color:
                            prefix = f'<font color="{color}">{prefix}</font>'
                    text = f"{prefix}{text}"

            # Timestamp line with cue settings
            ts_line = (
                f"{cls._format_timestamp(sup.start)} --> "
                f"{cls._format_timestamp(sup.end)}"
                f"{cls._format_cue_settings(sup, vtt_config)}"
            )

            lines.append(str(idx))
            lines.append(ts_line)
            lines.append(text)
            lines.append("")

        return "\n".join(lines).encode("utf-8")

    # =========================================================================
    # Writer: YouTube VTT (word-level timestamps)
    # =========================================================================

    @classmethod
    def _to_youtube_vtt_bytes(
        cls,
        supervisions: List[Supervision],
        include_speaker: bool = True,
        metadata: Optional[Dict] = None,
        translation_first: bool = False,
        vtt_config: Any = None,
        speaker_color: str = "",
    ) -> bytes:
        """Generate YouTube VTT format with word-level timestamps.

        Format: <00:00:10.559><c> word</c>
        """
        from ..colors import resolve_speaker_color_rgb
        from ..config import VTTConfig
        from .base import SpeakerTracker, render_bilingual_text

        vtt_config = vtt_config or VTTConfig()
        lines: List[str] = []
        color_cache: Dict[str, str] = {}

        cls._write_header(metadata, lines)

        tracker = SpeakerTracker()
        for sup in sorted(supervisions, key=lambda x: x.start):
            text = sup.text or ""
            alignment = getattr(sup, "alignment", None)
            words = alignment.get("word") if alignment else None

            cue_settings_str = cls._format_cue_settings(sup, vtt_config)
            show_speaker = cls._should_include_speaker(sup, include_speaker, tracker)

            if words:
                cue_start = words[0].start
                cue_end = words[-1].end
                lines.append(
                    f"{cls._format_timestamp(cue_start)} --> "
                    f"{cls._format_timestamp(cue_end)}{cue_settings_str}"
                )

                text_parts: List[str] = []
                if show_speaker and sup.speaker == ">>":
                    text_parts.append(">> ")
                for i, word in enumerate(words):
                    symbol = word.symbol
                    if i == 0 and show_speaker and sup.speaker and sup.speaker != ">>":
                        prefix = f"{sup.speaker}: "
                        if speaker_color:
                            color = resolve_speaker_color_rgb(sup.speaker, speaker_color, color_cache)
                            if color:
                                prefix = f'<font color="{color}">{prefix}</font>'
                        symbol = f"{prefix}{symbol}"
                    text_parts.append(f"<{cls._format_timestamp(word.start)}><c> {symbol}</c>")
                lines.append("".join(text_parts))
            else:
                text = render_bilingual_text(sup, translation_first=translation_first)
                lines.append(
                    f"{cls._format_timestamp(sup.start)} --> "
                    f"{cls._format_timestamp(sup.end)}{cue_settings_str}"
                )
                if show_speaker and sup.speaker:
                    sep = " " if sup.speaker == ">>" else ": "
                    prefix = f"{sup.speaker}{sep}"
                    if speaker_color and sup.speaker != ">>":
                        color = resolve_speaker_color_rgb(sup.speaker, speaker_color, color_cache)
                        if color:
                            prefix = f'<font color="{color}">{prefix}</font>'
                    text = f"{prefix}{text}"
                lines.append(text)
            lines.append("")

        return "\n".join(lines).encode("utf-8")

    # =========================================================================
    # Writer: ASS-to-CSS style conversion
    # =========================================================================

    @staticmethod
    def _ass_color_to_css(ass_color: str) -> str:
        """Convert ASS color &HAABBGGRR to CSS hex #RRGGBB or rgba."""
        c = ass_color.replace("&H", "").replace("&h", "").lstrip("0") or "0"
        c = c.zfill(8)  # pad to AABBGGRR
        aa, bb, gg, rr = c[0:2], c[2:4], c[4:6], c[6:8]
        alpha = int(aa, 16)
        if alpha > 0:
            # ASS alpha: 0=opaque, 255=transparent
            css_alpha = round(1 - alpha / 255, 2)
            return f"rgba({int(rr, 16)},{int(gg, 16)},{int(bb, 16)},{css_alpha})"
        return f"#{rr}{gg}{bb}"

    @classmethod
    def _build_vtt_style_block(cls, metadata: Dict) -> List[str]:
        """Build VTT STYLE block from ass_styles metadata."""
        ass_styles = metadata.get("ass_styles", {})
        default_style = ass_styles.get("Default")
        if not default_style:
            return []

        props: List[str] = []
        if "fontname" in default_style:
            props.append(f"  font-family: {default_style['fontname']};")
        if "bold" in default_style and default_style["bold"]:
            props.append("  font-weight: bold;")
        if "italic" in default_style and default_style["italic"]:
            props.append("  font-style: italic;")
        if "primarycolor" in default_style:
            props.append(f"  color: {cls._ass_color_to_css(default_style['primarycolor'])};")

        borderstyle = default_style.get("borderstyle", 1)
        if borderstyle == 3 and "outlinecolor" in default_style:
            # Opaque box mode: ASS borderstyle=3 uses OutlineColour as the box fill
            props.append(f"  background-color: {cls._ass_color_to_css(default_style['outlinecolor'])};")
            props.append("  text-shadow: none;")
        else:
            props.append("  background-color: transparent;")
            outline_w = default_style.get("outline", 0)
            if outline_w > 0 and "outlinecolor" in default_style:
                oc = cls._ass_color_to_css(default_style["outlinecolor"])
                w = max(1, round(outline_w))
                # 4-direction text-shadow for outline
                props.append(f"  text-shadow: -{w}px 0 {oc}, {w}px 0 {oc}, 0 -{w}px {oc}, 0 {w}px {oc};")

        if not props:
            return []
        return ["STYLE", "::cue {"] + props + ["}"]
