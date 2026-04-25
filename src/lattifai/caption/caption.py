"""Caption data structure for storing subtitle information with metadata."""

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from .config import ASSConfig, LRCConfig, SRTConfig, RenderConfig, StandardizationConfig
    from .formats.nle.audition import AuditionCSVConfig, EdiMarkerConfig
    from .formats.nle.avid import AvidDSConfig
    from .formats.nle.fcpxml import FCPXMLConfig
    from .formats.nle.premiere import PremiereXMLConfig
    from .formats.ttml import TTMLConfig

    FormatConfig = Union[
        ASSConfig, LRCConfig, SRTConfig, TTMLConfig,
        FCPXMLConfig, PremiereXMLConfig,
        AvidDSConfig, AuditionCSVConfig, EdiMarkerConfig,
    ]

from .config import InputCaptionFormat, OutputCaptionFormat  # noqa: F401
from .exceptions import CaptionParseError, FormatDetectionError, FormatNotSupportedError
from .formats import detect_format, detect_format_from_content, get_reader, get_writer
from .supervision import AlignmentItem, Pathlike, Supervision, fastcopy


@dataclass
class Caption:
    """Container for caption/subtitle data with metadata.

    Encapsulates a list of supervisions (subtitle segments) along with
    metadata such as language, kind, format information, and source file details.
    """

    supervisions: List[Supervision] = field(default_factory=list)
    """List of supervision segments containing text and timing information."""

    language: Optional[str] = None
    """Language code (e.g., 'en', 'zh', 'es')."""

    target_lang: Optional[str] = None
    """Target language code for translation."""

    kind: Optional[str] = None
    """Caption kind/type (e.g., 'captions', 'subtitles', 'descriptions')."""

    source_format: Optional[str] = None
    """Original format of the caption file (e.g., 'vtt', 'srt', 'json')."""

    source_path: Optional[Pathlike] = None
    """Path to the source caption file."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional custom metadata as key-value pairs."""

    def __len__(self) -> int:
        """Return the number of supervision segments."""
        return len(self.supervisions)

    def __iter__(self):
        """Iterate over supervision segments."""
        return iter(self.supervisions)

    def __getitem__(self, index):
        """Get supervision segment by index."""
        return self.supervisions[index]

    def __bool__(self) -> bool:
        """Return True if caption has supervisions."""
        return len(self) > 0

    @property
    def is_empty(self) -> bool:
        """Check if caption has no supervisions."""
        return len(self.supervisions) == 0

    @property
    def duration(self) -> Optional[float]:
        """
        Get total duration of the caption in seconds.

        Returns:
            Total duration from first to last supervision, or None if empty
        """
        if not self.supervisions:
            return None
        return self.supervisions[-1].end - self.supervisions[0].start

    @property
    def start_time(self) -> Optional[float]:
        """Get start time of first supervision."""
        if not self.supervisions:
            return None
        return self.supervisions[0].start

    @property
    def end_time(self) -> Optional[float]:
        """Get end time of last supervision."""
        if not self.supervisions:
            return None
        return self.supervisions[-1].end

    def append(self, supervision: Supervision) -> None:
        """Add a supervision segment to the caption."""
        self.supervisions.append(supervision)

    def extend(self, supervisions: List[Supervision]) -> None:
        """Add multiple supervision segments to the caption."""
        self.supervisions.extend(supervisions)

    def filter_by_speaker(self, speaker: str) -> "Caption":
        """
        Create a new Caption with only supervisions from a specific speaker.

        Args:
            speaker: Speaker identifier to filter by

        Returns:
            New Caption instance with filtered supervisions
        """
        filtered_sups = [sup for sup in self.supervisions if sup.speaker == speaker]
        return Caption(
            supervisions=filtered_sups,
            language=self.language,
            kind=self.kind,
            source_format=self.source_format,
            source_path=self.source_path,
            metadata=self.metadata.copy(),
        )

    def get_speakers(self) -> List[str]:
        """
        Get list of unique speakers in the caption.

        Returns:
            Sorted list of unique speaker identifiers
        """
        speakers = {sup.speaker for sup in self.supervisions if sup.speaker}
        return sorted(speakers)

    @property
    def has_translation(self) -> bool:
        """Check if any supervision has translation data."""
        return any(sup.has_translation for sup in self.supervisions)

    def set_translations(self, translations: List[str], target_lang: Optional[str] = None) -> "Caption":
        """Set translations for supervisions.

        Args:
            translations: List of translated strings, one per supervision
            target_lang: Language code of the translations (e.g., 'zh', 'ja')

        Returns:
            Self for chaining
        """
        if len(translations) != len(self.supervisions):
            raise ValueError(
                f"Number of translations ({len(translations)}) must match "
                f"number of supervisions ({len(self.supervisions)})"
            )
        for sup, trans in zip(self.supervisions, translations):
            sup.translation = trans
            if target_lang:
                sup.target_lang = target_lang
        if target_lang:
            self.target_lang = target_lang
        return self

    def strip_translations(self) -> "Caption":
        """Remove all translation data from supervisions.

        Returns:
            Self for chaining
        """
        for sup in self.supervisions:
            sup.translation = None
            sup.target_lang = None
        self.target_lang = None
        return self

    def shift_time(self, seconds: float) -> "Caption":
        """
        Create a new Caption with all timestamps shifted by given seconds.

        Args:
            seconds: Number of seconds to shift (positive delays, negative advances)

        Returns:
            New Caption instance with shifted timestamps
        """
        shifted_sups = []
        for sup in self.supervisions:
            # Calculate physical time range
            raw_start = sup.start + seconds
            raw_end = sup.end + seconds

            # Skip segments that end before 0
            if raw_end <= 0:
                continue

            # Clip start to 0 if negative
            if raw_start < 0:
                final_start = 0.0
                final_duration = raw_end
            else:
                final_start = raw_start
                final_duration = sup.duration

            # Handle alignment (word-level timestamps)
            final_alignment = None
            original_alignment = getattr(sup, "alignment", None)
            if original_alignment and "word" in original_alignment:
                new_words = []
                for word in original_alignment["word"]:
                    w_start = word.start + seconds
                    w_end = w_start + word.duration

                    # Skip words that end before 0
                    if w_end <= 0:
                        continue

                    # Clip start to 0 if negative
                    if w_start < 0:
                        w_final_start = 0.0
                        w_final_duration = w_end
                    else:
                        w_final_start = w_start
                        w_final_duration = word.duration

                    new_words.append(
                        AlignmentItem(
                            symbol=word.symbol,
                            start=w_final_start,
                            duration=w_final_duration,
                            score=word.score,
                        )
                    )

                # Copy original alignment dict structure and update words
                final_alignment = original_alignment.copy()
                final_alignment["word"] = new_words

            shifted_sups.append(
                Supervision(
                    text=sup.text,
                    start=final_start,
                    duration=final_duration,
                    speaker=sup.speaker,
                    id=sup.id,
                    recording_id=sup.recording_id if hasattr(sup, "recording_id") else "",
                    channel=getattr(sup, "channel", 0),
                    language=sup.language,
                    alignment=final_alignment,
                    custom=sup.custom,
                )
            )

        return Caption(
            supervisions=shifted_sups,
            language=self.language,
            kind=self.kind,
            source_format=self.source_format,
            source_path=self.source_path,
            metadata=self.metadata.copy(),
        )

    def with_margins(
        self,
        start_margin: float = 0.10,
        end_margin: float = 0.10,
        min_gap: float = 0.08,
        collision_mode: str = "trim",
    ) -> "Caption":
        """
        Create a new Caption with segment boundaries adjusted based on word-level alignment.

        Uses supervision.alignment['word'] to recalculate segment start/end times
        with the specified margins applied around the actual speech boundaries.

        Args:
            start_margin: Seconds to extend before the first word (default: 0.10)
            end_margin: Seconds to extend after the last word (default: 0.10)
            min_gap: Minimum gap between segments for collision handling (default: 0.08)
            collision_mode: How to handle segment overlap - 'trim' or 'gap' (default: 'trim')

        Returns:
            New Caption instance with adjusted timestamps

        Note:
            Segments without alignment data will keep their original timestamps.

        Example:
            >>> caption = Caption.read("aligned.srt")
            >>> adjusted = caption.with_margins(start_margin=0.05, end_margin=0.15)
            >>> adjusted.write("output.srt")
        """
        from .standardize import apply_margins_to_captions

        adjusted_sups = apply_margins_to_captions(
            self.supervisions,
            start_margin=start_margin,
            end_margin=end_margin,
            min_gap=min_gap,
            collision_mode=collision_mode,
        )

        return Caption(
            supervisions=adjusted_sups,
            language=self.language,
            kind=self.kind,
            source_format=self.source_format,
            source_path=self.source_path,
            metadata=self.metadata.copy(),
        )

    def to_string(
        self,
        format: str = "srt",
        render: Optional["RenderConfig"] = None,
        format_config: Optional["FormatConfig"] = None,
    ) -> str:
        """
        Return caption content in specified format.

        Args:
            format: Output format (e.g., 'srt', 'vtt', 'ass')
            render: RenderConfig controlling rendering and output behavior
            format_config: Format-specific configuration (ASSConfig, TTMLConfig, etc.)

        Returns:
            String containing formatted captions
        """
        return self.to_bytes(output_format=format, render=render, format_config=format_config).decode("utf-8")

    def to_dict(self) -> Dict:
        """
        Convert Caption to dictionary representation.

        Returns:
            Dictionary with caption data and metadata
        """
        return {
            "supervisions": [sup.to_dict() for sup in self.supervisions],
            "language": self.language,
            "target_lang": self.target_lang,
            "kind": self.kind,
            "source_format": self.source_format,
            "source_path": str(self.source_path) if self.source_path else None,
            "metadata": self.metadata,
            "duration": self.duration,
            "num_segments": len(self.supervisions),
            "speakers": self.get_speakers(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Caption":
        """Create Caption from a dictionary (inverse of to_dict).

        Accepts the same structure as to_dict() output and CaptionData schema.
        Ignores computed fields (duration, num_segments, speakers).

        Args:
            data: Dictionary with caption fields.

        Returns:
            New Caption instance.
        """
        sups = data.get("supervisions", [])
        supervisions = [
            Supervision.from_dict(s) if isinstance(s, dict) else s
            for s in sups
        ]
        return cls(
            supervisions=supervisions,
            language=data.get("language"),
            target_lang=data.get("target_lang"),
            kind=data.get("kind"),
            source_format=data.get("source_format"),
            source_path=data.get("source_path"),
            metadata=data.get("metadata") or {},
        )

    @classmethod
    def from_supervisions(
        cls,
        supervisions: List[Supervision],
        language: Optional[str] = None,
        target_lang: Optional[str] = None,
        kind: Optional[str] = None,
        source_format: Optional[str] = None,
        source_path: Optional[Pathlike] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> "Caption":
        """
        Create Caption from a list of supervisions.

        Args:
            supervisions: List of supervision segments
            language: Language code
            target_lang: Target language code for bilingual captions
            kind: Caption kind/type
            source_format: Original format
            source_path: Source file path
            metadata: Additional metadata

        Returns:
            New Caption instance
        """
        return cls(
            supervisions=supervisions,
            language=language,
            target_lang=target_lang,
            kind=kind,
            source_format=source_format,
            source_path=source_path,
            metadata=metadata or {},
        )

    @classmethod
    def from_string(
        cls,
        content: str,
        format: Optional[str] = None,
        normalize_text: bool = True,
    ) -> "Caption":
        """
        Create Caption from string content.

        Args:
            content: Caption content as string
            format: Caption format (e.g., 'srt', 'vtt', 'ass').
                Auto-detected from content when omitted.
            normalize_text: Whether to normalize text during reading

        Returns:
            New Caption instance

        Raises:
            FormatDetectionError: If format cannot be auto-detected from content.
            FormatNotSupportedError: If the specified format has no registered reader.
            CaptionParseError: If the reader fails to parse the content.

        Example:
            >>> caption = Caption.from_string(srt_content)          # auto-detect
            >>> caption = Caption.from_string(srt_content, "srt")   # explicit
        """
        if not format or format == "auto":
            format = detect_format_from_content(content)
            if not format:
                raise FormatDetectionError(
                    "Unable to detect caption format from content. "
                    "Please specify the 'format' parameter explicitly."
                )

        reader_cls = get_reader(format)
        if not reader_cls:
            from .formats.pysubs2 import Pysubs2Format

            reader_cls = Pysubs2Format

        # Sentinel newline: readers use "\n" presence to distinguish
        # content from file paths. Without this, single-line content
        # (e.g., short CJK text) would be misidentified.
        if "\n" not in content:
            content += "\n"

        try:
            result = reader_cls.parse(content, normalize_text=normalize_text)
        except (FormatDetectionError, FormatNotSupportedError):
            raise
        except Exception as exc:
            raise CaptionParseError(f"Failed to parse {format} content: {exc}") from exc

        return cls(
            supervisions=result.supervisions,
            language=result.language,
            target_lang=result.target_lang,
            kind=result.kind,
            source_format=format,
            metadata=result.format_metadata,
        )

    def to_bytes(
        self,
        output_format: Optional[str] = None,
        render: Optional["RenderConfig"] = None,
        format_config: Optional["FormatConfig"] = None,
    ) -> bytes:
        """
        Convert caption to bytes.

        Args:
            output_format: Output format (e.g., 'srt', 'vtt', 'ass'). Defaults to source_format or 'srt'
            render: RenderConfig controlling rendering and output behavior
            format_config: Format-specific configuration (ASSConfig, TTMLConfig, etc.)

        Returns:
            Caption content as bytes

        Example:
            >>> caption = Caption.read("input.srt")
            >>> data = caption.to_bytes()
            >>> vtt_data = caption.to_bytes(output_format="vtt")
        """
        return self.write(
            None,
            format_config=format_config,
            render=render,
            _output_format=output_format,
        )

    @classmethod
    def read(
        cls,
        path: Union[Pathlike, io.BytesIO, io.StringIO],
        format: Optional[str] = None,
        normalize_text: bool = True,
        encoding: str = "utf-8",
    ) -> "Caption":
        """
        Read caption file or in-memory data and return Caption object.

        Args:
            path: Path to caption file, or BytesIO/StringIO object with caption content.
            format: Caption format. Auto-detected from file extension or content
                when omitted.
            normalize_text: Whether to normalize text during reading.
            encoding: Character encoding for BytesIO / file reading (default utf-8).

        Returns:
            Caption object containing supervisions and metadata.

        Raises:
            ValueError: If format cannot be determined.
            FileNotFoundError: If file path does not exist.
        """
        source_path: Optional[str] = None
        detected_encoding: Optional[str] = None

        # --- Load content into memory string ---
        if isinstance(path, (io.BytesIO, io.StringIO)):
            content = path.read().decode(encoding, errors="replace") if isinstance(path, io.BytesIO) else path.read()
        else:
            file_path = Path(str(path))
            if not file_path.is_file():
                raise FileNotFoundError(f"Caption file not found: {file_path}")
            source_path = str(file_path)
            if not format or format == "auto":
                format = detect_format(source_path) or file_path.suffix.lstrip(".").lower()
            # Use encoding detection for robust handling of UTF-16/GBK/GB18030 files.
            # Pure-utf-8 files round-trip through the BOM branch unchanged.
            from .formats.pysubs2 import detect_file_encoding

            content, detected_encoding = detect_file_encoding(file_path)

        # --- Resolve format: explicit > file extension > content sniffing ---
        if not format or format == "auto":
            format = detect_format_from_content(content)

        # --- Parse ---
        caption = cls.from_string(content, format=format, normalize_text=normalize_text)
        caption.source_path = source_path
        # Preserve the real on-disk encoding for downstream consumers (e.g.
        # roundtripping back to the original file encoding). parse()'s
        # from_string branch can't see it because the string is already decoded.
        if detected_encoding and "encoding" not in caption.metadata:
            caption.metadata["encoding"] = detected_encoding

        # Detect dominant line terminator so writers can preserve Windows-style
        # (CRLF) files (common in ASS output from Arctime/Aegisub on Windows).
        # Without this, pysubs2's ``\n`` output silently flips CRLF to LF and
        # turns a byte-faithful roundtrip into a full-file diff.
        if content and "line_terminator" not in caption.metadata:
            crlf = content.count("\r\n")
            total_lf = content.count("\n")
            bare_lf = total_lf - crlf
            if crlf > bare_lf and crlf > 0:
                caption.metadata["line_terminator"] = "\r\n"
            elif bare_lf > 0:
                caption.metadata["line_terminator"] = "\n"

        # P2-3: detect language from filename when not already set by the reader.
        # Patterns like ".简体中文&英文.ass" / ".CN&EN.srt" / ".双语.ass" are
        # free metadata available at zero parse cost.
        if not caption.language and source_path:
            from .parsers.text_parser import detect_language_from_filename

            lang, tgt = detect_language_from_filename(source_path)
            if lang:
                caption.language = lang
            if tgt and not caption.target_lang:
                caption.target_lang = tgt

        return caption

    def write(
        self,
        path: Union[Pathlike, io.BytesIO, None] = None,
        format_config: Optional["FormatConfig"] = None,
        render: Optional["RenderConfig"] = None,
        standardization: Optional["StandardizationConfig"] = None,
        _output_format: Optional[str] = None,
    ) -> Union[Pathlike, bytes]:
        """
        Write caption to file or return as bytes.

        Args:
            path: Path to output caption file, BytesIO object, or None to return bytes
            format_config: Format-specific configuration (ASSConfig, TTMLConfig, etc.)
            render: RenderConfig controlling include_speaker, word_level, translation_first
            standardization: Broadcast standardization (min/max duration, CPS, margins)

        Returns:
            Path to the written file if path is a file path, or bytes if path is BytesIO/None
        """
        from .config import ASSConfig, RenderConfig, apply_color_scheme

        effective_render = render or RenderConfig()

        # Apply karaoke color scheme from ASSConfig
        if isinstance(format_config, ASSConfig) and format_config.karaoke_color_scheme:
            format_config = apply_color_scheme(format_config.karaoke_color_scheme, format_config)

        supervisions = self.supervisions

        # Apply broadcast standardization if configured
        if standardization:
            from .standardize import CaptionStandardizer

            standardizer = CaptionStandardizer(
                min_duration=standardization.min_duration,
                max_duration=standardization.max_duration,
                min_gap=standardization.min_gap,
                max_lines=standardization.max_lines,
                max_chars_per_line=standardization.max_chars_per_line,
            )
            supervisions = standardizer.process(supervisions)
            if standardization.start_margin is not None:
                supervisions = standardizer.apply_margins(
                    supervisions,
                    start_margin=standardization.start_margin,
                    end_margin=standardization.end_margin or 0.10,
                )

        # Roundtrip metadata: merge format_metadata with Caption-level attrs
        # so writers (e.g., VTT) can access kind/language from the metadata dict.
        effective_metadata = dict(self.metadata) if self.metadata else {}
        if self.kind:
            effective_metadata.setdefault("kind", self.kind)
        if self.language:
            effective_metadata.setdefault("language", self.language)

        # For JSON format: build full Caption-level metadata dict
        caption_level_metadata = {
            "language": self.language,
            "target_lang": self.target_lang,
            "kind": self.kind,
            "source_format": self.source_format,
            "metadata": effective_metadata,
        }
        caption_level_metadata = {k: v for k, v in caption_level_metadata.items() if v is not None}

        # Determine output format: explicit > path extension > source format > "srt"
        if _output_format:
            fmt = _output_format.lower()
        elif isinstance(path, (io.BytesIO, type(None))):
            fmt = self.source_format or "srt"
        else:
            fmt = detect_format(str(path)) or Path(str(path)).suffix.lstrip(".").lower() or "srt"

        # Special casing for professional formats
        ext = fmt
        if isinstance(path, (str, Path)):
            path_str = str(path)
            if path_str.endswith("_avid.txt"):
                ext = "avid_ds"
            elif "audition" in path_str.lower() and path_str.endswith(".csv"):
                ext = "audition_csv"
            elif "edimarker" in path_str.lower() and path_str.endswith(".csv"):
                ext = "edimarker_csv"
            elif "imsc" in path_str.lower() and path_str.endswith(".ttml"):
                ext = "imsc1"
            elif "ebu" in path_str.lower() and path_str.endswith(".ttml"):
                ext = "ebu_tt_d"

        writer_cls = get_writer(ext)
        if not writer_cls:
            from .formats.pysubs2 import Pysubs2Format

            writer_cls = Pysubs2Format

        writer_metadata = caption_level_metadata if ext == "json" else effective_metadata

        if isinstance(path, (str, Path)):
            return writer_cls.write(
                supervisions,
                path,
                metadata=writer_metadata,
                render=effective_render,
                config=format_config,
            )

        content = writer_cls.to_bytes(
            supervisions,
            metadata=writer_metadata,
            render=effective_render,
            config=format_config,
        )
        if isinstance(path, io.BytesIO):
            path.write(content)
            path.seek(0)
        return content

    def __repr__(self) -> str:
        """String representation of Caption."""
        lang = f"lang={self.language}" if self.language else "lang=unknown"
        kind_str = f"kind={self.kind}" if self.kind else ""
        parts = [f"Caption({len(self.supervisions)} segments", lang]
        if kind_str:
            parts.append(kind_str)
        if self.duration:
            parts.append(f"duration={self.duration:.2f}s")
        return ", ".join(parts) + ")"
