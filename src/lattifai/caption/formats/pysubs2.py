"""Standard subtitle formats using pysubs2 library.

Handles: SRT, VTT, ASS, SSA, SUB (MicroDVD), SAMI/SMI
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

import pysubs2

from ..colors import SPEAKER_PALETTE, hex_rgb_to_bgr, resolve_speaker_color
from ..config import ASSConfig
from ..parsers.text_parser import detect_speaker_candidates
from ..parsers.text_parser import normalize_text as normalize_text_fn
from ..parsers.text_parser import parse_speaker_text, set_speaker_candidates
from ..supervision import Supervision
from . import register_format
from .base import FormatHandler


class Pysubs2Format(FormatHandler):
    """Base class for formats handled by pysubs2."""

    # Subclasses should set these
    pysubs2_format: str = ""

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
            with open(path, "r", encoding="utf-8") as f:
                content = cls._preprocess_content(f.read())

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
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read(4096)
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

        subs = pysubs2.SSAFile()

        tf = behavior.translation_first
        for sup in supervisions:
            text = render_bilingual_text(sup, translation_first=tf)
            if cls._should_include_speaker(sup, include_speaker):
                text = f"{cls._format_speaker_prefix(sup.speaker)}{text}"

            subs.append(
                pysubs2.SSAEvent(
                    start=int(sup.start * 1000),
                    end=int(sup.end * 1000),
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
    """SRT (SubRip) format - the most widely used subtitle format."""

    extensions = [".srt"]
    pysubs2_format = "srt"
    description = "SubRip Subtitle format - universal compatibility"

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
        """
        content = super().to_bytes(supervisions, **kwargs)

        # Add BOM if requested or if original had BOM
        add_bom = use_bom
        if metadata and metadata.get("encoding") == "utf-8-sig":
            add_bom = True

        if add_bom:
            content = b"\xef\xbb\xbf" + content

        return content


@register_format("ass")
class ASSFormat(Pysubs2Format):
    """Advanced SubStation Alpha format with karaoke support."""

    extensions = [".ass"]
    pysubs2_format = "ass"
    description = "Advanced SubStation Alpha - rich styling support"

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
        try:
            if cls.is_content(source):
                subs = pysubs2.SSAFile.from_string(source, format_=cls.pysubs2_format)
            else:
                subs = pysubs2.load(
                    str(source), encoding="utf-8", format_=cls.pysubs2_format
                )
        except Exception:
            if cls.is_content(source):
                subs = pysubs2.SSAFile.from_string(source)
            else:
                subs = pysubs2.load(str(source), encoding="utf-8")

        # Auto-detect title-case speaker candidates from text content
        all_texts = [e.text for e in subs.events]
        candidates = detect_speaker_candidates(all_texts)
        if candidates:
            set_speaker_candidates(candidates)

        supervisions = []
        for event in subs.events:
            text = event.text
            if normalize_text:
                text = normalize_text_fn(text)

            speaker, text = parse_speaker_text(text)

            # When parse_speaker_text can't extract speaker from text but
            # the Name field has one, strip the matching prefix from text
            # so roundtripping include_speaker_in_text=True doesn't duplicate.
            if not speaker and event.name:
                for sep in (": ", "： "):
                    prefix = event.name + sep
                    if text.startswith(prefix):
                        text = text[len(prefix) :]
                        break

            # Preserve ASS-specific event attributes
            custom = {
                "ass_style": event.style,
                "ass_layer": event.layer,
                "ass_margin_l": event.marginl,
                "ass_margin_r": event.marginr,
                "ass_margin_v": event.marginv,
                "ass_effect": event.effect,
            }

            supervisions.append(
                Supervision(
                    text=text,
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
        """
        from .base import ParseResult

        try:
            if cls.is_content(source):
                subs = pysubs2.SSAFile.from_string(source, format_=cls.pysubs2_format)
            else:
                subs = pysubs2.load(
                    str(source), encoding="utf-8", format_=cls.pysubs2_format
                )
        except Exception:
            if cls.is_content(source):
                subs = pysubs2.SSAFile.from_string(source)
            else:
                subs = pysubs2.load(str(source), encoding="utf-8")

        # --- Extract supervisions from events ---
        all_texts = [e.text for e in subs.events]
        candidates = detect_speaker_candidates(all_texts)
        if candidates:
            set_speaker_candidates(candidates)

        supervisions = []
        for event in subs.events:
            text = event.text
            if normalize_text:
                text = normalize_text_fn(text)

            speaker, text = parse_speaker_text(text)
            if not speaker and event.name:
                for sep in (": ", "： "):
                    prefix = event.name + sep
                    if text.startswith(prefix):
                        text = text[len(prefix) :]
                        break

            custom = {
                "ass_style": event.style,
                "ass_layer": event.layer,
                "ass_margin_l": event.marginl,
                "ass_margin_r": event.marginr,
                "ass_margin_v": event.marginv,
                "ass_effect": event.effect,
            }

            supervisions.append(
                Supervision(
                    text=text,
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

        return ParseResult(
            supervisions=supervisions,
            format_metadata={
                "ass_info": dict(subs.info),
                "ass_styles": styles_dict,
            },
        )

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

        speaker_color = config.speaker_color

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

        # Speaker color cache: maps speaker name → BBGGRR color string (assigned on first appearance)
        _speaker_color_cache = {}

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
                )
                karaoke_text = karaoke_text.replace("\n", "\\N")
                if cls._should_include_speaker(sup, include_speaker):
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
                event_start = int(word_items[0].start * 1000)
                event_end = int(word_items[-1].end * 1000)

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
                if cls._should_include_speaker(sup, include_speaker):
                    prefix = cls._format_speaker_prefix(sup.speaker)
                    spk_color = cls._resolve_speaker_color(
                        sup.speaker, speaker_color, _speaker_color_cache
                    )
                    if spk_color:
                        text = f"{{\\c&H{spk_color}&}}{prefix}{{\\c}}{text}"
                    else:
                        text = f"{prefix}{text}"

                if config.kinetic_style is not None:
                    from ..kinetic import build_line_override, resolve_kinetic

                    resolved = resolve_kinetic(config.kinetic_style, word_level=False)
                    if resolved is not None:
                        scope, impl = resolved
                        # resolve_kinetic with word_level=False only returns
                        # scope="line" or raises — so we are safe to prepend.
                        override = build_line_override(impl)
                        if override:
                            text = f"{{{override}}}{text}"

                event = cls._create_event_from_supervision(sup, text)
                subs.append(event)

        return subs.to_string(format_="ass").encode("utf-8")

    DEFAULT_FONTSIZE = 64.0

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

        Args:
            sup: Supervision with optional custom dict containing ass_* attributes
            text: Processed text content

        Returns:
            pysubs2.SSAEvent with restored attributes
        """
        custom = getattr(sup, "custom", None) or {}

        # Convert \n to \N for ASS format (pysubs2 uses \N for inline line breaks)
        text = text.replace("\n", "\\N")

        return pysubs2.SSAEvent(
            start=int(sup.start * 1000),
            end=int(sup.end * 1000),
            text=text,
            name=sup.speaker or "",
            style=custom.get("ass_style", "Default"),
            layer=custom.get("ass_layer", 0),
            marginl=custom.get("ass_margin_l", 0),
            marginr=custom.get("ass_margin_r", 0),
            marginv=custom.get("ass_margin_v", 0),
            effect=custom.get("ass_effect", ""),
        )

    @classmethod
    def _build_ass_style(cls, config: ASSConfig) -> pysubs2.SSAStyle:
        """Build pysubs2 SSAStyle from self-contained ASSConfig.

        Args:
            config: ASSConfig with all visual, positioning, and rendering fields

        Returns:
            pysubs2.SSAStyle object
        """
        alignment = pysubs2.Alignment(config.alignment)

        # When background_color is set, switch to borderstyle=3 (opaque box)
        # ASS borderstyle=3 semantics:
        #   OutlineColour = opaque box FILL color
        #   BackColour    = box shadow color
        has_bg = bool(config.background_color)
        if has_bg:
            outline_clr = cls._hex_to_ass_color(config.background_color)
            back = cls._hex_to_ass_color(config.back_color)
            borderstyle = 3
            shadow = 0
        else:
            outline_clr = cls._hex_to_ass_color(config.outline_color)
            back = cls._hex_to_ass_color(config.back_color)
            borderstyle = 1
            shadow = config.shadow_depth

        return pysubs2.SSAStyle(
            fontname=config.font_name,
            fontsize=config.font_size,
            primarycolor=cls._hex_to_ass_color(config.primary_color),
            secondarycolor=cls._hex_to_ass_color(config.secondary_color),
            outlinecolor=outline_clr,
            backcolor=back,
            bold=config.bold,
            italic=config.italic,
            outline=config.outline_width,
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
                kinetic_override = build_word_overrides(word_impl, word_start_ms)
                parts.append(
                    f"{separators[i]}{{\\{tag}{centiseconds}{kinetic_override}}}{word.symbol}"
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

        from .base import render_bilingual_text

        subs = cls._create_ass_file(metadata, config)

        tf = behavior.translation_first
        for sup in supervisions:
            text = render_bilingual_text(sup, separator="\\N", translation_first=tf)
            if cls._should_include_speaker(sup, include_speaker):
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
