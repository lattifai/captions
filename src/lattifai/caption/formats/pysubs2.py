"""Standard subtitle formats using pysubs2 library.

Handles: SRT, VTT, ASS, SSA, SUB (MicroDVD), SAMI/SMI
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

import pysubs2

from ..colors import SPEAKER_PALETTE, resolve_speaker_color
from ..config import CaptionStyle, KaraokeConfig
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
                    duration=(event.end - event.start) / 1000.0 if event.end is not None else 0,
                )
            )

        # Clear candidates to avoid leaking into subsequent reads
        if candidates:
            set_speaker_candidates(set())

        return supervisions

    @classmethod
    def extract_metadata(cls, source, **kwargs) -> Dict[str, str]:
        """Extract metadata from VTT or SRT."""
        import re
        from pathlib import Path

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
        if cls.pysubs2_format == "vtt" or (isinstance(source, str) and source.startswith("WEBVTT")):
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
    def write(
        cls,
        supervisions: List[Supervision],
        output_path,
        include_speaker: bool = True,
        fps: float = 25.0,
        **kwargs,
    ) -> Path:
        """Write caption using pysubs2."""
        output_path = Path(output_path)
        content = cls.to_bytes(supervisions, include_speaker=include_speaker, fps=fps, **kwargs)
        output_path.write_bytes(content)
        return output_path

    @classmethod
    def to_bytes(
        cls,
        supervisions: List[Supervision],
        include_speaker: bool = True,
        fps: float = 25.0,
        word_level: bool = False,
        karaoke_config: Optional[KaraokeConfig] = None,
        **kwargs,
    ) -> bytes:
        """Convert to bytes using pysubs2.

        Args:
            supervisions: List of Supervision objects
            include_speaker: Whether to include speaker in output
            fps: Frames per second (for MicroDVD format)
            word_level: If True and alignment exists, output word-per-segment
            karaoke_config: Karaoke configuration. When provided with enabled=True,
                use karaoke styling (format-specific)

        Returns:
            Subtitle content as bytes
        """
        from .base import expand_to_word_supervisions

        # Check if karaoke is enabled
        karaoke_enabled = karaoke_config is not None and karaoke_config.enabled

        # Expand to word-per-segment if word_level=True and karaoke is not enabled
        if word_level and not karaoke_enabled:
            supervisions = expand_to_word_supervisions(supervisions)

        from .base import render_bilingual_text

        subs = pysubs2.SSAFile()

        for sup in supervisions:
            text = render_bilingual_text(sup)
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
        include_speaker: bool = True,
        use_bom: bool = False,
        metadata: Optional[Dict] = None,
        **kwargs,
    ) -> bytes:
        """Generate SRT with proper formatting (comma for milliseconds).

        Args:
            supervisions: List of supervision segments
            include_speaker: Whether to include speaker in output
            use_bom: Whether to add BOM for Windows compatibility
            metadata: Optional metadata dict. If encoding is 'utf-8-sig', adds BOM.
        """
        content = super().to_bytes(supervisions, include_speaker=include_speaker, **kwargs)

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
                subs = pysubs2.load(str(source), encoding="utf-8", format_=cls.pysubs2_format)
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
                    duration=(event.end - event.start) / 1000.0 if event.end is not None else 0,
                    custom=custom,
                )
            )

        if candidates:
            set_speaker_candidates(set())

        return supervisions

    @classmethod
    def extract_metadata(cls, source, **kwargs) -> Dict:
        """Extract ASS global metadata including Script Info and Styles.

        Returns:
            Dict containing:
            - ass_info: Script Info section as dict
            - ass_styles: Style definitions as dict of dicts
        """
        try:
            if cls.is_content(source):
                subs = pysubs2.SSAFile.from_string(source, format_=cls.pysubs2_format)
            else:
                subs = pysubs2.load(str(source), encoding="utf-8", format_=cls.pysubs2_format)
        except Exception:
            return {}

        # Convert styles to serializable dict
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

        return {
            "ass_info": dict(subs.info),
            "ass_styles": styles_dict,
        }

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
        include_speaker: bool = True,
        fps: float = 25.0,
        word_level: bool = False,
        karaoke_config: Optional[KaraokeConfig] = None,
        metadata: Optional[Dict] = None,
        **kwargs,
    ) -> bytes:
        """Convert to ASS bytes with style preservation and optional karaoke tags.

        Args:
            supervisions: List of supervision segments
            include_speaker: Whether to include speaker in output
            fps: Frames per second (not used for ASS)
            word_level: If True and alignment exists, output word-per-segment or karaoke
            karaoke_config: Karaoke configuration. When provided with enabled=True,
                generate karaoke tags
            metadata: Optional metadata dict containing ass_info and ass_styles
                to restore original ASS formatting

        Returns:
            ASS content as bytes
        """
        from .base import expand_to_word_supervisions

        speaker_color = kwargs.pop("speaker_color", "")
        background_color = kwargs.pop("background_color", "")
        karaoke_enabled = karaoke_config is not None and karaoke_config.enabled

        # Expand to word-per-segment if word_level=True and karaoke is not enabled
        if word_level and not karaoke_enabled:
            supervisions = expand_to_word_supervisions(supervisions)

        # Create ASS file and restore global styles from metadata
        subs = cls._create_ass_file_with_metadata(metadata)

        # Add karaoke style with priority:
        # 1. metadata ass_styles.Karaoke (already loaded above)
        # 2. Copy from metadata ass_styles.Default (inherits user's font/color settings)
        # 3. Fallback to karaoke_config.style defaults
        has_metadata_styles = metadata and "ass_styles" in metadata
        if karaoke_enabled and "Karaoke" not in subs.styles:
            if has_metadata_styles and "Default" in metadata["ass_styles"]:
                subs.styles["Karaoke"] = subs.styles["Default"].copy()
            else:
                subs.styles["Karaoke"] = cls._create_karaoke_style(karaoke_config.style)

        # Apply background_color to Default style (non-karaoke mode)
        if background_color and not karaoke_enabled and "Default" in subs.styles:
            subs.styles["Default"].backcolor = cls._hex_to_ass_color(background_color)
            subs.styles["Default"].borderstyle = 3
            subs.styles["Default"].shadow = 0

        # Speaker color cache: maps speaker name → BBGGRR color string (assigned on first appearance)
        _speaker_color_cache = {}

        for sup in supervisions:
            alignment = getattr(sup, "alignment", None)
            word_items = alignment.get("word") if alignment else None

            # Karaoke mode with word alignment
            if word_level and karaoke_enabled and word_items:
                karaoke_text = cls._build_karaoke_text(word_items, karaoke_config.effect)
                if cls._should_include_speaker(sup, include_speaker):
                    prefix = cls._format_speaker_prefix(sup.speaker)
                    spk_color = cls._resolve_speaker_color(sup.speaker, speaker_color, _speaker_color_cache)
                    if spk_color:
                        karaoke_text = f"{{\\c&H{spk_color}&}}{prefix}{{\\c}}{karaoke_text}"
                    else:
                        karaoke_text = f"{prefix}{karaoke_text}"
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
                # Standard mode: restore custom attributes from supervision
                from .base import render_bilingual_text

                text = render_bilingual_text(sup, separator="\\N")
                if cls._should_include_speaker(sup, include_speaker):
                    prefix = cls._format_speaker_prefix(sup.speaker)
                    spk_color = cls._resolve_speaker_color(sup.speaker, speaker_color, _speaker_color_cache)
                    if spk_color:
                        text = f"{{\\c&H{spk_color}&}}{prefix}{{\\c}}{text}"
                    else:
                        text = f"{prefix}{text}"

                event = cls._create_event_from_supervision(sup, text)
                subs.append(event)

        return subs.to_string(format_="ass").encode("utf-8")

    # Default reference resolution for ASS font scaling.
    # Players scale all coordinates and font sizes from PlayRes to actual video size.
    DEFAULT_PLAY_RES_X = 1920
    DEFAULT_PLAY_RES_Y = 1080
    DEFAULT_FONTSIZE = 48.0

    @classmethod
    def _create_ass_file_with_metadata(cls, metadata: Optional[Dict]) -> pysubs2.SSAFile:
        """Create SSAFile and restore global styles from metadata.

        When no PlayRes is provided, defaults to 1920x1080 reference resolution
        so fonts scale proportionally to the actual video size.

        Args:
            metadata: Dict containing ass_info, ass_styles, play_res_x, play_res_y

        Returns:
            pysubs2.SSAFile with restored styles
        """
        subs = pysubs2.SSAFile()

        # Set sensible defaults for font scaling (overridable by metadata)
        subs.info["PlayResX"] = str(cls.DEFAULT_PLAY_RES_X)
        subs.info["PlayResY"] = str(cls.DEFAULT_PLAY_RES_Y)
        subs.styles["Default"].fontsize = cls.DEFAULT_FONTSIZE

        if not metadata:
            return subs

        # Restore Script Info (may override PlayRes defaults)
        if "ass_info" in metadata:
            subs.info.update(metadata["ass_info"])

        # Explicit PlayRes overrides take highest priority
        if "play_res_x" in metadata:
            subs.info["PlayResX"] = str(metadata["play_res_x"])
        if "play_res_y" in metadata:
            subs.info["PlayResY"] = str(metadata["play_res_y"])

        # Restore Styles
        if "ass_styles" in metadata:
            for name, style_dict in metadata["ass_styles"].items():
                subs.styles[name] = cls._dict_to_style(style_dict)

        return subs

    @classmethod
    def _dict_to_style(cls, style_dict: Dict) -> pysubs2.SSAStyle:
        """Convert style dict back to pysubs2.SSAStyle."""
        return pysubs2.SSAStyle(
            fontname=style_dict.get("fontname", "Arial"),
            fontsize=style_dict.get("fontsize", 20.0),
            primarycolor=cls._str_to_color(style_dict.get("primarycolor", "&H00FFFFFF")),
            secondarycolor=cls._str_to_color(style_dict.get("secondarycolor", "&H000000FF")),
            tertiarycolor=cls._str_to_color(style_dict.get("tertiarycolor", "&H00000000")),
            outlinecolor=cls._str_to_color(style_dict.get("outlinecolor", "&H00000000")),
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
    def _create_event_from_supervision(cls, sup: Supervision, text: str) -> pysubs2.SSAEvent:
        """Create SSAEvent from Supervision, restoring custom attributes.

        Args:
            sup: Supervision with optional custom dict containing ass_* attributes
            text: Processed text content

        Returns:
            pysubs2.SSAEvent with restored attributes
        """
        custom = getattr(sup, "custom", None) or {}

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
    def _create_karaoke_style(cls, style: CaptionStyle) -> pysubs2.SSAStyle:
        """Create pysubs2 SSAStyle from CaptionStyle config.

        Args:
            style: KaraokeStyle configuration

        Returns:
            pysubs2.SSAStyle object
        """
        # Convert int alignment to pysubs2.Alignment enum
        alignment = pysubs2.Alignment(style.alignment)

        # When background_color is set, switch to borderstyle=3 (opaque box)
        has_bg = bool(style.background_color)
        if has_bg:
            back = cls._hex_to_ass_color(style.background_color)
            borderstyle = 3
            shadow = 0  # ASS ignores shadow in borderstyle=3
        else:
            back = cls._hex_to_ass_color(style.back_color)
            borderstyle = 1
            shadow = style.shadow_depth

        return pysubs2.SSAStyle(
            fontname=style.font_name,
            fontsize=style.font_size,
            primarycolor=cls._hex_to_ass_color(style.primary_color),
            secondarycolor=cls._hex_to_ass_color(style.secondary_color),
            outlinecolor=cls._hex_to_ass_color(style.outline_color),
            backcolor=back,
            bold=style.bold,
            italic=style.italic,
            outline=style.outline_width,
            shadow=shadow,
            borderstyle=borderstyle,
            alignment=alignment,
            marginl=style.margin_l,
            marginr=style.margin_r,
            marginv=style.margin_v,
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
    def _build_karaoke_text(words: list, effect: str = "sweep") -> str:
        """Build karaoke tag text.

        Args:
            words: List of AlignmentItem objects
            effect: Karaoke effect type ("sweep", "instant", "outline")

        Returns:
            Text with karaoke tags, e.g. "{\\kf45}Hello {\\kf55}world"
        """
        tag_map = {"sweep": "kf", "instant": "k", "outline": "ko"}
        tag = tag_map.get(effect, "kf")

        parts = []
        for word in words:
            # Duration in centiseconds (multiply by 100)
            centiseconds = int(word.duration * 100)
            parts.append(f"{{\\{tag}{centiseconds}}}{word.symbol}")

        return " ".join(parts)

    # Backward-compatible alias — canonical source is colors.SPEAKER_PALETTE
    _SPEAKER_PALETTE = SPEAKER_PALETTE

    @classmethod
    def _resolve_speaker_color(cls, speaker: str, speaker_color_spec: str, cache: dict) -> str:
        """Resolve ASS BBGGRR color string for a speaker.

        Delegates to colors.resolve_speaker_color() — kept as thin wrapper
        for backward compatibility with tests referencing ASSFormat._resolve_speaker_color.
        """
        return resolve_speaker_color(speaker, speaker_color_spec, cache)


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
        include_speaker: bool = True,
        fps: float = 25.0,
        word_level: bool = False,
        karaoke_config: Optional[KaraokeConfig] = None,
        metadata: Optional[Dict] = None,
        **kwargs,
    ) -> bytes:
        """Convert to SSA bytes with style preservation."""
        from .base import expand_to_word_supervisions

        if word_level and not (karaoke_config and karaoke_config.enabled):
            supervisions = expand_to_word_supervisions(supervisions)

        from .base import render_bilingual_text

        subs = cls._create_ass_file_with_metadata(metadata)

        for sup in supervisions:
            text = render_bilingual_text(sup, separator="\\N")
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
