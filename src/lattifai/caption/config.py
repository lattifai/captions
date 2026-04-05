"""Caption I/O configuration for LattifAI."""

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, get_args

# Re-export color data for backward compatibility (canonical source: colors.py)
from .colors import KARAOKE_COLOR_SCHEMES, resolve_karaoke_color_scheme  # noqa: F401

# =============================================================================
# Caption Style Configuration Classes
# =============================================================================


class CaptionFonts:
    """Common caption font constants.

    These are reference constants for popular fonts. You can use any
    system font name as the font_name parameter in CaptionStyle.
    """

    # Western fonts
    ARIAL = "Arial"
    IMPACT = "Impact"
    VERDANA = "Verdana"
    HELVETICA = "Helvetica"

    # Chinese fonts
    NOTO_SANS_SC = "Noto Sans SC"
    MICROSOFT_YAHEI = "Microsoft YaHei"
    PINGFANG_SC = "PingFang SC"
    SIMHEI = "SimHei"

    # Japanese fonts
    NOTO_SANS_JP = "Noto Sans JP"
    MEIRYO = "Meiryo"
    HIRAGINO_SANS = "Hiragino Sans"

    # Korean fonts
    NOTO_SANS_KR = "Noto Sans KR"
    MALGUN_GOTHIC = "Malgun Gothic"


@dataclass(frozen=True)
class CaptionStyle:
    """Caption style configuration for ASS/TTML formats.

    Frozen dataclass — instances are immutable after creation.
    Use dataclasses.replace() to create modified copies.
    """

    # -- Colors (#RRGGBB format) --

    primary_color: str = "#FFFFFF"
    """Main text color (#RRGGBB)."""

    secondary_color: str = "#00FFFF"
    """Secondary/highlight color (#RRGGBB). Used as karaoke sweep target color."""

    outline_color: str = "#000000"
    """Text outline color (#RRGGBB)."""

    back_color: str = "#000000"
    """Shadow/back color (#RRGGBB). In ASS borderstyle=1 this is the drop shadow color."""

    # -- Font --

    font_name: str = CaptionFonts.ARIAL
    """Font family name. Use CaptionFonts constants or any system font."""

    font_size: int = 20
    """Font size in points."""

    bold: bool = False
    """Enable bold text."""

    italic: bool = False
    """Enable italic text."""

    # -- Border and shadow --

    outline_width: float = 0
    """Outline thickness (px). Recommended 2.0-2.5 for karaoke."""

    shadow_depth: float = 1.0
    """Shadow distance (px). Set to 0 for karaoke — shadows interfere with color sweep."""

    # -- Background box --

    background_color: str = ""
    """Subtitle background box color.
    - "":           no background box (default — text floats on video)
    - "#RRGGBB":    solid opaque background box
    - "#RRGGBBAA":  semi-transparent background box (e.g., "#00000080" = 50% black)
    Supported formats: ASS (borderstyle=3), TTML, FCPXML, VTT (CSS).
    Silently ignored by formats without background support (SRT, LRC, etc.).
    """

    # -- Position --

    alignment: int = 2
    """ASS alignment (1-9, numpad style). 2=bottom-center (default)."""

    margin_l: int = 20
    """Left margin in pixels."""

    margin_r: int = 20
    """Right margin in pixels."""

    margin_v: int = 20
    """Vertical margin in pixels."""

    # -- Speaker --

    speaker_color: str = ""
    """Speaker name color mode for ASS output.
    - "":           no special color (default)
    - "#RRGGBB":    single color for all speakers
    - "#RRGGBB,#00BFFF,...": comma-separated, auto-assigned per speaker
    - "auto":       built-in 10-color palette, auto-assigned per speaker
    """

    # -- Output behavior --

    include_speaker_in_text: bool = True
    """Include speaker labels in caption text (e.g., '[Alice] Hello' vs 'Hello')."""

    word_level: bool = False
    """Word-level output: word-per-segment in normal mode, word timestamps in karaoke/JSON."""

    translation_first: bool = False
    """Place translation text above original text in bilingual output."""


@dataclass
class KaraokeConfig:
    """Karaoke export configuration.

    Karaoke-specific settings only. Subtitle styling (font, colors, background)
    lives in CaptionStyle, not here.

    Attributes:
        enabled: Whether karaoke mode is enabled
        effect: Karaoke effect type
            - "sweep": Gradual fill from left to right (ASS \\kf tag)
            - "instant": Instant highlight (ASS \\k tag)
            - "outline": Outline then fill (ASS \\ko tag)
        color_scheme: Predefined color scheme name (overrides style colors)
        lrc_precision: LRC time precision ("centisecond" or "millisecond")
        lrc_metadata: LRC metadata dict (ar, ti, al, etc.)
        ttml_timing_mode: TTML timing attribute ("Word" or "Line")
    """

    enabled: bool = False
    effect: Literal["sweep", "instant", "outline"] = "sweep"
    color_scheme: str = ""
    """Karaoke color scheme name. When set, overrides style colors.
    See KARAOKE_COLOR_SCHEMES in colors.py for available schemes.
    Use "" (empty) for manual style configuration."""

    # LRC specific
    lrc_precision: Literal["centisecond", "millisecond"] = "millisecond"
    lrc_metadata: Dict[str, str] = field(default_factory=dict)

    # TTML specific
    ttml_timing_mode: Literal["Word", "Line"] = "Word"


def apply_color_scheme(style: CaptionStyle, scheme_name: str) -> CaptionStyle:
    """Return a new CaptionStyle with karaoke color scheme applied.

    Font, alignment, margins, and other non-color fields are preserved
    from the original style. The input style is never mutated.

    Args:
        style: Source CaptionStyle (not modified)
        scheme_name: Color scheme name (e.g., "azure-gold")

    Returns:
        New CaptionStyle with colors overridden, or the original style
        unchanged if the scheme is not found.
    """
    from dataclasses import replace

    resolved = resolve_karaoke_color_scheme(scheme_name)
    if not resolved:
        return style

    overrides = {
        "primary_color": resolved["primary_color"],
        "secondary_color": resolved["secondary_color"],
        "outline_color": resolved["outline_color"],
        "back_color": resolved["back_color"],
        "shadow_depth": resolved.get("shadow_depth", 0.0),
    }
    if "outline_width" in resolved:
        overrides["outline_width"] = resolved["outline_width"]
    if "background_color" in resolved:
        overrides["background_color"] = resolved["background_color"]

    return replace(style, **overrides)


@dataclass
class StandardizationConfig:
    """Caption standardization configuration following broadcast guidelines.

    Reference Standards:
    - Netflix Timed Text Style Guide
    - BBC Subtitle Guidelines
    - EBU-TT-D Standard

    Attributes:
        min_duration: Minimum segment duration (seconds). Netflix recommends 5/6s, BBC 0.3s
        max_duration: Maximum segment duration (seconds). Netflix/BBC recommends 7s
        min_gap: Minimum gap between segments (seconds). 80ms prevents subtitle flicker
        max_lines: Maximum lines per segment. Broadcast standard is typically 2
        max_chars_per_line: Maximum characters per line. CJK auto-adjusted by ÷2 (e.g., 42 → 21)
        optimal_cps: Optimal reading speed (chars/sec). Netflix recommends 17-20 CPS
        start_margin: Start margin (seconds) before first word. None = no adjustment (default)
        end_margin: End margin (seconds) after last word. None = no adjustment (default)
        margin_collision_mode: How to handle collisions: 'trim' (reduce margin) or 'gap' (maintain min_gap)
    """

    min_duration: float = 0.8
    max_duration: float = 7.0
    min_gap: float = 0.08
    max_lines: int = 2
    max_chars_per_line: int = 42
    optimal_cps: float = 17.0
    start_margin: Optional[float] = None
    end_margin: Optional[float] = None
    margin_collision_mode: Literal["trim", "gap"] = "trim"

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.min_duration <= 0:
            raise ValueError("min_duration must be positive")
        if self.max_duration <= self.min_duration:
            raise ValueError("max_duration must be greater than min_duration")
        if self.min_gap < 0:
            raise ValueError("min_gap cannot be negative")
        if self.max_lines < 1:
            raise ValueError("max_lines must be at least 1")
        if self.max_chars_per_line < 10:
            raise ValueError("max_chars_per_line must be at least 10")
        if self.start_margin is not None and self.start_margin < 0:
            raise ValueError("start_margin cannot be negative")
        if self.end_margin is not None and self.end_margin < 0:
            raise ValueError("end_margin cannot be negative")
        if self.margin_collision_mode not in ("trim", "gap"):
            raise ValueError("margin_collision_mode must be 'trim' or 'gap'")


# =============================================================================
# Format Type Definitions (Single Source of Truth)
# =============================================================================

# Type alias for input caption formats (all formats with registered readers)
InputCaptionFormat = Literal[
    # Standard subtitle formats
    "srt",
    "vtt",  # WebVTT (auto-detects YouTube VTT with word-level timestamps)
    "ass",
    "ssa",
    "sub",
    "sbv",
    "txt",
    "sami",
    "smi",
    # Tabular formats
    "csv",
    "tsv",
    "aud",
    "json",
    # Specialized formats
    "textgrid",  # Praat TextGrid
    "gemini",  # Gemini/YouTube transcript format
    # Professional NLE formats
    "avid_ds",
    "fcpxml",
    "premiere_xml",
    "audition_csv",
    # Special
    "auto",  # Auto-detect format
]

# Type alias for output caption formats (all formats with registered writers)
OutputCaptionFormat = Literal[
    # Standard subtitle formats
    "srt",
    "vtt",  # WebVTT (use karaoke.enabled=True for YouTube VTT style output)
    "ass",
    "ssa",
    "sub",
    "sbv",
    "txt",
    "sami",
    "smi",
    # Tabular formats
    "csv",
    "tsv",
    "aud",
    "json",
    # Specialized formats
    "textgrid",  # Praat TextGrid
    "gemini",  # Gemini/YouTube transcript format
    # TTML profiles (write-only)
    "ttml",  # Generic TTML
    "imsc1",  # IMSC1 (Netflix/streaming) TTML profile
    "ebu_tt_d",  # EBU-TT-D (European broadcast) TTML profile
    # Professional NLE formats
    "avid_ds",  # Avid Media Composer SubCap format
    "fcpxml",  # Final Cut Pro XML
    "premiere_xml",  # Adobe Premiere Pro XML (graphic clips)
    "audition_csv",  # Adobe Audition markers
    "edimarker_csv",  # Pro Tools (via EdiMarker) markers
]

# =============================================================================
# Runtime Format Lists (Derived from Type Definitions)
# =============================================================================

# Input caption formats list (derived from InputCaptionFormat)
INPUT_CAPTION_FORMATS: list[str] = list(get_args(InputCaptionFormat))

# Output caption formats list (derived from OutputCaptionFormat)
OUTPUT_CAPTION_FORMATS: list[str] = list(get_args(OutputCaptionFormat))

# Standard caption formats (formats with both reader and writer)
CAPTION_FORMATS: list[str] = ["srt", "vtt", "ass", "ssa", "sub", "sbv", "txt", "sami", "smi"]

# All caption formats combined (for file detection, excludes "auto")
ALL_CAPTION_FORMATS: list[str] = list(set(INPUT_CAPTION_FORMATS + OUTPUT_CAPTION_FORMATS) - {"auto"})
