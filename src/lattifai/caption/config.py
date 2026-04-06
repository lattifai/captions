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
    system font name as the font_name parameter in ASSConfig.
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


@dataclass
class OutputBehavior:
    """Output behavior configuration for caption writing.

    Controls how caption content is structured in the output, independent of
    visual styling or format-specific rendering.
    """

    include_speaker_in_text: bool = True
    """Include speaker labels in caption text (e.g., 'Alice: Hello' vs 'Hello')."""

    word_level: bool = False
    """Word-level output: word-per-segment in normal mode, word timestamps in karaoke/JSON."""

    translation_first: bool = False
    """Place translation text above original text in bilingual output."""


@dataclass
class ASSConfig:
    """Self-contained configuration for ASS/SSA export.

    Includes rendering context (PlayRes, wrap), visual styling (font, colors,
    outline, shadow), and positioning (alignment, margins).
    """

    # -- Visual style --

    font_name: str = CaptionFonts.ARIAL
    """Font family name."""

    font_size: int = 48
    """Font size in points (relative to PlayRes)."""

    primary_color: str = "#FFFFFF"
    """Main text color (#RRGGBB)."""

    secondary_color: str = "#00FFFF"
    """Karaoke sweep target color (#RRGGBB)."""

    outline_color: str = "#000000"
    """Text outline color (#RRGGBB)."""

    back_color: str = "#000000"
    """Shadow/back color (#RRGGBB). In borderstyle=1 this is the drop shadow color."""

    bold: bool = False
    """Enable bold text."""

    italic: bool = False
    """Enable italic text."""

    background_color: str = ""
    """Subtitle background box color (#RRGGBB or #RRGGBBAA).
    When set, switches to borderstyle=3 (opaque box)."""

    outline_width: float = 0
    """Outline thickness (px). Recommended 2.0-2.5 for karaoke."""

    shadow_depth: float = 1.0
    """Shadow distance (px). Set to 0 for karaoke."""

    # -- Positioning --

    alignment: int = 2
    """ASS alignment (1-9, numpad style). 2=bottom-center."""

    margin_l: int = 20
    """Left margin in pixels."""

    margin_r: int = 20
    """Right margin in pixels."""

    margin_v: int = 20
    """Vertical margin in pixels."""

    # -- Rendering context --

    play_res_x: int = 1920
    """Reference resolution width for coordinate scaling."""

    play_res_y: int = 1080
    """Reference resolution height for coordinate scaling."""

    scaled_border_and_shadow: bool = True
    """Scale border/shadow with PlayRes (ScaledBorderAndShadow: yes)."""

    wrap_style: int = 0
    """Line wrapping mode. 0=smart, 1=EOL, 2=none, 3=smart+lower wide."""

    # -- Speaker --

    speaker_color: str = ""
    """Speaker name color mode for ASS override tags.
    - "":           no special color (default)
    - "#RRGGBB":    single color for all speakers
    - "#RRGGBB,#00BFFF,...": comma-separated, auto-assigned per speaker
    - "auto":       built-in 10-color palette, auto-assigned per speaker
    """


@dataclass
class KaraokeConfig:
    """Karaoke export configuration.

    Karaoke-specific settings only. Subtitle styling (font, colors, background)
    lives in ASSConfig, not here. Format-specific settings live in their own
    config classes (LRCConfig, TTMLConfig, etc.).
    """

    enabled: bool = False
    """Whether karaoke mode is enabled."""

    effect: Literal["sweep", "instant", "outline"] = "sweep"
    """Karaoke effect type:
    - "sweep": Gradual fill from left to right (ASS \\kf tag)
    - "instant": Instant highlight (ASS \\k tag)
    - "outline": Outline then fill (ASS \\ko tag)
    """

    color_scheme: str = ""
    """Karaoke color scheme name. When set, overrides style colors.
    See KARAOKE_COLOR_SCHEMES in colors.py for available schemes.
    Use "" (empty) for manual style configuration."""


@dataclass
class LRCConfig:
    """Configuration for LRC lyric format export.

    Pass as format_config to Caption.write() for LRC output.
    """

    precision: Literal["centisecond", "millisecond"] = "millisecond"
    """Timestamp precision: "centisecond" ([mm:ss.xx]) or "millisecond" ([mm:ss.xxx])."""

    metadata: Dict[str, str] = field(default_factory=dict)
    """LRC metadata header fields (ar, ti, al, by, offset, etc.)."""


def apply_color_scheme(scheme_name: str, config: Optional[ASSConfig] = None) -> ASSConfig:
    """Apply karaoke color scheme to ASSConfig.

    All color fields (primary, secondary, outline, back, background) are in ASSConfig.
    The input config is never mutated — a new instance is returned.

    Args:
        scheme_name: Color scheme name (e.g., "azure-gold")
        config: Source ASSConfig (not modified). Defaults to ASSConfig().

    Returns:
        New ASSConfig with colors overridden, or the original config
        unchanged if the scheme is not found.
    """
    from dataclasses import replace

    config = config or ASSConfig()

    resolved = resolve_karaoke_color_scheme(scheme_name)
    if not resolved:
        return config

    overrides = {}
    for key in (
        "primary_color",
        "secondary_color",
        "outline_color",
        "back_color",
        "shadow_depth",
        "outline_width",
        "background_color",
    ):
        if key in resolved:
            overrides[key] = resolved[key]

    return replace(config, **overrides) if overrides else config


@dataclass
class StandardizationConfig:
    """Caption standardization configuration following broadcast guidelines.

    Reference Standards:
    - Netflix Timed Text Style Guide
    - BBC Subtitle Guidelines
    - EBU-TT-D Standard
    """

    min_duration: float = 0.8
    """Minimum segment duration (seconds). Netflix recommends 5/6s, BBC 0.3s."""

    max_duration: float = 7.0
    """Maximum segment duration (seconds). Netflix/BBC recommends 7s."""

    min_gap: float = 0.08
    """Minimum gap between segments (seconds). 80ms prevents subtitle flicker."""

    max_lines: int = 2
    """Maximum lines per segment. Broadcast standard is typically 2."""

    max_chars_per_line: int = 42
    """Maximum characters per line. CJK auto-adjusted by ÷2 (e.g., 42 → 21)."""

    optimal_cps: float = 17.0
    """Optimal reading speed (chars/sec). Netflix recommends 17-20 CPS."""

    start_margin: Optional[float] = None
    """Start margin (seconds) before first word. None = no adjustment (default)."""

    end_margin: Optional[float] = None
    """End margin (seconds) after last word. None = no adjustment (default)."""

    margin_collision_mode: Literal["trim", "gap"] = "trim"
    """How to handle collisions: 'trim' (reduce margin) or 'gap' (maintain min_gap)."""

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
