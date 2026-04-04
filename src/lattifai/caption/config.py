"""Caption I/O configuration for LattifAI."""

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, get_args

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


@dataclass
class CaptionStyle:
    """Caption style configuration for ASS/TTML formats.

    Attributes:
        primary_color: Main text color (#RRGGBB)
        secondary_color: Secondary/highlight color (#RRGGBB)
        outline_color: Text outline color (#RRGGBB)
        back_color: Shadow color (#RRGGBB)
        font_name: Font family name (use CaptionFonts constants or any system font)
        font_size: Font size in points
        bold: Enable bold text
        italic: Enable italic text
        outline_width: Outline thickness
        shadow_depth: Shadow distance
        alignment: ASS alignment (1-9, numpad style), 2=bottom-center
        margin_l: Left margin in pixels
        margin_r: Right margin in pixels
        margin_v: Vertical margin in pixels
    """

    # Colors (#RRGGBB format)
    primary_color: str = "#FFFFFF"
    secondary_color: str = "#00FFFF"
    outline_color: str = "#000000"
    back_color: str = "#000000"

    # Font
    font_name: str = CaptionFonts.ARIAL
    font_size: int = 20
    bold: bool = False
    italic: bool = False

    # Border and shadow
    outline_width: float = 0
    shadow_depth: float = 1.0

    # Position
    alignment: int = 2
    margin_l: int = 20
    margin_r: int = 20
    margin_v: int = 20


@dataclass
class KaraokeConfig:
    """Karaoke export configuration.

    Attributes:
        enabled: Whether karaoke mode is enabled
        effect: Karaoke effect type
            - "sweep": Gradual fill from left to right (ASS \\kf tag)
            - "instant": Instant highlight (ASS \\k tag)
            - "outline": Outline then fill (ASS \\ko tag)
        style: Caption style configuration (font, colors, position)
        lrc_precision: LRC time precision ("centisecond" or "millisecond")
        lrc_metadata: LRC metadata dict (ar, ti, al, etc.)
        ttml_timing_mode: TTML timing attribute ("Word" or "Line")
    """

    enabled: bool = False
    effect: Literal["sweep", "instant", "outline"] = "sweep"
    color_scheme: str = ""
    """Karaoke color scheme name. When set, overrides style colors.
    Available schemes: azure-gold, sakura-purple, mint-ocean, gardenia-green,
    sunset-warm, prussian-elegant, burgundy-classic, langgan-spring,
    mars-teal, spring-field, navy-pink, apricot-dark.
    Use "" (empty) for manual style configuration."""

    style: CaptionStyle = field(default_factory=CaptionStyle)

    # LRC specific
    lrc_precision: Literal["centisecond", "millisecond"] = "millisecond"
    lrc_metadata: Dict[str, str] = field(default_factory=dict)

    # TTML specific
    ttml_timing_mode: Literal["Word", "Line"] = "Word"

    def __post_init__(self):
        """Apply color scheme to style if one is specified."""
        if self.color_scheme:
            resolved = resolve_karaoke_color_scheme(self.color_scheme)
            if resolved:
                self.style.primary_color = resolved["primary_color"]
                self.style.secondary_color = resolved["secondary_color"]
                self.style.outline_color = resolved["outline_color"]
                self.style.back_color = resolved["back_color"]
                if "outline_width" in resolved:
                    self.style.outline_width = resolved["outline_width"]
                if "shadow_depth" in resolved:
                    self.style.shadow_depth = resolved["shadow_depth"]


# Karaoke color schemes — curated from 不二创艺, 徐挺好, 色彩中国, 莱利纺织
# Each scheme: primary (unsung text), secondary (highlight sweep), outline, back (shadow)
KARAOKE_COLOR_SCHEMES: Dict[str, Dict[str, str]] = {
    "azure-gold": {
        "primary_color": "#FFFFFF",
        "secondary_color": "#FFC209",  # 金柠暖阳
        "outline_color": "#1387C0",  # 晴空海蓝
        "back_color": "#0A3D5C",
        "outline_width": 2.0,
    },
    "sakura-purple": {
        "primary_color": "#F7C3D9",  # 柔樱粉
        "secondary_color": "#7953B1",  # 紫鸢深紫
        "outline_color": "#063C85",  # 深海藏蓝
        "back_color": "#1A1A2E",
        "outline_width": 2.0,
    },
    "mint-ocean": {
        "primary_color": "#A1FEEF",  # 薄荷冰青
        "secondary_color": "#658AE4",  # 柔空蓝
        "outline_color": "#28314E",  # 深海夜蓝
        "back_color": "#0A0A1A",
        "outline_width": 2.0,
    },
    "gardenia-green": {
        "primary_color": "#FFFFFF",
        "secondary_color": "#9DC92A",  # 苹果绿
        "outline_color": "#77964A",  # 碧山
        "back_color": "#1C2B1A",
        "outline_width": 2.0,
    },
    "sunset-warm": {
        "primary_color": "#FAEDD1",  # 奶油米白
        "secondary_color": "#F4520D",  # 暖橙光
        "outline_color": "#1387C0",  # 晴空海蓝
        "back_color": "#0A1628",
        "outline_width": 2.0,
    },
    "prussian-elegant": {
        "primary_color": "#FFFFFF",
        "secondary_color": "#FBC03D",  # 栀子黄
        "outline_color": "#003153",  # 普鲁士蓝
        "back_color": "#001A2C",
        "outline_width": 2.5,
    },
    "burgundy-classic": {
        "primary_color": "#F7F2DF",  # 宣纸白
        "secondary_color": "#CC5D84",  # 琅玕紫
        "outline_color": "#800020",  # 勃艮第红
        "back_color": "#2A000D",
        "outline_width": 2.0,
    },
    "langgan-spring": {
        "primary_color": "#C1D796",  # 春辰 (unsung text)
        "secondary_color": "#CC5D84",  # 琅玕紫 (highlight sweep)
        "outline_color": "#8A3A5A",  # 琅玕紫暗化
        "back_color": "#2A1020",
        "outline_width": 2.0,
    },
    "mars-teal": {
        "primary_color": "#FFFFFF",
        "secondary_color": "#008C8C",  # 马尔斯绿
        "outline_color": "#003153",  # 普鲁士蓝
        "back_color": "#001A1A",
        "outline_width": 2.0,
    },
    "spring-field": {
        "primary_color": "#FBFFF2",  # 荔枝白
        "secondary_color": "#46B065",  # Spring Fields 中绿
        "outline_color": "#008E6B",  # Spring Fields 深绿
        "back_color": "#0A2A1A",
        "outline_width": 2.0,
    },
    "navy-pink": {
        "primary_color": "#FFFFFF",
        "secondary_color": "#F7C3D9",  # 柔樱粉
        "outline_color": "#063C85",  # 深海藏蓝
        "back_color": "#021A3A",
        "outline_width": 2.0,
    },
    "apricot-dark": {
        "primary_color": "#FEA72E",  # 杏黄
        "secondary_color": "#F7F2DF",  # 宣纸白
        "outline_color": "#3A3C50",  # 玄青
        "back_color": "#1A1A28",
        "outline_width": 2.0,
    },
}


def resolve_karaoke_color_scheme(name: str) -> Optional[Dict]:
    """Resolve a karaoke color scheme name to style dict. Returns None if not found."""
    return KARAOKE_COLOR_SCHEMES.get(name.lower().strip())


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
    "vtt",  # WebVTT (use karaoke_config.enabled=True for YouTube VTT style output)
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
