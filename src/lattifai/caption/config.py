"""Caption I/O configuration for LattifAI."""

from dataclasses import dataclass, field, fields
from typing import Any, Dict, Literal, Optional, get_args  # noqa: F401

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
class RenderConfig:
    """Cross-format rendering configuration for caption output.

    Controls how caption content is structured in the output, independent of
    visual styling or format-specific rendering.
    """

    include_speaker_in_text: bool = True
    """Include speaker labels in caption text (e.g., 'Alice: Hello' vs 'Hello')."""

    word_level: Optional[bool] = None
    """Tri-state word-level output control.

    - None (default): per-format default. Renderer formats (SRT, VTT, ASS,
      LRC, TTML, SRV3, Premiere, FCPXML) stay segment-level — equivalent to
      the historical ``False`` behavior, so existing pipelines see no change.
      Lossless serializers (JSON, TextGrid) preserve word data when
      ``alignment["word"]`` is non-empty. ASS karaoke is triggered by
      ``ASSConfig.karaoke_effect`` alone, independent of this flag.
    - True: force word-level output. Renderers emit per-word cues / inline
      timestamps; if word alignment is missing or empty, a warning is logged
      and the writer degrades gracefully to segment-level output for that
      supervision (mixed batches emit a partial-fallback warning naming the
      unaligned count).
    - False: force segment-level output. Lossless writers (JSON / TextGrid)
      drop their word data; ASS karaoke is disabled even when
      ``karaoke_effect`` is set (a warning is emitted).
    """

    translation_first: bool = False
    """Place translation text above original text in bilingual output."""

    speaker_color: str = ""
    """Speaker color spec for formats that support inline coloring.

    - "": no coloring (default)
    - "auto": use built-in SPEAKER_PALETTE
    - "#RRGGBB": single color for all speakers
    - "#RRGGBB,#00BFFF,...": comma-separated palette, auto-assigned per speaker

    Format-specific behavior:
    - ASS: wraps speaker with {\\c&HBBGGRR&}...{\\c} override tags
    - SRT: wraps speaker with <font color="#RRGGBB">...</font> HTML tags
    - VTT: wraps speaker with <c.color>...</c> or <font> tags

    Format-specific config (ASSConfig.speaker_color, SRTConfig.speaker_color)
    takes precedence over this field when set.
    """


@dataclass
class ASSConfig:
    """Self-contained configuration for ASS/SSA export.

    Includes rendering context (PlayRes, wrap), visual styling (font, colors,
    outline, shadow), and positioning (alignment, margins).
    """

    # -- Visual style --

    font_name: str = CaptionFonts.ARIAL
    """Font family name."""

    font_size: int = 64
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

    underline: bool = False
    """Enable underline text."""

    strikeout: bool = False
    """Enable strikeout (strikethrough) text."""

    scalex: float = 100.0
    """Horizontal text scaling (%). 100=normal."""

    scaley: float = 100.0
    """Vertical text scaling (%). 100=normal.
    Kinetic presets (zoom, rise, bounce, etc.) use this as their baseline."""

    spacing: float = 0.0
    """Extra inter-character spacing (px). 0=normal."""

    angle: float = 0.0
    """Text rotation angle (degrees, counterclockwise). 0=normal.
    Kinetic presets (shake, swing, wave) use this as their baseline."""

    background_color: str = ""
    """Subtitle background box color (#RRGGBB or #RRGGBBAA).
    When set and borderstyle is None, auto-switches to borderstyle=3 (opaque box)."""

    borderstyle: Optional[int] = None
    """ASS BorderStyle (1=outline+shadow, 3=opaque box).
    None=auto-derive: borderstyle=3 when background_color is set, else 1.
    Explicit value takes precedence over background_color logic."""

    outline_width: float = 0
    """Outline thickness (px). Recommended 2.0-2.5 for karaoke.
    In borderstyle=3 (opaque box) this controls box padding."""

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

    # -- Karaoke --

    karaoke_effect: Optional[Literal["sweep", "instant", "outline"]] = None
    """Karaoke effect type. None = karaoke disabled.
    - "sweep": Gradual fill from left to right (ASS \\kf tag)
    - "instant": Instant highlight (ASS \\k tag)
    - "outline": Outline then fill (ASS \\ko tag)
    """

    karaoke_color_scheme: str = ""
    """Karaoke color scheme name. When set, overrides style colors.
    See KARAOKE_COLOR_SCHEMES in colors.py for available schemes."""

    translation_color: Optional[str] = None
    """Color for the translation line in bilingual karaoke mode.

    Why this exists: in bilingual karaoke (target text + translation on a
    second \\N line), libass treats the translation as a trailing karaoke
    syllable. While the karaoke is mid-sweep the translation tracks the
    \\k color tween, then snaps to PrimaryColour when the sweep ends —
    visible as a color "jump". The fix is to (a) reset karaoke state via
    \\rKaraoke and (b) lock both \\1c + \\2c so any residual animation
    becomes invisible.

    Accepted values:
        None | "primary"   → use Karaoke style PrimaryColour (the
                             "sung/filled" end state — most cohesive)
        "secondary"        → use Karaoke style SecondaryColour (the
                             "unsung/initial" start state)
        "#RRGGBB"          → explicit color, regardless of scheme

    Default is None → primary, because the eye reads the translation as
    the visual "rest position" alongside the karaoke's end state."""

    kinetic_style: Optional[
        Literal[
            "bounce",
            "pop",
            "shake",
            "pulse",
            "swing",
            "spotlight",
            "fade",
            "zoom",
            "rise",
            "reveal",
            "typewriter",
            "blur_in",
            "glow",
            "neon",
            "wave",
            "flicker",
            "stagger",
        ]
    ] = None
    """Word-level kinetic typography style (the 'motion' layer).

    Composes orthogonally with karaoke_effect and karaoke_color_scheme:
        karaoke_effect       -> how to reveal (sweep/instant/outline)
        karaoke_color_scheme -> what color    (13 presets)
        kinetic_style        -> how to move   (17 presets)

    Styles grouped by feel:
        Impact:    bounce, pop, shake, pulse, swing, spotlight
        Smooth:    fade, zoom, rise, reveal, typewriter, blur_in
        Stylized:  glow, neon, wave, flicker, stagger

    Unknown values raise ValueError at construction (fail-fast).
    """

    # -- High-level style preset --

    style_preset: Optional[
        Literal[
            "classic",
            "tiktok",
            "modern_box",
            "cinematic",
            "outline",
            "bold_center",
        ]
    ] = None
    """High-level visual preset that bundles font / colour / alignment /
    karaoke / kinetic values into a single named look.

    Composition: a preset only fills fields the user did NOT explicitly
    override. Any kwarg passed alongside ``style_preset`` wins over the
    preset's value. See ``styles.py`` for the registered presets.
    """

    @classmethod
    def from_preset(cls, name: str, **overrides: Any) -> "ASSConfig":
        """Build an ASSConfig from a named preset, optionally overriding
        individual fields. Equivalent to ``ASSConfig(style_preset=name,
        **overrides)`` but reads more declaratively at call sites."""
        return cls(style_preset=name, **overrides)

    def __post_init__(self) -> None:
        # 1) Apply preset BEFORE validation so spotlight/kinetic introduced
        # by the preset gets validated below.
        if self.style_preset is not None:
            self._apply_style_preset()

        # 2) Validate kinetic_style (after preset injection — preset may
        # have set it).
        # Lazy import: kinetic imports nothing from config, but we keep
        # the dependency direction one-way to avoid circulars.
        from .kinetic import validate_kinetic_style

        validate_kinetic_style(self.kinetic_style)

    def _apply_style_preset(self) -> None:
        """Fill unset fields from the named preset.

        A field is considered "unset" when its current value equals the
        dataclass default. Explicit overrides (kwargs that happen to equal
        the default) will be re-filled by the preset — this is acceptable
        because the user's intent in that case is identical to the
        preset's value.
        """
        from .styles import resolve_style_preset

        preset = resolve_style_preset(self.style_preset)

        # Build a {field_name: default_value} map once.
        defaults = {f.name: f.default for f in fields(self) if f.name != "style_preset"}

        for fname, pvalue in preset.items():
            if fname not in defaults:
                # Preset references a field not declared on ASSConfig —
                # styles.py is misaligned with the dataclass schema.
                raise AttributeError(
                    f"style_preset {self.style_preset!r} sets unknown ASSConfig field {fname!r}"
                )
            current = getattr(self, fname)
            if current != defaults[fname]:
                # User explicitly set this field — preset stays out.
                continue
            setattr(self, fname, pvalue)


@dataclass
class SRTConfig:
    """Configuration for SRT subtitle format export.

    Pass as format_config to Caption.write() for SRT output.
    SRT supports limited HTML tags (<b>, <i>, <u>, <font color>)
    in most modern players (VLC, PotPlayer, etc.).
    """

    speaker_color: str = ""
    """Speaker color spec: "", "auto", "#RRGGBB", or comma-separated palette.
    When set, wraps speaker prefix with <font color="#RRGGBB">...</font> tag."""


@dataclass
class VTTConfig:
    """Configuration for WebVTT export.

    Pass as format_config to Caption.write() for VTT output.
    Controls cue positioning, voice tags, and inline formatting.
    """

    # -- Default cue settings (applied when supervision lacks vtt_* custom fields) --

    default_align: Optional[str] = None
    """Default text alignment for all cues (start/center/end/left/right)."""

    default_line: Optional[str] = None
    """Default vertical position (percentage like '90%' or integer like '-1')."""

    default_position: Optional[str] = None
    """Default horizontal position (percentage like '20%')."""

    default_size: Optional[str] = None
    """Default cue box width (percentage like '80%')."""

    default_vertical: Optional[str] = None
    """Default writing direction ('rl' or 'lr' for vertical text)."""

    # -- Voice tag behavior --

    voice_tag: bool = False
    """Use <v Speaker>text</v> tags for speaker labeling.
    When False, uses 'Speaker: text' prefix (default, backward-compatible)."""

    # -- Speaker --

    speaker_color: str = ""
    """Speaker color spec for VTT output.
    - "": no coloring (default)
    - "auto": use built-in SPEAKER_PALETTE
    - "#RRGGBB": single color for all speakers
    - "#RRGGBB,#00BFFF,...": comma-separated palette, auto-assigned per speaker

    Wraps speaker prefix with <font color="#RRGGBB">...</font> tags.
    Takes precedence over RenderConfig.speaker_color when set."""

    # -- Inline formatting --

    preserve_formatting: bool = True
    """Preserve <b>, <i>, <u>, <c>, <ruby>, <lang> inline tags in output.
    When False, strips all inline tags from text."""

    def __post_init__(self) -> None:
        valid_aligns = {None, "start", "center", "end", "left", "right"}
        if self.default_align not in valid_aligns:
            raise ValueError(f"default_align must be one of {valid_aligns}, got {self.default_align!r}")
        valid_verticals = {None, "rl", "lr"}
        if self.default_vertical not in valid_verticals:
            raise ValueError(f"default_vertical must be one of {valid_verticals}, got {self.default_vertical!r}")


@dataclass
class LRCConfig:
    """Configuration for LRC lyric format export.

    Pass as format_config to Caption.write() for LRC output.
    """

    precision: Literal["centisecond", "millisecond"] = "millisecond"
    """Timestamp precision: "centisecond" ([mm:ss.xx]) or "millisecond" ([mm:ss.xxx])."""

    metadata: Dict[str, str] = field(default_factory=dict)
    """LRC metadata header fields (ar, ti, al, by, offset, etc.)."""


def apply_color_scheme(
    scheme_name: str, config: Optional[ASSConfig] = None
) -> ASSConfig:
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
CAPTION_FORMATS: list[str] = [
    "srt",
    "vtt",
    "ass",
    "ssa",
    "sub",
    "sbv",
    "txt",
    "sami",
    "smi",
]

# All caption formats combined (for file detection, excludes "auto")
ALL_CAPTION_FORMATS: list[str] = list(
    set(INPUT_CAPTION_FORMATS + OUTPUT_CAPTION_FORMATS) - {"auto"}
)


# =============================================================================
# Format Config Resolution
# =============================================================================

# 格式 → config 类映射（延迟导入避免循环依赖）
_FORMAT_CONFIG_MAP: Optional[Dict[str, type]] = None


def _get_format_config_map() -> Dict[str, type]:
    """Lazy-init format → config class mapping."""
    global _FORMAT_CONFIG_MAP
    if _FORMAT_CONFIG_MAP is None:
        from .formats.nle.audition import AuditionCSVConfig, EdiMarkerConfig
        from .formats.nle.avid import AvidDSConfig
        from .formats.nle.fcpxml import FCPXMLConfig
        from .formats.nle.premiere import PremiereXMLConfig
        from .formats.ttml import TTMLConfig

        _FORMAT_CONFIG_MAP = {
            "ass": ASSConfig,
            "ssa": ASSConfig,
            "srt": SRTConfig,
            "vtt": VTTConfig,
            "lrc": LRCConfig,
            "ttml": TTMLConfig,
            "imsc1": TTMLConfig,
            "ebu_tt_d": TTMLConfig,
            "premiere_xml": PremiereXMLConfig,
            "fcpxml": FCPXMLConfig,
            "avid_ds": AvidDSConfig,
            "audition_csv": AuditionCSVConfig,
            "edimarker_csv": EdiMarkerConfig,
        }
    return _FORMAT_CONFIG_MAP


def resolve_format_config(output_format: str, config_dict: Optional[Dict] = None):
    """Convert a plain dict to the appropriate format-specific config dataclass.

    Args:
        output_format: Output caption format (e.g., 'ass', 'lrc', 'ttml').
        config_dict: Dictionary of config values. None returns None.

    Returns:
        Instantiated config dataclass, or None if format has no config or dict is None.
    """
    if not config_dict:
        return None
    config_cls = _get_format_config_map().get(output_format)
    if not config_cls:
        return None
    return config_cls(**config_dict)
