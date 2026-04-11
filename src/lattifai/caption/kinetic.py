"""Kinetic typography styles for word-level ASS caption animation.

This module provides the "motion" layer that composes with karaoke_effect
(sweep/instant/outline) and karaoke_color_scheme. The three axes are
orthogonal:

    karaoke_effect       -> how to reveal     (\\k, \\kf, \\ko)
    karaoke_color_scheme -> what color        (12 presets)
    kinetic_style        -> how to move       (15 presets, this module)

Each preset is a KineticTemplate with two fields:

    initial      — static ASS override tags applied at event start. Required
                   for "entrance" effects (fade, pop, zoom, rise, blur_in,
                   typewriter) where the word must be hidden/small/blurry
                   BEFORE its activation time. Without this, the word is
                   fully visible at event start, and the animation looks
                   like "flash off → fade in" which reads as a flicker.

    transitions  — ordered list of (t1_ms, t2_ms, override_tags). At build
                   time each entry is rendered as
                   `\\t(ws+t1, ws+t2, tags)` where `ws` is the word's
                   cumulative start offset from the Dialogue event's
                   beginning. This makes every word animate at its own
                   activation time rather than all at event start.

The stagger style is special: it needs character-level time offsets, so
`expand_stagger_word()` wraps each character individually.

Horizontal scale (`\\fscx`) is DELIBERATELY avoided everywhere. libass
treats `\\fscx` as a change to the glyph's advance width, which triggers
line re-flow — subsequent words get pushed right as one word scales up,
creating visible horizontal jitter across the whole line. Vertical scale
(`\\fscy`) does not affect advance width and is safe. See the
`karaoke-vs-shortform-captions-research` plan for the full analysis.

ASS override tags used (all libass-compatible, no absolute positioning):
    \\fscy            — vertical scale (%)
    \\alpha          — transparency (&H00&=opaque ... &HFF&=transparent)
    \\frz            — z-axis rotation (degrees)
    \\bord            — outline width (px)
    \\blur            — Gaussian blur radius
    \\t(t1,t2,tags)  — time-based animation between two states

Deliberately NOT used: \\move, \\pos, \\org, \\fscx, \\fs — these either
require absolute coordinates or change horizontal advance width.
"""

from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional, Tuple

# =============================================================================
# Data model
# =============================================================================


@dataclass(frozen=True)
class KineticTemplate:
    """A word-level kinetic typography preset.

    Attributes:
        initial: Static ASS override tags applied at event start. Required
            for "entrance" animations so the word has the correct hidden /
            small / blurry state BEFORE its activation time. Empty string
            means "use the style defaults" (e.g. bounce, glow, pulse which
            start from the normal rendered state and animate on top).
        transitions: List of (t1_ms, t2_ms, tags) tuples rendered as
            `\\t(ws+t1, ws+t2, tags)` at the word's cumulative offset.
    """

    initial: str = ""
    transitions: Tuple[Tuple[int, int, str], ...] = field(default_factory=tuple)


# =============================================================================
# Public style names — keep in sync with ASSConfig.kinetic_style Literal
# =============================================================================

KINETIC_STYLE_NAMES: Tuple[str, ...] = (
    # Impact — high-energy word reveal
    "bounce",
    "pop",
    "shake",
    "pulse",
    "swing",
    # Smooth — graceful entrance
    "fade",
    "zoom",
    "rise",
    "typewriter",
    "blur_in",
    # Stylized — special rendering
    "glow",
    "neon",
    "wave",
    "flicker",
    "stagger",
)


# =============================================================================
# Template table
# =============================================================================

_KINETIC_TEMPLATES: "dict[str, KineticTemplate]" = {
    # ----- Impact ---------------------------------------------------------
    # bounce: word snaps to 130% vertical scale at activation and springs
    # back over 150ms. Vertical-only to avoid horizontal advance-width
    # reflow (would shake the entire line on rapid speech).
    "bounce": KineticTemplate(
        transitions=(
            (0, 1, r"\fscy130"),
            (1, 151, r"\fscy100"),
        ),
    ),
    # pop: word is vertically squashed and invisible from event start; at
    # its activation it grows and fades in over 120ms. The static initial
    # keeps it hidden until activation instead of flashing into view.
    "pop": KineticTemplate(
        initial=r"\fscy60\alpha&HFF&",
        transitions=((0, 120, r"\fscy100\alpha&H00&"),),
    ),
    # shake: three-stage rotation jitter. Rotation is around the glyph's
    # render origin and does not affect advance width.
    "shake": KineticTemplate(
        transitions=(
            (0, 60, r"\frz3"),
            (60, 120, r"\frz-3"),
            (120, 180, r"\frz0"),
        ),
    ),
    # pulse: 100% → 115% → 100% breathing over 400ms. Vertical-only.
    "pulse": KineticTemplate(
        transitions=(
            (0, 200, r"\fscy115"),
            (200, 400, r"\fscy100"),
        ),
    ),
    # swing: pendulum rotation -8° → 8° → 0° over 400ms.
    "swing": KineticTemplate(
        transitions=(
            (0, 1, r"\frz-8"),
            (1, 201, r"\frz8"),
            (201, 400, r"\frz0"),
        ),
    ),
    # ----- Smooth ---------------------------------------------------------
    # fade: word is fully transparent from event start; at activation it
    # fades to opaque over 150ms. No "flash off then fade in" bug.
    "fade": KineticTemplate(
        initial=r"\alpha&HFF&",
        transitions=((0, 150, r"\alpha&H00&"),),
    ),
    # zoom: word is vertically squashed from event start; at activation it
    # grows to full height over 150ms. Smooth, no elasticity.
    "zoom": KineticTemplate(
        initial=r"\fscy80",
        transitions=((0, 150, r"\fscy100"),),
    ),
    # rise: word has zero height from event start; at activation it rises
    # to full height over 180ms, looking like it's "sprouting" from baseline.
    "rise": KineticTemplate(
        initial=r"\fscy0",
        transitions=((0, 180, r"\fscy100"),),
    ),
    # typewriter: word is invisible from event start; at activation it
    # appears instantly (1 ms ramp). Hard cut, no easing.
    "typewriter": KineticTemplate(
        initial=r"\alpha&HFF&",
        transitions=((0, 1, r"\alpha&H00&"),),
    ),
    # blur_in: word is blurred from event start; at activation it sharpens
    # to zero blur over 150ms.
    "blur_in": KineticTemplate(
        initial=r"\blur4",
        transitions=((0, 150, r"\blur0"),),
    ),
    # ----- Stylized -------------------------------------------------------
    # glow: outline and blur pulse at activation. No initial state because
    # the word should look normal before activation; the glow is a pulse
    # on top of the normal render.
    "glow": KineticTemplate(
        transitions=(
            (0, 100, r"\bord4\blur3"),
            (100, 250, r"\bord2\blur1"),
        ),
    ),
    # neon: stronger glow pulse with a pre-state and a sustained tail.
    "neon": KineticTemplate(
        transitions=(
            (0, 1, r"\bord2\blur2"),
            (1, 151, r"\bord6\blur5"),
            (151, 400, r"\bord3\blur2"),
        ),
    ),
    # wave: vertical only up-down-settle ripple. No initial state.
    "wave": KineticTemplate(
        transitions=(
            (0, 200, r"\fscy110"),
            (200, 400, r"\fscy90"),
            (400, 600, r"\fscy100"),
        ),
    ),
    # flicker: two rapid alpha flashes at activation. Word is visible
    # before activation (default state), flickers briefly, then stays on.
    "flicker": KineticTemplate(
        transitions=(
            (0, 50, r"\alpha&HA0&"),
            (50, 100, r"\alpha&H00&"),
            (150, 200, r"\alpha&HA0&"),
            (200, 250, r"\alpha&H00&"),
        ),
    ),
    # stagger: char-level. Handled by expand_stagger_word() — the entry
    # here is a sentinel so validate_kinetic_style accepts the name.
    "stagger": KineticTemplate(),
}


# Styles that require character-level expansion instead of the transition
# table. Currently only `stagger` — see `expand_stagger_word()`.
_CHAR_LEVEL_STYLES: FrozenSet[str] = frozenset({"stagger"})


# =============================================================================
# Public API
# =============================================================================


def list_kinetic_styles() -> List[str]:
    """Return all 15 supported kinetic style names, in canonical order."""
    return list(KINETIC_STYLE_NAMES)


def validate_kinetic_style(style: Optional[str]) -> None:
    """Raise ValueError if `style` is not None and not a known style.

    Fail-fast: unsupported styles are a user error, not a silent fallback.
    """
    if style is None:
        return
    if style not in KINETIC_STYLE_NAMES:
        available = ", ".join(KINETIC_STYLE_NAMES)
        raise ValueError(f"Unknown kinetic_style: {style!r}. Available: {available}")


def is_char_level_style(style: Optional[str]) -> bool:
    """True if the style requires character-level expansion (stagger)."""
    return style in _CHAR_LEVEL_STYLES


def get_kinetic_template(style: str) -> KineticTemplate:
    """Return the KineticTemplate for a named style. Raises on unknown."""
    validate_kinetic_style(style)
    return _KINETIC_TEMPLATES[style]


def build_kinetic_overrides(style: Optional[str], word_start_ms: int) -> str:
    """Return concatenated static + `\\t(...)` ASS overrides for a word.

    Format: "<static tags><\\t block 1><\\t block 2>..."
        e.g. "\\alpha&HFF&\\t(450,600,\\alpha&H00&)"

    Args:
        style: Kinetic style name or None.
        word_start_ms: Cumulative ms from Dialogue event start to this word.

    Returns:
        Empty string if `style` is None or stagger (char-level is handled
        separately via expand_stagger_word).

    Raises:
        ValueError: unknown style name (fail-fast).
    """
    if style is None:
        return ""
    validate_kinetic_style(style)
    if is_char_level_style(style):
        return ""  # Stagger is handled separately via expand_stagger_word
    template = _KINETIC_TEMPLATES[style]
    transitions = "".join(
        f"\\t({word_start_ms + t1},{word_start_ms + t2},{tags})"
        for t1, t2, tags in template.transitions
    )
    return template.initial + transitions


# =============================================================================
# Stagger: character-level expansion
# =============================================================================

# Stagger defaults — kept conservative so a 3-char CJK word finishes well
# within 500ms (3 chars * 30ms delay + 100ms window = 190ms total).
_STAGGER_CHAR_DELAY_MS = 30
_STAGGER_CHAR_WINDOW_MS = 100
_STAGGER_INITIAL_FSCY = 60


def expand_stagger_word(
    word_text: str,
    word_start_ms: int,
    char_delay_ms: int = _STAGGER_CHAR_DELAY_MS,
    char_window_ms: int = _STAGGER_CHAR_WINDOW_MS,
) -> str:
    """Wrap each character of a word with staggered vertical-scale-in tags.

    Produces output like:
        {\\alpha&HFF&\\fscy60\\t(450,550,\\alpha&H00&\\fscy100)}H
        {\\alpha&HFF&\\fscy60\\t(480,580,\\alpha&H00&\\fscy100)}e
        {\\alpha&HFF&\\fscy60\\t(510,610,\\alpha&H00&\\fscy100)}l
        ...

    Each char is invisible and squashed from event start (static `initial`
    inside its own override block), then at `word_start_ms + i*char_delay_ms`
    it fades in and grows to full height over `char_window_ms`.

    Uses `\\fscy` only (no `\\fscx`) to avoid horizontal reflow. Handles CJK
    correctly because Python string iteration yields code points.

    Args:
        word_text: Raw word text (may contain CJK, Latin, mixed).
        word_start_ms: Cumulative ms from Dialogue event start to this word.
        char_delay_ms: Delay between successive characters.
        char_window_ms: Settle animation duration per character.

    Returns:
        Tag-wrapped text ready to be placed after the karaoke `\\k{cs}` tag.
    """
    if not word_text:
        return ""

    parts: List[str] = []
    for i, ch in enumerate(word_text):
        ch_start = word_start_ms + i * char_delay_ms
        ch_settle = ch_start + char_window_ms
        parts.append(
            f"{{\\alpha&HFF&\\fscy{_STAGGER_INITIAL_FSCY}"
            f"\\t({ch_start},{ch_settle},\\alpha&H00&\\fscy100)}}{ch}"
        )
    return "".join(parts)
