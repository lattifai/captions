"""Kinetic typography styles for word-level ASS caption animation.

This module provides the "motion" layer that composes with karaoke_effect
(sweep/instant/outline) and karaoke_color_scheme. The three axes are
orthogonal:

    karaoke_effect       -> how to reveal     (\\k, \\kf, \\ko)
    karaoke_color_scheme -> what color        (12 presets)
    kinetic_style        -> how to move       (15 presets, this module)

All 14 word-level styles are implemented as ordered lists of transition
tuples `(t1_ms, t2_ms, override_tags)`. At build time each transition is
rendered as an ASS `\\t(ws+t1, ws+t2, tags)` block where `ws` is the word's
cumulative start offset from the Dialogue event's beginning. This makes
every word animate at its own activation time rather than all at event
start — which is what differentiates a real kinetic caption from a static
override.

The stagger style is special: it needs character-level time offsets, so it
is handled by `expand_stagger_word()` instead of the transition table.

ASS override tags used (all libass-compatible, no absolute positioning):
    \\fscx, \\fscy   — horizontal/vertical scale (%)
    \\alpha          — transparency (&H00&=opaque ... &HFF&=transparent)
    \\frz            — z-axis rotation (degrees)
    \\bord            — outline width (px)
    \\blur            — Gaussian blur radius
    \\t(t1,t2,tags)  — time-based animation between two states

Deliberately NOT used: \\move, \\pos, \\org — these require absolute
coordinates and break libass automatic layout.
"""

from typing import Dict, FrozenSet, List, Optional, Tuple

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
# Word-level transition tables
# =============================================================================

# Each entry: list of (t1_ms, t2_ms, ass_override_tags).
# At build time the writer adds `word_start_ms` to both t1 and t2 and wraps
# as `\\t(t1,t2,tags)`. Multiple tuples are concatenated; libass applies
# them in order and animates the current text state through each window.
#
# Convention: use a 1-ms (t, t+1) window for instant state jumps (e.g. set
# the initial scale before the settle animation). This keeps the "jump" from
# being visually interpolated while remaining a valid `\\t` block.

_KINETIC_TRANSITIONS: Dict[str, List[Tuple[int, int, str]]] = {
    # ----- Impact ---------------------------------------------------------
    "bounce": [
        (0, 1, r"\fscx120\fscy120"),
        (1, 151, r"\fscx100\fscy100"),
    ],
    "pop": [
        (0, 1, r"\fscx60\fscy60\alpha&HFF&"),
        (1, 121, r"\fscx100\fscy100\alpha&H00&"),
    ],
    "shake": [
        (0, 60, r"\frz3"),
        (60, 120, r"\frz-3"),
        (120, 180, r"\frz0"),
    ],
    "pulse": [
        (0, 200, r"\fscx110\fscy110"),
        (200, 400, r"\fscx100\fscy100"),
    ],
    "swing": [
        (0, 1, r"\frz-8"),
        (1, 201, r"\frz8"),
        (201, 400, r"\frz0"),
    ],
    # ----- Smooth ---------------------------------------------------------
    "fade": [
        (0, 1, r"\alpha&HFF&"),
        (1, 151, r"\alpha&H00&"),
    ],
    "zoom": [
        (0, 1, r"\fscx80\fscy80"),
        (1, 151, r"\fscx100\fscy100"),
    ],
    "rise": [
        (0, 1, r"\fscy0"),
        (1, 181, r"\fscy100"),
    ],
    "typewriter": [],  # Hard cut — relies on \k alone, no extra transitions
    "blur_in": [
        (0, 1, r"\blur4"),
        (1, 151, r"\blur0"),
    ],
    # ----- Stylized -------------------------------------------------------
    "glow": [
        (0, 100, r"\bord4\blur3"),
        (100, 250, r"\bord2\blur1"),
    ],
    "neon": [
        (0, 1, r"\bord2\blur2"),
        (1, 151, r"\bord6\blur5"),
        (151, 400, r"\bord3\blur2"),
    ],
    "wave": [
        (0, 200, r"\fscy110"),
        (200, 400, r"\fscy90"),
        (400, 600, r"\fscy100"),
    ],
    "flicker": [
        (0, 50, r"\alpha&HA0&"),
        (50, 100, r"\alpha&H00&"),
        (150, 200, r"\alpha&HA0&"),
        (200, 250, r"\alpha&H00&"),
    ],
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
    """True if the style requires character-level expansion (stagger).

    Char-level styles cannot use the normal word-prefix pipeline; the writer
    must call `expand_stagger_word()` to replace the word text with a
    per-character tagged sequence.
    """
    return style in _CHAR_LEVEL_STYLES


def build_kinetic_overrides(style: Optional[str], word_start_ms: int) -> str:
    """Return concatenated `\\t(...)` ASS override blocks for a word.

    Args:
        style: Kinetic style name or None.
        word_start_ms: Cumulative ms from Dialogue event start to this word.

    Returns:
        Empty string if `style` is None, typewriter, or stagger.
        Otherwise a string like `\\t(450,451,\\fscx120\\fscy120)\\t(451,601,\\fscx100\\fscy100)`.

    Raises:
        ValueError: unknown style name (fail-fast).
    """
    if style is None:
        return ""
    validate_kinetic_style(style)
    if is_char_level_style(style):
        return ""  # Stagger is handled separately via expand_stagger_word
    transitions = _KINETIC_TRANSITIONS[style]
    if not transitions:
        return ""  # typewriter
    return "".join(
        f"\\t({word_start_ms + t1},{word_start_ms + t2},{tags})"
        for t1, t2, tags in transitions
    )


# =============================================================================
# Stagger: character-level expansion
# =============================================================================

# Stagger defaults — kept conservative so a 3-char CJK word finishes well
# within 500ms (3 chars * 30ms delay + 100ms window = 190ms total).
_STAGGER_CHAR_DELAY_MS = 30
_STAGGER_CHAR_WINDOW_MS = 100
_STAGGER_INITIAL_SCALE = 60


def expand_stagger_word(
    word_text: str,
    word_start_ms: int,
    char_delay_ms: int = _STAGGER_CHAR_DELAY_MS,
    char_window_ms: int = _STAGGER_CHAR_WINDOW_MS,
) -> str:
    """Wrap each character of a word with staggered scale-in tags.

    Produces output like:
        {\\t(450,451,\\fscx60\\fscy60)\\t(451,551,\\fscx100\\fscy100)}H
        {\\t(480,481,\\fscx60\\fscy60)\\t(481,581,\\fscx100\\fscy100)}e
        {\\t(510,511,\\fscx60\\fscy60)\\t(511,611,\\fscx100\\fscy100)}l
        ...

    Each char enters at `word_start_ms + i * char_delay_ms` and settles
    over `char_window_ms`. Handles CJK correctly because Python string
    iteration yields code points.

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
            f"{{\\t({ch_start},{ch_start + 1},"
            f"\\fscx{_STAGGER_INITIAL_SCALE}\\fscy{_STAGGER_INITIAL_SCALE})"
            f"\\t({ch_start + 1},{ch_settle},\\fscx100\\fscy100)}}{ch}"
        )
    return "".join(parts)
