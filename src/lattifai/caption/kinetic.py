"""Kinetic typography styles for ASS caption animation.

This module provides the "motion" layer that composes with karaoke_effect
(sweep/instant/outline) and karaoke_color_scheme. The three axes are
orthogonal:

    karaoke_effect       -> how to reveal     (\\k, \\kf, \\ko)
    karaoke_color_scheme -> what color        (12 presets)
    kinetic_style        -> how to move       (15 presets, this module)

Each preset (KineticPreset) holds up to two implementations (KineticImpl),
one for each scope:

    scope="line"  — a single override block at the event start; all text
                    animates uniformly together. Used when the caller does
                    NOT have word-level alignment, or when the caller
                    explicitly wants a whole-line entrance. Safe to use
                    \\fscy, \\fad, \\blur, \\bord, \\frz — all scale/deform
                    tags affect the whole block uniformly so there is no
                    "one word scaling while others stay" reflow.

    scope="word"  — a per-word override block wrapping each word, with
                    \\t() transitions computed from the word's cumulative
                    start offset inside the event. Each word animates at
                    its own activation time as karaoke sweep reaches it.
                    Word-scope implementations MUST use only metric-safe
                    tags (no \\fscx, \\fscy, \\fs, \\fsp) so individual
                    word animations don't push neighbours horizontally or
                    vertically — any glyph-dimension change during
                    per-word animation triggers libass line reflow and
                    the whole line visibly shakes.

The stagger style is special: it needs character-level time offsets, so
it is handled by expand_stagger_word() instead of the KineticImpl
transition table. stagger is word-scope only.

Scope selection is driven by RenderConfig.word_level:
    word_level=True  -> preset.word if available, else preset.line
    word_level=False -> preset.line if available, else raise ValueError

This keeps the public API surface small (no new kinetic_scope field) while
letting callers control scope via the existing word_level switch.

Line-level tags that must NOT appear in word-scope impls:
    \\fscx, \\fscy — change horizontal/vertical glyph advance; trigger
                     reflow across the whole line
    \\fs            — font size change, same problem as \\fscx/\\fscy
    \\fsp           — extra letter spacing, changes horizontal advance
    \\move, \\pos   — absolute positioning, breaks libass auto-layout

Safe word-scope tags:
    \\alpha, \\1a, \\3a, \\4a    — transparency channels
    \\c, \\1c, \\2c, \\3c, \\4c  — color channels
    \\bord, \\xbord, \\ybord     — outline width
    \\blur, \\be                  — gaussian blur / edge blur
    \\frz, \\frx, \\fry           — 3D rotation (bounding box unchanged)
    \\fax, \\fay                  — axis shear (small amounts only)
    \\xshad, \\yshad              — shadow offset
"""

from dataclasses import dataclass, field
from typing import FrozenSet, List, Literal, Optional, Tuple

Scope = Literal["line", "word"]


# =============================================================================
# Data model
# =============================================================================


@dataclass(frozen=True)
class KineticImpl:
    """A single-scope kinetic implementation.

    Attributes:
        initial: Static ASS override tags applied at the start of the
            override block. For word-scope, this means each word starts
            with these tags (e.g. \\alpha&HFF& for an invisible baseline).
            For line-scope, this is the text-position-0 static tag (e.g.
            \\fscy80 as the entrance pre-state, or \\fad(300,0) which is
            itself an event-level effect).
        transitions: List of (t1_ms, t2_ms, tags) tuples rendered as
            \\t(t1+offset, t2+offset, tags). For word-scope, offset is
            the word's cumulative ms from event start. For line-scope,
            offset is 0.
    """

    initial: str = ""
    transitions: Tuple[Tuple[int, int, str], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class KineticPreset:
    """A named kinetic preset with optional line and word implementations.

    At least one of `line` or `word` must be set. Presets that make sense
    only at a single scope (rise = line-only, stagger = word-only) leave
    the other side None.
    """

    line: Optional[KineticImpl] = None
    word: Optional[KineticImpl] = None


# =============================================================================
# Public style names — keep in sync with ASSConfig.kinetic_style Literal
# =============================================================================

KINETIC_STYLE_NAMES: Tuple[str, ...] = (
    # Smooth entrance
    "fade",
    "zoom",
    "rise",
    "blur_in",
    "pop",
    # Impact
    "bounce",
    "shake",
    "pulse",
    "swing",
    # Stylized
    "glow",
    "neon",
    "wave",
    "flicker",
    "typewriter",
    "stagger",
)


# =============================================================================
# Preset registry — 15 presets with line and/or word impls
# =============================================================================

_PRESETS: "dict[str, KineticPreset]" = {
    # ===== Smooth entrance ====================================================
    # fade — whole line fades in (line) or each word fades in at activation (word)
    "fade": KineticPreset(
        line=KineticImpl(initial=r"\fad(300,0)"),
        word=KineticImpl(
            initial=r"\alpha&HFF&",
            transitions=((0, 300, r"\alpha&H00&"),),
        ),
    ),
    # zoom — whole block vertically expands from 80%; per-word outline grows
    "zoom": KineticPreset(
        line=KineticImpl(
            initial=r"\fscy80",
            transitions=((0, 300, r"\fscy100"),),
        ),
        word=KineticImpl(
            initial=r"\bord0\blur2",
            transitions=((0, 300, r"\bord3\blur0"),),
        ),
    ),
    # rise — whole block rises from zero height; no word-scope (would equal fade)
    "rise": KineticPreset(
        line=KineticImpl(
            initial=r"\fscy0",
            transitions=((0, 400, r"\fscy100"),),
        ),
        word=None,
    ),
    # blur_in — gaussian blur clears from 6 to 0 (both scopes identical pattern)
    "blur_in": KineticPreset(
        line=KineticImpl(
            initial=r"\blur6",
            transitions=((0, 300, r"\blur0"),),
        ),
        word=KineticImpl(
            initial=r"\blur6",
            transitions=((0, 300, r"\blur0"),),
        ),
    ),
    # pop — line: \fscy squash + alpha fade; word: alpha+blur ramp with bord overshoot
    "pop": KineticPreset(
        line=KineticImpl(
            initial=r"\fscy60\alpha&HFF&",
            transitions=((0, 250, r"\fscy100\alpha&H00&"),),
        ),
        word=KineticImpl(
            initial=r"\alpha&HFF&\blur4\bord0",
            transitions=(
                (0, 1, r"\bord5\alpha&H00&\blur2"),
                (1, 250, r"\bord2\blur0"),
            ),
        ),
    ),
    # ===== Impact =============================================================
    # bounce — line: vertical 130% springback; word: bord+frz impact (chosen
    # from Codex's 3 alternatives: \frx is visually weaker, \bord+\blur too
    # close to glow, \bord+\frz gives distinct "impact plus recoil" identity)
    "bounce": KineticPreset(
        line=KineticImpl(
            initial=r"\fscy130",
            transitions=((0, 300, r"\fscy100"),),
        ),
        word=KineticImpl(
            transitions=(
                (0, 1, r"\bord6\frz3"),
                (1, 200, r"\bord2\frz0"),
            ),
        ),
    ),
    # shake — 3-stage rotation jitter (same pattern both scopes, shorter window for word)
    "shake": KineticPreset(
        line=KineticImpl(
            transitions=(
                (0, 80, r"\frz3"),
                (80, 160, r"\frz-3"),
                (160, 240, r"\frz0"),
            ),
        ),
        word=KineticImpl(
            transitions=(
                (0, 60, r"\frz3"),
                (60, 120, r"\frz-3"),
                (120, 180, r"\frz0"),
            ),
        ),
    ),
    # pulse — line: single fscy breathe; word: bord+blur breathe
    "pulse": KineticPreset(
        line=KineticImpl(
            transitions=(
                (0, 400, r"\fscy108"),
                (400, 800, r"\fscy100"),
            ),
        ),
        word=KineticImpl(
            transitions=(
                (0, 200, r"\bord4\blur2"),
                (200, 400, r"\bord2\blur0"),
            ),
        ),
    ),
    # swing — pendulum rotation, same shape both scopes
    "swing": KineticPreset(
        line=KineticImpl(
            initial=r"\frz-6",
            transitions=(
                (0, 200, r"\frz6"),
                (200, 400, r"\frz0"),
            ),
        ),
        word=KineticImpl(
            initial=r"\frz-8",
            transitions=(
                (0, 200, r"\frz8"),
                (200, 400, r"\frz0"),
            ),
        ),
    ),
    # ===== Stylized ===========================================================
    # glow — outline + blur pulse (metric-safe, same shape both scopes)
    "glow": KineticPreset(
        line=KineticImpl(
            transitions=(
                (0, 150, r"\bord5\blur4"),
                (150, 400, r"\bord2\blur1"),
            ),
        ),
        word=KineticImpl(
            transitions=(
                (0, 100, r"\bord5\blur4"),
                (100, 300, r"\bord2\blur1"),
            ),
        ),
    ),
    # neon — stronger glow pulse with pre-state
    "neon": KineticPreset(
        line=KineticImpl(
            initial=r"\bord2\blur1",
            transitions=(
                (0, 200, r"\bord7\blur6"),
                (200, 500, r"\bord3\blur2"),
            ),
        ),
        word=KineticImpl(
            transitions=(
                (0, 1, r"\bord2\blur1"),
                (1, 200, r"\bord7\blur6"),
                (200, 500, r"\bord3\blur2"),
            ),
        ),
    ),
    # wave — line: vertical ripple; word: per-word frz phase sway (metric-safe)
    "wave": KineticPreset(
        line=KineticImpl(
            transitions=(
                (0, 300, r"\fscy112"),
                (300, 600, r"\fscy92"),
                (600, 900, r"\fscy100"),
            ),
        ),
        word=KineticImpl(
            transitions=(
                (0, 200, r"\frz4"),
                (200, 400, r"\frz-4"),
                (400, 600, r"\frz0"),
            ),
        ),
    ),
    # flicker — two quick alpha flashes (both scopes, slightly tighter for word)
    "flicker": KineticPreset(
        line=KineticImpl(
            transitions=(
                (0, 60, r"\alpha&HC0&"),
                (60, 120, r"\alpha&H00&"),
                (180, 240, r"\alpha&HC0&"),
                (240, 300, r"\alpha&H00&"),
            ),
        ),
        word=KineticImpl(
            transitions=(
                (0, 50, r"\alpha&HC0&"),
                (50, 100, r"\alpha&H00&"),
                (150, 200, r"\alpha&HC0&"),
                (200, 250, r"\alpha&H00&"),
            ),
        ),
    ),
    # typewriter — line: 1ms hard-cut fade in; word: per-word instant reveal
    "typewriter": KineticPreset(
        line=KineticImpl(initial=r"\fad(1,0)"),
        word=KineticImpl(
            initial=r"\alpha&HFF&",
            transitions=((0, 1, r"\alpha&H00&"),),
        ),
    ),
    # stagger — word-scope only; char-level expansion handled by
    # expand_stagger_word(), not this transition table. The KineticImpl is
    # a sentinel so that resolve_kinetic() succeeds.
    "stagger": KineticPreset(line=None, word=KineticImpl()),
}


_CHAR_LEVEL_STYLES: FrozenSet[str] = frozenset({"stagger"})


# =============================================================================
# Public API
# =============================================================================


def list_kinetic_styles() -> List[str]:
    """Return all 15 supported kinetic style names, in canonical order."""
    return list(KINETIC_STYLE_NAMES)


def validate_kinetic_style(style: Optional[str]) -> None:
    """Raise ValueError if `style` is not None and not a known style."""
    if style is None:
        return
    if style not in KINETIC_STYLE_NAMES:
        available = ", ".join(KINETIC_STYLE_NAMES)
        raise ValueError(f"Unknown kinetic_style: {style!r}. Available: {available}")


def is_char_level_style(style: Optional[str]) -> bool:
    """True if the style requires character-level expansion (stagger)."""
    return style in _CHAR_LEVEL_STYLES


def get_kinetic_preset(style: str) -> KineticPreset:
    """Return the KineticPreset for a named style. Raises on unknown."""
    validate_kinetic_style(style)
    return _PRESETS[style]


def resolve_kinetic(
    style: Optional[str],
    word_level: bool,
) -> Optional[Tuple[Scope, KineticImpl]]:
    """Pick the scope and impl for a style based on the word_level preference.

    Returns None if style is None.

    Scope selection:
        word_level=True  -> preset.word if available, else fall back to
                            preset.line (preset is word-only-unavailable,
                            e.g. rise, so use line impl uniformly across
                            the dialogue event).
        word_level=False -> preset.line if available, else raise (preset
                            requires word_level=True, e.g. stagger).

    Raises:
        ValueError: unknown style, or preset has no suitable impl.
    """
    if style is None:
        return None
    validate_kinetic_style(style)
    preset = _PRESETS[style]

    if word_level:
        if preset.word is not None:
            return "word", preset.word
        if preset.line is not None:
            return "line", preset.line
    else:
        if preset.line is not None:
            return "line", preset.line
        if preset.word is not None:
            raise ValueError(
                f"kinetic_style={style!r} requires word_level=True "
                f"(preset has no line-scope implementation)"
            )

    raise ValueError(f"kinetic preset {style!r} has no implementation")


def build_line_override(impl: KineticImpl) -> str:
    """Render a line-scope impl as a single ASS override block body.

    The returned string goes INSIDE a {...} at event-text position 0,
    e.g. "\\fad(300,0)" or "\\fscy80\\t(0,300,\\fscy100)".
    """
    transitions = "".join(f"\\t({t1},{t2},{tags})" for t1, t2, tags in impl.transitions)
    return impl.initial + transitions


def build_word_overrides(impl: KineticImpl, word_start_ms: int) -> str:
    """Render a word-scope impl as ASS overrides offset to this word's time.

    The returned string concatenates the static initial with transitions
    whose t1/t2 are shifted by word_start_ms, producing per-word animation
    timed to the word's activation moment rather than event start.

    Format: "<initial>\\t(ws+t1,ws+t2,tags)\\t(ws+t3,ws+t4,tags)..."
    """
    transitions = "".join(
        f"\\t({word_start_ms + t1},{word_start_ms + t2},{tags})"
        for t1, t2, tags in impl.transitions
    )
    return impl.initial + transitions


# =============================================================================
# Stagger: character-level expansion
# =============================================================================

_STAGGER_CHAR_DELAY_MS = 40
_STAGGER_CHAR_WINDOW_MS = 120


def expand_stagger_word(
    word_text: str,
    word_start_ms: int,
    char_delay_ms: int = _STAGGER_CHAR_DELAY_MS,
    char_window_ms: int = _STAGGER_CHAR_WINDOW_MS,
) -> str:
    """Wrap each character with staggered alpha-fade tags.

    Produces output like:
        {\\alpha&HFF&\\t(450,570,\\alpha&H00&)}H
        {\\alpha&HFF&\\t(490,610,\\alpha&H00&)}e
        {\\alpha&HFF&\\t(530,650,\\alpha&H00&)}l
        ...

    Each char is invisible from event start (static `\\alpha&HFF&` inside
    its own override block), then at `word_start_ms + i*char_delay_ms` it
    fades in over `char_window_ms`. Uses only alpha (no \\fsc*) so
    characters don't push their neighbours — metric-safe. Handles CJK
    correctly because Python string iteration yields code points.

    Args:
        word_text: Raw word text (may contain CJK, Latin, mixed).
        word_start_ms: Cumulative ms from Dialogue event start to this word.
        char_delay_ms: Delay between successive characters.
        char_window_ms: Fade-in duration per character.

    Returns:
        Tag-wrapped text ready to be placed after the karaoke \\k{cs} tag.
    """
    if not word_text:
        return ""

    parts: List[str] = []
    for i, ch in enumerate(word_text):
        ch_start = word_start_ms + i * char_delay_ms
        ch_end = ch_start + char_window_ms
        parts.append(f"{{\\alpha&HFF&\\t({ch_start},{ch_end},\\alpha&H00&)}}{ch}")
    return "".join(parts)
