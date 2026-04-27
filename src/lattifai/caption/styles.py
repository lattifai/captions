"""High-level ASS style presets — six common social-video looks.

Each preset is a one-shot bundle of ASSConfig field values. Mirrors the
"pick a style" experience common to short-video editors: the user names a
look (``"tiktok"``, ``"cinematic"``, ...) and the writer fills in the
matching font, color, outline, alignment, and (where relevant) karaoke /
kinetic / progress-bar values.

Composition rules (enforced by ``ASSConfig.__post_init__``):

    1. A preset only fills fields the user has NOT explicitly set.
    2. An explicit kwarg always wins over a preset value.
    3. Presets stay vendor-neutral — they describe a visual look, not a
       specific product.

Usage:

    from lattifai.caption.config import ASSConfig
    cfg = ASSConfig.from_preset("bold_center")
    # or override:
    cfg = ASSConfig(style_preset="bold_center", font_size=160)

Preset catalogue:

    classic      Opaque black 80% box, white normal-weight text, bottom
                 centre. Reads as a clean "broadcast caption".
    tiktok       Bold uppercase white text + double drop shadow with the
                 active word swapped to a saturated yellow via karaoke +
                 spotlight emphasis on the active word.
    modern_box   Solid white box with bold black text, bottom centre.
                 Inverse of the classic dark box look.
    cinematic    Italic light-weight white text, wide tracking, soft
                 shadow only — sits inside the letterbox bar.
    outline      White bold + thin (1.5px) black stroke + light shadow.
    bold_center  Massive screen-centre bold text with a thin black
                 outline, white glow shadow, and per-word "gradual
                 reveal" fade-in driven by karaoke instant + the
                 ``reveal`` kinetic preset.
"""

from typing import Any, Dict, List

# =============================================================================
# Style preset registry
# =============================================================================
# Reference target: 1920x1080 frame. play_res_x/y are not stored here so the
# preset composes cleanly with user-chosen render resolutions.
#
# Each preset stores ONLY the fields it cares about. Anything not listed
# falls back to the ASSConfig dataclass default.
# -----------------------------------------------------------------------------

STYLE_PRESETS: Dict[str, Dict[str, Any]] = {
    # 1) classic — opaque black 80% box, white normal-weight text. Mirrors
    # the "broadcast caption" look: a solid translucent rectangle hosts the
    # text at bottom-centre. ``background_color`` triggers borderstyle=3
    # (opaque box) automatically; ``outline_width=16`` becomes box padding.
    "classic": {
        "font_name": "Arial",
        "font_size": 64,
        "bold": False,
        "primary_color": "#FFFFFF",
        "background_color": "#000000CC",  # 80% opaque black
        "outline_width": 16,  # padding in borderstyle=3
        "shadow_depth": 0,
        "alignment": 2,  # bottom centre
        "margin_v": 80,
    },
    # 2) tiktok — bold uppercase white body with double drop shadow; the
    # active (sung) word swaps to a saturated yellow via the ``yellow-pop``
    # karaoke scheme. The ``spotlight`` kinetic adds a metric-safe
    # \alpha+\bord contrast pulse so the active word visibly "pops"
    # against the static white body.
    #
    # Note: ASS karaoke does not natively render a per-word background
    # pill (the pill in editor previews is a CSS-only effect). The yellow
    # text + spotlight contrast is the closest legibility-preserving
    # approximation that works in any libass-driven player.
    "tiktok": {
        "font_name": "Arial Black",
        "font_size": 80,
        "bold": True,
        "outline_color": "#000000",
        "outline_width": 3.0,
        "shadow_depth": 2.0,  # double drop-shadow approximation
        "borderstyle": 1,
        "alignment": 5,  # screen centre
        "margin_v": 0,
        "spacing": 1.5,  # ~0.04em letter-spacing
        "karaoke_effect": "instant",
        "karaoke_color_scheme": "yellow-pop",
        "kinetic_style": "spotlight",
    },
    # 3) modern_box — solid white opaque box with bold black text. Inverse
    # of the classic dark-box look. Uses borderstyle=3 (auto-derived from
    # background_color) and outline_width=16 as box padding.
    "modern_box": {
        "font_name": "Helvetica",
        "font_size": 58,
        "bold": True,  # font-semibold ≈ ASS bold
        "primary_color": "#000000",
        "background_color": "#FFFFFFFF",  # solid white
        "outline_width": 16,  # padding in borderstyle=3
        "shadow_depth": 3.0,  # shadow-lg
        "alignment": 2,
        "margin_v": 80,
    },
    # 4) cinematic — italic light-weight white, wide letter-spacing, soft
    # shadow only (no outline), deep vertical margin so the text sits in
    # the letterbox area. ``primary_color`` carries 90% alpha to soften
    # the body; the shadow handles legibility on bright footage.
    "cinematic": {
        "font_name": "Helvetica Neue",
        "font_size": 48,
        "bold": False,
        "italic": True,
        "primary_color": "#FFFFFFE6",  # ~90% white
        "outline_color": "#000000",
        "outline_width": 0,  # no outline (soft cinematic look)
        "shadow_depth": 4.0,
        "spacing": 4.0,  # ~0.1em wide tracking
        "borderstyle": 1,
        "alignment": 2,
        "margin_v": 120,
    },
    # 5) outline — white bold + thin black stroke (1.5px) and a light
    # shadow. Reads as the "Tailwind Webkit-text-stroke" feel without
    # over-baking the outline thickness.
    "outline": {
        "font_name": "Arial Black",
        "font_size": 64,
        "bold": True,
        "primary_color": "#FFFFFF",
        "outline_color": "#000000",
        "outline_width": 1.5,
        "shadow_depth": 1.0,
        "borderstyle": 1,
        "alignment": 2,
        "margin_v": 60,
    },
    # 6) bold_center — massive screen-centre bold text with a thin black
    # half-opaque stroke, soft drop shadow (acts as the "white glow"),
    # and per-word "gradual reveal" via karaoke instant + the reveal
    # kinetic preset.
    #
    # ``wrap_style=2`` disables libass smart-wrap: at 100px Arial Black
    # on a 1920-wide frame each line fits ~16-22 chars, so default
    # smart-wrap collapses any long cue into stacked rows and destroys
    # the single-row screen-centre look. Callers SHOULD pair this preset
    # with an aggressive ``StandardizationConfig`` (e.g.
    # ``max_chars_per_line=16, max_lines=1``) so cues fit on one row.
    #
    # No progress bar / playhead indicator — the reference visual ("Bold
    # Center" in short-video editors) is a single emphasis line.
    "bold_center": {
        "font_name": "Arial Black",
        "font_size": 100,
        "bold": True,
        "primary_color": "#FFFFFF",
        "outline_color": "#00000099",  # 60% opaque black stroke
        "outline_width": 1.5,
        "shadow_depth": 4.0,  # stand-in for the white-glow textShadow
        "borderstyle": 1,
        "alignment": 5,  # screen centre
        "margin_v": 0,
        "wrap_style": 2,  # no automatic line wrap
        "karaoke_effect": "instant",  # drives word-by-word reveal timing
        "kinetic_style": "reveal",  # 180ms alpha fade-in per word
    },
}


# =============================================================================
# Public API
# =============================================================================


def list_style_presets() -> List[str]:
    """Return all registered preset names in catalogue order."""
    return list(STYLE_PRESETS.keys())


def resolve_style_preset(name: str) -> Dict[str, Any]:
    """Return a fresh copy of the preset's field map.

    Raises ValueError if ``name`` is not a registered preset.
    """
    if name not in STYLE_PRESETS:
        valid = ", ".join(STYLE_PRESETS.keys())
        raise ValueError(f"Unknown style_preset {name!r}. Valid: {valid}")
    return dict(STYLE_PRESETS[name])
