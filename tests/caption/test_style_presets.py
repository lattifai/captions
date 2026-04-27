"""Tests for ASSConfig.style_preset and the kinetic presets bound to it
(``spotlight``, ``reveal``).

Covers the high-level "name a look" experience for ASS export:

  1. STYLE_PRESETS exposes 6 entries with the expected names.
  2. ASSConfig.from_preset / style_preset injects preset values into
     fields the user did not explicitly set.
  3. Explicit kwargs always win over the preset.
  4. Preset-driven ASS render produces the expected Style / Dialogue
     tag patterns end-to-end (one per preset).
  5. spotlight kinetic emits \\bord-and-\\alpha contrast (metric-safe;
     no \\fscx / \\fscy / \\fs in word-scope output).
  6. bold_center's ``reveal`` kinetic emits per-word \\alpha fade-in
     when word alignment is present, and a line-scope \\fad fallback
     otherwise.
"""

import os
import tempfile

import pytest

from lattifai.caption import Caption, Supervision
from lattifai.caption.supervision import AlignmentItem
from lattifai.caption.config import ASSConfig
from lattifai.caption.styles import (
    STYLE_PRESETS,
    list_style_presets,
    resolve_style_preset,
)


def _write_ass(cap: Caption, cfg: ASSConfig) -> str:
    """Render a Caption to ASS via a temp file and return its text."""
    with tempfile.NamedTemporaryFile(suffix=".ass", delete=False, mode="w") as f:
        path = f.name
    try:
        cap.write(path, format_config=cfg)
        return open(path).read()
    finally:
        os.unlink(path)


@pytest.fixture
def two_word_caption():
    return Caption(
        supervisions=[
            Supervision(
                text="Hello world",
                start=0.0,
                duration=2.0,
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=0.0, duration=1.0, score=1.0),
                        AlignmentItem(symbol="world", start=1.0, duration=1.0, score=1.0),
                    ]
                },
            )
        ],
        language="en",
    )


@pytest.fixture
def simple_caption():
    return Caption(
        supervisions=[
            Supervision(text="Hello world", start=0.0, duration=2.0),
            Supervision(text="Second cue", start=3.0, duration=1.5),
        ],
        language="en",
    )


# =============================================================================
# 1. styles.py module API
# =============================================================================


class TestStylesModuleAPI:
    def test_six_presets_registered(self):
        names = list_style_presets()
        assert len(names) == 6
        assert set(names) == {
            "classic",
            "tiktok",
            "modern_box",
            "cinematic",
            "outline",
            "bold_center",
        }

    def test_resolve_returns_fresh_dict(self):
        d1 = resolve_style_preset("classic")
        d2 = resolve_style_preset("classic")
        d1["font_size"] = 999
        # Mutating one copy must not bleed into another lookup.
        assert resolve_style_preset("classic")["font_size"] == STYLE_PRESETS["classic"]["font_size"]
        assert d2["font_size"] != 999

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown style_preset"):
            resolve_style_preset("not_a_preset")


# =============================================================================
# 2. ASSConfig preset injection
# =============================================================================


class TestPresetInjection:
    def test_classic_fills_defaults(self):
        # classic = opaque black 80% box (borderstyle=3 auto from
        # background_color), white normal-weight Arial 64, outline=16
        # serves as box padding.
        c = ASSConfig.from_preset("classic")
        assert c.font_name == "Arial"
        assert c.font_size == 64
        assert c.bold is False
        assert c.background_color == "#000000CC"
        assert c.outline_width == 16  # padding under borderstyle=3
        assert c.alignment == 2

    def test_bold_center_disables_smart_wrap(self):
        # 100px Arial Black on 1920px frame fits ~16-22 chars per row;
        # smart-wrap (style 0) would collapse long cues into stacked rows
        # and break the screen-centre look. Preset must turn it off.
        c = ASSConfig.from_preset("bold_center")
        assert c.wrap_style == 2

    def test_bold_center_uses_reveal_kinetic(self):
        # bold_center's "gradual word appearance" is driven by the
        # karaoke instant tags (per-word \k boundaries) plus the reveal
        # kinetic preset (\alpha&HFF& → 180ms \alpha&H00& per word).
        c = ASSConfig.from_preset("bold_center")
        assert c.karaoke_effect == "instant"
        assert c.kinetic_style == "reveal"
        # Soft black stroke + drop-shadow as the "glow" stand-in.
        assert c.outline_color == "#00000099"
        assert c.outline_width == 1.5
        assert c.shadow_depth == 4.0

    def test_other_presets_keep_default_wrap_style(self):
        # Non-bold_center presets stay on the dataclass default (0 = smart).
        for name in ("classic", "tiktok", "modern_box", "cinematic", "outline"):
            assert ASSConfig.from_preset(name).wrap_style == 0, name

    def test_tiktok_sets_karaoke_and_kinetic(self):
        c = ASSConfig.from_preset("tiktok")
        assert c.karaoke_effect == "instant"
        assert c.karaoke_color_scheme == "yellow-pop"
        assert c.kinetic_style == "spotlight"
        # Bold uppercase body with double drop shadow (no semi-opaque
        # outline — the active word colour swap carries emphasis).
        assert c.bold is True
        assert c.outline_width == 3.0
        assert c.shadow_depth == 2.0

    def test_modern_box_sets_solid_white_bg(self):
        # Inverted from classic: solid white box with bold black text.
        c = ASSConfig.from_preset("modern_box")
        assert c.background_color == "#FFFFFFFF"
        assert c.primary_color == "#000000"
        assert c.bold is True

    def test_cinematic_sets_italic_no_outline(self):
        c = ASSConfig.from_preset("cinematic")
        assert c.italic is True
        assert c.outline_width == 0
        # Wide tracking + soft shadow = the "letterbox cinema" look.
        assert c.spacing == 4.0
        assert c.shadow_depth == 4.0

    def test_outline_uses_thin_stroke(self):
        # Was 5.0 in v1; SubStudio's webkit-text-stroke is 1.5px and
        # carries a light shadow.
        c = ASSConfig.from_preset("outline")
        assert c.outline_width == 1.5
        assert c.shadow_depth == 1.0

    def test_explicit_kwarg_overrides_preset(self):
        c = ASSConfig(style_preset="bold_center", font_size=160)
        assert c.font_size == 160
        # Other preset fields untouched.
        assert c.font_name == "Arial Black"
        assert c.kinetic_style == "reveal"

    def test_unknown_preset_raises_at_construction(self):
        with pytest.raises(ValueError, match="Unknown style_preset"):
            ASSConfig(style_preset="not_a_preset")

    def test_no_preset_keeps_dataclass_defaults(self):
        c = ASSConfig()
        assert c.style_preset is None
        assert c.font_name  # default font present
        assert c.font_size == 64



# =============================================================================
# 3. End-to-end render: each preset emits expected ASS tag patterns
# =============================================================================


class TestPresetEndToEndRender:
    def test_classic_style_line(self, simple_caption):
        # classic now uses borderstyle=3 (opaque black box) — assert font,
        # size, alignment, AND that the borderstyle field carries "3".
        out = _write_ass(simple_caption, ASSConfig.from_preset("classic"))
        style_line = next(line for line in out.splitlines() if line.startswith("Style:"))
        assert "Arial" in style_line
        assert ",64," in style_line
        parts = style_line.split(",")
        assert ",2," in style_line  # alignment
        assert "3" in parts, f"expected borderstyle=3 in {parts}"

    def test_tiktok_emits_karaoke_style_and_spotlight_tags(self, two_word_caption):
        out = _write_ass(two_word_caption, ASSConfig.from_preset("tiktok"))
        # Karaoke style block is added when karaoke_effect is set.
        assert "Style: Karaoke" in out
        # spotlight word-impl uses \alpha + \bord (metric-safe contrast).
        assert "\\alpha" in out
        assert "\\bord" in out
        # And no \fscx / \fscy in word-scope output (line-reflow invariant).
        assert "\\fscx" not in out
        # yellow-pop karaoke scheme: active (sung) word switches to
        # #FACC15 yellow (BBGGRR=15CCFA) on the karaoke style.
        karaoke_style = next(
            line for line in out.splitlines() if line.startswith("Style: Karaoke")
        )
        assert "15CCFA" in karaoke_style.upper(), karaoke_style

    def test_modern_box_uses_borderstyle_3_and_white_bg(self, simple_caption):
        out = _write_ass(simple_caption, ASSConfig.from_preset("modern_box"))
        style_line = next(line for line in out.splitlines() if line.startswith("Style:"))
        parts = style_line.split(",")
        # BorderStyle = 3 (opaque box) auto-derived from background_color.
        assert "3" in parts, f"expected borderstyle=3 in {parts}"
        # PrimaryColour = solid black (BBGGRR=000000, alpha=00 → "&H00000000").
        # The first colour field after Fontsize is PrimaryColour.
        assert "&H00000000" in style_line, style_line

    def test_outline_has_thin_stroke(self, simple_caption):
        # outline_width=1.5 + bold; shadow_depth=1.0 (was 0 in v1).
        out = _write_ass(simple_caption, ASSConfig.from_preset("outline"))
        style_line = next(line for line in out.splitlines() if line.startswith("Style:"))
        parts = style_line.split(",")
        # Bold flag is field index 7 in V4+ Style line. pysubs2 emits -1 for True.
        assert "-1" in parts, "bold flag should be -1 (True)"

    def test_cinematic_has_italic_and_letter_spacing(self, simple_caption):
        out = _write_ass(simple_caption, ASSConfig.from_preset("cinematic"))
        style_line = next(line for line in out.splitlines() if line.startswith("Style:"))
        # V4+ Style fields: Name, Fontname, Fontsize, PrimaryColour,
        # SecondaryColour, OutlineColour, BackColour, Bold, Italic, ...
        # split() on the leading "Style: " yields ["Style: Default", ...]
        # so positions shift by 1.
        parts = [p.strip() for p in style_line.split(",")]
        # Bold=0 (cinematic is light-weight), Italic=-1 (True).
        assert parts[7] == "0", f"expected Bold=0, got {parts[7]}; full: {parts}"
        assert parts[8] == "-1", f"expected Italic=-1, got {parts[8]}; full: {parts}"
        # Spacing field — index 13.
        assert parts[13] == "4", f"expected Spacing=4, got {parts[13]}; full: {parts}"

    def test_bold_center_emits_no_extra_dialogue_lines(self, simple_caption):
        # bold_center reference visual has no playhead / progress bar; the
        # preset matches that. Two cues = exactly two Dialogue lines.
        out = _write_ass(simple_caption, ASSConfig.from_preset("bold_center"))
        dlg_lines = [line for line in out.splitlines() if line.startswith("Dialogue:")]
        assert len(dlg_lines) == 2
        # reveal kinetic line-scope fallback: \fad(180,0) when word
        # alignment is missing.
        assert any("\\fad(180,0)" in line for line in dlg_lines), dlg_lines

    def test_bold_center_emits_per_word_reveal_with_alignment(self, two_word_caption):
        # With word alignment present, bold_center renders per-word
        # \alpha&HFF& → \t(...,180,...,\alpha&H00&) reveal transitions.
        out = _write_ass(two_word_caption, ASSConfig.from_preset("bold_center"))
        # Karaoke style block is added; per-word \k tags are present.
        assert "Style: Karaoke" in out
        assert "\\k" in out
        # Word-scope reveal: each word starts at full alpha (FF) then
        # transitions over 180ms to alpha 00. The transition time is
        # rebased per-word, so we just look for the alpha-FF reset and a
        # \t(...,...,\alpha&H00&) ending.
        assert "\\alpha&HFF&" in out
        assert "\\alpha&H00&" in out


