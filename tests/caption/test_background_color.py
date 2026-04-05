"""Tests for subtitle background color (background_color field)."""

import pytest

from lattifai.caption.config import CaptionStyle, KaraokeConfig
from lattifai.caption.supervision import Supervision


def _make_sups():
    return [
        Supervision(text="Hello world", start=0.0, duration=2.0, speaker="Alice"),
        Supervision(text="Goodbye", start=2.5, duration=1.5, speaker="Bob"),
    ]


class TestCaptionStyleBackgroundColor:
    """CaptionStyle.background_color field behavior."""

    def test_default_is_empty(self):
        style = CaptionStyle()
        assert style.background_color == ""

    def test_solid_color(self):
        style = CaptionStyle(background_color="#000000")
        assert style.background_color == "#000000"

    def test_alpha_color(self):
        style = CaptionStyle(background_color="#00000080")
        assert style.background_color == "#00000080"

    def test_apply_color_scheme_with_background(self):
        """apply_color_scheme returns new style with background_color applied."""
        from lattifai.caption.colors import KARAOKE_COLOR_SCHEMES
        from lattifai.caption.config import apply_color_scheme

        original = KARAOKE_COLOR_SCHEMES["azure-gold"].copy()
        KARAOKE_COLOR_SCHEMES["azure-gold"]["background_color"] = "#1387C080"
        try:
            style = CaptionStyle()
            new_style = apply_color_scheme(style, "azure-gold")
            assert new_style.background_color == "#1387C080"
            assert style.background_color == ""  # original unchanged
        finally:
            KARAOKE_COLOR_SCHEMES["azure-gold"] = original


class TestASSBackgroundColor:
    """ASS writer background_color handling."""

    def _write_ass(self, sups, karaoke=None, **kwargs):
        from lattifai.caption.formats.pysubs2 import ASSFormat

        return ASSFormat.to_bytes(
            sups,
            include_speaker=False,
            karaoke=karaoke,
            **kwargs,
        ).decode("utf-8")

    def test_no_background_borderstyle_1(self):
        """Default: no background_color → borderstyle=1."""
        result = self._write_ass(_make_sups())
        # Default style should have BorderStyle: 1 (or not have BorderStyle: 3)
        assert "BorderStyle: 3" not in result or "BorderStyle: 1" in result

    def test_solid_background_borderstyle_3(self):
        """style.background_color="#000000" → borderstyle=3 in Default style."""
        style = CaptionStyle(background_color="#000000")
        result = self._write_ass(_make_sups(), style=style)
        lines = result.split("\n")
        default_style = [l for l in lines if l.startswith("Style: Default")]
        assert len(default_style) == 1
        fields = default_style[0].split(",")
        assert fields[15].strip() == "3"  # borderstyle field index 15

    def test_alpha_background_inverted(self):
        """#RRGGBBAA alpha is inverted for ASS (FF=opaque → 00 in ASS)."""
        from lattifai.caption.formats.pysubs2 import ASSFormat

        # Fully opaque in standard hex
        color = ASSFormat._hex_to_ass_color("#FF000000")  # Red, fully transparent
        assert color.a == 255  # ASS: FF = fully transparent

        # Fully transparent → ASS fully opaque
        color2 = ASSFormat._hex_to_ass_color("#FF0000FF")  # Red, fully opaque
        assert color2.a == 0  # ASS: 00 = fully opaque

        # 50% opacity
        color3 = ASSFormat._hex_to_ass_color("#00000080")  # 50% opaque
        assert color3.a == 127  # ASS: 255 - 128 = 127

    def test_hex6_no_alpha(self):
        """#RRGGBB (no alpha) → fully opaque in ASS (a=0)."""
        from lattifai.caption.formats.pysubs2 import ASSFormat

        color = ASSFormat._hex_to_ass_color("#FF0000")
        assert color.r == 255
        assert color.g == 0
        assert color.b == 0
        assert color.a == 0  # Fully opaque

    def test_karaoke_with_background(self):
        """Karaoke style should use borderstyle=3 when background_color is set via style param."""
        config = KaraokeConfig(enabled=True)
        style = CaptionStyle(background_color="#00000080")
        sups = _make_sups()
        from lattifai.caption.supervision import AlignmentItem

        sups[0].alignment = {
            "word": [
                AlignmentItem(symbol="Hello", start=0.0, duration=0.5),
                AlignmentItem(symbol="world", start=0.6, duration=0.4),
            ]
        }
        result = self._write_ass(sups, karaoke=config, word_level=True, style=style)
        lines = result.split("\n")
        karaoke_style_line = [l for l in lines if l.startswith("Style: Karaoke")]
        assert len(karaoke_style_line) == 1
        # borderstyle is field index 15 in ASS style format (after split by comma)
        fields = karaoke_style_line[0].split(",")
        assert len(fields) > 16
        borderstyle = fields[15].strip()
        assert borderstyle == "3"

    def test_shadow_disabled_in_box_mode(self):
        """When background_color is set, shadow should be 0."""
        config = KaraokeConfig(enabled=True)
        style = CaptionStyle(background_color="#00000080", shadow_depth=2.0)
        sups = _make_sups()
        from lattifai.caption.supervision import AlignmentItem

        sups[0].alignment = {
            "word": [
                AlignmentItem(symbol="Hello", start=0.0, duration=0.5),
                AlignmentItem(symbol="world", start=0.6, duration=0.4),
            ]
        }
        result = self._write_ass(sups, karaoke=config, word_level=True, style=style)
        karaoke_style_line = [l for l in result.split("\n") if l.startswith("Style: Karaoke")]
        assert len(karaoke_style_line) == 1
        fields = karaoke_style_line[0].split(",")
        # shadow is field index 17 in ASS style format
        shadow = fields[17].strip()
        assert shadow == "0" or shadow == "0.0"


class TestNonKaraokeBackgroundColor:
    """Background color in standard (non-karaoke) ASS mode."""

    def _write_ass(self, sups, style=None):
        from lattifai.caption.formats.pysubs2 import ASSFormat

        return ASSFormat.to_bytes(sups, include_speaker=True, style=style).decode("utf-8")

    def test_default_style_gets_borderstyle_3(self):
        """Non-karaoke ASS with background_color should set Default style borderstyle=3."""
        style = CaptionStyle(background_color="#000000")
        result = self._write_ass(_make_sups(), style=style)
        lines = result.split("\n")
        default_style = [l for l in lines if l.startswith("Style: Default")]
        assert len(default_style) == 1
        fields = default_style[0].split(",")
        borderstyle = fields[15].strip()
        assert borderstyle == "3"

    def test_no_background_keeps_borderstyle_1(self):
        """Non-karaoke ASS without background_color keeps borderstyle=1."""
        result = self._write_ass(_make_sups())
        lines = result.split("\n")
        default_style = [l for l in lines if l.startswith("Style: Default")]
        assert len(default_style) == 1
        fields = default_style[0].split(",")
        borderstyle = fields[15].strip()
        assert borderstyle == "1"
