"""Tests for ASSConfig style fields: scalex, scaley, spacing, angle, underline, strikeout.

Also covers kinetic baseline rebase when scaley/angle deviate from defaults.
"""

import re

import pytest

from lattifai.caption.config import ASSConfig, RenderConfig
from lattifai.caption.formats.pysubs2 import ASSFormat
from lattifai.caption.supervision import AlignmentItem, Supervision


def _make_sups():
    return [Supervision(text="Hello world", start=0.0, duration=2.0)]


def _make_word_sups():
    return [
        Supervision(
            text="Hello world",
            start=0.0,
            duration=2.0,
            alignment={
                "word": [
                    AlignmentItem(symbol="Hello", start=0.0, duration=0.8),
                    AlignmentItem(symbol="world", start=1.0, duration=0.8),
                ]
            },
        )
    ]


def _get_default_style(config):
    """Write ASS and extract the Default style line fields."""
    result = ASSFormat.to_bytes(_make_sups(), config=config).decode("utf-8")
    lines = result.split("\n")
    default_style = [l for l in lines if l.startswith("Style: Default")][0]
    return default_style.split(",")


# ASS style field indices (after split by comma):
# 0=Name, 1=Fontname, 2=Fontsize, 3=PrimaryColour, 4=SecondaryColour,
# 5=OutlineColour, 6=BackColour, 7=Bold, 8=Italic, 9=Underline,
# 10=StrikeOut, 11=ScaleX, 12=ScaleY, 13=Spacing, 14=Angle,
# 15=BorderStyle, 16=Outline, 17=Shadow, 18=Alignment, ...


class TestASSConfigDefaults:
    """New fields should have pysubs2-aligned defaults."""

    def test_scalex_default(self):
        assert ASSConfig().scalex == 100.0

    def test_scaley_default(self):
        assert ASSConfig().scaley == 100.0

    def test_spacing_default(self):
        assert ASSConfig().spacing == 0.0

    def test_angle_default(self):
        assert ASSConfig().angle == 0.0

    def test_underline_default(self):
        assert ASSConfig().underline is False

    def test_strikeout_default(self):
        assert ASSConfig().strikeout is False


class TestASSStyleOutput:
    """Fields should appear in the correct ASS style line positions."""

    def test_scalex_in_output(self):
        fields = _get_default_style(ASSConfig(scalex=75.0))
        assert fields[11].strip() == "75"

    def test_scaley_in_output(self):
        fields = _get_default_style(ASSConfig(scaley=120.0))
        assert fields[12].strip() == "120"

    def test_spacing_in_output(self):
        fields = _get_default_style(ASSConfig(spacing=2.5))
        assert fields[13].strip() == "2.5"

    def test_angle_in_output(self):
        fields = _get_default_style(ASSConfig(angle=5.0))
        assert fields[14].strip() == "5"

    def test_underline_in_output(self):
        fields = _get_default_style(ASSConfig(underline=True))
        assert fields[9].strip() == "-1"  # ASS uses -1 for True

    def test_strikeout_in_output(self):
        fields = _get_default_style(ASSConfig(strikeout=True))
        assert fields[10].strip() == "-1"

    def test_defaults_match_pysubs2(self):
        """Default values should produce same output as pysubs2 defaults."""
        fields = _get_default_style(ASSConfig())
        assert fields[11].strip() == "100"  # scalex
        assert fields[12].strip() == "100"  # scaley
        assert fields[13].strip() == "0"    # spacing
        assert fields[14].strip() == "0"    # angle
        assert fields[9].strip() == "0"     # underline
        assert fields[10].strip() == "0"    # strikeout


class TestKineticBaselineRebase:
    """Kinetic presets should respect config scaley/angle as baseline."""

    def _write_ass_karaoke(self, config):
        sups = _make_word_sups()
        render = RenderConfig(include_speaker_in_text=False, word_level=True)
        return ASSFormat.to_bytes(sups, render=render, config=config).decode("utf-8")

    def _write_ass_line(self, config):
        sups = _make_sups()
        render = RenderConfig(include_speaker_in_text=False)
        return ASSFormat.to_bytes(sups, render=render, config=config).decode("utf-8")

    def test_zoom_line_baseline_scaley(self):
        """zoom line-scope: \\fscy80 -> \\fscy{0.8*scaley}, \\fscy100 -> \\fscy{scaley}."""
        config = ASSConfig(kinetic_style="zoom", scaley=120.0)
        result = self._write_ass_line(config)
        # End state should be \fscy120 (= config.scaley), not \fscy100
        assert "\\fscy120" in result
        # Start state should be \fscy96 (= 80/100 * 120), not \fscy80
        assert "\\fscy96" in result

    def test_zoom_line_default_scaley_unchanged(self):
        """With default scaley=100, preset values should be unchanged."""
        config = ASSConfig(kinetic_style="zoom")
        result = self._write_ass_line(config)
        assert "\\fscy80" in result
        assert "\\fscy100" in result

    def test_shake_word_baseline_angle(self):
        """shake word-scope: \\frz0 -> \\frz{angle}, \\frz3 -> \\frz{3+angle}."""
        config = ASSConfig(
            karaoke_effect="sweep", kinetic_style="shake", angle=5.0
        )
        result = self._write_ass_karaoke(config)
        # Return to baseline: \frz0 -> \frz5
        assert "\\frz5" in result
        # Offset: \frz3 -> \frz8
        assert "\\frz8" in result
        # Offset: \frz-3 -> \frz2
        assert "\\frz2" in result

    def test_shake_default_angle_unchanged(self):
        """With default angle=0, preset values should be unchanged."""
        config = ASSConfig(karaoke_effect="sweep", kinetic_style="shake")
        result = self._write_ass_karaoke(config)
        assert "\\frz0" in result
        assert "\\frz3" in result
        assert "\\frz-3" in result

    def test_rise_line_baseline_scaley(self):
        """rise line-scope: \\fscy0 start, \\fscy{scaley} end."""
        config = ASSConfig(kinetic_style="rise", scaley=80.0)
        result = self._write_ass_line(config)
        assert "\\fscy0" in result   # start from 0 always
        assert "\\fscy80" in result  # end at scaley
