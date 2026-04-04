"""Tests for ASS karaoke tag generation."""

import pytest

from lattifai.caption.config import CaptionStyle, KaraokeConfig
from lattifai.caption.formats.pysubs2 import ASSFormat
from lattifai.caption.supervision import AlignmentItem, Supervision


class TestASSKaraoke:
    """Test ASS karaoke tag generation."""

    def test_karaoke_sweep_effect(self):
        """Sweep effect should use \\kf tag."""
        sups = [
            Supervision(
                text="Hello world",
                start=15.2,
                duration=3.3,
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=15.2, duration=0.45),
                        AlignmentItem(symbol="world", start=15.65, duration=2.85),
                    ]
                },
            )
        ]
        karaoke_config = KaraokeConfig(enabled=True)
        result = ASSFormat.to_bytes(sups, word_level=True, karaoke_config=karaoke_config)
        content = result.decode("utf-8")

        # \kf45 means 45 centiseconds (0.45s)
        assert "{\\kf45}Hello" in content
        assert "{\\kf285}world" in content

    def test_karaoke_instant_effect(self):
        """Instant effect should use \\k tag."""
        sups = [
            Supervision(
                text="Hello world",
                start=15.2,
                duration=3.3,
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=15.2, duration=0.45),
                        AlignmentItem(symbol="world", start=15.65, duration=2.85),
                    ]
                },
            )
        ]
        config = KaraokeConfig(enabled=True, effect="instant")
        result = ASSFormat.to_bytes(sups, word_level=True, karaoke_config=config)
        content = result.decode("utf-8")

        assert "{\\k45}Hello" in content
        assert "{\\k285}world" in content

    def test_karaoke_outline_effect(self):
        """Outline effect should use \\ko tag."""
        sups = [
            Supervision(
                text="Hello",
                start=0.0,
                duration=1.0,
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=0.0, duration=0.5),
                    ]
                },
            )
        ]
        config = KaraokeConfig(enabled=True, effect="outline")
        result = ASSFormat.to_bytes(sups, word_level=True, karaoke_config=config)
        content = result.decode("utf-8")

        assert "{\\ko50}Hello" in content

    def test_karaoke_style_in_output(self):
        """Karaoke style should be defined in ASS output."""
        sups = [
            Supervision(
                text="Hello",
                start=0.0,
                duration=1.0,
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=0.0, duration=0.5),
                    ]
                },
            )
        ]
        karaoke_config = KaraokeConfig(enabled=True)
        result = ASSFormat.to_bytes(sups, word_level=True, karaoke_config=karaoke_config)
        content = result.decode("utf-8")

        # Should have Karaoke style defined
        assert "Style: Karaoke" in content or "Karaoke," in content

    def test_fallback_without_alignment(self):
        """Without alignment, should output normal ASS."""
        sups = [Supervision(text="No alignment", start=10.0, duration=2.0)]
        karaoke_config = KaraokeConfig(enabled=True)
        result = ASSFormat.to_bytes(sups, word_level=True, karaoke_config=karaoke_config)
        content = result.decode("utf-8")

        assert "No alignment" in content
        assert "{\\k" not in content  # No karaoke tags

    def test_metadata_karaoke_style_takes_precedence(self):
        """When metadata provides ass_styles.Karaoke, it should NOT be overwritten by default.

        KaraokeConfig defaults: font_name=Arial, font_size=128, primary_color=#FFFFFF.
        Metadata uses deliberately different values so we can verify precedence.
        """
        sups = [
            Supervision(
                text="Hello world",
                start=0.0,
                duration=2.0,
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=0.0, duration=0.5),
                        AlignmentItem(symbol="world", start=0.5, duration=1.5),
                    ]
                },
            )
        ]
        # Metadata values differ from CaptionStyle defaults on purpose:
        #   font: "Comic Sans MS" vs default "Arial"
        #   size: 36 vs default 128
        #   primary: &H0000FFFF (yellow) vs default &H00FFFFFF (white)
        #   outline: &H00FF0000 (blue) vs default &H00000000 (black)
        metadata = {
            "ass_styles": {
                "Default": {
                    "fontname": "Comic Sans MS",
                    "fontsize": 36,
                    "primarycolor": "&H0000FFFF",
                    "outlinecolor": "&H00FF0000",
                    "alignment": 2,
                },
                "Karaoke": {
                    "fontname": "Comic Sans MS",
                    "fontsize": 36,
                    "primarycolor": "&H0000FFFF",
                    "outlinecolor": "&H00FF0000",
                    "alignment": 2,
                },
            }
        }
        karaoke_config = KaraokeConfig(enabled=True)
        result = ASSFormat.to_bytes(
            sups,
            word_level=True,
            karaoke_config=karaoke_config,
            metadata=metadata,
        )
        content = result.decode("utf-8")

        # Parse the Karaoke style line
        # ASS Format: Style: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,...
        karaoke_line = [l for l in content.splitlines() if l.startswith("Style: Karaoke,")]
        assert len(karaoke_line) == 1, f"Expected one Karaoke style, got: {karaoke_line}"
        parts = karaoke_line[0].split(",")

        # Font name (index 1)
        assert parts[1] == "Comic Sans MS", f"Expected 'Comic Sans MS', got '{parts[1]}'"
        # Font size (index 2) — must be 36, not default 128
        assert float(parts[2]) == 36.0, f"Expected fontsize 36, got {parts[2]}"
        # PrimaryColour (index 3) — &H0000FFFF, not default &H00FFFFFF
        assert parts[3] == "&H0000FFFF", f"Expected &H0000FFFF, got {parts[3]}"
        # OutlineColour (index 5) — &H00FF0000, not default &H00000000
        assert parts[5] == "&H00FF0000", f"Expected &H00FF0000, got {parts[5]}"

    def test_karaoke_style_inherits_from_default_when_no_karaoke_in_metadata(self):
        """When metadata has Default but no Karaoke, Karaoke should copy from Default."""
        sups = [
            Supervision(
                text="Hello",
                start=0.0,
                duration=1.0,
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=0.0, duration=0.5),
                    ]
                },
            )
        ]
        metadata = {
            "ass_styles": {
                "Default": {
                    "fontname": "Georgia",
                    "fontsize": 42,
                    "primarycolor": "&H00FF00FF",
                    "outlinecolor": "&H0000FF00",
                    "alignment": 5,
                },
            }
        }
        karaoke_config = KaraokeConfig(enabled=True)
        result = ASSFormat.to_bytes(
            sups,
            word_level=True,
            karaoke_config=karaoke_config,
            metadata=metadata,
        )
        content = result.decode("utf-8")

        karaoke_line = [l for l in content.splitlines() if l.startswith("Style: Karaoke,")]
        assert len(karaoke_line) == 1
        parts = karaoke_line[0].split(",")

        # Should inherit Default's values, not CaptionStyle defaults (Arial/128)
        assert parts[1] == "Georgia", f"Expected 'Georgia', got '{parts[1]}'"
        assert float(parts[2]) == 42.0, f"Expected fontsize 42, got {parts[2]}"
        assert parts[3] == "&H00FF00FF", f"Expected &H00FF00FF, got {parts[3]}"
        assert parts[5] == "&H0000FF00", f"Expected &H0000FF00, got {parts[5]}"

    def test_default_karaoke_style_when_no_metadata(self):
        """Without metadata Karaoke style, should create from karaoke_config defaults."""
        sups = [
            Supervision(
                text="Hello",
                start=0.0,
                duration=1.0,
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=0.0, duration=0.5),
                    ]
                },
            )
        ]
        # Custom style with font_size=64 passed via style param
        custom_style = CaptionStyle(font_size=64, font_name="Courier")
        karaoke_config = KaraokeConfig(enabled=True)
        result = ASSFormat.to_bytes(sups, word_level=True, karaoke_config=karaoke_config, style=custom_style)
        content = result.decode("utf-8")

        karaoke_line = [l for l in content.splitlines() if l.startswith("Style: Karaoke,")]
        assert len(karaoke_line) == 1
        parts = karaoke_line[0].split(",")
        assert parts[1] == "Courier", f"Expected Courier, got {parts[1]}"
        assert parts[2] == "64" or parts[2] == "64.0", f"Expected fontsize 64, got {parts[2]}"

    def test_word_level_false_uses_original(self):
        """word_level=False should use original behavior."""
        sups = [
            Supervision(
                text="Hello world",
                start=15.2,
                duration=3.3,
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=15.2, duration=0.45),
                        AlignmentItem(symbol="world", start=15.65, duration=2.85),
                    ]
                },
            )
        ]
        result = ASSFormat.to_bytes(sups, word_level=False)
        content = result.decode("utf-8")

        # Should NOT have karaoke tags
        assert "{\\k" not in content
