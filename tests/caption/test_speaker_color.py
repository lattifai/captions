"""Tests for speaker_color feature in ASS output.

Covers: ASSFormat._resolve_speaker_color(), _SPEAKER_PALETTE, and speaker_color
parameter flowing through ASSFormat.to_bytes() in both karaoke and non-karaoke modes.
"""

import pytest

from lattifai.caption.config import CaptionStyle, KaraokeConfig
from lattifai.caption.formats.pysubs2 import ASSFormat
from lattifai.caption.supervision import AlignmentItem, Supervision


def _make_sups_with_speakers(*speakers: str) -> list:
    """Create supervisions with given speaker names, each with word alignment."""
    sups = []
    for i, spk in enumerate(speakers):
        start = float(i * 3)
        sups.append(
            Supervision(
                text=f"Line by {spk}",
                start=start,
                duration=2.0,
                speaker=spk,
                alignment={
                    "word": [
                        AlignmentItem(symbol="Line", start=start, duration=0.4),
                        AlignmentItem(symbol="by", start=start + 0.4, duration=0.3),
                        AlignmentItem(symbol=spk, start=start + 0.7, duration=1.3),
                    ]
                },
            )
        )
    return sups


def _make_sups_no_alignment(*speakers: str) -> list:
    """Create supervisions without word alignment (non-karaoke mode)."""
    sups = []
    for i, spk in enumerate(speakers):
        start = float(i * 3)
        sups.append(
            Supervision(
                text=f"Line by {spk}",
                start=start,
                duration=2.0,
                speaker=spk,
            )
        )
    return sups


class TestResolveSpeakerColor:
    """Unit tests for ASSFormat._resolve_speaker_color()."""

    def test_empty_spec_returns_empty(self):
        """Empty speaker_color spec should return no color."""
        cache = {}
        result = ASSFormat._resolve_speaker_color("Alice", "", cache)
        assert result == ""
        assert cache == {}

    def test_auto_assigns_palette_colors(self):
        """'auto' should assign colors from the built-in palette in order."""
        from lattifai.caption.colors import hex_rgb_to_bgr

        cache = {}
        color_a = ASSFormat._resolve_speaker_color("Alice", "auto", cache)
        color_b = ASSFormat._resolve_speaker_color("Bob", "auto", cache)
        assert color_a != ""
        assert color_b != ""
        assert color_a != color_b
        # First speaker gets palette[0] converted to BBGGRR, second gets palette[1]
        assert color_a == hex_rgb_to_bgr(ASSFormat._SPEAKER_PALETTE[0])
        assert color_b == hex_rgb_to_bgr(ASSFormat._SPEAKER_PALETTE[1])

    def test_auto_caches_same_speaker(self):
        """Same speaker should always get the same cached color."""
        cache = {}
        c1 = ASSFormat._resolve_speaker_color("Alice", "auto", cache)
        c2 = ASSFormat._resolve_speaker_color("Alice", "auto", cache)
        assert c1 == c2

    def test_auto_cycles_when_more_speakers_than_palette(self):
        """With more speakers than palette entries, colors should cycle."""
        cache = {}
        palette_size = len(ASSFormat._SPEAKER_PALETTE)
        speakers = [f"Speaker_{i}" for i in range(palette_size + 2)]
        colors = []
        for spk in speakers:
            c = ASSFormat._resolve_speaker_color(spk, "auto", cache)
            colors.append(c)
        # Speaker at index palette_size should cycle back to palette[0]
        assert colors[palette_size] == colors[0]
        assert colors[palette_size + 1] == colors[1]

    def test_single_hex_color(self):
        """Single #RRGGBB should convert to BBGGRR and apply to all speakers."""
        cache = {}
        c1 = ASSFormat._resolve_speaker_color("Alice", "#FF0000", cache)
        c2 = ASSFormat._resolve_speaker_color("Bob", "#FF0000", cache)
        # #FF0000 → BBGGRR = 0000FF
        assert c1 == "0000FF"
        assert c2 == "0000FF"

    def test_comma_separated_colors(self):
        """Comma-separated #RRGGBB values should assign in order."""
        cache = {}
        spec = "#FF0000,#00FF00,#0000FF"
        c1 = ASSFormat._resolve_speaker_color("Alice", spec, cache)
        c2 = ASSFormat._resolve_speaker_color("Bob", spec, cache)
        c3 = ASSFormat._resolve_speaker_color("Carol", spec, cache)
        # FF0000 → 0000FF, 00FF00 → 00FF00, 0000FF → FF0000
        assert c1 == "0000FF"
        assert c2 == "00FF00"
        assert c3 == "FF0000"

    def test_comma_separated_cycles(self):
        """Comma-separated palette should cycle for extra speakers."""
        cache = {}
        spec = "#FF0000,#00FF00"
        colors = []
        for spk in ["A", "B", "C", "D"]:
            colors.append(ASSFormat._resolve_speaker_color(spk, spec, cache))
        assert colors[0] == colors[2]  # A and C get same color
        assert colors[1] == colors[3]  # B and D get same color

    def test_invalid_color_returns_empty(self):
        """An invalid color string should return empty (no valid palette entries)."""
        cache = {}
        result = ASSFormat._resolve_speaker_color("Alice", "not-a-color", cache)
        assert result == ""

    def test_palette_size(self):
        """Built-in palette should have exactly 10 colors."""
        assert len(ASSFormat._SPEAKER_PALETTE) == 10


class TestSpeakerColorKaraokeMode:
    """Test speaker_color in karaoke ASS output (word_level=True, karaoke enabled)."""

    def test_auto_produces_different_colors(self):
        """speaker_color='auto' should produce different \\c tags for different speakers."""
        sups = _make_sups_with_speakers("Alice", "Bob")
        config = KaraokeConfig(enabled=True)
        result = ASSFormat.to_bytes(sups, word_level=True, karaoke=config, style=CaptionStyle(speaker_color="auto"))
        content = result.decode("utf-8")

        # Both speakers should have color override tags
        assert "{\\c&H" in content
        # Extract the color tags: {\\c&HBBGGRR&}
        import re

        color_tags = re.findall(r"\{\\c&H([0-9A-Fa-f]{6})&\}", content)
        assert len(color_tags) >= 2
        # The two speakers should have different colors
        unique_colors = set(color_tags)
        assert len(unique_colors) >= 2, f"Expected at least 2 unique colors, got {unique_colors}"

    def test_single_color_applied_to_all(self):
        """speaker_color='#FF0000' should apply the same color to all speakers."""
        sups = _make_sups_with_speakers("Alice", "Bob")
        config = KaraokeConfig(enabled=True)
        result = ASSFormat.to_bytes(sups, word_level=True, karaoke=config, style=CaptionStyle(speaker_color="#FF0000"))
        content = result.decode("utf-8")

        import re

        color_tags = re.findall(r"\{\\c&H([0-9A-Fa-f]{6})&\}", content)
        assert len(color_tags) >= 2
        # All should be the same color (0000FF = BBGGRR for #FF0000)
        assert all(c == "0000FF" for c in color_tags)

    def test_comma_separated_assignment(self):
        """speaker_color='#FF0000,#00FF00' should assign different colors per speaker."""
        sups = _make_sups_with_speakers("Alice", "Bob")
        config = KaraokeConfig(enabled=True)
        result = ASSFormat.to_bytes(
            sups, word_level=True, karaoke=config, style=CaptionStyle(speaker_color="#FF0000,#00FF00")
        )
        content = result.decode("utf-8")

        import re

        color_tags = re.findall(r"\{\\c&H([0-9A-Fa-f]{6})&\}", content)
        assert len(color_tags) >= 2
        unique_colors = set(color_tags)
        assert len(unique_colors) == 2

    def test_empty_speaker_color_no_color_tags(self):
        """speaker_color='' should produce no \\c color override tags."""
        sups = _make_sups_with_speakers("Alice", "Bob")
        config = KaraokeConfig(enabled=True)
        result = ASSFormat.to_bytes(sups, word_level=True, karaoke=config, style=CaptionStyle(speaker_color=""))
        content = result.decode("utf-8")

        # Should have karaoke tags but no speaker color tags
        assert "{\\kf" in content
        assert "{\\c&H" not in content

    def test_speaker_prefix_present_with_color(self):
        """Speaker label prefix should appear in the karaoke text alongside color tags."""
        sups = _make_sups_with_speakers("Alice")
        config = KaraokeConfig(enabled=True)
        result = ASSFormat.to_bytes(
            sups, word_level=True, karaoke=config, style=CaptionStyle(speaker_color="auto"), include_speaker=True
        )
        content = result.decode("utf-8")

        # Speaker prefix should appear
        assert "Alice" in content


class TestSpeakerColorNonKaraokeMode:
    """Test speaker_color in non-karaoke ASS output (standard mode)."""

    def test_auto_in_standard_mode(self):
        """speaker_color='auto' should work in standard (non-karaoke) mode."""
        sups = _make_sups_no_alignment("Alice", "Bob")
        result = ASSFormat.to_bytes(sups, word_level=False, style=CaptionStyle(speaker_color="auto"))
        content = result.decode("utf-8")

        assert "{\\c&H" in content
        # Both speakers should get colored prefixes
        import re

        color_tags = re.findall(r"\{\\c&H([0-9A-Fa-f]{6})&\}", content)
        assert len(color_tags) >= 2
        unique_colors = set(color_tags)
        assert len(unique_colors) >= 2

    def test_single_color_in_standard_mode(self):
        """Single #RRGGBB in standard mode should color all speaker prefixes the same."""
        sups = _make_sups_no_alignment("Alice", "Bob")
        result = ASSFormat.to_bytes(sups, word_level=False, style=CaptionStyle(speaker_color="#0000FF"))
        content = result.decode("utf-8")

        import re

        color_tags = re.findall(r"\{\\c&H([0-9A-Fa-f]{6})&\}", content)
        # #0000FF → BBGGRR = FF0000
        assert all(c == "FF0000" for c in color_tags)

    def test_empty_no_color_in_standard_mode(self):
        """speaker_color='' in standard mode should not add \\c tags."""
        sups = _make_sups_no_alignment("Alice", "Bob")
        result = ASSFormat.to_bytes(sups, word_level=False, style=CaptionStyle(speaker_color=""))
        content = result.decode("utf-8")

        assert "{\\c&H" not in content

    def test_comma_separated_in_standard_mode(self):
        """Comma-separated colors in standard mode should assign per speaker."""
        sups = _make_sups_no_alignment("Alice", "Bob", "Carol")
        result = ASSFormat.to_bytes(sups, word_level=False, style=CaptionStyle(speaker_color="#FF0000,#00FF00,#0000FF"))
        content = result.decode("utf-8")

        import re

        color_tags = re.findall(r"\{\\c&H([0-9A-Fa-f]{6})&\}", content)
        unique_colors = set(color_tags)
        assert len(unique_colors) == 3

    def test_no_speaker_means_no_color_tag(self):
        """Supervisions without speaker names should not get color tags."""
        sups = [
            Supervision(text="No speaker here", start=0.0, duration=2.0, speaker=""),
        ]
        result = ASSFormat.to_bytes(sups, word_level=False, style=CaptionStyle(speaker_color="auto"))
        content = result.decode("utf-8")

        # No speaker → no color override
        assert "{\\c&H" not in content


class TestSpeakerColorEdgeCases:
    """Edge cases for speaker_color handling."""

    def test_single_speaker_auto(self):
        """A single speaker with 'auto' should still get a color."""
        sups = _make_sups_with_speakers("Solo")
        config = KaraokeConfig(enabled=True)
        result = ASSFormat.to_bytes(sups, word_level=True, karaoke=config, style=CaptionStyle(speaker_color="auto"))
        content = result.decode("utf-8")

        assert "{\\c&H" in content

    def test_color_reset_tag_present(self):
        """After speaker color, a \\c reset tag should appear to restore default color."""
        sups = _make_sups_with_speakers("Alice")
        config = KaraokeConfig(enabled=True)
        result = ASSFormat.to_bytes(sups, word_level=True, karaoke=config, style=CaptionStyle(speaker_color="auto"))
        content = result.decode("utf-8")

        # Should have reset tag {\\c} after the speaker prefix
        assert "{\\c}" in content

    def test_hex_color_with_hash(self):
        """Colors with # prefix should be handled correctly."""
        cache = {}
        c = ASSFormat._resolve_speaker_color("Test", "#AABBCC", cache)
        # AABBCC → CCBBAA
        assert c == "CCBBAA"

    def test_hex_color_without_hash(self):
        """Colors without # prefix should also work (lstrip handles it)."""
        cache = {}
        c = ASSFormat._resolve_speaker_color("Test", "AABBCC", cache)
        # AABBCC → CCBBAA
        assert c == "CCBBAA"

    def test_mixed_speakers_and_empty(self):
        """Mix of supervisions with and without speakers should only color those with speakers."""
        sups = [
            Supervision(text="Hello", start=0.0, duration=1.0, speaker="Alice"),
            Supervision(text="World", start=1.0, duration=1.0, speaker=""),
            Supervision(text="Bye", start=2.0, duration=1.0, speaker="Bob"),
        ]
        result = ASSFormat.to_bytes(sups, word_level=False, style=CaptionStyle(speaker_color="auto"))
        content = result.decode("utf-8")

        # Find dialogue lines
        lines = [l for l in content.splitlines() if l.startswith("Dialogue:")]
        assert len(lines) == 3

        # Lines with speakers should have color; the one without should not
        assert "{\\c&H" in lines[0]  # Alice
        assert "{\\c&H" not in lines[1]  # no speaker
        assert "{\\c&H" in lines[2]  # Bob
