"""Tests for WebVTT W3C standard compliance.

Covers features added to achieve full spec alignment:
- Cue settings (align, line, position, size, vertical, region)
- Voice tags (<v name>)
- Inline formatting tags (<b>, <i>, <u>)
- Class tags (<c.classname>)
- Ruby annotations (<ruby><rt>)
- Language tags (<lang>)
- REGION blocks
- STYLE block read/roundtrip
- Multi-line NOTE comments
- VTTConfig
"""

import pytest

from lattifai.caption import Caption, Supervision
from lattifai.caption.config import RenderConfig, VTTConfig
from lattifai.caption.formats.vtt import VTTFormat


# =============================================================================
# Cue Settings
# =============================================================================


class TestCueSettings:
    """Cue settings are stored in Supervision.custom['vtt_*'] on read,
    and written back to the timestamp line on write."""

    def test_read_align(self):
        content = """WEBVTT

00:00:01.000 --> 00:00:03.000 align:center
Centered text
"""
        sups = VTTFormat.parse(content).supervisions
        assert len(sups) == 1
        assert sups[0].custom["vtt_align"] == "center"
        assert sups[0].text == "Centered text"

    def test_read_multiple_settings(self):
        content = """WEBVTT

00:00:01.000 --> 00:00:03.000 align:center line:90% size:80%
Styled text
"""
        sups = VTTFormat.parse(content).supervisions
        assert sups[0].custom["vtt_align"] == "center"
        assert sups[0].custom["vtt_line"] == "90%"
        assert sups[0].custom["vtt_size"] == "80%"

    def test_read_vertical(self):
        content = """WEBVTT

00:00:01.000 --> 00:00:03.000 vertical:rl
Vertical text
"""
        sups = VTTFormat.parse(content).supervisions
        assert sups[0].custom["vtt_vertical"] == "rl"

    def test_read_position(self):
        content = """WEBVTT

00:00:01.000 --> 00:00:03.000 position:20%
Positioned text
"""
        sups = VTTFormat.parse(content).supervisions
        assert sups[0].custom["vtt_position"] == "20%"

    def test_read_region_ref(self):
        content = """WEBVTT

00:00:01.000 --> 00:00:03.000 region:scroll_area
Region text
"""
        sups = VTTFormat.parse(content).supervisions
        assert sups[0].custom["vtt_region"] == "scroll_area"

    def test_read_line_integer(self):
        content = """WEBVTT

00:00:01.000 --> 00:00:03.000 line:-1
Bottom text
"""
        sups = VTTFormat.parse(content).supervisions
        assert sups[0].custom["vtt_line"] == "-1"

    def test_no_settings_means_no_custom(self):
        content = """WEBVTT

00:00:01.000 --> 00:00:03.000
Plain text
"""
        sups = VTTFormat.parse(content).supervisions
        assert sups[0].custom is None

    def test_write_cue_settings_roundtrip(self):
        """Cue settings survive read → write → read roundtrip."""
        content = """WEBVTT

00:00:01.000 --> 00:00:03.000 align:center line:90% size:80%
Styled text
"""
        sups = VTTFormat.parse(content).supervisions
        output = VTTFormat.to_bytes(sups).decode("utf-8")

        assert "align:center" in output
        assert "line:90%" in output
        assert "size:80%" in output

        # Re-read and verify
        sups2 = VTTFormat.parse(output).supervisions
        assert sups2[0].custom["vtt_align"] == "center"
        assert sups2[0].custom["vtt_line"] == "90%"
        assert sups2[0].custom["vtt_size"] == "80%"

    def test_write_settings_spec_order(self):
        """Cue settings are written in W3C spec order: vertical, line, position, size, align, region."""
        sup = Supervision(
            text="Test",
            start=1.0,
            duration=2.0,
            custom={"vtt_align": "center", "vtt_vertical": "rl", "vtt_size": "80%"},
        )
        output = VTTFormat.to_bytes([sup]).decode("utf-8")
        # Find the timestamp line
        for line in output.split("\n"):
            if "-->" in line:
                # vertical should come before size which should come before align
                v_idx = line.index("vertical:")
                s_idx = line.index("size:")
                a_idx = line.index("align:")
                assert v_idx < s_idx < a_idx
                break

    def test_vttconfig_defaults_applied(self):
        """VTTConfig.default_* values appear in output when supervision has no custom."""
        sup = Supervision(text="Test", start=1.0, duration=2.0)
        config = VTTConfig(default_align="center", default_line="90%")
        output = VTTFormat.to_bytes([sup], config=config).decode("utf-8")

        assert "align:center" in output
        assert "line:90%" in output

    def test_supervision_custom_overrides_config_default(self):
        """Per-supervision vtt_* overrides VTTConfig defaults."""
        sup = Supervision(
            text="Test",
            start=1.0,
            duration=2.0,
            custom={"vtt_align": "end"},
        )
        config = VTTConfig(default_align="center")
        output = VTTFormat.to_bytes([sup], config=config).decode("utf-8")

        assert "align:end" in output
        assert "align:center" not in output

    def test_youtube_vtt_cue_settings(self):
        """YouTube VTT also extracts cue settings."""
        content = """\
WEBVTT

00:00:00.080 --> 00:00:02.550 align:start position:0%
The<00:00:00.320><c> diffusion</c>
"""
        sups = VTTFormat.parse(content).supervisions
        assert len(sups) == 1
        assert sups[0].custom["vtt_align"] == "start"
        assert sups[0].custom["vtt_position"] == "0%"


# =============================================================================
# Voice Tags (<v>)
# =============================================================================


class TestVoiceTags:
    """<v name> voice tags map to Supervision.speaker."""

    def test_read_voice_tag_closed(self):
        content = """WEBVTT

00:00:01.000 --> 00:00:03.000
<v Alice>Hello world</v>
"""
        sups = VTTFormat.parse(content).supervisions
        assert sups[0].speaker == "Alice"
        assert sups[0].text == "Hello world"

    def test_read_voice_tag_unclosed(self):
        content = """WEBVTT

00:00:01.000 --> 00:00:03.000
<v Bob>Hi there
"""
        sups = VTTFormat.parse(content).supervisions
        assert sups[0].speaker == "Bob"
        assert sups[0].text == "Hi there"

    def test_read_voice_tag_with_formatting(self):
        content = """WEBVTT

00:00:01.000 --> 00:00:03.000
<v Alice>Hello <b>world</b></v>
"""
        sups = VTTFormat.parse(content).supervisions
        assert sups[0].speaker == "Alice"
        assert "<b>world</b>" in sups[0].text

    def test_write_voice_tag_enabled(self):
        """VTTConfig.voice_tag=True writes <v Speaker>text</v>."""
        sup = Supervision(text="Hello world", start=1.0, duration=2.0, speaker="Alice")
        config = VTTConfig(voice_tag=True)
        output = VTTFormat.to_bytes([sup], config=config).decode("utf-8")

        assert "<v Alice>Hello world</v>" in output

    def test_write_voice_tag_disabled(self):
        """VTTConfig.voice_tag=False uses 'Speaker: text' prefix (default)."""
        sup = Supervision(text="Hello world", start=1.0, duration=2.0, speaker="Alice")
        output = VTTFormat.to_bytes([sup]).decode("utf-8")

        assert "Alice: Hello world" in output
        assert "<v" not in output

    def test_voice_tag_roundtrip(self):
        """Read <v> → write with voice_tag=True → read back."""
        content = """WEBVTT

00:00:01.000 --> 00:00:03.000
<v Alice>Hello world</v>
"""
        sups = VTTFormat.parse(content).supervisions
        assert sups[0].speaker == "Alice"

        output = VTTFormat.to_bytes(sups, config=VTTConfig(voice_tag=True)).decode("utf-8")
        assert "<v Alice>" in output

        sups2 = VTTFormat.parse(output).supervisions
        assert sups2[0].speaker == "Alice"
        assert sups2[0].text == "Hello world"

    def test_write_no_speaker_no_voice_tag(self):
        """No speaker → no <v> tag regardless of config."""
        sup = Supervision(text="Hello", start=1.0, duration=2.0)
        config = VTTConfig(voice_tag=True)
        output = VTTFormat.to_bytes([sup], config=config).decode("utf-8")

        assert "<v" not in output


# =============================================================================
# Inline Formatting Tags
# =============================================================================


class TestInlineFormatting:
    """Inline tags (<b>, <i>, <u>, <c>, <ruby>, <lang>) are preserved in text."""

    def test_bold_preserved(self):
        content = """WEBVTT

00:00:01.000 --> 00:00:03.000
Hello <b>world</b>
"""
        sups = VTTFormat.parse(content).supervisions
        assert "<b>world</b>" in sups[0].text

    def test_italic_preserved(self):
        content = """WEBVTT

00:00:01.000 --> 00:00:03.000
<i>italic text</i>
"""
        sups = VTTFormat.parse(content).supervisions
        assert "<i>italic text</i>" in sups[0].text

    def test_underline_preserved(self):
        content = """WEBVTT

00:00:01.000 --> 00:00:03.000
<u>underlined</u>
"""
        sups = VTTFormat.parse(content).supervisions
        assert "<u>underlined</u>" in sups[0].text

    def test_class_tag_preserved(self):
        content = """WEBVTT

00:00:01.000 --> 00:00:03.000
Welcome to <c.brand-text>LattifAI</c>
"""
        sups = VTTFormat.parse(content).supervisions
        assert "<c.brand-text>LattifAI</c>" in sups[0].text

    def test_ruby_preserved(self):
        content = """WEBVTT

00:00:01.000 --> 00:00:03.000
<ruby>漢字<rt>かんじ</rt></ruby>
"""
        sups = VTTFormat.parse(content).supervisions
        assert "<ruby>" in sups[0].text
        assert "<rt>かんじ</rt>" in sups[0].text

    def test_lang_preserved(self):
        content = """WEBVTT

00:00:01.000 --> 00:00:03.000
He said <lang zh>你好世界</lang> to everyone
"""
        sups = VTTFormat.parse(content).supervisions
        assert "<lang zh>你好世界</lang>" in sups[0].text

    def test_nested_tags_preserved(self):
        content = """WEBVTT

00:00:01.000 --> 00:00:03.000
<v Alice><b>Bold</b> and <i>italic</i></v>
"""
        sups = VTTFormat.parse(content).supervisions
        assert sups[0].speaker == "Alice"
        assert "<b>Bold</b>" in sups[0].text
        assert "<i>italic</i>" in sups[0].text

    def test_formatting_roundtrip(self):
        """Inline tags survive read → write → read."""
        content = """WEBVTT

00:00:01.000 --> 00:00:03.000
Hello <b>bold</b> and <i>italic</i>
"""
        sups = VTTFormat.parse(content).supervisions
        output = VTTFormat.to_bytes(sups).decode("utf-8")
        assert "<b>bold</b>" in output
        assert "<i>italic</i>" in output

        sups2 = VTTFormat.parse(output).supervisions
        assert "<b>bold</b>" in sups2[0].text
        assert "<i>italic</i>" in sups2[0].text


# =============================================================================
# REGION Blocks
# =============================================================================


class TestRegions:
    """REGION blocks are parsed from input and written back in output."""

    REGION_VTT = """WEBVTT

REGION
id:scroll_area
width:40%
lines:3
regionanchor:0%,100%
viewportanchor:10%,90%
scroll:up

00:00:01.000 --> 00:00:03.000 region:scroll_area
Scrolling text

00:00:03.000 --> 00:00:05.000
Normal text
"""

    def test_read_region_metadata(self):
        result = VTTFormat.parse(self.REGION_VTT)
        regions = result.format_metadata.get("vtt_regions")
        assert regions is not None
        assert len(regions) == 1
        assert regions[0]["id"] == "scroll_area"
        assert regions[0]["width"] == "40%"
        assert regions[0]["lines"] == "3"
        assert regions[0]["scroll"] == "up"

    def test_read_region_cue_ref(self):
        sups = VTTFormat.parse(self.REGION_VTT).supervisions
        assert sups[0].custom["vtt_region"] == "scroll_area"
        assert sups[1].custom is None

    def test_write_region_roundtrip(self):
        result = VTTFormat.parse(self.REGION_VTT)
        output = VTTFormat.to_bytes(
            result.supervisions,
            metadata=result.format_metadata,
        ).decode("utf-8")

        assert "REGION" in output
        assert "id:scroll_area" in output
        assert "width:40%" in output
        assert "scroll:up" in output
        assert "region:scroll_area" in output

    def test_multiple_regions(self):
        content = """WEBVTT

REGION
id:top
width:50%
lines:2

REGION
id:bottom
width:80%
lines:3
scroll:up

00:00:01.000 --> 00:00:03.000 region:top
Top region text
"""
        result = VTTFormat.parse(content)
        regions = result.format_metadata["vtt_regions"]
        assert len(regions) == 2
        assert regions[0]["id"] == "top"
        assert regions[1]["id"] == "bottom"


# =============================================================================
# STYLE Blocks
# =============================================================================


class TestStyleReading:
    """STYLE blocks are parsed from input and roundtripped."""

    def test_read_style_block(self):
        content = """WEBVTT

STYLE
::cue {
  color: white;
  background-color: rgba(0,0,0,0.8);
}

00:00:01.000 --> 00:00:03.000
Styled text
"""
        result = VTTFormat.parse(content)
        styles = result.format_metadata.get("vtt_styles")
        assert styles is not None
        assert len(styles) == 1
        assert "color: white;" in styles[0]
        assert "background-color:" in styles[0]

    def test_style_roundtrip(self):
        content = """WEBVTT

STYLE
::cue {
  color: white;
  font-family: Arial;
}

00:00:01.000 --> 00:00:03.000
Styled text
"""
        result = VTTFormat.parse(content)
        output = VTTFormat.to_bytes(
            result.supervisions,
            metadata=result.format_metadata,
        ).decode("utf-8")

        assert "STYLE" in output
        assert "color: white;" in output
        assert "font-family: Arial;" in output

    def test_multiple_style_blocks(self):
        content = """WEBVTT

STYLE
::cue {
  color: white;
}

STYLE
::cue(v[voice="Alice"]) {
  color: blue;
}

00:00:01.000 --> 00:00:03.000
Text
"""
        result = VTTFormat.parse(content)
        styles = result.format_metadata["vtt_styles"]
        assert len(styles) == 2

    def test_vtt_styles_take_precedence_over_ass_styles(self):
        """When both vtt_styles and ass_styles exist, vtt_styles are used."""
        sups = [Supervision(text="Test", start=1.0, duration=2.0)]
        metadata = {
            "vtt_styles": ["::cue { color: red; }"],
            "ass_styles": {"Default": {"primarycolor": "&H00FFFFFF"}},
        }
        output = VTTFormat.to_bytes(sups, metadata=metadata).decode("utf-8")

        assert "color: red;" in output
        # ass_styles should NOT generate a second STYLE block
        assert output.count("STYLE") == 1


# =============================================================================
# NOTE Comments
# =============================================================================


class TestNoteComments:
    """NOTE blocks are parsed and roundtripped."""

    def test_single_line_note(self):
        content = """WEBVTT

NOTE author: John Doe

00:00:01.000 --> 00:00:03.000
Text
"""
        result = VTTFormat.parse(content)
        assert result.format_metadata.get("author") == "John Doe"
        notes = result.format_metadata.get("vtt_notes")
        assert notes is not None
        assert "author: John Doe" in notes[0]

    def test_multi_line_note(self):
        content = """WEBVTT

NOTE
This is a multi-line
comment block

00:00:01.000 --> 00:00:03.000
Text
"""
        result = VTTFormat.parse(content)
        notes = result.format_metadata["vtt_notes"]
        assert len(notes) == 1
        assert "multi-line" in notes[0]
        assert "comment block" in notes[0]

    def test_note_roundtrip(self):
        content = """WEBVTT

NOTE author: Generated by LattifAI

00:00:01.000 --> 00:00:03.000
Text
"""
        result = VTTFormat.parse(content)
        output = VTTFormat.to_bytes(
            result.supervisions,
            metadata=result.format_metadata,
        ).decode("utf-8")

        assert "NOTE author: Generated by LattifAI" in output


# =============================================================================
# VTTConfig
# =============================================================================


class TestVTTConfig:
    """VTTConfig controls WebVTT output behavior."""

    def test_default_config(self):
        config = VTTConfig()
        assert config.default_align is None
        assert config.voice_tag is False
        assert config.preserve_formatting is True

    def test_invalid_align_raises(self):
        with pytest.raises(ValueError, match="default_align"):
            VTTConfig(default_align="bad")

    def test_invalid_vertical_raises(self):
        with pytest.raises(ValueError, match="default_vertical"):
            VTTConfig(default_vertical="up")

    def test_valid_values(self):
        config = VTTConfig(
            default_align="center",
            default_vertical="rl",
            default_line="90%",
            default_position="50%",
            default_size="80%",
            voice_tag=True,
        )
        assert config.default_align == "center"
        assert config.default_vertical == "rl"

    def test_config_in_format_config_map(self):
        """VTTConfig is registered in the format config map."""
        from lattifai.caption.config import resolve_format_config

        config = resolve_format_config("vtt", {"default_align": "center"})
        assert isinstance(config, VTTConfig)
        assert config.default_align == "center"


# =============================================================================
# Comprehensive Roundtrip
# =============================================================================


class TestComprehensiveRoundtrip:
    """End-to-end roundtrip with all features combined."""

    FULL_VTT = """WEBVTT
Kind: captions
Language: en

REGION
id:scroll_area
width:40%
lines:3
regionanchor:0%,100%
viewportanchor:10%,90%
scroll:up

STYLE
::cue {
  color: white;
  background-color: rgba(0,0,0,0.8);
}

NOTE Generated by LattifAI

1
00:00:00.500 --> 00:00:03.000 align:center line:90% size:80%
<v Host>Welcome to <c.brand>LattifAI</c></v>

2
00:00:03.500 --> 00:00:06.000 vertical:rl line:0
Vertical text

3
00:00:06.500 --> 00:00:09.000 region:scroll_area
<v Guest>Thank you for having me</v>
"""

    def test_full_parse(self):
        result = VTTFormat.parse(self.FULL_VTT)

        # Metadata
        assert result.language == "en"
        assert result.kind == "captions"

        # Regions
        regions = result.format_metadata["vtt_regions"]
        assert len(regions) == 1
        assert regions[0]["id"] == "scroll_area"

        # Styles
        styles = result.format_metadata["vtt_styles"]
        assert len(styles) == 1
        assert "color: white;" in styles[0]

        # Notes
        notes = result.format_metadata["vtt_notes"]
        assert "Generated by LattifAI" in notes[0]

        # Supervisions
        sups = result.supervisions
        assert len(sups) == 3

        # Cue 1: voice tag + cue settings + class tag
        assert sups[0].speaker == "Host"
        assert "<c.brand>LattifAI</c>" in sups[0].text
        assert sups[0].custom["vtt_align"] == "center"
        assert sups[0].custom["vtt_line"] == "90%"
        assert sups[0].custom["vtt_size"] == "80%"

        # Cue 2: vertical text
        assert sups[1].custom["vtt_vertical"] == "rl"
        assert sups[1].custom["vtt_line"] == "0"

        # Cue 3: region ref + voice tag
        assert sups[2].speaker == "Guest"
        assert sups[2].custom["vtt_region"] == "scroll_area"

    def test_full_roundtrip(self):
        """Parse → write → re-parse preserves all features."""
        result = VTTFormat.parse(self.FULL_VTT)

        metadata = {
            **result.format_metadata,
            "kind": result.kind,
            "language": result.language,
        }
        output = VTTFormat.to_bytes(
            result.supervisions,
            metadata=metadata,
            config=VTTConfig(voice_tag=True),
        ).decode("utf-8")

        # Re-parse
        result2 = VTTFormat.parse(output)

        assert result2.kind == "captions"
        assert result2.language == "en"
        assert len(result2.format_metadata["vtt_regions"]) == 1
        assert len(result2.format_metadata["vtt_styles"]) == 1

        sups2 = result2.supervisions
        assert len(sups2) == 3
        assert sups2[0].speaker == "Host"
        assert sups2[0].custom["vtt_align"] == "center"
        assert sups2[2].custom["vtt_region"] == "scroll_area"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge cases and backward compatibility."""

    def test_cue_with_id(self):
        """Numeric cue IDs are preserved."""
        content = """WEBVTT

1
00:00:01.000 --> 00:00:03.000
First cue

intro
00:00:03.000 --> 00:00:05.000
Second cue
"""
        sups = VTTFormat.parse(content).supervisions
        assert len(sups) == 2
        assert sups[0].id == "1"
        assert sups[1].id == "intro"

    def test_empty_vtt(self):
        content = """WEBVTT
"""
        sups = VTTFormat.parse(content).supervisions
        assert len(sups) == 0

    def test_multiline_cue_text(self):
        """Multi-line cue text is joined with space when normalize_text=True."""
        content = """WEBVTT

00:00:01.000 --> 00:00:03.000
Line one
Line two
"""
        sups = VTTFormat.parse(content, normalize_text=True).supervisions
        assert sups[0].text == "Line one Line two"

    def test_voice_tag_with_full_name(self):
        """Voice tag with spaces in name."""
        content = """WEBVTT

00:00:01.000 --> 00:00:03.000
<v John Doe>Hello world</v>
"""
        sups = VTTFormat.parse(content).supervisions
        assert sups[0].speaker == "John Doe"

    def test_mixed_cues_with_and_without_settings(self):
        content = """WEBVTT

00:00:01.000 --> 00:00:03.000
Plain text

00:00:03.000 --> 00:00:05.000 align:center
Centered text
"""
        sups = VTTFormat.parse(content).supervisions
        assert sups[0].custom is None
        assert sups[1].custom["vtt_align"] == "center"

    def test_caption_write_with_vttconfig(self):
        """VTTConfig works through Caption.write() API."""
        sups = [Supervision(text="Test", start=1.0, duration=2.0, speaker="Alice")]
        caption = Caption.from_supervisions(sups)
        from lattifai.caption.config import VTTConfig

        output = caption.to_bytes(
            output_format="vtt",
            format_config=VTTConfig(voice_tag=True, default_align="center"),
        ).decode("utf-8")

        assert "<v Alice>" in output
        assert "align:center" in output


# =============================================================================
# Speaker preservation across consecutive cues
#
# The text-prefix encoding of speakers ("ALICE: hello") on VTT historically
# applied a tracker-based dedup, dropping the prefix on the second cue when
# the speaker hadn't changed. That broke fidelity for roundtrips and for
# cases where the source explicitly tagged every cue. The contract is now:
#
#   - ``original_speaker=True`` (the default) → emit the prefix on every
#     cue with a speaker, regardless of the previous cue's speaker. This
#     guarantees the read side can always recover the speaker and matches
#     the source author's intent.
#   - ``original_speaker=False`` → speaker was inherited / back-filled by
#     the producer (e.g. sentence splitter); suppress the prefix when it
#     equals the previous cue's speaker so display stays clean.
#
# These tests pin both halves of that contract.
# =============================================================================


class TestVTTSpeakerConsecutivePreservation:
    """Cover the speaker-front-to-back change matrix on VTT writes."""

    def _write_and_reparse(self, sups):
        caption = Caption.from_supervisions(sups)
        text = caption.to_bytes(output_format="vtt").decode("utf-8")
        reparsed = VTTFormat.parse(text).supervisions
        return text, reparsed

    def test_consecutive_same_explicit_speaker_preserved(self):
        """ALICE → ALICE (both explicit) must keep prefix on both cues."""
        sups = [
            Supervision(text="hello", start=0.0, duration=1.0, speaker="ALICE"),
            Supervision(text="hello again", start=1.5, duration=1.0, speaker="ALICE"),
        ]
        text, reparsed = self._write_and_reparse(sups)
        # Wire-level: every cue line carries its speaker prefix.
        assert text.count("ALICE:") == 2, f"Expected two ALICE: prefixes; output:\n{text}"
        # Roundtrip: speaker recovered on both cues. Reader keeps the
        # trailing colon as part of the parsed speaker label.
        assert [s.speaker for s in reparsed] == ["ALICE:", "ALICE:"]

    def test_three_consecutive_same_explicit_speaker_all_preserved(self):
        """A → A → A keeps the prefix on every cue (no silent drop in the middle)."""
        sups = [
            Supervision(text="line one", start=0.0, duration=1.0, speaker="A"),
            Supervision(text="line two", start=1.5, duration=1.0, speaker="A"),
            Supervision(text="line three", start=3.0, duration=1.0, speaker="A"),
        ]
        text, reparsed = self._write_and_reparse(sups)
        assert text.count("A:") == 3, f"Expected three A: prefixes; output:\n{text}"
        assert [s.speaker for s in reparsed] == ["A:", "A:", "A:"]

    def test_speaker_change_pattern_alice_none_alice(self):
        """ALICE → None → ALICE. The continuation cue stays empty; the
        recurrence carries its own prefix."""
        sups = [
            Supervision(text="first", start=0.0, duration=1.0, speaker="ALICE"),
            Supervision(text="middle continuation", start=1.5, duration=1.0, speaker=None),
            Supervision(text="alice again", start=3.0, duration=1.0, speaker="ALICE"),
        ]
        text, reparsed = self._write_and_reparse(sups)
        assert [s.speaker for s in reparsed] == ["ALICE:", None, "ALICE:"], reparsed
        assert text.count("ALICE:") == 2

    def test_alternating_speakers_each_prefix_kept(self):
        """ALICE → BOB → ALICE → BOB; every cue keeps its prefix."""
        sups = [
            Supervision(text=f"line {i}", start=i * 1.5, duration=1.0, speaker=("ALICE" if i % 2 == 0 else "BOB"))
            for i in range(4)
        ]
        text, reparsed = self._write_and_reparse(sups)
        assert text.count("ALICE:") == 2
        assert text.count("BOB:") == 2
        assert [s.speaker for s in reparsed] == ["ALICE:", "BOB:", "ALICE:", "BOB:"]

    def test_inherited_speaker_consecutive_dedup_suppresses_prefix(self):
        """Legacy contract: when the producer flags a cue as inherited
        (``custom['original_speaker']=False``), the writer should suppress
        the prefix on consecutive duplicates so the rendered output stays
        clean. The first cue still emits."""
        sups = [
            Supervision(
                text="first part",
                start=0.0,
                duration=1.0,
                speaker="ALICE",
                custom={"original_speaker": True},
            ),
            Supervision(
                text="continuation chunk",
                start=1.5,
                duration=1.0,
                speaker="ALICE",
                custom={"original_speaker": False},  # inherited from neighbour
            ),
            Supervision(
                text="another inherited",
                start=3.0,
                duration=1.0,
                speaker="ALICE",
                custom={"original_speaker": False},
            ),
        ]
        text, reparsed = self._write_and_reparse(sups)
        # Only the first cue emits ALICE: ; the two inherited cues are suppressed.
        assert text.count("ALICE:") == 1, f"Expected exactly one ALICE: prefix; got:\n{text}"
        # Roundtrip then collapses the inherited speakers to None on read.
        assert [s.speaker for s in reparsed] == ["ALICE:", None, None], reparsed

    def test_explicit_after_inherited_resumes_prefix(self):
        """Inherited duplicates suppress the prefix, but a subsequent
        explicit cue (even with the same speaker) must re-emit so the
        reader can recover the speaker again."""
        sups = [
            Supervision(text="first", start=0.0, duration=1.0, speaker="ALICE"),  # default original
            Supervision(
                text="inherited",
                start=1.5,
                duration=1.0,
                speaker="ALICE",
                custom={"original_speaker": False},
            ),
            Supervision(text="explicit again", start=3.0, duration=1.0, speaker="ALICE"),  # default original
        ]
        text, reparsed = self._write_and_reparse(sups)
        # First and third explicit cues emit prefix; middle inherited is suppressed.
        assert text.count("ALICE:") == 2, f"Expected two ALICE: prefixes; got:\n{text}"
        assert [s.speaker for s in reparsed] == ["ALICE:", None, "ALICE:"], reparsed

    def test_eight_supervision_multi_speaker_layout_roundtrips(self):
        """End-to-end: the eight-cue multi-speaker layout used by the
        backend regression suite must roundtrip without dropping the
        explicitly-tagged BOB on the second consecutive BOB cue."""
        # Use colon-suffixed labels to match what the parser returns on
        # read-back (``parse_speaker_text`` keeps the trailing colon).
        layout = [
            "ALICE:",
            None,
            "BOB:",
            "BOB:",  # ← previously dropped by VTT dedup
            "[SPEAKER_02]:",
            None,
            ">> DAVID:",
            "ALICE:",
        ]
        sups = [
            Supervision(text=f"line {i}", start=i * 2.0, duration=1.5, speaker=sp) for i, sp in enumerate(layout)
        ]
        text, reparsed = self._write_and_reparse(sups)
        assert [s.speaker for s in reparsed] == layout, reparsed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
