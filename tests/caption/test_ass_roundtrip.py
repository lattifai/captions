"""P0-2: ASS raw text preservation for roundtrip fidelity.

Verifies that ASS override tags, drawing commands, and bilingual \\N separators
are preserved through read → write roundtrip.
"""

import pysubs2
import pytest

from lattifai.caption.config import ASSConfig, RenderConfig
from lattifai.caption.formats.pysubs2 import ASSFormat
from lattifai.caption.supervision import Supervision


class TestASSRawTextPreservation:
    """event.text (with override tags) must be stored in custom['ass_raw_text']."""

    def test_read_stores_raw_text(self):
        """ASS reader should store event.text in custom['ass_raw_text']."""
        ass_content = (
            "[Script Info]\nScriptType: v4.00+\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, "
            "Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
            "0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            r"Dialogue: 0,0:00:01.00,0:00:05.00,Default,,0,0,0,,{\an8}Hello World" + "\n"
        )
        sups = ASSFormat.parse(ass_content).supervisions
        assert len(sups) == 1
        # custom should contain ass_raw_text with override tags
        assert "ass_raw_text" in sups[0].custom
        assert r"{\an8}" in sups[0].custom["ass_raw_text"]
        # sup.text should be plain text (no tags)
        assert r"{\an8}" not in sups[0].text
        assert "Hello World" in sups[0].text

    def test_read_preserves_bilingual_newline(self):
        """\\N in ASS text should be preserved and accessible."""
        ass_content = (
            "[Script Info]\nScriptType: v4.00+\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, "
            "Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
            "0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            r"Dialogue: 0,0:00:01.00,0:00:05.00,Default,,0,0,0,,他决不会放弃塔伯特\NHe would never give up on Talbot"
            + "\n"
        )
        sups = ASSFormat.parse(ass_content).supervisions
        assert len(sups) == 1
        # plaintext should have \n instead of \N
        assert "\n" in sups[0].text
        assert "他决不会放弃塔伯特" in sups[0].text
        assert "He would never give up on Talbot" in sups[0].text
        # raw text should have \N
        assert r"\N" in sups[0].custom["ass_raw_text"]

    def test_roundtrip_preserves_override_tags(self):
        """Override tags should survive read → write roundtrip."""
        ass_content = (
            "[Script Info]\nScriptType: v4.00+\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, "
            "Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
            "0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            r"Dialogue: 0,0:00:01.00,0:00:05.00,Default,,0,0,0,,{\an8\fad(0,500)}Hello World" + "\n"
        )
        sups = ASSFormat.parse(ass_content).supervisions
        metadata = ASSFormat.parse(ass_content).format_metadata

        # Write back
        output = ASSFormat.to_bytes(sups, metadata=metadata, render=RenderConfig(include_speaker_in_text=False))
        output_text = output.decode("utf-8")

        # The override tags should be in the output
        assert r"{\an8\fad(0,500)}" in output_text

    def test_read_detects_drawing_command(self):
        """P1-2: ASS drawing events should be marked as line_type='drawing'."""
        ass_content = (
            "[Script Info]\nScriptType: v4.00+\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, "
            "Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
            "0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            r"Dialogue: 0,0:00:01.00,0:00:05.00,Default,,0,0,0,,{\p1}m 20 0 l 209 0 b 218 0" + "\n"
            "Dialogue: 0,0:00:05.00,0:00:10.00,Default,,0,0,0,,Normal dialogue\n"
        )
        sups = ASSFormat.parse(ass_content).supervisions
        assert len(sups) == 2
        assert sups[0].custom.get("line_type") == "drawing"
        assert sups[1].custom.get("line_type") is None  # normal dialogue, no line_type

    def test_read_preserves_comment_events(self):
        """P1-3: ASS Comment events should be marked in custom['ass_is_comment']."""
        ass_content = (
            "[Script Info]\nScriptType: v4.00+\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, "
            "Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
            "0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            "Comment: 0,0:00:01.00,0:00:05.00,Default,,0,0,0,,This is a comment\n"
            "Dialogue: 0,0:00:05.00,0:00:10.00,Default,,0,0,0,,This is dialogue\n"
        )
        sups = ASSFormat.parse(ass_content).supervisions
        assert len(sups) == 2
        assert sups[0].custom.get("ass_is_comment") is True
        assert sups[1].custom.get("ass_is_comment") is False

    def test_roundtrip_preserves_comment_type(self):
        """Comment events should remain Comment (not become Dialogue) on roundtrip."""
        ass_content = (
            "[Script Info]\nScriptType: v4.00+\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, "
            "Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
            "0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            "Comment: 0,0:00:01.00,0:00:05.00,Default,,0,0,0,,Hidden comment\n"
            "Dialogue: 0,0:00:05.00,0:00:10.00,Default,,0,0,0,,Visible dialogue\n"
        )
        sups = ASSFormat.parse(ass_content).supervisions
        metadata = ASSFormat.parse(ass_content).format_metadata
        output = ASSFormat.to_bytes(sups, metadata=metadata, render=RenderConfig(include_speaker_in_text=False))
        output_text = output.decode("utf-8")
        assert "Comment:" in output_text
        assert "Dialogue:" in output_text

    def test_roundtrip_preserves_script_info(self):
        """P0-3: Script Info section should be preserved exactly on roundtrip."""
        ass_content = (
            "[Script Info]\n"
            "; // The sub is created by AssToolkit\n"
            "Title: My Custom Title\n"
            "ScriptType: v4.00+\n"
            "Synch Point:0\n"
            "Timer:100.0000\n"
            "PlayResX: 1280\n"
            "PlayResY: 720\n\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, "
            "Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
            "0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            "Dialogue: 0,0:00:01.00,0:00:05.00,Default,,0,0,0,,Hello World\n"
        )
        sups = ASSFormat.parse(ass_content).supervisions
        metadata = ASSFormat.parse(ass_content).format_metadata
        output = ASSFormat.to_bytes(sups, metadata=metadata, render=RenderConfig(include_speaker_in_text=False))
        output_text = output.decode("utf-8")

        # Custom comment should be preserved
        assert "; // The sub is created by AssToolkit" in output_text
        # pysubs2 default comment should NOT be present
        assert "; Script generated by pysubs2" not in output_text
        # Non-standard fields should be preserved
        assert "Synch Point:0" in output_text
        assert "Timer:100.0000" in output_text

    def test_roundtrip_preserves_bilingual_structure(self):
        """Bilingual \\N separator should survive read → write roundtrip."""
        ass_content = (
            "[Script Info]\nScriptType: v4.00+\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, "
            "Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
            "0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            r"Dialogue: 0,0:00:01.00,0:00:05.00,Default,,0,0,0,,你好世界\NHello World" + "\n"
        )
        sups = ASSFormat.parse(ass_content).supervisions
        metadata = ASSFormat.parse(ass_content).format_metadata

        output = ASSFormat.to_bytes(sups, metadata=metadata, render=RenderConfig(include_speaker_in_text=False))
        output_text = output.decode("utf-8")

        # \N should be preserved, not \n
        assert r"\NHello World" in output_text or r"\N" in output_text
