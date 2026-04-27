#!/usr/bin/env python3
"""
Comprehensive test suite for all caption formats
"""

from pathlib import Path

import pytest

from lattifai.caption import Caption, Supervision
from lattifai.caption.config import RenderConfig


class TestCaptionFormats:
    """Test all supported caption formats."""

    @pytest.mark.parametrize(
        "format_ext,content_template",
        [
            ("srt", "1\n00:00:01,000 --> 00:00:03,000\n{text}\n"),
            ("vtt", "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\n{text}\n"),
            ("ass", "[Events]\nDialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,{text}\n"),
            ("ssa", "[Events]\nDialogue: Marked=0,0:00:01.00,0:00:03.00,Default,NTP,0,0,0,,{text}\n"),
        ],
    )
    def test_pysubs2_format_read(self, tmp_path, format_ext, content_template):
        """Test reading various pysubs2-supported formats."""
        test_text = "Test subtitle text"
        content = content_template.format(text=test_text)

        file_path = tmp_path / f"test.{format_ext}"
        file_path.write_text(content)

        caption = Caption.read(file_path)
        assert isinstance(caption, Caption)
        assert len(caption.supervisions) > 0
        assert test_text in caption.supervisions[0].text
        print(f"✓ Read {format_ext.upper()} format successfully")

    @pytest.mark.parametrize(
        "format_ext",
        ["srt", "vtt", "ass", "ssa", "sub", "sbv", "txt", "json", "TextGrid"],
    )
    def test_format_write(self, tmp_path, format_ext):
        """Test writing various caption formats."""
        supervisions = [
            Supervision(text="First line", start=1.0, duration=2.0),
            Supervision(text="Second line", start=4.0, duration=2.0),
        ]

        caption = Caption.from_supervisions(supervisions)
        output_file = tmp_path / f"output.{format_ext}"

        result_path = caption.write(output_file)
        assert output_file.exists()
        assert result_path == output_file
        print(f"✓ Write {format_ext.upper()} format successfully")

    def test_sbv_format_complete(self, tmp_path):
        """Test SBV format with comprehensive scenarios."""
        # Test with multiline text
        sbv_content = """0:00:01.000,0:00:03.500
First line
Second line of same subtitle

0:00:04.000,0:00:06.500
Single line subtitle

0:00:07.000,0:00:09.000
SPEAKER ONE: Dialogue with speaker
"""
        sbv_file = tmp_path / "test.sbv"
        sbv_file.write_text(sbv_content)

        caption = Caption.read(sbv_file)
        assert len(caption.supervisions) == 3
        assert "First line Second line" in caption.supervisions[0].text
        assert caption.supervisions[2].speaker == "SPEAKER ONE:"
        print("✓ SBV multiline and speaker handling works")

    def test_txt_format_with_timestamps(self, tmp_path):
        """Test TXT format with timestamp markers."""
        txt_content = """[1.0-3.0] First line with timestamp
[4.0-6.0] SPEAKER: Second line with speaker
Plain line without timestamp
"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text(txt_content)

        caption = Caption.read(txt_file)
        assert len(caption.supervisions) == 3
        assert caption.supervisions[0].start == 1.0
        assert caption.supervisions[1].speaker == "SPEAKER:"
        print("✓ TXT format with timestamps works")

    def test_format_round_trip(self, tmp_path):
        """Test write and read back maintains data integrity."""
        original_supervisions = [
            Supervision(text="First line", start=1.0, duration=2.0, speaker="ALICE"),
            Supervision(text="Second line", start=4.0, duration=2.0, speaker="BOB"),
        ]

        formats_to_test = ["srt", "vtt", "sbv", "json"]

        for fmt in formats_to_test:
            caption = Caption.from_supervisions(original_supervisions)
            output_file = tmp_path / f"test.{fmt}"

            # Write
            caption.write(output_file)

            # Read back
            caption_readback = Caption.read(output_file)

            # Verify
            assert len(caption_readback.supervisions) == len(original_supervisions)
            assert abs(caption_readback.supervisions[0].start - 1.0) < 0.1
            assert abs(caption_readback.supervisions[1].start - 4.0) < 0.1

            print(f"✓ Round-trip for {fmt.upper()} format successful")

    def test_textgrid_format(self, tmp_path):
        """Test TextGrid format writing."""
        supervisions = [
            Supervision(text="First utterance", start=0.0, duration=2.0, speaker="SPEAKER_01"),
            Supervision(text="Second utterance", start=2.5, duration=1.5, speaker="SPEAKER_02"),
        ]

        caption = Caption.from_supervisions(supervisions)
        output_file = tmp_path / "output.TextGrid"

        caption.write(output_file)
        assert output_file.exists()

        # Read back and verify - speakers should be separate tiers
        content = output_file.read_text()
        assert "SPEAKER_01" in content
        assert "SPEAKER_02" in content
        assert "First utterance" in content
        assert "Second utterance" in content

        # Verify roundtrip preserves speaker as tier name
        read_back = Caption.read(output_file)
        speakers = {s.speaker for s in read_back.supervisions}
        assert "SPEAKER_01" in speakers
        assert "SPEAKER_02" in speakers
        print("✓ TextGrid format works correctly")

    @pytest.mark.parametrize(
        "special_chars",
        [
            "Text with 'quotes'",
            'Text with "double quotes"',
            "Text with <tags>",
            "Text with & ampersand",
            "Text with émojis 😀",
        ],
    )
    def test_special_characters_handling(self, tmp_path, special_chars):
        """Test handling of special characters in various formats."""
        supervisions = [Supervision(text=special_chars, start=1.0, duration=2.0)]
        caption = Caption.from_supervisions(supervisions)

        # Test formats that support round-trip
        for fmt in ["srt", "vtt", "sbv", "json"]:
            output_file = tmp_path / f"special_{fmt}.{fmt}"
            caption.write(output_file)

            # Read back
            caption_readback = Caption.read(output_file)
            # Basic check - should not crash
            assert len(caption_readback.supervisions) > 0

        print(f"✓ Special characters '{special_chars[:20]}...' handled correctly")


class TestFormatCoverage:
    """Test format coverage completeness."""

    def test_all_input_formats_defined(self):
        """Verify all input formats are properly defined."""
        from lattifai.caption.config import INPUT_CAPTION_FORMATS

        expected_formats = ["srt", "vtt", "ass", "ssa", "sub", "sbv", "txt", "sami", "smi", "auto", "gemini"]

        for fmt in expected_formats:
            assert fmt in INPUT_CAPTION_FORMATS, f"Format {fmt} missing from INPUT_CAPTION_FORMATS"

        print(f"✓ All {len(expected_formats)} input formats are defined")

    def test_all_output_formats_defined(self):
        """Verify all output formats are properly defined."""
        from lattifai.caption.config import OUTPUT_CAPTION_FORMATS

        expected_formats = ["srt", "vtt", "ass", "ssa", "sub", "sbv", "txt", "ttml", "sami", "smi", "textgrid", "json"]

        for fmt in expected_formats:
            assert fmt in OUTPUT_CAPTION_FORMATS, f"Format {fmt} missing from OUTPUT_CAPTION_FORMATS"

        print(f"✓ All {len(expected_formats)} output formats are defined")

    def test_format_detection_coverage(self):
        """Test that format detection works for all common extensions."""
        from lattifai.caption.config import ALL_CAPTION_FORMATS

        common_formats = ["srt", "vtt", "ass", "ssa", "sub", "sbv", "txt", "textgrid", "json", "gemini"]

        for fmt in common_formats:
            assert fmt in ALL_CAPTION_FORMATS, f"Format {fmt} not in ALL_CAPTION_FORMATS"

        print(f"✓ All {len(common_formats)} common formats are detected")


class TestPysubs2SpeakerFormat:
    """Test pysubs2 format speaker handling including word-level output."""

    def test_pysubs2_speaker_in_text(self, tmp_path):
        """Test that speaker is included in text when include_speaker_in_text=True."""
        supervisions = [
            Supervision(text="Hello world", start=1.0, duration=2.0, speaker="ALICE"),
            Supervision(text="Goodbye", start=4.0, duration=1.0, speaker="BOB"),
        ]

        caption = Caption.from_supervisions(supervisions)
        output_file = tmp_path / "output.srt"
        caption.write(output_file)

        content = output_file.read_text()
        # Speaker prepended with ': ' separator for parseable roundtripping
        assert "ALICE: Hello world" in content
        assert "BOB: Goodbye" in content

    def test_pysubs2_speaker_not_in_text(self, tmp_path):
        """Test that speaker is not in text when include_speaker_in_text=False."""
        supervisions = [
            Supervision(text="Hello world", start=1.0, duration=2.0, speaker="ALICE"),
        ]

        caption = Caption.from_supervisions(supervisions)
        output_file = tmp_path / "output.srt"
        caption.write(output_file, render=RenderConfig(include_speaker_in_text=False))

        content = output_file.read_text()
        assert "Hello world" in content
        assert "ALICE Hello world" not in content

    def test_pysubs2_word_level_with_speaker(self, tmp_path):
        """Test word-level output uses consistent speaker across all words."""
        from lattifai.caption.supervision import AlignmentItem

        # Create supervision with word-level alignment
        supervisions = [
            Supervision(
                text="Hello world",
                start=1.0,
                duration=2.0,
                speaker="SPEAKER_01",
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=1.0, duration=0.5),
                        AlignmentItem(symbol="world", start=1.6, duration=0.4),
                    ]
                },
            ),
        ]

        caption = Caption.from_supervisions(supervisions)
        output_file = tmp_path / "output.vtt"
        caption.write(output_file)

        content = output_file.read_text()
        # Word-level output should have each word as separate subtitle
        assert "Hello" in content
        assert "world" in content

    def test_pysubs2_word_level_without_speaker(self, tmp_path):
        """Test word-level output without speaker inclusion."""
        from lattifai.caption.supervision import AlignmentItem

        # Create supervision with word-level alignment
        supervisions = [
            Supervision(
                text="Hello world",
                start=1.0,
                duration=2.0,
                speaker="SPEAKER_01",
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=1.0, duration=0.5),
                        AlignmentItem(symbol="world", start=1.6, duration=0.4),
                    ]
                },
            ),
        ]

        caption = Caption.from_supervisions(supervisions)
        output_file = tmp_path / "output.vtt"
        caption.write(output_file, render=RenderConfig(include_speaker_in_text=False))

        content = output_file.read_text()
        # Words should be present
        assert "Hello" in content
        assert "world" in content

    def test_pysubs2_null_speaker_handling(self, tmp_path):
        """Test that null speaker is handled correctly."""
        supervisions = [
            Supervision(text="No speaker here", start=1.0, duration=2.0, speaker=None),
        ]

        caption = Caption.from_supervisions(supervisions)
        output_file = tmp_path / "output.srt"
        caption.write(output_file)

        content = output_file.read_text()
        assert "No speaker here" in content
        # Should not have "None" or empty prefix
        assert "None No speaker" not in content


class TestPysubs2Preprocessing:
    """Test preprocessing of Gemini-style thinking/meta blocks in pysubs2 formats."""

    def test_srt_with_yaml_frontmatter(self, tmp_path):
        """Test SRT file with YAML front matter is parsed correctly."""
        srt_with_frontmatter = """---
title: Test Video
language: en
---
1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:04,000 --> 00:00:06,000
Second line
"""
        srt_file = tmp_path / "frontmatter.srt"
        srt_file.write_text(srt_with_frontmatter)

        caption = Caption.read(srt_file)
        assert len(caption.supervisions) == 2
        assert caption.supervisions[0].text == "Hello world"
        assert caption.supervisions[1].text == "Second line"
        print("✓ SRT with YAML front matter parsed correctly")

    def test_srt_with_thinking_block(self, tmp_path):
        """Test SRT file with thinking block is parsed correctly."""
        srt_with_thinking = """<thinking>
This is internal reasoning that should be filtered out.
Let me analyze the content...
</thinking>
1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:04,000 --> 00:00:06,000
Second line
"""
        srt_file = tmp_path / "thinking.srt"
        srt_file.write_text(srt_with_thinking)

        caption = Caption.read(srt_file)
        assert len(caption.supervisions) == 2
        assert caption.supervisions[0].text == "Hello world"
        assert "thinking" not in caption.supervisions[0].text.lower()
        print("✓ SRT with thinking block parsed correctly")

    def test_srt_with_both_frontmatter_and_thinking(self, tmp_path):
        """Test SRT file with both front matter and thinking block."""
        srt_mixed = """---
title: Mixed Test
---
<thinking>
Internal analysis...
</thinking>
1
00:00:01,000 --> 00:00:03,000
First subtitle

2
00:00:05,000 --> 00:00:07,000
Second subtitle
"""
        srt_file = tmp_path / "mixed.srt"
        srt_file.write_text(srt_mixed)

        caption = Caption.read(srt_file)
        assert len(caption.supervisions) == 2
        assert caption.supervisions[0].text == "First subtitle"
        assert caption.supervisions[1].text == "Second subtitle"
        print("✓ SRT with both front matter and thinking block parsed correctly")

    def test_vtt_with_thinking_block(self, tmp_path):
        """Test VTT file with thinking block is parsed correctly."""
        vtt_with_thinking = """<thinking>
Gemini reasoning content...
</thinking>
WEBVTT

00:00:01.000 --> 00:00:03.000
Hello world
"""
        vtt_file = tmp_path / "thinking.vtt"
        vtt_file.write_text(vtt_with_thinking)

        caption = Caption.read(vtt_file)
        assert len(caption.supervisions) == 1
        assert caption.supervisions[0].text == "Hello world"
        print("✓ VTT with thinking block parsed correctly")

    def test_srt_content_string_with_frontmatter(self):
        """Test parsing SRT content string with front matter."""
        from lattifai.caption.formats.pysubs2 import SRTFormat

        srt_content = """---
meta: data
---
1
00:00:01,000 --> 00:00:02,000
Test line
"""
        supervisions = SRTFormat.parse(srt_content).supervisions
        assert len(supervisions) == 1
        assert supervisions[0].text == "Test line"
        print("✓ SRT content string with front matter parsed correctly")

    def test_normal_srt_unaffected(self, tmp_path):
        """Test that normal SRT without meta blocks still works."""
        normal_srt = """1
00:00:01,000 --> 00:00:03,000
Normal subtitle

2
00:00:04,000 --> 00:00:06,000
Another line
"""
        srt_file = tmp_path / "normal.srt"
        srt_file.write_text(normal_srt)

        caption = Caption.read(srt_file)
        assert len(caption.supervisions) == 2
        assert caption.supervisions[0].text == "Normal subtitle"
        print("✓ Normal SRT still works correctly")


class TestSrtRawTextSplice:
    """Tests for SRT writer's raw-cue splicing behavior.

    The reader caches each cue's raw text in `sup.custom['srt_raw_text']` so
    override tags (e.g. `{\\an1}{\\pos(...)}`) can be preserved on a
    pure round-trip. The writer must NOT splice when the user has mutated
    `sup.text` or set `sup.translation`, otherwise translation work is
    silently discarded.
    """

    def _round_trip_with_mutation(
        self, tmp_path, mutate, expect_in_output, must_not_in_output
    ):
        srt_in = """1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:04,000 --> 00:00:06,000
Another line
"""
        srt_file = tmp_path / "in.srt"
        srt_file.write_text(srt_in, encoding="utf-8")
        caption = Caption.read(srt_file)
        for sup in caption.supervisions:
            mutate(sup)
        out = tmp_path / "out.srt"
        caption.write(out)
        # Caption writer always emits UTF-8; specify it explicitly so this
        # test passes on Windows where ``read_text`` defaults to cp1252 and
        # cannot decode CJK bytes injected by the bilingual mutation.
        text = out.read_text(encoding="utf-8")
        for needle in expect_in_output:
            assert needle in text, f"expected {needle!r} in:\n{text}"
        for needle in must_not_in_output:
            assert needle not in text, f"unexpected {needle!r} in:\n{text}"

    def test_replace_mode_no_silent_drop(self, tmp_path):
        """Mutating sup.text must propagate to the SRT output (no splice override)."""
        replacements = ["Hola mundo", "Otra linea"]

        def mutate(sup):
            sup.text = replacements.pop(0)

        self._round_trip_with_mutation(
            tmp_path,
            mutate=mutate,
            expect_in_output=["Hola mundo", "Otra linea"],
            must_not_in_output=["Hello world", "Another line"],
        )

    def test_bilingual_translation_preserved(self, tmp_path):
        """Setting sup.translation must produce a bilingual SRT (no splice override)."""
        translations = ["你好世界", "另一行"]

        def mutate(sup):
            sup.translation = translations.pop(0)
            sup.target_lang = "zh"

        self._round_trip_with_mutation(
            tmp_path,
            mutate=mutate,
            expect_in_output=["Hello world", "你好世界", "Another line", "另一行"],
            must_not_in_output=[],
        )

    def test_unchanged_round_trip_still_splices_overrides(self, tmp_path):
        """Pure round-trip with override tags still preserves them."""
        srt_in = """1
00:00:01,000 --> 00:00:03,000
{\\an8}Top line

2
00:00:04,000 --> 00:00:06,000
{\\pos(960,540)}Centered
"""
        srt_file = tmp_path / "in.srt"
        srt_file.write_text(srt_in, encoding="utf-8")
        caption = Caption.read(srt_file)
        # NO mutation — pure round-trip
        out = tmp_path / "out.srt"
        caption.write(out)
        text = out.read_text(encoding="utf-8")
        assert "{\\an8}Top line" in text, f"override tag dropped:\n{text}"
        assert "{\\pos(960,540)}Centered" in text, f"override tag dropped:\n{text}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
