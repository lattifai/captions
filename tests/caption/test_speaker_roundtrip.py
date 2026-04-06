#!/usr/bin/env python3
"""
Test suite for speaker label roundtrip consistency
"""

import tempfile
from pathlib import Path

import pytest

from lattifai.caption import Caption, Supervision
from lattifai.caption.config import RenderConfig
from lattifai.caption.supervision import AlignmentItem


class TestSpeakerRoundtrip:
    """Test speaker label preservation across read/write operations."""

    def test_srt_speaker_roundtrip_lattifai_format(self, tmp_path):
        """Test [SPEAKER]: format roundtrip in SRT."""
        # Create supervisions with speaker labels
        supervisions = [
            Supervision(text="Hello world", start=1.0, duration=2.0, speaker="[SPEAKER_01]:"),
            Supervision(text="How are you", start=4.0, duration=2.0, speaker="ALICE:"),
            Supervision(text="I'm fine", start=7.0, duration=2.0, speaker="BOB:"),
        ]

        # Write to SRT file
        srt_file = tmp_path / "test.srt"
        caption = Caption.from_supervisions(supervisions)
        caption.write(srt_file)

        # Read back
        caption_read = Caption.read(srt_file)

        # Verify speaker labels are preserved
        assert len(caption_read.supervisions) == 3
        # Note: parse_speaker_text returns the full prefix including colon
        assert caption_read.supervisions[0].speaker == "[SPEAKER_01]:"
        assert caption_read.supervisions[0].text == "Hello world"
        assert caption_read.supervisions[1].speaker == "ALICE:"
        assert caption_read.supervisions[1].text == "How are you"
        assert caption_read.supervisions[2].speaker == "BOB:"
        assert caption_read.supervisions[2].text == "I'm fine"

        print("✓ SRT [SPEAKER]: format roundtrip successful")

    def test_vtt_speaker_roundtrip(self, tmp_path):
        """Test speaker roundtrip in VTT format."""
        supervisions = [
            Supervision(text="First line", start=1.0, duration=2.0, speaker="[SPEAKER_00]:"),
            Supervision(text="Second line", start=4.0, duration=2.0, speaker="[SPEAKER_01]:"),
        ]

        # Write to VTT file
        vtt_file = tmp_path / "test.vtt"
        caption = Caption.from_supervisions(supervisions)
        caption.write(vtt_file)

        # Read back
        caption_read = Caption.read(vtt_file)

        # Verify
        assert len(caption_read.supervisions) == 2
        assert caption_read.supervisions[0].speaker == "[SPEAKER_00]:"
        assert caption_read.supervisions[0].text == "First line"
        assert caption_read.supervisions[1].speaker == "[SPEAKER_01]:"
        assert caption_read.supervisions[1].text == "Second line"

        print("✓ VTT speaker roundtrip successful")

    def test_txt_speaker_roundtrip(self, tmp_path):
        """Test speaker roundtrip in TXT format.

        Note: New format uses "speaker text" instead of "[speaker]: text".
        """
        supervisions = [
            Supervision(text="Line one", start=1.0, duration=2.0, speaker="[NARRATOR]:"),
            Supervision(text="Line two", start=4.0, duration=2.0, speaker="[ALICE]:"),
        ]

        # Write to TXT file
        txt_file = tmp_path / "test.txt"
        caption = Caption.from_supervisions(supervisions)
        caption.write(txt_file)

        # Read and check content format - new format: "speaker text"
        content = txt_file.read_text()
        assert "[NARRATOR]: Line one" in content
        assert "[ALICE]: Line two" in content

        print("✓ TXT speaker format correct")

    def test_speaker_format_parsing(self, tmp_path):
        """Test that different speaker formats are correctly parsed."""
        # Create SRT with different speaker formats
        srt_content = """1
00:00:01,000 --> 00:00:03,000
[SPEAKER_01]: First speaker format

2
00:00:04,000 --> 00:00:06,000
>> ALICE: Second speaker format

3
00:00:07,000 --> 00:00:09,000
BOB: Third speaker format
"""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(srt_content)

        # Read and verify all formats are parsed
        caption = Caption.read(srt_file)

        assert len(caption.supervisions) == 3

        # [SPEAKER_01]: format
        assert caption.supervisions[0].speaker == "[SPEAKER_01]:"
        assert caption.supervisions[0].text == "First speaker format"

        # >> ALICE: format
        assert caption.supervisions[1].speaker == ">> ALICE:"
        assert caption.supervisions[1].text == "Second speaker format"

        # BOB: format (Gemini style)
        assert caption.supervisions[2].speaker == "BOB:"
        assert caption.supervisions[2].text == "Third speaker format"

        print("✓ Multiple speaker formats parsed correctly")

    def test_write_without_speaker(self, tmp_path):
        """Test writing without speaker labels."""
        supervisions = [
            Supervision(text="Hello", start=1.0, duration=2.0, speaker="[ALICE]:"),
            Supervision(text="World", start=4.0, duration=2.0, speaker="[BOB]:"),
        ]

        # Write without speaker
        srt_file = tmp_path / "test.srt"
        caption = Caption.from_supervisions(supervisions)
        caption.write(srt_file, render=RenderConfig(include_speaker_in_text=False))

        # Read back
        caption_read = Caption.read(srt_file)

        # Speaker info should still be in event.name but not in text
        assert caption_read.supervisions[0].text == "Hello"
        assert caption_read.supervisions[1].text == "World"
        # pysubs2 stores speaker in event.name, which gets assigned to supervision.speaker
        # When include_speaker_in_text=False, only event.name is used (no colon format)
        assert not caption_read.supervisions[0].speaker
        assert not caption_read.supervisions[1].speaker

        print("✓ Write without speaker in text successful")

    def test_srt_titlecase_speaker_roundtrip(self, tmp_path):
        """Test title-case speaker names (no colon) roundtrip in SRT.

        SRT has no dedicated speaker field. Title-case names are detected by
        detect_speaker_candidates which requires ≥3 occurrences of the same
        name pattern. This test uses a realistic interview scenario.
        """
        supervisions = [
            Supervision(text="Welcome to the show.", start=1.0, duration=3.0, speaker="Host"),
            Supervision(text="Thanks for having me.", start=5.0, duration=2.0, speaker="Terence Tao"),
            Supervision(text="Let's start with math.", start=8.0, duration=2.0, speaker="Host"),
            Supervision(text="Sure, great topic.", start=11.0, duration=2.0, speaker="Terence Tao"),
            Supervision(text="How did you get into it?", start=14.0, duration=2.0, speaker="Host"),
            Supervision(text="It started early.", start=17.0, duration=2.0, speaker="Terence Tao"),
        ]

        srt_file = tmp_path / "test.srt"
        caption = Caption.from_supervisions(supervisions)
        caption.write(srt_file)

        # Verify written format has separator
        content = srt_file.read_text()
        assert "Host: Welcome to the show." in content
        assert "Terence Tao: Thanks for having me." in content

        # Read back — detect_speaker_candidates finds Host (3x) and Terence Tao (3x)
        caption_read = Caption.read(srt_file)
        assert len(caption_read.supervisions) == 6
        for sup in caption_read.supervisions:
            assert sup.speaker is not None, f"Speaker lost for: {sup.text}"
            assert not sup.text.startswith("Host:"), f"Speaker prefix leaked into text: {sup.text}"
            assert not sup.text.startswith("Terence Tao:"), f"Speaker prefix leaked into text: {sup.text}"

    def test_ass_speaker_roundtrip_with_include(self, tmp_path):
        """Test ASS roundtrip: include_speaker_in_text=True preserves speaker."""
        supervisions = [
            Supervision(text="Hello world", start=1.0, duration=2.0, speaker="Host"),
            Supervision(text="Goodbye", start=4.0, duration=2.0, speaker="Guest"),
        ]

        ass_file = tmp_path / "test.ass"
        caption = Caption.from_supervisions(supervisions)
        caption.write(ass_file)

        content = ass_file.read_text()
        # Name field should have speaker
        assert ",Host," in content
        assert ",Guest," in content
        # Text should have speaker with separator
        assert "Host: Hello world" in content
        assert "Guest: Goodbye" in content

        # Read back
        caption_read = Caption.read(ass_file)
        assert len(caption_read.supervisions) == 2
        assert caption_read.supervisions[0].speaker is not None
        assert caption_read.supervisions[0].text == "Hello world"
        assert caption_read.supervisions[1].text == "Goodbye"

    def test_speaker_change_marker_no_colon_srt(self, tmp_path):
        """>> speaker should produce '>> text' not '>>: text' in SRT output."""
        supervisions = [
            Supervision(text="Hello world", start=1.0, duration=2.0, speaker=">>"),
            Supervision(text="Goodbye", start=4.0, duration=2.0, speaker="Alice"),
        ]

        srt_file = tmp_path / "test.srt"
        caption = Caption.from_supervisions(supervisions)
        caption.write(srt_file)

        content = srt_file.read_text()
        # >> speaker should NOT have a colon prefix
        assert ">>:" not in content, f"Found '>>:' in SRT output:\n{content}"
        assert ">> Hello world" in content, f"Expected '>> Hello world' in SRT output:\n{content}"

    def test_speaker_change_marker_no_colon_vtt(self, tmp_path):
        """>> speaker should produce '>> text' not '>>: text' in VTT output."""
        supervisions = [
            Supervision(text="Hello world", start=1.0, duration=2.0, speaker=">>"),
            Supervision(text="Goodbye", start=4.0, duration=2.0, speaker="Alice"),
        ]

        vtt_file = tmp_path / "test.vtt"
        caption = Caption.from_supervisions(supervisions)
        caption.write(vtt_file)

        content = vtt_file.read_text()
        assert ">>:" not in content, f"Found '>>:' in VTT output:\n{content}"
        assert ">> Hello world" in content, f"Expected '>> Hello world' in VTT output:\n{content}"

    def test_word_level_youtube_vtt_speaker_change_roundtrip(self, tmp_path):
        """Word-level YouTube VTT should preserve bare >> speaker markers on roundtrip."""
        supervisions = [
            Supervision(
                text="Hello world",
                start=1.0,
                duration=1.0,
                speaker=">>",
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=1.0, duration=0.5),
                        AlignmentItem(symbol="world", start=1.5, duration=0.5),
                    ]
                },
            )
        ]

        vtt_file = tmp_path / "word_level_speaker_change.vtt"
        caption = Caption.from_supervisions(supervisions)
        caption.write(
            vtt_file,
            render=RenderConfig(word_level=True),
        )

        content = vtt_file.read_text()
        assert ">> <00:00:01.000><c> Hello</c><00:00:01.500><c> world</c>" in content

        caption_read = Caption.read(vtt_file)

        assert len(caption_read.supervisions) == 1
        assert caption_read.supervisions[0].speaker == ">>"
        assert caption_read.supervisions[0].text == "Hello world"
        assert caption_read.supervisions[0].alignment is not None
        assert caption_read.supervisions[0].alignment["word"][0].symbol == "Hello"

    def test_speaker_change_marker_no_colon_ass(self, tmp_path):
        """>> speaker should produce '>> text' not '>>: text' in ASS output."""
        supervisions = [
            Supervision(text="Hello world", start=1.0, duration=2.0, speaker=">>"),
            Supervision(text="Goodbye", start=4.0, duration=2.0, speaker="Alice"),
        ]

        ass_file = tmp_path / "test.ass"
        caption = Caption.from_supervisions(supervisions)
        caption.write(ass_file)

        content = ass_file.read_text()
        assert ">>:" not in content, f"Found '>>:' in ASS output:\n{content}"
        assert ">> Hello world" in content, f"Expected '>> Hello world' in ASS output:\n{content}"

    def test_speaker_change_marker_no_colon_txt(self, tmp_path):
        """>> speaker should produce '>> text' not '>>: text' in TXT output."""
        supervisions = [
            Supervision(text="Hello world", start=1.0, duration=2.0, speaker=">>"),
            Supervision(text="Goodbye", start=4.0, duration=2.0, speaker="Alice"),
        ]

        txt_file = tmp_path / "test.txt"
        caption = Caption.from_supervisions(supervisions)
        caption.write(txt_file)

        content = txt_file.read_text()
        assert ">>:" not in content, f"Found '>>:' in TXT output:\n{content}"
        assert ">> Hello world" in content, f"Expected '>> Hello world' in TXT output:\n{content}"

    def test_format_speaker_prefix_with_speaker_change(self):
        """_format_speaker_prefix should return '>> ' for '>>' speaker (no colon)."""
        from lattifai.caption.formats.base import FormatWriter

        assert FormatWriter._format_speaker_prefix(">>") == ">> "
        # Normal speakers still get colon
        assert FormatWriter._format_speaker_prefix("Alice") == "Alice: "
        assert FormatWriter._format_speaker_prefix("BOB:") == "BOB: "

    def test_speaker_change_marker_word_level_vtt_roundtrip(self, tmp_path):
        """>> speaker with word-level alignment must survive VTT write/read roundtrip."""
        supervisions = [
            Supervision(
                text="Hello world",
                start=1.0,
                duration=2.0,
                speaker=">>",
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=1.0, duration=0.8),
                        AlignmentItem(symbol="world", start=1.9, duration=1.1),
                    ]
                },
            ),
        ]

        vtt_file = tmp_path / "test.vtt"
        caption = Caption.from_supervisions(supervisions)
        caption.write(
            vtt_file,
            render=RenderConfig(word_level=True),
        )

        content = vtt_file.read_text()
        # >> should NOT be inside a <c> token — it should be a bare prefix
        assert ">>:" not in content, f"Found '>>:' in word-level VTT:\n{content}"
        assert ">> <00:00:01.000><c> Hello</c><00:00:01.900><c> world</c>" in content

        # Read back
        caption_read = Caption.read(vtt_file)
        assert len(caption_read.supervisions) == 1
        sup = caption_read.supervisions[0]
        assert sup.speaker == ">>", f"Speaker lost on roundtrip, got: {sup.speaker}"
        assert not sup.text.startswith(">>"), f">> leaked into text: {sup.text}"
        # First word alignment should be clean
        if sup.alignment and sup.alignment.get("word"):
            first_word = sup.alignment["word"][0].symbol
            assert not first_word.startswith(">>"), f">> leaked into first word: {first_word}"

    def test_ass_opaque_box_color_exports_correctly_to_vtt(self, tmp_path):
        """ASS borderstyle=3 should export OutlineColour (fill) as VTT background-color."""
        from lattifai.caption.config import ASSConfig

        supervisions = [
            Supervision(text="Hello", start=1.0, duration=2.0),
        ]

        # Red fill box, black shadow — these must be different to catch the bug
        style = ASSConfig(background_color="#FF0000", back_color="#000000")

        ass_file = tmp_path / "test.ass"
        caption = Caption.from_supervisions(supervisions)
        caption.write(ass_file, format_config=style)

        # Read back ASS to get metadata with styles
        caption_read = Caption.read(ass_file)

        # Now write to VTT (which converts ASS style to CSS)
        vtt_file = tmp_path / "test.vtt"
        caption_read.write(vtt_file)

        vtt_content = vtt_file.read_text()
        # The CSS background-color should be the fill (red #FF0000), not the shadow (black #000000)
        assert "background-color" in vtt_content, f"No background-color in VTT:\n{vtt_content}"
        bg_line = [l for l in vtt_content.splitlines() if "background-color" in l][0]
        # Should contain red, not black
        assert "#000000" not in bg_line, \
            f"VTT background-color is shadow color (black) instead of fill (red):\n{bg_line}"

    def test_ass_speaker_roundtrip_without_include(self, tmp_path):
        """Test ASS roundtrip: include_speaker_in_text=False uses Name field only."""
        supervisions = [
            Supervision(text="Hello world", start=1.0, duration=2.0, speaker="Host"),
            Supervision(text="Goodbye", start=4.0, duration=2.0, speaker="Guest"),
        ]

        ass_file = tmp_path / "test.ass"
        caption = Caption.from_supervisions(supervisions)
        caption.write(ass_file, render=RenderConfig(include_speaker_in_text=False))

        content = ass_file.read_text()
        # Name field should still have speaker
        assert ",Host," in content
        assert ",Guest," in content
        # Text should NOT have speaker prefix
        assert ",,Hello world" in content
        assert ",,Goodbye" in content

        # Read back — speaker recovered from Name field
        caption_read = Caption.read(ass_file)
        assert len(caption_read.supervisions) == 2
        assert caption_read.supervisions[0].speaker == "Host"
        assert caption_read.supervisions[0].text == "Hello world"
        assert caption_read.supervisions[1].speaker == "Guest"
        assert caption_read.supervisions[1].text == "Goodbye"


def run_tests():
    """Run all tests."""
    print("🧪 Running Speaker Roundtrip Tests\n")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_suite = TestSpeakerRoundtrip()

        print("\n📄 Testing speaker roundtrip...")
        test_suite.test_srt_speaker_roundtrip_lattifai_format(tmp_path)
        test_suite.test_vtt_speaker_roundtrip(tmp_path)
        test_suite.test_txt_speaker_roundtrip(tmp_path)

        print("\n📄 Testing speaker format parsing...")
        test_suite.test_speaker_format_parsing(tmp_path)
        test_suite.test_write_without_speaker(tmp_path)

    print("\n" + "=" * 60)
    print("✅ All speaker roundtrip tests passed!")
    print("\n📝 Summary:")
    print("   • [SPEAKER]: format used for writing")
    print("   • All formats (>>, [SPEAKER]:, NAME:) correctly parsed")
    print("   • Speaker labels preserved in roundtrip")
    print("   • Supports writing with/without speaker in text")


if __name__ == "__main__":
    import sys

    try:
        run_tests()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
