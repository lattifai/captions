"""Tests for YouTube SRV3/YTT format reader and writer."""

import tempfile
from pathlib import Path

import pytest

from lattifai.caption import Caption, Supervision
from lattifai.caption.config import RenderConfig
from lattifai.caption.formats.srv3 import SRV3Format
from lattifai.caption.supervision import AlignmentItem

# Test data directory
TEST_DATA_DIR = Path(__file__).parent.parent / "data" / "captions"
SRV3_FILE = TEST_DATA_DIR / "DoesFastChargingHurttheBatteryRaw.srv3"


class TestSRV3Format:
    """Test SRV3 format reader."""

    def test_can_read_by_extension(self):
        """Test format detection by file extension."""
        assert SRV3Format.can_read("test.srv3")
        assert SRV3Format.can_read("test.ytt")
        assert not SRV3Format.can_read("test.vtt")
        assert not SRV3Format.can_read("test.srt")

    def test_can_read_by_content(self):
        """Test format detection by content."""
        srv3_content = '<?xml version="1.0"?><timedtext format="3"><body></body></timedtext>'
        assert SRV3Format.can_read(srv3_content)

        vtt_content = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello"
        assert not SRV3Format.can_read(vtt_content)

    def test_read_real_file(self):
        """Test reading real SRV3 file."""
        if not SRV3_FILE.exists():
            pytest.skip(f"Test file not found: {SRV3_FILE}")

        supervisions = SRV3Format.read(SRV3_FILE)

        # Should have multiple segments
        assert len(supervisions) > 0

        # First segment should be "Does fast charging hurt the battery?"
        first = supervisions[0]
        assert "Does" in first.text
        assert "fast" in first.text
        assert "charging" in first.text

        # Check timing (first segment starts at 240ms = 0.24s)
        assert abs(first.start - 0.24) < 0.01

        # Check word-level alignment exists
        assert first.alignment is not None
        assert "word" in first.alignment
        assert len(first.alignment["word"]) > 0

    def test_word_level_timing(self):
        """Test word-level timing extraction."""
        if not SRV3_FILE.exists():
            pytest.skip(f"Test file not found: {SRV3_FILE}")

        supervisions = SRV3Format.read(SRV3_FILE)
        first = supervisions[0]

        # Check word alignment
        words = first.alignment["word"]

        # First word "Does" starts at paragraph start (240ms)
        assert words[0].symbol.strip() == "Does"
        assert abs(words[0].start - 0.24) < 0.01

        # Second word "fast" has offset t="320" -> 240+320 = 560ms
        assert "fast" in words[1].symbol
        assert abs(words[1].start - 0.56) < 0.01

    def test_html_entity_decoding(self):
        """Test HTML entity decoding (e.g., &#39; -> ')."""
        if not SRV3_FILE.exists():
            pytest.skip(f"Test file not found: {SRV3_FILE}")

        supervisions = SRV3Format.read(SRV3_FILE)

        # Find segment with "didn't" (originally &#39;)
        found_apostrophe = False
        for sup in supervisions:
            if "didn't" in sup.text or "didn't" in sup.text:
                found_apostrophe = True
                break

        assert found_apostrophe, "HTML entity &#39; should be decoded to apostrophe"

    def test_skip_empty_paragraphs(self):
        """Test that paragraphs with a='1' (empty) are skipped."""
        # Create test content with empty paragraph
        content = """<?xml version="1.0" encoding="utf-8" ?>
        <timedtext format="3">
        <body>
        <p t="1000" d="2000" w="1"><s ac="0">Hello</s><s t="500" ac="0"> world</s></p>
        <p t="2000" d="1000" w="1" a="1"></p>
        <p t="3000" d="2000" w="1"><s ac="0">Goodbye</s></p>
        </body>
        </timedtext>"""

        supervisions = SRV3Format.read(content)

        # Should only have 2 segments (empty one skipped)
        assert len(supervisions) == 2
        assert "Hello" in supervisions[0].text
        assert "Goodbye" in supervisions[1].text

    def test_extract_metadata(self):
        """Test metadata extraction."""
        if not SRV3_FILE.exists():
            pytest.skip(f"Test file not found: {SRV3_FILE}")

        metadata = SRV3Format.extract_metadata(SRV3_FILE)

        assert metadata.get("source_format") == "srv3"
        assert metadata.get("srv3_format") == "3"

    def test_read_from_content_string(self):
        """Test reading from content string."""
        content = """<?xml version="1.0" encoding="utf-8" ?>
        <timedtext format="3">
        <body>
        <p t="1000" d="2000" w="1"><s ac="0">Hello</s><s t="500" ac="0"> world</s></p>
        </body>
        </timedtext>"""

        supervisions = SRV3Format.read(content)

        assert len(supervisions) == 1
        assert "Hello" in supervisions[0].text
        assert "world" in supervisions[0].text

        # Check timing: 1000ms = 1.0s
        assert abs(supervisions[0].start - 1.0) < 0.01
        assert abs(supervisions[0].duration - 2.0) < 0.01


class TestSRV3Integration:
    """Integration tests with Caption class."""

    def test_read_via_caption_class(self):
        """Test reading SRV3 via Caption.read()."""
        if not SRV3_FILE.exists():
            pytest.skip(f"Test file not found: {SRV3_FILE}")

        caption = Caption.read(SRV3_FILE)

        assert len(caption) > 0
        assert caption.source_format == "srv3"

    def test_convert_srv3_to_srt(self):
        """Test converting SRV3 to SRT format."""
        if not SRV3_FILE.exists():
            pytest.skip(f"Test file not found: {SRV3_FILE}")

        caption = Caption.read(SRV3_FILE)

        # Convert to SRT string
        srt_content = caption.to_string("srt")

        # Should have SRT structure
        assert "1\n" in srt_content
        assert "-->" in srt_content
        assert "Does" in srt_content

    def test_convert_srv3_to_vtt(self):
        """Test converting SRV3 to VTT format."""
        if not SRV3_FILE.exists():
            pytest.skip(f"Test file not found: {SRV3_FILE}")

        caption = Caption.read(SRV3_FILE)

        # Convert to VTT string
        vtt_content = caption.to_string("vtt")

        # Should have VTT structure
        assert "WEBVTT" in vtt_content
        assert "-->" in vtt_content

    def test_convert_srv3_to_json_with_words(self):
        """Test converting SRV3 to JSON preserving word-level timing."""
        if not SRV3_FILE.exists():
            pytest.skip(f"Test file not found: {SRV3_FILE}")

        caption = Caption.read(SRV3_FILE)

        # Convert to JSON with word-level
        json_content = caption.to_string("json", render=RenderConfig(word_level=True))

        # Should contain words array
        assert '"words"' in json_content
        assert '"word"' in json_content


class TestSRV3EdgeCases:
    """Test edge cases and error handling."""

    def test_empty_content(self):
        """Test handling of empty content."""
        content = '<?xml version="1.0"?><timedtext format="3"><body></body></timedtext>'
        supervisions = SRV3Format.read(content)
        assert supervisions == []

    def test_invalid_xml(self):
        """Test handling of invalid XML."""
        # Content that looks like SRV3 but is malformed XML
        content = '<timedtext format="3"><body>not closed properly'
        supervisions = SRV3Format.read(content)
        assert supervisions == []

    def test_missing_body(self):
        """Test handling of missing body element."""
        content = '<?xml version="1.0"?><timedtext format="3"><head></head></timedtext>'
        supervisions = SRV3Format.read(content)
        assert supervisions == []

    def test_paragraph_without_duration(self):
        """Test handling of paragraph without duration."""
        content = """<?xml version="1.0"?>
        <timedtext format="3">
        <body>
        <p t="1000" w="1"><s ac="0">No duration</s></p>
        <p t="2000" d="1000" w="1"><s ac="0">Has duration</s></p>
        </body>
        </timedtext>"""

        supervisions = SRV3Format.read(content)

        # Only paragraph with duration should be included
        assert len(supervisions) == 1
        assert "Has duration" in supervisions[0].text

    def test_word_without_text(self):
        """Test handling of span without text content."""
        content = """<?xml version="1.0"?>
        <timedtext format="3">
        <body>
        <p t="1000" d="2000" w="1"><s ac="0"></s><s t="500" ac="0">world</s></p>
        </body>
        </timedtext>"""

        supervisions = SRV3Format.read(content)

        assert len(supervisions) == 1
        # Empty span should be skipped
        assert supervisions[0].text.strip() == "world"


class TestSRV3Writer:
    """Test SRV3 format writer."""

    def test_write_basic(self):
        """Test basic SRV3 writing."""
        supervisions = [
            Supervision(start=1.0, duration=2.0, text="Hello world"),
            Supervision(start=4.0, duration=1.5, text="Goodbye"),
        ]

        content = SRV3Format.to_bytes(supervisions)
        content_str = content.decode("utf-8")

        # Check XML structure
        assert '<?xml version="1.0" encoding="utf-8" ?>' in content_str
        assert '<timedtext format="3">' in content_str
        assert "<body>" in content_str

        # Check timing (1000ms, 2000ms)
        assert 't="1000"' in content_str
        assert 'd="2000"' in content_str

        # Check text content
        assert "Hello" in content_str
        assert "world" in content_str
        assert "Goodbye" in content_str

    def test_write_with_word_level_timing(self):
        """Test SRV3 writing with word-level timing."""
        supervisions = [
            Supervision(
                start=1.0,
                duration=2.0,
                text="Hello world",
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=1.0, duration=0.5),
                        AlignmentItem(symbol="world", start=1.6, duration=1.4),
                    ]
                },
            ),
        ]

        content = SRV3Format.to_bytes(supervisions, word_level=True)
        content_str = content.decode("utf-8")

        # Check word timing offsets
        # First word at offset 0 (implicit)
        # Second word at offset 600ms (1.6 - 1.0 = 0.6s = 600ms)
        assert 't="600"' in content_str
        assert "Hello" in content_str
        assert "world" in content_str

    def test_write_to_file(self):
        """Test writing SRV3 to file."""
        supervisions = [
            Supervision(start=0.5, duration=1.5, text="Test caption"),
        ]

        with tempfile.NamedTemporaryFile(suffix=".srv3", delete=False) as f:
            output_path = Path(f.name)

        try:
            SRV3Format.write(supervisions, output_path)

            # Verify file exists and has content
            assert output_path.exists()
            content = output_path.read_text()
            assert '<timedtext format="3">' in content
            assert "Test" in content
            assert "caption" in content
        finally:
            output_path.unlink(missing_ok=True)

    def test_write_with_speaker(self):
        """Test SRV3 writing with speaker labels."""
        supervisions = [
            Supervision(start=1.0, duration=2.0, text="Hello", speaker="Alice"),
        ]

        content = SRV3Format.to_bytes(supervisions, include_speaker=True)
        content_str = content.decode("utf-8")

        # Speaker should be included in text
        assert "[Alice]" in content_str
        assert "Hello" in content_str

    def test_write_without_speaker(self):
        """Test SRV3 writing without speaker labels."""
        supervisions = [
            Supervision(start=1.0, duration=2.0, text="Hello", speaker="Alice"),
        ]

        content = SRV3Format.to_bytes(supervisions, include_speaker=False)
        content_str = content.decode("utf-8")

        # Speaker should not be included
        assert "[Alice]" not in content_str
        assert "Hello" in content_str


class TestSRV3RoundTrip:
    """Test SRV3 read/write round-trip."""

    def test_roundtrip_basic(self):
        """Test basic round-trip without word-level timing."""
        original_supervisions = [
            Supervision(start=1.0, duration=2.0, text="Hello world"),
            Supervision(start=4.0, duration=1.5, text="How are you"),
        ]

        # Write to bytes
        content = SRV3Format.to_bytes(original_supervisions, word_level=False)

        # Read back
        read_supervisions = SRV3Format.read(content.decode("utf-8"))

        # Verify count
        assert len(read_supervisions) == 2

        # Verify timing (within 1ms tolerance due to ms conversion)
        assert abs(read_supervisions[0].start - 1.0) < 0.002
        assert abs(read_supervisions[0].duration - 2.0) < 0.002

        # Verify text content
        assert "Hello" in read_supervisions[0].text
        assert "world" in read_supervisions[0].text

    def test_roundtrip_with_word_level(self):
        """Test round-trip with word-level timing."""
        original_supervisions = [
            Supervision(
                start=1.0,
                duration=2.0,
                text="Hello world",
                alignment={
                    "word": [
                        AlignmentItem(symbol="Hello", start=1.0, duration=0.5),
                        AlignmentItem(symbol="world", start=1.6, duration=1.4),
                    ]
                },
            ),
        ]

        # Write with word-level
        content = SRV3Format.to_bytes(original_supervisions, word_level=True)

        # Read back
        read_supervisions = SRV3Format.read(content.decode("utf-8"))

        # Verify word-level alignment was preserved
        assert len(read_supervisions) == 1
        assert read_supervisions[0].alignment is not None
        assert "word" in read_supervisions[0].alignment

        words = read_supervisions[0].alignment["word"]
        assert len(words) == 2

        # Check word timing (within 1ms tolerance)
        assert abs(words[0].start - 1.0) < 0.002
        assert abs(words[1].start - 1.6) < 0.002

    def test_roundtrip_real_file(self):
        """Test round-trip with real SRV3 file."""
        if not SRV3_FILE.exists():
            pytest.skip(f"Test file not found: {SRV3_FILE}")

        # Read original
        original = SRV3Format.read(SRV3_FILE)

        # Write to bytes with word-level
        content = SRV3Format.to_bytes(original, word_level=True)

        # Read back
        roundtrip = SRV3Format.read(content.decode("utf-8"))

        # Verify segment count preserved
        assert len(roundtrip) == len(original)

        # Verify first segment timing preserved
        assert abs(roundtrip[0].start - original[0].start) < 0.002
        assert abs(roundtrip[0].duration - original[0].duration) < 0.002

    def test_roundtrip_via_caption_class(self):
        """Test round-trip using Caption class."""
        if not SRV3_FILE.exists():
            pytest.skip(f"Test file not found: {SRV3_FILE}")

        # Read via Caption
        caption = Caption.read(SRV3_FILE)
        original_count = len(caption)

        # Write to srv3 string
        srv3_str = caption.to_string("srv3", render=RenderConfig(word_level=True))

        # Read back
        caption2 = Caption.from_string(srv3_str, format="srv3")

        # Verify count preserved
        assert len(caption2) == original_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
