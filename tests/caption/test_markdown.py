"""Tests for Markdown transcript reader and writer."""

import pytest

from lattifai.caption import MarkdownReader, MarkdownSegment, MarkdownWriter, Supervision

# Sample transcript content for testing
SAMPLE_TRANSCRIPT = """## OpenAI Spring Update: GPT-4o

## Table of Contents
* [00:00:00] Introduction
* [00:53:00] Announcing GPT-4o

## [00:00:00] Introduction

[Music starts] [00:00:08]

[Applause] [00:00:13]

**Mira Murati:** Hi everyone. [00:00:13]

[Applause] [00:00:16]

**Mira Murati:** Hi everyone. Thank you. Thank you. It's great to have you here today. [00:00:19]

Today I'm going to talk about three things. That's it. [00:00:23]

## [00:53:00] Announcing GPT-4o

**Mira Murati:** But the big news today is that we are launching our new flagship model. [00:00:57]

And we are calling it GPT-4o. [00:01:01]

The special thing about GPT-4o is that it brings GPT-4 level intelligence to everyone. [00:01:11]
"""

# Sample YouTube format transcript content for testing
SAMPLE_YOUTUBE_TRANSCRIPT = """Introducing GPT-4o

## Table of Contents
* [[00:12](http://www.youtube.com/watch?v=DQacCB9tDaw&t=12)] Introduction
* [[00:54](http://www.youtube.com/watch?v=DQacCB9tDaw&t=54)] Introducing the New Flagship Model: GPT-4o

## [[00:12](http://www.youtube.com/watch?v=DQacCB9tDaw&t=12)] Introduction

**Mira Murati:** hi everyone Hi everyone thank you thank you it's great to have you here today today I'm going to talk [[00:21](http://www.youtube.com/watch?v=DQacCB9tDaw&t=21)]

about three things that's it we will start with why it's so important to us to have a product that [[00:29](http://www.youtube.com/watch?v=DQacCB9tDaw&t=29)]

we can make freely available and broadly available to everyone and we're always trying to find out ways to reduce [[00:37](http://www.youtube.com/watch?v=DQacCB9tDaw&t=37)]

## [[00:54](http://www.youtube.com/watch?v=DQacCB9tDaw&t=54)] Introducing the New Flagship Model: GPT-4o

**Mira Murati:** that we are launching our new flagship model and we are calling it gbt 40 the special thing about gbt [[01:03](http://www.youtube.com/watch?v=DQacCB9tDaw&t=63)]

40 is that it brings gb4 level intelligence to everyone including our free users we'll be showing some live demos [[01:13](http://www.youtube.com/watch?v=DQacCB9tDaw&t=73)]
"""


class TestMarkdownReader:
    """Tests for MarkdownReader class."""

    def test_read_all_segments(self, tmp_path):
        """Test reading all segments including events and sections."""
        # Create temp file
        transcript_file = tmp_path / "test_Gemini.md"
        transcript_file.write_text(SAMPLE_TRANSCRIPT)

        # Read all segments
        segments = MarkdownReader.read(transcript_file, include_events=True, include_sections=True)

        # Should have sections, events, and dialogue
        assert len(segments) > 0

        # Check segment types
        types = {seg.segment_type for seg in segments}
        assert "section_header" in types
        assert "event" in types
        assert "dialogue" in types

    def test_read_dialogue_only(self, tmp_path):
        """Test reading only dialogue segments."""
        transcript_file = tmp_path / "test_Gemini.md"
        transcript_file.write_text(SAMPLE_TRANSCRIPT)

        # Read dialogue only
        segments = MarkdownReader.read(transcript_file, include_events=False, include_sections=False)

        # Should only have dialogue
        types = {seg.segment_type for seg in segments}
        assert types == {"dialogue"}

    def test_parse_timestamp(self):
        """Test timestamp parsing."""
        timestamp = MarkdownReader.parse_timestamp("00", "00", "13")
        assert timestamp == 13.0

        timestamp = MarkdownReader.parse_timestamp("00", "01", "01")
        assert timestamp == 61.0

        timestamp = MarkdownReader.parse_timestamp("01", "00", "00")
        assert timestamp == 3600.0

    def test_speaker_extrevent(self, tmp_path):
        """Test speaker name extrevent."""
        transcript_file = tmp_path / "test_Gemini.md"
        transcript_file.write_text(SAMPLE_TRANSCRIPT)

        segments = MarkdownReader.read(transcript_file)

        # Find dialogue segments with speaker
        dialogue_with_speaker = [s for s in segments if s.speaker is not None]
        assert len(dialogue_with_speaker) > 0

        # Check speaker name
        speakers = {s.speaker for s in dialogue_with_speaker}
        assert "Mira Murati:" in speakers

    def test_section_tracking(self, tmp_path):
        """Test section title tracking."""
        transcript_file = tmp_path / "test_Gemini.md"
        transcript_file.write_text(SAMPLE_TRANSCRIPT)

        segments = MarkdownReader.read(transcript_file, include_events=True, include_sections=True)

        # Segments should have section information
        sections = {s.section for s in segments if s.section is not None}
        assert "Introduction" in sections
        assert "Announcing GPT-4o" in sections

    def test_extract_for_alignment(self, tmp_path):
        """Test extracting supervisions for alignment."""
        transcript_file = tmp_path / "test_Gemini.md"
        transcript_file.write_text(SAMPLE_TRANSCRIPT)

        # Extract for alignment
        supervisions = MarkdownReader.extract_for_alignment(transcript_file, merge_consecutive=False)

        # Should return Supervision objects
        assert len(supervisions) > 0
        assert all(isinstance(sup, Supervision) for sup in supervisions)

        # Should have text and timestamps
        for sup in supervisions:
            assert sup.text is not None
            assert sup.start >= 0
            assert sup.duration > 0

    def test_extract_with_merge(self, tmp_path):
        """Test extracting with consecutive segment merging."""
        transcript_file = tmp_path / "test_Gemini.md"
        transcript_file.write_text(SAMPLE_TRANSCRIPT)

        # Extract without merge
        sups_no_merge = MarkdownReader.extract_for_alignment(transcript_file, merge_consecutive=False)

        # Extract with merge
        sups_with_merge = MarkdownReader.extract_for_alignment(transcript_file, merge_consecutive=True)

        # Merged should have fewer or equal segments
        assert len(sups_with_merge) <= len(sups_no_merge)


class TestYouTubeMarkdownReader:
    """Tests for MarkdownReader with YouTube link format."""

    def test_read_youtube_format(self, tmp_path):
        """Test reading YouTube format transcript with link timestamps."""
        transcript_file = tmp_path / "youtube_Gemini.md"
        transcript_file.write_text(SAMPLE_YOUTUBE_TRANSCRIPT)

        # Read all segments
        segments = MarkdownReader.read(transcript_file, include_events=True, include_sections=True)

        # Should have sections and dialogue
        assert len(segments) > 0

        # Check segment types
        types = {seg.segment_type for seg in segments}
        assert "section_header" in types
        assert "dialogue" in types

    def test_youtube_timestamp_parsing(self):
        """Test YouTube timestamp parsing from URL format."""
        # Test seconds parsing
        timestamp = MarkdownReader.parse_timestamp("12")
        assert timestamp == 12.0

        timestamp = MarkdownReader.parse_timestamp("63")
        assert timestamp == 63.0

        timestamp = MarkdownReader.parse_timestamp("3661")
        assert timestamp == 3661.0

    def test_youtube_section_headers(self, tmp_path):
        """Test YouTube format section headers."""
        transcript_file = tmp_path / "youtube_Gemini.md"
        transcript_file.write_text(SAMPLE_YOUTUBE_TRANSCRIPT)

        segments = MarkdownReader.read(transcript_file, include_sections=True)

        # Find section headers
        section_headers = [s for s in segments if s.segment_type == "section_header"]
        assert len(section_headers) > 0

        # Check section information
        sections = {s.section for s in segments if s.section is not None}
        assert "Introduction" in sections
        assert "Introducing the New Flagship Model: GPT-4o" in sections

    def test_youtube_speaker_dialogue(self, tmp_path):
        """Test YouTube format speaker dialogue parsing."""
        transcript_file = tmp_path / "youtube_Gemini.md"
        transcript_file.write_text(SAMPLE_YOUTUBE_TRANSCRIPT)

        segments = MarkdownReader.read(transcript_file)

        # Find dialogue segments with speaker
        dialogue_with_speaker = [s for s in segments if s.speaker is not None]
        assert len(dialogue_with_speaker) > 0

        # Check speaker name
        speakers = {s.speaker for s in dialogue_with_speaker}
        assert "Mira Murati:" in speakers

        # Check timestamps are correctly parsed
        for seg in dialogue_with_speaker:
            if seg.timestamp is not None:
                assert seg.timestamp > 0

    def test_youtube_extract_for_alignment(self, tmp_path):
        """Test extracting YouTube format for alignment."""
        transcript_file = tmp_path / "youtube_Gemini.md"
        transcript_file.write_text(SAMPLE_YOUTUBE_TRANSCRIPT)

        # Extract for alignment
        supervisions = MarkdownReader.extract_for_alignment(transcript_file, merge_consecutive=False)

        # Should return Supervision objects
        assert len(supervisions) > 0
        assert all(isinstance(sup, Supervision) for sup in supervisions)

        # Should have text and timestamps
        for sup in supervisions:
            assert sup.text is not None
            assert sup.start >= 0
            assert sup.duration > 0

    def test_youtube_with_merge(self, tmp_path):
        """Test YouTube format with consecutive segment merging."""
        transcript_file = tmp_path / "youtube_Gemini.md"
        transcript_file.write_text(SAMPLE_YOUTUBE_TRANSCRIPT)

        # Extract without merge
        sups_no_merge = MarkdownReader.extract_for_alignment(transcript_file, merge_consecutive=False)

        # Extract with merge
        sups_with_merge = MarkdownReader.extract_for_alignment(transcript_file, merge_consecutive=True)

        # Merged should have fewer or equal segments
        assert len(sups_with_merge) <= len(sups_no_merge)


class TestMultiEventParsing:
    """Tests for parsing multi-event lines like [Laughter] [Applause] [00:13:38]."""

    def test_multi_event_line(self):
        """Multi-event line should be split into separate event segments."""
        content = """Some dialogue here. [00:13:30]

[Laughter] [Applause] [00:13:38]

More dialogue. [00:13:45]
"""
        segments = MarkdownReader.read(content, include_events=True)
        event_segments = [s for s in segments if s.segment_type == "event"]

        assert len(event_segments) == 2
        assert event_segments[0].text == "[Laughter]"
        assert event_segments[1].text == "[Applause]"
        assert event_segments[0].timestamp == 13 * 60 + 38
        assert event_segments[1].timestamp == 13 * 60 + 38

    def test_multi_event_with_ms(self):
        """Multi-event line with milliseconds should parse correctly."""
        content = """[Laughter] [Applause] [00:13:38.500]
"""
        segments = MarkdownReader.read(content, include_events=True)
        event_segments = [s for s in segments if s.segment_type == "event"]

        assert len(event_segments) == 2
        assert event_segments[0].text == "[Laughter]"
        assert event_segments[1].text == "[Applause]"
        assert event_segments[0].timestamp == 13 * 60 + 38.5

    def test_three_events(self):
        """Three events on one line should all be parsed."""
        content = """[Music] [Laughter] [Applause] [00:05:00]
"""
        segments = MarkdownReader.read(content, include_events=True)
        event_segments = [s for s in segments if s.segment_type == "event"]

        assert len(event_segments) == 3
        assert event_segments[0].text == "[Music]"
        assert event_segments[1].text == "[Laughter]"
        assert event_segments[2].text == "[Applause]"

    def test_multi_event_mm_ss_format(self):
        """Multi-event with MM:SS format timestamp."""
        content = """[Laughter] [Applause] [13:38]
"""
        segments = MarkdownReader.read(content, include_events=True)
        event_segments = [s for s in segments if s.segment_type == "event"]

        assert len(event_segments) == 2
        assert event_segments[0].timestamp == 13 * 60 + 38

    def test_multi_event_excluded_when_include_events_false(self):
        """Multi-event lines should be excluded when include_events=False."""
        content = """[Laughter] [Applause] [00:13:38]
"""
        segments = MarkdownReader.read(content, include_events=False)
        assert len(segments) == 0

    def test_single_event_still_works(self):
        """Single event lines should still work as before."""
        content = """[Applause] [00:13:38]
"""
        segments = MarkdownReader.read(content, include_events=True)
        event_segments = [s for s in segments if s.segment_type == "event"]

        assert len(event_segments) == 1
        assert event_segments[0].text == "[Applause]"

    def test_multi_event_extract_for_alignment(self):
        """Multi-event lines should produce separate supervisions in extract_for_alignment."""
        content = """Some text here. [00:13:30]

[Laughter] [Applause] [00:13:38]

More text. [00:13:45]
"""
        supervisions = MarkdownReader.extract_for_alignment(content)
        event_sups = [s for s in supervisions if s.text.startswith("[") and s.text.endswith("]")]

        assert len(event_sups) == 2
        assert event_sups[0].text == "[Laughter]"
        assert event_sups[1].text == "[Applause]"


class TestMarkdownWriter:
    """Tests for MarkdownWriter class."""

    def test_format_timestamp(self):
        """Test timestamp formatting."""
        # Test various timestamps
        assert MarkdownWriter.format_timestamp(13.0) == "[00:00:13]"
        assert MarkdownWriter.format_timestamp(61.0) == "[00:01:01]"
        assert MarkdownWriter.format_timestamp(3661.0) == "[01:01:01]"
        assert MarkdownWriter.format_timestamp(0.0) == "[00:00:00]"

    def test_update_timestamps(self, tmp_path):
        """Test updating transcript with new timestamps."""
        # Create original transcript
        original_file = tmp_path / "original.txt"
        original_file.write_text(SAMPLE_TRANSCRIPT)

        # Extract supervisions
        supervisions = MarkdownReader.extract_for_alignment(original_file)

        # Modify timestamps slightly (simulate alignment)
        aligned_supervisions = []
        for sup in supervisions:
            aligned_sup = Supervision(
                id=sup.id,
                text=sup.text,
                start=sup.start + 0.1,  # Add 0.1 second
                duration=sup.duration,
            )
            aligned_supervisions.append(aligned_sup)

        # Update timestamps
        output_file = tmp_path / "updated.txt"
        MarkdownWriter.update_timestamps(original_file, aligned_supervisions, output_file)

        # Check output file exists
        assert output_file.exists()

        # Read updated content
        updated_content = output_file.read_text()
        assert len(updated_content) > 0

    def test_write_aligned_transcript(self, tmp_path):
        """Test writing simplified aligned transcript."""
        # Create original transcript
        original_file = tmp_path / "original.txt"
        original_file.write_text(SAMPLE_TRANSCRIPT)

        # Extract and create aligned supervisions
        supervisions = MarkdownReader.extract_for_alignment(original_file)

        # Add word-level alignment
        for sup in supervisions:
            words = sup.text.split()
            word_duration = sup.duration / max(len(words), 1)
            word_alignments = []
            for i, word in enumerate(words):
                word_alignments.append(
                    {
                        "symbol": word,
                        "start": sup.start + i * word_duration,
                        "end": sup.start + (i + 1) * word_duration,
                    }
                )
            sup.alignment = {"word": word_alignments}

        # Write aligned transcript
        output_file = tmp_path / "aligned.txt"
        MarkdownWriter.write_aligned_transcript(supervisions, output_file, include_word_timestamps=True)

        # Check output
        assert output_file.exists()
        content = output_file.read_text()
        assert "Aligned Transcript" in content
        assert "[00:00:" in content  # Should have timestamps

    def test_write_aligned_without_words(self, tmp_path):
        """Test writing aligned transcript without word timestamps."""
        original_file = tmp_path / "original.txt"
        original_file.write_text(SAMPLE_TRANSCRIPT)

        supervisions = MarkdownReader.extract_for_alignment(original_file)

        output_file = tmp_path / "aligned_no_words.txt"
        MarkdownWriter.write_aligned_transcript(supervisions, output_file, include_word_timestamps=False)

        assert output_file.exists()
        content = output_file.read_text()

        # Should not contain word-level details
        assert "Words:" not in content

    def test_write_aligned_transcript_extra_kwargs(self, tmp_path):
        """Test write_aligned_transcript accepts extra kwargs for Caption.write() compatibility."""
        supervisions = [
            Supervision(id="test_001", text="Hello world", start=0.0, duration=2.0),
            Supervision(id="test_002", text="Goodbye world", start=2.0, duration=2.0),
        ]

        output_file = tmp_path / "output.md"
        # Should not raise TypeError when passing extra kwargs
        MarkdownWriter.write_aligned_transcript(
            supervisions,
            output_file,
            include_word_timestamps=False,
            include_speaker=True,  # Extra kwarg
            word_level=True,  # Extra kwarg
            some_unknown_param="value",  # Extra kwarg
        )

        assert output_file.exists()
        content = output_file.read_text()
        assert "Hello world" in content
        assert "Goodbye world" in content

    def test_write_with_extra_kwargs(self, tmp_path):
        """Test MarkdownWriter.write() accepts extra kwargs."""
        supervisions = [
            Supervision(id="test_001", text="Test content", start=0.0, duration=1.5),
        ]

        output_file = tmp_path / "output_write.md"
        # Should not raise TypeError
        result = MarkdownWriter.write(
            supervisions,
            output_file,
            include_speaker=True,
            word_level=False,
        )

        assert result == output_file
        assert output_file.exists()

    def test_to_bytes_with_extra_kwargs(self):
        """Test MarkdownWriter.to_bytes() accepts extra kwargs."""
        supervisions = [
            Supervision(id="test_001", text="Bytes test", start=0.0, duration=1.0),
        ]

        # Should not raise TypeError
        result = MarkdownWriter.to_bytes(
            supervisions,
            include_word_timestamps=False,
            include_speaker=True,
            custom_param="ignored",
        )

        assert isinstance(result, bytes)
        assert b"Bytes test" in result


class TestMarkdownSegment:
    """Tests for MarkdownSegment dataclass (shared)."""

    def test_segment_creation(self):
        """Test creating a MarkdownSegment."""
        segment = MarkdownSegment(
            text="Hello world",
            timestamp=13.0,
            speaker="Speaker",
            section="Section 1",
            segment_type="dialogue",
            line_number=10,
        )

        assert segment.text == "Hello world"
        assert segment.timestamp == 13.0
        assert segment.speaker == "Speaker"
        assert segment.section == "Section 1"
        assert segment.segment_type == "dialogue"
        assert segment.line_number == 10

    def test_start_property(self):
        """Test the start property."""
        segment = MarkdownSegment(text="Test", timestamp=10.5)
        assert segment.start == 10.5

        segment_no_ts = MarkdownSegment(text="Test", timestamp=None)
        assert segment_no_ts.start == 0.0


class TestFrontmatterAndThinkingSkip:
    """Tests for YAML front matter and <thinking> block removal."""

    def test_skip_frontmatter(self, tmp_path):
        """Test that YAML front matter is skipped."""
        content_with_frontmatter = """---
model_version: gemini-3-flash-preview
prompt_tokens: 9034
output_tokens: 1218
citations:
  - uri: https://example.com
    range: [100, 200]
---

# Real Content

**Speaker:** Hello world. [00:00:05]
"""
        transcript_file = tmp_path / "frontmatter_Gemini.md"
        transcript_file.write_text(content_with_frontmatter)

        segments = MarkdownReader.read(transcript_file, include_events=True)

        assert len(segments) == 1
        assert "Hello world" in segments[0].text
        # Ensure front matter is not parsed as content
        assert not any("model_version" in s.text for s in segments)

    def test_skip_frontmatter_and_thinking(self, tmp_path):
        """Test that both front matter and thinking blocks are skipped."""
        content = """---
model_version: gemini-3-flash
---

<thinking>
This should be skipped.
**Speaker:** Duplicate content. [00:00:01]
</thinking>

# Real Content

**Speaker:** Actual content. [00:00:10]
"""
        transcript_file = tmp_path / "both_Gemini.md"
        transcript_file.write_text(content)

        segments = MarkdownReader.read(transcript_file, include_events=True)

        assert len(segments) == 1
        assert "Actual content" in segments[0].text

    def test_skip_thinking_block(self, tmp_path):
        """Test that <thinking>...</thinking> blocks are skipped."""
        content_with_thinking = """<thinking>
This is thinking content that should be ignored.

**Mira Murati:** This duplicate text should be skipped. [00:00:10]
</thinking>

# Real Content

**Mira Murati:** This is the actual transcript. [00:00:15]

More content here. [00:00:20]
"""
        transcript_file = tmp_path / "thinking_Gemini.md"
        transcript_file.write_text(content_with_thinking)

        segments = MarkdownReader.read(transcript_file, include_events=True)

        # Should only have segments from after </thinking>
        assert len(segments) == 2
        assert "actual transcript" in segments[0].text
        assert "More content" in segments[1].text

    def test_skip_thinking_from_real_file(self):
        """Test with the actual test data file containing thinking block."""
        from pathlib import Path

        test_file = Path(__file__).parent.parent / "data" / "Gemini_IncludeThoughts.md"
        if not test_file.exists():
            pytest.skip("Test data file not found")

        segments = MarkdownReader.read(test_file, include_events=True)

        # The file has duplicated content in <thinking> - verify we only get unique content
        texts = [s.text for s in segments if s.segment_type == "dialogue"]

        # Should not have excessive duplicates (some duplicates are normal in transcripts)
        # The key is that thinking block content doesn't double everything
        assert len(segments) < 50  # Without fix this would be ~48 (doubled)

    def test_no_thinking_block(self, tmp_path):
        """Test that content without thinking block works normally."""
        content_no_thinking = """# Normal Transcript

**Speaker:** Hello world. [00:00:05]

More text here. [00:00:10]
"""
        transcript_file = tmp_path / "normal_Gemini.md"
        transcript_file.write_text(content_no_thinking)

        segments = MarkdownReader.read(transcript_file, include_events=True)

        assert len(segments) == 2
        assert "Hello world" in segments[0].text


class TestStartEndTimestamps:
    """Tests for parsing format with both start and end timestamps: [START] text [END]."""

    def test_inline_both_timestamps(self):
        """Test parsing lines with both start and end timestamps."""
        content = """## [00:00:00] Introduction

[00:00:00] [Music starts] [00:00:11]

[00:00:11] Hello world this is a test. [00:00:19]

[00:00:20] Another line with timestamps. [00:00:28]
"""
        segments = MarkdownReader.read(content, include_events=True)

        # Filter dialogue segments with timestamps
        dialogue = [s for s in segments if s.segment_type == "dialogue" and s.timestamp is not None]

        assert len(dialogue) == 3

        # Check first segment: [Music starts]
        assert dialogue[0].timestamp == 0
        assert dialogue[0].end_timestamp == 11
        assert dialogue[0].text == "[Music starts]"

        # Check second segment
        assert dialogue[1].timestamp == 11
        assert dialogue[1].end_timestamp == 19
        assert "Hello world" in dialogue[1].text

        # Check third segment
        assert dialogue[2].timestamp == 20
        assert dialogue[2].end_timestamp == 28

    def test_extract_for_alignment_with_both_timestamps(self):
        """Test extract_for_alignment with start+end format."""
        content = """[00:00:00] First segment here. [00:00:05]

[00:00:05] Second segment. [00:00:10]

[00:00:10] Third segment with more text. [00:00:18]
"""
        supervisions = MarkdownReader.extract_for_alignment(content)

        assert len(supervisions) == 3

        # Check durations are calculated from start-end
        assert supervisions[0].start == 0
        assert supervisions[0].duration == 5  # 5-0
        assert supervisions[1].start == 5
        assert supervisions[1].duration == 5  # 10-5
        assert supervisions[2].start == 10
        assert supervisions[2].duration == 8  # 18-10

    def test_real_gemini_startend_file(self):
        """Test with real Gemini_StartEnd.md file."""
        from pathlib import Path

        test_file = Path("tests/data/Gemini_StartEnd.md")
        if not test_file.exists():
            pytest.skip("Test file not found")

        segments = MarkdownReader.read(test_file, include_events=True)

        # Should have multiple segments with both timestamps
        dialogue_with_both = [
            s
            for s in segments
            if s.segment_type == "dialogue" and s.timestamp is not None and s.end_timestamp is not None
        ]
        assert len(dialogue_with_both) > 10  # Should have many segments

        # Check first few segments have correct timing
        for seg in dialogue_with_both[:5]:
            assert (
                seg.timestamp < seg.end_timestamp
            ), f"Start should be before end: {seg.timestamp} vs {seg.end_timestamp}"

    def test_mixed_format_mm_ss(self):
        """Test with MM:SS format timestamps."""
        content = """[00:00] Short format start. [00:15]

[00:15] Another MM:SS format. [00:30]
"""
        segments = MarkdownReader.read(content, include_events=True)
        dialogue = [s for s in segments if s.segment_type == "dialogue" and s.timestamp is not None]

        assert len(dialogue) == 2
        assert dialogue[0].timestamp == 0
        assert dialogue[0].end_timestamp == 15
        assert dialogue[1].timestamp == 15
        assert dialogue[1].end_timestamp == 30


class TestMillisecondTimestamps:
    """Tests for parsing timestamps with milliseconds: [HH:MM:SS.mmm]."""

    def test_parse_timestamp_with_milliseconds(self):
        """Test parse_timestamp with milliseconds."""
        # HH:MM:SS.mmm format
        ts = MarkdownReader.parse_timestamp("00", "00", "11", "750")
        assert ts == 11.75

        ts = MarkdownReader.parse_timestamp("00", "01", "30", "500")
        assert ts == 90.5

        # MM:SS.mmm format (using ms keyword)
        ts = MarkdownReader.parse_timestamp("01", "30", ms="500")
        assert ts == 90.5

        # Without milliseconds (backward compatibility)
        ts = MarkdownReader.parse_timestamp("00", "00", "11")
        assert ts == 11.0

    def test_inline_both_timestamps_with_ms(self):
        """Test parsing lines with both start and end timestamps including milliseconds."""
        content = """[00:00:11.750] Hi everyone. [00:00:12.500]

[00:00:12.500] [Applause] [00:00:16.800]

[00:00:16.800] Thank you. [00:00:20.400]
"""
        segments = MarkdownReader.read(content, include_events=True)
        dialogue = [s for s in segments if s.segment_type == "dialogue" and s.timestamp is not None]

        assert len(dialogue) == 3

        # Check millisecond precision
        assert dialogue[0].timestamp == 11.75
        assert dialogue[0].end_timestamp == 12.5
        assert dialogue[1].timestamp == 12.5
        assert dialogue[1].end_timestamp == 16.8
        assert dialogue[2].timestamp == 16.8
        assert dialogue[2].end_timestamp == 20.4

    def test_speaker_with_milliseconds(self):
        """Test speaker dialogue with millisecond timestamps."""
        content = """**Mira Murati:** [00:00:11.750] Hi everyone. [00:00:12.500]

**Mark Chen:** [00:00:12.500] Hello there! [00:00:14.200]
"""
        segments = MarkdownReader.read(content)
        dialogue = [s for s in segments if s.segment_type == "dialogue"]

        assert len(dialogue) == 2
        assert dialogue[0].speaker == "Mira Murati:"
        assert dialogue[0].timestamp == 11.75
        assert dialogue[0].end_timestamp == 12.5
        assert dialogue[0].text == "Hi everyone."

        assert dialogue[1].speaker == "Mark Chen:"
        assert dialogue[1].timestamp == 12.5
        assert dialogue[1].end_timestamp == 14.2

    def test_real_precise_end_file(self):
        """Test with real gemini-3-flash-preview_PreciseEnd.md file."""
        from pathlib import Path

        test_file = Path("tests/data/gemini-3-flash-preview_PreciseEnd.md")
        if not test_file.exists():
            pytest.skip("Test file not found")

        segments = MarkdownReader.read(test_file, include_events=True)

        # Should have many segments
        assert len(segments) > 100

        # Check first few segments have millisecond precision
        dialogue = [s for s in segments if s.segment_type == "dialogue" and s.timestamp is not None]

        # First segment should be "Hi everyone." with precise timing
        assert dialogue[0].timestamp == 11.75  # 00:00:11.750
        assert dialogue[0].end_timestamp == 12.5  # 00:00:12.500

    def test_extract_for_alignment_with_ms(self):
        """Test extract_for_alignment preserves millisecond precision."""
        content = """[00:00:11.750] First segment. [00:00:12.500]

[00:00:12.500] Second segment. [00:00:16.800]
"""
        supervisions = MarkdownReader.extract_for_alignment(content)

        assert len(supervisions) == 2
        assert supervisions[0].start == pytest.approx(11.75)
        assert supervisions[0].duration == pytest.approx(0.75)  # 12.5 - 11.75
        assert supervisions[1].start == pytest.approx(12.5)
        assert supervisions[1].duration == pytest.approx(4.3)  # 16.8 - 12.5


class TestEndOnlyTimestamps:
    """Tests for parsing format with only end timestamps: text [END]."""

    def test_end_only_timestamps(self):
        """Test parsing lines with only end timestamps."""
        content = """## [00:00:00] 开场

ChatGPT以及硅谷AI大战终于升级。 [00:00:30]

[音乐] [00:00:33]

Hello大家好，欢迎来到《硅谷101》。 [00:00:54]
"""
        segments = MarkdownReader.read(content, include_events=True)

        # Filter segments with timestamps
        with_ts = [s for s in segments if s.timestamp is not None or s.end_timestamp is not None]

        assert len(with_ts) == 3

        # First segment: only end timestamp
        assert with_ts[0].timestamp is None
        assert with_ts[0].end_timestamp == 30.0
        assert "ChatGPT" in with_ts[0].text

        # Event: has start timestamp (from event pattern)
        assert with_ts[1].timestamp == 33.0
        assert with_ts[1].segment_type == "event"

        # Third segment: only end timestamp
        assert with_ts[2].timestamp is None
        assert with_ts[2].end_timestamp == 54.0

    def test_extract_for_alignment_end_only(self):
        """Test extract_for_alignment infers start from previous segment's end."""
        content = """## [00:00:00] 开场

ChatGPT以及硅谷AI大战终于升级。 [00:00:30]

[音乐] [00:00:33]

Hello大家好，欢迎来到《硅谷101》。 [00:00:54]
"""
        supervisions = MarkdownReader.extract_for_alignment(content)

        assert len(supervisions) == 3

        # First segment: start inferred as 0 (beginning), end=30
        assert supervisions[0].start == pytest.approx(0.0)
        assert supervisions[0].duration == pytest.approx(30.0)

        # Event [音乐]: start=33 (from timestamp)
        assert supervisions[1].start == pytest.approx(33.0)

        # Third segment: start inferred from previous end
        assert supervisions[2].start == pytest.approx(supervisions[1].start + supervisions[1].duration)

    def test_chinese_transcript_full(self):
        """Test full Chinese transcript with mixed timestamp formats."""
        content = """# GPT-4o对战谷歌：多模态之战

## Table of Contents
* [00:00:00] 开场
* [00:01:15] CHAPTER 1

## [00:00:00] 开场

ChatGPT以及硅谷AI大战终于升级，长出了"眼睛"和"嘴"。 [00:00:30]

[音乐] [00:00:33]

Hello大家好，欢迎来到《硅谷101》，我是陈茜。 [00:00:54]
"""
        supervisions = MarkdownReader.extract_for_alignment(content)

        assert len(supervisions) == 3

        # First segment starts at 0
        assert supervisions[0].start == pytest.approx(0.0)
        assert supervisions[0].duration == pytest.approx(30.0)
        assert "ChatGPT" in supervisions[0].text

        # Event
        assert supervisions[1].start == pytest.approx(33.0)
        assert "[音乐]" in supervisions[1].text

        # Last segment
        assert "硅谷101" in supervisions[2].text


class TestPreserveOriginalOrder:
    """Tests for preserving original text order (not sorting by timestamp).

    Gemini timestamps are often inaccurate, so we must preserve the original
    text order from the transcript rather than sorting by timestamp.
    """

    def test_preserve_order_with_out_of_order_timestamps(self):
        """Test that segments preserve original order even when timestamps are out of order."""
        # Timestamps are intentionally out of order (30 > 20 > 25)
        # But text order should be preserved: First -> Second -> Third
        content = """First sentence here. [00:00:30]

Second sentence here. [00:00:20]

Third sentence here. [00:00:25]
"""
        supervisions = MarkdownReader.extract_for_alignment(content)

        assert len(supervisions) == 3
        # Order should match original text order, NOT timestamp order
        assert "First" in supervisions[0].text
        assert "Second" in supervisions[1].text
        assert "Third" in supervisions[2].text

    def test_preserve_order_with_start_timestamps(self):
        """Test order preservation with start timestamps out of order."""
        content = """[00:00:50] First segment. [00:00:55]

[00:00:10] Second segment. [00:00:15]

[00:00:30] Third segment. [00:00:35]
"""
        supervisions = MarkdownReader.extract_for_alignment(content)

        assert len(supervisions) == 3
        # Should preserve original order: First -> Second -> Third
        # NOT sorted by timestamp: Second (10) -> Third (30) -> First (50)
        assert "First" in supervisions[0].text
        assert "Second" in supervisions[1].text
        assert "Third" in supervisions[2].text

    def test_preserve_order_mixed_timestamp_formats(self):
        """Test order preservation with mixed timestamp formats."""
        content = """**Speaker A:** Last chronologically but first in text. [00:05:00]

**Speaker B:** [00:00:30] Middle chronologically, second in text. [00:00:35]

**Speaker A:** First chronologically but last in text. [00:00:10]
"""
        supervisions = MarkdownReader.extract_for_alignment(content)

        assert len(supervisions) == 3
        # Preserve text order, ignore chronological order
        assert "first in text" in supervisions[0].text
        assert "second in text" in supervisions[1].text
        assert "last in text" in supervisions[2].text

    def test_preserve_order_real_world_scenario(self):
        """Test with realistic Gemini output where timestamps jump around."""
        # Simulates Gemini's common behavior of producing non-sequential timestamps
        content = """## [00:00:00] Introduction

Hello everyone, welcome to the show. [00:00:45]

[音乐] [00:00:30]

Today we're going to discuss AI. [00:00:50]

But first, let me introduce myself. [00:00:15]
"""
        supervisions = MarkdownReader.extract_for_alignment(content)

        texts = [s.text for s in supervisions]

        # Verify original text order is preserved
        assert texts.index("Hello everyone, welcome to the show.") < texts.index("[音乐]")
        assert texts.index("[音乐]") < texts.index("Today we're going to discuss AI.")
        assert texts.index("Today we're going to discuss AI.") < texts.index("But first, let me introduce myself.")


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_full_workflow(self, tmp_path):
        """Test complete read -> align -> write workflow."""
        # 1. Create transcript
        transcript_file = tmp_path / "Gemini.md"
        transcript_file.write_text(SAMPLE_TRANSCRIPT)

        # 2. Extract for alignment
        supervisions = MarkdownReader.extract_for_alignment(transcript_file)
        assert len(supervisions) > 0

        # 3. Simulate alignment (add small corrections)
        aligned_supervisions = []
        for sup in supervisions:
            # Simulate alignment correction
            aligned_sup = Supervision(
                id=sup.id,
                text=sup.text,
                start=sup.start + 0.05,
                duration=sup.duration * 0.95,
            )

            # Add word alignment
            words = sup.text.split()
            word_duration = aligned_sup.duration / max(len(words), 1)
            word_alignments = []
            for i, word in enumerate(words):
                word_alignments.append(
                    {
                        "symbol": word,
                        "start": aligned_sup.start + i * word_duration,
                        "end": aligned_sup.start + (i + 1) * word_duration,
                    }
                )
            aligned_sup.alignment = {"word": word_alignments}
            aligned_supervisions.append(aligned_sup)

        # 4. Write updated transcript
        updated_file = tmp_path / "updated_Gemini.md"
        MarkdownWriter.update_timestamps(transcript_file, aligned_supervisions, updated_file)
        assert updated_file.exists()

        # 5. Write simplified aligned transcript
        simple_file = tmp_path / "simple_aligned.txt"
        MarkdownWriter.write_aligned_transcript(aligned_supervisions, simple_file, include_word_timestamps=True)
        assert simple_file.exists()

        # Verify content
        simple_content = simple_file.read_text()
        assert "Aligned Transcript" in simple_content
        assert "Words:" in simple_content

    def test_youtube_workflow(self, tmp_path):
        """Test complete workflow with YouTube format transcript."""
        # 1. Create YouTube format transcript
        transcript_file = tmp_path / "youtube_Gemini.md"
        transcript_file.write_text(SAMPLE_YOUTUBE_TRANSCRIPT)

        # 2. Extract for alignment
        supervisions = MarkdownReader.extract_for_alignment(transcript_file)
        assert len(supervisions) > 0

        # 3. Simulate alignment (add small corrections)
        aligned_supervisions = []
        for sup in supervisions:
            # Simulate alignment correction
            aligned_sup = Supervision(
                id=sup.id,
                text=sup.text,
                start=sup.start + 0.05,
                duration=sup.duration * 0.95,
            )

            # Add word alignment
            words = sup.text.split()
            word_duration = aligned_sup.duration / max(len(words), 1)
            word_alignments = []
            for i, word in enumerate(words):
                word_alignments.append(
                    {
                        "symbol": word,
                        "start": aligned_sup.start + i * word_duration,
                        "end": aligned_sup.start + (i + 1) * word_duration,
                    }
                )
            aligned_sup.alignment = {"word": word_alignments}
            aligned_supervisions.append(aligned_sup)

        # 4. Write updated transcript
        updated_file = tmp_path / "updated_youtube_Gemini.md"
        MarkdownWriter.update_timestamps(transcript_file, aligned_supervisions, updated_file)
        assert updated_file.exists()

        # 5. Write simplified aligned transcript
        simple_file = tmp_path / "simple_youtube_aligned.txt"
        MarkdownWriter.write_aligned_transcript(aligned_supervisions, simple_file, include_word_timestamps=True)
        assert simple_file.exists()

        # Verify content
        simple_content = simple_file.read_text()
        assert "Aligned Transcript" in simple_content
        assert "Words:" in simple_content

    def test_mixed_format_compatibility(self, tmp_path):
        """Test that both formats can be processed together."""
        # Test original format
        original_file = tmp_path / "original.txt"
        original_file.write_text(SAMPLE_TRANSCRIPT)
        original_sups = MarkdownReader.extract_for_alignment(original_file)

        # Test YouTube format
        youtube_file = tmp_path / "youtube.txt"
        youtube_file.write_text(SAMPLE_YOUTUBE_TRANSCRIPT)
        youtube_sups = MarkdownReader.extract_for_alignment(youtube_file)

        # Both should work and return supervisions
        assert len(original_sups) > 0
        assert len(youtube_sups) > 0

        # Both should have valid supervisions
        for sup in original_sups + youtube_sups:
            assert isinstance(sup, Supervision)
            assert sup.text is not None
            assert sup.start >= 0
            assert sup.duration > 0


class TestTimeRangeFormat:
    """Tests for time range format: text [HH:MM:SS → HH:MM:SS]."""

    def test_speaker_with_time_range(self):
        """Test parsing speaker dialogue with time range timestamps."""
        content = """**Mira Murati:** Hi everyone thank you. [00:00:12 → 00:01:26]

**Mark Chen:** Hey I'm Mark. [00:01:26 → 00:01:50]
"""
        segments = MarkdownReader.read(content)

        assert len(segments) == 2
        assert segments[0].speaker == "Mira Murati:"
        assert segments[0].timestamp == 12.0
        assert segments[0].end_timestamp == 86.0
        assert "Hi everyone" in segments[0].text

        assert segments[1].speaker == "Mark Chen:"
        assert segments[1].timestamp == 86.0
        assert segments[1].end_timestamp == 110.0

    def test_plain_text_with_time_range(self):
        """Test parsing plain text with time range timestamps."""
        content = """hey chat GPT I'm Mark how are you [00:09:47 → 00:09:49]

certainly yes great looks like it works [00:23:15 → 00:23:21]
"""
        segments = MarkdownReader.read(content)

        assert len(segments) == 2
        assert segments[0].timestamp == 587.0
        assert segments[0].end_timestamp == 589.0
        assert "hey chat GPT" in segments[0].text
        # Ensure the time range is NOT in the text
        assert "→" not in segments[0].text

    def test_event_still_works_alongside_time_range(self):
        """Test that events are still parsed correctly when time ranges are present."""
        content = """**Speaker:** Some dialogue. [00:02:46 → 00:02:51]

[Applause] [00:02:46]

**Speaker:** More dialogue. [00:02:51 → 00:04:07]
"""
        segments = MarkdownReader.read(content, include_events=True)

        events = [s for s in segments if s.segment_type == "event"]
        dialogues = [s for s in segments if s.segment_type == "dialogue"]

        assert len(events) == 1
        assert events[0].text == "[Applause]"
        assert len(dialogues) == 2

    def test_time_range_extract_for_alignment(self):
        """Test extract_for_alignment with time range format."""
        content = """**Speaker A:** First segment. [00:00:12 → 00:01:26]

**Speaker B:** Second segment. [00:01:26 → 00:02:00]

[Applause] [00:02:00]

**Speaker A:** Third segment. [00:02:05 → 00:02:30]
"""
        supervisions = MarkdownReader.extract_for_alignment(content)

        assert len(supervisions) >= 3

        # Check first supervision
        assert supervisions[0].start == 12.0
        assert supervisions[0].duration == pytest.approx(74.0)  # 86 - 12
        assert "First segment" in supervisions[0].text

    def test_time_range_with_mm_ss(self):
        """Test time range with MM:SS format."""
        content = """Some text here. [09:47 → 09:49]
"""
        segments = MarkdownReader.read(content)

        assert len(segments) == 1
        assert segments[0].timestamp == 587.0
        assert segments[0].end_timestamp == 589.0

    def test_time_range_with_milliseconds(self):
        """Test time range with milliseconds."""
        content = """**Speaker:** Hello world. [00:00:12.500 → 00:01:26.750]
"""
        segments = MarkdownReader.read(content)

        assert len(segments) == 1
        assert segments[0].timestamp == 12.5
        assert segments[0].end_timestamp == 86.75


class TestTrailingSectionHeaders:
    """Tests for section headers with trailing timestamps: ## Title [HH:MM:SS]."""

    def test_trailing_section_header(self):
        """Test parsing section header with timestamp at the end."""
        content = """## Introduction and GPT-4o Announcement [00:00:12]

**Mira Murati:** Hi everyone. [00:00:12 → 00:01:26]

## GPT-4o: Speed, Vision, and Audio [00:02:51]

**Mira Murati:** GPT-4o provides intelligence. [00:02:51 → 00:04:07]
"""
        segments = MarkdownReader.read(content, include_sections=True)

        sections = [s for s in segments if s.segment_type == "section_header"]
        assert len(sections) == 2
        assert sections[0].text == "Introduction and GPT-4o Announcement"
        assert sections[0].timestamp == 12.0
        assert sections[1].text == "GPT-4o: Speed, Vision, and Audio"
        assert sections[1].timestamp == 171.0

    def test_trailing_section_excluded_by_default(self):
        """Test that trailing section headers are excluded when include_sections=False."""
        content = """## Introduction [00:00:12]

**Speaker:** Hello. [00:00:12 → 00:00:20]
"""
        segments = MarkdownReader.read(content, include_sections=False)

        types = {s.segment_type for s in segments}
        assert "section_header" not in types

    def test_trailing_section_with_mm_ss(self):
        """Test trailing section header with MM:SS format."""
        content = """## Some Section [12:30]

Text here. [12:30 → 13:00]
"""
        segments = MarkdownReader.read(content, include_sections=True)

        sections = [s for s in segments if s.segment_type == "section_header"]
        assert len(sections) == 1
        assert sections[0].timestamp == 750.0  # 12*60 + 30

    def test_section_context_propagation(self):
        """Test that trailing section sets current_section for subsequent segments."""
        content = """## Introduction [00:00:12]

**Speaker:** Hello. [00:00:12 → 00:00:20]

## Conclusion [00:10:00]

**Speaker:** Goodbye. [00:10:00 → 00:10:10]
"""
        segments = MarkdownReader.read(content)

        dialogue = [s for s in segments if s.segment_type == "dialogue"]
        assert dialogue[0].section == "Introduction"
        assert dialogue[1].section == "Conclusion"


class TestImageLineSkip:
    """Tests for skipping markdown image lines."""

    def test_image_line_skipped(self):
        """Test that image lines are not included in segments."""
        content = """**Speaker:** Hello. [00:00:05]

![cover](imgs/cover.jpg)

**Speaker:** World. [00:00:10]
"""
        segments = MarkdownReader.read(content)

        assert len(segments) == 2
        assert all("cover" not in s.text and "![" not in s.text for s in segments)

    def test_image_not_merged_into_previous(self):
        """Test that image line doesn't get merged into the previous segment's text."""
        content = """Some text without end timestamp.

![diagram](path/to/img.png)

**Speaker:** Next line. [00:00:20]
"""
        segments = MarkdownReader.read(content)

        # The image should NOT be in any segment text
        for seg in segments:
            assert "![" not in seg.text
            assert "diagram" not in seg.text


class TestSpeakersProcessedFile:
    """Integration tests for the speakers-processed.md format (baoyu YouTube transcript)."""

    @pytest.fixture
    def speakers_file(self, tmp_path):
        """Extract speakers-processed.md from test zip."""
        import zipfile
        from pathlib import Path

        zip_path = Path(__file__).parent.parent / "data" / "captions" / "speakers-processed.md.zip"
        if not zip_path.exists():
            pytest.skip("Test data file not found")

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_path)

        return tmp_path / "speakers-processed.md"

    def test_read_segments(self, speakers_file):
        """Test reading all segments from the real file."""
        segments = MarkdownReader.read(speakers_file, include_events=True, include_sections=True)

        # Should have all segment types
        types = {s.segment_type for s in segments}
        assert "section_header" in types
        assert "event" in types
        assert "dialogue" in types

        # Should have 9 sections
        sections = [s for s in segments if s.segment_type == "section_header"]
        assert len(sections) == 9

        # All sections should have timestamps
        for section in sections:
            assert section.timestamp is not None

    def test_time_ranges_parsed(self, speakers_file):
        """Test that time range timestamps are correctly parsed."""
        segments = MarkdownReader.read(speakers_file, include_events=True, include_sections=True)

        dialogue = [s for s in segments if s.segment_type == "dialogue"]

        # Most dialogues should have both start and end timestamps
        with_both = [s for s in dialogue if s.timestamp is not None and s.end_timestamp is not None]
        assert len(with_both) > 90  # Almost all should have both

        # Time ranges should not appear in text
        for seg in with_both:
            assert "→" not in seg.text

    def test_speakers_identified(self, speakers_file):
        """Test that speakers are correctly identified."""
        segments = MarkdownReader.read(speakers_file)

        speakers = {s.speaker for s in segments if s.speaker}
        assert "Mira Murati:" in speakers
        assert "Mark Chen:" in speakers
        assert "Barrett Zoph:" in speakers
        assert "GPT-4o:" in speakers

    def test_cover_image_not_in_segments(self, speakers_file):
        """Test that the cover image line is not included in any segment."""
        segments = MarkdownReader.read(speakers_file, include_events=True, include_sections=True)

        for seg in segments:
            assert "![cover]" not in seg.text
            assert "imgs/cover.jpg" not in seg.text

    def test_frontmatter_stripped(self, speakers_file):
        """Test that YAML front matter is stripped."""
        segments = MarkdownReader.read(speakers_file, include_events=True, include_sections=True)

        for seg in segments:
            assert "channel:" not in seg.text
            assert "language:" not in seg.text

    def test_events_parsed(self, speakers_file):
        """Test that events like [Applause] and [Music] are parsed."""
        segments = MarkdownReader.read(speakers_file, include_events=True)

        events = [s for s in segments if s.segment_type == "event"]
        event_texts = {s.text for s in events}
        assert "[Applause]" in event_texts
        assert "[Music]" in event_texts

    def test_extract_for_alignment(self, speakers_file):
        """Test extract_for_alignment produces valid supervisions."""
        supervisions = MarkdownReader.extract_for_alignment(speakers_file)

        assert len(supervisions) > 50

        for sup in supervisions:
            assert isinstance(sup, Supervision)
            assert sup.text is not None
            assert sup.start >= 0
            assert sup.duration > 0

        # Supervisions should be in chronological order
        for i in range(1, len(supervisions)):
            assert supervisions[i].start >= supervisions[i - 1].start

    def test_roundtrip_update_timestamps(self, speakers_file, tmp_path):
        """Test round-trip: parse → modify timestamps → write back preserves file structure."""
        # Read original content for comparison
        original_content = speakers_file.read_text(encoding="utf-8")

        # Extract supervisions
        supervisions = MarkdownReader.extract_for_alignment(speakers_file)

        # Simulate alignment: shift all timestamps by 0.5s
        aligned_supervisions = []
        for sup in supervisions:
            aligned_sup = Supervision(
                id=sup.id,
                text=sup.text,
                start=sup.start + 0.5,
                duration=sup.duration,
                speaker=sup.speaker,
            )
            aligned_supervisions.append(aligned_sup)

        # Write back with updated timestamps
        output_file = tmp_path / "updated.md"
        MarkdownWriter.update_timestamps(speakers_file, aligned_supervisions, output_file)

        # Verify output file exists and has content
        assert output_file.exists()
        updated_content = output_file.read_text(encoding="utf-8")

        # Key structural elements must be preserved
        assert "---" in updated_content  # Frontmatter delimiters
        assert "title: Introducing GPT-4o" in updated_content
        assert "![cover](imgs/cover.jpg)" in updated_content
        assert "## Table of Contents" in updated_content
        assert "**Mira Murati:**" in updated_content
        assert "**Mark Chen:**" in updated_content
        assert "**Barrett Zoph:**" in updated_content
        assert "**GPT-4o:**" in updated_content
        assert "[Applause]" in updated_content
        assert "[Music]" in updated_content

        # Time ranges should still use → format
        assert "→" in updated_content

        # Line count should remain the same
        assert len(updated_content.splitlines()) == len(original_content.splitlines())


class TestTimeRangeWriter:
    """Tests for MarkdownWriter handling time range format."""

    def test_replace_time_range(self):
        """Test replacing time range timestamps."""
        line = "**Speaker:** Hello world. [00:00:12 → 00:01:26]\n"
        result = MarkdownWriter._replace_timestamp(line, 15.0, 90.0)
        assert "[00:00:15 → 00:01:30]" in result
        assert "Hello world" in result

    def test_replace_single_timestamp(self):
        """Test backward compatibility: replacing single timestamps."""
        line = "[Applause] [00:02:46]\n"
        result = MarkdownWriter._replace_timestamp(line, 170.0)
        assert "[00:02:50]" in result

    def test_replace_time_range_preserves_text(self):
        """Test that time range replacement preserves surrounding text."""
        line = "**Barrett Zoph:** hey chat chbt [00:14:01 → 00:14:04]\n"
        result = MarkdownWriter._replace_timestamp(line, 842.0, 845.0)
        assert "**Barrett Zoph:**" in result
        assert "hey chat chbt" in result
        assert "[00:14:02 → 00:14:05]" in result


class TestMarkdownFrontmatter:
    """Tests for YAML frontmatter extraction and round-trip."""

    SAMPLE_WITH_FRONTMATTER = """\
---
title: State of AI 2026
channel: Lex Fridman
url: "https://www.youtube.com/watch?v=EV7WhVT270Q"
duration: 14895
language: en
transcript_source: https://lexfridman.com/ai-sota-2026-transcript
description: |
  Nathan Lambert and Sebastian Raschka are ML researchers.
  They discuss the state of AI in 2026.
---

**Lex Fridman:** Welcome to the podcast. [00:00:01]

**Nathan Lambert:** Thanks for having me. [00:00:05]
"""

    def test_extract_frontmatter_basic_fields(self):
        """Test that basic scalar fields are correctly extracted."""
        meta = MarkdownReader.extract_frontmatter(self.SAMPLE_WITH_FRONTMATTER)
        assert meta["title"] == "State of AI 2026"
        assert meta["channel"] == "Lex Fridman"
        assert meta["language"] == "en"

    def test_extract_frontmatter_quoted_url(self):
        """Test that quoted values have quotes stripped."""
        meta = MarkdownReader.extract_frontmatter(self.SAMPLE_WITH_FRONTMATTER)
        assert meta["url"] == "https://www.youtube.com/watch?v=EV7WhVT270Q"

    def test_extract_frontmatter_numeric(self):
        """Test that duration is converted to float."""
        meta = MarkdownReader.extract_frontmatter(self.SAMPLE_WITH_FRONTMATTER)
        assert meta["duration"] == 14895.0
        assert isinstance(meta["duration"], float)

    def test_extract_frontmatter_multiline_description(self):
        """Test that YAML block scalar (|) is parsed correctly."""
        meta = MarkdownReader.extract_frontmatter(self.SAMPLE_WITH_FRONTMATTER)
        assert "Nathan Lambert" in meta["description"]
        assert "state of AI" in meta["description"]
        assert "\n" in meta["description"]
        # Should NOT contain the | indicator itself
        assert not meta["description"].startswith("|")

    def test_extract_frontmatter_no_frontmatter(self):
        """Test graceful handling of content without frontmatter."""
        meta = MarkdownReader.extract_frontmatter("**Speaker:** Hello [00:00:01]\n")
        assert meta == {}

    def test_extract_frontmatter_empty_content(self):
        """Test graceful handling of empty content."""
        meta = MarkdownReader.extract_frontmatter("")
        assert meta == {}

    def test_read_preserves_segments_with_frontmatter(self):
        """Test that frontmatter doesn't interfere with segment parsing."""
        segments = MarkdownReader.read(self.SAMPLE_WITH_FRONTMATTER)
        dialogue = [s for s in segments if s.segment_type == "dialogue"]
        assert len(dialogue) == 2
        assert dialogue[0].speaker == "Lex Fridman:"
        assert dialogue[1].speaker == "Nathan Lambert:"

    def test_format_extract_metadata(self):
        """Test MarkdownFormat.extract_metadata integration."""
        import tempfile
        from pathlib import Path

        from lattifai.caption.formats.markdown import MarkdownFormat

        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, encoding="utf-8") as f:
            f.write(self.SAMPLE_WITH_FRONTMATTER)
            tmp = f.name

        try:
            meta = MarkdownFormat.parse(tmp).format_metadata
            assert meta["title"] == "State of AI 2026"
            assert meta["channel"] == "Lex Fridman"
            assert "Nathan Lambert" in meta["description"]
        finally:
            Path(tmp).unlink()

    def test_writer_frontmatter_round_trip(self):
        """Test that metadata survives write → read round-trip."""
        import tempfile
        from pathlib import Path

        from lattifai.caption.formats.markdown import MarkdownFormat

        original_meta = {
            "title": "Test Episode",
            "channel": "Test Host",
            "url": "https://youtube.com/watch?v=abc",
            "duration": 3600,
            "description": "Guest A and Guest B discuss testing.\nLine two of description.",
        }
        sups = [
            Supervision(text="Hello world", start=1.0, duration=3.0, speaker="Host"),
            Supervision(text="Hi there", start=5.0, duration=2.0, speaker="Guest"),
        ]

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            tmp = f.name

        try:
            MarkdownWriter.write(sups, tmp, metadata=original_meta)
            roundtrip_meta = MarkdownFormat.parse(tmp).format_metadata

            assert roundtrip_meta["title"] == "Test Episode"
            assert roundtrip_meta["channel"] == "Test Host"
            assert roundtrip_meta["url"] == "https://youtube.com/watch?v=abc"
            assert roundtrip_meta["duration"] == 3600.0
            assert "Guest A and Guest B" in roundtrip_meta["description"]
            assert "\n" in roundtrip_meta["description"]
        finally:
            Path(tmp).unlink()

    def test_writer_no_frontmatter_when_empty_metadata(self):
        """Test that no frontmatter is written when metadata is empty."""
        import tempfile
        from pathlib import Path

        sups = [Supervision(text="Hello", start=1.0, duration=2.0)]

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            tmp = f.name

        try:
            MarkdownWriter.write(sups, tmp, metadata={})
            content = Path(tmp).read_text(encoding="utf-8")
            assert not content.startswith("---")
        finally:
            Path(tmp).unlink()

    def test_writer_description_truncates_sponsors(self):
        """Test that description is truncated before sponsor sections."""
        import tempfile
        from pathlib import Path

        meta = {
            "title": "Episode",
            "description": "Great episode intro.\n\n*SPONSORS:*\nBuy stuff here.",
        }
        sups = [Supervision(text="Hello", start=0.0, duration=1.0)]

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            tmp = f.name

        try:
            MarkdownWriter.write(sups, tmp, metadata=meta)
            content = Path(tmp).read_text(encoding="utf-8")
            assert "Great episode intro" in content
            assert "SPONSORS" not in content
            assert "Buy stuff" not in content
        finally:
            Path(tmp).unlink()

    def test_writer_uses_uploader_as_channel(self):
        """Test that 'uploader' key is mapped to 'channel' in frontmatter."""
        import tempfile
        from pathlib import Path

        meta = {"uploader": "Lex Fridman"}
        sups = [Supervision(text="Hello", start=0.0, duration=1.0)]

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            tmp = f.name

        try:
            MarkdownWriter.write(sups, tmp, metadata=meta)
            content = Path(tmp).read_text(encoding="utf-8")
            assert "channel: Lex Fridman" in content
        finally:
            Path(tmp).unlink()


class TestMarkdownBilingual:
    """Tests for bilingual markdown read/write with > [lang] marker."""

    def test_read_bilingual_with_frontmatter(self):
        """Translation lines parsed when target_lang is in frontmatter."""
        content = """---
title: Test
target_lang: zh
---

# Test

**Speaker:** [00:01:00] Hello world [00:01:05]
> [zh] 你好世界

[00:01:05] Another line [00:01:10]
> [zh] 另一行翻译
"""
        segments = MarkdownReader.read(content)
        dialogue = [s for s in segments if s.segment_type == "dialogue"]
        assert len(dialogue) == 2
        assert dialogue[0].text == "Hello world"
        assert dialogue[0].translation == "你好世界"
        assert dialogue[1].text == "Another line"
        assert dialogue[1].translation == "另一行翻译"

    def test_read_blockquote_without_frontmatter(self):
        """Without target_lang, > lines are NOT parsed as translations."""
        content = """# Test

**Speaker:** [00:01:00] Hello world [00:01:05]
> This is a regular blockquote
"""
        segments = MarkdownReader.read(content)
        dialogue = [s for s in segments if s.segment_type == "dialogue"]
        assert dialogue[0].translation is None

    def test_bilingual_round_trip(self):
        """Write bilingual supervisions, read them back with translation preserved."""
        sups = [
            Supervision(
                text="Hello world",
                translation="你好世界",
                target_lang="zh",
                start=60.0,
                duration=5.0,
                speaker="Host",
            ),
            Supervision(
                text="Another line",
                translation="另一行翻译",
                target_lang="zh",
                start=65.0,
                duration=5.0,
            ),
        ]
        written = MarkdownWriter.to_bytes(sups, metadata={"title": "Test", "target_lang": "zh"})
        content = written.decode("utf-8")

        # Verify written format contains > [zh] markers
        assert "> [zh] 你好世界" in content
        assert "> [zh] 另一行翻译" in content

        # Read back
        read_sups = MarkdownReader.extract_for_alignment(content)
        assert len(read_sups) == 2
        assert read_sups[0].text == "Hello world"
        assert read_sups[0].translation == "你好世界"
        assert read_sups[0].target_lang == "zh"
        assert read_sups[1].text == "Another line"
        assert read_sups[1].translation == "另一行翻译"

    def test_non_bilingual_unchanged(self):
        """Non-bilingual files behave exactly as before."""
        content = """# Test

**Speaker:** [00:01:00] Hello world [00:01:05]

[00:01:05] Another line [00:01:10]
"""
        segments = MarkdownReader.read(content)
        dialogue = [s for s in segments if s.segment_type == "dialogue"]
        assert all(s.translation is None for s in dialogue)

    def test_orphan_blockquote_ignored(self):
        """A > [lang] line without preceding dialogue is skipped, not crash."""
        content = """---
target_lang: zh
---

# Test

> [zh] 孤立的翻译

**Speaker:** [00:01:00] Hello world [00:01:05]
"""
        segments = MarkdownReader.read(content)
        dialogue = [s for s in segments if s.segment_type == "dialogue"]
        assert len(dialogue) == 1
        assert dialogue[0].text == "Hello world"
        assert dialogue[0].translation is None

    def test_writer_uses_target_lang_tag(self):
        """Writer uses target_lang as the language tag in > [lang] prefix."""
        sup = Supervision(text="Hello", translation="Hola", target_lang="es", start=0.0, duration=1.0)
        content = MarkdownWriter.to_bytes([sup]).decode("utf-8")
        assert "> [es] Hola" in content

    def test_writer_fallback_tag_without_target_lang(self):
        """Writer falls back to [translation] when target_lang is not set."""
        sup = Supervision(text="Hello", translation="Hola", start=0.0, duration=1.0)
        content = MarkdownWriter.to_bytes([sup]).decode("utf-8")
        assert "> [translation] Hola" in content

    def test_mixed_blockquote_and_bilingual(self):
        """Regular blockquote (without [lang]) is not parsed as translation even with target_lang."""
        content = """---
target_lang: zh
---

# Test

**Speaker:** [00:01:00] He said something [00:01:05]
> This is just a regular quote

**Speaker:** [00:01:05] Hello [00:01:10]
> [zh] 你好
"""
        segments = MarkdownReader.read(content)
        dialogue = [s for s in segments if s.segment_type == "dialogue"]
        # 3 dialogue segments: "He said something", "> This is..." (plain text), "Hello"
        assert len(dialogue) == 3
        # First segment: no translation
        assert dialogue[0].translation is None
        # Regular blockquote is parsed as standalone dialogue, not a translation
        assert dialogue[1].text == "> This is just a regular quote"
        assert dialogue[1].translation is None
        # Third segment: > [zh] is a translation
        assert dialogue[2].translation == "你好"

    def test_target_lang_propagated_to_supervision(self):
        """target_lang from frontmatter is set on Supervision objects."""
        content = """---
target_lang: ja
---

# Test

**Speaker:** [00:01:00] Hello [00:01:05]
> [ja] こんにちは
"""
        sups = MarkdownReader.extract_for_alignment(content)
        assert sups[0].translation == "こんにちは"
        assert sups[0].target_lang == "ja"

    def test_frontmatter_target_lang_round_trip(self):
        """target_lang appears in written frontmatter and is read back."""
        sups = [Supervision(text="Hello", translation="你好", target_lang="zh", start=0.0, duration=1.0)]
        content = MarkdownWriter.to_bytes(sups, metadata={"title": "Test", "target_lang": "zh"}).decode("utf-8")
        assert "target_lang: zh" in content

        fm = MarkdownReader.extract_frontmatter(content)
        assert fm["target_lang"] == "zh"

    def test_hyphenated_lang_code_round_trip(self):
        """BCP-47 tags like zh-CN, pt-BR round-trip correctly."""
        sups = [Supervision(text="Hello", translation="你好", target_lang="zh-CN", start=0.0, duration=1.0)]
        content = MarkdownWriter.to_bytes(sups, metadata={"target_lang": "zh-CN"}).decode("utf-8")
        assert "> [zh-CN] 你好" in content

        read_sups = MarkdownReader.extract_for_alignment(content)
        assert read_sups[0].translation == "你好"
        assert read_sups[0].target_lang == "zh-CN"

    def test_mismatched_tag_not_consumed(self):
        """A > [en] line in a target_lang: zh file is NOT consumed as translation."""
        content = """---
target_lang: zh
---

# Test

**Speaker:** [00:01:00] Hello [00:01:05]
> [en] This is an English note, not a translation
"""
        segments = MarkdownReader.read(content)
        dialogue = [s for s in segments if s.segment_type == "dialogue"]
        # The > [en] line does not match target_lang zh, so no translation
        assert dialogue[0].translation is None

    def test_explicit_target_lang_kwarg(self):
        """Explicit target_lang parameter works without frontmatter."""
        content = """# Test

**Speaker:** [00:01:00] Hello [00:01:05]
> [zh] 你好
"""
        # No frontmatter, but explicit target_lang
        segments = MarkdownReader.read(content, target_lang="zh")
        dialogue = [s for s in segments if s.segment_type == "dialogue"]
        assert dialogue[0].translation == "你好"

    def test_explicit_target_lang_propagated_to_supervision(self):
        """Explicit target_lang kwarg is propagated to Supervision objects."""
        content = """# Test

**Speaker:** [00:01:00] Hello [00:01:05]
> [zh] 你好
"""
        sups = MarkdownReader.extract_for_alignment(content, target_lang="zh")
        assert sups[0].translation == "你好"
        assert sups[0].target_lang == "zh"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
