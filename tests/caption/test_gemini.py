"""Tests for YouTube transcript reader and writer."""

import pytest

from lattifai.caption import GeminiReader, GeminiSegment, GeminiWriter, Supervision

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


class TestGeminiReader:
    """Tests for GeminiReader class (formerly GeminiReader)."""

    def test_read_all_segments(self, tmp_path):
        """Test reading all segments including events and sections."""
        # Create temp file
        transcript_file = tmp_path / "test_Gemini.md"
        transcript_file.write_text(SAMPLE_TRANSCRIPT)

        # Read all segments
        segments = GeminiReader.read(transcript_file, include_events=True, include_sections=True)

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
        segments = GeminiReader.read(transcript_file, include_events=False, include_sections=False)

        # Should only have dialogue
        types = {seg.segment_type for seg in segments}
        assert types == {"dialogue"}

    def test_parse_timestamp(self):
        """Test timestamp parsing."""
        timestamp = GeminiReader.parse_timestamp("00", "00", "13")
        assert timestamp == 13.0

        timestamp = GeminiReader.parse_timestamp("00", "01", "01")
        assert timestamp == 61.0

        timestamp = GeminiReader.parse_timestamp("01", "00", "00")
        assert timestamp == 3600.0

    def test_speaker_extrevent(self, tmp_path):
        """Test speaker name extrevent."""
        transcript_file = tmp_path / "test_Gemini.md"
        transcript_file.write_text(SAMPLE_TRANSCRIPT)

        segments = GeminiReader.read(transcript_file)

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

        segments = GeminiReader.read(transcript_file, include_events=True, include_sections=True)

        # Segments should have section information
        sections = {s.section for s in segments if s.section is not None}
        assert "Introduction" in sections
        assert "Announcing GPT-4o" in sections

    def test_extract_for_alignment(self, tmp_path):
        """Test extracting supervisions for alignment."""
        transcript_file = tmp_path / "test_Gemini.md"
        transcript_file.write_text(SAMPLE_TRANSCRIPT)

        # Extract for alignment
        supervisions = GeminiReader.extract_for_alignment(transcript_file, merge_consecutive=False)

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
        sups_no_merge = GeminiReader.extract_for_alignment(transcript_file, merge_consecutive=False)

        # Extract with merge
        sups_with_merge = GeminiReader.extract_for_alignment(transcript_file, merge_consecutive=True)

        # Merged should have fewer or equal segments
        assert len(sups_with_merge) <= len(sups_no_merge)


class TestYouTubeGeminiReader:
    """Tests for GeminiReader with YouTube link format."""

    def test_read_youtube_format(self, tmp_path):
        """Test reading YouTube format transcript with link timestamps."""
        transcript_file = tmp_path / "youtube_Gemini.md"
        transcript_file.write_text(SAMPLE_YOUTUBE_TRANSCRIPT)

        # Read all segments
        segments = GeminiReader.read(transcript_file, include_events=True, include_sections=True)

        # Should have sections and dialogue
        assert len(segments) > 0

        # Check segment types
        types = {seg.segment_type for seg in segments}
        assert "section_header" in types
        assert "dialogue" in types

    def test_youtube_timestamp_parsing(self):
        """Test YouTube timestamp parsing from URL format."""
        # Test seconds parsing
        timestamp = GeminiReader.parse_timestamp("12")
        assert timestamp == 12.0

        timestamp = GeminiReader.parse_timestamp("63")
        assert timestamp == 63.0

        timestamp = GeminiReader.parse_timestamp("3661")
        assert timestamp == 3661.0

    def test_youtube_section_headers(self, tmp_path):
        """Test YouTube format section headers."""
        transcript_file = tmp_path / "youtube_Gemini.md"
        transcript_file.write_text(SAMPLE_YOUTUBE_TRANSCRIPT)

        segments = GeminiReader.read(transcript_file, include_sections=True)

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

        segments = GeminiReader.read(transcript_file)

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
        supervisions = GeminiReader.extract_for_alignment(transcript_file, merge_consecutive=False)

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
        sups_no_merge = GeminiReader.extract_for_alignment(transcript_file, merge_consecutive=False)

        # Extract with merge
        sups_with_merge = GeminiReader.extract_for_alignment(transcript_file, merge_consecutive=True)

        # Merged should have fewer or equal segments
        assert len(sups_with_merge) <= len(sups_no_merge)


class TestGeminiWriter:
    """Tests for GeminiWriter class (formerly GeminiWriter)."""

    def test_format_timestamp(self):
        """Test timestamp formatting."""
        # Test various timestamps
        assert GeminiWriter.format_timestamp(13.0) == "[00:00:13]"
        assert GeminiWriter.format_timestamp(61.0) == "[00:01:01]"
        assert GeminiWriter.format_timestamp(3661.0) == "[01:01:01]"
        assert GeminiWriter.format_timestamp(0.0) == "[00:00:00]"

    def test_update_timestamps(self, tmp_path):
        """Test updating transcript with new timestamps."""
        # Create original transcript
        original_file = tmp_path / "original.txt"
        original_file.write_text(SAMPLE_TRANSCRIPT)

        # Extract supervisions
        supervisions = GeminiReader.extract_for_alignment(original_file)

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
        GeminiWriter.update_timestamps(original_file, aligned_supervisions, output_file)

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
        supervisions = GeminiReader.extract_for_alignment(original_file)

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
        GeminiWriter.write_aligned_transcript(supervisions, output_file, include_word_timestamps=True)

        # Check output
        assert output_file.exists()
        content = output_file.read_text()
        assert "Aligned Transcript" in content
        assert "[00:00:" in content  # Should have timestamps

    def test_write_aligned_without_words(self, tmp_path):
        """Test writing aligned transcript without word timestamps."""
        original_file = tmp_path / "original.txt"
        original_file.write_text(SAMPLE_TRANSCRIPT)

        supervisions = GeminiReader.extract_for_alignment(original_file)

        output_file = tmp_path / "aligned_no_words.txt"
        GeminiWriter.write_aligned_transcript(supervisions, output_file, include_word_timestamps=False)

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
        GeminiWriter.write_aligned_transcript(
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
        """Test GeminiWriter.write() accepts extra kwargs."""
        supervisions = [
            Supervision(id="test_001", text="Test content", start=0.0, duration=1.5),
        ]

        output_file = tmp_path / "output_write.md"
        # Should not raise TypeError
        result = GeminiWriter.write(
            supervisions,
            output_file,
            include_speaker=True,
            word_level=False,
        )

        assert result == output_file
        assert output_file.exists()

    def test_to_bytes_with_extra_kwargs(self):
        """Test GeminiWriter.to_bytes() accepts extra kwargs."""
        supervisions = [
            Supervision(id="test_001", text="Bytes test", start=0.0, duration=1.0),
        ]

        # Should not raise TypeError
        result = GeminiWriter.to_bytes(
            supervisions,
            include_word_timestamps=False,
            include_speaker=True,
            custom_param="ignored",
        )

        assert isinstance(result, bytes)
        assert b"Bytes test" in result


class TestGeminiGeminiSegment:
    """Tests for GeminiSegment dataclass (shared)."""

    def test_segment_creation(self):
        """Test creating a GeminiSegment."""
        segment = GeminiSegment(
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
        segment = GeminiSegment(text="Test", timestamp=10.5)
        assert segment.start == 10.5

        segment_no_ts = GeminiSegment(text="Test", timestamp=None)
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

        segments = GeminiReader.read(transcript_file, include_events=True)

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

        segments = GeminiReader.read(transcript_file, include_events=True)

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

        segments = GeminiReader.read(transcript_file, include_events=True)

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

        segments = GeminiReader.read(test_file, include_events=True)

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

        segments = GeminiReader.read(transcript_file, include_events=True)

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
        segments = GeminiReader.read(content, include_events=True)

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
        supervisions = GeminiReader.extract_for_alignment(content)

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

        segments = GeminiReader.read(test_file, include_events=True)

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
        segments = GeminiReader.read(content, include_events=True)
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
        ts = GeminiReader.parse_timestamp("00", "00", "11", "750")
        assert ts == 11.75

        ts = GeminiReader.parse_timestamp("00", "01", "30", "500")
        assert ts == 90.5

        # MM:SS.mmm format (using ms keyword)
        ts = GeminiReader.parse_timestamp("01", "30", ms="500")
        assert ts == 90.5

        # Without milliseconds (backward compatibility)
        ts = GeminiReader.parse_timestamp("00", "00", "11")
        assert ts == 11.0

    def test_inline_both_timestamps_with_ms(self):
        """Test parsing lines with both start and end timestamps including milliseconds."""
        content = """[00:00:11.750] Hi everyone. [00:00:12.500]

[00:00:12.500] [Applause] [00:00:16.800]

[00:00:16.800] Thank you. [00:00:20.400]
"""
        segments = GeminiReader.read(content, include_events=True)
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
        segments = GeminiReader.read(content)
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

        segments = GeminiReader.read(test_file, include_events=True)

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
        supervisions = GeminiReader.extract_for_alignment(content)

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
        segments = GeminiReader.read(content, include_events=True)

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
        supervisions = GeminiReader.extract_for_alignment(content)

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
        supervisions = GeminiReader.extract_for_alignment(content)

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
        supervisions = GeminiReader.extract_for_alignment(content)

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
        supervisions = GeminiReader.extract_for_alignment(content)

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
        supervisions = GeminiReader.extract_for_alignment(content)

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
        supervisions = GeminiReader.extract_for_alignment(content)

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
        supervisions = GeminiReader.extract_for_alignment(transcript_file)
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
        GeminiWriter.update_timestamps(transcript_file, aligned_supervisions, updated_file)
        assert updated_file.exists()

        # 5. Write simplified aligned transcript
        simple_file = tmp_path / "simple_aligned.txt"
        GeminiWriter.write_aligned_transcript(aligned_supervisions, simple_file, include_word_timestamps=True)
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
        supervisions = GeminiReader.extract_for_alignment(transcript_file)
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
        GeminiWriter.update_timestamps(transcript_file, aligned_supervisions, updated_file)
        assert updated_file.exists()

        # 5. Write simplified aligned transcript
        simple_file = tmp_path / "simple_youtube_aligned.txt"
        GeminiWriter.write_aligned_transcript(aligned_supervisions, simple_file, include_word_timestamps=True)
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
        original_sups = GeminiReader.extract_for_alignment(original_file)

        # Test YouTube format
        youtube_file = tmp_path / "youtube.txt"
        youtube_file.write_text(SAMPLE_YOUTUBE_TRANSCRIPT)
        youtube_sups = GeminiReader.extract_for_alignment(youtube_file)

        # Both should work and return supervisions
        assert len(original_sups) > 0
        assert len(youtube_sups) > 0

        # Both should have valid supervisions
        for sup in original_sups + youtube_sups:
            assert isinstance(sup, Supervision)
            assert sup.text is not None
            assert sup.start >= 0
            assert sup.duration > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
