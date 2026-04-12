"""Tests for podcast transcript support in MarkdownReader.

Podcast transcripts use bare timestamp headings (#### HH:MM:SS Title) and
plain speaker labels (Name: text) instead of bold markers (**Name:**).
These are now handled within the existing MarkdownReader via podcast_mode.
"""

import pytest

from lattifai.caption.formats import detect_format_from_content
from lattifai.caption.formats.markdown import MarkdownReader

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

SIMPLE_TRANSCRIPT = """\
#### 00:00:37 Why AI matters for commerce

Erik Torenberg: So you guys have both been thinking about how AI changes commerce.

Alex Rampell: Well, I started a company years ago that was doing price comparison.

#### 00:06:08 Dynamic pricing

Erik Torenberg: How much will AI result in dynamic pricing?

Alex Rampell: I think it will be huge. The idea that every product has a single price is outdated.
"""

TRANSCRIPT_WITH_TOC = """\
### Timecodes:
[00:37 Why AI matters for commerce](https://example.com/1)
[06:08 Dynamic pricing](https://example.com/2)

_This transcript has been edited lightly for readability._

#### 00:00:37 Why AI matters for commerce

Erik Torenberg: So you guys have both been thinking about this topic.

#### 00:06:08 Dynamic pricing

Erik Torenberg: How much will AI result in dynamic pricing?
"""

TRANSCRIPT_WITH_CONTINUATION = """\
#### 00:00:00 Introduction

Host Name: This is a really long paragraph that continues
on the next line without a speaker prefix.

Guest Speaker: And here is another turn.
"""

TRANSCRIPT_WITH_FRONTMATTER = """\
---
title: AI Podcast
date: 2026-01-15
---

#### 00:00:10 Opening

Alice: Welcome to the show.
"""

TRANSCRIPT_TWO_HASH_HEADING = """\
## 00:01:00 Section with two hashes

Speaker One: Content under two-hash heading.
"""


class TestPodcastTranscriptParsing:
    """Test podcast transcript reading via MarkdownReader."""

    def test_basic_parsing(self):
        """Parse simple transcript with two sections."""
        segments = MarkdownReader.read(SIMPLE_TRANSCRIPT)
        dialogue = [s for s in segments if s.segment_type == "dialogue"]
        assert len(dialogue) == 4

    def test_speaker_extraction(self):
        """Speaker names are correctly extracted."""
        segments = MarkdownReader.read(SIMPLE_TRANSCRIPT)
        dialogue = [s for s in segments if s.segment_type == "dialogue"]
        assert dialogue[0].speaker == "Erik Torenberg"
        assert dialogue[1].speaker == "Alex Rampell"
        assert dialogue[2].speaker == "Erik Torenberg"
        assert dialogue[3].speaker == "Alex Rampell"

    def test_text_extraction(self):
        """Dialogue text is correctly extracted (no speaker prefix)."""
        segments = MarkdownReader.read(SIMPLE_TRANSCRIPT)
        dialogue = [s for s in segments if s.segment_type == "dialogue"]
        assert dialogue[0].text.startswith("So you guys")
        assert "Erik" not in dialogue[0].text

    def test_toc_links_skipped(self):
        """TOC links like [MM:SS Title](url) are skipped."""
        segments = MarkdownReader.read(TRANSCRIPT_WITH_TOC)
        all_text = " ".join(s.text for s in segments)
        assert "example.com" not in all_text

    def test_editorial_disclaimer_skipped(self):
        """Italic editorial notes are skipped."""
        segments = MarkdownReader.read(TRANSCRIPT_WITH_TOC)
        all_text = " ".join(s.text for s in segments)
        assert "edited lightly" not in all_text

    def test_continuation_lines(self):
        """Lines without speaker prefix are appended to previous turn."""
        segments = MarkdownReader.read(TRANSCRIPT_WITH_CONTINUATION)
        dialogue = [s for s in segments if s.segment_type == "dialogue"]
        assert len(dialogue) == 2
        assert "continues on the next line" in dialogue[0].text
        assert dialogue[0].speaker == "Host Name"
        assert dialogue[1].speaker == "Guest Speaker"

    def test_frontmatter_stripped(self):
        """YAML front matter is removed before parsing."""
        segments = MarkdownReader.read(TRANSCRIPT_WITH_FRONTMATTER)
        dialogue = [s for s in segments if s.segment_type == "dialogue"]
        assert len(dialogue) == 1
        assert dialogue[0].speaker == "Alice"
        assert dialogue[0].text == "Welcome to the show."

    def test_two_hash_heading(self):
        """## headings with bare timestamps are also recognized."""
        segments = MarkdownReader.read(TRANSCRIPT_TWO_HASH_HEADING)
        dialogue = [s for s in segments if s.segment_type == "dialogue"]
        assert len(dialogue) == 1

    def test_section_headers_with_include(self):
        """Section headers are returned when include_sections=True."""
        segments = MarkdownReader.read(SIMPLE_TRANSCRIPT, include_sections=True)
        headers = [s for s in segments if s.segment_type == "section_header"]
        assert len(headers) == 2
        assert headers[0].text == "Why AI matters for commerce"
        assert headers[0].timestamp == pytest.approx(37.0)
        assert headers[1].text == "Dynamic pricing"
        assert headers[1].timestamp == pytest.approx(368.0)

    def test_extract_for_alignment(self):
        """extract_for_alignment produces Supervision objects."""
        sups = MarkdownReader.extract_for_alignment(SIMPLE_TRANSCRIPT)
        assert len(sups) == 4
        assert sups[0].speaker == "Erik Torenberg"
        assert sups[0].start == pytest.approx(37.0)


class TestPodcastTranscriptDetection:
    """Test format detection for podcast transcripts."""

    def test_detect_from_content(self):
        """detect_format_from_content identifies podcast transcript as markdown."""
        assert detect_format_from_content(SIMPLE_TRANSCRIPT) == "markdown"

    def test_not_confused_with_bold_markdown(self):
        """Markdown with **Speaker:** bold labels is also detected as markdown."""
        md_content = """\
## [00:00:37] Section title

**Erik:** So you guys have been thinking about this.
"""
        result = detect_format_from_content(md_content)
        assert result == "markdown"


class TestPodcastModeIsolation:
    """Ensure plain speaker detection only activates in podcast mode."""

    def test_plain_text_not_falsely_detected_as_speaker(self):
        """Without bare headings, 'Name: text' is NOT detected as speaker."""
        content = """\
## [00:00:10] Introduction

Note: This is an important note, not a speaker.

Second line of content.
"""
        segments = MarkdownReader.read(content)
        dialogue = [s for s in segments if s.segment_type == "dialogue"]
        # "Note:" should NOT be parsed as speaker since there are no bare headings
        for seg in dialogue:
            assert seg.speaker != "Note"

    def test_podcast_mode_activates_on_bare_heading(self):
        """Plain speaker detection activates only after bare timestamp heading."""
        content = """\
## [00:00:05] Standard section

Note: This should NOT be a speaker.

#### 00:00:10 Podcast section

Alice: This SHOULD be detected as a speaker.
"""
        segments = MarkdownReader.read(content)
        dialogue = [s for s in segments if s.segment_type == "dialogue"]
        # "Alice:" should be detected since we saw bare heading
        alice_segs = [s for s in dialogue if s.speaker == "Alice"]
        assert len(alice_segs) == 1
        # "Note:" should NOT be a speaker (before podcast_mode activated)
        note_segs = [s for s in dialogue if s.speaker == "Note"]
        assert len(note_segs) == 0


class TestPodcastEdgeCases:
    """Edge cases for podcast transcript support."""

    def test_hyphenated_speaker_name(self):
        """Speaker names with hyphens are recognized."""
        content = """\
#### 00:00:00 Test

Mary-Jane Watson: Hello there.
"""
        segments = MarkdownReader.read(content)
        dialogue = [s for s in segments if s.segment_type == "dialogue"]
        assert dialogue[0].speaker == "Mary-Jane Watson"

    def test_apostrophe_in_name(self):
        """Speaker names with apostrophes are recognized."""
        content = """\
#### 00:00:00 Test

O'Brien: Top of the morning.
"""
        segments = MarkdownReader.read(content)
        dialogue = [s for s in segments if s.segment_type == "dialogue"]
        assert dialogue[0].speaker == "O'Brien"

    def test_single_word_speaker(self):
        """Single-word speaker names are recognized."""
        content = """\
#### 00:00:00 Test

Moderator: Let's begin.
"""
        segments = MarkdownReader.read(content)
        dialogue = [s for s in segments if s.segment_type == "dialogue"]
        assert dialogue[0].speaker == "Moderator"

    def test_millisecond_timestamps(self):
        """Timestamps with milliseconds are parsed correctly."""
        content = """\
#### 00:01:30.500 Section with millis

Speaker: Content here.
"""
        segments = MarkdownReader.read(content, include_sections=True)
        headers = [s for s in segments if s.segment_type == "section_header"]
        assert headers[0].timestamp == pytest.approx(90.5)
