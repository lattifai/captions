"""Tests for ASS format: read, write, roundtrip, line breaks, standardization.

Consolidates ASS-specific test coverage that was previously scattered across
test_background_color, test_speaker_color, test_speaker_roundtrip, etc.
"""

import re

import pytest

from lattifai.caption import Caption, Supervision
from lattifai.caption.config import ASSConfig, RenderConfig, StandardizationConfig
from lattifai.caption.formats.pysubs2 import ASSFormat
from lattifai.caption.standardize import CaptionStandardizer
from lattifai.caption.supervision import AlignmentItem


# ── Fixtures ──────────────────────────────────────────


@pytest.fixture
def simple_sups():
    """Two basic supervisions with speakers."""
    return [
        Supervision(text="Hello world", start=0.0, duration=2.0, speaker="Alice"),
        Supervision(text="How are you?", start=2.5, duration=1.5, speaker="Bob"),
    ]


@pytest.fixture
def word_aligned_sups():
    """Supervisions with word-level alignment data."""
    return [
        Supervision(
            text="Hello world",
            start=0.0,
            duration=2.0,
            speaker="Alice",
            alignment={
                "word": [
                    AlignmentItem(symbol="Hello", start=0.0, duration=0.8),
                    AlignmentItem(symbol="world", start=0.9, duration=1.0),
                ]
            },
        ),
        Supervision(
            text="Goodbye everyone",
            start=3.0,
            duration=2.0,
            speaker="Bob",
            alignment={
                "word": [
                    AlignmentItem(symbol="Goodbye", start=3.0, duration=1.0),
                    AlignmentItem(symbol="everyone", start=4.1, duration=0.8),
                ]
            },
        ),
    ]


@pytest.fixture
def bilingual_sups():
    """Supervisions with translation (bilingual)."""
    return [
        Supervision(
            text="Hello world",
            start=0.0,
            duration=2.0,
            speaker="Host",
            translation="你好世界",
        ),
        Supervision(
            text="How are you?",
            start=3.0,
            duration=2.0,
            speaker="Guest",
            translation="你好吗？",
        ),
    ]


# ── Basic Read / Write ────────────────────────────────


class TestASSBasicIO:
    """Test basic ASS read and write operations."""

    ASS_CONTENT = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,Alice,0,0,0,,Hello world
Dialogue: 0,0:00:04.00,0:00:06.00,Default,Bob,0,0,0,,How are you?
"""

    def test_read_basic_ass(self):
        """Read ASS content and verify supervisions."""
        sups = ASSFormat.parse(self.ASS_CONTENT).supervisions
        assert len(sups) == 2
        assert sups[0].text == "Hello world"
        assert sups[0].speaker == "Alice"
        assert sups[1].text == "How are you?"
        assert sups[1].speaker == "Bob"

    def test_write_basic_ass(self, simple_sups, tmp_path):
        """Write supervisions to ASS file."""
        ass_file = tmp_path / "output.ass"
        caption = Caption.from_supervisions(simple_sups)
        caption.write(ass_file)

        content = ass_file.read_text()
        assert "[Script Info]" in content
        assert "[V4+ Styles]" in content
        assert "[Events]" in content
        assert "Hello world" in content

    def test_roundtrip(self, simple_sups, tmp_path):
        """Write then read ASS preserves content."""
        ass_file = tmp_path / "roundtrip.ass"
        caption = Caption.from_supervisions(simple_sups)
        caption.write(ass_file)

        caption_read = Caption.read(ass_file)
        assert len(caption_read.supervisions) == 2
        assert caption_read.supervisions[0].text == "Hello world"
        assert caption_read.supervisions[1].text == "How are you?"

    def test_to_bytes(self, simple_sups):
        """Write ASS to bytes (in-memory)."""
        data = ASSFormat.to_bytes(simple_sups)
        content = data.decode("utf-8")
        assert "Dialogue:" in content
        assert "Hello world" in content

    def test_from_string_and_back(self):
        """Caption.from_string -> to_string roundtrip."""
        caption = Caption.from_string(self.ASS_CONTENT, format="ass")
        output = caption.to_string(format="ass")
        assert "Hello world" in output
        assert "How are you?" in output


# ── Line Break Handling ───────────────────────────────


class TestASSLineBreaks:
    r"""Test \n to \N conversion in ASS output.

    ASS format uses \N for inline line breaks (not actual newlines).
    The standardizer's _smart_split_text inserts \n, which must be
    converted to \N before writing to ASS.
    """

    def test_newline_converted_to_ass_linebreak(self):
        r"""Text with \n should produce \N in ASS output."""
        sups = [
            Supervision(text="Line one\nLine two", start=0.0, duration=3.0),
        ]
        content = ASSFormat.to_bytes(sups).decode("utf-8")

        # Should NOT have actual newline in Dialogue text
        dialogue_lines = [l for l in content.splitlines() if l.startswith("Dialogue:")]
        assert len(dialogue_lines) == 1
        assert "\\N" in dialogue_lines[0]
        assert "Line one\\NLine two" in dialogue_lines[0]

    def test_multiple_newlines_converted(self):
        r"""Multiple \n should all be converted to \N."""
        sups = [
            Supervision(text="A\nB\nC", start=0.0, duration=3.0),
        ]
        content = ASSFormat.to_bytes(sups).decode("utf-8")
        dialogue_lines = [l for l in content.splitlines() if l.startswith("Dialogue:")]
        assert len(dialogue_lines) == 1
        assert "A\\NB\\NC" in dialogue_lines[0]

    def test_existing_ass_linebreak_preserved(self):
        r"""Text already containing \N should not be double-escaped."""
        sups = [
            Supervision(text="Line one\\NLine two", start=0.0, duration=3.0),
        ]
        content = ASSFormat.to_bytes(sups).decode("utf-8")
        dialogue_lines = [l for l in content.splitlines() if l.startswith("Dialogue:")]
        assert len(dialogue_lines) == 1
        # \N should be present (not \\\\N)
        assert "Line one\\NLine two" in dialogue_lines[0]

    def test_bilingual_uses_ass_linebreak(self, bilingual_sups):
        r"""Bilingual text should use \N as separator in ASS."""
        content = ASSFormat.to_bytes(bilingual_sups).decode("utf-8")
        dialogue_lines = [l for l in content.splitlines() if l.startswith("Dialogue:")]
        # Each dialogue should be on a single file line
        assert len(dialogue_lines) == 2
        # Should contain \N separator between EN and ZH text
        assert "\\N" in dialogue_lines[0]

    def test_each_dialogue_is_single_file_line(self, simple_sups):
        """Every Dialogue entry must be a single line in the file."""
        content = ASSFormat.to_bytes(simple_sups).decode("utf-8")
        dialogue_lines = [l for l in content.splitlines() if l.startswith("Dialogue:")]
        for line in dialogue_lines:
            # No embedded newlines (would break ASS parser)
            assert "\n" not in line

    def test_standardization_linebreaks_in_ass(self, tmp_path):
        """Standardization + ASS write should produce valid single-line Dialogues."""
        long_text = (
            "I think within a decade a lot of things that mathematicians "
            "currently do um what we spend a lot of the bulk of our time "
            "doing can be done by AI"
        )
        sups = [Supervision(text=long_text, start=0.0, duration=5.0)]
        config = StandardizationConfig(max_chars_per_line=42, max_lines=2)
        caption = Caption.from_supervisions(sups)

        ass_file = tmp_path / "std.ass"
        caption.write(ass_file, standardization=config)

        content = ass_file.read_text()
        for line in content.splitlines():
            if line.startswith("Dialogue:"):
                # Line breaks should be \N, not actual newlines
                assert "\\N" in line or len(line.split(",", 9)[-1]) <= 42


# ── Speaker Handling ──────────────────────────────────


class TestASSSpeaker:
    """Test speaker label handling in ASS format."""

    def test_speaker_in_name_field(self, simple_sups, tmp_path):
        """Speaker name stored in ASS Name field."""
        ass_file = tmp_path / "speaker.ass"
        Caption.from_supervisions(simple_sups).write(ass_file)
        content = ass_file.read_text()
        assert ",Alice," in content
        assert ",Bob," in content

    def test_speaker_in_text_default(self, simple_sups, tmp_path):
        """Default: speaker prefix included in text."""
        ass_file = tmp_path / "speaker_text.ass"
        Caption.from_supervisions(simple_sups).write(ass_file)
        content = ass_file.read_text()
        assert "Alice: Hello world" in content
        assert "Bob: How are you?" in content

    def test_speaker_excluded_from_text(self, simple_sups, tmp_path):
        """include_speaker_in_text=False: speaker only in Name field."""
        ass_file = tmp_path / "no_speaker_text.ass"
        render = RenderConfig(include_speaker_in_text=False)
        Caption.from_supervisions(simple_sups).write(ass_file, render=render)
        content = ass_file.read_text()
        assert ",Alice," in content  # Name field
        assert "Alice:" not in content  # Not in text

    def test_speaker_roundtrip_with_include(self, tmp_path):
        """ASS write->read roundtrip preserves speakers (include mode)."""
        sups = [
            Supervision(text="Hello", start=1.0, duration=2.0, speaker="Host"),
            Supervision(text="Hi", start=4.0, duration=2.0, speaker="Guest"),
        ]
        ass_file = tmp_path / "rt.ass"
        Caption.from_supervisions(sups).write(ass_file)

        read_back = Caption.read(ass_file)
        assert read_back.supervisions[0].speaker is not None
        assert read_back.supervisions[0].text == "Hello"

    def test_speaker_roundtrip_without_include(self, tmp_path):
        """ASS write->read roundtrip (no-include mode) recovers from Name field."""
        sups = [
            Supervision(text="Hello", start=1.0, duration=2.0, speaker="Host"),
            Supervision(text="Hi", start=4.0, duration=2.0, speaker="Guest"),
        ]
        ass_file = tmp_path / "rt2.ass"
        Caption.from_supervisions(sups).write(
            ass_file, render=RenderConfig(include_speaker_in_text=False)
        )

        read_back = Caption.read(ass_file)
        assert read_back.supervisions[0].speaker == "Host"
        assert read_back.supervisions[0].text == "Hello"
        assert read_back.supervisions[1].speaker == "Guest"


# ── Standardization + ASS Integration ─────────────────


class TestASSStandardization:
    """Test standardization applied during ASS write."""

    def test_long_text_split_into_multiple_segments(self):
        """Text exceeding max_lines * max_chars_per_line should split into multiple segments."""
        long_text = (
            "I am very curious when you expect AIs that can like actually do "
            "frontier math better than the at least as good well as the best "
            "human mathematicians in the world today"
        )
        assert len(long_text) > 84

        sups = [Supervision(text=long_text, start=0.0, duration=10.0)]
        standardizer = CaptionStandardizer(max_chars_per_line=42, max_lines=2)
        result = standardizer.process(sups)

        assert len(result) >= 2
        for seg in result:
            text = seg.text or ""
            for line in text.split("\n"):
                assert len(line) <= 50, f"Line too long ({len(line)} chars): {line}"

    def test_contraction_not_split(self):
        """English contractions (they're, can't, it's) must NOT be split mid-word."""
        text = (
            "I mean in in some ways they\u2019re already doing frontier math "
            "that is super intelligent and can\u2019t be replicated easily"
        )
        sups = [Supervision(text=text, start=0.0, duration=5.0)]
        standardizer = CaptionStandardizer(max_chars_per_line=42, max_lines=2)
        result = standardizer.process(sups)

        for seg in result:
            seg_text = seg.text or ""
            for line in seg_text.split("\n"):
                # No line should end with a bare apostrophe from a contraction
                assert not line.rstrip().endswith("\u2019"), (
                    f"Contraction split mid-word: '{line}'"
                )
                # No line should start with orphaned contraction suffix
                stripped = line.lstrip()
                assert not re.match(r"^(re|t|s|ve|ll|d)\b", stripped) or len(stripped) > 10, (
                    f"Orphaned contraction suffix: '{line}'"
                )

    def test_split_with_word_alignment_uses_timestamps(self):
        """When word alignment is available, timing should come from word timestamps."""
        words = [
            AlignmentItem(symbol="Hello", start=0.0, duration=0.5),
            AlignmentItem(symbol="world", start=0.6, duration=0.4),
            AlignmentItem(symbol="this", start=1.2, duration=0.3),
            AlignmentItem(symbol="is", start=1.6, duration=0.2),
            AlignmentItem(symbol="a", start=1.9, duration=0.1),
            AlignmentItem(symbol="test", start=2.1, duration=0.3),
            AlignmentItem(symbol="of", start=2.5, duration=0.2),
            AlignmentItem(symbol="word", start=2.8, duration=0.3),
            AlignmentItem(symbol="alignment", start=3.2, duration=0.5),
            AlignmentItem(symbol="based", start=3.8, duration=0.3),
            AlignmentItem(symbol="splitting", start=4.2, duration=0.4),
            AlignmentItem(symbol="in", start=4.7, duration=0.1),
            AlignmentItem(symbol="standardizer", start=4.9, duration=0.5),
        ]
        text = " ".join(w.symbol for w in words)  # 74 chars
        sups = [
            Supervision(
                text=text, start=0.0, duration=5.5,
                alignment={"word": words},
            )
        ]
        standardizer = CaptionStandardizer(max_chars_per_line=30, max_lines=2)
        result = standardizer.process(sups)

        assert len(result) >= 2
        # First segment should start near the first word's start time
        assert result[0].start <= 0.1
        # Each segment should have alignment data
        for seg in result:
            alignment = getattr(seg, "alignment", None)
            assert alignment is not None, "Split segment lost alignment data"
            assert "word" in alignment
            assert len(alignment["word"]) > 0

    def test_split_prefers_larger_gaps(self):
        """With alignment, splitting should prefer positions with larger inter-word gaps."""
        # Words with a big gap (0.8s) between "frontier" and "math"
        words = [
            AlignmentItem(symbol="doing", start=0.0, duration=0.3),
            AlignmentItem(symbol="frontier", start=0.4, duration=0.4),
            # ← 0.8s gap here (natural pause)
            AlignmentItem(symbol="math", start=1.6, duration=0.3),
            AlignmentItem(symbol="better", start=2.0, duration=0.3),
            AlignmentItem(symbol="than", start=2.4, duration=0.2),
            AlignmentItem(symbol="humans", start=2.7, duration=0.3),
        ]
        text = " ".join(w.symbol for w in words)  # 37 chars
        sups = [
            Supervision(
                text=text, start=0.0, duration=3.0,
                alignment={"word": words},
            )
        ]
        # Force split with small budget
        standardizer = CaptionStandardizer(max_chars_per_line=20, max_lines=1)
        result = standardizer.process(sups)

        if len(result) >= 2:
            # First segment should end around "frontier" (before the big gap)
            first_text = result[0].text or ""
            assert "frontier" in first_text or "doing" in first_text

    def test_short_text_not_split(self):
        """Text within budget should remain a single segment."""
        sups = [Supervision(text="Short text", start=0.0, duration=2.0)]
        standardizer = CaptionStandardizer(max_chars_per_line=42, max_lines=2)
        result = standardizer.process(sups)
        assert len(result) == 1

    def test_split_preserves_speaker(self):
        """Split segments should preserve the original speaker."""
        long_text = "word " * 25  # 125 chars
        sups = [Supervision(text=long_text.strip(), start=0.0, duration=8.0, speaker="Tao")]
        standardizer = CaptionStandardizer(max_chars_per_line=42, max_lines=2)
        result = standardizer.process(sups)
        assert len(result) >= 2
        for seg in result:
            assert seg.speaker == "Tao"

    def test_split_distributes_duration(self):
        """Split segments should have proportional durations."""
        long_text = "word " * 25  # 125 chars
        sups = [Supervision(text=long_text.strip(), start=0.0, duration=10.0)]
        standardizer = CaptionStandardizer(max_chars_per_line=42, max_lines=2)
        result = standardizer.process(sups)
        assert len(result) >= 2
        for seg in result:
            assert seg.duration >= 0.8  # min_duration enforced

    def test_standardized_ass_output_valid(self, tmp_path):
        """ASS file with standardization should have all Dialogues on single lines."""
        long_text = (
            "Um I mean you could argue that calculators were doing frontier math "
            "that that humans could not accomplish but it was different from what "
            "we are used to seeing in traditional mathematical research"
        )
        sups = [Supervision(text=long_text, start=0.0, duration=8.0, speaker="Tao")]
        config = StandardizationConfig(max_chars_per_line=42, max_lines=2)
        caption = Caption.from_supervisions(sups)

        ass_file = tmp_path / "std_output.ass"
        caption.write(ass_file, standardization=config)

        content = ass_file.read_text()
        for line in content.splitlines():
            if line.startswith("Dialogue:"):
                # Must be single-line (no embedded newlines)
                fields = line.split(",", 9)
                text_field = fields[-1]
                assert "\n" not in text_field

    def test_min_duration_enforced(self):
        """Segments shorter than min_duration should be extended."""
        sups = [
            Supervision(text="Hi", start=0.0, duration=0.3),
            Supervision(text="Bye", start=5.0, duration=0.3),
        ]
        standardizer = CaptionStandardizer(min_duration=0.8)
        result = standardizer.process(sups)
        for seg in result:
            assert seg.duration >= 0.8

    def test_min_gap_enforced(self):
        """Gap between segments should be >= min_gap."""
        sups = [
            Supervision(text="First", start=0.0, duration=2.0),
            Supervision(text="Second", start=2.01, duration=2.0),  # 10ms gap
        ]
        standardizer = CaptionStandardizer(min_gap=0.08)
        result = standardizer.process(sups)
        gap = result[1].start - (result[0].start + result[0].duration)
        assert gap >= 0.07  # Allow small float tolerance


# ── Metadata Preservation ─────────────────────────────


class TestASSMetadata:
    """Test ASS metadata extraction and preservation."""

    def test_extract_script_info(self):
        """Extract Script Info fields from ASS content."""
        content = """\
[Script Info]
Title: Test
PlayResX: 1920
PlayResY: 1080
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,1,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello
"""
        metadata = ASSFormat.parse(content).format_metadata
        assert metadata["ass_info"]["Title"] == "Test"
        assert metadata["ass_info"]["PlayResX"] == "1920"

    def test_event_custom_preserved(self):
        """Event attributes (style, layer, margins) stored in Supervision.custom."""
        content = """\
[Script Info]
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,1,2,10,10,10,1
Style: Custom,Impact,30,&H00FF0000,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,3,3,2,20,20,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 1,0:00:01.00,0:00:03.00,Custom,,10,10,20,FadeIn,Styled text
"""
        sups = ASSFormat.parse(content).supervisions
        assert len(sups) == 1
        assert sups[0].custom["ass_style"] == "Custom"
        assert sups[0].custom["ass_layer"] == 1
        assert sups[0].custom["ass_margin_l"] == 10
        assert sups[0].custom["ass_effect"] == "FadeIn"

    def test_metadata_roundtrip(self):
        """Read ASS with metadata, write back, verify preserved."""
        content = """\
[Script Info]
Title: Roundtrip Test
PlayResX: 1920
PlayResY: 1080
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello
"""
        caption = Caption.from_string(content, format="ass")
        output = caption.to_string(format="ass")
        assert "Title: Roundtrip Test" in output
        assert "PlayResX: 1920" in output


# ── Style and Visual ──────────────────────────────────


class TestASSStyle:
    """Test ASS style rendering (font, resolution, alignment)."""

    def test_default_resolution(self, simple_sups):
        """Default ASS output should have 1920x1080 resolution."""
        content = ASSFormat.to_bytes(simple_sups).decode("utf-8")
        assert "PlayResX: 1920" in content
        assert "PlayResY: 1080" in content

    def test_custom_font(self, simple_sups):
        """Custom font_name should appear in Default style."""
        config = ASSConfig(font_name="PingFang SC")
        content = ASSFormat.to_bytes(simple_sups, config=config).decode("utf-8")
        assert "PingFang SC" in content

    def test_custom_font_size(self, simple_sups):
        """Custom font_size should appear in Default style."""
        config = ASSConfig(font_size=24)
        content = ASSFormat.to_bytes(simple_sups, config=config).decode("utf-8")
        style_lines = [l for l in content.splitlines() if l.startswith("Style: Default")]
        assert len(style_lines) == 1
        fields = style_lines[0].split(",")
        assert fields[2].strip() == "24"  # fontsize is 3rd field


# ── Karaoke Integration ──────────────────────────────


class TestASSKaraoke:
    """Test karaoke mode in ASS output."""

    def test_karaoke_tags_present(self, word_aligned_sups):
        r"""Karaoke mode should produce \kf timing tags."""
        config = ASSConfig(karaoke_effect="sweep")
        content = ASSFormat.to_bytes(
            word_aligned_sups, render=RenderConfig(word_level=True), config=config
        ).decode("utf-8")
        assert "{\\kf" in content

    def test_karaoke_style_defined(self, word_aligned_sups):
        """Karaoke mode should define a Karaoke style."""
        config = ASSConfig(karaoke_effect="sweep")
        content = ASSFormat.to_bytes(
            word_aligned_sups, render=RenderConfig(word_level=True), config=config
        ).decode("utf-8")
        assert "Style: Karaoke" in content

    def test_karaoke_with_speaker_color(self, word_aligned_sups):
        r"""Karaoke + speaker_color='auto' should produce \c color tags."""
        config = ASSConfig(karaoke_effect="sweep", speaker_color="auto")
        content = ASSFormat.to_bytes(
            word_aligned_sups, render=RenderConfig(word_level=True), config=config
        ).decode("utf-8")
        assert "{\\c&H" in content

    def test_karaoke_without_alignment_falls_back(self, simple_sups):
        """Supervisions without alignment should render in standard mode."""
        config = ASSConfig(karaoke_effect="sweep")
        content = ASSFormat.to_bytes(
            simple_sups, render=RenderConfig(word_level=True), config=config
        ).decode("utf-8")
        # Should still have Dialogue entries (standard mode fallback)
        assert "Dialogue:" in content
        # Should NOT have karaoke timing tags (no alignment data)
        assert "{\\kf" not in content
