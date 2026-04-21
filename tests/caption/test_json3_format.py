"""Tests for YouTube JSON3 format handler."""

import json
import tempfile
from pathlib import Path

import pytest

from lattifai.caption.formats import detect_format, detect_format_from_content, get_reader, get_writer
from lattifai.caption.formats.json3 import JSON3Format
from lattifai.caption.supervision import Supervision

DATA_DIR = Path(__file__).parent.parent / "data" / "captions"
SAMPLE_JSON3 = DATA_DIR / "a16z_sample.json3"


# ---------------------------------------------------------------------------
# Minimal fixture (inline) for tests that don't need a real file
# ---------------------------------------------------------------------------
MINIMAL_JSON3 = json.dumps({
    "wireMagic": "pb3",
    "pens": [{}],
    "wsWinStyles": [{}],
    "wpWinPositions": [{}],
    "events": [
        {"tStartMs": 0, "dDurationMs": 10000, "id": 1, "wpWinPosId": 1, "wsWinStyleId": 1},
        {
            "tStartMs": 100,
            "dDurationMs": 2000,
            "wWinId": 1,
            "segs": [
                {"utf8": "Hello", "acAsrConf": 0},
                {"utf8": " world", "tOffsetMs": 500, "acAsrConf": 0},
            ],
        },
        {"tStartMs": 1900, "dDurationMs": 200, "wWinId": 1, "aAppend": 1, "segs": [{"utf8": "\n"}]},
        {
            "tStartMs": 2100,
            "dDurationMs": 1500,
            "wWinId": 1,
            "segs": [
                {"utf8": ">> Goodbye", "acAsrConf": 0},
                {"utf8": " all", "tOffsetMs": 800, "acAsrConf": 0},
            ],
        },
    ],
})


class TestJSON3Reader:
    """Test JSON3 format reading."""

    def test_read_from_file(self):
        """Read from the real YouTube JSON3 sample file."""
        sups = JSON3Format.parse(SAMPLE_JSON3).supervisions
        # Sample has window event + 11 content events + 10 append events
        assert len(sups) == 11

    def test_read_from_content(self):
        """Read from inline JSON string."""
        sups = JSON3Format.parse(MINIMAL_JSON3).supervisions
        assert len(sups) == 2

    def test_word_level_alignment(self):
        """Each supervision has word-level alignment with correct timing."""
        sups = JSON3Format.parse(MINIMAL_JSON3).supervisions

        first = sups[0]
        assert first.alignment is not None
        words = first.alignment["word"]
        assert len(words) == 2
        assert words[0].symbol == "Hello"
        assert words[0].start == pytest.approx(0.1)
        assert words[1].symbol == "world"
        assert words[1].start == pytest.approx(0.6)  # 100ms + 500ms offset

    def test_word_duration_calculation(self):
        """Word durations are calculated from inter-word gaps."""
        sups = JSON3Format.parse(MINIMAL_JSON3).supervisions
        words = sups[0].alignment["word"]

        # First word duration = next_word_start - this_word_start
        assert words[0].duration == pytest.approx(0.5)
        # Last word duration = event_end - word_start
        assert words[1].duration == pytest.approx(1.5)  # (100+2000)/1000 - 0.6

    def test_append_events_skipped(self):
        """Append/continuation events (aAppend=1) are skipped."""
        sups = JSON3Format.parse(MINIMAL_JSON3).supervisions
        # Only 2 content events, append events filtered out
        assert len(sups) == 2

    def test_window_events_skipped(self):
        """Window/config events (no segs) are skipped."""
        sups = JSON3Format.parse(MINIMAL_JSON3).supervisions
        assert len(sups) == 2

    def test_speaker_change_marker(self):
        """>> prefix is stripped from text and set as speaker marker."""
        sups = JSON3Format.parse(MINIMAL_JSON3).supervisions

        # First segment: no speaker change
        assert sups[0].speaker is None
        assert "Hello world" in sups[0].text

        # Second segment: >> speaker change
        assert sups[1].speaker == ">>"
        assert ">>" not in sups[1].text
        assert "Goodbye all" in sups[1].text

    def test_speaker_change_word_alignment_clean(self):
        """>> is stripped from word alignment symbols too."""
        sups = JSON3Format.parse(MINIMAL_JSON3).supervisions
        speaker_sup = sups[1]
        words = speaker_sup.alignment["word"]
        assert words[0].symbol == "Goodbye"
        assert not words[0].symbol.startswith(">>")

    def test_real_sample_first_segment(self):
        """Verify first content segment from real YouTube data."""
        sups = JSON3Format.parse(SAMPLE_JSON3).supervisions
        first = sups[0]
        assert first.start == pytest.approx(0.08)
        assert first.text == "The diffusion of AI capability is going"
        assert first.speaker is None

        words = first.alignment["word"]
        assert len(words) == 7
        assert words[0].symbol == "The"
        assert words[0].start == pytest.approx(0.08)
        assert words[1].symbol == "diffusion"
        assert words[1].start == pytest.approx(0.32)

    def test_real_sample_speaker_change(self):
        """Verify >> speaker change in real YouTube data (event index 7)."""
        sups = JSON3Format.parse(SAMPLE_JSON3).supervisions
        # Event at tStartMs=4640 has ">> It's"
        speaker_sup = next(s for s in sups if s.speaker == ">>")
        assert "It's" in speaker_sup.text
        assert ">>" not in speaker_sup.text

    def test_real_sample_segment_count(self):
        """Verify segment count from real sample (11 content events)."""
        sups = JSON3Format.parse(SAMPLE_JSON3).supervisions
        assert len(sups) == 11

        # Check speaker distribution
        speakers = [s.speaker for s in sups]
        assert speakers.count(">>") == 2  # Events with >> prefix
        assert speakers.count(None) == 9

    def test_normalize_text_html_entities(self):
        """HTML entities in utf8 fields are decoded."""
        content = json.dumps({
            "wireMagic": "pb3",
            "events": [{
                "tStartMs": 0,
                "dDurationMs": 1000,
                "segs": [{"utf8": "it&#39;s", "acAsrConf": 0}],
            }],
        })
        sups = JSON3Format.parse(content).supervisions
        assert sups[0].text == "it's"

    def test_normalize_text_disabled(self):
        """normalize_text=False preserves raw text."""
        content = json.dumps({
            "wireMagic": "pb3",
            "events": [{
                "tStartMs": 0,
                "dDurationMs": 1000,
                "segs": [{"utf8": "it&#39;s", "acAsrConf": 0}],
            }],
        })
        sups = JSON3Format.parse(content, normalize_text=False).supervisions
        assert "&#39;" in sups[0].text

    def test_zero_duration_events_skipped(self):
        """Events with dDurationMs <= 0 are skipped."""
        content = json.dumps({
            "wireMagic": "pb3",
            "events": [
                {"tStartMs": 0, "dDurationMs": 0, "segs": [{"utf8": "skip"}]},
                {"tStartMs": 100, "dDurationMs": 500, "segs": [{"utf8": "keep"}]},
            ],
        })
        sups = JSON3Format.parse(content).supervisions
        assert len(sups) == 1
        assert sups[0].text == "keep"


class TestJSON3Writer:
    """Test JSON3 format writing."""

    def test_write_basic(self):
        """Write supervisions to JSON3 format."""
        sups = [
            Supervision(text="Hello world", start=0.1, duration=2.0),
            Supervision(text="Goodbye", start=3.0, duration=1.0),
        ]
        output = JSON3Format.to_bytes(sups)
        data = json.loads(output)

        assert data["wireMagic"] == "pb3"
        assert "events" in data
        # 1 window + 2 content + 2 append = 5
        assert len(data["events"]) == 5

    def test_write_with_word_level(self):
        """Write with word_level=True includes timing offsets."""
        from lattifai.caption.supervision import AlignmentItem

        sups = [
            Supervision(
                text="Hello world",
                start=0.1,
                duration=2.0,
                alignment={"word": [
                    AlignmentItem(symbol="Hello", start=0.1, duration=0.5),
                    AlignmentItem(symbol="world", start=0.6, duration=1.5),
                ]},
            ),
        ]
        output = JSON3Format.to_bytes(sups, word_level=True)
        data = json.loads(output)

        # Find the content event (skip window event)
        content_events = [e for e in data["events"] if "segs" in e and not e.get("aAppend")]
        assert len(content_events) == 1

        segs = content_events[0]["segs"]
        assert len(segs) == 2
        assert segs[0]["utf8"] == "Hello"
        assert "tOffsetMs" not in segs[0]  # First word has 0 offset, omitted
        assert segs[1]["utf8"] == " world"
        assert segs[1]["tOffsetMs"] == 500

    def test_write_speaker_change(self):
        """Speaker change marker >> is preserved in output."""
        sups = [Supervision(text="Hello", start=0.0, duration=1.0, speaker=">>")]
        output = JSON3Format.to_bytes(sups, include_speaker=True)
        data = json.loads(output)

        content_events = [e for e in data["events"] if "segs" in e and not e.get("aAppend")]
        first_seg = content_events[0]["segs"][0]
        assert first_seg["utf8"].startswith(">> ")

    def test_write_to_file(self, tmp_path):
        """Write to file and read back."""
        sups = [Supervision(text="Test caption", start=1.0, duration=2.0)]
        out = tmp_path / "test.json3"
        JSON3Format.write(sups, out)

        assert out.exists()
        data = json.loads(out.read_text())
        assert data["wireMagic"] == "pb3"

    def test_roundtrip_text(self):
        """Read → write → read preserves text content."""
        sups_in = JSON3Format.parse(MINIMAL_JSON3).supervisions
        output = JSON3Format.to_bytes(sups_in)
        sups_out = JSON3Format.parse(output.decode("utf-8")).supervisions

        assert len(sups_out) == len(sups_in)
        for a, b in zip(sups_in, sups_out):
            assert a.text == b.text


class TestJSON3Detection:
    """Test format detection for JSON3."""

    def test_detect_by_extension(self):
        """detect_format recognizes .json3 extension."""
        assert detect_format("video.en.json3") == "json3"

    def test_detect_from_content_wire_magic(self):
        """detect_format_from_content finds wireMagic marker."""
        assert detect_format_from_content(MINIMAL_JSON3) == "json3"

    def test_detect_from_content_events_marker(self):
        """detect_format_from_content finds events+tStartMs combination."""
        content = json.dumps({"events": [{"tStartMs": 0, "dDurationMs": 100}]})
        assert detect_format_from_content(content) == "json3"

    def test_can_read_file(self):
        """can_read accepts .json3 path."""
        assert JSON3Format.can_read("output.json3")
        assert JSON3Format.can_read("/path/to/captions.en.json3")

    def test_can_read_content(self):
        """can_read accepts JSON3 content string."""
        assert JSON3Format.can_read(MINIMAL_JSON3)

    def test_not_confused_with_json(self):
        """JSON3 is not confused with regular JSON format."""
        regular_json = json.dumps([{"text": "hello", "start": 0, "end": 1}])
        assert detect_format_from_content(regular_json) == "json"

    def test_registered_in_readers(self):
        """json3 is registered as both reader and writer."""
        assert get_reader("json3") is JSON3Format
        assert get_writer("json3") is JSON3Format


class TestJSON3Metadata:
    """Test metadata extraction."""

    def test_extract_metadata(self):
        """Extract window style and position counts."""
        meta = JSON3Format.parse(MINIMAL_JSON3).format_metadata
        assert meta["source_format"] == "json3"
        assert meta["json3_wire_magic"] == "pb3"

    def test_extract_metadata_from_file(self):
        """Extract metadata from real file."""
        meta = JSON3Format.parse(SAMPLE_JSON3).format_metadata
        assert meta["source_format"] == "json3"
        assert meta["json3_window_styles"] == 2
        assert meta["json3_window_positions"] == 2
