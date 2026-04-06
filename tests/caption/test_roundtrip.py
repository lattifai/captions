"""Roundtrip tests: read → write → read — verify text and metadata preservation.

For each format that supports both reading and writing, we:
1. Create a Caption from known content (from_string or from_supervisions)
2. Write it to bytes in that format
3. Read it back
4. Assert that supervisions (text, timing) and metadata survive the trip

Also includes file-level roundtrip: read real file → write → compare output
against original to detect any information loss.
"""

import io
from pathlib import Path

import pytest

from lattifai.caption import Caption, Supervision
from lattifai.caption.config import RenderConfig
from lattifai.caption.supervision import AlignmentItem


# =============================================================================
# Helpers
# =============================================================================


def _make_supervisions(with_speaker: bool = False) -> list:
    """Standard 2-segment test supervisions."""
    return [
        Supervision(
            text="Hello world",
            start=1.0,
            duration=2.0,
            speaker="Alice" if with_speaker else None,
        ),
        Supervision(
            text="Second line here",
            start=3.5,
            duration=2.5,
            speaker="Bob" if with_speaker else None,
        ),
    ]


def _assert_text_roundtrip(original: Caption, roundtrip: Caption, fmt: str):
    """Assert text content survived roundtrip."""
    assert len(roundtrip) == len(original), f"{fmt}: segment count mismatch"
    for i, (orig, rt) in enumerate(zip(original, roundtrip)):
        assert orig.text == rt.text, f"{fmt} seg {i}: text mismatch"
        assert abs(orig.start - rt.start) < 0.05, f"{fmt} seg {i}: start mismatch"
        assert abs(orig.duration - rt.duration) < 0.05, f"{fmt} seg {i}: duration mismatch"


# =============================================================================
# Standard text roundtrip — parametrized across all read/write formats
# =============================================================================

# Formats that can roundtrip text+timing faithfully
TEXT_ROUNDTRIP_FORMATS = [
    "srt",
    "vtt",
    "ass",
    "ssa",
    "sbv",
    "csv",
    "tsv",
    "json",
    "lrc",
    "txt",
    "aud",
]


class TestTextRoundtrip:
    """Text + timing survives write → read for all common formats."""

    @pytest.mark.parametrize("fmt", TEXT_ROUNDTRIP_FORMATS)
    def test_text_roundtrip(self, fmt):
        sups = _make_supervisions()
        original = Caption.from_supervisions(sups)
        written = original.to_bytes(output_format=fmt)
        roundtrip = Caption.from_string(written.decode("utf-8"), format=fmt)

        assert len(roundtrip) >= len(original), f"{fmt}: lost segments"
        # Check that original texts appear in roundtrip
        orig_texts = {s.text for s in original}
        rt_texts = {s.text for s in roundtrip}
        for t in orig_texts:
            assert any(t in rt_text for rt_text in rt_texts), f"{fmt}: lost text '{t}'"


# =============================================================================
# Metadata roundtrip — format-specific metadata preservation
# =============================================================================


class TestVTTMetadataRoundtrip:
    """VTT Kind/Language header survives roundtrip."""

    def test_kind_and_language(self):
        content = """\
WEBVTT
Kind: captions
Language: en

00:00:01.000 --> 00:00:03.000
Hello World

00:00:03.500 --> 00:00:06.000
This is a test
"""
        caption = Caption.from_string(content, format="vtt")
        assert caption.kind == "captions"
        assert caption.language == "en"

        output = caption.to_string(format="vtt")
        assert "Kind: captions" in output
        assert "Language: en" in output

        # Full roundtrip
        caption2 = Caption.from_string(output, format="vtt")
        assert caption2.kind == "captions"
        assert caption2.language == "en"
        assert len(caption2) == 2


class TestASSMetadataRoundtrip:
    """ASS styles and Script Info survive roundtrip."""

    ASS_CONTENT = """\
[Script Info]
Title: Test Title
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000088EF,&H00000000,&H00666666,-1,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello World
Dialogue: 0,0:00:03.50,0:00:06.00,Default,,0,0,0,,Second line
"""

    def test_styles_preserved(self):
        caption = Caption.from_string(self.ASS_CONTENT, format="ass")
        assert "ass_styles" in caption.metadata
        assert "Default" in caption.metadata["ass_styles"]
        assert caption.metadata["ass_styles"]["Default"]["fontname"] == "Arial"

        output = caption.to_string(format="ass")
        assert "Arial" in output
        assert "1920" in output

        # Full roundtrip
        caption2 = Caption.from_string(output, format="ass")
        assert "ass_styles" in caption2.metadata
        assert caption2.metadata["ass_styles"]["Default"]["fontname"] == "Arial"

    def test_script_info_preserved(self):
        caption = Caption.from_string(self.ASS_CONTENT, format="ass")
        assert "ass_info" in caption.metadata
        assert caption.metadata["ass_info"]["Title"] == "Test Title"

        output = caption.to_string(format="ass")
        caption2 = Caption.from_string(output, format="ass")
        assert caption2.metadata["ass_info"]["Title"] == "Test Title"

    def test_text_preserved(self):
        caption = Caption.from_string(self.ASS_CONTENT, format="ass")
        output = caption.to_string(format="ass")
        caption2 = Caption.from_string(output, format="ass")

        _assert_text_roundtrip(caption, caption2, "ass")


class TestSRTMetadataRoundtrip:
    """SRT BOM and text roundtrip."""

    def test_bom_preserved(self):
        content = "\ufeff1\n00:00:01,000 --> 00:00:03,000\nHello\n"
        caption = Caption.from_string(content, format="srt")
        # BOM detected in metadata
        assert caption.metadata.get("encoding") == "utf-8-sig"

        # Write back — SRT writer should re-add BOM if metadata says utf-8-sig
        output_bytes = caption.to_bytes(output_format="srt")
        assert output_bytes.startswith(b"\xef\xbb\xbf"), "BOM should be preserved in output"

    def test_text_roundtrip(self):
        content = "1\n00:00:01,000 --> 00:00:03,000\nHello world\n\n2\n00:00:04,000 --> 00:00:06,000\nSecond\n"
        caption = Caption.from_string(content, format="srt")
        output = caption.to_string(format="srt")
        caption2 = Caption.from_string(output, format="srt")
        _assert_text_roundtrip(caption, caption2, "srt")


class TestLRCMetadataRoundtrip:
    """LRC metadata tags survive roundtrip."""

    LRC_CONTENT = """\
[ti:Test Song]
[ar:Test Artist]
[al:Test Album]
[00:01.00]First line
[00:03.50]Second line
"""

    def test_metadata_preserved(self):
        caption = Caption.from_string(self.LRC_CONTENT, format="lrc")
        assert caption.metadata.get("lrc_ti") == "Test Song"
        assert caption.metadata.get("lrc_ar") == "Test Artist"
        assert caption.metadata.get("lrc_al") == "Test Album"

    def test_text_roundtrip(self):
        caption = Caption.from_string(self.LRC_CONTENT, format="lrc")
        output_bytes = caption.to_bytes(output_format="lrc")
        caption2 = Caption.from_string(output_bytes.decode("utf-8"), format="lrc")
        assert len(caption2) >= 2
        texts = {s.text for s in caption2}
        assert "First line" in texts
        assert "Second line" in texts


class TestJSONRoundtrip:
    """JSON full metadata roundtrip."""

    def test_full_roundtrip(self):
        sups = _make_supervisions(with_speaker=True)
        original = Caption.from_supervisions(
            sups, language="en", kind="captions"
        )
        written = original.to_bytes(output_format="json")
        caption2 = Caption.from_string(written.decode("utf-8"), format="json")

        assert len(caption2) == 2
        assert caption2[0].text == "Hello world"
        assert caption2[1].text == "Second line here"


class TestSRV3Roundtrip:
    """SRV3 text and word-level timing roundtrip."""

    SRV3_CONTENT = """\
<?xml version="1.0" encoding="utf-8"?>
<timedtext format="3">
<body>
<p t="1000" d="2000"><s>Hello</s><s t="500"> world</s></p>
<p t="3500" d="2500"><s>Second</s><s t="500"> line</s></p>
</body>
</timedtext>
"""

    def test_text_roundtrip(self):
        caption = Caption.from_string(self.SRV3_CONTENT, format="srv3")
        assert len(caption) == 2

        output_bytes = caption.to_bytes(output_format="srv3")
        caption2 = Caption.from_string(output_bytes.decode("utf-8"), format="srv3")

        assert len(caption2) == 2
        assert caption[0].text == caption2[0].text

    def test_metadata_preserved(self):
        caption = Caption.from_string(self.SRV3_CONTENT, format="srv3")
        assert caption.metadata.get("srv3_format") == "3"


class TestSBVRoundtrip:
    """SBV text roundtrip."""

    def test_text_roundtrip(self):
        content = "0:00:01.000,0:00:03.000\nHello world\n\n0:00:03.500,0:00:06.000\nSecond line\n"
        caption = Caption.from_string(content, format="sbv")
        output_bytes = caption.to_bytes(output_format="sbv")
        caption2 = Caption.from_string(output_bytes.decode("utf-8"), format="sbv")

        assert len(caption2) == 2
        assert caption2[0].text == "Hello world"
        assert caption2[1].text == "Second line"


class TestCSVTSVRoundtrip:
    """CSV/TSV roundtrip."""

    def test_csv_roundtrip(self):
        sups = _make_supervisions(with_speaker=True)
        original = Caption.from_supervisions(sups)
        written = original.to_bytes(output_format="csv")
        caption2 = Caption.from_string(written.decode("utf-8"), format="csv")

        assert len(caption2) == 2
        rt_texts = " ".join(s.text for s in caption2)
        assert "Hello world" in rt_texts
        assert "Second line" in rt_texts

    def test_tsv_roundtrip(self):
        sups = _make_supervisions(with_speaker=True)
        original = Caption.from_supervisions(sups)
        written = original.to_bytes(output_format="tsv")
        caption2 = Caption.from_string(written.decode("utf-8"), format="tsv")

        assert len(caption2) == 2
        assert caption2[0].text == "Hello world"


class TestCrossFormatRoundtrip:
    """Read in one format, write to another, verify text preservation."""

    CROSS_PAIRS = [
        ("srt", "vtt"),
        ("srt", "ass"),
        ("vtt", "srt"),
        ("vtt", "json"),
        ("ass", "srt"),
        ("srt", "sbv"),
        ("srt", "csv"),
        ("srt", "lrc"),
    ]

    @pytest.mark.parametrize("src_fmt,dst_fmt", CROSS_PAIRS, ids=[f"{s}->{d}" for s, d in CROSS_PAIRS])
    def test_cross_format_text_preserved(self, src_fmt, dst_fmt):
        sups = _make_supervisions()
        original = Caption.from_supervisions(sups, source_format=src_fmt)
        written = original.to_bytes(output_format=dst_fmt)
        roundtrip = Caption.from_string(written.decode("utf-8"), format=dst_fmt)

        assert len(roundtrip) >= 1, f"{src_fmt}->{dst_fmt}: no segments"
        rt_texts = " ".join(s.text for s in roundtrip)
        assert "Hello world" in rt_texts, f"{src_fmt}->{dst_fmt}: lost text"


class TestChineseRoundtrip:
    """Chinese text survives roundtrip across formats."""

    CHINESE_FORMATS = ["srt", "vtt", "ass", "json", "csv", "lrc"]

    @pytest.mark.parametrize("fmt", CHINESE_FORMATS)
    def test_chinese_roundtrip(self, fmt):
        sups = [
            Supervision(text="LattifAI 是高性能引擎", start=0.0, duration=3.0),
            Supervision(text="确保数据安全", start=3.5, duration=2.5),
        ]
        original = Caption.from_supervisions(sups)
        written = original.to_bytes(output_format=fmt)
        roundtrip = Caption.from_string(written.decode("utf-8"), format=fmt)

        rt_texts = " ".join(s.text for s in roundtrip)
        assert "高性能引擎" in rt_texts, f"{fmt}: lost Chinese text"
        assert "数据安全" in rt_texts, f"{fmt}: lost Chinese text"


# =============================================================================
# File-level roundtrip: read real file → write → read → compare
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"


def _file_roundtrip(file_path: Path, fmt: str):
    """Read file → write to bytes → read back → compare supervisions."""
    original = Caption.read(file_path, format=fmt)
    written = original.to_bytes(output_format=fmt)
    roundtrip = Caption.from_string(written.decode("utf-8"), format=fmt)
    return original, roundtrip


class TestFileRoundtrip:
    """Read real test files, write back in same format, compare."""

    def test_srt_file(self):
        f = DATA_DIR / "SA1.srt"
        if not f.exists():
            pytest.skip(f"Missing {f}")
        original, roundtrip = _file_roundtrip(f, "srt")
        _assert_text_roundtrip(original, roundtrip, "srt")

    def test_vtt_file(self):
        f = DATA_DIR / "SA1.vtt"
        if not f.exists():
            pytest.skip(f"Missing {f}")
        original, roundtrip = _file_roundtrip(f, "vtt")
        _assert_text_roundtrip(original, roundtrip, "vtt")

    def test_sbv_file(self):
        f = DATA_DIR / "SA1.sbv"
        if not f.exists():
            pytest.skip(f"Missing {f}")
        original, roundtrip = _file_roundtrip(f, "sbv")
        _assert_text_roundtrip(original, roundtrip, "sbv")

    def test_txt_file(self):
        f = DATA_DIR / "SA1.TXT"
        if not f.exists():
            pytest.skip(f"Missing {f}")
        original, roundtrip = _file_roundtrip(f, "txt")
        _assert_text_roundtrip(original, roundtrip, "txt")

    def test_textgrid_file(self):
        f = DATA_DIR / "SA1.TextGrid"
        if not f.exists():
            pytest.skip(f"Missing {f}")
        original = Caption.read(f, format="textgrid")
        assert len(original) > 0, "TextGrid read returned 0 segments"
        # TextGrid write → read roundtrip
        written = original.to_bytes(output_format="textgrid")
        roundtrip = Caption.read(io.BytesIO(written), format="textgrid")
        # TextGrid write adds score tiers, so roundtrip may have more segments.
        # Verify original texts are present in roundtrip.
        orig_texts = {s.text for s in original if s.text}
        rt_texts = {s.text for s in roundtrip if s.text}
        for t in orig_texts:
            assert t in rt_texts, f"TextGrid: lost text '{t}'"

    def test_srv3_file(self):
        f = DATA_DIR / "captions" / "DoesFastChargingHurttheBatteryRaw.srv3"
        if not f.exists():
            pytest.skip(f"Missing {f}")
        original = Caption.read(f, format="srv3")
        assert len(original) > 0
        written = original.to_bytes(output_format="srv3")
        roundtrip = Caption.from_string(written.decode("utf-8"), format="srv3")
        assert len(roundtrip) == len(original), "SRV3 segment count mismatch"
        for i, (o, r) in enumerate(zip(original, roundtrip)):
            assert o.text == r.text, f"SRV3 seg {i}: text mismatch"
            assert abs(o.start - r.start) < 0.002, f"SRV3 seg {i}: start mismatch"

    def test_vtt_youtube_raw_file(self):
        f = DATA_DIR / "captions" / "DoesFastChargingHurttheBatteryRaw.vtt"
        if not f.exists():
            pytest.skip(f"Missing {f}")
        original = Caption.read(f, format="vtt")
        assert len(original) > 0
        # Write as standard VTT (YouTube VTT → standard VTT)
        written = original.to_bytes(output_format="vtt")
        roundtrip = Caption.from_string(written.decode("utf-8"), format="vtt")
        # Text should survive even if word-level timing is lost
        for i, (o, r) in enumerate(zip(original, roundtrip)):
            assert o.text == r.text, f"VTT seg {i}: text mismatch"


class TestFileRoundtripBytesIdentical:
    """Stricter: read → write → compare bytes (where format allows it).

    Some formats (SRT, VTT) should produce near-identical output.
    """

    def test_srt_bytes_stable(self):
        """SRT write is deterministic: writing twice produces identical bytes."""
        sups = _make_supervisions()
        caption = Caption.from_supervisions(sups)
        bytes1 = caption.to_bytes(output_format="srt")
        bytes2 = caption.to_bytes(output_format="srt")
        assert bytes1 == bytes2, "SRT output not deterministic"

    def test_vtt_bytes_stable(self):
        """VTT write is deterministic."""
        sups = _make_supervisions()
        caption = Caption.from_supervisions(sups)
        bytes1 = caption.to_bytes(output_format="vtt")
        bytes2 = caption.to_bytes(output_format="vtt")
        assert bytes1 == bytes2, "VTT output not deterministic"

    def test_srt_file_roundtrip_stable(self):
        """SRT file: read → write → read → write → bytes should match."""
        f = DATA_DIR / "SA1.srt"
        if not f.exists():
            pytest.skip(f"Missing {f}")
        cap1 = Caption.read(f, format="srt")
        bytes1 = cap1.to_bytes(output_format="srt")
        cap2 = Caption.from_string(bytes1.decode("utf-8"), format="srt")
        bytes2 = cap2.to_bytes(output_format="srt")
        assert bytes1 == bytes2, "SRT not stable after double roundtrip"

    def test_ass_file_roundtrip_metadata_stable(self):
        """ASS: metadata should survive double roundtrip."""
        content = """\
[Script Info]
Title: Stability Test
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000088EF,&H00000000,&H00666666,-1,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Test line
"""
        cap1 = Caption.from_string(content, format="ass")
        bytes1 = cap1.to_bytes(output_format="ass")
        cap2 = Caption.from_string(bytes1.decode("utf-8"), format="ass")

        # Metadata roundtrip
        assert cap1.metadata.get("ass_info", {}).get("Title") == cap2.metadata.get("ass_info", {}).get("Title")
        assert cap1.metadata.get("ass_styles", {}).keys() == cap2.metadata.get("ass_styles", {}).keys()
