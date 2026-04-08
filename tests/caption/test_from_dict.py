"""Tests for Caption.from_dict(), resolve_format_config(), and Supervision.from_dict() edge cases.

Covers the new APIs added to support CaptionData round-trip between backend read/write.
"""

import pytest

from lattifai.caption import Caption, Supervision, resolve_format_config
from lattifai.caption.config import ASSConfig, LRCConfig, StandardizationConfig
from lattifai.caption.supervision import AlignmentItem


# =============================================================================
# Caption.from_dict — inverse of to_dict
# =============================================================================


class TestCaptionFromDict:
    """Caption.from_dict() reconstructs Caption from to_dict() output."""

    def test_roundtrip_basic(self):
        """to_dict → from_dict preserves all fields."""
        original = Caption(
            supervisions=[
                Supervision(text="Hello", start=0.0, duration=2.0),
                Supervision(text="World", start=2.5, duration=1.5),
            ],
            language="en",
            target_lang="zh",
            kind="subtitles",
            source_format="srt",
            metadata={"title": "Test"},
        )
        d = original.to_dict()
        restored = Caption.from_dict(d)

        assert len(restored) == 2
        assert restored.language == "en"
        assert restored.target_lang == "zh"
        assert restored.kind == "subtitles"
        assert restored.source_format == "srt"
        assert restored.metadata == {"title": "Test"}
        assert restored[0].text == "Hello"
        assert restored[1].text == "World"

    def test_roundtrip_with_alignment(self):
        """Alignment wire format [[symbol, start, dur]] is deserialized to AlignmentItem."""
        d = {
            "supervisions": [
                {
                    "id": "1",
                    "recording_id": "test",
                    "start": 0.0,
                    "duration": 2.0,
                    "text": "Hello world",
                    "alignment": {
                        "word": [
                            ["Hello", 0.0, 0.8],
                            ["world", 0.9, 1.0],
                        ]
                    },
                }
            ],
        }
        caption = Caption.from_dict(d)
        assert len(caption) == 1
        alignment = caption[0].alignment
        assert alignment is not None
        assert len(alignment["word"]) == 2
        assert isinstance(alignment["word"][0], AlignmentItem)
        assert alignment["word"][0].symbol == "Hello"
        assert alignment["word"][1].start == 0.9

    def test_empty_supervisions(self):
        """Empty supervisions list creates empty Caption."""
        caption = Caption.from_dict({"supervisions": []})
        assert len(caption) == 0
        assert caption.metadata == {}

    def test_missing_optional_fields(self):
        """Missing optional fields default to None/empty."""
        caption = Caption.from_dict({
            "supervisions": [{"text": "Hi", "start": 0.0, "duration": 1.0}],
        })
        assert caption.language is None
        assert caption.target_lang is None
        assert caption.kind is None
        assert caption.source_format is None
        assert caption.metadata == {}
        assert len(caption) == 1

    def test_ignores_computed_fields(self):
        """Computed fields (duration, num_segments, speakers) are ignored."""
        caption = Caption.from_dict({
            "supervisions": [{"text": "Hi", "start": 0.0, "duration": 1.0}],
            "duration": 999.0,  # should be ignored
            "num_segments": 999,  # should be ignored
            "speakers": ["Alice"],  # should be ignored
        })
        assert caption.duration == 1.0  # computed from supervision, not 999
        assert len(caption) == 1

    def test_source_path_preserved(self):
        """source_path is preserved through from_dict round-trip."""
        original = Caption(
            supervisions=[Supervision(text="Hi", start=0.0, duration=1.0)],
            source_path="/path/to/input.srt",
        )
        d = original.to_dict()
        assert d["source_path"] == "/path/to/input.srt"
        restored = Caption.from_dict(d)
        assert restored.source_path == "/path/to/input.srt"

    def test_metadata_none_becomes_empty_dict(self):
        """metadata=None in input should become {}."""
        caption = Caption.from_dict({
            "supervisions": [],
            "metadata": None,
        })
        assert caption.metadata == {}

    def test_chinese_text_preserved(self):
        """CJK text is preserved through roundtrip."""
        original = Caption(
            supervisions=[
                Supervision(text="你好世界", start=0.0, duration=2.0),
                Supervision(text="こんにちは", start=2.5, duration=1.5),
            ],
            language="zh",
        )
        restored = Caption.from_dict(original.to_dict())
        assert restored[0].text == "你好世界"
        assert restored[1].text == "こんにちは"

    def test_ass_metadata_roundtrip(self):
        """ASS metadata (ass_info, ass_styles) survives roundtrip."""
        meta = {
            "ass_info": {"Title": "Test", "ScriptType": "v4.00+"},
            "ass_styles": {"Default": {"fontname": "Arial", "fontsize": 20}},
        }
        caption = Caption.from_dict({
            "supervisions": [{"text": "Hello", "start": 0.0, "duration": 1.0}],
            "source_format": "ass",
            "metadata": meta,
        })
        assert caption.metadata["ass_info"]["Title"] == "Test"
        assert caption.metadata["ass_styles"]["Default"]["fontname"] == "Arial"

    def test_supervision_objects_passthrough(self):
        """Pre-constructed Supervision objects are accepted without conversion."""
        sup = Supervision(text="Hello", start=0.0, duration=1.0)
        caption = Caption.from_dict({"supervisions": [sup]})
        assert caption[0] is sup

    def test_from_dict_then_write(self):
        """from_dict output can be written to bytes without error."""
        d = {
            "supervisions": [
                {"id": "1", "recording_id": "t", "start": 0.0, "duration": 2.0, "text": "Hello"},
                {"id": "2", "recording_id": "t", "start": 2.5, "duration": 1.5, "text": "World"},
            ],
            "language": "en",
        }
        caption = Caption.from_dict(d)
        srt_bytes = caption.write(path=None, _output_format="srt")
        assert b"Hello" in srt_bytes
        assert b"World" in srt_bytes

    def test_from_dict_with_alignment_then_write_json(self):
        """from_dict with alignment wire format can write to JSON."""
        d = {
            "supervisions": [
                {
                    "id": "1",
                    "recording_id": "t",
                    "start": 0.0,
                    "duration": 2.0,
                    "text": "Hello world",
                    "alignment": {"word": [["Hello", 0.0, 0.8], ["world", 0.9, 1.0]]},
                }
            ],
        }
        caption = Caption.from_dict(d)
        json_bytes = caption.write(path=None, _output_format="json")
        assert b"Hello" in json_bytes

    def test_from_dict_with_standardization(self):
        """from_dict output works with standardization (margin adjustment)."""
        d = {
            "supervisions": [
                {"id": "1", "recording_id": "t", "start": 0.0, "duration": 2.0, "text": "Hello"},
            ],
        }
        caption = Caption.from_dict(d)
        std = StandardizationConfig(start_margin=0.05, end_margin=0.05)
        content = caption.write(path=None, standardization=std, _output_format="srt")
        assert b"Hello" in content


# =============================================================================
# resolve_format_config
# =============================================================================


class TestResolveFormatConfig:
    """resolve_format_config() converts dict to format-specific config dataclass."""

    def test_ass_config(self):
        cfg = resolve_format_config("ass", {"font_name": "Arial", "font_size": 24})
        assert isinstance(cfg, ASSConfig)
        assert cfg.font_name == "Arial"
        assert cfg.font_size == 24

    def test_ssa_maps_to_ass_config(self):
        cfg = resolve_format_config("ssa", {"font_name": "Noto Sans"})
        assert isinstance(cfg, ASSConfig)
        assert cfg.font_name == "Noto Sans"

    def test_lrc_config(self):
        cfg = resolve_format_config("lrc", {"precision": "centisecond"})
        assert isinstance(cfg, LRCConfig)
        assert cfg.precision == "centisecond"

    def test_ttml_config(self):
        from lattifai.caption.formats.ttml import TTMLConfig
        cfg = resolve_format_config("ttml", {"default_region": True})
        assert isinstance(cfg, TTMLConfig)

    def test_imsc1_maps_to_ttml(self):
        from lattifai.caption.formats.ttml import TTMLConfig
        cfg = resolve_format_config("imsc1", {"default_region": True})
        assert isinstance(cfg, TTMLConfig)

    def test_none_dict_returns_none(self):
        assert resolve_format_config("ass", None) is None

    def test_empty_dict_returns_none(self):
        assert resolve_format_config("ass", {}) is None

    def test_unknown_format_returns_none(self):
        assert resolve_format_config("srt", {"key": "value"}) is None

    def test_premiere_xml_config(self):
        from lattifai.caption.formats.nle.premiere import PremiereXMLConfig
        cfg = resolve_format_config("premiere_xml", {"fps": 24.0})
        assert isinstance(cfg, PremiereXMLConfig)

    def test_fcpxml_config(self):
        from lattifai.caption.formats.nle.fcpxml import FCPXMLConfig
        cfg = resolve_format_config("fcpxml", {"fps": 24.0})
        assert isinstance(cfg, FCPXMLConfig)

    def test_avid_ds_config(self):
        from lattifai.caption.formats.nle.avid import AvidDSConfig
        cfg = resolve_format_config("avid_ds", {"fps": 25.0})
        assert isinstance(cfg, AvidDSConfig)

    def test_audition_csv_config(self):
        from lattifai.caption.formats.nle.audition import AuditionCSVConfig
        cfg = resolve_format_config("audition_csv", {"sample_rate": 48000})
        assert isinstance(cfg, AuditionCSVConfig)

    def test_edimarker_csv_config(self):
        from lattifai.caption.formats.nle.audition import EdiMarkerConfig
        cfg = resolve_format_config("edimarker_csv", {"marker_prefix": "PT"})
        assert isinstance(cfg, EdiMarkerConfig)
        assert cfg.marker_prefix == "PT"


# =============================================================================
# Supervision.from_dict — alignment=None edge case
# =============================================================================


class TestSupervisionFromDictEdgeCases:
    """Supervision.from_dict() handles alignment edge cases."""

    def test_alignment_none(self):
        """alignment=None (from Pydantic model_dump) should not crash."""
        sup = Supervision.from_dict({
            "text": "Hello",
            "start": 0.0,
            "duration": 1.0,
            "alignment": None,
        })
        assert sup.text == "Hello"
        assert sup.alignment is None

    def test_alignment_empty_dict(self):
        """alignment={} should be kept as empty dict."""
        sup = Supervision.from_dict({
            "text": "Hello",
            "start": 0.0,
            "duration": 1.0,
            "alignment": {},
        })
        assert sup.alignment == {}

    def test_alignment_wire_format(self):
        """Standard wire format is deserialized to AlignmentItem."""
        sup = Supervision.from_dict({
            "text": "Hello world",
            "start": 0.0,
            "duration": 2.0,
            "alignment": {
                "word": [["Hello", 0.0, 0.8], ["world", 0.9, 1.0]],
            },
        })
        assert len(sup.alignment["word"]) == 2
        assert isinstance(sup.alignment["word"][0], AlignmentItem)

    def test_alignment_with_score(self):
        """Wire format with optional score [symbol, start, dur, score]."""
        sup = Supervision.from_dict({
            "text": "Hello",
            "start": 0.0,
            "duration": 1.0,
            "alignment": {
                "word": [["Hello", 0.0, 0.8, 0.95]],
            },
        })
        assert sup.alignment["word"][0].score == 0.95

    def test_no_alignment_key(self):
        """Missing alignment key is fine — defaults to None."""
        sup = Supervision.from_dict({
            "text": "Hello",
            "start": 0.0,
            "duration": 1.0,
        })
        assert sup.alignment is None

    def test_unknown_fields_ignored(self):
        """Unknown fields are silently dropped for forward-compat."""
        sup = Supervision.from_dict({
            "text": "Hello",
            "start": 0.0,
            "duration": 1.0,
            "future_field": "should_be_ignored",
        })
        assert sup.text == "Hello"
        assert not hasattr(sup, "future_field")
