"""Tests for bilingual subtitle support."""

import json

import pytest

from lattifai.caption import Caption
from lattifai.caption.bilingual import merge_bilingual
from lattifai.caption.supervision import Supervision, fastcopy

# ---------------------------------------------------------------------------
# Supervision-level tests
# ---------------------------------------------------------------------------


class TestSupervisionTranslation:
    def test_translation_fields_default_none(self):
        sup = Supervision(text="Hello", start=1.0, duration=2.0)
        assert sup.translation is None
        assert sup.target_lang is None
        assert sup.has_translation is False

    def test_has_translation(self):
        sup = Supervision(text="Hello", translation="你好", start=1.0, duration=2.0)
        assert sup.has_translation is True

    def test_has_translation_empty_string(self):
        sup = Supervision(text="Hello", translation="", start=1.0, duration=2.0)
        assert sup.has_translation is False

    def test_fastcopy_preserves_translation(self):
        sup = Supervision(text="Hello", translation="你好", target_lang="zh", start=1.0, duration=2.0)
        copy = fastcopy(sup, text="Hi")
        assert copy.translation == "你好"
        assert copy.target_lang == "zh"

    def test_with_offset_preserves_translation(self):
        sup = Supervision(text="Hello", translation="你好", target_lang="zh", start=1.0, duration=2.0)
        shifted = sup.with_offset(5.0)
        assert shifted.start == 6.0
        assert shifted.translation == "你好"
        assert shifted.target_lang == "zh"

    def test_to_dict_includes_translation(self):
        sup = Supervision(text="Hello", translation="你好", target_lang="zh", start=1.0, duration=2.0)
        d = sup.to_dict()
        assert d["translation"] == "你好"
        assert d["target_lang"] == "zh"

    def test_to_dict_excludes_none_translation(self):
        sup = Supervision(text="Hello", start=1.0, duration=2.0)
        d = sup.to_dict()
        assert "translation" not in d
        assert "target_lang" not in d

    def test_from_dict_with_translation(self):
        d = {"text": "Hello", "start": 1.0, "duration": 2.0, "translation": "你好", "target_lang": "zh"}
        sup = Supervision.from_dict(d)
        assert sup.translation == "你好"
        assert sup.target_lang == "zh"
        assert sup.has_translation is True


# ---------------------------------------------------------------------------
# Caption-level tests
# ---------------------------------------------------------------------------


class TestCaptionBilingual:
    def _make_caption(self, texts, translations=None):
        sups = [Supervision(text=t, start=float(i), duration=1.0) for i, t in enumerate(texts)]
        c = Caption(supervisions=sups)
        if translations:
            c.set_translations(translations, target_lang="zh")
        return c

    def test_has_translation_false(self):
        c = self._make_caption(["Hello", "World"])
        assert c.has_translation is False

    def test_set_translations(self):
        c = self._make_caption(["Hello", "World"])
        c.set_translations(["你好", "世界"], target_lang="zh")
        assert c.has_translation is True
        assert c.target_lang == "zh"
        assert c.supervisions[0].translation == "你好"
        assert c.supervisions[1].translation == "世界"

    def test_set_translations_length_mismatch(self):
        c = self._make_caption(["Hello", "World"])
        with pytest.raises(ValueError, match="must match"):
            c.set_translations(["你好"])

    def test_strip_translations(self):
        c = self._make_caption(["Hello"], ["你好"])
        assert c.has_translation is True
        c.strip_translations()
        assert c.has_translation is False
        assert c.supervisions[0].translation is None

    def test_merge_bilingual_line_by_line(self):
        sups = [
            Supervision(text="Hello\n你好", start=0.0, duration=2.0),
            Supervision(text="World\n世界", start=2.0, duration=2.0),
        ]
        c = Caption(supervisions=sups)
        merged = merge_bilingual(
            c.supervisions, c.source_format,
            mode="line_by_line", primary_language="en", secondary_language="zh",
        )
        assert merged[0].text == "Hello"
        assert merged[0].translation == "你好"
        assert merged[1].text == "World"
        assert merged[1].translation == "世界"
        # Per-supervision language/target_lang are stamped; caller propagates
        # to the container if needed.
        assert merged[0].language == "en"
        assert merged[0].target_lang == "zh"

    def test_merge_bilingual_line_by_line_single_line(self):
        """Single-line text should have no translation."""
        sups = [Supervision(text="Hello", start=0.0, duration=2.0)]
        c = Caption(supervisions=sups)
        merged = merge_bilingual(c.supervisions, c.source_format, mode="line_by_line")
        assert merged[0].text == "Hello"
        assert merged[0].translation is None

    def test_merge_bilingual_same_timing_pairs(self):
        sups = [
            Supervision(text="Hello", start=0.0, duration=2.0),
            Supervision(text="你好", start=0.0, duration=2.0),
            Supervision(text="World", start=2.0, duration=2.0),
            Supervision(text="世界", start=2.0, duration=2.0),
        ]
        c = Caption(supervisions=sups)
        merged = merge_bilingual(
            c.supervisions, c.source_format,
            mode="same_timing_pairs", primary_language="en", secondary_language="zh",
        )
        assert len(merged) == 2
        assert merged[0].text == "Hello"
        assert merged[0].translation == "你好"
        assert merged[1].text == "World"
        assert merged[1].translation == "世界"

    def test_merge_bilingual_invalid_mode(self):
        c = self._make_caption(["Hello"])
        with pytest.raises(ValueError, match="Unknown mode"):
            merge_bilingual(c.supervisions, c.source_format, mode="invalid")


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


class TestJSONBilingual:
    def test_json_round_trip(self):
        sup = Supervision(text="Hello", translation="你好", target_lang="zh", start=1.0, duration=2.0)
        c = Caption(supervisions=[sup])
        data = c.to_bytes(output_format="json")
        parsed = json.loads(data)
        assert parsed["supervisions"][0]["translation"] == "你好"
        assert parsed["supervisions"][0]["target_lang"] == "zh"

        # Read back
        c2 = Caption.from_string(data.decode("utf-8"), format="json")
        assert c2.supervisions[0].translation == "你好"
        assert c2.supervisions[0].target_lang == "zh"

    def test_json_no_translation_no_field(self):
        sup = Supervision(text="Hello", start=1.0, duration=2.0)
        c = Caption(supervisions=[sup])
        data = json.loads(c.to_bytes(output_format="json"))
        assert "translation" not in data["supervisions"][0]
        assert "target_lang" not in data["supervisions"][0]


# ---------------------------------------------------------------------------
# SRT output
# ---------------------------------------------------------------------------


class TestSRTBilingual:
    def test_srt_bilingual_output(self):
        sup = Supervision(text="Hello world", translation="你好世界", start=1.0, duration=2.0)
        c = Caption(supervisions=[sup])
        srt = c.to_string(format="srt")
        # SRT should have both lines separated by newline
        assert "Hello world" in srt
        assert "你好世界" in srt

    def test_srt_no_translation(self):
        sup = Supervision(text="Hello world", start=1.0, duration=2.0)
        c = Caption(supervisions=[sup])
        srt = c.to_string(format="srt")
        assert "Hello world" in srt


# ---------------------------------------------------------------------------
# VTT output
# ---------------------------------------------------------------------------


class TestVTTBilingual:
    def test_vtt_bilingual_output(self):
        sup = Supervision(text="Hello world", translation="你好世界", start=1.0, duration=2.0)
        c = Caption(supervisions=[sup])
        vtt = c.to_string(format="vtt")
        assert "Hello world" in vtt
        assert "你好世界" in vtt


# ---------------------------------------------------------------------------
# LRC output
# ---------------------------------------------------------------------------


class TestLRCBilingual:
    def test_lrc_bilingual_output(self):
        sup = Supervision(text="Hello world", translation="你好世界", start=15.0, duration=5.0)
        c = Caption(supervisions=[sup])
        lrc = c.to_string(format="lrc")
        assert "Hello world" in lrc
        assert "你好世界" in lrc


# ---------------------------------------------------------------------------
# CSV/TSV output
# ---------------------------------------------------------------------------


class TestTabularBilingual:
    def test_csv_bilingual_header(self):
        sup = Supervision(text="Hello", translation="你好", start=1.0, duration=2.0)
        c = Caption(supervisions=[sup])
        csv_str = c.to_string(format="csv")
        lines = csv_str.strip().split("\n")
        assert "translation" in lines[0]
        assert "你好" in lines[1]

    def test_csv_no_translation_no_column(self):
        sup = Supervision(text="Hello", start=1.0, duration=2.0)
        c = Caption(supervisions=[sup])
        csv_str = c.to_string(format="csv")
        assert "translation" not in csv_str

    def test_tsv_bilingual_header(self):
        sup = Supervision(text="Hello", translation="你好", start=1.0, duration=2.0)
        c = Caption(supervisions=[sup])
        tsv_str = c.to_string(format="tsv")
        lines = tsv_str.strip().split("\n")
        assert "translation" in lines[0]
        assert "你好" in lines[1]


# ---------------------------------------------------------------------------
# End-to-end: set_translations -> write -> read
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_set_translations_write_json_read(self):
        c = Caption(
            supervisions=[
                Supervision(text="Hello", start=0.0, duration=2.0),
                Supervision(text="World", start=2.0, duration=2.0),
            ]
        )
        c.set_translations(["你好", "世界"], target_lang="zh")
        json_bytes = c.to_bytes(output_format="json")
        c2 = Caption.from_string(json_bytes.decode("utf-8"), format="json")
        assert c2.supervisions[0].translation == "你好"
        assert c2.supervisions[1].translation == "世界"
        assert c2.supervisions[0].has_translation is True

    def test_bilingual_srt_round_trip(self):
        """Write bilingual SRT, re-read, and verify text contains both lines."""
        sup = Supervision(text="Hello", translation="你好", start=1.0, duration=2.0)
        c = Caption(supervisions=[sup])
        srt_str = c.to_string(format="srt")
        c2 = Caption.from_string(srt_str, format="srt")
        # SRT reader sees "Hello\n你好" as single text block
        assert "Hello" in c2.supervisions[0].text
        assert "你好" in c2.supervisions[0].text
