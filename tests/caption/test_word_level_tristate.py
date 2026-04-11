#!/usr/bin/env python3
"""Tests for the tri-state word_level semantics.

word_level: Optional[bool]
  - None  → smart default (per-format behavior)
  - True  → force word-level output (warn if alignment missing)
  - False → force segment-level output (strip word data even from lossless formats)
"""

import json as json_module
import logging
from pathlib import Path

import pytest

from lattifai.caption import Caption, Supervision
from lattifai.caption.config import ASSConfig, RenderConfig
from lattifai.caption.supervision import AlignmentItem


# ── Fixtures ─────────────────────────────────────────────────────────────


def _sup_with_words(text="hello world", start=0.0, end=2.0):
    """Build a Supervision with word-level alignment."""
    words = text.split()
    per_word = (end - start) / len(words)
    return Supervision(
        text=text,
        start=start,
        duration=end - start,
        alignment={
            "word": [
                AlignmentItem(symbol=w, start=start + i * per_word, duration=per_word)
                for i, w in enumerate(words)
            ]
        },
    )


def _sup_no_words(text="hello world", start=0.0, end=2.0):
    """Build a Supervision without word-level alignment."""
    return Supervision(text=text, start=start, duration=end - start)


@pytest.fixture
def caption_with_words():
    return Caption(supervisions=[_sup_with_words("hello world", 0.0, 2.0)])


@pytest.fixture
def caption_without_words():
    return Caption(supervisions=[_sup_no_words("hello world", 0.0, 2.0)])


# ── RenderConfig API ─────────────────────────────────────────────────────


class TestRenderConfigTristate:
    def test_default_is_none(self):
        """Default value of word_level must be None (smart default)."""
        cfg = RenderConfig()
        assert cfg.word_level is None

    def test_accepts_true_false_none(self):
        """Field must accept all three states without raising."""
        RenderConfig(word_level=None)
        RenderConfig(word_level=True)
        RenderConfig(word_level=False)


# ── Smart default (None) ─────────────────────────────────────────────────


class TestSmartDefault:
    """word_level=None → format decides based on data and format-specific rules."""

    def test_srv3_none_default_segment(self, tmp_path, caption_with_words):
        """SRV3: None defaults to segment output (backward compatible).

        Word-level <s t=""> timing requires explicit word_level=True opt-in;
        the data-driven behavior is reserved for lossless serializers.
        """
        out = tmp_path / "out.srv3"
        caption_with_words.write(out, render=RenderConfig(word_level=None))
        content = out.read_text()
        body = content.split("<body>")[1]
        # No <s t="..."> word offsets in segment mode
        assert '<s t=' not in body and 's t="' not in body

    def test_srv3_true_emits_word_timing(self, tmp_path, caption_with_words):
        """SRV3: True forces per-word timing."""
        out = tmp_path / "out.srv3"
        caption_with_words.write(out, render=RenderConfig(word_level=True))
        content = out.read_text()
        assert 't="' in content

    def test_vtt_none_default_segment(self, tmp_path, caption_with_words):
        """VTT: None defaults to plain segment VTT (backward compatible)."""
        out = tmp_path / "out.vtt"
        caption_with_words.write(out, render=RenderConfig(word_level=None))
        content = out.read_text()
        assert "WEBVTT" in content
        assert "<00:" not in content
        assert "<c>" not in content

    def test_vtt_true_emits_inline_timing(self, tmp_path, caption_with_words):
        """VTT: True forces YouTube-style inline word timestamps."""
        out = tmp_path / "out.vtt"
        caption_with_words.write(out, render=RenderConfig(word_level=True))
        content = out.read_text()
        assert "<00:" in content or "<c>" in content

    def test_srt_none_segment_default(self, tmp_path, caption_with_words):
        """SRT: segment-format-by-default; None must NOT auto-expand to word cues."""
        out = tmp_path / "out.srt"
        caption_with_words.write(out, render=RenderConfig(word_level=None))
        content = out.read_text()
        # Should produce ONE cue, not two
        assert content.count("-->") == 1

    def test_lrc_none_segment_default(self, tmp_path, caption_with_words):
        """LRC: segment by default; None must NOT inline word timestamps."""
        out = tmp_path / "out.lrc"
        caption_with_words.write(out, render=RenderConfig(word_level=None))
        content = out.read_text()
        # Inline word timing in enhanced LRC uses <mm:ss.xx> markers — must be absent
        assert "<00:" not in content

    def test_ass_none_no_karaoke_segment(self, tmp_path, caption_with_words):
        """ASS: None + no karaoke_effect → segment line, no \\k tags."""
        out = tmp_path / "out.ass"
        caption_with_words.write(
            out,
            render=RenderConfig(word_level=None),
            format_config=ASSConfig(),
        )
        content = out.read_text()
        assert "\\k" not in content
        # Single Dialogue line
        assert content.count("Dialogue:") == 1

    def test_ass_none_with_karaoke_emits_k_tags(self, tmp_path, caption_with_words):
        """ASS: None + karaoke_effect set → karaoke_effect alone implies word-level."""
        out = tmp_path / "out.ass"
        caption_with_words.write(
            out,
            render=RenderConfig(word_level=None),
            format_config=ASSConfig(karaoke_effect="sweep"),
        )
        content = out.read_text()
        # \kf is the sweep karaoke tag
        assert "\\kf" in content


# ── Force word-level (True) ──────────────────────────────────────────────


class TestForceWordLevel:
    def test_srt_true_expands_to_word_cues(self, tmp_path, caption_with_words):
        """SRT: True → one cue per word."""
        out = tmp_path / "out.srt"
        caption_with_words.write(out, render=RenderConfig(word_level=True))
        content = out.read_text()
        # 2 words → 2 cues
        assert content.count("-->") == 2

    def test_lrc_true_inlines_word_timestamps(self, tmp_path, caption_with_words):
        """LRC: True → enhanced LRC with <mm:ss.xx> word markers."""
        out = tmp_path / "out.lrc"
        caption_with_words.write(out, render=RenderConfig(word_level=True))
        content = out.read_text()
        assert "<00:" in content

    def test_true_without_alignment_warns(self, tmp_path, caption_without_words, caplog):
        """word_level=True but supervision has no alignment → logger.warning."""
        out = tmp_path / "out.srt"
        with caplog.at_level(logging.WARNING, logger="lattifai.caption"):
            caption_without_words.write(out, render=RenderConfig(word_level=True))
        # At least one warning mentioning word_level / alignment / fallback
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            "word" in r.message.lower() and ("align" in r.message.lower() or "fallback" in r.message.lower())
            for r in warnings
        ), f"expected a degradation warning, got: {[r.message for r in warnings]}"


# ── Force segment-level (False) ──────────────────────────────────────────


class TestForceSegmentLevel:
    def test_json_false_strips_word_data(self, tmp_path, caption_with_words):
        """JSON: False → suppress 'words' array even when alignment is present."""
        out = tmp_path / "out.json"
        caption_with_words.write(out, render=RenderConfig(word_level=False))
        data = json_module.loads(out.read_text())
        assert "supervisions" in data
        for sup in data["supervisions"]:
            assert "words" not in sup

    def test_json_none_preserves_word_data(self, tmp_path, caption_with_words):
        """JSON: None (default) → still preserves word data (lossless)."""
        out = tmp_path / "out.json"
        caption_with_words.write(out, render=RenderConfig(word_level=None))
        data = json_module.loads(out.read_text())
        assert "words" in data["supervisions"][0]

    def test_textgrid_false_omits_words_tier(self, tmp_path, caption_with_words):
        """TextGrid: False → no 'words' tier even when alignment exists."""
        out = tmp_path / "out.TextGrid"
        caption_with_words.write(out, render=RenderConfig(word_level=False))
        content = out.read_text()
        assert '"words"' not in content

    def test_textgrid_none_emits_words_tier(self, tmp_path, caption_with_words):
        """TextGrid: None → words tier present when alignment exists."""
        out = tmp_path / "out.TextGrid"
        caption_with_words.write(out, render=RenderConfig(word_level=None))
        content = out.read_text()
        assert '"words"' in content

    def test_vtt_false_forces_standard_vtt(self, tmp_path, caption_with_words):
        """VTT: False → standard VTT even when alignment exists."""
        out = tmp_path / "out.vtt"
        caption_with_words.write(out, render=RenderConfig(word_level=False))
        content = out.read_text()
        assert "<00:" not in content
        assert "<c>" not in content

    def test_srv3_false_forces_segment(self, tmp_path, caption_with_words):
        """SRV3: False → no per-word t= even when alignment exists."""
        out = tmp_path / "out.srv3"
        caption_with_words.write(out, render=RenderConfig(word_level=False))
        content = out.read_text()
        # In segment mode, only the <p t=...> exists; inner <s> elements have no t attr
        # Count t= attributes outside <p> tags is hard with regex; check no nested t on <s>
        # Easier check: no s element with t="
        assert 't="' not in content.split("<body>")[1].split("</body>")[0].replace(
            '<p t="', ""
        ).replace('<w t="', "")

    def test_ass_false_with_karaoke_effect_disables_karaoke(self, tmp_path, caption_with_words):
        """ASS: False + karaoke_effect → karaoke is overridden, no \\k tags."""
        out = tmp_path / "out.ass"
        caption_with_words.write(
            out,
            render=RenderConfig(word_level=False),
            format_config=ASSConfig(karaoke_effect="sweep"),
        )
        content = out.read_text()
        assert "\\k" not in content


# ── Premiere unification ─────────────────────────────────────────────────


class TestPremiereUnification:
    """The .xml extension defaults to TTML in detect_format(), so these tests
    use to_bytes(output_format='premiere_xml') to target the Premiere writer
    explicitly."""

    def test_premiere_reads_render_word_level_true(self, caption_with_words):
        """Premiere XML: render.word_level=True must expand to per-word clips
        (replaces deprecated PremiereXMLConfig.use_word_level)."""
        content = caption_with_words.to_bytes(
            output_format="premiere_xml", render=RenderConfig(word_level=True)
        ).decode("utf-8")
        # 2 words → at least 2 clipitem entries
        assert content.count("<clipitem") >= 2

    def test_premiere_reads_render_word_level_none_segment(self, caption_with_words):
        """Premiere XML: render.word_level=None defaults to segment level."""
        content = caption_with_words.to_bytes(
            output_format="premiere_xml", render=RenderConfig(word_level=None)
        ).decode("utf-8")
        # 1 supervision → 1 clipitem
        assert content.count("<clipitem") == 1


# ── Backward-compat shims ────────────────────────────────────────────────


class TestDeprecatedShims:
    """Verify the deprecated entry points still work and emit DeprecationWarning."""

    def test_premiere_use_word_level_still_works(self, caption_with_words):
        from lattifai.caption.formats.nle.premiere import PremiereXMLConfig

        with pytest.warns(DeprecationWarning, match="use_word_level is deprecated"):
            cfg = PremiereXMLConfig(use_word_level=True)
        # The deprecated flag must still force per-word expansion.
        content = caption_with_words.to_bytes(
            output_format="premiere_xml", format_config=cfg
        ).decode("utf-8")
        assert content.count("<clipitem") >= 2

    def test_fcpxml_write_with_word_level_still_works(self, tmp_path, caption_with_words):
        from lattifai.caption.formats.nle.fcpxml import FCPXMLConfig, FCPXMLWriter

        # FCPXMLConfig defaults to use_bundle=True (writes a .fcpxmld directory).
        out = tmp_path / "out"
        with pytest.warns(DeprecationWarning, match="write_with_word_level"):
            written = FCPXMLWriter.write_with_word_level(
                caption_with_words.supervisions, out, FCPXMLConfig(use_bundle=False)
            )
        assert Path(written).exists()


# ── Edge cases caught in code review ─────────────────────────────────────


class TestEmptyWordList:
    """Empty alignment["word"] = [] must be treated as no data, not as data."""

    def _sup_with_empty_word_list(self):
        return Supervision(
            text="hello world",
            start=0.0,
            duration=2.0,
            alignment={"word": []},
        )

    def test_lrc_does_not_crash_on_empty_word_list(self, tmp_path):
        """Regression: previously _use_word went True on [] then word_items[0] crashed."""
        cap = Caption(supervisions=[self._sup_with_empty_word_list()])
        out = tmp_path / "out.lrc"
        cap.write(out, render=RenderConfig(word_level=True))
        # Should fall back to plain LRC output
        assert "[00:" in out.read_text()

    def test_srt_does_not_drop_segments_on_empty_word_list(self, tmp_path):
        """Empty word list should not silently drop the segment in SRT/SSA/ASS expansion."""
        cap = Caption(supervisions=[self._sup_with_empty_word_list()])
        out = tmp_path / "out.srt"
        cap.write(out, render=RenderConfig(word_level=True))
        content = out.read_text()
        # Original segment text must still appear (not silently dropped)
        assert "hello world" in content

    def test_json_does_not_emit_empty_words_array(self, tmp_path):
        """Empty alignment word list should not produce 'words': [] in JSON output."""
        cap = Caption(supervisions=[self._sup_with_empty_word_list()])
        out = tmp_path / "out.json"
        cap.write(out)
        data = json_module.loads(out.read_text())
        for sup in data["supervisions"]:
            assert "words" not in sup or sup["words"]


class TestPartialBatchWarning:
    """word_level=True with mixed alignment must emit a partial-fallback warning."""

    def test_partial_batch_warns(self, tmp_path, caplog):
        sup_with = _sup_with_words("hello world", 0.0, 2.0)
        sup_without = _sup_no_words("missing alignment", 3.0, 5.0)
        cap = Caption(supervisions=[sup_with, sup_without])

        out = tmp_path / "out.srt"
        with caplog.at_level(logging.WARNING, logger="lattifai.caption"):
            cap.write(out, render=RenderConfig(word_level=True))

        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            "1/2" in m or "lack word alignment" in m for m in warnings
        ), f"expected partial-fallback warning, got: {warnings}"


class TestASSKaraokeWarnings:
    """ASS karaoke must warn instead of silently producing segment output when
    word alignment is missing."""

    def test_karaoke_no_word_data_warns(self, tmp_path, caption_without_words, caplog):
        """karaoke_effect set + no alignment → warning."""
        out = tmp_path / "out.ass"
        with caplog.at_level(logging.WARNING, logger="lattifai.caption"):
            caption_without_words.write(
                out, format_config=ASSConfig(karaoke_effect="sweep")
            )
        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("karaoke_effect" in m and "no word alignment" in m for m in warnings), (
            f"expected ass karaoke fallback warning, got: {warnings}"
        )

    def test_karaoke_partial_data_warns(self, tmp_path, caplog):
        """karaoke_effect set + mixed batch → partial warning."""
        sup_with = _sup_with_words("hello world", 0.0, 2.0)
        sup_without = _sup_no_words("missing alignment", 3.0, 5.0)
        cap = Caption(supervisions=[sup_with, sup_without])

        out = tmp_path / "out.ass"
        with caplog.at_level(logging.WARNING, logger="lattifai.caption"):
            cap.write(out, format_config=ASSConfig(karaoke_effect="sweep"))

        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            "karaoke_effect" in m and ("1/2" in m or "lack word alignment" in m)
            for m in warnings
        ), f"expected ass karaoke partial warning, got: {warnings}"
