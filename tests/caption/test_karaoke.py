"""Comprehensive karaoke tests: config, ASS tags, TTML timing, cross-format integration."""

import re

import pytest

from lattifai.caption.config import CaptionFonts, CaptionStyle, KaraokeConfig
from lattifai.caption.formats import get_writer
from lattifai.caption.formats.pysubs2 import ASSFormat
from lattifai.caption.formats.ttml import TTMLFormat
from lattifai.caption.supervision import AlignmentItem, Supervision


# =============================================================================
# Shared fixtures
# =============================================================================


def _sup(text="Hello world", start=15.2, duration=3.3, words=None):
    """Shorthand for creating a supervision with word alignment."""
    if words is None:
        words = [
            AlignmentItem(symbol="Hello", start=15.2, duration=0.45),
            AlignmentItem(symbol="world", start=15.65, duration=2.85),
        ]
    return Supervision(text=text, start=start, duration=duration, alignment={"word": words})


@pytest.fixture
def sup_hello_world():
    return _sup()


@pytest.fixture
def sup_hello_beautiful_world():
    return _sup(
        text="Hello beautiful world",
        words=[
            AlignmentItem(symbol="Hello", start=15.2, duration=0.45),
            AlignmentItem(symbol="beautiful", start=15.65, duration=0.75),
            AlignmentItem(symbol="world", start=16.4, duration=2.1),
        ],
    )


@pytest.fixture
def sup_with_gaps():
    """Supervision where word timestamps have gaps (10.5→11.3, 11.5→12.7, 13.0→14.2)."""
    return Supervision(
        text="Hello beautiful world",
        start=10.0,
        duration=5.0,
        alignment={
            "word": [
                AlignmentItem(symbol="Hello", start=10.5, duration=0.8),
                AlignmentItem(symbol="beautiful", start=11.5, duration=1.2),
                AlignmentItem(symbol="world", start=13.0, duration=1.2),
            ]
        },
    )


# =============================================================================
# 1. Config: CaptionStyle, KaraokeConfig, color schemes
# =============================================================================


class TestCaptionFonts:
    def test_western_fonts(self):
        assert CaptionFonts.ARIAL == "Arial"
        assert CaptionFonts.IMPACT == "Impact"

    def test_cjk_fonts(self):
        assert CaptionFonts.NOTO_SANS_SC == "Noto Sans SC"
        assert CaptionFonts.NOTO_SANS_JP == "Noto Sans JP"


class TestCaptionStyleDefaults:
    def test_defaults(self):
        style = CaptionStyle()
        assert style.primary_color == "#FFFFFF"
        assert style.secondary_color == "#00FFFF"
        assert style.font_name == CaptionFonts.ARIAL
        assert style.font_size == 48
        assert style.bold is False

    def test_custom(self):
        style = CaptionStyle(primary_color="#FF00FF", font_name=CaptionFonts.NOTO_SANS_SC, font_size=56, bold=True)
        assert style.primary_color == "#FF00FF"
        assert style.font_name == "Noto Sans SC"
        assert style.font_size == 56


class TestKaraokeConfigDefaults:
    def test_defaults(self):
        config = KaraokeConfig()
        assert config.enabled is False
        assert config.effect == "sweep"
        assert config.lrc_precision == "millisecond"
        assert config.ttml_timing_mode == "Word"

    def test_effects(self):
        for effect in ("sweep", "instant", "outline"):
            assert KaraokeConfig(effect=effect).effect == effect

    def test_lrc_metadata(self):
        config = KaraokeConfig(lrc_metadata={"ar": "Artist", "ti": "Title"})
        assert config.lrc_metadata["ar"] == "Artist"


class TestColorSchemes:
    def test_apply_preserves_font(self):
        from lattifai.caption.config import apply_color_scheme

        style = CaptionStyle(font_name="PingFang SC", font_size=24)
        apply_color_scheme(style, "azure-gold")
        assert style.font_name == "PingFang SC"
        assert style.font_size == 24

    def test_apply_overrides_colors(self):
        from lattifai.caption.config import apply_color_scheme

        style = CaptionStyle(primary_color="#FF0000")
        new_style = apply_color_scheme(style, "sakura-purple")
        assert new_style.primary_color == "#F7C3D9"
        assert style.primary_color == "#FF0000"  # original unchanged

    def test_all_12_schemes_resolve(self):
        from lattifai.caption.config import KARAOKE_COLOR_SCHEMES, resolve_karaoke_color_scheme

        assert len(KARAOKE_COLOR_SCHEMES) == 12
        for name in KARAOKE_COLOR_SCHEMES:
            result = resolve_karaoke_color_scheme(name)
            assert result is not None
            assert "primary_color" in result

    def test_unknown_scheme_no_change(self):
        from lattifai.caption.config import apply_color_scheme

        style = CaptionStyle()
        result = apply_color_scheme(style, "nonexistent")
        assert result is style  # same object returned when scheme not found
        assert result.primary_color == "#FFFFFF"

    def test_case_insensitive(self):
        from lattifai.caption.config import resolve_karaoke_color_scheme

        assert resolve_karaoke_color_scheme("Azure-Gold") is not None
        assert resolve_karaoke_color_scheme("  azure-gold  ") is not None


# =============================================================================
# 2. ASS karaoke tags
# =============================================================================


class TestASSKaraokeEffects:
    def test_sweep(self, sup_hello_world):
        content = ASSFormat.to_bytes([sup_hello_world], word_level=True, karaoke=KaraokeConfig(enabled=True)).decode()
        assert "{\\kf45}Hello" in content
        assert "{\\kf285}world" in content

    def test_instant(self, sup_hello_world):
        content = ASSFormat.to_bytes(
            [sup_hello_world], word_level=True, karaoke=KaraokeConfig(enabled=True, effect="instant")
        ).decode()
        assert "{\\k45}Hello" in content

    def test_outline(self):
        sup = _sup("Hello", 0.0, 1.0, [AlignmentItem(symbol="Hello", start=0.0, duration=0.5)])
        content = ASSFormat.to_bytes(
            [sup], word_level=True, karaoke=KaraokeConfig(enabled=True, effect="outline")
        ).decode()
        assert "{\\ko50}Hello" in content


class TestASSKaraokeStyle:
    def test_karaoke_style_defined(self, sup_hello_world):
        content = ASSFormat.to_bytes([sup_hello_world], word_level=True, karaoke=KaraokeConfig(enabled=True)).decode()
        assert "Style: Karaoke" in content

    def test_metadata_karaoke_takes_precedence(self):
        sup = _sup("Hello world", 0.0, 2.0, [
            AlignmentItem(symbol="Hello", start=0.0, duration=0.5),
            AlignmentItem(symbol="world", start=0.5, duration=1.5),
        ])
        metadata = {
            "ass_styles": {
                "Default": {"fontname": "Comic Sans MS", "fontsize": 36, "primarycolor": "&H0000FFFF",
                            "outlinecolor": "&H00FF0000", "alignment": 2},
                "Karaoke": {"fontname": "Comic Sans MS", "fontsize": 36, "primarycolor": "&H0000FFFF",
                            "outlinecolor": "&H00FF0000", "alignment": 2},
            }
        }
        content = ASSFormat.to_bytes([sup], word_level=True, karaoke=KaraokeConfig(enabled=True), metadata=metadata).decode()
        parts = [l for l in content.splitlines() if l.startswith("Style: Karaoke,")][0].split(",")
        assert parts[1] == "Comic Sans MS"
        assert float(parts[2]) == 36.0

    def test_inherits_default_when_no_karaoke_style(self):
        sup = _sup("Hello", 0.0, 1.0, [AlignmentItem(symbol="Hello", start=0.0, duration=0.5)])
        metadata = {"ass_styles": {"Default": {"fontname": "Georgia", "fontsize": 42,
                                               "primarycolor": "&H00FF00FF", "outlinecolor": "&H0000FF00", "alignment": 5}}}
        content = ASSFormat.to_bytes([sup], word_level=True, karaoke=KaraokeConfig(enabled=True), metadata=metadata).decode()
        parts = [l for l in content.splitlines() if l.startswith("Style: Karaoke,")][0].split(",")
        assert parts[1] == "Georgia"
        assert float(parts[2]) == 42.0

    def test_custom_style_param(self):
        sup = _sup("Hello", 0.0, 1.0, [AlignmentItem(symbol="Hello", start=0.0, duration=0.5)])
        style = CaptionStyle(font_size=64, font_name="Courier")
        content = ASSFormat.to_bytes([sup], word_level=True, karaoke=KaraokeConfig(enabled=True), style=style).decode()
        parts = [l for l in content.splitlines() if l.startswith("Style: Karaoke,")][0].split(",")
        assert parts[1] == "Courier"

    def test_fallback_without_alignment(self):
        sup = Supervision(text="No alignment", start=10.0, duration=2.0)
        content = ASSFormat.to_bytes([sup], word_level=True, karaoke=KaraokeConfig(enabled=True)).decode()
        assert "No alignment" in content
        assert "{\\k" not in content

    def test_word_level_false(self, sup_hello_world):
        content = ASSFormat.to_bytes([sup_hello_world], word_level=False).decode()
        assert "{\\k" not in content

    def test_multiline_newline_converted_to_ass_linebreak(self):
        """Literal \\n in multiline karaoke text must become \\N in ASS output."""
        sup = _sup(
            text="I'm very curious\nlike actually do",
            start=0.0,
            duration=5.0,
            words=[
                AlignmentItem(symbol="I'm", start=0.0, duration=0.12),
                AlignmentItem(symbol="very", start=0.12, duration=0.40),
                AlignmentItem(symbol="curious", start=0.52, duration=0.74),
                AlignmentItem(symbol="like", start=1.26, duration=0.21),
                AlignmentItem(symbol="actually", start=1.47, duration=1.24),
                AlignmentItem(symbol="do", start=2.71, duration=1.44),
            ],
        )
        content = ASSFormat.to_bytes(
            [sup], word_level=True, karaoke=KaraokeConfig(enabled=True)
        ).decode()
        # Must NOT contain literal newline inside a Dialogue line
        for line in content.splitlines():
            if line.startswith("Dialogue:"):
                assert "\n" not in line  # trivially true per splitlines
                assert "\\N" in line  # ASS line break tag present
                assert "\\n" not in line.split(",", 9)[-1]  # no literal \n in text field


# =============================================================================
# 3. TTML word timing
# =============================================================================


class TestTTMLWordTiming:
    def test_word_timing_attribute(self, sup_hello_world):
        content = TTMLFormat.to_bytes([sup_hello_world], word_level=True, karaoke=KaraokeConfig(enabled=True)).decode()
        assert 'itunes:timing="Word"' in content or "timing" in content.lower()

    def test_word_spans(self, sup_hello_world):
        content = TTMLFormat.to_bytes([sup_hello_world], word_level=True, karaoke=KaraokeConfig(enabled=True)).decode()
        assert "<span" in content
        assert 'begin="00:00:15.200"' in content
        assert "Hello" in content

    def test_paragraph_timing(self, sup_hello_world):
        content = TTMLFormat.to_bytes([sup_hello_world], word_level=True, karaoke=KaraokeConfig(enabled=True)).decode()
        assert "<p " in content
        assert 'begin="00:00:15.200"' in content

    def test_fallback_without_alignment(self):
        sup = Supervision(text="No alignment", start=10.0, duration=2.0)
        content = TTMLFormat.to_bytes([sup], word_level=True, karaoke=KaraokeConfig(enabled=True)).decode()
        assert "No alignment" in content
        assert content.count("<span") <= 1

    def test_word_per_paragraph_without_karaoke(self, sup_hello_world):
        content = TTMLFormat.to_bytes([sup_hello_world], word_level=True).decode()
        assert content.count("<p ") == 2
        assert 'itunes:timing="Word"' not in content

    def test_word_level_false(self, sup_hello_world):
        content = TTMLFormat.to_bytes([sup_hello_world], word_level=False).decode()
        assert 'itunes:timing="Word"' not in content


# =============================================================================
# 4. Cross-format word-level integration
# =============================================================================


class TestCrossFormatWordLevel:
    def test_all_formats_support_word_level(self, sup_hello_beautiful_world):
        karaoke = KaraokeConfig(enabled=True)
        for fmt in ["lrc", "ass", "ttml"]:
            writer = get_writer(fmt)
            assert writer is not None
            result = writer.to_bytes([sup_hello_beautiful_world], word_level=True, karaoke=karaoke)
            assert len(result) > 0

    def test_custom_config(self, sup_hello_beautiful_world):
        style = CaptionStyle(primary_color="#FF00FF", font_name=CaptionFonts.NOTO_SANS_SC)
        config = KaraokeConfig(enabled=True, effect="instant", lrc_metadata={"ar": "Test Artist"})

        lrc_result = get_writer("lrc").to_bytes([sup_hello_beautiful_world], word_level=True, karaoke=config)
        assert b"[ar:Test Artist]" in lrc_result

        ass_result = get_writer("ass").to_bytes([sup_hello_beautiful_world], word_level=True, karaoke=config, style=style)
        assert b"{\\k" in ass_result

    def test_graceful_fallback(self):
        sup = Supervision(text="No alignment data", start=0.0, duration=1.0)
        karaoke = KaraokeConfig(enabled=True)
        for fmt in ["lrc", "ass", "ttml"]:
            result = get_writer(fmt).to_bytes([sup], word_level=True, karaoke=karaoke)
            assert b"No alignment" in result


# =============================================================================
# 5. Karaoke timestamp boundaries (gap-aware timing)
# =============================================================================


class TestKaraokeTimestampBoundary:
    def test_vtt_uses_word_timestamps(self, sup_with_gaps):
        content = get_writer("vtt").to_bytes([sup_with_gaps], word_level=True, karaoke=KaraokeConfig(enabled=True)).decode()
        ts_match = re.search(r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})", content)
        assert ts_match

        def _ts(s):
            h, m, rest = s.split(":")
            sec, ms = rest.split(".")
            return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000

        assert abs(_ts(ts_match.group(1)) - 10.5) < 0.01
        assert abs(_ts(ts_match.group(2)) - 14.2) < 0.01

    def test_vtt_word_timestamps_within_cue(self, sup_with_gaps):
        content = get_writer("vtt").to_bytes([sup_with_gaps], word_level=True, karaoke=KaraokeConfig(enabled=True)).decode()

        def _ts(s):
            h, m, rest = s.split(":")
            sec, ms = rest.split(".")
            return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000

        cue_match = re.search(r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})", content)
        cue_start, cue_end = _ts(cue_match.group(1)), _ts(cue_match.group(2))

        word_timestamps = [_ts(m) for m in re.findall(r"<(\d{2}:\d{2}:\d{2}\.\d{3})><c>", content)]
        assert len(word_timestamps) == 3
        for wt in word_timestamps:
            assert cue_start - 0.01 <= wt <= cue_end + 0.01

    def test_ass_uses_word_timestamps(self, sup_with_gaps):
        content = get_writer("ass").to_bytes([sup_with_gaps], word_level=True, karaoke=KaraokeConfig(enabled=True)).decode()
        match = re.search(r"Dialogue:\s*\d+,(\d+:\d+:\d+\.\d+),(\d+:\d+:\d+\.\d+)", content)
        assert match

        def _ts(s):
            h, m, rest = s.split(":")
            sec, cs = rest.split(".")
            return int(h) * 3600 + int(m) * 60 + int(sec) + int(cs) / 100

        assert abs(_ts(match.group(1)) - 10.5) < 0.02
        assert abs(_ts(match.group(2)) - 14.2) < 0.02

    def test_ass_gap_aware_durations(self, sup_with_gaps):
        content = get_writer("ass").to_bytes(
            [sup_with_gaps], word_level=True, karaoke=KaraokeConfig(enabled=True, effect="sweep")
        ).decode()
        kf_values = [int(m) for m in re.findall(r"\\kf(\d+)", content)]
        assert len(kf_values) == 3
        # Gap-aware: Hello=100cs (10.5→11.5), beautiful=150cs (11.5→13.0), world=120cs
        assert abs(kf_values[0] - 100) <= 2
        assert abs(kf_values[1] - 150) <= 2
        assert abs(kf_values[2] - 120) <= 2
        assert abs(sum(kf_values) - 370) <= 5

    def test_lrc_timestamps_monotonic(self, sup_with_gaps):
        content = get_writer("lrc").to_bytes([sup_with_gaps], word_level=True, karaoke=KaraokeConfig(enabled=True)).decode()
        matches = re.findall(r"<(\d+):(\d+)\.(\d+)>", content)
        assert len(matches) >= 3

        prev = -1
        for m, s, ms in matches:
            ms_val = int(ms) * 10 if len(ms) == 2 else int(ms)
            ts = int(m) * 60 + int(s) + ms_val / 1000
            assert ts >= prev
            prev = ts

    def test_ttml_spans_within_paragraph(self, sup_with_gaps):
        content = get_writer("ttml").to_bytes([sup_with_gaps], word_level=True, karaoke=KaraokeConfig(enabled=True)).decode()
        p_match = re.search(r'<p[^>]*begin="([^"]+)"[^>]*end="([^"]+)"', content)
        if p_match:

            def _ts(ts):
                parts = ts.replace(",", ".").split(":")
                h, m, rest = parts
                return int(h) * 3600 + int(m) * 60 + float(rest)

            p_start, p_end = _ts(p_match.group(1)), _ts(p_match.group(2))
            span_starts = [_ts(m) for m in re.findall(r'<span[^>]*begin="([^"]+)"', content)]
            for st in span_starts:
                assert p_start - 0.01 <= st <= p_end + 0.01
