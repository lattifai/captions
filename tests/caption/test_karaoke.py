"""Comprehensive karaoke tests: config, ASS tags, TTML timing, cross-format integration."""

import re

import pytest

from lattifai.caption.config import ASSConfig, CaptionFonts, RenderConfig
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
    """Supervision where word timestamps have gaps (10.5->11.3, 11.5->12.7, 13.0->14.2)."""
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
# 1. Config: RenderConfig, color schemes
# =============================================================================


class TestCaptionFonts:
    def test_western_fonts(self):
        assert CaptionFonts.ARIAL == "Arial"
        assert CaptionFonts.IMPACT == "Impact"

    def test_cjk_fonts(self):
        assert CaptionFonts.NOTO_SANS_SC == "Noto Sans SC"
        assert CaptionFonts.NOTO_SANS_JP == "Noto Sans JP"


class TestRenderConfigDefaults:
    def test_defaults(self):
        behavior = RenderConfig()
        assert behavior.include_speaker_in_text is True
        # word_level is tri-state (Optional[bool]); None == "per-format default".
        # For Renderer formats this matches the historical False behavior.
        assert behavior.word_level is None
        assert behavior.translation_first is False

    def test_custom(self):
        behavior = RenderConfig(include_speaker_in_text=False, word_level=True, translation_first=True)
        assert behavior.include_speaker_in_text is False
        assert behavior.word_level is True
        assert behavior.translation_first is True

    def test_ass_config_defaults(self):
        config = ASSConfig()
        assert config.font_size == 64
        assert config.secondary_color == "#00FFFF"
        assert config.outline_color == "#000000"
        assert config.alignment == 2

    def test_ass_config_custom(self):
        config = ASSConfig(font_size=56, secondary_color="#FF0000")
        assert config.font_size == 56
        assert config.secondary_color == "#FF0000"


class TestDefaults:
    def test_ass_config_karaoke_defaults(self):
        config = ASSConfig()
        assert config.karaoke_effect is None
        assert config.karaoke_color_scheme == ""

    def test_effects(self):
        for effect in ("sweep", "instant", "outline"):
            config = ASSConfig(karaoke_effect=effect)
            assert config.karaoke_effect == effect

    def test_lrc_config_defaults(self):
        from lattifai.caption.config import LRCConfig

        config = LRCConfig()
        assert config.precision == "millisecond"
        assert config.metadata == {}

    def test_lrc_config_metadata(self):
        from lattifai.caption.config import LRCConfig

        config = LRCConfig(metadata={"ar": "Artist", "ti": "Title"})
        assert config.metadata["ar"] == "Artist"


class TestColorSchemes:
    def test_apply_preserves_font(self):
        from lattifai.caption.config import apply_color_scheme

        config = ASSConfig(font_name="PingFang SC", font_size=24)
        new_config = apply_color_scheme("azure-gold", config=config)
        assert new_config.font_name == "PingFang SC"
        assert new_config.font_size == 24

    def test_apply_overrides_colors(self):
        from lattifai.caption.config import apply_color_scheme

        config = ASSConfig(primary_color="#FF0000")
        new_config = apply_color_scheme("sakura-purple", config=config)
        assert new_config.primary_color == "#F7C3D9"
        assert config.primary_color == "#FF0000"  # original unchanged

    def test_all_schemes_resolve(self):
        from lattifai.caption.config import KARAOKE_COLOR_SCHEMES, resolve_karaoke_color_scheme

        # 13 schemes: 12 stylistic palettes + yellow-pop (TikTok-style
        # active-word highlight). Add new schemes here when registered.
        assert len(KARAOKE_COLOR_SCHEMES) == 13
        for name in KARAOKE_COLOR_SCHEMES:
            result = resolve_karaoke_color_scheme(name)
            assert result is not None
            assert "primary_color" in result

    def test_unknown_scheme_no_change(self):
        from lattifai.caption.config import apply_color_scheme

        config = ASSConfig()
        result_config = apply_color_scheme("nonexistent", config=config)
        assert result_config is config  # same object returned when scheme not found
        assert result_config.primary_color == "#FFFFFF"

    def test_case_insensitive(self):
        from lattifai.caption.config import resolve_karaoke_color_scheme

        assert resolve_karaoke_color_scheme("Azure-Gold") is not None
        assert resolve_karaoke_color_scheme("  azure-gold  ") is not None


# =============================================================================
# 2. ASS karaoke tags
# =============================================================================


class TestASSKaraokeEffects:
    def test_sweep(self, sup_hello_world):
        config = ASSConfig(karaoke_effect="sweep")
        content = ASSFormat.to_bytes([sup_hello_world], render=RenderConfig(word_level=True), config=config).decode()
        assert "{\\kf45}Hello" in content
        assert "{\\kf285}world" in content

    def test_instant(self, sup_hello_world):
        config = ASSConfig(karaoke_effect="instant")
        content = ASSFormat.to_bytes(
            [sup_hello_world], render=RenderConfig(word_level=True), config=config
        ).decode()
        assert "{\\k45}Hello" in content

    def test_outline(self):
        sup = _sup("Hello", 0.0, 1.0, [AlignmentItem(symbol="Hello", start=0.0, duration=0.5)])
        config = ASSConfig(karaoke_effect="outline")
        content = ASSFormat.to_bytes(
            [sup], render=RenderConfig(word_level=True), config=config
        ).decode()
        assert "{\\ko50}Hello" in content


class TestASSKaraokeStyle:
    def test_karaoke_style_defined(self, sup_hello_world):
        config = ASSConfig(karaoke_effect="sweep")
        content = ASSFormat.to_bytes([sup_hello_world], render=RenderConfig(word_level=True), config=config).decode()
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
        config = ASSConfig(karaoke_effect="sweep")
        content = ASSFormat.to_bytes([sup], render=RenderConfig(word_level=True), config=config, metadata=metadata).decode()
        parts = [l for l in content.splitlines() if l.startswith("Style: Karaoke,")][0].split(",")
        assert parts[1] == "Comic Sans MS"
        assert float(parts[2]) == 36.0

    def test_inherits_default_when_no_karaoke_style(self):
        sup = _sup("Hello", 0.0, 1.0, [AlignmentItem(symbol="Hello", start=0.0, duration=0.5)])
        metadata = {"ass_styles": {"Default": {"fontname": "Georgia", "fontsize": 42,
                                               "primarycolor": "&H00FF00FF", "outlinecolor": "&H0000FF00", "alignment": 5}}}
        config = ASSConfig(karaoke_effect="sweep")
        content = ASSFormat.to_bytes([sup], render=RenderConfig(word_level=True), config=config, metadata=metadata).decode()
        parts = [l for l in content.splitlines() if l.startswith("Style: Karaoke,")][0].split(",")
        assert parts[1] == "Georgia"
        assert float(parts[2]) == 42.0

    def test_custom_style_param(self):
        sup = _sup("Hello", 0.0, 1.0, [AlignmentItem(symbol="Hello", start=0.0, duration=0.5)])
        config = ASSConfig(font_name="Courier", font_size=64, karaoke_effect="sweep")
        content = ASSFormat.to_bytes([sup], render=RenderConfig(word_level=True), config=config).decode()
        parts = [l for l in content.splitlines() if l.startswith("Style: Karaoke,")][0].split(",")
        assert parts[1] == "Courier"

    def test_fallback_without_alignment(self):
        sup = Supervision(text="No alignment", start=10.0, duration=2.0)
        config = ASSConfig(karaoke_effect="sweep")
        content = ASSFormat.to_bytes([sup], render=RenderConfig(word_level=True), config=config).decode()
        assert "No alignment" in content
        assert "{\\k" not in content

    def test_word_level_false(self, sup_hello_world):
        content = ASSFormat.to_bytes([sup_hello_world]).decode()
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
        config = ASSConfig(karaoke_effect="sweep")
        content = ASSFormat.to_bytes(
            [sup], render=RenderConfig(word_level=True), config=config
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
        content = TTMLFormat.to_bytes([sup_hello_world], render=RenderConfig(word_level=True)).decode()
        assert 'itunes:timing="Word"' in content or "timing" in content.lower()

    def test_word_spans(self, sup_hello_world):
        content = TTMLFormat.to_bytes([sup_hello_world], render=RenderConfig(word_level=True)).decode()
        assert "<span" in content
        assert 'begin="00:00:15.200"' in content
        assert "Hello" in content

    def test_paragraph_timing(self, sup_hello_world):
        content = TTMLFormat.to_bytes([sup_hello_world], render=RenderConfig(word_level=True)).decode()
        assert "<p " in content
        assert 'begin="00:00:15.200"' in content

    def test_fallback_without_alignment(self):
        sup = Supervision(text="No alignment", start=10.0, duration=2.0)
        content = TTMLFormat.to_bytes([sup], render=RenderConfig(word_level=True)).decode()
        assert "No alignment" in content
        assert content.count("<span") <= 1

    def test_word_spans_with_word_level(self, sup_hello_world):
        """word_level=True with alignment produces spans inside a single <p>."""
        content = TTMLFormat.to_bytes([sup_hello_world], render=RenderConfig(word_level=True)).decode()
        assert content.count("<p ") == 1
        assert content.count("<span") == 2
        assert 'itunes:timing="Word"' in content

    def test_word_level_false(self, sup_hello_world):
        content = TTMLFormat.to_bytes([sup_hello_world]).decode()
        assert 'itunes:timing="Word"' not in content


# =============================================================================
# 4. Cross-format word-level integration
# =============================================================================


class TestCrossFormatWordLevel:
    def test_all_formats_support_word_level(self, sup_hello_beautiful_world):
        for fmt in ["lrc", "ass", "ttml"]:
            writer = get_writer(fmt)
            assert writer is not None
            if fmt == "ass":
                result = writer.to_bytes(
                    [sup_hello_beautiful_world], render=RenderConfig(word_level=True),
                    config=ASSConfig(karaoke_effect="sweep"),
                )
            else:
                result = writer.to_bytes([sup_hello_beautiful_world], render=RenderConfig(word_level=True))
            assert len(result) > 0

    def test_custom_config(self, sup_hello_beautiful_world):
        from lattifai.caption.config import LRCConfig

        ass_config = ASSConfig(primary_color="#FF00FF", font_name=CaptionFonts.NOTO_SANS_SC, karaoke_effect="instant")
        lrc_config = LRCConfig(metadata={"ar": "Test Artist"})

        lrc_result = get_writer("lrc").to_bytes(
            [sup_hello_beautiful_world], render=RenderConfig(word_level=True), config=lrc_config
        )
        assert b"[ar:Test Artist]" in lrc_result

        ass_result = get_writer("ass").to_bytes(
            [sup_hello_beautiful_world], render=RenderConfig(word_level=True), config=ass_config
        )
        assert b"{\\k" in ass_result

    def test_graceful_fallback(self):
        sup = Supervision(text="No alignment data", start=0.0, duration=1.0)
        for fmt in ["lrc", "ass", "ttml"]:
            if fmt == "ass":
                result = get_writer(fmt).to_bytes(
                    [sup], render=RenderConfig(word_level=True), config=ASSConfig(karaoke_effect="sweep"),
                )
            else:
                result = get_writer(fmt).to_bytes([sup], render=RenderConfig(word_level=True))
            assert b"No alignment" in result


# =============================================================================
# 5. Karaoke timestamp boundaries (gap-aware timing)
# =============================================================================


class TestKaraokeTimestampBoundary:
    def test_vtt_uses_word_timestamps(self, sup_with_gaps):
        content = get_writer("vtt").to_bytes([sup_with_gaps], render=RenderConfig(word_level=True)).decode()
        ts_match = re.search(r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})", content)
        assert ts_match

        def _ts(s):
            h, m, rest = s.split(":")
            sec, ms = rest.split(".")
            return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000

        assert abs(_ts(ts_match.group(1)) - 10.5) < 0.01
        assert abs(_ts(ts_match.group(2)) - 14.2) < 0.01

    def test_vtt_word_timestamps_within_cue(self, sup_with_gaps):
        content = get_writer("vtt").to_bytes([sup_with_gaps], render=RenderConfig(word_level=True)).decode()

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
        config = ASSConfig(karaoke_effect="sweep")
        content = get_writer("ass").to_bytes([sup_with_gaps], render=RenderConfig(word_level=True), config=config).decode()
        match = re.search(r"Dialogue:\s*\d+,(\d+:\d+:\d+\.\d+),(\d+:\d+:\d+\.\d+)", content)
        assert match

        def _ts(s):
            h, m, rest = s.split(":")
            sec, cs = rest.split(".")
            return int(h) * 3600 + int(m) * 60 + int(sec) + int(cs) / 100

        assert abs(_ts(match.group(1)) - 10.5) < 0.02
        assert abs(_ts(match.group(2)) - 14.2) < 0.02

    def test_ass_gap_aware_durations(self, sup_with_gaps):
        config = ASSConfig(karaoke_effect="sweep")
        content = get_writer("ass").to_bytes(
            [sup_with_gaps], render=RenderConfig(word_level=True), config=config
        ).decode()
        kf_values = [int(m) for m in re.findall(r"\\kf(\d+)", content)]
        assert len(kf_values) == 3
        # Gap-aware: Hello=100cs (10.5->11.5), beautiful=150cs (11.5->13.0), world=120cs
        assert abs(kf_values[0] - 100) <= 2
        assert abs(kf_values[1] - 150) <= 2
        assert abs(kf_values[2] - 120) <= 2
        assert abs(sum(kf_values) - 370) <= 5

    def test_lrc_timestamps_monotonic(self, sup_with_gaps):
        content = get_writer("lrc").to_bytes([sup_with_gaps], render=RenderConfig(word_level=True)).decode()
        matches = re.findall(r"<(\d+):(\d+)\.(\d+)>", content)
        assert len(matches) >= 3

        prev = -1
        for m, s, ms in matches:
            ms_val = int(ms) * 10 if len(ms) == 2 else int(ms)
            ts = int(m) * 60 + int(s) + ms_val / 1000
            assert ts >= prev
            prev = ts

    def test_ttml_spans_within_paragraph(self, sup_with_gaps):
        content = get_writer("ttml").to_bytes([sup_with_gaps], render=RenderConfig(word_level=True)).decode()
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


# =============================================================================
# New API tests: KaraokeConfig deleted, karaoke via ASSConfig + RenderConfig
# =============================================================================


class TestNewKaraokeAPI:
    """Tests for the new API where karaoke is controlled via ASSConfig fields."""

    def test_render_config_exists(self):
        """RenderConfig should replace OutputBehavior."""
        from lattifai.caption.config import RenderConfig

        rc = RenderConfig(word_level=True)
        assert rc.word_level is True
        assert rc.include_speaker_in_text is True
        assert rc.translation_first is False

    def test_ass_config_has_karaoke_fields(self):
        """ASSConfig should have karaoke_effect and karaoke_color_scheme."""
        config = ASSConfig(karaoke_effect="sweep", karaoke_color_scheme="azure-gold")
        assert config.karaoke_effect == "sweep"
        assert config.karaoke_color_scheme == "azure-gold"

    def test_ass_config_karaoke_default_none(self):
        """ASSConfig.karaoke_effect should default to None (disabled)."""
        config = ASSConfig()
        assert config.karaoke_effect is None
        assert config.karaoke_color_scheme == ""

    def test_ass_karaoke_output(self, sup_hello_world):
        """ASS with karaoke_effect should produce \\k tags."""
        from lattifai.caption.config import RenderConfig

        ass_config = ASSConfig(karaoke_effect="sweep")
        render = RenderConfig(word_level=True)
        result = ASSFormat.to_bytes(
            [sup_hello_world], config=ass_config, render=render
        )
        assert b"\\k" in result or b"\\kf" in result

    def test_lrc_word_level_without_karaoke_config(self, sup_hello_world):
        """LRC with word_level=True should produce enhanced LRC without KaraokeConfig."""
        from lattifai.caption.config import RenderConfig

        render = RenderConfig(word_level=True)
        result = get_writer("lrc").to_bytes([sup_hello_world], render=render)
        content = result.decode("utf-8")
        # Enhanced LRC has inline timestamps
        assert "<" in content


_KARAOKE_ASS_HEADER = (
    "[Script Info]\n"
    "ScriptType: v4.00+\n\n"
    "[V4+ Styles]\n"
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
    "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
    "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
    "Alignment, MarginL, MarginR, MarginV, AlphaLevel, Encoding\n"
    "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
    "0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,0,1\n\n"
    "[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
    "Effect, Text\n"
)


class TestASSKaraokeReader:
    """ASS karaoke ``\\k*`` tags must parse into word-level alignment so
    downstream pipelines (forced alignment, sentence splitting, translation)
    can operate uniformly.
    """

    def test_kf_tags_populate_word_alignment(self, tmp_path):
        from lattifai.caption import Caption

        ass_src = (
            _KARAOKE_ASS_HEADER
            + "Dialogue: 0,0:00:01.00,0:00:05.00,Default,,0,0,0,,"
            + r"{\kf100}hello {\kf100}beautiful {\kf100}world"
            + "\n"
        )
        src = tmp_path / "k.ass"
        src.write_text(ass_src)

        cap = Caption.read(src)
        sup = cap.supervisions[0]

        assert sup.text == "hello beautiful world"
        assert sup.alignment is not None
        words = sup.alignment["word"]
        assert [w.symbol for w in words] == ["hello ", "beautiful ", "world"]
        assert [w.duration for w in words] == [1.0, 1.0, 1.0]
        # Each syllable's start advances by previous duration, anchored to event start.
        assert words[0].start == 1.0
        assert words[1].start == 2.0
        assert words[2].start == 3.0
        # ass_raw_text must be stripped of \k tags so write-back without
        # karaoke_effect doesn't leak stale timings.
        assert "\\k" not in sup.custom["ass_raw_text"]
        assert sup.custom["ass_raw_text"] == "hello beautiful world"

    def test_non_karaoke_override_tags_preserved(self, tmp_path):
        from lattifai.caption import Caption

        ass_src = (
            _KARAOKE_ASS_HEADER
            + "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,"
            + r"{\an8\kf30}hi{\kf20}there{\fad(0,500)}"
            + "\n"
        )
        src = tmp_path / "k.ass"
        src.write_text(ass_src)

        sup = Caption.read(src).supervisions[0]
        # \an8 and \fad survive; \kf gone
        assert "\\an8" in sup.custom["ass_raw_text"]
        assert "\\fad" in sup.custom["ass_raw_text"]
        assert "\\k" not in sup.custom["ass_raw_text"]

    def test_no_karaoke_keeps_raw_text_intact(self, tmp_path):
        from lattifai.caption import Caption

        ass_src = (
            _KARAOKE_ASS_HEADER
            + "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,"
            + r"{\an8}plain text"
            + "\n"
        )
        src = tmp_path / "k.ass"
        src.write_text(ass_src)

        sup = Caption.read(src).supervisions[0]
        assert sup.alignment is None
        # Non-karaoke event: raw text is verbatim from event.text
        assert sup.custom["ass_raw_text"] == r"{\an8}plain text"

    def test_regenerate_with_new_alignment_produces_new_kf_values(self, tmp_path):
        """Read karaoke ASS, replace alignment with new timings, write with
        karaoke_effect — output must reflect the NEW timings, not the old ones.
        """
        from lattifai.caption import Caption
        from lattifai.caption.config import ASSConfig, RenderConfig
        from lattifai.caption.supervision import AlignmentItem

        ass_src = (
            _KARAOKE_ASS_HEADER
            + "Dialogue: 0,0:00:01.00,0:00:05.00,Default,,0,0,0,,"
            + r"{\kf100}hello {\kf100}beautiful {\kf100}world"
            + "\n"
        )
        src = tmp_path / "k.ass"
        src.write_text(ass_src)

        cap = Caption.read(src)
        sup = cap.supervisions[0]
        sup.alignment = {
            "word": [
                AlignmentItem(symbol="hello",     start=1.00, duration=0.30),
                AlignmentItem(symbol="beautiful", start=1.30, duration=0.50),
                AlignmentItem(symbol="world",     start=1.80, duration=0.40),
            ]
        }
        sup.start = 1.00
        sup.duration = 1.20

        out = tmp_path / "out.ass"
        cap.write(
            out,
            format_config=ASSConfig(karaoke_effect="basic"),
            render=RenderConfig(word_level=True),
        )
        content = out.read_text()
        dialogue = next(
            ln for ln in content.splitlines() if ln.startswith("Dialogue:")
        )
        # New per-syllable durations: 30cs, 50cs, 40cs (was 100cs each).
        assert "\\kf30" in dialogue
        assert "\\kf50" in dialogue
        assert "\\kf40" in dialogue
        assert "\\kf100" not in dialogue

    def test_write_without_karaoke_effect_drops_stale_kf(self, tmp_path):
        """Read karaoke ASS, write back without karaoke_effect — output must
        be plaintext (no stale ``\\k`` tags). The previous behavior silently
        kept the original ``\\kf100`` in the Text field even after the
        segment-level Start/End was updated.
        """
        from lattifai.caption import Caption

        ass_src = (
            _KARAOKE_ASS_HEADER
            + "Dialogue: 0,0:00:01.00,0:00:05.00,Default,,0,0,0,,"
            + r"{\kf100}hello {\kf100}beautiful {\kf100}world"
            + "\n"
        )
        src = tmp_path / "k.ass"
        src.write_text(ass_src)

        out = tmp_path / "out.ass"
        Caption.read(src).write(out)
        content = out.read_text()
        dialogue = next(
            ln for ln in content.splitlines() if ln.startswith("Dialogue:")
        )
        assert "\\k" not in dialogue, dialogue
        assert dialogue.endswith("hello beautiful world")
