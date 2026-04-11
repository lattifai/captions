"""Kinetic typography tests: 15 style presets + char-level stagger + integration."""

import pytest

from lattifai.caption.config import ASSConfig, RenderConfig
from lattifai.caption.formats.pysubs2 import ASSFormat
from lattifai.caption.kinetic import (
    KINETIC_STYLE_NAMES,
    build_kinetic_overrides,
    expand_stagger_word,
    is_char_level_style,
    list_kinetic_styles,
    validate_kinetic_style,
)
from lattifai.caption.supervision import AlignmentItem, Supervision


# =============================================================================
# Fixtures
# =============================================================================


def _sup(text="Hello world", start=0.0, duration=1.0, words=None):
    if words is None:
        words = [
            AlignmentItem(symbol="Hello", start=0.0, duration=0.5),
            AlignmentItem(symbol="world", start=0.5, duration=0.5),
        ]
    return Supervision(
        text=text, start=start, duration=duration, alignment={"word": words}
    )


@pytest.fixture
def sup_two_words():
    return _sup()


@pytest.fixture
def sup_three_words():
    return _sup(
        text="Hello beautiful world",
        duration=3.0,
        words=[
            AlignmentItem(symbol="Hello", start=0.0, duration=1.0),
            AlignmentItem(symbol="beautiful", start=1.0, duration=1.0),
            AlignmentItem(symbol="world", start=2.0, duration=1.0),
        ],
    )


@pytest.fixture
def sup_cjk():
    return _sup(
        text="你好世界",
        duration=2.0,
        words=[
            AlignmentItem(symbol="你", start=0.0, duration=0.5),
            AlignmentItem(symbol="好", start=0.5, duration=0.5),
            AlignmentItem(symbol="世", start=1.0, duration=0.5),
            AlignmentItem(symbol="界", start=1.5, duration=0.5),
        ],
    )


def _render(sup, **ass_kwargs):
    config = ASSConfig(karaoke_effect="sweep", **ass_kwargs)
    return ASSFormat.to_bytes(
        [sup], render=RenderConfig(word_level=True), config=config
    ).decode()


# =============================================================================
# 1. Module-level API
# =============================================================================


class TestKineticModuleAPI:
    def test_all_15_styles_listed(self):
        assert len(KINETIC_STYLE_NAMES) == 15
        assert len(list_kinetic_styles()) == 15

    def test_style_groups_present(self):
        names = set(KINETIC_STYLE_NAMES)
        impact = {"bounce", "pop", "shake", "pulse", "swing"}
        smooth = {"fade", "zoom", "rise", "typewriter", "blur_in"}
        stylized = {"glow", "neon", "wave", "flicker", "stagger"}
        assert impact | smooth | stylized == names

    def test_validate_none_ok(self):
        validate_kinetic_style(None)  # Should not raise

    def test_validate_known_ok(self):
        for style in KINETIC_STYLE_NAMES:
            validate_kinetic_style(style)

    def test_validate_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown kinetic_style"):
            validate_kinetic_style("nonexistent")

    def test_validate_error_lists_available(self):
        with pytest.raises(ValueError, match="bounce"):
            validate_kinetic_style("not_a_style")

    def test_is_char_level_stagger_only(self):
        assert is_char_level_style("stagger") is True
        for style in KINETIC_STYLE_NAMES:
            if style != "stagger":
                assert is_char_level_style(style) is False
        assert is_char_level_style(None) is False


# =============================================================================
# 2. build_kinetic_overrides — word-level transition emission
# =============================================================================


class TestBuildKineticOverrides:
    def test_none_returns_empty(self):
        assert build_kinetic_overrides(None, 0) == ""

    def test_typewriter_returns_empty(self):
        assert build_kinetic_overrides("typewriter", 1000) == ""

    def test_stagger_returns_empty(self):
        # stagger is char-level; word-level pipeline must get empty string
        assert build_kinetic_overrides("stagger", 500) == ""

    def test_bounce_at_zero_offset(self):
        out = build_kinetic_overrides("bounce", 0)
        assert r"\t(0,1,\fscx120\fscy120)" in out
        assert r"\t(1,151,\fscx100\fscy100)" in out

    def test_bounce_offset_shifts_both_times(self):
        out = build_kinetic_overrides("bounce", 450)
        assert r"\t(450,451,\fscx120\fscy120)" in out
        assert r"\t(451,601,\fscx100\fscy100)" in out

    def test_pop_has_alpha_and_scale(self):
        out = build_kinetic_overrides("pop", 0)
        assert r"\alpha&HFF&" in out
        assert r"\alpha&H00&" in out
        assert r"\fscx60" in out
        assert r"\fscx100" in out

    def test_shake_three_rotations(self):
        out = build_kinetic_overrides("shake", 0)
        assert out.count(r"\t(") == 3
        assert r"\frz3" in out
        assert r"\frz-3" in out
        assert r"\frz0" in out

    def test_pulse_long_window(self):
        out = build_kinetic_overrides("pulse", 100)
        assert r"\t(100,300," in out
        assert r"\t(300,500," in out

    def test_swing_rotation(self):
        out = build_kinetic_overrides("swing", 0)
        assert r"\frz-8" in out
        assert r"\frz8" in out

    def test_fade_alpha_only(self):
        out = build_kinetic_overrides("fade", 0)
        assert r"\alpha&HFF&" in out
        assert r"\alpha&H00&" in out
        assert r"\fscx" not in out

    def test_zoom_scale(self):
        out = build_kinetic_overrides("zoom", 0)
        assert r"\fscx80\fscy80" in out
        assert r"\fscx100\fscy100" in out

    def test_rise_vertical_only(self):
        out = build_kinetic_overrides("rise", 0)
        assert r"\fscy0" in out
        assert r"\fscy100" in out
        # Rise is pure vertical — no horizontal scale
        assert r"\fscx" not in out

    def test_blur_in(self):
        out = build_kinetic_overrides("blur_in", 0)
        assert r"\blur4" in out
        assert r"\blur0" in out

    def test_glow(self):
        out = build_kinetic_overrides("glow", 0)
        assert r"\bord4\blur3" in out
        assert r"\bord2\blur1" in out

    def test_neon_three_stages(self):
        out = build_kinetic_overrides("neon", 0)
        assert out.count(r"\t(") == 3
        assert r"\bord6\blur5" in out

    def test_wave(self):
        out = build_kinetic_overrides("wave", 0)
        assert r"\fscy110" in out
        assert r"\fscy90" in out

    def test_flicker_four_transitions(self):
        out = build_kinetic_overrides("flicker", 0)
        assert out.count(r"\t(") == 4
        assert r"\alpha&HA0&" in out

    def test_unknown_style_raises(self):
        with pytest.raises(ValueError, match="Unknown kinetic_style"):
            build_kinetic_overrides("nonexistent", 0)


# =============================================================================
# 3. expand_stagger_word — char-level expansion
# =============================================================================


class TestExpandStaggerWord:
    def test_empty_word(self):
        assert expand_stagger_word("", 0) == ""

    def test_single_char(self):
        out = expand_stagger_word("A", 0)
        assert out.startswith("{")
        assert out.endswith("A")
        assert r"\fscx60\fscy60" in out
        assert r"\fscx100\fscy100" in out

    def test_multi_char_latin(self):
        out = expand_stagger_word("cat", 100)
        # Three char groups
        assert out.count("{") == 3
        # Each char preserved
        assert "c" in out and "a" in out and "t" in out
        # Time offsets advance by 30ms per char
        assert r"\t(100,101," in out
        assert r"\t(130,131," in out
        assert r"\t(160,161," in out

    def test_multi_char_cjk(self):
        out = expand_stagger_word("你好", 500)
        assert out.count("{") == 2
        assert "你" in out
        assert "好" in out
        assert r"\t(500,501," in out
        assert r"\t(530,531," in out

    def test_mixed_script(self):
        out = expand_stagger_word("Hi你", 0)
        assert out.count("{") == 3
        for ch in ("H", "i", "你"):
            assert ch in out


# =============================================================================
# 4. ASS writer integration
# =============================================================================


class TestASSKineticIntegration:
    def test_kinetic_none_matches_baseline(self, sup_two_words):
        """kinetic_style=None must produce byte-identical output to omitting it."""
        baseline = ASSFormat.to_bytes(
            [sup_two_words],
            render=RenderConfig(word_level=True),
            config=ASSConfig(karaoke_effect="sweep"),
        )
        explicit_none = ASSFormat.to_bytes(
            [sup_two_words],
            render=RenderConfig(word_level=True),
            config=ASSConfig(karaoke_effect="sweep", kinetic_style=None),
        )
        assert baseline == explicit_none

    def test_kinetic_none_has_no_transform_tags(self, sup_two_words):
        content = _render(sup_two_words)
        assert r"\t(" not in content
        assert r"\fscx" not in content

    def test_bounce_emits_transform(self, sup_two_words):
        content = _render(sup_two_words, kinetic_style="bounce")
        assert r"\fscx120\fscy120" in content
        assert r"\fscx100\fscy100" in content
        # \t(...) blocks must be present
        assert r"\t(0,1," in content

    def test_bounce_second_word_has_offset_time(self, sup_two_words):
        """First word at ms 0, second word at ms 500 (after 50cs of 'Hello')."""
        content = _render(sup_two_words, kinetic_style="bounce")
        # Word 1 animation starts at 0
        assert r"\t(0,1,\fscx120\fscy120)" in content
        # Word 2 animation starts at 500 (50cs * 10)
        assert r"\t(500,501,\fscx120\fscy120)" in content

    def test_three_words_cumulative_offsets(self, sup_three_words):
        content = _render(sup_three_words, kinetic_style="bounce")
        # Each word is 100cs = 1000ms
        assert r"\t(0,1,\fscx120\fscy120)" in content
        assert r"\t(1000,1001,\fscx120\fscy120)" in content
        assert r"\t(2000,2001,\fscx120\fscy120)" in content

    def test_typewriter_no_transforms(self, sup_two_words):
        content = _render(sup_two_words, kinetic_style="typewriter")
        # Typewriter adds no \t blocks but \kf must still be there
        assert r"\t(" not in content
        assert r"\kf" in content

    def test_stagger_expands_chars(self):
        sup = _sup(
            text="cat",
            words=[AlignmentItem(symbol="cat", start=0.0, duration=1.0)],
        )
        content = _render(sup, kinetic_style="stagger")
        # Each char gets its own override block
        assert r"\t(0,1," in content
        assert r"\t(30,31," in content
        assert r"\t(60,61," in content
        # Karaoke tag still present on the word
        assert r"\kf" in content

    def test_stagger_cjk(self, sup_cjk):
        content = _render(sup_cjk, kinetic_style="stagger")
        # 4 words × 1 char each — but stagger within a single-char word still
        # emits one char block per word with the char delay starting fresh
        for ch in ("你", "好", "世", "界"):
            assert ch in content

    def test_kinetic_with_color_scheme(self, sup_two_words):
        content = _render(
            sup_two_words, kinetic_style="glow", karaoke_color_scheme="azure-gold"
        )
        assert r"\bord4\blur3" in content
        assert "Style: Karaoke" in content

    def test_kinetic_preserves_karaoke_tags(self, sup_two_words):
        content = _render(sup_two_words, kinetic_style="pop")
        # Sweep \kf tag and kinetic \t block coexist in same override group
        assert r"\kf" in content
        assert r"\alpha&HFF&" in content

    @pytest.mark.parametrize("style", list(KINETIC_STYLE_NAMES))
    def test_all_15_styles_render_without_error(self, sup_three_words, style):
        content = _render(sup_three_words, kinetic_style=style)
        assert "Dialogue:" in content
        assert "Style: Karaoke" in content


# =============================================================================
# 5. Config validation
# =============================================================================


class TestASSConfigKineticValidation:
    def test_default_is_none(self):
        config = ASSConfig()
        assert config.kinetic_style is None

    def test_valid_style_accepted(self):
        config = ASSConfig(kinetic_style="bounce")
        assert config.kinetic_style == "bounce"

    def test_invalid_style_raises_at_construction(self):
        with pytest.raises(ValueError, match="Unknown kinetic_style"):
            ASSConfig(kinetic_style="totally_fake")  # type: ignore[arg-type]

    def test_all_15_styles_accepted(self):
        for style in KINETIC_STYLE_NAMES:
            config = ASSConfig(kinetic_style=style)
            assert config.kinetic_style == style


# =============================================================================
# 6. Backward compatibility
# =============================================================================


class TestBackwardCompatibility:
    def test_existing_karaoke_effects_still_work(self, sup_two_words):
        for effect in ("sweep", "instant", "outline"):
            content = ASSFormat.to_bytes(
                [sup_two_words],
                render=RenderConfig(word_level=True),
                config=ASSConfig(karaoke_effect=effect),
            ).decode()
            assert "Dialogue:" in content

    def test_color_scheme_without_kinetic(self, sup_two_words):
        content = ASSFormat.to_bytes(
            [sup_two_words],
            render=RenderConfig(word_level=True),
            config=ASSConfig(
                karaoke_effect="sweep", karaoke_color_scheme="sakura-purple"
            ),
        ).decode()
        assert r"\t(" not in content  # No kinetic → no \t transforms
