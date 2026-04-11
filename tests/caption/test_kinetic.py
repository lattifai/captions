"""Kinetic typography tests: 15 style presets + char-level stagger + integration."""

import pytest

from lattifai.caption.config import ASSConfig, RenderConfig
from lattifai.caption.formats.pysubs2 import ASSFormat
from lattifai.caption.kinetic import (
    KINETIC_STYLE_NAMES,
    KineticTemplate,
    build_kinetic_overrides,
    expand_stagger_word,
    get_kinetic_template,
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
        validate_kinetic_style(None)

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

    def test_get_template_returns_dataclass(self):
        tpl = get_kinetic_template("fade")
        assert isinstance(tpl, KineticTemplate)
        assert tpl.initial == r"\alpha&HFF&"

    def test_get_template_unknown_raises(self):
        with pytest.raises(ValueError):
            get_kinetic_template("nope")


# =============================================================================
# 2. No-fscx guarantee (the whole point of Phase 1)
# =============================================================================


class TestNoHorizontalReflow:
    """Phase 1 invariant: no preset may emit \\fscx anywhere.

    libass treats \\fscx as a change to glyph advance width, which
    re-flows the line and produces visible horizontal jitter during
    rapid-fire speech. Vertical scale (\\fscy) is safe.
    """

    @pytest.mark.parametrize("style", list(KINETIC_STYLE_NAMES))
    def test_no_fscx_in_overrides(self, style):
        if is_char_level_style(style):
            out = expand_stagger_word("word", 0)
        else:
            out = build_kinetic_overrides(style, 0)
        assert r"\fscx" not in out, f"style {style!r} leaked \\fscx into overrides"

    @pytest.mark.parametrize("style", list(KINETIC_STYLE_NAMES))
    def test_no_fscx_in_rendered_ass(self, sup_three_words, style):
        content = _render(sup_three_words, kinetic_style=style)
        assert r"\fscx" not in content, f"style {style!r} leaked \\fscx into ASS output"


# =============================================================================
# 3. Entrance presets must have static initial state
# =============================================================================


class TestEntrancePresets:
    """fade, pop, zoom, rise, typewriter, blur_in all hide/squash the word
    from event start and reveal it at the word's activation time. Previously
    they emitted an inverted animation that flashed the word off at ws then
    faded back in, which read as a flicker on rapid speech.
    """

    def test_fade_has_static_alpha_invisible(self):
        tpl = get_kinetic_template("fade")
        assert tpl.initial == r"\alpha&HFF&"

    def test_fade_transition_fades_in_only(self):
        tpl = get_kinetic_template("fade")
        assert len(tpl.transitions) == 1
        t1, t2, tags = tpl.transitions[0]
        assert tags == r"\alpha&H00&"

    def test_pop_has_squashed_invisible_initial(self):
        tpl = get_kinetic_template("pop")
        assert r"\alpha&HFF&" in tpl.initial
        assert r"\fscy60" in tpl.initial

    def test_zoom_has_squashed_initial(self):
        tpl = get_kinetic_template("zoom")
        assert tpl.initial == r"\fscy80"

    def test_rise_has_zero_height_initial(self):
        tpl = get_kinetic_template("rise")
        assert tpl.initial == r"\fscy0"

    def test_typewriter_has_invisible_initial(self):
        tpl = get_kinetic_template("typewriter")
        assert tpl.initial == r"\alpha&HFF&"

    def test_blur_in_has_blurred_initial(self):
        tpl = get_kinetic_template("blur_in")
        assert tpl.initial == r"\blur4"

    @pytest.mark.parametrize(
        "style", ["fade", "pop", "zoom", "rise", "typewriter", "blur_in"]
    )
    def test_entrance_initial_appears_in_output(self, style, sup_two_words):
        content = _render(sup_two_words, kinetic_style=style)
        tpl = get_kinetic_template(style)
        assert tpl.initial in content


# =============================================================================
# 4. build_kinetic_overrides — transition emission
# =============================================================================


class TestBuildKineticOverrides:
    def test_none_returns_empty(self):
        assert build_kinetic_overrides(None, 0) == ""

    def test_stagger_returns_empty(self):
        assert build_kinetic_overrides("stagger", 500) == ""

    def test_bounce_vertical_only(self):
        out = build_kinetic_overrides("bounce", 0)
        assert r"\fscy130" in out
        assert r"\fscy100" in out
        assert r"\t(0,1," in out
        assert r"\t(1,151," in out

    def test_bounce_offset_shifts_both_times(self):
        out = build_kinetic_overrides("bounce", 450)
        assert r"\t(450,451," in out
        assert r"\t(451,601," in out

    def test_pop_has_initial_prefix(self):
        out = build_kinetic_overrides("pop", 0)
        # Static initial tags appear before any \t(
        assert out.startswith(r"\fscy60\alpha&HFF&")
        assert r"\t(0,120,\fscy100\alpha&H00&)" in out

    def test_pop_offset(self):
        out = build_kinetic_overrides("pop", 300)
        assert out.startswith(r"\fscy60\alpha&HFF&")
        assert r"\t(300,420," in out

    def test_shake_three_rotations(self):
        out = build_kinetic_overrides("shake", 0)
        assert out.count(r"\t(") == 3
        assert r"\frz3" in out
        assert r"\frz-3" in out
        assert r"\frz0" in out

    def test_pulse_vertical_only(self):
        out = build_kinetic_overrides("pulse", 100)
        assert r"\fscy115" in out
        assert r"\fscy100" in out
        assert r"\t(100,300," in out
        assert r"\t(300,500," in out

    def test_swing_rotation(self):
        out = build_kinetic_overrides("swing", 0)
        assert r"\frz-8" in out
        assert r"\frz8" in out

    def test_fade_static_then_single_transition(self):
        out = build_kinetic_overrides("fade", 0)
        assert out.startswith(r"\alpha&HFF&")
        assert r"\t(0,150,\alpha&H00&)" in out
        assert out.count(r"\t(") == 1

    def test_zoom_static_then_grow(self):
        out = build_kinetic_overrides("zoom", 0)
        assert out.startswith(r"\fscy80")
        assert r"\t(0,150,\fscy100)" in out

    def test_rise_static_zero_height(self):
        out = build_kinetic_overrides("rise", 0)
        assert out.startswith(r"\fscy0")
        assert r"\t(0,180,\fscy100)" in out

    def test_typewriter_hard_cut(self):
        out = build_kinetic_overrides("typewriter", 0)
        assert out.startswith(r"\alpha&HFF&")
        assert r"\t(0,1,\alpha&H00&)" in out

    def test_blur_in_static_blurred(self):
        out = build_kinetic_overrides("blur_in", 0)
        assert out.startswith(r"\blur4")
        assert r"\t(0,150,\blur0)" in out

    def test_glow_pulse(self):
        out = build_kinetic_overrides("glow", 0)
        assert out.startswith(r"\t(")  # No static initial
        assert r"\bord4\blur3" in out
        assert r"\bord2\blur1" in out

    def test_neon_three_stages(self):
        out = build_kinetic_overrides("neon", 0)
        assert out.count(r"\t(") == 3
        assert r"\bord6\blur5" in out

    def test_wave_vertical_ripple(self):
        out = build_kinetic_overrides("wave", 0)
        assert r"\fscy110" in out
        assert r"\fscy90" in out
        assert r"\fscy100" in out

    def test_flicker_four_transitions(self):
        out = build_kinetic_overrides("flicker", 0)
        assert out.count(r"\t(") == 4
        assert r"\alpha&HA0&" in out

    def test_unknown_style_raises(self):
        with pytest.raises(ValueError, match="Unknown kinetic_style"):
            build_kinetic_overrides("nonexistent", 0)


# =============================================================================
# 5. expand_stagger_word — char-level expansion
# =============================================================================


class TestExpandStaggerWord:
    def test_empty_word(self):
        assert expand_stagger_word("", 0) == ""

    def test_single_char_vertical_only(self):
        out = expand_stagger_word("A", 0)
        assert out.startswith("{")
        assert out.endswith("A")
        assert r"\fscy60" in out
        assert r"\fscy100" in out
        assert r"\fscx" not in out
        # Static initial hides + squashes char from event start
        assert r"\alpha&HFF&" in out
        assert r"\alpha&H00&" in out

    def test_multi_char_latin(self):
        out = expand_stagger_word("cat", 100)
        assert out.count("{") == 3
        assert "c" in out and "a" in out and "t" in out
        # Each char has its own \t(ch_start, ch_start + window, ...)
        assert r"\t(100,200," in out
        assert r"\t(130,230," in out
        assert r"\t(160,260," in out

    def test_multi_char_cjk(self):
        out = expand_stagger_word("你好", 500)
        assert out.count("{") == 2
        assert "你" in out
        assert "好" in out
        assert r"\t(500,600," in out
        assert r"\t(530,630," in out

    def test_mixed_script(self):
        out = expand_stagger_word("Hi你", 0)
        assert out.count("{") == 3
        for ch in ("H", "i", "你"):
            assert ch in out


# =============================================================================
# 6. ASS writer integration
# =============================================================================


class TestASSKineticIntegration:
    def test_kinetic_none_matches_baseline(self, sup_two_words):
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
        assert r"\fscy" not in content

    def test_bounce_emits_vertical_transform(self, sup_two_words):
        content = _render(sup_two_words, kinetic_style="bounce")
        assert r"\fscy130" in content
        assert r"\fscy100" in content
        assert r"\t(0,1," in content

    def test_bounce_second_word_has_offset_time(self, sup_two_words):
        content = _render(sup_two_words, kinetic_style="bounce")
        # First word at ms 0
        assert r"\t(0,1,\fscy130)" in content
        # Second word at ms 500 (50cs of "Hello" × 10)
        assert r"\t(500,501,\fscy130)" in content

    def test_three_words_cumulative_offsets(self, sup_three_words):
        content = _render(sup_three_words, kinetic_style="bounce")
        assert r"\t(0,1,\fscy130)" in content
        assert r"\t(1000,1001,\fscy130)" in content
        assert r"\t(2000,2001,\fscy130)" in content

    def test_typewriter_has_reveal_ramp(self, sup_two_words):
        content = _render(sup_two_words, kinetic_style="typewriter")
        # Static invisible initial + 1 ms reveal
        assert r"\alpha&HFF&\t(0,1,\alpha&H00&)" in content
        assert r"\kf" in content

    def test_stagger_expands_chars_vertical_only(self):
        sup = _sup(
            text="cat",
            words=[AlignmentItem(symbol="cat", start=0.0, duration=1.0)],
        )
        content = _render(sup, kinetic_style="stagger")
        # Each char has static initial + \t animation
        assert r"\fscy60" in content
        assert r"\fscy100" in content
        assert r"\fscx" not in content
        assert r"\alpha&HFF&" in content
        # Karaoke tag still present on the word
        assert r"\kf" in content

    def test_stagger_cjk(self, sup_cjk):
        content = _render(sup_cjk, kinetic_style="stagger")
        for ch in ("你", "好", "世", "界"):
            assert ch in content

    def test_kinetic_with_color_scheme(self, sup_two_words):
        content = _render(
            sup_two_words, kinetic_style="glow", karaoke_color_scheme="azure-gold"
        )
        assert r"\bord4\blur3" in content
        assert "Style: Karaoke" in content

    def test_fade_no_inverted_flash(self, sup_two_words):
        """Regression: fade must emit static \\alpha&HFF& prefix per word, not
        a \\t(0,1,\\alpha&HFF&) snap that flashes the word off at activation."""
        content = _render(sup_two_words, kinetic_style="fade")
        # No inverted snap pattern
        assert r"\t(0,1,\alpha&HFF&)" not in content
        # Static prefix present; second word also has static prefix
        assert content.count(r"\alpha&HFF&") >= 2

    def test_pop_no_inverted_flash(self, sup_two_words):
        content = _render(sup_two_words, kinetic_style="pop")
        assert r"\t(0,1,\fscy60" not in content  # No inverted snap
        # Static initial appears per word (2 words → at least 2 occurrences)
        assert content.count(r"\fscy60\alpha&HFF&") >= 2

    @pytest.mark.parametrize("style", list(KINETIC_STYLE_NAMES))
    def test_all_15_styles_render_without_error(self, sup_three_words, style):
        content = _render(sup_three_words, kinetic_style=style)
        assert "Dialogue:" in content
        assert "Style: Karaoke" in content


# =============================================================================
# 7. Config validation
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
# 8. Backward compatibility
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
        assert r"\t(" not in content
