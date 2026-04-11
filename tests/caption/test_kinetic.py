"""Kinetic typography tests: dual-scope (line/word) preset architecture."""

import pytest

from lattifai.caption.config import ASSConfig, RenderConfig
from lattifai.caption.formats.pysubs2 import ASSFormat
from lattifai.caption.kinetic import (
    KINETIC_STYLE_NAMES,
    KineticImpl,
    KineticPreset,
    build_line_override,
    build_word_overrides,
    expand_stagger_word,
    get_kinetic_preset,
    is_char_level_style,
    list_kinetic_styles,
    resolve_kinetic,
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
def sup_no_alignment():
    return Supervision(text="Plain line with no word timing", start=0.0, duration=2.0)


def _render_word(sup, **ass_kwargs):
    """Render with word_level=True (karaoke word-scope path)."""
    config = ASSConfig(karaoke_effect="sweep", **ass_kwargs)
    return ASSFormat.to_bytes(
        [sup], render=RenderConfig(word_level=True), config=config
    ).decode()


def _render_line(sup, **ass_kwargs):
    """Render with word_level=False (standard line-scope path)."""
    config = ASSConfig(**ass_kwargs)
    return ASSFormat.to_bytes(
        [sup], render=RenderConfig(word_level=False), config=config
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
        smooth = {"fade", "zoom", "rise", "blur_in", "pop"}
        impact = {"bounce", "shake", "pulse", "swing"}
        stylized = {"glow", "neon", "wave", "flicker", "typewriter", "stagger"}
        assert smooth | impact | stylized == names

    def test_validate_none_ok(self):
        validate_kinetic_style(None)

    def test_validate_known_ok(self):
        for style in KINETIC_STYLE_NAMES:
            validate_kinetic_style(style)

    def test_validate_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown kinetic_style"):
            validate_kinetic_style("nonexistent")

    def test_is_char_level_stagger_only(self):
        assert is_char_level_style("stagger") is True
        for style in KINETIC_STYLE_NAMES:
            if style != "stagger":
                assert is_char_level_style(style) is False

    def test_get_preset_returns_kinetic_preset(self):
        preset = get_kinetic_preset("fade")
        assert isinstance(preset, KineticPreset)
        assert preset.line is not None
        assert preset.word is not None

    def test_rise_is_line_only(self):
        preset = get_kinetic_preset("rise")
        assert preset.line is not None
        assert preset.word is None

    def test_stagger_is_word_only(self):
        preset = get_kinetic_preset("stagger")
        assert preset.line is None
        assert preset.word is not None


# =============================================================================
# 2. Scope resolution
# =============================================================================


class TestResolveKinetic:
    def test_none_returns_none(self):
        assert resolve_kinetic(None, word_level=True) is None
        assert resolve_kinetic(None, word_level=False) is None

    def test_dual_preset_word_level_true_picks_word(self):
        scope, _ = resolve_kinetic("fade", word_level=True)
        assert scope == "word"

    def test_dual_preset_word_level_false_picks_line(self):
        scope, _ = resolve_kinetic("fade", word_level=False)
        assert scope == "line"

    def test_rise_word_level_false_picks_line(self):
        scope, _ = resolve_kinetic("rise", word_level=False)
        assert scope == "line"

    def test_rise_word_level_true_falls_back_to_line(self):
        scope, _ = resolve_kinetic("rise", word_level=True)
        assert scope == "line"

    def test_stagger_word_level_true_picks_word(self):
        scope, _ = resolve_kinetic("stagger", word_level=True)
        assert scope == "word"

    def test_stagger_word_level_false_raises(self):
        with pytest.raises(ValueError, match="requires word_level"):
            resolve_kinetic("stagger", word_level=False)

    def test_unknown_style_raises(self):
        with pytest.raises(ValueError, match="Unknown kinetic_style"):
            resolve_kinetic("nonexistent", word_level=True)


# =============================================================================
# 3. Builder functions
# =============================================================================


class TestBuildLineOverride:
    def test_fade_uses_fad_tag(self):
        _, impl = resolve_kinetic("fade", word_level=False)
        out = build_line_override(impl)
        assert out == r"\fad(300,0)"

    def test_zoom_has_initial_and_transition(self):
        _, impl = resolve_kinetic("zoom", word_level=False)
        out = build_line_override(impl)
        assert out.startswith(r"\fscy80")
        assert r"\t(0,300,\fscy100)" in out

    def test_rise_goes_from_zero_height(self):
        _, impl = resolve_kinetic("rise", word_level=False)
        out = build_line_override(impl)
        assert r"\fscy0" in out
        assert r"\fscy100" in out

    def test_blur_in_starts_blurred(self):
        _, impl = resolve_kinetic("blur_in", word_level=False)
        out = build_line_override(impl)
        assert r"\blur6" in out
        assert r"\blur0" in out

    def test_pop_has_scale_and_alpha(self):
        _, impl = resolve_kinetic("pop", word_level=False)
        out = build_line_override(impl)
        assert r"\fscy60" in out
        assert r"\alpha&HFF&" in out
        assert r"\fscy100" in out

    def test_shake_no_static_only_rotations(self):
        _, impl = resolve_kinetic("shake", word_level=False)
        out = build_line_override(impl)
        assert r"\frz3" in out
        assert r"\frz-3" in out
        assert r"\fscx" not in out
        assert r"\fscy" not in out

    def test_typewriter_uses_fad_one_ms(self):
        _, impl = resolve_kinetic("typewriter", word_level=False)
        out = build_line_override(impl)
        assert out == r"\fad(1,0)"


class TestBuildWordOverrides:
    def test_fade_word_has_static_alpha(self):
        _, impl = resolve_kinetic("fade", word_level=True)
        out = build_word_overrides(impl, 500)
        assert out.startswith(r"\alpha&HFF&")
        assert r"\t(500,800,\alpha&H00&)" in out

    def test_bounce_word_has_bord_and_blur(self):
        _, impl = resolve_kinetic("bounce", word_level=True)
        out = build_word_overrides(impl, 0)
        # Static reset + impact peak + decay
        assert out.startswith(r"\bord2\blur0")
        assert r"\bord8\blur4" in out

    def test_bounce_word_no_metric_nor_rotation(self):
        """Word scope bounce uses pure \\bord+\\blur impact — no \\fscy (metric
        reflow) and no \\frz (origin-displacement at off-center words)."""
        _, impl = resolve_kinetic("bounce", word_level=True)
        out = build_word_overrides(impl, 0)
        assert r"\fscy" not in out
        assert r"\fscx" not in out
        assert r"\frz" not in out

    def test_word_offset_shifts_times(self):
        _, impl = resolve_kinetic("shake", word_level=True)
        out = build_word_overrides(impl, 1000)
        assert r"\t(1000,1060," in out
        assert r"\t(1060,1120," in out
        assert r"\t(1120,1180," in out


# =============================================================================
# 4. Metric-safe invariant (the whole point of Phase 2)
# =============================================================================


_METRIC_UNSAFE_TAGS = (r"\fscx", r"\fscy", r"\fs", r"\fsp")


class TestWordScopeMetricSafe:
    """Word-scope impls must not emit any advance-width-affecting tag.

    Any of these tags in a per-word override block triggers libass line
    reflow, which visibly shakes the whole line during rapid speech.
    """

    @pytest.mark.parametrize("style", list(KINETIC_STYLE_NAMES))
    def test_no_metric_unsafe_tags_in_word_impl(self, style):
        resolved = resolve_kinetic(style, word_level=True)
        if resolved is None:
            return
        scope, impl = resolved
        if scope != "word":
            return  # rise falls back to line; line-scope may use fscy safely
        out = build_word_overrides(impl, 0)
        for bad in _METRIC_UNSAFE_TAGS:
            assert bad not in out, f"word scope for {style!r} leaked {bad}"


class TestLineScopeNoFscxAutowrapSafety:
    """Line scope may use \\fscy safely (whole block scales uniformly) but
    must avoid \\fscx and \\fsp which can re-trigger libass word-wrap on
    automatically-wrapped lines. See Codex review point 3."""

    @pytest.mark.parametrize("style", list(KINETIC_STYLE_NAMES))
    def test_no_fscx_or_fsp_in_line_impl(self, style):
        try:
            resolved = resolve_kinetic(style, word_level=False)
        except ValueError:
            return  # stagger has no line impl
        if resolved is None:
            return
        scope, impl = resolved
        if scope != "line":
            return
        out = build_line_override(impl)
        assert r"\fscx" not in out, f"line scope for {style!r} leaked \\fscx"
        assert r"\fsp" not in out, f"line scope for {style!r} leaked \\fsp"


# =============================================================================
# 5. expand_stagger_word — char-level expansion
# =============================================================================


class TestExpandStaggerWord:
    def test_empty_word(self):
        assert expand_stagger_word("", 0) == ""

    def test_latin_word(self):
        out = expand_stagger_word("cat", 100)
        assert out.count("{") == 3
        assert "c" in out and "a" in out and "t" in out
        # Per-char static alpha + transition
        assert r"\alpha&HFF&" in out
        assert r"\alpha&H00&" in out
        # Advance by char_delay_ms (40ms default)
        assert r"\t(100,220," in out
        assert r"\t(140,260," in out
        assert r"\t(180,300," in out

    def test_no_metric_tags(self):
        out = expand_stagger_word("abc", 0)
        for bad in _METRIC_UNSAFE_TAGS:
            assert bad not in out

    def test_cjk_word(self):
        out = expand_stagger_word("你好", 500)
        assert out.count("{") == 2
        assert "你" in out and "好" in out


# =============================================================================
# 6. ASS writer integration — word-scope path
# =============================================================================


class TestWordScopeIntegration:
    def test_fade_word_level_per_word_reveal(self, sup_two_words):
        content = _render_word(sup_two_words, kinetic_style="fade")
        # Each word has its own static invisible prefix
        assert content.count(r"\alpha&HFF&") >= 2
        # Cumulative offsets per word (50cs × 10 = 500ms per word)
        assert r"\t(0,300,\alpha&H00&)" in content
        assert r"\t(500,800,\alpha&H00&)" in content

    def test_bounce_word_level_impact(self, sup_two_words):
        content = _render_word(sup_two_words, kinetic_style="bounce")
        # Each word's override block has a static \bord2\blur0 reset to
        # break libass's cumulative override inheritance from the previous
        # word's in-flight animation.
        assert content.count(r"\bord2\blur0") >= 2  # reset + decay target
        assert r"\bord8\blur4" in content  # impact peak
        assert r"\frz" not in content  # no rotation
        # \kf still present for karaoke sweep
        assert r"\kf" in content

    def test_bounce_word_has_static_reset_per_word(self, sup_two_words):
        """Each word must open with the static reset so inheritance from the
        previous word's animation cannot leak into this word's rendering.
        The per-word block starts with \\k<cs> for the karaoke timing,
        immediately followed by the kinetic reset \\bord2\\blur0."""
        content = _render_word(sup_two_words, kinetic_style="bounce")
        import re

        # Match `{\kf<digits>\bord2\blur0` at the start of each word block.
        hits = re.findall(r"\{\\kf\d+\\bord2\\blur0", content)
        assert len(hits) >= 2

    def test_rise_word_level_falls_back_to_line_prefix(self, sup_two_words):
        """rise has no word impl, so word_level=True uses the line impl as a
        prefix at event start so the whole line rises while \\k sweeps."""
        content = _render_word(sup_two_words, kinetic_style="rise")
        assert r"\fscy0" in content
        assert r"\t(0,400,\fscy100)" in content
        assert r"\kf" in content

    def test_stagger_word_level_per_char(self, sup_two_words):
        content = _render_word(sup_two_words, kinetic_style="stagger")
        # Each char wrapped with its own override block
        assert content.count(r"\alpha&HFF&") >= len("Hello") + len("world")
        # No metric-unsafe tags
        assert r"\fscx" not in content
        assert r"\fscy" not in content

    @pytest.mark.parametrize(
        "style",
        [s for s in KINETIC_STYLE_NAMES if s != "rise"],  # rise special case above
    )
    def test_word_scope_no_fscx_fscy_in_output(self, sup_three_words, style):
        content = _render_word(sup_three_words, kinetic_style=style)
        # rise is the only preset where \fscy may appear in the output because
        # it falls back to its line impl. All other word-scope impls must be
        # metric-safe.
        if style != "rise":
            assert r"\fscx" not in content
            assert r"\fscy" not in content

    @pytest.mark.parametrize("style", list(KINETIC_STYLE_NAMES))
    def test_word_scope_renders_without_error(self, sup_three_words, style):
        content = _render_word(sup_three_words, kinetic_style=style)
        assert "Dialogue:" in content


# =============================================================================
# 7. ASS writer integration — line-scope path (word_level=False)
# =============================================================================


class TestLineScopeIntegration:
    def test_fade_line_uses_fad_event_tag(self, sup_no_alignment):
        content = _render_line(sup_no_alignment, kinetic_style="fade")
        assert r"{\fad(300,0)}" in content
        # No per-word alpha reveal
        assert r"\t(0,300,\alpha&H00&)" not in content

    def test_zoom_line_has_fscy_at_event_start(self, sup_no_alignment):
        content = _render_line(sup_no_alignment, kinetic_style="zoom")
        assert r"{\fscy80\t(0,300,\fscy100)}" in content

    def test_rise_line(self, sup_no_alignment):
        content = _render_line(sup_no_alignment, kinetic_style="rise")
        assert r"\fscy0" in content
        assert r"\t(0,400,\fscy100)" in content

    def test_blur_in_line(self, sup_no_alignment):
        content = _render_line(sup_no_alignment, kinetic_style="blur_in")
        assert r"\blur6" in content
        assert r"\blur0" in content

    def test_pop_line(self, sup_no_alignment):
        content = _render_line(sup_no_alignment, kinetic_style="pop")
        assert r"\fscy60" in content
        assert r"\alpha&HFF&" in content
        assert r"\fscy100" in content

    def test_bounce_line(self, sup_no_alignment):
        content = _render_line(sup_no_alignment, kinetic_style="bounce")
        assert r"\fscy130" in content
        assert r"\t(0,300,\fscy100)" in content

    def test_typewriter_line_uses_fad_one_ms(self, sup_no_alignment):
        content = _render_line(sup_no_alignment, kinetic_style="typewriter")
        assert r"{\fad(1,0)}" in content

    def test_stagger_line_raises(self, sup_no_alignment):
        """stagger has no line-scope impl — must fail fast."""
        with pytest.raises(ValueError, match="requires word_level"):
            _render_line(sup_no_alignment, kinetic_style="stagger")

    @pytest.mark.parametrize(
        "style",
        [s for s in KINETIC_STYLE_NAMES if s != "stagger"],
    )
    def test_line_scope_renders_without_error(self, sup_no_alignment, style):
        content = _render_line(sup_no_alignment, kinetic_style=style)
        assert "Dialogue:" in content


# =============================================================================
# 8. Config validation
# =============================================================================


class TestASSConfigKineticValidation:
    def test_default_is_none(self):
        config = ASSConfig()
        assert config.kinetic_style is None

    def test_valid_style_accepted(self):
        config = ASSConfig(kinetic_style="bounce")
        assert config.kinetic_style == "bounce"

    def test_invalid_style_raises(self):
        with pytest.raises(ValueError, match="Unknown kinetic_style"):
            ASSConfig(kinetic_style="totally_fake")  # type: ignore[arg-type]

    def test_all_15_styles_accepted(self):
        for style in KINETIC_STYLE_NAMES:
            config = ASSConfig(kinetic_style=style)
            assert config.kinetic_style == style


# =============================================================================
# 9. Backward compatibility
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
        assert r"\t(" not in content  # no kinetic → no \t tags

    def test_kinetic_none_has_no_overrides(self, sup_two_words):
        content = _render_word(sup_two_words)
        assert r"\t(" not in content
