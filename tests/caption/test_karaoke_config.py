"""Tests for caption style and karaoke configuration classes."""

import pytest

from lattifai.caption.config import CaptionFonts, CaptionStyle, KaraokeConfig


class TestCaptionFonts:
    """Test CaptionFonts constants."""

    def test_western_fonts_exist(self):
        """Western font constants should be defined."""
        assert CaptionFonts.ARIAL == "Arial"
        assert CaptionFonts.IMPACT == "Impact"
        assert CaptionFonts.VERDANA == "Verdana"

    def test_chinese_fonts_exist(self):
        """Chinese font constants should be defined."""
        assert CaptionFonts.NOTO_SANS_SC == "Noto Sans SC"
        assert CaptionFonts.MICROSOFT_YAHEI == "Microsoft YaHei"
        assert CaptionFonts.PINGFANG_SC == "PingFang SC"

    def test_japanese_fonts_exist(self):
        """Japanese font constants should be defined."""
        assert CaptionFonts.NOTO_SANS_JP == "Noto Sans JP"
        assert CaptionFonts.MEIRYO == "Meiryo"


class TestCaptionStyle:
    """Test CaptionStyle dataclass."""

    def test_default_values(self):
        """Default style should have sensible defaults."""
        style = CaptionStyle()
        assert style.primary_color == "#FFFFFF"
        assert style.secondary_color == "#00FFFF"
        assert style.font_name == CaptionFonts.ARIAL
        assert style.font_size == 20
        assert style.bold is False

    def test_custom_values(self):
        """Custom values should override defaults."""
        style = CaptionStyle(
            primary_color="#FF00FF",
            font_name=CaptionFonts.NOTO_SANS_SC,
            font_size=56,
            bold=True,
        )
        assert style.primary_color == "#FF00FF"
        assert style.font_name == "Noto Sans SC"
        assert style.font_size == 56
        assert style.bold is True


class TestKaraokeConfig:
    """Test KaraokeConfig dataclass."""

    def test_default_config(self):
        """Default config should work."""
        config = KaraokeConfig()
        assert config.enabled is False  # Default is False, must be explicitly enabled
        assert config.effect == "sweep"
        assert isinstance(config.style, CaptionStyle)
        assert config.lrc_precision == "millisecond"
        assert config.ttml_timing_mode == "Word"

    def test_effect_options(self):
        """Effect should support sweep, instant, outline."""
        config_sweep = KaraokeConfig(effect="sweep")
        assert config_sweep.effect == "sweep"

        config_instant = KaraokeConfig(effect="instant")
        assert config_instant.effect == "instant"

        config_outline = KaraokeConfig(effect="outline")
        assert config_outline.effect == "outline"

    def test_lrc_metadata(self):
        """LRC metadata should be configurable."""
        config = KaraokeConfig(lrc_metadata={"ar": "Artist", "ti": "Title"})
        assert config.lrc_metadata["ar"] == "Artist"
        assert config.lrc_metadata["ti"] == "Title"


class TestKaraokeColorScheme:
    """Test KaraokeConfig color_scheme field and resolve_karaoke_color_scheme()."""

    def test_color_scheme_applies_colors_to_style(self):
        """Setting color_scheme should auto-fill style colors via __post_init__."""
        config = KaraokeConfig(color_scheme="azure-gold")
        assert config.style.primary_color == "#FFFFFF"
        assert config.style.secondary_color == "#FFC209"
        assert config.style.outline_color == "#1387C0"
        assert config.style.back_color == "#0A3D5C"
        assert config.style.outline_width == 2.0

    def test_color_scheme_overrides_manual_style(self):
        """Preset should win over manually-set style colors (applied in __post_init__)."""
        manual_style = CaptionStyle(primary_color="#FF0000", secondary_color="#00FF00")
        config = KaraokeConfig(color_scheme="sakura-purple", style=manual_style)
        # Preset colors should overwrite the manual ones
        assert config.style.primary_color == "#F7C3D9"
        assert config.style.secondary_color == "#7953B1"
        assert config.style.outline_color == "#063C85"
        assert config.style.back_color == "#1A1A2E"

    def test_all_12_color_schemes_resolve(self):
        """All 12 documented color schemes should resolve to a valid color dict."""
        from lattifai.caption.config import KARAOKE_COLOR_SCHEMES, resolve_karaoke_color_scheme

        expected_names = [
            "azure-gold",
            "sakura-purple",
            "mint-ocean",
            "gardenia-green",
            "sunset-warm",
            "prussian-elegant",
            "burgundy-classic",
            "langgan-spring",
            "mars-teal",
            "spring-field",
            "navy-pink",
            "apricot-dark",
        ]
        assert len(KARAOKE_COLOR_SCHEMES) == 12

        for name in expected_names:
            result = resolve_karaoke_color_scheme(name)
            assert result is not None, f"Preset '{name}' should exist"
            assert "primary_color" in result
            assert "secondary_color" in result
            assert "outline_color" in result
            assert "back_color" in result

    def test_all_color_schemes_apply_via_config(self):
        """Every color_scheme should be usable via KaraokeConfig(color_scheme=...) without error."""
        from lattifai.caption.config import KARAOKE_COLOR_SCHEMES

        for name in KARAOKE_COLOR_SCHEMES:
            config = KaraokeConfig(color_scheme=name)
            assert config.style.primary_color != ""
            assert config.style.secondary_color != ""

    def test_unknown_color_scheme_does_not_crash(self):
        """An unrecognized color_scheme name should leave style defaults untouched."""
        config = KaraokeConfig(color_scheme="nonexistent-preset")
        # Style should remain at CaptionStyle defaults
        assert config.style.primary_color == "#FFFFFF"
        assert config.style.secondary_color == "#00FFFF"
        assert config.style.outline_color == "#000000"
        assert config.style.back_color == "#000000"

    def test_resolve_karaoke_color_scheme_returns_none_for_unknown(self):
        """resolve_karaoke_color_scheme() should return None for unknown names."""
        from lattifai.caption.config import resolve_karaoke_color_scheme

        assert resolve_karaoke_color_scheme("does-not-exist") is None
        assert resolve_karaoke_color_scheme("") is None

    def test_resolve_karaoke_color_scheme_case_insensitive(self):
        """resolve_karaoke_color_scheme() should be case-insensitive and strip whitespace."""
        from lattifai.caption.config import resolve_karaoke_color_scheme

        assert resolve_karaoke_color_scheme("Azure-Gold") is not None
        assert resolve_karaoke_color_scheme("  azure-gold  ") is not None
        assert resolve_karaoke_color_scheme("LANGGAN-SPRING") is not None

    def test_color_scheme_empty_string_keeps_defaults(self):
        """Empty color_scheme string (default) should not modify style."""
        config = KaraokeConfig(color_scheme="")
        assert config.style.primary_color == "#FFFFFF"
        assert config.style.secondary_color == "#00FFFF"

    def test_color_scheme_applies_optional_outline_and_shadow(self):
        """Presets with outline_width and shadow_depth should set them on style."""
        from lattifai.caption.config import KARAOKE_COLOR_SCHEMES

        # All current presets have outline_width
        config = KaraokeConfig(color_scheme="prussian-elegant")
        scheme_data = KARAOKE_COLOR_SCHEMES["prussian-elegant"]
        assert config.style.outline_width == scheme_data["outline_width"]
