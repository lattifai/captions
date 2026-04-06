"""P2-1/P2-2: Auto bilingual detection, merge, and staff credit tests."""

import pytest

from lattifai.caption import Caption
from lattifai.caption.parsers.text_parser import classify_line_type, cjk_ratio
from lattifai.caption.supervision import Supervision


# =============================================================================
# CJK ratio utility
# =============================================================================


class TestCJKRatio:
    """CJK character ratio detection."""

    def test_pure_chinese(self):
        assert cjk_ratio("他决不会放弃塔伯特") > 0.9

    def test_pure_english(self):
        assert cjk_ratio("He would never give up on Talbot") < 0.05

    def test_mixed(self):
        ratio = cjk_ratio("Hello世界")
        assert 0.2 < ratio < 0.8

    def test_empty(self):
        assert cjk_ratio("") == 0.0

    def test_punctuation_only(self):
        assert cjk_ratio("...!!!") == 0.0


# =============================================================================
# Auto mode detection
# =============================================================================


class TestMergeBilingualAuto:
    """merge_bilingual(mode='auto') should detect and handle all patterns."""

    def test_auto_line_by_line(self):
        """Pattern A/B: text contains \\n separating CJK and Latin lines."""
        sups = [
            Supervision(text="你好世界\nHello World", start=0, duration=5),
            Supervision(text="再见\nGoodbye", start=5, duration=5),
        ]
        cap = Caption(supervisions=sups)
        result = cap.merge_bilingual(mode="auto")

        assert result.supervisions[0].text == "你好世界"
        assert result.supervisions[0].translation == "Hello World"
        assert result.supervisions[1].text == "再见"
        assert result.supervisions[1].translation == "Goodbye"

    def test_auto_alternating(self):
        """Pattern: consecutive sups with same timing, one CJK one Latin."""
        sups = [
            Supervision(text="你好世界", start=0, duration=5),
            Supervision(text="Hello World", start=0, duration=5),
            Supervision(text="再见", start=5, duration=5),
            Supervision(text="Goodbye", start=5, duration=5),
        ]
        cap = Caption(supervisions=sups)
        result = cap.merge_bilingual(mode="auto")

        assert len(result.supervisions) == 2
        assert result.supervisions[0].text == "你好世界"
        assert result.supervisions[0].translation == "Hello World"

    def test_auto_ass_style_split(self):
        """Pattern D: Different ASS styles for different languages."""
        sups = [
            Supervision(text="你好世界", start=0, duration=5,
                        custom={"ass_style": "Default"}),
            Supervision(text="Hello World", start=0, duration=5,
                        custom={"ass_style": "English"}),
            Supervision(text="再见", start=5, duration=5,
                        custom={"ass_style": "Default"}),
            Supervision(text="Goodbye", start=5, duration=5,
                        custom={"ass_style": "English"}),
        ]
        cap = Caption(supervisions=sups)
        result = cap.merge_bilingual(mode="auto")

        assert len(result.supervisions) == 2
        assert result.supervisions[0].text == "你好世界"
        assert result.supervisions[0].translation == "Hello World"

    def test_auto_monolingual_unchanged(self):
        """Monolingual content should pass through unchanged."""
        sups = [
            Supervision(text="Hello World", start=0, duration=5),
            Supervision(text="How are you?", start=5, duration=5),
        ]
        cap = Caption(supervisions=sups)
        result = cap.merge_bilingual(mode="auto")

        assert len(result.supervisions) == 2
        assert result.supervisions[0].translation is None

    def test_auto_explicit_languages(self):
        """Auto mode should respect explicit language parameters."""
        sups = [
            Supervision(text="你好\nHello", start=0, duration=5),
        ]
        cap = Caption(supervisions=sups)
        result = cap.merge_bilingual(
            mode="auto", primary_language="zh", secondary_language="en"
        )

        assert result.language == "zh"
        assert result.target_lang == "en"


# =============================================================================
# P2-3: Filename language detection
# =============================================================================


class TestFilenameLangDetection:
    """detect_language_from_filename() should extract language info."""

    @pytest.mark.parametrize("filename,lang,target", [
        ("Show.S01E01.简体中文&英文.ass", "zh", "en"),
        ("Show.S01E01.简体&英文.srt", "zh", "en"),
        ("Show.CN&EN.ass", "zh", "en"),
        ("Show.chs&eng.srt", "zh", "en"),
        ("Show.繁体&英文.ass", "zh_tw", "en"),
        ("Show.繁體&英文.srt", "zh_tw", "en"),
        ("Show.简体中文.ass", "zh", None),
        ("Show.简体.srt", "zh", None),
        ("Show.chs.ass", "zh", None),
        ("Show.CN.srt", "zh", None),
        ("Show.繁体.ass", "zh_tw", None),
        ("Show.繁體.srt", "zh_tw", None),
        ("Show.cht.ass", "zh_tw", None),
        ("Show.英文.srt", "en", None),
        ("Show.EN.srt", "en", None),
        ("Show.eng.ass", "en", None),
        ("Show.双语.srt", "zh", "en"),
        ("Show.bilingual.ass", "zh", "en"),
    ])
    def test_language_patterns(self, filename, lang, target):
        from lattifai.caption.parsers.text_parser import detect_language_from_filename
        result_lang, result_target = detect_language_from_filename(filename)
        assert result_lang == lang, f"{filename}: expected lang={lang}, got {result_lang}"
        assert result_target == target, f"{filename}: expected target={target}, got {result_target}"

    def test_no_language_marker(self):
        from lattifai.caption.parsers.text_parser import detect_language_from_filename
        lang, target = detect_language_from_filename("Movie.2024.720p.srt")
        assert lang is None
        assert target is None


# =============================================================================
# P2-2: Staff credit and branding detection
# =============================================================================


class TestLineTypeClassification:
    """classify_line_type() should detect staff credits and branding."""

    def test_staff_credit_translation(self):
        assert classify_line_type("翻译 张三", start=10.0) == "staff_credit"

    def test_staff_credit_proofreading(self):
        assert classify_line_type("校对 李四", start=30.0) == "staff_credit"

    def test_staff_credit_timing(self):
        assert classify_line_type("时间轴 王五", start=60.0) == "staff_credit"

    def test_staff_credit_supervisor(self):
        assert classify_line_type("总监 赵六", start=5.0) == "staff_credit"

    def test_staff_credit_too_late(self):
        """Staff credits after 120s should not be classified."""
        assert classify_line_type("翻译 张三", start=150.0) is None

    def test_staff_credit_too_long(self):
        """Long text is not a staff credit."""
        long_text = "翻译 " + "张三李四" * 20
        assert classify_line_type(long_text, start=10.0) is None

    def test_branding_yyets(self):
        assert classify_line_type("YYeTs人人影视出品", start=5.0) == "branding"

    def test_branding_url(self):
        assert classify_line_type("www.zimuzu.tv", start=10.0) == "branding"

    def test_branding_disclaimer(self):
        assert classify_line_type("仅供交流学习 禁止商用", start=30.0) == "branding"

    def test_branding_subtitle_group(self):
        assert classify_line_type("某某字幕组出品", start=15.0) == "branding"

    def test_normal_dialogue(self):
        """Normal dialogue should return None."""
        assert classify_line_type("他决不会放弃塔伯特", start=300.0) is None

    def test_normal_dialogue_early(self):
        """Normal dialogue even early should not match."""
        assert classify_line_type("你好世界", start=5.0) is None
