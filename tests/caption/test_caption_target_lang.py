"""Tests for Caption target_lang support in from_supervisions and to_dict."""

from lattifai.caption.caption import Caption
from lattifai.caption.supervision import Supervision


class TestCaptionFromSupervisionsTargetLang:
    """Verify Caption.from_supervisions accepts and stores target_lang."""

    def test_target_lang_none_by_default(self):
        sups = [Supervision(text="hello", start=0.0, duration=1.0)]
        caption = Caption.from_supervisions(sups, language="en")
        assert caption.target_lang is None

    def test_target_lang_set(self):
        sups = [Supervision(text="hello", start=0.0, duration=1.0)]
        caption = Caption.from_supervisions(sups, language="en", target_lang="zh")
        assert caption.target_lang == "zh"

    def test_target_lang_preserved_in_to_dict(self):
        sups = [Supervision(text="hello", start=0.0, duration=1.0)]
        caption = Caption.from_supervisions(sups, language="en", target_lang="ja")
        d = caption.to_dict()
        assert d["target_lang"] == "ja"

    def test_target_lang_none_in_to_dict(self):
        sups = [Supervision(text="hello", start=0.0, duration=1.0)]
        caption = Caption.from_supervisions(sups, language="en")
        d = caption.to_dict()
        assert d["target_lang"] is None


class TestCaptionHasTranslationProperty:
    """Verify Caption.has_translation reflects supervision translation state."""

    def test_no_translation_without_translation_field(self):
        sups = [Supervision(text="hello", start=0.0, duration=1.0)]
        caption = Caption.from_supervisions(sups)
        assert not caption.has_translation

    def test_has_translation_with_translation_field(self):
        sups = [Supervision(text="hello", start=0.0, duration=1.0, translation="你好")]
        caption = Caption.from_supervisions(sups)
        assert caption.has_translation

    def test_set_translations(self):
        sups = [
            Supervision(text="hello", start=0.0, duration=1.0),
            Supervision(text="world", start=1.0, duration=1.0),
        ]
        caption = Caption.from_supervisions(sups, language="en")
        caption.set_translations(["你好", "世界"], target_lang="zh")

        assert caption.target_lang == "zh"
        assert caption.has_translation
        assert caption.supervisions[0].translation == "你好"
        assert caption.supervisions[1].translation == "世界"
        assert caption.supervisions[0].target_lang == "zh"

    def test_strip_translations(self):
        sups = [
            Supervision(text="hello", start=0.0, duration=1.0, translation="你好", target_lang="zh"),
        ]
        caption = Caption.from_supervisions(sups, target_lang="zh")
        caption.strip_translations()

        assert caption.supervisions[0].translation is None
        assert caption.supervisions[0].target_lang is None
