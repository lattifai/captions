"""Tests for Supervision dataclass — score field, serialization, and field propagation."""

from lattifai.caption.supervision import AlignmentItem, Supervision


class TestSupervisionScore:
    """Verify the score field on Supervision dataclass."""

    def test_score_default_none(self):
        s = Supervision(text="hello")
        assert s.score is None

    def test_score_set(self):
        s = Supervision(text="hello", score=0.95)
        assert s.score == 0.95

    def test_score_in_to_dict(self):
        s = Supervision(text="hello", score=0.88)
        d = s.to_dict()
        assert d["score"] == 0.88

    def test_score_none_excluded_from_to_dict(self):
        """_asdict_nonull should omit score when None."""
        s = Supervision(text="hello")
        d = s.to_dict()
        assert "score" not in d

    def test_score_roundtrip_from_dict(self):
        s = Supervision(text="hello", score=0.77)
        d = s.to_dict()
        s2 = Supervision.from_dict(d)
        assert s2.score == 0.77

    def test_score_preserved_by_with_offset(self):
        s = Supervision(text="hello", start=1.0, duration=0.5, score=0.92)
        s2 = s.with_offset(2.0)
        assert s2.score == 0.92
        assert s2.start == 3.0

    def test_score_preserved_by_trim(self):
        """trim() uses fastcopy, should preserve score."""
        s = Supervision(text="hello", start=1.0, duration=2.0, score=0.85)
        s2 = s.trim(end=2.5, start=0.0)
        assert s2.score == 0.85


class TestSupervisionTranslation:
    """Verify translation and target_lang fields."""

    def test_translation_default_none(self):
        s = Supervision(text="hello")
        assert s.translation is None
        assert s.target_lang is None

    def test_translation_set(self):
        s = Supervision(text="hello", translation="你好", target_lang="zh")
        assert s.translation == "你好"
        assert s.target_lang == "zh"

    def test_translation_in_to_dict(self):
        s = Supervision(text="hello", translation="Hola", target_lang="es")
        d = s.to_dict()
        assert d["translation"] == "Hola"
        assert d["target_lang"] == "es"

    def test_translation_roundtrip(self):
        s = Supervision(text="hello", translation="こんにちは", target_lang="ja")
        d = s.to_dict()
        s2 = Supervision.from_dict(d)
        assert s2.translation == "こんにちは"
        assert s2.target_lang == "ja"

    def test_has_translation_property(self):
        s1 = Supervision(text="hello")
        assert not s1.has_translation

        s2 = Supervision(text="hello", translation="你好")
        assert s2.has_translation

    def test_translation_preserved_by_with_offset(self):
        s = Supervision(text="hi", start=0.0, duration=1.0, translation="嗨", target_lang="zh")
        s2 = s.with_offset(5.0)
        assert s2.translation == "嗨"
        assert s2.target_lang == "zh"

    def test_translation_preserved_by_trim(self):
        s = Supervision(text="hi", start=0.0, duration=2.0, translation="嗨", target_lang="zh")
        s2 = s.trim(end=1.5)
        assert s2.translation == "嗨"
        assert s2.target_lang == "zh"


class TestSupervisionFromDict:
    """Verify from_dict handles all fields correctly."""

    def test_full_roundtrip(self):
        """All fields survive to_dict → from_dict."""
        s = Supervision(
            text="Hello world",
            start=1.5,
            duration=2.0,
            id="seg-0",
            recording_id="rec-1",
            channel=0,
            language="en",
            speaker="Alice",
            gender="Female",
            score=0.93,
            translation="你好世界",
            target_lang="zh",
            custom={"source": "youtube"},
            alignment={
                "word": [
                    AlignmentItem(symbol="Hello", start=1.5, duration=0.8, score=0.95),
                    AlignmentItem(symbol="world", start=2.3, duration=1.2, score=0.91),
                ]
            },
        )
        d = s.to_dict()
        s2 = Supervision.from_dict(d)

        assert s2.text == "Hello world"
        assert s2.start == 1.5
        assert s2.duration == 2.0
        assert s2.id == "seg-0"
        assert s2.language == "en"
        assert s2.speaker == "Alice"
        assert s2.gender == "Female"
        assert s2.score == 0.93
        assert s2.translation == "你好世界"
        assert s2.target_lang == "zh"
        assert s2.custom == {"source": "youtube"}
        assert len(s2.alignment["word"]) == 2

    def test_unknown_fields_ignored(self):
        """from_dict should silently ignore unknown fields (forward-compat)."""
        d = {"text": "hi", "start": 0.0, "duration": 1.0, "future_field": "ignored"}
        s = Supervision.from_dict(d)
        assert s.text == "hi"
        assert not hasattr(s, "future_field") or s.custom is None or "future_field" not in (s.custom or {})
