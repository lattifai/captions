"""Tests for ``Caption.extract_for_alignment()``.

Contract:
    extract_for_alignment() -> Tuple[str, List[Supervision], List[Supervision]]
        (caption_type, primary_sups, secondary_sups)

    caption_type: one of ``"mono"`` / ``"bilingual_inline"`` / ``"bilingual_dual_row"``.

    primary_sups:
        Fastcopies of ``self.supervisions`` restricted to the *priority*
        language, carrying enough state for ``apply_alignment`` to write
        results back. Each Supervision has:
          - ``align_index`` (on sup.custom) : 0-based index into the
                            original ``self.supervisions`` for write-back.
                            Inline bilingual shares one index across both
                            sides (F1); dual-row keeps two.
          - ``text``      : plaintext in the priority language, with ASS
                            override tags stripped
          - ``language``  : ISO-639-1 of the priority language
          - ``start`` / ``duration`` : original timestamps
        Non-dialogue rows (staff_credit / sign / karaoke / title / banner /
        translator_note / branding) and timing-less rows
        (``duration ≤ 0.01``) are excluded.

    secondary_sups:
        Same shape, but for the secondary language. ``[]`` for mono.

Priority language heuristic: the language contributing the most total
characters across the dialogue rows. Ties broken by ISO-639-1 lex order
for deterministic output.

``self`` is not mutated.
"""

import pytest

from lattifai.caption import Caption
from lattifai.caption.supervision import Supervision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cap(supervisions, source_format="srt"):
    """Build a Caption with the given supervision shorthand.

    Each entry is a dict describing one Supervision: text / start / duration,
    plus optional custom for ASS-specific tests.
    """
    sups = []
    for s in supervisions:
        sups.append(
            Supervision(
                text=s["text"],
                start=s["start"],
                duration=s["duration"],
                custom=s.get("custom"),
            )
        )
    return Caption(supervisions=sups, source_format=source_format)


# ---------------------------------------------------------------------------
# MONO
# ---------------------------------------------------------------------------


def test_mono_english_all_rows_in_primary() -> None:
    """Pure-English caption → caption_type='mono', primary=all rows, secondary=[]."""
    caption = _cap([
        {"text": "We all think a lot of you, you know?", "start": 1.0, "duration": 3.0},
        {"text": "Morse, please come in.", "start": 5.0, "duration": 3.0},
    ])
    cap_type, primary, secondary = caption.extract_for_alignment()
    assert cap_type == "mono"
    assert len(primary) == 2
    assert [s.align_index for s in primary] == [0, 1]
    assert primary[0].text == "We all think a lot of you, you know?"
    assert primary[0].language == "en"
    assert secondary == []


def test_mono_chinese_all_rows_in_primary() -> None:
    """Pure-Chinese caption → caption_type='mono', primary language='zh'."""
    caption = _cap([
        {"text": "我们都很看好你", "start": 1.0, "duration": 3.0},
        {"text": "摩斯 请进来", "start": 5.0, "duration": 3.0},
    ])
    cap_type, primary, secondary = caption.extract_for_alignment()
    assert cap_type == "mono"
    assert len(primary) == 2
    assert primary[0].language == "zh"
    assert [s.align_index for s in primary] == [0, 1]
    assert secondary == []


def test_mono_does_not_mutate_original() -> None:
    """extract_for_alignment() must leave the caller's caption untouched."""
    caption = _cap([
        {"text": "Hello world.", "start": 0.0, "duration": 2.0},
    ])
    before_texts = [s.text for s in caption.supervisions]
    before_starts = [s.start for s in caption.supervisions]
    caption.extract_for_alignment()
    assert [s.text for s in caption.supervisions] == before_texts
    assert [s.start for s in caption.supervisions] == before_starts


# ---------------------------------------------------------------------------
# BILINGUAL — inline (F1, \n within a single cue)
# ---------------------------------------------------------------------------


def test_bilingual_inline_zh_primary_en_secondary() -> None:
    """Subtitle-group F1 pattern: each cue has ``<zh>\\n<en>`` with shared index."""
    caption = _cap([
        {"text": "我们都很看好你\nWe all think a lot of you",
         "start": 1.0, "duration": 3.0},
        {"text": "摩斯 请进来\nMorse, please come in.",
         "start": 5.0, "duration": 3.0},
    ])
    cap_type, primary, secondary = caption.extract_for_alignment()
    assert cap_type == "bilingual_inline"
    # Same source row → both sides share align_index.
    assert [s.align_index for s in primary] == [0, 1]
    assert [s.align_index for s in secondary] == [0, 1]
    # Subtitle-group convention: CJK (zh) is primary.
    assert primary[0].language == "zh"
    assert primary[0].text == "我们都很看好你"
    assert secondary[0].language == "en"
    assert secondary[0].text == "We all think a lot of you"


# ---------------------------------------------------------------------------
# BILINGUAL — dual-row (F2, two Dialogue rows at the same timestamp)
# ---------------------------------------------------------------------------


def test_bilingual_dual_row_distinct_indices_per_language() -> None:
    """F2 pattern: same (start, end) but two rows, ASS Style name encodes lang.

    Each language side targets a different original index for write-back.
    """
    caption = _cap(
        [
            {"text": "我们都很看好你", "start": 1.0, "duration": 3.0,     # idx 0
             "custom": {"ass_style": "中文 1080"}},
            {"text": "We all think a lot of you", "start": 1.0,          # idx 1
             "duration": 3.0, "custom": {"ass_style": "英文 1080"}},
            {"text": "摩斯 请进来", "start": 5.0, "duration": 3.0,        # idx 2
             "custom": {"ass_style": "中文 1080"}},
            {"text": "Morse, please come in.", "start": 5.0,             # idx 3
             "duration": 3.0, "custom": {"ass_style": "英文 1080"}},
        ],
        source_format="ass",
    )
    cap_type, primary, secondary = caption.extract_for_alignment()
    assert cap_type == "bilingual_dual_row"
    # Two rows per language, different indices.
    assert len(primary) == 2
    assert len(secondary) == 2
    primary_indices = {s.align_index for s in primary}
    secondary_indices = {s.align_index for s in secondary}
    assert primary_indices.isdisjoint(secondary_indices)
    # Subtitle-group convention: CJK (zh) is primary.
    assert primary[0].language == "zh"
    assert secondary[0].language == "en"
    assert primary_indices == {0, 2}
    assert secondary_indices == {1, 3}


# ---------------------------------------------------------------------------
# Non-dialogue row handling
# ---------------------------------------------------------------------------


def test_staff_credit_row_excluded_from_alignment() -> None:
    """``翻译 张三`` looks like dialogue but is staff-credit; must be dropped."""
    caption = _cap([
        {"text": "翻译 张三", "start": 10.0, "duration": 3.0},           # idx 0
        {"text": "We all think a lot of you", "start": 20.0,           # idx 1
         "duration": 3.0},
    ])
    cap_type, primary, secondary = caption.extract_for_alignment()
    assert cap_type == "mono"
    assert [s.align_index for s in primary] == [1]


def test_zero_duration_row_excluded_from_alignment() -> None:
    """Dialogue rows whose timestamps collapse to zero length are excluded.

    They'll be re-timed via interpolation inside ``apply_alignment``.
    """
    caption = _cap([
        {"text": "No timing yet.", "start": 0.0, "duration": 0.0},    # idx 0
        {"text": "Morse, please come in.", "start": 5.0,              # idx 1
         "duration": 3.0},
    ])
    cap_type, primary, secondary = caption.extract_for_alignment()
    assert cap_type == "mono"
    assert [s.align_index for s in primary] == [1]


# ---------------------------------------------------------------------------
# Override-tag stripping
# ---------------------------------------------------------------------------


def test_ass_override_tags_stripped_from_extracted_text() -> None:
    """Extracted text must be plaintext — no {\\an8}{\\pos(...)} leftover."""
    caption = _cap(
        [
            {"text": r"{\an8}{\pos(100,200)}Hello world", "start": 1.0,
             "duration": 2.0},
        ],
        source_format="ass",
    )
    _, primary, _ = caption.extract_for_alignment()
    assert primary[0].text == "Hello world"
