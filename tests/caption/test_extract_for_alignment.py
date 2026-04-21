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


# ---------------------------------------------------------------------------
# Real-file regressions — discovered by E2E smoke testing
# ---------------------------------------------------------------------------


SRT_INLINE_BILINGUAL = """\
1
00:00:01,000 --> 00:00:04,000
我们都很看好你
We all think a lot of you, you know?

2
00:00:05,000 --> 00:00:08,000
你可得照顾好自己
You must look after yourself.

3
00:00:09,000 --> 00:00:12,000
只要你需要
As long as you need.
"""


def test_srt_inline_bilingual_is_recognised_after_read(tmp_path) -> None:
    """Regression: a zh/en F1 SRT read from disk must classify as bilingual_inline.

    pysubs2's SRT reader previously folded the ``\\n`` inside each cue into
    a space (via ``normalize_text`` collapsing ``\\s+``), so the downstream
    bilingual detector couldn't see the inline split and fell back to mono.
    Real subtitle-group SRTs (e.g. Endeavour 简体&英文.srt) fail without
    this fix.
    """
    path = tmp_path / "inline.srt"
    path.write_text(SRT_INLINE_BILINGUAL, encoding="utf-8")
    caption = Caption.read(str(path))

    # The reader must preserve the newline so downstream can see both halves.
    multiline = [s for s in caption.supervisions if "\n" in (s.text or "")]
    assert len(multiline) == 3, (
        "Expected 3 multi-line cues; "
        f"got {len(multiline)} — SRT reader is folding \\n to space"
    )

    ctype, primary, secondary = caption.extract_for_alignment()
    assert ctype == "bilingual_inline"
    assert len(primary) == 3 and primary[0].language == "zh"
    assert len(secondary) == 3 and secondary[0].language == "en"


def test_ass_inline_bilingual_not_misclassified_by_star_style_variant() -> None:
    """Regression: ASS files that mix Sign rows (Style="Default") with
    inline bilingual dialogue rows (Style="*Default") must still classify
    as ``bilingual_inline``.

    ``_detect_bilingual_mode`` step 1 compares average CJK ratios per ASS
    style — and pysubs2 exposes ``Default`` and ``*Default`` as separate
    style names. Sign-only rows land on ``Default`` (100 % CJK) while the
    zh+en inline dialogue lands on ``*Default`` (~50 % CJK), so step 1
    thinks it's seeing two language tracks (dual-row) when really we have
    "metadata-style vs dialogue-style" inside a single track. Step 3's
    newline signal is the more reliable judge.
    """
    sups = [
        # Sign rows — small, top-of-screen, pure-CJK text in Style=Default.
        {"text": "摩斯探长前传", "start": 10.0, "duration": 3.0,
         "custom": {"ass_style": "Default"}},
        {"text": "第九季 第一集", "start": 15.0, "duration": 3.0,
         "custom": {"ass_style": "Default"}},
        # Inline bilingual dialogue in Style=*Default with \n between halves.
        {"text": "我们都很看好你\nWe all think a lot of you, you know?",
         "start": 20.0, "duration": 3.0,
         "custom": {"ass_style": "*Default"}},
        {"text": "你可得照顾好自己\nYou must look after yourself.",
         "start": 25.0, "duration": 3.0,
         "custom": {"ass_style": "*Default"}},
        {"text": "只要你需要\nAs long as you need.",
         "start": 30.0, "duration": 3.0,
         "custom": {"ass_style": "*Default"}},
    ]
    caption = _cap(sups, source_format="ass")
    ctype, primary, secondary = caption.extract_for_alignment()
    assert ctype == "bilingual_inline", (
        "star-style variant of Default should not trigger dual_row judgment"
    )
    assert primary[0].language == "zh"
    assert secondary and secondary[0].language == "en"
