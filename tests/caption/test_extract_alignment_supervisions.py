"""Tests for ``Caption.extract_alignment_supervisions()``.

Contract:
    extract_alignment_supervisions()
        -> Tuple[List[Supervision], List[Supervision]]
           (primary_sups, secondary_sups)

    primary_sups / secondary_sups: fastcopies of ``self.supervisions``
    restricted to one language each, carrying enough state for
    ``apply_alignment`` to write back:
      - ``align_index`` (on sup.custom): 0-based index into the original
        ``self.supervisions``. F1 inline shares one index across both
        sides; F2 dual-row keeps two.
      - ``text``: plaintext in that language, ASS override tags stripped.
      - ``language``: ISO-639-1 decided by group-level aggregate voting
        (primary's ``language`` is the same for every row in the list).
      - ``start`` / ``duration``: original timestamps.

    Mono captions return ``secondary_sups == []``; the presence or
    absence of the second list is the sole bilingual/mono signal.

    Non-dialogue rows (classify_line_type ≠ None) and zero-duration rows
    (duration ≤ 0.01s) are excluded from both lists.

    ``self`` is not mutated.
"""

import pytest

import lattifai.caption.parsers.text_parser as text_parser

from lattifai.caption import Caption
from lattifai.caption.supervision import Supervision


def _cap(supervisions, source_format="srt"):
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


def test_mono_english_returns_all_rows_as_primary() -> None:
    caption = _cap([
        {"text": "We all think a lot of you, you know?", "start": 1.0, "duration": 3.0},
        {"text": "Morse, please come in.", "start": 5.0, "duration": 3.0},
    ])
    primary, secondary = caption.extract_alignment_supervisions()
    assert len(primary) == 2
    assert primary[0].language == "en"
    assert primary[0].text == "We all think a lot of you, you know?"
    assert [s.align_index for s in primary] == [0, 1]
    assert secondary == []


def test_mono_chinese_returns_all_rows_as_primary() -> None:
    caption = _cap([
        {"text": "我们都很看好你", "start": 1.0, "duration": 3.0},
        {"text": "摩斯 请进来", "start": 5.0, "duration": 3.0},
    ])
    primary, secondary = caption.extract_alignment_supervisions()
    assert len(primary) == 2
    assert primary[0].language == "zh"
    assert secondary == []


def test_mono_fast_path_skips_bilingual_detection_and_line_classification(monkeypatch) -> None:
    """Simple mono captions should bypass the heavy bilingual/classification path."""

    def fail_detect_mode(*args, **kwargs):
        raise AssertionError("fast path should not call _detect_bilingual_mode")

    def fail_classify(*args, **kwargs):
        raise AssertionError("fast path should not call classify_line_type")

    monkeypatch.setattr(Caption, "_detect_bilingual_mode", fail_detect_mode)
    monkeypatch.setattr(text_parser, "classify_line_type", fail_classify)

    caption = _cap([
        {"text": "We all think a lot of you, you know?", "start": 1.0, "duration": 3.0},
        {"text": "Morse, please come in.", "start": 5.0, "duration": 3.0},
    ])
    primary, secondary = caption.extract_alignment_supervisions()

    assert [s.align_index for s in primary] == [0, 1]
    assert primary[0].language == "en"
    assert secondary == []


def test_does_not_mutate_original_caption() -> None:
    caption = _cap([
        {"text": "Hello world.", "start": 0.0, "duration": 2.0},
    ])
    before_texts = [s.text for s in caption.supervisions]
    before_starts = [s.start for s in caption.supervisions]
    caption.extract_alignment_supervisions()
    assert [s.text for s in caption.supervisions] == before_texts
    assert [s.start for s in caption.supervisions] == before_starts


def test_does_not_mutate_original_custom_fields() -> None:
    caption = _cap(
        [
            {
                "text": "我们都很看好你\nWe all think a lot of you",
                "start": 1.0,
                "duration": 3.0,
                "custom": {"ass_style": "*Default"},
            },
        ],
        source_format="ass",
    )
    before_custom = dict(caption.supervisions[0].custom)
    caption.extract_alignment_supervisions()
    assert caption.supervisions[0].custom == before_custom


# ---------------------------------------------------------------------------
# BILINGUAL — F1 inline (\n within a cue)
# ---------------------------------------------------------------------------


def test_bilingual_inline_yields_two_groups_sharing_align_index() -> None:
    """F1: primary (zh) and secondary (en) point at the same source row."""
    caption = _cap([
        {"text": "我们都很看好你\nWe all think a lot of you",
         "start": 1.0, "duration": 3.0},
        {"text": "摩斯 请进来\nMorse, please come in.",
         "start": 5.0, "duration": 3.0},
    ])
    primary, secondary = caption.extract_alignment_supervisions()
    assert primary[0].language == "zh"
    assert primary[0].text == "我们都很看好你"
    assert secondary[0].language == "en"
    assert secondary[0].text == "We all think a lot of you"
    # Same original row serves both sides.
    assert [s.align_index for s in primary] == [0, 1]
    assert [s.align_index for s in secondary] == [0, 1]


# ---------------------------------------------------------------------------
# BILINGUAL — F2 dual-row (two Dialogue rows at the same timestamp)
# ---------------------------------------------------------------------------


def test_bilingual_dual_row_targets_distinct_original_indices() -> None:
    """F2: each language side writes back to its own original row."""
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
    primary, secondary = caption.extract_alignment_supervisions()
    assert primary[0].language == "zh"
    assert secondary[0].language == "en"
    assert {s.align_index for s in primary} == {0, 2}
    assert {s.align_index for s in secondary} == {1, 3}


# ---------------------------------------------------------------------------
# Non-dialogue row handling
# ---------------------------------------------------------------------------


def test_staff_credit_row_excluded_from_alignment() -> None:
    """``翻译 张三`` is staff-credit; must be dropped."""
    caption = _cap([
        {"text": "翻译 张三", "start": 10.0, "duration": 3.0},            # idx 0
        {"text": "We all think a lot of you", "start": 20.0,            # idx 1
         "duration": 3.0},
    ])
    primary, secondary = caption.extract_alignment_supervisions()
    assert [s.align_index for s in primary] == [1]
    assert secondary == []


def test_zero_duration_row_excluded_from_alignment() -> None:
    """Zero-duration rows are excluded; they're re-timed by apply_alignment."""
    caption = _cap([
        {"text": "No timing yet.", "start": 0.0, "duration": 0.0},    # idx 0
        {"text": "Morse, please come in.", "start": 5.0,              # idx 1
         "duration": 3.0},
    ])
    primary, secondary = caption.extract_alignment_supervisions()
    assert [s.align_index for s in primary] == [1]


# ---------------------------------------------------------------------------
# Override-tag stripping
# ---------------------------------------------------------------------------


def test_ass_override_tags_stripped_from_extracted_text() -> None:
    caption = _cap(
        [
            {"text": r"{\an8}{\pos(100,200)}Hello world", "start": 1.0,
             "duration": 2.0},
        ],
        source_format="ass",
    )
    primary, _ = caption.extract_alignment_supervisions()
    assert primary[0].text == "Hello world"


# ---------------------------------------------------------------------------
# Cross-validation — short-text lingua outliers must not become a fake
# secondary track.
# ---------------------------------------------------------------------------


def test_mostly_english_mono_is_not_split_by_lingua_short_text_noise() -> None:
    """Mono English with a few short lines that lingua sometimes mislabels
    (e.g. ``"Morse."`` → ``de``) must still come back as mono-en, not a
    zh/en split. Layer-2 rollback folds Latin-script "secondary" back into
    primary when the two groups share alignment script.
    """
    caption = _cap([
        {"text": "The storm's a proper one tonight.", "start": 1.0, "duration": 2.5},
        {"text": "You know he won't give up.", "start": 4.0, "duration": 2.5},
        {"text": "Tell her I'm coming home.", "start": 7.0, "duration": 2.5},
        {"text": "Morse.", "start": 10.0, "duration": 1.5},
        {"text": "Sir.", "start": 12.0, "duration": 1.5},
        {"text": "Please come in, sergeant.", "start": 14.0, "duration": 2.5},
        {"text": "The valley's quiet this time of year.", "start": 17.0, "duration": 2.5},
        {"text": "I shan't forget.", "start": 20.0, "duration": 2.5},
        {"text": "Right then.", "start": 23.0, "duration": 1.5},
        {"text": "He looked troubled this morning.", "start": 25.0, "duration": 2.5},
        {"text": "No.", "start": 28.0, "duration": 1.0},
        {"text": "We'll meet again before the week's out.", "start": 30.0, "duration": 3.0},
    ])
    primary, secondary = caption.extract_alignment_supervisions()
    assert secondary == [], (
        "Layer-2 rollback should have folded any Latin-script outliers back "
        "into primary."
    )
    assert primary[0].language == "en"
    # All rows (including the short ones) are in primary.
    assert len(primary) == 12


def test_chinese_mono_with_multiline_numeric_cues_stays_mono() -> None:
    """Pure-Chinese cues that split across ``\\n`` due to digits (e.g.
    dates) must not be classified as bilingual just because a permissive
    script threshold is near-triggered.
    """
    caption = _cap([
        {"text": "梅林根酒店账单\n1972年3月21日周二入住",
         "start": 10.0, "duration": 3.0},
        {"text": "泰晤士河谷城堡门警局", "start": 15.0, "duration": 3.0},
        {"text": "瑟斯戴 牛津2831", "start": 20.0, "duration": 3.0},
    ])
    primary, secondary = caption.extract_alignment_supervisions()
    assert secondary == []
    assert primary[0].language == "zh"


# ---------------------------------------------------------------------------
# Read-time regressions — discovered via E2E smoke tests
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


def test_srt_inline_bilingual_recognised_after_read(tmp_path) -> None:
    """Regression: zh/en F1 SRT read from disk must surface as bilingual."""
    path = tmp_path / "inline.srt"
    path.write_text(SRT_INLINE_BILINGUAL, encoding="utf-8")
    caption = Caption.read(str(path))

    # Reader must preserve the \n between the two halves.
    multiline = [s for s in caption.supervisions if "\n" in (s.text or "")]
    assert len(multiline) == 3

    primary, secondary = caption.extract_alignment_supervisions()
    assert primary and primary[0].language == "zh"
    assert secondary and secondary[0].language == "en"
    assert len(primary) == 3 and len(secondary) == 3


def test_ass_inline_bilingual_not_misclassified_when_sign_and_dialogue_share_default_style() -> None:
    """Regression: ASS files where sign rows (Style=Default) coexist with
    inline bilingual dialogue (Style=*Default) must surface as bilingual.

    Signs land on "Default" (pure CJK), inline dialogue on "*Default"
    (zh\\nen). The style-CJK split would otherwise fool step 1 of
    ``_detect_bilingual_mode`` into declaring dual-row; step 3's newline
    signal takes precedence.
    """
    sups = [
        {"text": "摩斯探长前传", "start": 10.0, "duration": 3.0,
         "custom": {"ass_style": "Default"}},
        {"text": "第九季 第一集", "start": 15.0, "duration": 3.0,
         "custom": {"ass_style": "Default"}},
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
    primary, secondary = caption.extract_alignment_supervisions()
    assert primary[0].language == "zh"
    assert secondary and secondary[0].language == "en"


# ---------------------------------------------------------------------------
# Corpus-discovered bugs — both surfaced by the audit in
# ``scripts/smoke_roundtrip_corpus.py`` against yyets-subtitles.
# ---------------------------------------------------------------------------


def test_bilingual_pair_skipped_when_secondary_side_is_non_dialogue() -> None:
    """Regression: both sides of an inline bilingual cue must pass
    ``classify_line_type``, not just the primary side.

    Real-world repro: the opening two cues of many subtitle-group SRTs
    are pure-Chinese disclaimers split across ``\\n``. The first line is
    usually benign enough to escape the classifier (returns ``None``)
    while the second line matches the ``branding`` keyword set. The old
    implementation probed only ``t1``, so the whole pair slipped through
    and the ``branding`` line landed in ``secondary``.
    """
    caption = _cap([
        # Subtitle-group disclaimer — t1 escapes classify_line_type,
        # t2 ("仅供学习交流...") is caught as ``branding``.
        {"text": "视频资料来自网络 版权归BBC所有\n仅供学习交流使用 禁止用于任何商业盈利行为",
         "start": 0.0, "duration": 1.68},
        # Real bilingual dialogue so language voting has a leg to stand on.
        {"text": "今晚 一只狗狗出墙来\nTonight, a dog looks over a wall.",
         "start": 4.68, "duration": 4.0},
        {"text": "我们骑椅子\nWe drive some chairs.",
         "start": 9.8, "duration": 1.76},
        {"text": "大家好\nHello everybody.",
         "start": 23.08, "duration": 1.32},
    ])

    primary, secondary = caption.extract_alignment_supervisions()

    # The disclaimer pair at idx 0 must be rejected in its entirety —
    # neither primary nor secondary may carry an idx=0 entry.
    primary_idx = [getattr(s, "align_index", None) for s in primary]
    secondary_idx = [getattr(s, "align_index", None) for s in secondary]
    assert 0 not in primary_idx, "disclaimer t1 leaked into primary"
    assert 0 not in secondary_idx, "disclaimer t2 leaked into secondary"

    # All remaining rows must be genuine dialogue.
    for side_name, rows in (("primary", primary), ("secondary", secondary)):
        for row in rows:
            assert text_parser.classify_line_type(
                row.text,
                start=row.start,
                duration=row.duration,
            ) is None, f"non-dialogue leaked into {side_name}: {row.text!r}"


def test_mono_with_two_sync_pairs_is_not_misdetected_as_alternating() -> None:
    """Regression: a mostly-mono caption where only a couple of
    same-timing pairs exist (e.g. a CJK-heavy title card stacked over a
    Latin-heavy subtitle) must not flip the bilingual mode to
    ``alternating``.

    The old detector tripped the alternating branch as soon as
    ``pair_count >= 2``, producing an absurd split like 1228 / 1 on
    1000+ line Japanese captions. The fix requires the same-timing
    pairs to cover a non-trivial fraction of the corpus, and — as a
    belt-and-braces measure — the extractor rolls an extreme
    primary/secondary skew back to mono.
    """
    sups = []
    # 30 mono Chinese dialogue rows — the clear majority signal.
    for i in range(30):
        sups.append({"text": f"今天的天气真好 第{i}句对白",
                     "start": 10.0 + i * 3.0, "duration": 2.5})
    # Two stray CJK-vs-Latin same-timing pairs (e.g. a title card stacked
    # over a transliterated effect line). Both have cjk_ratio diff > 0.4
    # so they trip the old ``pair_count >= 2 and diff/pair > 0.5``
    # alternating branch despite being a tiny minority.
    sups.append({"text": "梅林根酒店账单",
                 "start": 120.0, "duration": 5.0})
    sups.append({"text": "MERINGEN HOTEL BILL",
                 "start": 120.0, "duration": 5.0})
    sups.append({"text": "泰晤士河谷城堡门警局",
                 "start": 130.0, "duration": 5.0})
    sups.append({"text": "THAMES VALLEY POLICE",
                 "start": 130.0, "duration": 5.0})

    caption = _cap(sups)
    primary, secondary = caption.extract_alignment_supervisions()

    assert secondary == [], (
        "Mostly-mono captions with a handful of same-timing pairs must "
        "stay mono rather than degenerate into an extreme "
        f"primary/secondary split (got primary={len(primary)}, "
        f"secondary={len(secondary)})."
    )
    assert primary, "primary must not be empty for a mono caption"


def test_cue_with_only_override_primary_is_dropped_entirely() -> None:
    """Regression: a cue whose top line is pure ASS override tags (``{\\a6}``,
    ``{\\pos(...)}``) leaves an *empty* primary after stripping. Before the
    fix, the cue still leaked its second line into ``secondary``, producing
    a partial align_index overlap that broke the F1/F2 topology
    invariant. The whole cue must be dropped: if primary has nothing to
    align, secondary has no 1:1 counterpart either.
    """
    caption = _cap([
        # Override-only primary line + non-empty secondary — this is the
        # shape of the sign/disclaimer card in many subtitle-group SRTs.
        {"text": "{\\a6}\n声明：本字幕仅作学习交流之用",
         "start": 1.0, "duration": 3.0},
        # Enough genuine bilingual cues so ``_detect_bilingual_mode``
        # locks onto line_by_line.
        {"text": "我们都很看好你\nWe all think a lot of you",
         "start": 5.0, "duration": 3.0},
        {"text": "摩斯 请进来\nMorse, please come in.",
         "start": 9.0, "duration": 3.0},
        {"text": "只要你需要\nAs long as you need.",
         "start": 13.0, "duration": 3.0},
    ])
    primary, secondary = caption.extract_alignment_supervisions()

    # idx 0 must land in neither side.
    assert 0 not in [s.align_index for s in primary]
    assert 0 not in [s.align_index for s in secondary]
    # Topology invariant: F1 inline ⇒ primary/secondary align_index sets
    # are identical (every bilingual row is represented on both sides).
    assert {s.align_index for s in primary} == {s.align_index for s in secondary}


def test_embedded_newlines_are_flattened_in_extracted_text() -> None:
    """Alignment text should be single-line. When a bilingual cue's side
    carries a soft wrap (``医生\\n先生`` kind of thing, or an alternating
    partner that ran long and got broken across lines), the extractor
    must flatten the embedded whitespace rather than surface raw
    newlines that confuse the lattice tokenizer and pollute audits.
    """
    # Alternating bilingual where the English partner on one pair
    # carries a hard line break inside its text.
    sups = [
        {"text": "今天天气真好", "start": 1.0, "duration": 3.0},
        {"text": "What a lovely day\nit is today.", "start": 1.0, "duration": 3.0},
        {"text": "明天我们去郊游", "start": 5.0, "duration": 3.0},
        {"text": "Tomorrow we'll go for a picnic.", "start": 5.0, "duration": 3.0},
        {"text": "带上你的相机", "start": 9.0, "duration": 3.0},
        {"text": "Bring your camera along.", "start": 9.0, "duration": 3.0},
        {"text": "希望不要下雨", "start": 13.0, "duration": 3.0},
        {"text": "Hope it doesn't rain.", "start": 13.0, "duration": 3.0},
    ]
    caption = _cap(sups)
    primary, secondary = caption.extract_alignment_supervisions()

    # Neither side may leak raw newlines.
    for side_name, rows in (("primary", primary), ("secondary", secondary)):
        for row in rows:
            assert "\n" not in row.text, (
                f"{side_name} row still contains newline: {row.text!r}"
            )

    # And the flattened text preserves the words (no tokens eaten).
    assert any("What a lovely day it is today." == s.text for s in secondary)


def test_mostly_mono_with_sparse_bilingual_rows_stays_mono() -> None:
    """Regression: a mostly-mono caption where only a small fraction of
    cues are bilingual (e.g. a live-broadcast dub that quotes a handful
    of original English lines for flavour) must not be classified as
    ``line_by_line``. The old detector tripped as soon as
    ``newline_bilingual / newline_total > 0.5``, ignoring the fact that
    the bilingual rows might be a tiny fraction of the whole caption.
    """
    sups = []
    # 40 mono Chinese rows — the dominant signal.
    for i in range(40):
        sups.append({"text": f"这是第{i}句中文对白 没有英文对照",
                     "start": 10.0 + i * 3.0, "duration": 2.5})
    # 5 sparse bilingual rows tucked in the tail.
    sups.append({"text": "欢迎来到第87届奥斯卡颁奖典礼\nWelcome to the 87th Academy Awards",
                 "start": 200.0, "duration": 3.0})
    sups.append({"text": "获奖者是\nAnd the winner is",
                 "start": 210.0, "duration": 3.0})
    sups.append({"text": "有请颁奖嘉宾\nPlease welcome our presenter",
                 "start": 220.0, "duration": 3.0})
    sups.append({"text": "感谢大家\nThank you everyone",
                 "start": 230.0, "duration": 3.0})
    sups.append({"text": "再见\nGoodbye",
                 "start": 240.0, "duration": 3.0})

    caption = _cap(sups)
    primary, secondary = caption.extract_alignment_supervisions()

    # 5/45 = 11 % coverage — below the bilingual threshold, so we want
    # mono. Secondary must be empty (or rolled back to empty).
    assert secondary == [], (
        "Sparse bilingual rows (<< 20 % of the caption) must not flip "
        "the mode to line_by_line bilingual."
    )
    # Primary carries every row; bilingual rows keep their embedded
    # newline flattened to a space.
    assert len(primary) == 45
    assert any("欢迎来到第87届奥斯卡颁奖典礼 Welcome to" in s.text for s in primary)
