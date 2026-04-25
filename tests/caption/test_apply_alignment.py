"""Tests for ``Caption.apply_alignment()``.

Contract:
    apply_alignment(
        aligned_primary: List[Supervision],
        aligned_secondary: Optional[List[Supervision]] = None,
        *,
        plan: AlignmentPlan,
    ) -> None

    Mutates ``self.supervisions`` in place:

    1. For each ``aligned_primary[i]``, copy its ``start`` / ``duration`` /
       ``alignment["word"]`` onto ``self.supervisions[plan.source_indices_primary[i]]``.
       Same for the secondary side. Out-of-range indices are silently skipped.

    2. For each ``InterpRun`` in ``plan.interp_runs``, distribute the run's
       zero-duration dialogue rows uniformly between the run's left and
       right anchors (anchors are read *after* the write-back pass).
"""

from typing import List, Optional, Tuple

from lattifai.caption import Caption
from lattifai.caption.bilingual import AlignmentPlan, InterpRun
from lattifai.caption.supervision import AlignmentItem, Supervision


def _cap(supervisions, source_format="srt"):
    sups = []
    for s in supervisions:
        sups.append(
            Supervision(
                text=s["text"],
                start=s["start"],
                duration=s["duration"],
                custom=s.get("custom"),
                alignment=s.get("alignment"),
            )
        )
    return Caption(supervisions=sups, source_format=source_format)


def _aligned(start: float, duration: float, words=None) -> Supervision:
    """Build a fake alignment result (timing only — source mapping lives in plan)."""
    return Supervision(
        text="", start=start, duration=duration,
        alignment={"word": words} if words is not None else None,
    )


def _plan(
    primary_idx: Tuple[int, ...] = (),
    secondary_idx: Tuple[int, ...] = (),
    interp_runs: Tuple[InterpRun, ...] = (),
    break_indices=frozenset(),
) -> AlignmentPlan:
    return AlignmentPlan(
        source_indices_primary=primary_idx,
        source_indices_secondary=secondary_idx,
        interp_runs=interp_runs,
        break_indices=break_indices,
    )


# ---------------------------------------------------------------------------
# index-based write-back
# ---------------------------------------------------------------------------


def test_apply_writes_start_duration_for_matched_index() -> None:
    """One aligned row → one row in self updated in place."""
    caption = _cap([
        {"text": "Hello", "start": 1.0, "duration": 3.0},
    ])
    caption.apply_alignment([_aligned(1.234, 2.876)], plan=_plan(primary_idx=(0,)))
    assert caption.supervisions[0].start == 1.234
    assert caption.supervisions[0].duration == 2.876


def test_apply_writes_word_alignment_onto_matched_row() -> None:
    """Word-level alignment from aligned sup is copied over."""
    words = [
        AlignmentItem(symbol="Hello", start=1.0, duration=0.5, score=0.92),
        AlignmentItem(symbol="world", start=1.6, duration=0.4, score=0.88),
    ]
    caption = _cap([
        {"text": "Hello world", "start": 0.0, "duration": 3.0},
    ])
    caption.apply_alignment(
        [_aligned(1.0, 1.0, words=words)],
        plan=_plan(primary_idx=(0,)),
    )
    assert caption.supervisions[0].alignment is not None
    assert caption.supervisions[0].alignment["word"] == words


def test_apply_leaves_unaddressed_self_rows_untouched() -> None:
    """Rows in self whose index isn't targeted keep their original timing."""
    caption = _cap([
        {"text": "dialogue", "start": 1.0, "duration": 3.0},
        {"text": "staff", "start": 100.0, "duration": 5.0},
    ])
    caption.apply_alignment(
        [_aligned(1.234, 2.876)],
        plan=_plan(primary_idx=(0,)),
    )
    assert caption.supervisions[0].start == 1.234
    assert caption.supervisions[1].start == 100.0
    assert caption.supervisions[1].duration == 5.0


def test_apply_silently_ignores_out_of_range_indices() -> None:
    """An aligned sup mapped to an out-of-range index must NOT raise — just be skipped."""
    caption = _cap([
        {"text": "Hello", "start": 1.0, "duration": 3.0},
    ])
    caption.apply_alignment(
        [_aligned(1.234, 2.876), _aligned(10.0, 5.0)],
        plan=_plan(primary_idx=(0, 99)),
    )
    assert caption.supervisions[0].start == 1.234


def test_apply_mutates_in_place_and_returns_none() -> None:
    """apply_alignment is an imperative method; returns None."""
    caption = _cap([
        {"text": "Hi", "start": 0.0, "duration": 2.0},
    ])
    result = caption.apply_alignment(
        [_aligned(1.0, 1.0)],
        plan=_plan(primary_idx=(0,)),
    )
    assert result is None
    assert caption.supervisions[0].start == 1.0


# ---------------------------------------------------------------------------
# Dual-row (F2): each language row has its own source index
# ---------------------------------------------------------------------------


def test_apply_dual_row_updates_aligned_language_row_only() -> None:
    """If alignment only carries the English track, only that row gets updated.

    The Chinese row (different source index) stays at its original timestamps.
    """
    caption = _cap([
        {"text": "我们都很看好你", "start": 1.0, "duration": 3.0},          # idx 0
        {"text": "We all think a lot of you", "start": 1.0, "duration": 3.0},  # idx 1
    ])
    caption.apply_alignment(
        [],
        [_aligned(1.200, 2.600)],
        plan=_plan(primary_idx=(), secondary_idx=(1,)),
    )
    assert caption.supervisions[1].start == 1.200
    assert caption.supervisions[1].duration == 2.600
    assert caption.supervisions[0].start == 1.0
    assert caption.supervisions[0].duration == 3.0


# ---------------------------------------------------------------------------
# No-timing interpolation via plan.interp_runs
# ---------------------------------------------------------------------------


def test_apply_interpolates_zero_duration_dialogue_between_neighbours() -> None:
    """Plan-driven uniform interpolation between two anchors."""
    caption = _cap([
        {"text": "First line", "start": 1.0, "duration": 2.0},    # idx 0
        {"text": "Middle line", "start": 0.0, "duration": 0.0},   # idx 1
        {"text": "Last line", "start": 7.0, "duration": 3.0},     # idx 2
    ])
    plan = _plan(
        primary_idx=(0, 2),
        interp_runs=(InterpRun(rows=(1,), left_anchor=0, right_anchor=2),),
    )
    caption.apply_alignment(
        [_aligned(1.0, 2.0), _aligned(7.0, 3.0)],
        plan=plan,
    )
    middle = caption.supervisions[1]
    assert 3.0 <= middle.start < 7.0
    assert middle.duration > 0.0
    assert middle.start + middle.duration <= 7.0


def test_apply_does_not_touch_zero_duration_non_dialogue_rows() -> None:
    """A staff_credit row should never appear in plan.interp_runs."""
    caption = _cap([
        {"text": "First line", "start": 1.0, "duration": 2.0},     # idx 0
        {"text": "翻译 张三", "start": 0.0, "duration": 0.0},        # idx 1
        {"text": "Last line", "start": 7.0, "duration": 3.0},      # idx 2
    ])
    # Plan does NOT include idx 1 in any interp_run — the row is a
    # staff credit, classify_line_type returns non-None.
    plan = _plan(primary_idx=(0, 2))
    caption.apply_alignment(
        [_aligned(1.0, 2.0), _aligned(7.0, 3.0)],
        plan=plan,
    )
    credit = caption.supervisions[1]
    assert credit.start == 0.0
    assert credit.duration == 0.0


# ---------------------------------------------------------------------------
# break-aware interpolation
# ---------------------------------------------------------------------------


def test_apply_one_sided_extend_when_only_left_anchor_exists() -> None:
    """A run with only a left anchor extends from the left, duration=0."""
    caption = _cap([
        {"text": "A", "start": 0.0, "duration": 1.0},      # idx 0 timed
        {"text": "B", "start": 1.0, "duration": 0.0},      # idx 1 zero-dur dialogue
        {"text": "C", "start": 50.0, "duration": 1.0},     # idx 2 timed, post-gap
    ])
    # Plan: idx 1 belongs to a run whose right anchor is None (blocked
    # by the break boundary at idx 2).
    plan = _plan(
        primary_idx=(0, 2),
        interp_runs=(InterpRun(rows=(1,), left_anchor=0, right_anchor=None),),
        break_indices=frozenset({2}),
    )
    caption.apply_alignment(
        [_aligned(0.0, 1.0), _aligned(50.0, 1.0)],
        plan=plan,
    )
    assert caption.supervisions[1].start == 1.0
    assert caption.supervisions[1].duration == 0.0
    assert caption.supervisions[2].start == 50.0
    assert caption.supervisions[2].duration == 1.0


def test_apply_splits_zero_duration_run_into_two_runs_at_break_boundary() -> None:
    """Two independent interp_runs, one per break segment."""
    caption = _cap([
        {"text": "A", "start": 0.0, "duration": 1.0},       # idx 0 timed
        {"text": "B1", "start": 1.0, "duration": 0.0},      # idx 1 pre-break dialogue
        {"text": "B2", "start": 1.5, "duration": 0.0},      # idx 2 pre-break dialogue
        {"text": "C", "start": 50.0, "duration": 1.0},      # idx 3 timed, post-gap
        {"text": "D1", "start": 51.0, "duration": 0.0},     # idx 4 post-break dialogue
        {"text": "D2", "start": 51.5, "duration": 0.0},     # idx 5 post-break dialogue
        {"text": "E", "start": 60.0, "duration": 1.0},      # idx 6 timed
    ])
    # Plan: pre-break run uses left anchor only (right blocked); post-break
    # run uses both anchors.
    plan = _plan(
        primary_idx=(0, 3, 6),
        interp_runs=(
            InterpRun(rows=(1, 2), left_anchor=0, right_anchor=None),
            InterpRun(rows=(4, 5), left_anchor=3, right_anchor=6),
        ),
        break_indices=frozenset({3}),
    )
    caption.apply_alignment(
        [_aligned(0.0, 1.0), _aligned(50.0, 1.0), _aligned(60.0, 1.0)],
        plan=plan,
    )
    assert caption.supervisions[1].duration == 0.0
    assert caption.supervisions[2].duration == 0.0
    assert caption.supervisions[4].duration > 0.0
    assert caption.supervisions[5].duration > 0.0
    assert abs(caption.supervisions[4].duration - 4.5) < 1e-3
    assert abs(caption.supervisions[5].duration - 4.5) < 1e-3
