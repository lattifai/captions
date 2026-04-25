"""Tests for ``lattifai.caption.bilingual.apply_alignment``.

Contract:
    apply_alignment(
        aligned_primary: List[Supervision],
        aligned_secondary: Optional[List[Supervision]] = None,
        *,
        plan: AlignmentPlan,
    ) -> None

    Mutates ``self.supervisions`` in place: for each ``aligned_primary[i]``,
    copy its ``start`` / ``duration`` / ``alignment["word"]`` onto
    ``self.supervisions[plan.source_indices_primary[i]]``. Same for the
    secondary side. Out-of-range indices are silently skipped.

    Source rows the aligner never saw (zero-duration / non-dialogue) keep
    their original timing — upstream callers decide how to handle them.
"""

from typing import Tuple

from lattifai.caption import Caption
from lattifai.caption.bilingual import AlignmentPlan, apply_alignment
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
) -> AlignmentPlan:
    return AlignmentPlan(
        source_indices_primary=primary_idx,
        source_indices_secondary=secondary_idx,
    )


# ---------------------------------------------------------------------------
# index-based write-back
# ---------------------------------------------------------------------------


def test_apply_writes_start_duration_for_matched_index() -> None:
    """One aligned row → one row in self updated in place."""
    caption = _cap([
        {"text": "Hello", "start": 1.0, "duration": 3.0},
    ])
    apply_alignment(caption.supervisions,[_aligned(1.234, 2.876)], plan=_plan(primary_idx=(0,)))
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
    apply_alignment(caption.supervisions,
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
    apply_alignment(caption.supervisions,
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
    apply_alignment(caption.supervisions,
        [_aligned(1.234, 2.876), _aligned(10.0, 5.0)],
        plan=_plan(primary_idx=(0, 99)),
    )
    assert caption.supervisions[0].start == 1.234


def test_apply_mutates_in_place_and_returns_none() -> None:
    """apply_alignment is an imperative method; returns None."""
    caption = _cap([
        {"text": "Hi", "start": 0.0, "duration": 2.0},
    ])
    result = apply_alignment(
        caption.supervisions,
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
    apply_alignment(caption.supervisions,
        [],
        [_aligned(1.200, 2.600)],
        plan=_plan(primary_idx=(), secondary_idx=(1,)),
    )
    assert caption.supervisions[1].start == 1.200
    assert caption.supervisions[1].duration == 2.600
    assert caption.supervisions[0].start == 1.0
    assert caption.supervisions[0].duration == 3.0


# ---------------------------------------------------------------------------
# Rows the aligner never saw — kept untouched
# ---------------------------------------------------------------------------


def test_apply_leaves_zero_duration_rows_untouched() -> None:
    """Source rows excluded from the plan keep their original timing.

    apply_alignment is pure write-back: rows the aligner never saw
    (zero-duration / non-dialogue / staff_credit etc.) remain at their
    original ``start`` / ``duration`` values. Upstream callers decide
    how to handle them — fail loudly, drop, or run a separate
    interpolation pass.
    """
    caption = _cap([
        {"text": "First line", "start": 1.0, "duration": 2.0},
        {"text": "Middle line", "start": 0.0, "duration": 0.0},
        {"text": "Last line", "start": 7.0, "duration": 3.0},
    ])
    plan = _plan(primary_idx=(0, 2))
    apply_alignment(caption.supervisions,
        [_aligned(1.234, 2.5), _aligned(7.1, 3.1)],
        plan=plan,
    )
    assert caption.supervisions[0].start == 1.234
    assert caption.supervisions[0].duration == 2.5
    # Middle row was not in the plan — original (0.0, 0.0) preserved.
    assert caption.supervisions[1].start == 0.0
    assert caption.supervisions[1].duration == 0.0
    assert caption.supervisions[2].start == 7.1
    assert caption.supervisions[2].duration == 3.1
